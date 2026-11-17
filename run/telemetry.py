"""Durable attempt records and offline summaries; no model dependencies."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

_context = ContextVar("autolink_telemetry", default=None)


def now():
    return time.perf_counter(), time.time()


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def scope(**fields):
    context = dict(_context.get() or {})
    context.update(fields)
    context.setdefault("_counts", {})
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def metadata():
    context = {k: v for k, v in (_context.get() or {}).items() if not k.startswith("_")}
    context.setdefault("run_id", os.environ.get("AUTOLINK_RUN_ID"))
    context.setdefault("session_id", os.environ.get("AUTOLINK_SESSION_ID"))
    context.setdefault("stage_attempt_id", os.environ.get("AUTOLINK_STAGE_ATTEMPT_ID"))
    context.setdefault("stage", os.environ.get("AUTOLINK_STAGE"))
    return context


def output_directory():
    return (_context.get() or {}).get("output_dir") or os.environ.get("AUTOLINK_OUTPUT_DIR")


def call_metadata(backend):
    result = metadata()
    context = _context.get()
    result.update(backend=backend, call_index=None, round_index=None)
    if context is not None:
        counts = context["_counts"]
        counts[backend] = counts.get(backend, 0) + 1
        result["call_index"] = counts[backend]
        if result.get("stage") in ("explore", "revise"):
            # Model-call order within this unit attempt; embedding follows the current round.
            result["round_index"] = counts.get("llm")
    return result


@lru_cache(maxsize=16)
def _identities(path, mtime):
    return {row["instance_id"]: row for row in json.loads(Path(path).read_text())}


def identity(output, instance_id):
    path = Path(output) / "identity.json"
    if not path.exists():
        return {"instance_id": instance_id}
    return _identities(str(path), path.stat().st_mtime_ns).get(instance_id, {"instance_id": instance_id})


class Record:
    def __init__(self, folder, prefix, *, started=None, **fields):
        self.id = uuid.uuid4().hex
        self.path = Path(folder) / f"{prefix}-{self.id}.json"
        self.perf, timestamp = started or now()
        self.data = {
            "schema_version": 1, "record_id": self.id, **metadata(), **fields,
            "status": "started", "started_at": utc(timestamp), "started_unix": timestamp,
            "ended_at": None, "ended_unix": None, "seconds": None,
        }
        for kind, key in (("sessions", "session_id"), ("stages", "stage_attempt_id"),
                          ("units", "unit_attempt_id")):
            if fields.get("kind") == kind:
                self.data[key] = self.id
        self._persist()

    def _persist(self):
        write_json(self.path, self.data)
        if self.data.get("backend") in ("llm", "embedding"):
            compact = {k: v for k, v in self.data.items() if k not in ("request", "response")}
            compact["request"] = {k: v for k, v in self.data.get("request", {}).items()
                                  if k not in ("messages", "input")}
            compact["usage"] = self.data.get("usage") or (self.data.get("response") or {}).get("usage")
            write_json(self.path.parent.parent / "telemetry/calls" / self.path.name, compact)

    def finish(self, status, **fields):
        ended = time.time()
        self.data.update(status=status, ended_at=utc(ended), ended_unix=ended,
                         seconds=time.perf_counter() - self.perf, **fields)
        self._persist()


@contextmanager
def span(output, kind, *, started=None, **fields):
    record = Record(Path(output) / "telemetry" / kind, kind, started=started, kind=kind, **fields)
    try:
        yield record
    except BaseException as exc:
        record.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        record.finish(record.data.get("outcome", "complete"))


@contextmanager
def unit_scope(output, stage, key):
    iid, _, candidate = key.partition("/")
    fields = dict(identity(output, iid), output_dir=str(Path(output).resolve()), stage=stage,
                  unit=key, candidate_index=int(candidate) if candidate.isdigit() else None,
                  _counts={})
    with scope(**fields):
        with span(output, "units") as record:
            with scope(unit_attempt_id=record.id):
                yield record


def usage_dict(usage):
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {key: getattr(usage, key) for key in (
        "prompt_tokens", "completion_tokens", "total_tokens"
    ) if getattr(usage, key, None) is not None} or None


def embedding_request_metadata(request):
    # Full vector responses and input texts are not duplicated into cost logs.
    texts = request["input"]
    return {**{k: v for k, v in request.items() if k != "input"},
            "input_count": len(texts),
            "input_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sum_known(values):
    numbers = [v for v in values if _number(v)]
    return sum(numbers) if numbers else None


def _latency(values):
    values = sorted(v for v in values if _number(v))
    return {"count": len(values), "sum": sum(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "p50": values[math.ceil(len(values) * .50) - 1] if values else None,
            "p95": values[math.ceil(len(values) * .95) - 1] if values else None,
            "max": values[-1] if values else None}


def _call_row(path, record, backend):
    usage = record.get("usage") or (record.get("response") or {}).get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    request = record.get("request") or {}
    fields = ("run_id", "session_id", "stage_attempt_id", "unit_attempt_id", "instance_id", "question_id", "db_id",
              "candidate_index", "pair_candidates", "call_index", "round_index", "batch_index", "status", "outcome",
              "started_at", "ended_at", "started_unix", "ended_unix", "endpoint", "ttft_seconds", "ttft_unavailable_reason")
    return {**{key: record.get(key) for key in fields}, "record_path": str(path),
            "record_id": record.get("record_id", path.stem), "backend": backend,
            "stage": record.get("stage") or path.name.split("-")[0],
            "model": request.get("model"),
            "latency_seconds": record.get("api_seconds", record.get("seconds")),
            "record_seconds": record.get("seconds"),
            "input_count": request.get("input_count"),
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "cached_tokens": prompt_details.get("cached_tokens"),
            "finish_reason": record.get("finish_reason")}


def _call_stats(rows):
    return {"attempts": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
            "outcome_counts": dict(Counter(r.get("outcome") or "unknown" for r in rows)),
            "unresolved_attempts": sum(r["status"] in ("started", "interrupted") for r in rows),
            "missing_total_tokens": sum(r.get("total_tokens") is None for r in rows),
            **{"reported_" + key: _sum_known(r.get(key) for r in rows)
               for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens")},
            "latency_seconds": _latency(r.get("latency_seconds") for r in rows)}


def _flat_stats(backend, stats):
    result = {}
    for key, value in stats.items():
        if key == "latency_seconds":
            result.update({f"{backend}_{key}_{stat}": number for stat, number in value.items()})
        else:
            result[backend + "_" + key] = value
    return result


def _csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or ["record_id"]
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items() if k in fields})
    temporary.replace(path)


def summarize(output):
    output = Path(output).resolve()
    warnings = []

    def read(path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            warnings.append(f"Cannot read {path}: {type(exc).__name__}")
            return default

    def records(folder):
        result = []
        for path in sorted((output / folder).glob("*.json")):
            source = path
            if folder in ("calls", "embedding_calls"):
                compact = output / "telemetry/calls" / path.name
                if compact.exists() and compact.stat().st_mtime_ns >= path.stat().st_mtime_ns:
                    source = compact
            record = read(source, {})
            if record:
                result.append((path, record))
        return result

    calls = [_call_row(p, r, backend) for folder, backend in (("calls", "llm"), ("embedding_calls", "embedding"))
             for p, r in records(folder)]
    calls.sort(key=lambda r: (r.get("started_unix") or 0, r["record_id"]))
    sessions = [r for _, r in records("telemetry/sessions")]
    stages = [r for _, r in records("telemetry/stages")]
    units = [r for _, r in records("telemetry/units")]
    identities = read(output / "identity.json", [])
    predictions = {r["instance_id"]: r for r in read(output / "predictions.json", [])}
    manifest = read(output / "manifest.json", {})
    stage_names = sorted({r["stage"] for r in calls + stages + units if r.get("stage")})
    stage_rows = []
    for stage in stage_names:
        row = {"stage": stage}
        attempts = [r for r in stages if r.get("stage") == stage]
        row.update(attempts=len(attempts), unfinished_attempts=sum(r["status"] == "started" for r in attempts),
                   wall_seconds_sum=_sum_known(r.get("seconds") for r in attempts),
                   skipped_attempts=sum(r["status"] == "skipped" for r in attempts))
        for backend in ("llm", "embedding"):
            stats = _call_stats([r for r in calls if r["stage"] == stage and r["backend"] == backend])
            row.update(_flat_stats(backend, stats))
        stage_rows.append(row)
    question_rows = []
    for item in identities:
        iid = item["instance_id"]
        related = [r for r in units if r.get("instance_id") == iid]
        active = [r for r in related if not r["status"].startswith("skipped")]
        by_session = defaultdict(list)
        for row in active:
            if row.get("session_id") and _number(row.get("ended_unix")):
                by_session[row["session_id"]].append(row)
        spans = [max(r["ended_unix"] for r in group) - min(r["started_unix"] for r in group)
                 for group in by_session.values()]
        error = read(output / "errors" / (iid + ".json"), {})
        row = {**item, "status": predictions.get(iid, {}).get("status", "error" if error else "pending"),
               "error_type": error.get("error_type"), "unit_attempts": len(active),
               "skipped_units": len(related) - len(active),
               "unfinished_units": sum(r["status"] == "started" for r in active),
               "active_seconds_sum": _sum_known(r.get("seconds") for r in active),
               "observed_span_seconds_sum": sum(spans) if spans else None}
        for backend in ("llm", "embedding"):
            stats = _call_stats([r for r in calls if r.get("instance_id") == iid and r["backend"] == backend])
            row.update(_flat_stats(backend, stats))
        question_rows.append(row)
    summary = {
        "schema_version": 1, "generated_at": utc(time.time()), "output_dir": str(output),
        "run_id": manifest.get("run_id", output.name), "dataset": manifest.get("dataset"), "split": manifest.get("split"),
        "model": manifest.get("model"), "settings": manifest.get("settings"),
        "registered_questions": len(identities), "prediction_status_counts": dict(Counter(
            row["status"] for row in question_rows)),
        "sessions": {"attempts": len(sessions), "status_counts": dict(Counter(r["status"] for r in sessions)),
                     "finalized_wall_seconds_sum": _sum_known(r.get("seconds") for r in sessions)},
        "llm": _call_stats([r for r in calls if r["backend"] == "llm"]),
        "embedding": _call_stats([r for r in calls if r["backend"] == "embedding"]),
        "unattributed_calls": _call_stats([r for r in calls if not r.get("instance_id")]),
        "legacy_calls_without_context": sum(not r.get("unit_attempt_id") and r["backend"] == "llm" for r in calls),
        "stage_metrics": stage_rows, "warnings": warnings,
        "definitions": {
            "attempts": "Persisted client attempts, not confirmed server executions; started may be in progress or interrupted.",
            "tokens": "Sum of reported fields only; missing usage is null, never estimated. Reasoning tokens are a subset of completion tokens, not an additional total.",
            "latency": "Client API round-trip seconds, including network/queue time, not GPU time; percentiles use nearest rank.",
            "wall_time": "Sum of finalized launch session durations, including validation, preparation and resume checks but excluding final summary generation; excludes downtime between launches. Unfinished sessions have unknown full duration.",
            "question_time": "active_seconds_sum sums unit-attempt wall time; observed_span_seconds_sum includes gaps between a question's units within each session, excluding downtime. Neither is a sum of parallel API latencies.",
            "shared_embedding": "Document embedding batches are shared preprocessing and are not assigned to individual questions.",
            "ttft": "null: requests are non-streaming; no first-token timestamp is observable.",
            "resume": "All retained attempts count, including failures and reruns. Checkpoint skips add no calls. Missing historical telemetry cannot be recovered.",
        },
    }
    for backend in ("llm", "embedding"):
        summary[backend]["attempts_per_registered_question"] = (
            summary[backend]["attempts"] / len(identities) if identities else None)
    for backend in ("llm", "embedding"):
        total = summary[backend]["reported_total_tokens"]
        summary[backend]["reported_total_tokens_per_registered_question"] = (
            total / len(identities) if total is not None and identities else None)
    summary["question_timing"] = {
        "active_seconds": _latency(row["active_seconds_sum"] for row in question_rows),
        "observed_span_seconds": _latency(row["observed_span_seconds_sum"] for row in question_rows),
    }
    metrics = output / "metrics"
    write_json(metrics / "summary.json", summary)
    _csv(metrics / "calls.csv", calls)
    _csv(metrics / "stages.csv", stage_rows)
    _csv(metrics / "questions.csv", question_rows)
    _csv(metrics / "sessions.csv", sessions)
    _csv(metrics / "units.csv", units)
    return summary
