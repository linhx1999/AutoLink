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



def summarize(output):
    """Raw attempt records are retained for later offline analysis."""
    return None
