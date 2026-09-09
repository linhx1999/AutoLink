"""Unified AutoLink dataset entry point. Validation/preparation never call an LLM."""

from __future__ import annotations

import argparse
import hashlib
import fcntl
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "run"))
import data_layer as data

LINK_STAGES = [
    "embeddings",
    "retrieve",
    "add_ids",
    "initial_schema",
    "explore",
    "merge",
    "final_schema",
]
SQL_STAGES = ["generate", "execute", "revise", "select", "export"]
ENV_DEFAULTS = {
    "TOP_N": "100",
    "NUM_CANDIDATES": "5",
    "MAX_TOKENS": "16384",
    "API_TIMEOUT": "600",
    "SQL_TIMEOUT": "30",
    "STAGE_TIMEOUT": "43200",
    "MAX_API_CALLS_PER_STAGE": "30000",
    "TEMPERATURE": "0",
    "TOP_P": "0.95",
    "TOP_K": "20",
    "SEED": "42",
    "EMBEDDING_MODEL": "BAAI/bge-large-en-v1.5",
    "EMBEDDING_DEVICE": "cpu",
}


class ExperimentFailure(BaseException):
    """Abort native unlimited retries; do not score an incomplete stage."""


class OutputTokenLimit(ExperimentFailure):
    """Terminal per-question generation failure, not an infrastructure outage."""

    def __init__(self, call_record: Path, max_tokens: int, usage: dict | None):
        super().__init__("Model output reached max_tokens")
        self.call_record = str(call_record.resolve())
        self.max_tokens = max_tokens
        self.usage = usage


def question_error(output: Path, instance_id: str) -> dict | None:
    path = output / "errors" / (instance_id + ".json")
    return data.read_json(path) if path.exists() else None


def mark_output_limit(
    output: Path, stage: str, key: str, error: OutputTokenLimit
) -> None:
    instance_id = key.split("/", 1)[0]
    data.write_json(
        output / "errors" / (instance_id + ".json"),
        {
            "status": "error",
            "error_type": "output_token_limit",
            "instance_id": instance_id,
            "stage": stage,
            "unit": key,
            "call_record": error.call_record,
        },
    )
    print("Question failed:", instance_id, "output_token_limit; continuing", flush=True)


class ModelClient:
    """Experiment-only request settings and accounting; no global SDK mutation."""

    def __init__(self, output: Path, stage: str):
        import httpx
        from openai import OpenAI

        self.api = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            max_retries=0,
            http_client=httpx.Client(
                trust_env=False, timeout=float(os.environ["API_TIMEOUT"])
            ),
        )
        self.output = output
        self.stage = stage
        self.count = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.count += 1
        folder = self.output / "calls"
        folder.mkdir(exist_ok=True)
        if sum(1 for _ in folder.glob(self.stage + "-*.json")) >= int(
            os.environ["MAX_API_CALLS_PER_STAGE"]
        ):
            raise ExperimentFailure(
                "Model-call stage budget exhausted (including previous attempts)"
            )
        request.update(
            max_tokens=int(os.environ["MAX_TOKENS"]),
            temperature=float(os.environ["TEMPERATURE"]),
            top_p=float(os.environ["TOP_P"]),
            seed=int(os.environ["SEED"]),
        )
        request["extra_body"] = {"top_k": int(os.environ["TOP_K"])}
        started = time.monotonic()
        record = folder / f"{self.stage}-{time.time_ns()}.json"
        data.write_json(record, {"status": "started", "request": request})
        try:
            response = self.api.chat.completions.create(**request)
            msg = response.choices[0].message
            if not getattr(msg, "reasoning_content", None):
                msg.reasoning_content = getattr(msg, "reasoning", None) or ""
            data.write_json(
                record,
                {
                    "status": "returned",
                    "request": request,
                    "response": response.model_dump(),
                    "seconds": time.monotonic() - started,
                },
            )
            if response.choices[0].finish_reason == "length":
                raise OutputTokenLimit(
                    record, request["max_tokens"], response.model_dump().get("usage")
                )
            if not msg.content:
                raise ExperimentFailure("No final content; stage remains incomplete")
            return response
        except Exception as exc:
            data.write_json(
                record,
                {
                    "status": "request_failure",
                    "error": str(exc),
                    "request": request,
                    "seconds": time.monotonic() - started,
                },
            )
            raise ExperimentFailure("Model request failed; see call log") from exc


def run_unit(
    output: Path,
    stage: str,
    key: str,
    operation,
    required: list[Path],
    outcomes: list[Path] | None = None,
) -> None:
    """Resume only fully persisted units; replay a partially written unit."""
    if question_error(output, key.split("/", 1)[0]):
        print("Resume: skip failed question", key, flush=True)
        return
    outcomes = outcomes or []
    checkpoint = (
        output
        / "checkpoints"
        / stage
        / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    )
    if checkpoint.exists():
        record = data.read_json(checkpoint)
        hashes = record.get("artifacts", {})
        if (
            record.get("status") == "complete"
            and hashes
            and all(
                Path(path).is_file() and data.file_hash(Path(path)) == digest
                for path, digest in hashes.items()
            )
        ):
            print("Resume: skip", stage, key, flush=True)
            return
    # These are outputs owned solely by this unit, never input datasets.
    for path in required + outcomes:
        path.unlink(missing_ok=True)
    try:
        operation()
    except OutputTokenLimit as exc:
        mark_output_limit(output, stage, key, exc)
        return
    if any(not path.is_file() for path in required):
        raise ExperimentFailure("Incomplete unit artifacts: " + stage + "/" + key)
    completed_outcomes = [path for path in outcomes if path.is_file()]
    if outcomes and len(completed_outcomes) != 1:
        raise ExperimentFailure(
            "Expected exactly one execution outcome: " + stage + "/" + key
        )
    paths = required + completed_outcomes
    data.write_json(
        checkpoint,
        {
            "status": "complete",
            "key": key,
            "artifacts": {str(path.resolve()): data.file_hash(path) for path in paths},
        },
    )


def run_stage(stage: str, output: Path) -> None:
    """Dispatch original method functions; all data access is through data_layer."""
    questions = data.load_questions()
    if stage in ("explore", "generate", "execute", "revise", "select"):
        questions = {
            iid: item
            for iid, item in questions.items()
            if not question_error(output, iid)
        }
        if not questions:
            print("No pending non-terminal questions in", stage, flush=True)
            return
    log = str(output / "logs")
    task = "experiment"
    count = int(os.environ["NUM_CANDIDATES"])
    Path(log).mkdir(exist_ok=True)
    if stage == "documents":
        import generate_docs

        for kind in ("sqlite", "bigquery", "snowflake"):
            generate_docs.generate_documents(data.resource_path("databases/" + kind))
    elif stage == "embeddings":
        import embedding_docs

        embedding_docs.embed_documents(
            data.artifact_path("documents/localdb.json"),
            data.artifact_path("embeddings/localdb"),
        )
    elif stage == "retrieve":
        import retrieve_topk_schema

        candidates = {}
        for iid, info in questions.items():
            saved = output / "retrieved" / (iid + ".json")

            def retrieve_one():
                for folder in ("cache", "status"):
                    Path(log, folder, iid + ".json").unlink(missing_ok=True)
                result = retrieve_topk_schema.process_batch_with_device(
                    {iid: info}, 0, int(os.environ["TOP_N"]), log
                )
                data.write_json(saved, result[iid])

            run_unit(
                output,
                stage,
                iid,
                retrieve_one,
                [
                    saved,
                    Path(log, "cache", iid + ".json"),
                    Path(log, "status", iid + ".json"),
                ],
            )
            candidates[iid] = data.read_json(saved)
        data.write_json(Path(log, "initial_candidates.json"), candidates)
    elif stage == "add_ids":
        import add_id

        add_id.add_pre_rule(log)
    elif stage in ("initial_schema", "final_schema"):
        import generate_schema

        generate_schema.generate_schema_prompt(
            log, is_initial=stage == "initial_schema"
        )
    elif stage == "explore":
        # Native module constructs a client at import; suppress process proxy discovery for local API only.
        import complete_schema

        complete_schema.client = ModelClient(output, stage)
        for folder in (
            "backup",
            "candidates",
            "model_output",
            "tool_calls",
            "input",
            "error",
        ):
            Path(log, folder).mkdir(exist_ok=True)
        initial = data.read_json(Path(log, "initial_candidates.json"))
        for iid in questions:
            complete_schema.backup_instance_state(iid, log)
            complete_schema.restore_instance_state(iid, log)
            status = data.read_json(Path(log, "status", iid + ".json"))
            if status["is_complete"]:
                continue
            run_unit(
                output,
                stage,
                iid,
                lambda: complete_schema.process_instance_batch(
                    {iid: initial[iid]}, log
                ),
                [
                    Path(log, folder, iid + suffix)
                    for folder, suffix in [
                        ("candidates", ".json"),
                        ("model_output", ".txt"),
                        ("tool_calls", ".json"),
                        ("input", ".txt"),
                    ]
                ],
            )
    elif stage == "merge":
        import postprocess

        postprocess.merge(log, is_preprocess=True)
    elif stage == "generate":
        import sql_generation as module

        client = ModelClient(output, stage)
        module.OpenAI = lambda **kwargs: client
        generator = module.SQLGenerator()
        for cid in range(count):
            for iid in questions:
                run_unit(
                    output,
                    stage,
                    f"{iid}/{cid}",
                    lambda: module.run_task(
                        generator,
                        questions,
                        f"{log}/final_schema_prompts",
                        log,
                        task,
                        iid,
                        cid,
                    ),
                    [
                        Path(log, "sql_gen", f"{task}_{kind}_{cid}", iid + ".txt")
                        for kind in ("sql_generation", "reasoning")
                    ],
                )
        for cid in range(count):
            if Path(log, "sql_gen", f"{task}_sql_generation_{cid}").is_dir():
                module.sql_clean(log, task, cid)
    elif stage == "execute":
        import sql_execution as module

        for cid in range(count):
            for folder in (
                f"{task}_execution_error_{cid}",
                f"{task}_execution_result_{cid}",
            ):
                Path(log, "sql_gen", folder).mkdir(parents=True, exist_ok=True)
            for iid, item in questions.items():
                sql = Path(
                    log, "sql_gen", f"{task}_sql_{cid}", iid + ".sql"
                ).read_text()
                run_unit(
                    output,
                    stage,
                    f"{iid}/{cid}",
                    lambda: module.execute(sql, iid, item["db_name"], cid, log, task),
                    [],
                    [
                        Path(
                            log,
                            "sql_gen",
                            f"{task}_execution_{kind}_{cid}",
                            iid + suffix,
                        )
                        for kind, suffix in [("error", ".txt"), ("result", ".csv")]
                    ],
                )
    elif stage == "revise":
        import sql_revise as module

        client = ModelClient(output, stage)
        module.OpenAI = lambda **kwargs: client
        reviser = module.SQLReviser()
        for cid in range(count):
            for iid in questions:
                if not Path(
                    log, "sql_gen", f"{task}_execution_error_{cid}", iid + ".txt"
                ).exists():
                    continue
                run_unit(
                    output,
                    stage,
                    f"{iid}/{cid}",
                    lambda: module.run_task(
                        reviser,
                        questions,
                        f"{log}/final_schema_prompts",
                        log,
                        task,
                        iid,
                        cid,
                    ),
                    [
                        Path(log, "sql_revise", f"{task}_sql_{cid}", iid + ".sql"),
                        Path(
                            log, "sql_revise", f"{task}_messages_{cid}", iid + ".json"
                        ),
                    ],
                    [
                        Path(log, "sql_revise", f"{task}_sql_{cid}", iid + suffix)
                        for suffix in (".csv", ".txt")
                    ],
                )
    elif stage == "select":
        import sql_selection as module

        client = ModelClient(output, stage)
        module.OpenAI = lambda **kwargs: client
        Path(log, "sql_selection").mkdir(exist_ok=True)
        args = SimpleNamespace(
            log_path=log,
            task=task,
            num_candidates=count,
            max_rows=1000,
            max_chars=10000,
        )
        for iid in questions:
            run_unit(
                output,
                stage,
                iid,
                lambda: module.process_instance(iid + ".txt", args, threading.Lock()),
                [Path(log, "sql_selection/final", iid, "selected.sql")],
                [
                    Path(log, "sql_selection/final", iid, filename)
                    for filename in ("result.csv", "error.txt")
                ],
            )
    elif stage == "export":
        rows = []
        for identity in data.read_json(output / "identity.json"):
            p = Path(
                log, "sql_selection/final", identity["instance_id"], "selected.sql"
            )
            failure = question_error(output, identity["instance_id"])
            sql = "" if failure else p.read_text().strip()
            if not failure and not sql:
                raise ExperimentFailure(
                    "Missing prediction: " + identity["instance_id"]
                )
            row = {
                **identity,
                "pred": sql,
                "status": "error" if failure else "complete",
            }
            if failure:
                row["error_type"] = failure["error_type"]
                row["error_stage"] = failure["stage"]
            rows.append(row)
        data.export_predictions(
            data.runtime()["dataset"],
            output,
            data.read_json(output / "identity.json"),
            rows,
        )
    else:
        raise ValueError(stage)


def validate_model_metadata(models: dict, model: str) -> None:
    if not any(item.get("id") == model for item in models.get("data", [])):
        raise ValueError("Configured model is not served: " + model)


def check_model_service(env: dict) -> None:
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        env["OPENAI_BASE_URL"].rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + env["OPENAI_API_KEY"]},
    )
    with opener.open(request, timeout=10) as response:
        models = json.load(response)
    validate_model_metadata(models, env.get("MODEL_NAME", ""))


def package_versions() -> dict:
    versions = {}
    for name in (
        "openai",
        "httpx",
        "pandas",
        "sentence-transformers",
        "faiss-cpu",
        "transformers",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=data.SUPPORTED_DATASETS, default=os.environ.get("DATASET")
    )
    parser.add_argument("--data_root", type=Path, default=os.environ.get("DATA_ROOT"))
    parser.add_argument(
        "--database_root", type=Path, default=os.environ.get("DATABASE_ROOT")
    )
    parser.add_argument("--output_dir", type=Path, default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    parser.add_argument(
        "--stage",
        choices=("validate", "prepare", "link", "sql", "all"),
        default="all",
    )
    compatibility = parser.add_mutually_exclusive_group()
    compatibility.add_argument("--check", action="store_true")
    compatibility.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--internal-stage", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_stage:
        run_stage(args.internal_stage, Path(os.environ["AUTOLINK_OUTPUT_DIR"]))
        return
    if not args.dataset or not args.data_root:
        parser.error("--dataset and --data_root are required")
    if args.check:
        args.stage = "validate"
    if args.prepare_only:
        args.stage = "prepare"
    layout = data.resolve_dataset(args.data_root, args.dataset, args.database_root)
    questions = layout.load_questions()
    if args.limit < 0 or args.limit > len(questions):
        raise ValueError("Invalid limit")
    if args.limit:
        questions = questions[: args.limit]
    report = layout.validate(questions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.stage == "validate":
        return
    if not args.output_dir:
        parser.error("--output_dir is required beyond validation")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = {k: os.environ.get(k, v) for k, v in ENV_DEFAULTS.items()}
    if int(settings["MAX_TOKENS"]) <= 0:
        raise ValueError("MAX_TOKENS must be positive")
    manifest = {
        "protocol": "autolink-data-layer-v1",
        "dataset": args.dataset,
        "instances": [q.instance_id for q in questions],
        "input_hashes": layout.source_hashes(questions),
        "settings": settings,
        "model": os.environ.get("MODEL_NAME"),
        "endpoint": os.environ.get("OPENAI_BASE_URL"),
        "code_hashes": {
            str(p.relative_to(ROOT)): data.file_hash(p)
            for p in [ROOT / "run_experiment.py", *sorted((ROOT / "run").glob("*.py"))]
        },
        "packages": package_versions(),
    }
    old = output / "manifest.json"
    if old.exists() and data.read_json(old) != manifest:
        raise ValueError("Configuration changed: use a new output directory")
    data.write_json(old, manifest)
    runtime_path = layout.prepare(output, questions)
    env = os.environ.copy()
    env.update(settings)
    env["PYTHONHASHSEED"] = settings["SEED"]
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["AUTOLINK_DATASET_CONFIG"] = str(runtime_path)
    env["AUTOLINK_OUTPUT_DIR"] = str(output)
    if args.stage in ("link", "sql", "all"):
        if not env.get("OPENAI_BASE_URL") or not env.get("OPENAI_API_KEY"):
            raise ValueError("Set API configuration in the experiment shell script")
        # Only the local endpoint bypasses process proxy settings; remote endpoints keep them.
        from urllib.parse import urlparse

        if urlparse(env["OPENAI_BASE_URL"]).hostname in (
            "127.0.0.1",
            "localhost",
            "::1",
        ):
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ):
                env.pop(key, None)
    stages = ["documents"]
    if args.stage in ("link", "all"):
        stages += LINK_STAGES
    if args.stage == "sql" and not (output / "stages/final_schema.json").exists():
        raise ValueError("Run --stage link first with the same output directory")
    if args.stage in ("sql", "all"):
        stages += SQL_STAGES
    if any(
        stage in ("explore", "generate", "revise", "select")
        and not (output / "stages" / (stage + ".json")).exists()
        for stage in stages
    ):
        check_model_service(env)
    for stage in stages:
        marker = output / "stages" / (stage + ".json")
        if marker.exists() and data.read_json(marker).get("status") == "complete":
            print("Resume: skip stage", stage, flush=True)
            continue
        with (output / (stage + ".log")).open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_experiment.py"),
                    "--internal-stage",
                    stage,
                ],
                env=env,
                cwd=output,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=float(settings["STAGE_TIMEOUT"]),
            )
        data.write_json(marker, {"status": "complete"})
        print("Completed stage:", stage, flush=True)
    print("Output:", output)


if __name__ == "__main__":
    main()
