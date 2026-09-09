"""Dataset adapters and native-runtime resource access; gold stays offline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = ("bird_minidev", "spider2_lite_sqlite")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    tmp.replace(path)


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def inside(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Resource path escapes dataset root")
    return result


def identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@dataclass(frozen=True)
class Question:
    instance_id: str
    question_id: int | str
    db_id: str
    question: str
    evidence: str
    dialect: str = "sqlite"


@dataclass(frozen=True)
class DatasetLayout:
    dataset: str
    data_root: Path
    database_root: Path

    @property
    def input_file(self) -> Path:
        return self.data_root / (
            "mini_dev_sqlite.json"
            if self.dataset == "bird_minidev"
            else "spider2-lite.jsonl"
        )

    def load_questions(self) -> list[Question]:
        if self.dataset == "bird_minidev":
            rows = read_json(self.input_file)
            if not isinstance(rows, list):
                raise ValueError("Mini-Dev questions must be an array")
            questions = [
                Question(
                    f'bird_{r["question_id"]}',
                    r["question_id"],
                    r["db_id"],
                    r["question"],
                    r.get("evidence") or "",
                )
                for r in rows
            ]
        else:
            rows = [
                json.loads(line)
                for line in self.input_file.read_text().splitlines()
                if line.strip()
            ]
            questions = []
            for row in rows:
                if not row["instance_id"].startswith("local"):
                    continue
                name = row.get("external_knowledge")
                evidence = (
                    inside(self.data_root / "resource/documents", name).read_text()
                    if name
                    else ""
                )
                questions.append(
                    Question(
                        row["instance_id"],
                        row["instance_id"],
                        row["db"],
                        row["question"],
                        evidence,
                    )
                )
        if not questions:
            raise ValueError("No supported questions")
        if len({q.instance_id for q in questions}) != len(questions):
            raise ValueError("Duplicate instance ID")
        for q in questions:
            if not q.question.strip():
                raise ValueError("Empty question")
        return questions

    def database_path(self, question: Question) -> Path:
        if self.dataset == "bird_minidev":
            path = inside(
                self.database_root, f"{question.db_id}/{question.db_id}.sqlite"
            )
        else:
            mapping = read_json(self.database_root / "local-map.jsonl")
            if question.instance_id not in mapping:
                raise ValueError("Missing SQLite mapping: " + question.instance_id)
            stem = mapping[question.instance_id]
            path = inside(
                self.database_root,
                stem if stem.endswith(".sqlite") else stem + ".sqlite",
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def metadata_directory(self, db_id: str) -> Path:
        root = self.data_root / "resource/databases/sqlite"
        exact = inside(root, db_id)
        if exact.is_dir():
            return exact
        # Metadata directory names use underscores/case variants of local filenames.
        normalize = lambda value: value.casefold().replace("-", "_")
        matches = [
            p
            for p in root.iterdir()
            if p.is_dir() and normalize(p.name) == normalize(db_id)
        ]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous metadata alias: " + db_id)
        return matches[0]

    def load_schema(self, question: Question) -> list[dict]:
        db = self.database_path(question)
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            if self.dataset == "spider2_lite_sqlite":
                tables = []
                for path in sorted(
                    self.metadata_directory(question.db_id).glob("*.json")
                ):
                    raw = read_json(path)
                    # Whitelist metadata; do not propagate unrelated annotations.
                    table = {
                        k: raw[k]
                        for k in (
                            "table_fullname",
                            "column_names",
                            "column_types",
                            "description",
                            "sample_rows",
                        )
                    }
                    name = table["table_fullname"].split(".")[-1]
                    columns = {
                        r[1].casefold()
                        for r in conn.execute(
                            "PRAGMA table_info(" + identifier(name) + ")"
                        )
                    }
                    if not columns or any(
                        c.casefold() not in columns for c in table["column_names"]
                    ):
                        raise ValueError("Schema/database mismatch: " + str(path))
                    tables.append(table)
                if not tables:
                    raise ValueError("Empty metadata directory")
                return tables
            tables = []
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ):
                columns = list(
                    conn.execute("PRAGMA table_info(" + identifier(name) + ")")
                )
                descriptions = {}
                description_file = inside(
                    self.database_root,
                    f"{question.db_id}/database_description/{name}.csv",
                )
                if description_file.exists():
                    with description_file.open(
                        encoding="utf-8-sig", errors="replace", newline=""
                    ) as f:
                        for row in csv.DictReader(f):
                            descriptions[
                                row.get("original_column_name", "").strip()
                            ] = " ".join(
                                row.get(k) or ""
                                for k in (
                                    "column_name",
                                    "column_description",
                                    "value_description",
                                )
                            ).strip()
                names = [c[1] for c in columns]
                samples = [
                    dict(zip(names, r))
                    for r in conn.execute(
                        "SELECT * FROM " + identifier(name) + " LIMIT 3"
                    )
                ]
                tables.append(
                    {
                        "table_fullname": name,
                        "column_names": names,
                        "column_types": [c[2] for c in columns],
                        "description": [descriptions.get(n, "") for n in names],
                        "sample_rows": samples,
                    }
                )
            return tables

    def validate(self, questions: list[Question] | None = None) -> dict:
        questions = questions if questions is not None else self.load_questions()
        seen = {}
        table_count = 0
        for q in questions:
            path = self.database_path(q)
            if q.db_id in seen and seen[q.db_id] != path:
                raise ValueError("Conflicting database mapping")
            if q.db_id not in seen:
                seen[q.db_id] = path
                for table in self.load_schema(q):
                    n = len(table["column_names"])
                    if n != len(table["column_types"]) or n != len(
                        table["description"]
                    ):
                        raise ValueError("Misaligned column metadata")
                    table_count += 1
        return {
            "dataset": self.dataset,
            "questions": len(questions),
            "databases": len(seen),
            "tables": table_count,
            "dialect": "sqlite",
        }

    def source_hashes(self, questions: list[Question]) -> dict[str, str]:
        paths = {self.input_file}
        if self.dataset == "spider2_lite_sqlite":
            paths.add(self.database_root / "local-map.jsonl")
            rows = [
                json.loads(l)
                for l in self.input_file.read_text().splitlines()
                if l.strip()
            ]
            wanted = {q.instance_id for q in questions}
            for row in rows:
                if row["instance_id"] in wanted and row.get("external_knowledge"):
                    paths.add(
                        inside(
                            self.data_root / "resource/documents",
                            row["external_knowledge"],
                        )
                    )
        for q in questions:
            paths.add(self.database_path(q))
            directory = (
                self.metadata_directory(q.db_id)
                if self.dataset == "spider2_lite_sqlite"
                else self.database_root / q.db_id / "database_description"
            )
            paths.update(
                directory.glob(
                    "*.json" if self.dataset == "spider2_lite_sqlite" else "*.csv"
                )
            )
        return {str(p): file_hash(p) for p in sorted(paths)}

    def prepare(self, output: Path, questions: list[Question]) -> Path:
        """Materialize model-independent resources, never copy executable code."""
        output = output.resolve()
        resources = output / "data/resources"
        online = {}
        identities = []
        databases = {}
        for i, q in enumerate(questions):
            evidence = resources / "documents" / f"{i:06d}.txt"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(q.evidence)
            online[q.instance_id] = {
                "question": q.question,
                "db_name": q.db_id,
                "external_knowledge": evidence.name,
                "dialect": q.dialect,
            }
            identities.append(
                {
                    "index": i,
                    "question_id": q.question_id,
                    "instance_id": q.instance_id,
                    "db_id": q.db_id,
                }
            )
            if q.db_id not in databases:
                databases[q.db_id] = str(self.database_path(q))
                for index, table in enumerate(self.load_schema(q)):
                    write_json(
                        resources / "databases/sqlite" / q.db_id / f"{index:04d}.json",
                        table,
                    )
        for kind in ("sqlite", "bigquery", "snowflake"):
            (resources / "databases" / kind).mkdir(parents=True, exist_ok=True)
        write_json(output / "data/questions.json", online)
        write_json(output / "identity.json", identities)
        config = {
            "dataset": self.dataset,
            "questions": str(output / "data/questions.json"),
            "resources": str(resources),
            "artifacts": str(output),
            "databases": databases,
        }
        path = output / "data/runtime.json"
        write_json(path, config)
        return path


def resolve_dataset(
    data_root: str | Path, dataset: str, database_root: str | Path | None = None
) -> DatasetLayout:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError("Unsupported dataset: " + dataset)
    root = Path(data_root).expanduser().resolve()
    dbroot = (
        Path(database_root).expanduser().resolve()
        if database_root
        else root
        / (
            "dev_databases"
            if dataset == "bird_minidev"
            else "resource/databases/spider2-localdb"
        )
    )
    return DatasetLayout(dataset, root, dbroot)


@lru_cache(maxsize=8)
def _runtime(path: str) -> dict:
    return read_json(Path(path)) if path else {}


def runtime() -> dict:
    return _runtime(os.environ.get("AUTOLINK_DATASET_CONFIG", ""))


def question_file() -> str:
    return runtime().get("questions", "spider2_data.json")


def load_questions() -> dict:
    return read_json(Path(question_file()))


def dialect(instance_id: str) -> str:
    if runtime():
        return load_questions()[instance_id]["dialect"]
    if instance_id.startswith(("bq", "ga")):
        return "bigquery"
    if instance_id.startswith("sf"):
        return "snowflake"
    if instance_id.startswith("local"):
        return "sqlite"
    raise ValueError("Unknown legacy dialect: " + instance_id)


def resource_path(relative: str = "") -> str:
    return str(inside(Path(runtime().get("resources", "resource")).resolve(), relative))


def artifact_path(relative: str = "") -> str:
    return str(Path(runtime().get("artifacts", ".")) / relative)


def embedding_path(instance_id: str) -> str:
    return artifact_path(
        "embeddings/"
        + ("localdb" if dialect(instance_id) == "sqlite" else dialect(instance_id))
    )


def sqlite_path(db_name: str) -> str:
    if runtime():
        return runtime()["databases"][db_name]
    return str(Path("resource/databases/spider2-localdb") / (db_name + ".sqlite"))


def connect_sqlite(
    database: str | Path, timeout: float | None = None
) -> sqlite3.Connection:
    path = Path(database).resolve()
    if runtime() and str(path) not in runtime()["databases"].values():
        raise ValueError("Database outside registered dataset")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")

    def authorize(action, name, argument, database, trigger):
        if action == sqlite3.SQLITE_PRAGMA:
            # AutoLink explores columns via pragma_table_info(); only metadata PRAGMAs are allowed.
            return (
                sqlite3.SQLITE_OK
                if (name or "").lower()
                in {
                    "table_info",
                    "table_xinfo",
                    "foreign_key_list",
                    "index_list",
                    "index_info",
                    "index_xinfo",
                }
                else sqlite3.SQLITE_DENY
            )
        return (
            sqlite3.SQLITE_OK
            if action
            in {
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }
            else sqlite3.SQLITE_DENY
        )

    conn.set_authorizer(authorize)
    deadline = time.monotonic() + (
        timeout if timeout is not None else float(os.environ.get("SQL_TIMEOUT", "30"))
    )
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    return conn


def export_predictions(
    dataset: str, output: Path, identities: list[dict], rows: list[dict]
) -> dict:
    """Export complete predictions in each benchmark's native submission format."""
    import re

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError("Unsupported export dataset: " + dataset)
    expected = [r["instance_id"] for r in identities]
    supplied = [r["instance_id"] for r in rows]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or len(set(supplied)) != len(supplied)
        or set(expected) != set(supplied)
    ):
        raise ValueError(
            "Predictions must cover every registered instance exactly once"
        )
    by_id = {r["instance_id"]: r for r in rows}
    ordered = []
    for index, identity in enumerate(identities):
        row = by_id[identity["instance_id"]]
        if identity["index"] != index or any(
            row.get(k) != identity[k] for k in ("index", "question_id", "db_id")
        ):
            raise ValueError("Prediction identity/order mismatch")
        if not isinstance(row.get("pred"), str):
            raise ValueError("Prediction must be a string")
        if not row["pred"].strip() and not (
            row.get("status") == "error"
            and row.get("error_type") == "output_token_limit"
        ):
            raise ValueError("Empty prediction requires an explicit terminal error")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identity["instance_id"]):
            raise ValueError("Unsafe submission filename")
        ordered.append(row)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    if dataset == "bird_minidev":
        separator = "\t----- bird -----\t"
        if any(separator in row["pred"] for row in ordered):
            raise ValueError("Prediction contains reserved BIRD separator")
        target = output / "predict_dev.json"
        write_json(
            target,
            {
                str(i): row["pred"].strip() + separator + row["db_id"]
                for i, row in enumerate(ordered)
            },
        )
        files[target.name] = file_hash(target)
        format_name = "bird-indexed-json"
    else:
        directory = output / "submission_sql"
        directory.mkdir(exist_ok=True)
        wanted = {iid + ".sql" for iid in expected}
        extra = {p.name for p in directory.glob("*.sql")} - wanted
        if extra:
            raise ValueError(
                "Unexpected SQL files in submission directory: "
                + ", ".join(sorted(extra))
            )
        for row in ordered:
            target = directory / (row["instance_id"] + ".sql")
            tmp = target.with_suffix(".sql.tmp")
            tmp.write_text(row["pred"].strip() + "\n", encoding="utf-8")
            tmp.replace(target)
            files[str(target.relative_to(output))] = file_hash(target)
        format_name = "spider2-instance-sql"
    write_json(output / "predictions.json", ordered)
    files["predictions.json"] = file_hash(output / "predictions.json")
    manifest = {
        "dataset": dataset,
        "format": format_name,
        "n": len(expected),
        "instance_ids": expected,
        "files": files,
    }
    write_json(output / "export_manifest.json", manifest)
    return manifest


def verify_export(output: Path, dataset: str) -> dict:
    """Check export coverage and hashes before invoking an official evaluator."""
    output = output.resolve()
    manifest = read_json(output / "export_manifest.json")
    identities = read_json(output / "identity.json")
    if (
        manifest["dataset"] != dataset
        or manifest["instance_ids"] != [r["instance_id"] for r in identities]
        or manifest["n"] != len(identities)
    ):
        raise ValueError("Export dataset or coverage differs from registered run")
    for relative, expected_hash in manifest["files"].items():
        path = inside(output, relative)
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError("Export missing or changed: " + relative)
    if dataset == "spider2_lite_sqlite":
        found = {p.stem for p in (output / "submission_sql").glob("*.sql")}
        if found != set(manifest["instance_ids"]):
            raise ValueError("Unexpected or missing submission IDs")
    return manifest
