"""Dataset contracts and native-stage integration, without model calls."""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
import data_layer as data


class DataLayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = self.root / "bird"
        dbdir = self.ds / "dev_databases/shop"
        dbdir.mkdir(parents=True)
        self.database = dbdir / "shop.sqlite"
        with closing(sqlite3.connect(self.database)) as conn:
            conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, paid REAL)")
            conn.execute("INSERT INTO orders VALUES (1, 80)")
            conn.commit()
        data.write_json(
            self.ds / "mini_dev_sqlite.json",
            [
                {
                    "question_id": 7,
                    "db_id": "shop",
                    "question": "Actual payment?",
                    "evidence": "paid is actual payment",
                    "SQL": "SECRET_GOLD",
                    "difficulty": "SECRET_DIFFICULTY",
                }
            ],
        )
        self.layout = data.resolve_dataset(self.ds, "bird_minidev")
        self.out = self.root / "output"
        config = self.layout.prepare(self.out, self.layout.load_questions())
        self.env = patch.dict(
            os.environ, {"AUTOLINK_DATASET_CONFIG": str(config), "SQL_TIMEOUT": "2"}
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        data._runtime.cache_clear()
        self.tmp.cleanup()

    def test_online_contract_excludes_gold_and_uses_explicit_dialect(self):
        row = data.load_questions()["bird_7"]
        self.assertEqual(
            set(row), {"question", "db_name", "external_knowledge", "dialect"}
        )
        self.assertEqual(data.dialect("bird_7"), "sqlite")
        self.assertNotIn("SECRET", json.dumps(row))
        self.assertEqual(
            Path(
                data.resource_path("documents/" + row["external_knowledge"])
            ).read_text(),
            "paid is actual payment",
        )

    def test_readonly_query_write_attach_and_timeout(self):
        with closing(data.connect_sqlite(data.sqlite_path("shop"))) as conn:
            self.assertEqual(
                conn.execute("SELECT paid FROM orders").fetchone(), (80.0,)
            )
            for sql in [
                "DELETE FROM orders",
                "ATTACH DATABASE ':memory:' AS other",
                "PRAGMA query_only=OFF",
            ]:
                with self.assertRaises(sqlite3.DatabaseError):
                    conn.execute(sql)
        with closing(data.connect_sqlite(self.database, timeout=0.001)) as conn:
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute(
                    "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n"
                ).fetchone()

    def test_native_schema_exploration_pragma_is_readonly(self):
        with closing(data.connect_sqlite(self.database)) as conn:
            names = conn.execute(
                "SELECT name FROM pragma_table_info('orders')"
            ).fetchall()
            self.assertEqual(names, [("id",), ("paid",)])
            self.assertEqual(
                len(conn.execute("PRAGMA table_info('orders')").fetchall()), 2
            )
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("PRAGMA writable_schema=ON")

    def test_unregistered_database_rejected(self):
        other = self.root / "other.sqlite"
        other.write_bytes(self.database.read_bytes())
        with self.assertRaises(ValueError):
            data.connect_sqlite(other)
        with self.assertRaises(ValueError):
            data.resource_path("../../outside")

    def test_legacy_fallback(self):
        with patch.dict(os.environ, {"AUTOLINK_DATASET_CONFIG": ""}):
            self.assertEqual(data.question_file(), "spider2_data.json")
            self.assertEqual(
                [data.dialect(i) for i in ["local001", "ga001", "bq001", "sf001"]],
                ["sqlite", "bigquery", "bigquery", "snowflake"],
            )

    def test_dataset_switch_does_not_reuse_old_questions(self):
        ds2 = self.root / "bird2"
        import shutil

        shutil.copytree(self.ds, ds2)
        rows = data.read_json(ds2 / "mini_dev_sqlite.json")
        rows[0]["question_id"] = 8
        data.write_json(ds2 / "mini_dev_sqlite.json", rows)
        layout = data.resolve_dataset(ds2, "bird_minidev")
        config = layout.prepare(self.root / "output2", layout.load_questions())
        with patch.dict(os.environ, {"AUTOLINK_DATASET_CONFIG": str(config)}):
            self.assertEqual(set(data.load_questions()), {"bird_8"})
        self.assertEqual(set(data.load_questions()), {"bird_7"})

    def test_duplicate_questions_rejected(self):
        rows = data.read_json(self.ds / "mini_dev_sqlite.json")
        data.write_json(self.ds / "mini_dev_sqlite.json", rows * 2)
        with self.assertRaises(ValueError):
            self.layout.load_questions()

    def test_spider_alias_metadata_and_evidence(self):
        ds = self.root / "spider"
        dbroot = self.root / "sqlite"
        dbroot.mkdir()
        (dbroot / "Db-IMDB.sqlite").write_bytes(self.database.read_bytes())
        data.write_json(dbroot / "local-map.jsonl", {"local001": "Db-IMDB"})
        docs = ds / "resource/documents"
        docs.mkdir(parents=True)
        (docs / "hint.md").write_text("allowed evidence")
        (ds / "spider2-lite.jsonl").write_text(
            json.dumps(
                {
                    "instance_id": "local001",
                    "db": "Db-IMDB",
                    "question": "Payment?",
                    "external_knowledge": "hint.md",
                }
            )
            + "\n"
        )
        table = self.layout.load_schema(self.layout.load_questions()[0])[0]
        data.write_json(ds / "resource/databases/sqlite/DB_IMDB/orders.json", table)
        layout = data.resolve_dataset(ds, "spider2_lite_sqlite", dbroot)
        self.assertEqual(
            layout.validate(),
            {
                "dataset": "spider2_lite_sqlite",
                "questions": 1,
                "databases": 1,
                "tables": 1,
                "dialect": "sqlite",
            },
        )
        self.assertEqual(layout.load_questions()[0].evidence, "allowed evidence")
        (docs / "hint.md").unlink()
        with self.assertRaises(FileNotFoundError):
            layout.load_questions()

    def test_native_documents_schema_and_merge(self):
        import generate_docs, generate_schema, postprocess

        for kind in ("sqlite", "bigquery", "snowflake"):
            generate_docs.generate_documents(data.resource_path("databases/" + kind))
        docs = data.read_json(Path(data.artifact_path("documents/localdb.json")))
        self.assertEqual(set(docs), {"shop"})
        info = docs["shop"]["orders"]
        candidate = {
            "question": "Actual payment?",
            "db_name": "shop",
            "table_candidates": ["orders"] * 2,
            "column_candidates": list(info["columns"]),
            "column_types": info["column_types"],
            "column_values": info["sample_values"],
            "descriptions": list(info["columns"].values()),
        }
        log = self.out / "logs"
        data.write_json(log / "unfilled_pre_rule.json", {"bird_7": candidate})
        generate_schema.generate_schema_prompt(str(log), is_initial=True)
        postprocess.merge(str(log), is_preprocess=True)
        generate_schema.generate_schema_prompt(str(log))
        prompt = (log / "final_schema_prompts/bird_7.txt").read_text()
        self.assertIn("paid", prompt)
        self.assertIn("paid is actual payment", prompt)
        self.assertNotIn("SECRET", prompt)

    def test_native_generation_dialect_without_local_id_prefix(self):
        import sql_generation

        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="```sql\nSELECT paid FROM orders\n```",
                            reasoning_content="",
                        )
                    )
                ]
            )

        fake = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.object(sql_generation, "OpenAI", return_value=fake):
            result = sql_generation.SQLGenerator().generate(
                "bird_7", data.load_questions()["bird_7"], "orders(id,paid)"
            )
        self.assertIn("SQLite", calls[0]["messages"][0]["content"])
        self.assertIn("SELECT paid", result["output"])

    def test_native_exploration_reads_registered_schema(self):
        # Importing the original module constructs an SDK client; replace only that constructor.
        with patch("openai.OpenAI", return_value=SimpleNamespace()):
            module = importlib.import_module("complete_schema")
        status, frame = module.sql_execution(
            "bird_7", "SELECT name FROM pragma_table_info('orders')", "shop"
        )
        self.assertEqual(status, "success")
        self.assertEqual(frame.iloc[:, 0].tolist(), ["id", "paid"])

    def test_native_execution_and_refinement_database_mapping(self):
        import sql_revise, sql_execution

        status, result = sql_revise.query_sqlite("shop", "SELECT paid FROM orders")
        self.assertEqual(status, "success")
        self.assertEqual(result.iloc[0, 0], 80)
        log = self.out / "logs"
        (log / "sql_gen/test_execution_result_0").mkdir(parents=True)
        (log / "sql_gen/test_execution_error_0").mkdir()
        sql_execution.execute(
            "SELECT paid FROM orders", "bird_7", "shop", 0, str(log), "test"
        )
        self.assertTrue((log / "sql_gen/test_execution_result_0/bird_7.csv").exists())
        sql_execution.execute(
            "DELETE FROM orders", "bird_7", "shop", 0, str(log), "test"
        )
        self.assertTrue((log / "sql_gen/test_execution_error_0/bird_7.txt").exists())


if __name__ == "__main__":
    unittest.main()
