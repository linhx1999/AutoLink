"""Official submission formats and model-context checks without model calls."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))
import data_layer as data
from run_experiment import validate_model_metadata


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.ids = [
            {
                "index": 0,
                "instance_id": "local002",
                "question_id": "local002",
                "db_id": "shop",
            },
            {
                "index": 1,
                "instance_id": "local003",
                "question_id": "local003",
                "db_id": "shop",
            },
        ]
        self.rows = [
            {**row, "pred": "SELECT 1" if i == 0 else "SELECT SQL ERROR"}
            for i, row in enumerate(self.ids)
        ]
        data.write_json(self.out / "identity.json", self.ids)

    def tearDown(self):
        self.tmp.cleanup()

    def test_spider_official_filename_discovery(self):
        data.export_predictions(
            "spider2_lite_sqlite", self.out, self.ids, list(reversed(self.rows))
        )
        directory = self.out / "submission_sql"
        # Same top-level .sql discovery rule as the official evaluator.
        discovered = {
            file.name.split(".")[0]
            for file in directory.iterdir()
            if file.name.endswith(".sql")
        }
        self.assertEqual(discovered, {"local002", "local003"})
        self.assertEqual(
            (directory / "local003.sql").read_text().strip(), "SELECT SQL ERROR"
        )
        self.assertEqual(data.verify_export(self.out, "spider2_lite_sqlite")["n"], 2)

    def test_bird_official_package_sqls_accepts_json(self):
        data.export_predictions("bird_minidev", self.out, self.ids, self.rows)
        # Execute the unchanged official parsing function, without importing unrelated DB drivers.
        source = ROOT.parent / "mini_dev/evaluation/evaluation_utils.py"
        tree = ast.parse(source.read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "package_sqls"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"json": json}
        exec(compile(module, str(source), "exec"), namespace)
        queries, _ = namespace["package_sqls"](
            str(self.out / "predict_dev.json"), "/unused/", mode="pred"
        )
        self.assertEqual(queries, ["SELECT 1", "SELECT SQL ERROR"])
        self.assertEqual(
            list(data.read_json(self.out / "predict_dev.json")), ["0", "1"]
        )
        self.assertEqual(data.verify_export(self.out, "bird_minidev")["n"], 2)

    def test_missing_duplicate_and_misaligned_predictions_rejected(self):
        for rows in [
            self.rows[:1],
            [self.rows[0], self.rows[0]],
            [{**self.rows[0], "db_id": "wrong"}, self.rows[1]],
        ]:
            with self.assertRaises(ValueError):
                data.export_predictions("bird_minidev", self.out, self.ids, rows)
        self.assertFalse((self.out / "predict_dev.json").exists())

    def test_tampered_or_extra_spider_submission_rejected(self):
        data.export_predictions("spider2_lite_sqlite", self.out, self.ids, self.rows)
        extra = self.out / "submission_sql/local999.sql"
        extra.write_text("SELECT 1")
        with self.assertRaises(ValueError):
            data.verify_export(self.out, "spider2_lite_sqlite")
        with self.assertRaises(ValueError):
            data.export_predictions(
                "spider2_lite_sqlite", self.out, self.ids, self.rows
            )
        extra.unlink()
        (self.out / "submission_sql/local002.sql").write_text("changed")
        with self.assertRaises(ValueError):
            data.verify_export(self.out, "spider2_lite_sqlite")

    def test_model_discovery_does_not_require_context_metadata(self):
        validate_model_metadata({"data": [{"id": "Qwen3-8B"}]}, "Qwen3-8B")
        validate_model_metadata(
            {"data": [{"id": "Qwen3-8B", "max_model_len": 32768}]}, "Qwen3-8B"
        )
        with self.assertRaises(ValueError):
            validate_model_metadata({"data": [{"id": "Qwen3-8B"}]}, "missing")


if __name__ == "__main__":
    unittest.main()
