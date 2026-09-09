"""A length finish is a terminal question error; infrastructure errors still stop."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiment as experiment
from run_experiment import data


def response(finish="stop", content="```sql\nSELECT 1\n```"):
    msg = SimpleNamespace(content=content, reasoning_content="thinking")
    result = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish)]
    )
    result.model_dump = lambda: {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"content": content, "reasoning_content": "thinking"},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 16384 if finish == "length" else 10,
            "total_tokens": 16484 if finish == "length" else 110,
        },
    }
    return result


class OutputLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ, {**experiment.ENV_DEFAULTS, "NUM_CANDIDATES": "2"}
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def client(self, responses):
        c = experiment.ModelClient.__new__(experiment.ModelClient)
        c.output = self.output
        c.stage = "generate"
        c.count = 0
        c.chat = SimpleNamespace(completions=c)
        c.api = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(side_effect=responses))
            )
        )
        return c

    def test_length_is_terminal_even_with_partial_content(self):
        for content in ["", "SELECT incomplete"]:
            c = self.client([response("length", content)])
            with self.assertRaises(experiment.OutputTokenLimit) as caught:
                c.create(model="test", messages=[])
            record = data.read_json(Path(caught.exception.call_record))
            self.assertEqual(record["status"], "returned")
            self.assertEqual(record["response"]["usage"]["completion_tokens"], 16384)

    def test_non_length_empty_response_is_not_silently_terminal(self):
        c = self.client([response("stop", "")])
        with self.assertRaises(experiment.ExperimentFailure) as caught:
            c.create(model="test", messages=[])
        self.assertNotIsInstance(caught.exception, experiment.OutputTokenLimit)

    def test_network_failure_still_propagates_and_is_retryable(self):
        c = self.client([ConnectionError("service unavailable")])
        with self.assertRaises(experiment.ExperimentFailure):
            experiment.run_unit(
                self.output,
                "generate",
                "local001/0",
                lambda: c.create(model="test", messages=[]),
                [],
            )
        self.assertFalse((self.output / "errors/local001.json").exists())
        self.assertEqual(
            data.read_json(next((self.output / "calls").glob("*.json")))["status"],
            "request_failure",
        )

    def test_generation_continues_next_question_and_skips_remaining_candidates(self):
        questions = {
            iid: {"question": iid, "db_name": "shop", "dialect": "sqlite"}
            for iid in ["local001", "local002"]
        }
        schema = self.output / "logs/final_schema_prompts"
        schema.mkdir(parents=True)
        for iid in questions:
            (schema / (iid + ".txt")).write_text("table t(id)")
        c = self.client([response("length", ""), response(), response()])
        with patch.object(experiment, "ModelClient", return_value=c), patch.object(
            data, "load_questions", return_value=questions
        ), patch.object(data, "dialect", return_value="sqlite"):
            experiment.run_stage("generate", self.output)
            experiment.run_stage("generate", self.output)
        self.assertEqual(c.api.chat.completions.create.call_count, 3)
        self.assertEqual(
            data.read_json(self.output / "errors/local001.json")["error_type"],
            "output_token_limit",
        )
        for cid in range(2):
            self.assertTrue(
                (
                    self.output / f"logs/sql_gen/experiment_sql_{cid}/local002.sql"
                ).exists()
            )
            self.assertFalse(
                (
                    self.output / f"logs/sql_gen/experiment_sql_{cid}/local001.sql"
                ).exists()
            )

    def test_export_keeps_failed_question_in_both_official_formats(self):
        ids = [
            {"index": i, "instance_id": iid, "question_id": i, "db_id": "shop"}
            for i, iid in enumerate(["local001", "local002"])
        ]
        data.write_json(self.output / "identity.json", ids)
        record = self.output / "call.json"
        data.write_json(record, {"finish_reason": "length"})
        experiment.mark_output_limit(
            self.output,
            "explore",
            "local001",
            experiment.OutputTokenLimit(record, 16384, {"completion_tokens": 16384}),
        )
        good = self.output / "logs/sql_selection/final/local002/selected.sql"
        good.parent.mkdir(parents=True)
        good.write_text("SELECT 1")
        for dataset in ["bird_minidev", "spider2_lite_sqlite"]:
            with patch.object(data, "load_questions", return_value={}), patch.object(
                data, "runtime", return_value={"dataset": dataset}
            ):
                experiment.run_stage("export", self.output)
            rows = data.read_json(self.output / "predictions.json")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["pred"], "")
            self.assertEqual(rows[0]["status"], "error")
            self.assertEqual(data.verify_export(self.output, dataset)["n"], 2)


if __name__ == "__main__":
    unittest.main()
