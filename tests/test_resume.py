"""Resume contracts using local files and fake operations, never model calls."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_experiment import ExperimentFailure, run_unit


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_completed_unit_is_not_repeated(self):
        result = self.output / "result.sql"
        calls = []

        def operation():
            calls.append(True)
            result.write_text("SELECT 1")

        run_unit(self.output, "generate", "q/0", operation, [result])
        run_unit(self.output, "generate", "q/0", operation, [result])
        self.assertEqual(len(calls), 1)

    def test_partial_unit_is_replayed(self):
        first = self.output / "raw.txt"
        second = self.output / "reasoning.txt"
        first.write_text("half-written")

        def operation():
            self.assertFalse(first.exists())
            first.write_text("complete")
            second.write_text("reasoning")

        run_unit(self.output, "generate", "q/0", operation, [first, second])
        self.assertEqual(first.read_text(), "complete")

    def test_corrupted_completed_file_is_not_accepted(self):
        result = self.output / "result.sql"
        calls = []

        def operation():
            calls.append(True)
            result.write_text("SELECT 1")

        run_unit(self.output, "generate", "q/0", operation, [result])
        result.write_text("broken")
        run_unit(self.output, "generate", "q/0", operation, [result])
        self.assertEqual(len(calls), 2)

    def test_interruption_does_not_mark_completion(self):
        result = self.output / "result.sql"

        def interrupted():
            result.write_text("partial")
            raise RuntimeError("interrupted")

        with self.assertRaises(RuntimeError):
            run_unit(self.output, "generate", "q/0", interrupted, [result])
        self.assertFalse(list((self.output / "checkpoints").rglob("*.json")))
        run_unit(
            self.output,
            "generate",
            "q/0",
            lambda: result.write_text("SELECT 1"),
            [result],
        )
        self.assertEqual(result.read_text(), "SELECT 1")

    def test_missing_execution_outcome_is_not_complete(self):
        csv = self.output / "result.csv"
        err = self.output / "error.txt"
        with self.assertRaises(ExperimentFailure):
            run_unit(self.output, "execute", "q/0", lambda: None, [], [csv, err])
        run_unit(
            self.output,
            "execute",
            "q/0",
            lambda: err.write_text("syntax error"),
            [],
            [csv, err],
        )
        run_unit(
            self.output,
            "execute",
            "q/0",
            lambda: self.fail("must skip"),
            [],
            [csv, err],
        )

    def test_shell_defaults_keep_full_dataset_and_explicit_generation_budget(self):
        capture = self.output / "python"
        capture.write_text(
            '#!/bin/sh\nprintf "%s|%s|%s|%s\\n" "${LIMIT-unset}" "${NUM_CANDIDATES-unset}" "${MAX_TOKENS-unset}" "$OUTPUT_DIR"\n'
        )
        capture.chmod(0o755)
        env = os.environ.copy()
        for key in ["LIMIT", "NUM_CANDIDATES", "MAX_TOKENS", "OUTPUT_DIR"]:
            env.pop(key, None)
        env["PYTHON_BIN"] = str(capture)
        for script in [
            "run_bird_minidev_qwen3_8b.sh",
            "run_spider2_lite_sqlite_qwen3_8b.sh",
        ]:
            result = subprocess.check_output(
                ["bash", str(ROOT / script)], env=env, text=True
            )
            self.assertTrue(result.startswith("unset|unset|16384|"), result)
            self.assertNotIn("_n135", result)
            self.assertNotIn("_n500", result)


if __name__ == "__main__":
    unittest.main()
