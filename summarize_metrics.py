"""Rebuild metrics from persisted attempts without making model requests."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "run"))
from telemetry import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error("output_dir does not exist")
    summary = summarize(args.output_dir)
    print("Metrics:", args.output_dir.resolve() / "metrics")
    print("Questions:", summary["registered_questions"],
          "LLM attempts:", summary["llm"]["attempts"],
          "Embedding attempts:", summary["embedding"]["attempts"])


if __name__ == "__main__":
    main()
