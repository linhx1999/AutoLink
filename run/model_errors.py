from pathlib import Path


class ExperimentFailure(BaseException):
    """Abort native unlimited retries; do not score an incomplete stage."""


class OutputTokenLimit(ExperimentFailure):
    """One model call exhausted its output budget."""

    def __init__(self, call_record: Path, max_tokens: int, usage: dict | None):
        super().__init__("Model output reached max_tokens")
        self.call_record = str(call_record.resolve())
        self.max_tokens = max_tokens
        self.usage = usage
