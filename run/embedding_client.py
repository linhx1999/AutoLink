"""Shared vLLM embeddings configuration for documents and retrieval queries."""
from dataclasses import asdict, dataclass
import math
import os
import threading
import time
from pathlib import Path
import telemetry
from urllib.parse import urlparse


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    model: str
    dimensions: int
    max_tokens: int

    @classmethod
    def from_env(cls):
        config = cls(
            os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
            os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            int(os.environ.get("EMBEDDING_DIMENSIONS", "0")),
            int(os.environ.get("EMBEDDING_MAX_TOKENS", "512")),
        )
        if config.dimensions < 0 or config.max_tokens <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be >= 0 and EMBEDDING_MAX_TOKENS > 0")
        if not config.model or urlparse(config.base_url).scheme not in ("http", "https"):
            raise ValueError("Invalid embedding model or service URL")
        return config

    def signature(self):
        return {"backend": "vllm", "truncation_side": "right", **asdict(self)}


class EmbeddingClient:
    def __init__(self, config=None, client=None):
        self.config = config or EmbeddingConfig.from_env()
        if client is None:
            from openai import DefaultHttpxClient, OpenAI
            local = urlparse(self.config.base_url).hostname in ("localhost", "127.0.0.1", "::1")
            client = OpenAI(
                base_url=self.config.base_url,
                api_key=os.environ.get("EMBEDDING_API_KEY", "EMPTY"),
                max_retries=0,
                http_client=DefaultHttpxClient(trust_env=not local),
            )
        self.client = client
        self._dimension = None
        self._lock = threading.Lock()

    def embed(self, texts):
        if not texts:
            return []
        request = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
            "extra_body": {
                "truncate_prompt_tokens": self.config.max_tokens,
                "truncation_side": "right",
            },
        }
        # 0 omits dimensions and uses the model's native size (1024 for BGE-M3/BGE-large-en-v1.5).
        # Non-Matryoshka models require this request field to be omitted.
        if self.config.dimensions:
            request["dimensions"] = self.config.dimensions
        output = telemetry.output_directory()
        fields = telemetry.call_metadata("embedding")
        record = telemetry.Record(
            Path(output) / "embedding_calls", fields.get("stage") or "embedding",
            **fields, request=telemetry.embedding_request_metadata(request), endpoint=self.config.base_url,
            ttft_seconds=None, ttft_unavailable_reason="embedding_response_not_streamed",
        ) if output else None
        started = time.perf_counter()
        received = False
        usage = None
        api_seconds = None
        try:
            response = self.client.embeddings.create(**request)
            received = True
            api_seconds = time.perf_counter() - started
            usage = telemetry.usage_dict(getattr(response, "usage", None))
            vectors = self._validate_response(response, texts)
        except BaseException as exc:
            if record:
                record.finish(
                    "returned" if received else "request_failure" if isinstance(exc, Exception) else "interrupted",
                    outcome="invalid_response" if received else "request_failure",
                    error_type=type(exc).__name__, error=str(exc), usage=usage,
                    api_seconds=api_seconds if api_seconds is not None else time.perf_counter() - started,
                )
            raise
        if record:
            record.finish("returned", outcome="success", usage=usage,
                          api_seconds=api_seconds, dimensions=len(vectors[0]))
        return vectors

    def _validate_response(self, response, texts):
        rows = sorted(response.data, key=lambda row: row.index)
        if [row.index for row in rows] != list(range(len(texts))):
            raise ValueError("Embedding service returned missing or duplicate indices")
        vectors = [row.embedding for row in rows]
        dimension = len(vectors[0])
        if not dimension or any(len(v) != dimension or not all(math.isfinite(x) for x in v)
                                for v in vectors):
            raise ValueError("Embedding service returned invalid vectors")
        if self.config.dimensions and dimension != self.config.dimensions:
            raise ValueError("Embedding response dimension differs from configured dimensions")
        with self._lock:
            if self._dimension is not None and dimension != self._dimension:
                raise ValueError("Embedding service changed output dimensions")
            self._dimension = dimension
        return vectors

    def encode(self, texts, convert_to_numpy=True):
        single = isinstance(texts, str)
        vectors = self.embed([texts] if single else texts)
        if convert_to_numpy:
            import numpy as np
            result = np.asarray(vectors, dtype=np.float32)
        else:
            result = vectors
        return result[0] if single else result
