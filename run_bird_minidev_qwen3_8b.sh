#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
# OpenAI客户端请求参数；max_tokens包含思考与最终输出。
export TEMPERATURE="${TEMPERATURE:-0}"
export MAX_TOKENS="${MAX_TOKENS:-16384}"
export DATASET="bird_minidev"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/workspace/kouan/datasets/bird/MINIDEV}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/autolink_bird_minidev_${MODEL_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" "$REPO_DIR/run_experiment.py" "$@"
