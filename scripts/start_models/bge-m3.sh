#!/usr/bin/env bash
set -euo pipefail

# vLLM嵌入服务配置；客户端EMBEDDING_MODEL需与SERVED_MODEL_NAME一致。
MODEL="${EMBEDDING_MODEL_PATH:-/root/autodl-tmp/models/bge-m3}"
SERVED_MODEL_NAME="${EMBEDDING_MODEL:-$(basename "${MODEL%/}")}"
HOST="${EMBEDDING_HOST:-127.0.0.1}"
PORT="${EMBEDDING_PORT:-9001}"
MAX_MODEL_LEN="${EMBEDDING_MAX_TOKENS:-8192}"
# 每轮调度最多处理的文本序列数；不等同于HTTP请求数（一个请求可含多条文本）。
MAX_NUM_SEQS="${EMBEDDING_MAX_NUM_SEQS:-32}"
# 默认使用FP16；可通过EMBEDDING_DTYPE覆盖。
DTYPE="${EMBEDDING_DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${EMBEDDING_GPU_MEMORY_UTILIZATION:-0.1}"
# 指定物理GPU编号，例如 CUDA_VISIBLE_DEVICES=1；客户端不再选择GPU。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

ARGS=(
    --model "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --runner pooling
    --host "$HOST"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --api-key "${EMBEDDING_API_KEY:-EMPTY}"
    "$@"
)

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
