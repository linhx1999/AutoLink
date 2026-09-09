#!/usr/bin/env bash

set -euo pipefail

MODEL="/root/autodl-tmp/models/Qwen3-8B"
SERVED_MODEL_NAME="$(basename "${MODEL%/}")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="32768"
TENSOR_PARALLEL_SIZE="1"
GPU_MEMORY_UTILIZATION="0.90"
MAX_NUM_SEQS="4"
DTYPE="auto"
REASONING_PARSER="qwen3"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 启动 vLLM 服务
# 通过 setsid 将 vLLM 及其张量并行 Worker 子进程放入同一进程组，便于在
# Ctrl+C / kill 时统一清理，避免 Worker 成为孤儿进程继续占用显存。
cleanup() {
    trap - INT TERM EXIT
    set +e
    echo ""
    echo "=========================================="
    echo " 正在停止 vLLM 服务并释放显存..."
    echo "=========================================="
    # 先优雅终止整个进程组（主进程 + 全部 Worker）
    kill -TERM "-$VLLM_PGID" 2>/dev/null
    # 最多等待 10 秒让其优雅退出
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "-$VLLM_PGID" 2>/dev/null || break
        sleep 1
    done
    # 仍有残留则强制终止整个进程组
    kill -KILL "-$VLLM_PGID" 2>/dev/null
    # 仅终止本服务进程组，不删除其他程序可能正在使用的共享内存。
    echo "服务已停止，显存已释放"
}

setsid "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype "$DTYPE" \
    --reasoning-parser "$REASONING_PARSER" \
    &
VLLM_PGID=$!

trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

wait "$VLLM_PGID"
