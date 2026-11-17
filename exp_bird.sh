#!/usr/bin/env bash
set -euo pipefail

# 参数初始化
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"

# OpenAI客户端请求参数；max_tokens包含思考与最终输出。
# SQL候选生成、探索、修订和选择均默认使用0。
export TEMPERATURE="${TEMPERATURE:-0}"
export MAX_TOKENS="${MAX_TOKENS:-32768}"

# 嵌入服务配置（独立vLLM服务；GPU在服务启动时指定）
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:9001/v1}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-EMPTY}"
# 与vLLM的 --served-model-name 一致。
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
# 0表示使用模型原生维度，不向API传递dimensions；BGE-M3和bge-large-en-v1.5原生均为1024维。
# 非0仅用于支持Matryoshka的模型；BGE-M3请保持0，显式传递dimensions会被vLLM拒绝。
export EMBEDDING_DIMENSIONS="${EMBEDDING_DIMENSIONS:-0}"
# 每条嵌入输入的token上限，超过时从右侧截断（保留开头）；不应超过服务的max-model-len。
export EMBEDDING_MAX_TOKENS="${EMBEDDING_MAX_TOKENS:-8192}"

export DATASET="bird"
export SPLIT="${SPLIT:-dev}"
export DATA_ROOT="${DATA_ROOT:-../datasets/bird}"
# 起始运行ID；多次运行时递增末尾数字，相同ID用于续跑。
RUN_ID="${RUN_ID:-run1}"
# 顺序运行次数；例如 RUNS=3 从run1开始执行run1、run2、run3。
RUNS="${RUNS:-1}"
# 同一阶段并发处理的题目数；设为1恢复串行。
WORKERS="${WORKERS:-16}"

# 参数拼接
# 提取脚本参数；输出目录在每次运行时单独设置。
USER_ARGS=()
OUTPUT_BASE="${OUTPUT_DIR:-}"
while (($#)); do
    case "$1" in
        --runs) RUNS="${2:?--runs requires a value}"; shift 2 ;;
        --runs=*) RUNS="${1#--runs=}"; shift ;;
        --output_dir) OUTPUT_BASE="${2:?--output_dir requires a value}"; shift 2 ;;
        --output_dir=*) OUTPUT_BASE="${1#--output_dir=}"; shift ;;
        *) USER_ARGS+=("$1"); shift ;;
    esac
done
if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS / --runs must be a positive integer" >&2
    exit 2
fi
for ((i = 0; i < ${#USER_ARGS[@]}; i++)); do
    case "${USER_ARGS[i]}" in
        --split) SPLIT="${USER_ARGS[i+1]:?--split requires a value}" ;;
        --split=*) SPLIT="${USER_ARGS[i]#--split=}" ;;
    esac
done
if ((RUNS > 1)); then
    if [[ "$RUN_ID" =~ ^(.*[^0-9])?([0-9]+)$ ]]; then
        RUN_PREFIX="${BASH_REMATCH[1]}"
        RUN_START=$((10#${BASH_REMATCH[2]}))
    else
        echo "Multiple runs require RUN_ID to end in a number, e.g. run1" >&2
        exit 2
    fi
fi
MODEL_TAG="${MODEL_NAME//\//_}"
ARGS=(
    --dataset "$DATASET"
    --split "$SPLIT"
    --workers "$WORKERS"
    --data_root "$DATA_ROOT"
    "${USER_ARGS[@]}"
)

# 运行：各run顺序执行；其中一次失败时停止，重跑同一命令可续跑。
for ((run_offset = 0; run_offset < RUNS; run_offset++)); do
    CURRENT_RUN_ID="$RUN_ID"
    if ((RUNS > 1)); then
        CURRENT_RUN_ID="${RUN_PREFIX}$((RUN_START + run_offset))"
    fi
    if [[ -n "$OUTPUT_BASE" ]]; then
        OUTPUT_DIR="$OUTPUT_BASE"
        if ((RUNS > 1)); then
            OUTPUT_DIR="${OUTPUT_BASE%/}/${CURRENT_RUN_ID}"
        fi
    else
        OUTPUT_DIR="$REPO_DIR/outputs/autolink_${DATASET}_${SPLIT}_${MODEL_TAG}_${CURRENT_RUN_ID}"
    fi
    export OUTPUT_DIR
    export AUTOLINK_RUN_ID="$CURRENT_RUN_ID"
    echo "Run $((run_offset + 1))/$RUNS: $CURRENT_RUN_ID -> $OUTPUT_DIR" >&2
    if ((RUNS == 1)); then
        exec python "$REPO_DIR/run_experiment.py" \
            "${ARGS[@]}" --output_dir "$OUTPUT_DIR"
    fi
    python "$REPO_DIR/run_experiment.py" \
        "${ARGS[@]}" --output_dir "$OUTPUT_DIR"
done
