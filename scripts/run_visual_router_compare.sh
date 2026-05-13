#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-MT-TTA}"

ROUTER_SOURCE="${1:-current_text_moe}"
GPU_ID="${2:-0}"
TAG="${3:-manual}"
VISUAL_PROMPT_SCALE_INIT="${4:-0.25}"
VISUAL_TOPK_PRIVATE_EXPERTS="${5:-1}"
VISUAL_SHARED_MIX_WEIGHT="${6:-0.5}"
NUM_VISUAL_PROMPTS="${7:-2}"
NUM_VISUAL_SHARED_EXPERTS="${8:-2}"
NUM_VISUAL_PRIVATE_EXPERTS="${9:-4}"
DISABLE_VISUAL_DOMAINS="${10:-}"

case "${ROUTER_SOURCE}" in
  fixed_zeroshot_text|current_text_moe)
    ;;
  *)
    echo "Unsupported router source: ${ROUTER_SOURCE}" >&2
    echo "Expected one of: fixed_zeroshot_text, current_text_moe" >&2
    exit 1
    ;;
esac

mkdir -p "${RESULTS_DIR}"

OUTPUT_JSON="${RESULTS_DIR}/joint_text_visual_router_${ROUTER_SOURCE}_${TAG}.json"
LOG_FILE="${RESULTS_DIR}/joint_text_visual_router_${ROUTER_SOURCE}_${TAG}.log"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python scripts/eval_officehome_moe_tta.py \
  --mode all \
  --device cuda:0 \
  --output "${OUTPUT_JSON}" \
  --num-visual-prompts "${NUM_VISUAL_PROMPTS}" \
  --visual-prompt-scale-init "${VISUAL_PROMPT_SCALE_INIT}" \
  --num-visual-shared-experts "${NUM_VISUAL_SHARED_EXPERTS}" \
  --num-visual-private-experts "${NUM_VISUAL_PRIVATE_EXPERTS}" \
  --visual-topk-private-experts "${VISUAL_TOPK_PRIVATE_EXPERTS}" \
  --visual-shared-mix-weight "${VISUAL_SHARED_MIX_WEIGHT}" \
  --visual-router-source "${ROUTER_SOURCE}" \
  ${DISABLE_VISUAL_DOMAINS:+--disable-visual-domains ${DISABLE_VISUAL_DOMAINS}} \
  |& tee "${LOG_FILE}"
