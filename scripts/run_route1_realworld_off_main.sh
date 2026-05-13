#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-MT-TTA}"

GPU_ID="${1:-0}"
TAG="${2:-main}"
SEED="${3:-0}"

OUTPUT_JSON="${RESULTS_DIR}/route1_realworld_off_${TAG}.json"
LOG_FILE="${RESULTS_DIR}/route1_realworld_off_${TAG}.log"

mkdir -p "${RESULTS_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

conda run --no-capture-output -n "${CONDA_ENV_NAME}" python scripts/eval_officehome_moe_tta.py \
  --mode all \
  --device cuda:0 \
  --output "${OUTPUT_JSON}" \
  --seed "${SEED}" \
  --num-visual-prompts 2 \
  --visual-prompt-scale-init 0.25 \
  --num-visual-shared-experts 2 \
  --num-visual-private-experts 4 \
  --visual-topk-private-experts 1 \
  --visual-shared-mix-weight 0.44 \
  --visual-router-source current_text_moe \
  --disable-visual-domains Real_World \
  |& tee "${LOG_FILE}"
