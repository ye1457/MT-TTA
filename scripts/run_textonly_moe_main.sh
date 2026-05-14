#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-MT-TTA}"

GPU_ID="${1:-0}"
TAG="${2:-main}"
SEED="${3:-0}"
shift $(( $# >= 3 ? 3 : $# ))
EXTRA_ARGS=("$@")

OUTPUT_JSON="${RESULTS_DIR}/textonly_moe_${TAG}.json"
LOG_FILE="${RESULTS_DIR}/textonly_moe_${TAG}.log"

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
  --num-visual-shared-experts 0 \
  --num-visual-private-experts 0 \
  "${EXTRA_ARGS[@]}" \
  |& tee "${LOG_FILE}"
