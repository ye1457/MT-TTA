#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_visual_router_compare.sh"
SESSION_NAME="${1:-vis_full_refine}"
TAG_PREFIX="${2:-full_refine_20260512}"
GPU_LIST="${GPU_LIST:-2 3 4 5 6 7}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Missing executable run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi

JOBS=(
  "p2_sc028_m050 0.28 1 0.50 2 2 4"
  "p2_sc029_m050 0.29 1 0.50 2 2 4"
  "p2_sc031_m050 0.31 1 0.50 2 2 4"
  "p2_sc032_m050 0.32 1 0.50 2 2 4"
  "p2_sc033_m050 0.33 1 0.50 2 2 4"
  "p2_sc030_m045 0.30 1 0.45 2 2 4"
  "p2_sc030_m048 0.30 1 0.48 2 2 4"
  "p2_sc030_m052 0.30 1 0.52 2 2 4"
  "p2_sc030_m055 0.30 1 0.55 2 2 4"
  "p2_sc028_m048 0.28 1 0.48 2 2 4"
  "p2_sc032_m048 0.32 1 0.48 2 2 4"
  "p2_sc031_m045 0.31 1 0.45 2 2 4"
)

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

declare -a WINDOW_COMMANDS=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  WINDOW_COMMANDS[i]="cd ${PROJECT_ROOT}"
done

for ((job_index = 0; job_index < ${#JOBS[@]}; job_index++)); do
  gpu_index=$((job_index % ${#GPUS[@]}))
  gpu_id="${GPUS[gpu_index]}"
  read -r job_tag scale topk mix num_prompts num_shared num_private <<< "${JOBS[job_index]}"
  full_tag="${job_tag}_${TAG_PREFIX}"
  WINDOW_COMMANDS[gpu_index]+=" && ./scripts/run_visual_router_compare.sh current_text_moe ${gpu_id} ${full_tag} ${scale} ${topk} ${mix} ${num_prompts} ${num_shared} ${num_private}"
done

tmux new-session -d -s "${SESSION_NAME}" -n "gpu${GPUS[0]}" "${WINDOW_COMMANDS[0]}"
for ((i = 1; i < ${#GPUS[@]}; i++)); do
  tmux new-window -t "${SESSION_NAME}:" -n "gpu${GPUS[i]}" "${WINDOW_COMMANDS[i]}"
done

echo "Launched session: ${SESSION_NAME}"
