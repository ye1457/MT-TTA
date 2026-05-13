#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_visual_router_compare.sh"
SESSION_NAME="${1:-vis_realworld_off}"
TAG_PREFIX="${2:-rw_off_20260512}"
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
  "rwoff_p2_sc025_m050 0.25 1 0.50 2 2 4"
  "rwoff_p2_sc030_m050 0.30 1 0.50 2 2 4"
  "rwoff_p2_sc025_m045 0.25 1 0.45 2 2 4"
  "rwoff_p2_sc030_m045 0.30 1 0.45 2 2 4"
  "rwoff_p2_sc030_m048 0.30 1 0.48 2 2 4"
  "rwoff_p2_sc030_m052 0.30 1 0.52 2 2 4"
)

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n "gpu${GPUS[0]}" "cd ${PROJECT_ROOT}"
for ((i = 1; i < ${#GPUS[@]}; i++)); do
  tmux new-window -t "${SESSION_NAME}:" -n "gpu${GPUS[i]}" "cd ${PROJECT_ROOT}"
done

for ((job_index = 0; job_index < ${#JOBS[@]}; job_index++)); do
  gpu_index=$((job_index % ${#GPUS[@]}))
  gpu_id="${GPUS[gpu_index]}"
  read -r job_tag scale topk mix num_prompts num_shared num_private <<< "${JOBS[job_index]}"
  full_tag="${job_tag}_${TAG_PREFIX}"
  cmd="./scripts/run_visual_router_compare.sh current_text_moe ${gpu_id} ${full_tag} ${scale} ${topk} ${mix} ${num_prompts} ${num_shared} ${num_private} Real_World"
  tmux send-keys -t "${SESSION_NAME}:$gpu_index" "${cmd}" C-m
done

echo "Launched session: ${SESSION_NAME}"
