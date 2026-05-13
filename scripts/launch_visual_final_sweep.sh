#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_visual_router_compare.sh"
SESSION_NAME="${1:-vis_final_sweep}"
TAG_PREFIX="${2:-finalsweep_20260512}"
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
  "rwoff rw_s023_m043 0.23 0.43 2 2 4 1"
  "rwoff rw_s023_m045 0.23 0.45 2 2 4 1"
  "rwoff rw_s023_m047 0.23 0.47 2 2 4 1"
  "rwoff rw_s024_m044 0.24 0.44 2 2 4 1"
  "rwoff rw_s024_m046 0.24 0.46 2 2 4 1"
  "rwoff rw_s024_m048 0.24 0.48 2 2 4 1"
  "rwoff rw_s025_m043 0.25 0.43 2 2 4 1"
  "rwoff rw_s025_m044 0.25 0.44 2 2 4 1"
  "rwoff rw_s025_m046 0.25 0.46 2 2 4 1"
  "rwoff rw_s025_m047 0.25 0.47 2 2 4 1"
  "rwoff rw_s026_m044 0.26 0.44 2 2 4 1"
  "rwoff rw_s026_m045 0.26 0.45 2 2 4 1"
  "rwoff rw_s026_m046 0.26 0.46 2 2 4 1"
  "rwoff rw_s026_m048 0.26 0.48 2 2 4 1"
  "rwoff rw_s027_m044 0.27 0.44 2 2 4 1"
  "rwoff rw_s027_m046 0.27 0.46 2 2 4 1"
  "rwoff rw_s027_m048 0.27 0.48 2 2 4 1"
  "rwoff rw_s025_m045_p3 0.25 0.45 3 2 4 1"
  "rwoff rw_s025_m045_p2_priv5 0.25 0.45 2 2 5 1"
  "rwoff rw_s025_m045_p2_sh3 0.25 0.45 2 3 4 1"
  "full full_s028_m049 0.28 0.49 2 2 4 1"
  "full full_s028_m050 0.28 0.50 2 2 4 1"
  "full full_s028_m051 0.28 0.51 2 2 4 1"
  "full full_s029_m048 0.29 0.48 2 2 4 1"
  "full full_s029_m049 0.29 0.49 2 2 4 1"
  "full full_s029_m050 0.29 0.50 2 2 4 1"
  "full full_s029_m051 0.29 0.51 2 2 4 1"
  "full full_s029_m052 0.29 0.52 2 2 4 1"
  "full full_s030_m048 0.30 0.48 2 2 4 1"
  "full full_s030_m049 0.30 0.49 2 2 4 1"
  "full full_s030_m050 0.30 0.50 2 2 4 1"
  "full full_s030_m051 0.30 0.51 2 2 4 1"
  "full full_s030_m052 0.30 0.52 2 2 4 1"
  "full full_s031_m049 0.31 0.49 2 2 4 1"
  "full full_s031_m050 0.31 0.50 2 2 4 1"
  "full full_s031_m051 0.31 0.51 2 2 4 1"
  "full full_s029_m050_p3 0.29 0.50 3 2 4 1"
  "full full_s029_m050_p2_priv5 0.29 0.50 2 2 5 1"
  "full full_s029_m050_p2_sh3 0.29 0.50 2 3 4 1"
  "full full_s029_m050_p1 0.29 0.50 1 2 4 1"
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
  read -r mode job_tag scale mix num_prompts num_shared num_private topk <<< "${JOBS[job_index]}"
  full_tag="${job_tag}_${TAG_PREFIX}"
  if [[ "${mode}" == "rwoff" ]]; then
    WINDOW_COMMANDS[gpu_index]+=" && ./scripts/run_visual_router_compare.sh current_text_moe ${gpu_id} ${full_tag} ${scale} ${topk} ${mix} ${num_prompts} ${num_shared} ${num_private} Real_World"
  else
    WINDOW_COMMANDS[gpu_index]+=" && ./scripts/run_visual_router_compare.sh current_text_moe ${gpu_id} ${full_tag} ${scale} ${topk} ${mix} ${num_prompts} ${num_shared} ${num_private}"
  fi
done

tmux new-session -d -s "${SESSION_NAME}" -n "gpu${GPUS[0]}" "${WINDOW_COMMANDS[0]}"
for ((i = 1; i < ${#GPUS[@]}; i++)); do
  tmux new-window -t "${SESSION_NAME}:" -n "gpu${GPUS[i]}" "${WINDOW_COMMANDS[i]}"
done

echo "Launched session: ${SESSION_NAME}"
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  echo "gpu${GPUS[i]} => ${WINDOW_COMMANDS[i]}"
done
