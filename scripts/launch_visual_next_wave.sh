#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_visual_router_compare.sh"
SESSION_NAME="${1:-vis_next_wave}"
TAG_PREFIX="${2:-wave_20260512}"
GPU_LIST="${GPU_LIST:-2 3 4 5 6 7}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Missing executable run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 6 ]]; then
  echo "GPU_LIST must contain exactly 6 GPU ids for this launcher." >&2
  exit 1
fi

RW_JOBS=(
  "rwoff_p2_sc025_m050 0.25 1 0.50 2 2 4"
  "rwoff_p2_sc030_m050 0.30 1 0.50 2 2 4"
  "rwoff_p2_sc025_m045 0.25 1 0.45 2 2 4"
  "rwoff_p2_sc030_m045 0.30 1 0.45 2 2 4"
  "rwoff_p2_sc030_m048 0.30 1 0.48 2 2 4"
  "rwoff_p2_sc030_m052 0.30 1 0.52 2 2 4"
)

FULL_JOBS=(
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
for ((i = 0; i < 6; i++)); do
  gpu_id="${GPUS[i]}"
  read -r rw_tag rw_scale rw_topk rw_mix rw_prompts rw_shared rw_private <<< "${RW_JOBS[i]}"
  full_index_a=$((i * 2))
  full_index_b=$((i * 2 + 1))
  read -r full_tag_a full_scale_a full_topk_a full_mix_a full_prompts_a full_shared_a full_private_a <<< "${FULL_JOBS[full_index_a]}"
  read -r full_tag_b full_scale_b full_topk_b full_mix_b full_prompts_b full_shared_b full_private_b <<< "${FULL_JOBS[full_index_b]}"
  WINDOW_COMMANDS[i]="cd ${PROJECT_ROOT}"
  WINDOW_COMMANDS[i]+=" && ./scripts/run_visual_router_compare.sh current_text_moe ${gpu_id} ${rw_tag}_${TAG_PREFIX} ${rw_scale} ${rw_topk} ${rw_mix} ${rw_prompts} ${rw_shared} ${rw_private} Real_World"
  WINDOW_COMMANDS[i]+=" && ./scripts/run_visual_router_compare.sh current_text_moe ${gpu_id} ${full_tag_a}_${TAG_PREFIX} ${full_scale_a} ${full_topk_a} ${full_mix_a} ${full_prompts_a} ${full_shared_a} ${full_private_a}"
  WINDOW_COMMANDS[i]+=" && ./scripts/run_visual_router_compare.sh current_text_moe ${gpu_id} ${full_tag_b}_${TAG_PREFIX} ${full_scale_b} ${full_topk_b} ${full_mix_b} ${full_prompts_b} ${full_shared_b} ${full_private_b}"
done

tmux new-session -d -s "${SESSION_NAME}" -n "gpu${GPUS[0]}" "${WINDOW_COMMANDS[0]}"
for ((i = 1; i < 6; i++)); do
  tmux new-window -t "${SESSION_NAME}:" -n "gpu${GPUS[i]}" "${WINDOW_COMMANDS[i]}"
done

echo "Launched session: ${SESSION_NAME}"
