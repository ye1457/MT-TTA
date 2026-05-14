#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-shadow_text_formal_validate}"
GPU_LIST="${GPU_LIST:-2 3 4 5 6 7}"
SEED="${SEED:-0}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

read -r -a GPUS <<< "${GPU_LIST}"
first_gpu="${GPUS[0]}"
first_tag="formal_seed${SEED}_gpu${first_gpu}"
tmux new-session -d -s "${SESSION_NAME}" -n "gpu${first_gpu}" \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_shadow_text_realworld_main.sh ${first_gpu} ${first_tag} ${SEED}; exec bash'"

for ((i = 1; i < ${#GPUS[@]}; i++)); do
  gpu_id="${GPUS[i]}"
  tag="formal_seed${SEED}_gpu${gpu_id}"
  tmux new-window -t "${SESSION_NAME}" -n "gpu${gpu_id}" \
    "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_shadow_text_realworld_main.sh ${gpu_id} ${tag} ${SEED}; exec bash'"
done

echo "Launched formal shadow-text validation session: ${SESSION_NAME}"
for gpu_id in "${GPUS[@]}"; do
  echo "gpu${gpu_id} => shadow_text_realworld seed=${SEED}"
done
