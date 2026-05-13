#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_route1_realworld_off_main.sh"
SESSION_NAME="${1:-route1_realworld_off}"
GPU_LIST="${GPU_LIST:-2 3 4 5 6 7}"
SEED_OFFSET="${SEED_OFFSET:-0}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Missing executable run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

first_gpu="${GPUS[0]}"
first_seed="${SEED_OFFSET}"
first_tag="seed${first_seed}_gpu${first_gpu}"
tmux new-session -d -s "${SESSION_NAME}" -n "gpu${first_gpu}" \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_route1_realworld_off_main.sh ${first_gpu} ${first_tag} ${first_seed}; exec bash'"

for ((i = 1; i < ${#GPUS[@]}; i++)); do
  gpu_id="${GPUS[i]}"
  seed=$((SEED_OFFSET + i))
  tag="seed${seed}_gpu${gpu_id}"
  tmux new-window -t "${SESSION_NAME}:" -n "gpu${gpu_id}" \
    "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_route1_realworld_off_main.sh ${gpu_id} ${tag} ${seed}; exec bash'"
done

echo "Launched session: ${SESSION_NAME}"
for ((i = 0; i < ${#GPUS[@]}; i++)); do
  gpu_id="${GPUS[i]}"
  seed=$((SEED_OFFSET + i))
  echo "gpu${gpu_id} => route1_realworld_off seed=${seed}"
done
