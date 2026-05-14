#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-next_routes_overnight}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

tmux new-session -d -s "${SESSION_NAME}" -n shadow_text_realworld \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_shadow_text_realworld_main.sh 2 overnight_seed0 0; exec bash'"

tmux new-window -t "${SESSION_NAME}" -n rw_no_update \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_route1_realworld_off_main.sh 3 routeB_rw_no_update_seed0 0 --disable-update-domains Real_World; exec bash'"

tmux new-window -t "${SESSION_NAME}" -n rw_thr090 \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_route1_realworld_off_main.sh 4 routeB_rw_thr090_seed0 0 --domain-confidence-threshold-overrides Real_World=0.90; exec bash'"

tmux new-window -t "${SESSION_NAME}" -n rw_thr095 \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_route1_realworld_off_main.sh 5 routeB_rw_thr095_seed0 0 --domain-confidence-threshold-overrides Real_World=0.95; exec bash'"

tmux new-window -t "${SESSION_NAME}" -n rw_thr099 \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_route1_realworld_off_main.sh 6 routeB_rw_thr099_seed0 0 --domain-confidence-threshold-overrides Real_World=0.99; exec bash'"

tmux new-window -t "${SESSION_NAME}" -n rw_no_update_thr095 \
  "bash -lc 'cd ${PROJECT_ROOT} && ./scripts/run_route1_realworld_off_main.sh 7 routeB_rw_no_update_thr095_seed0 0 --disable-update-domains Real_World --domain-confidence-threshold-overrides Real_World=0.95; exec bash'"

echo "Launched session: ${SESSION_NAME}"
echo "  GPU2 -> shadow text Real_World merge route"
echo "  GPU3 -> Real_World no-update route"
echo "  GPU4 -> Real_World threshold 0.90 route"
echo "  GPU5 -> Real_World threshold 0.95 route"
echo "  GPU6 -> Real_World threshold 0.99 route"
echo "  GPU7 -> Real_World no-update + threshold 0.95 route"
