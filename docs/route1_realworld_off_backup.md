# Route 1 Backup

This document records the current backup configuration for the best stable
`Route 1` result in this repository.

## Stable Backup Result

- Average top-1 accuracy: `86.1152%`
- Seed: `0`
- Protocol: text MoE + visual MoE on `Art/Clipart/Product`, visual MoE disabled on `Real_World`

Reference result file:

- `results/route1_realworld_off_repeat_seed0_run1_gpu2.json`

## Core Configuration

- `visual_router_source=current_text_moe`
- `num_visual_prompts=2`
- `num_visual_shared_experts=2`
- `num_visual_private_experts=4`
- `visual_topk_private_experts=1`
- `visual_prompt_scale_init=0.25`
- `visual_shared_mix_weight=0.44`
- `disable_visual_domains=["Real_World"]`
- `seed=0`

## Reproduction Entry Points

- Main launcher: `scripts/run_route1_realworld_off_main.sh`
- Multi-GPU validation launcher: `scripts/launch_route1_realworld_off_validate.sh`

## Notes

- This backup is stable only when the seed is fixed to `0`.
- With other seeds, the same configuration shows noticeable variation.
