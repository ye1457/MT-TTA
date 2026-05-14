# Shadow-Text Real_World Backup

This document records the current backup configuration for the best stable
`shadow-text Real_World` result in this repository.

## Stable Backup Result

- Average top-1 accuracy: `86.4882%`
- Seed: `0`
- Protocol: text MoE + visual MoE on `Art/Clipart/Product`, `Real_World` evaluated from a shadow text-only continual session

Reference result files:

- `results/shadow_text_realworld_formal_seed0_gpu2.json`
- `results/shadow_text_realworld_formal_seed0_gpu3.json`
- `results/shadow_text_realworld_formal_seed0_gpu4.json`
- `results/shadow_text_realworld_formal_seed0_gpu6.json`
- `results/shadow_text_realworld_formal_seed0_gpu7.json`

## Core Configuration

- `visual_router_source=current_text_moe`
- `num_visual_prompts=2`
- `num_visual_shared_experts=2`
- `num_visual_private_experts=4`
- `visual_topk_private_experts=1`
- `visual_prompt_scale_init=0.25`
- `visual_shared_mix_weight=0.44`
- `shadow_text_domains=["Real_World"]`
- `seed=0`

## Reproduction Entry Points

- Main launcher: `scripts/run_shadow_text_realworld_main.sh`
- Multi-GPU validation launcher: `scripts/launch_shadow_text_formal_validate.sh`

## Domain Metrics

- `Art`: `85.4965%`
- `Clipart`: `72.2108%`
- `Product`: `94.0752%`
- `Real_World`: `94.1703%`

## Comparison Against Route 1

- `shadow-text Real_World`: `86.4882%`
- `Route 1`: `86.1152%`
- absolute gain: `+0.3729%`

## Notes

- The `Art/Clipart/Product` domains come from the primary joint text+visual session.
- The `Real_World` domain comes from the shadow text-only continual session.
- Formal validation on GPUs `2/3/4/6/7` reproduced identical results.
