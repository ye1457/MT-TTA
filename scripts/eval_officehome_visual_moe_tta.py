#!/usr/bin/env python3
"""Evaluate visual-only mixture-of-experts prompt-based test-time adaptation for OfficeHome."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clip_officehome.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MOE_EXPERT_INIT_STD,
    DEFAULT_MOE_ROUTER_TEMPERATURE,
    DEFAULT_MOE_SEED,
    DEFAULT_MOE_SHARED_MIX_WEIGHT,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_ROOT,
    DEFAULT_TEMPLATE,
    DEFAULT_TTA_CLIPART_LR,
    DEFAULT_TTA_ENTROPY_WEIGHT,
    DEFAULT_TTA_LR,
    DEFAULT_TTA_PSEUDO_TOPK,
    DEFAULT_TTA_UPDATE_STEPS,
    DEFAULT_VISUAL_MOE_CONFIDENCE_THRESHOLD,
    DEFAULT_VISUAL_MOE_NUM_PRIVATE_EXPERTS,
    DEFAULT_VISUAL_MOE_NUM_SHARED_EXPERTS,
    DEFAULT_VISUAL_MOE_TOPK_PRIVATE_EXPERTS,
    DEFAULT_VISUAL_TTA_NUM_PROMPTS,
    DEFAULT_VISUAL_TTA_PROMPT_INIT_STD,
    DEFAULT_VISUAL_TTA_PROMPT_SCALE_INIT,
    STANDARD_DOMAINS,
)
from clip_officehome.officehome import build_all_domain_specs, get_domain_spec_by_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual-only mixture-of-experts prompt-based test-time adaptation for OfficeHome."
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--mode", type=str, choices=("all", "single"), default="all")
    parser.add_argument("--target-domain", type=str, choices=STANDARD_DOMAINS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--tta-lr", type=float, default=DEFAULT_TTA_LR)
    parser.add_argument("--clipart-tta-lr", type=float, default=DEFAULT_TTA_CLIPART_LR)
    parser.add_argument("--pseudo-topk", type=int, default=DEFAULT_TTA_PSEUDO_TOPK)
    parser.add_argument("--update-steps", type=int, default=DEFAULT_TTA_UPDATE_STEPS)
    parser.add_argument("--entropy-weight", type=float, default=DEFAULT_TTA_ENTROPY_WEIGHT)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_VISUAL_MOE_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--num-visual-prompts", type=int, default=DEFAULT_VISUAL_TTA_NUM_PROMPTS)
    parser.add_argument(
        "--visual-prompt-init-std",
        type=float,
        default=DEFAULT_VISUAL_TTA_PROMPT_INIT_STD,
    )
    parser.add_argument(
        "--visual-prompt-scale-init",
        type=float,
        default=DEFAULT_VISUAL_TTA_PROMPT_SCALE_INIT,
    )
    parser.add_argument(
        "--num-shared-experts",
        type=int,
        default=DEFAULT_VISUAL_MOE_NUM_SHARED_EXPERTS,
    )
    parser.add_argument(
        "--num-private-experts",
        type=int,
        default=DEFAULT_VISUAL_MOE_NUM_PRIVATE_EXPERTS,
    )
    parser.add_argument(
        "--topk-private-experts",
        type=int,
        default=DEFAULT_VISUAL_MOE_TOPK_PRIVATE_EXPERTS,
    )
    parser.add_argument("--shared-mix-weight", type=float, default=DEFAULT_MOE_SHARED_MIX_WEIGHT)
    parser.add_argument("--router-temperature", type=float, default=DEFAULT_MOE_ROUTER_TEMPERATURE)
    parser.add_argument("--expert-init-std", type=float, default=DEFAULT_MOE_EXPERT_INIT_STD)
    parser.add_argument("--seed", type=int, default=DEFAULT_MOE_SEED)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_output_path(output_arg: Path | None) -> Path:
    if output_arg is not None:
        return output_arg
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"officehome_visual_moe_tta_{timestamp}.json"


def resolve_effective_tta_lr(
    target_domain: str,
    tta_lr: float,
    clipart_tta_lr: float | None,
) -> float:
    if target_domain == "Clipart" and clipart_tta_lr is not None:
        return clipart_tta_lr
    return tta_lr


def print_result_table(results: list[dict]) -> None:
    print("\nVisual MoE TTA Summary")
    print("=" * 120)
    for result in results:
        print(
            f"Target: {result['target_domain']:<10} | "
            f"Images: {result['num_images']:<5} | "
            f"VPrompts: {result['num_visual_prompts']:<2} | "
            f"Shared: {result['num_shared_experts']:<2} | "
            f"Private: {result['num_private_experts']:<2} | "
            f"TopRoute: {str(result['most_selected_private_expert_ids'][:3]):<12} | "
            f"LR: {result['effective_tta_lr']:.2e} | "
            f"Top-1 Acc: {result['top1_accuracy'] * 100:.2f}%"
        )
    print("=" * 120)


def main() -> int:
    args = parse_args()
    if args.mode == "single" and not args.target_domain:
        raise ValueError("--target-domain is required when --mode single is used.")

    from clip_officehome.clip_utils import load_clip_model, resolve_device
    from clip_officehome.evaluator import save_results
    from clip_officehome.visual_moe_prompt_tta import PersistentVisualMoEPromptTTASession

    device = resolve_device(args.device)
    output_path = build_output_path(args.output)

    if args.mode == "all":
        domain_specs = build_all_domain_specs(args.split_root, args.raw_root)
    else:
        domain_specs = [
            get_domain_spec_by_target(args.split_root, args.raw_root, args.target_domain)
        ]

    model, preprocess = load_clip_model(args.model_name, device)
    session = PersistentVisualMoEPromptTTASession(
        model=model,
        preprocess=preprocess,
        reference_domain_spec=domain_specs[0],
        num_workers=args.num_workers,
        device=device,
        template=args.template,
        num_visual_prompts=args.num_visual_prompts,
        num_shared_experts=args.num_shared_experts,
        num_private_experts=args.num_private_experts,
        prompt_init_std=args.visual_prompt_init_std,
        prompt_scale_init=args.visual_prompt_scale_init,
        expert_init_std=args.expert_init_std,
        seed=args.seed,
    )

    results = []
    for domain_spec in domain_specs:
        effective_tta_lr = resolve_effective_tta_lr(
            target_domain=domain_spec.target_domain,
            tta_lr=args.tta_lr,
            clipart_tta_lr=args.clipart_tta_lr,
        )
        result = session.evaluate_domain(
            domain_spec=domain_spec,
            tta_lr=effective_tta_lr,
            pseudo_topk=args.pseudo_topk,
            update_steps=args.update_steps,
            entropy_weight=args.entropy_weight,
            confidence_threshold=args.confidence_threshold,
            shared_mix_weight=args.shared_mix_weight,
            topk_private_experts=args.topk_private_experts,
            router_temperature=args.router_temperature,
        )
        results.append(result.to_dict())
        print(
            f"[Done] {result.target_domain}: "
            f"top-1 accuracy = {result.top1_accuracy * 100:.2f}% "
            f"(lr={result.effective_tta_lr:.2e}, "
            f"selected={result.num_selected_samples}, "
            f"updates={result.num_updates})"
        )

    average_accuracy = sum(item["top1_accuracy"] for item in results) / len(results)
    print_result_table(results)
    print(f"Average Top-1 Accuracy: {average_accuracy * 100:.2f}%")
    print(f"Using device: {device}")

    payload = {
        "config": {
            "split_root": str(args.split_root),
            "raw_root": str(args.raw_root),
            "model_name": args.model_name,
            "mode": args.mode,
            "target_domain": args.target_domain,
            "num_workers": args.num_workers,
            "device": device,
            "method": "visual_moe_prompt_tta",
            "template": args.template,
            "adaptation_protocol": "continual_persistent_visual_prompt_bank",
            "routing_mode": "sample_wise",
            "tta_lr": args.tta_lr,
            "clipart_tta_lr": args.clipart_tta_lr,
            "pseudo_topk": args.pseudo_topk,
            "update_steps": args.update_steps,
            "entropy_weight": args.entropy_weight,
            "confidence_threshold": args.confidence_threshold,
            "num_visual_prompts": args.num_visual_prompts,
            "visual_prompt_init_std": args.visual_prompt_init_std,
            "visual_prompt_scale_init": args.visual_prompt_scale_init,
            "num_shared_experts": args.num_shared_experts,
            "num_private_experts": args.num_private_experts,
            "topk_private_experts": args.topk_private_experts,
            "shared_mix_weight": args.shared_mix_weight,
            "router_temperature": args.router_temperature,
            "expert_init_std": args.expert_init_std,
            "seed": args.seed,
            "task_sequence": [spec.target_domain for spec in domain_specs],
            "output": str(output_path),
        },
        "results": results,
        "average_top1_accuracy": average_accuracy,
    }
    save_results(output_path, payload)
    print(f"Saved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
