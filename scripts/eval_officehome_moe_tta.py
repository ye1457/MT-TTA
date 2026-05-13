#!/usr/bin/env python3
"""Evaluate mixture-of-experts prompt-based test-time adaptation for OfficeHome."""

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
    DEFAULT_DISTILL_KL_WEIGHT,
    DEFAULT_MOE_CONFIDENCE_THRESHOLD,
    DEFAULT_MOE_EXPERT_INIT_STD,
    DEFAULT_MOE_NUM_PRIVATE_EXPERTS,
    DEFAULT_MOE_NUM_SHARED_EXPERTS,
    DEFAULT_MOE_ROUTER_TEMPERATURE,
    DEFAULT_MOE_ROUTING_BUFFER_SIZE,
    DEFAULT_MOE_ROUTING_SCORE,
    DEFAULT_MOE_SEED,
    DEFAULT_MOE_SHARED_MIX_WEIGHT,
    DEFAULT_MOE_SHARED_ANCHOR_WEIGHT,
    DEFAULT_MOE_TOPK_PRIVATE_EXPERTS,
    DEFAULT_MOE_VISUAL_NUM_PRIVATE_EXPERTS,
    DEFAULT_MOE_VISUAL_NUM_PROMPTS,
    DEFAULT_MOE_VISUAL_ROUTER_SOURCE,
    DEFAULT_MOE_VISUAL_NUM_SHARED_EXPERTS,
    DEFAULT_MOE_VISUAL_PROMPT_INIT_STD,
    DEFAULT_MOE_VISUAL_PROMPT_SCALE_INIT,
    DEFAULT_MOE_VISUAL_ROUTER_TEMPERATURE,
    DEFAULT_MOE_VISUAL_SHARED_MIX_WEIGHT,
    DEFAULT_MOE_VISUAL_TOPK_PRIVATE_EXPERTS,
    DEFAULT_TTA_CLIPART_LR,
    DEFAULT_TTA_ENTROPY_WEIGHT,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_ROOT,
    DEFAULT_TEACHER_CONFIDENCE_THRESHOLD,
    DEFAULT_TEACHER_CE_WEIGHT,
    DEFAULT_TEACHER_MODEL_NAME,
    DEFAULT_TEACHER_SUPERVISION,
    DEFAULT_TEACHER_TEMPLATE,
    DEFAULT_TEACHER_TEMPERATURE,
    DEFAULT_TTA_LR,
    DEFAULT_TTA_NUM_CONTEXT,
    DEFAULT_TTA_PSEUDO_TOPK,
    DEFAULT_TTA_PROMPT_INIT,
    DEFAULT_TTA_UPDATE_STEPS,
    STANDARD_DOMAINS,
)
from clip_officehome.officehome import build_all_domain_specs, get_domain_spec_by_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixture-of-experts prompt-based test-time adaptation for OfficeHome."
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--mode", type=str, choices=("all", "single"), default="all")
    parser.add_argument("--target-domain", type=str, choices=STANDARD_DOMAINS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tta-lr", type=float, default=DEFAULT_TTA_LR)
    parser.add_argument("--clipart-tta-lr", type=float, default=DEFAULT_TTA_CLIPART_LR)
    parser.add_argument("--num-context", type=int, default=DEFAULT_TTA_NUM_CONTEXT)
    parser.add_argument("--prompt-init", type=str, default=DEFAULT_TTA_PROMPT_INIT)
    parser.add_argument("--pseudo-topk", type=int, default=DEFAULT_TTA_PSEUDO_TOPK)
    parser.add_argument("--update-steps", type=int, default=DEFAULT_TTA_UPDATE_STEPS)
    parser.add_argument("--entropy-weight", type=float, default=DEFAULT_TTA_ENTROPY_WEIGHT)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_MOE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--shared-mix-weight", type=float, default=DEFAULT_MOE_SHARED_MIX_WEIGHT)
    parser.add_argument("--num-shared-experts", type=int, default=DEFAULT_MOE_NUM_SHARED_EXPERTS)
    parser.add_argument("--num-private-experts", type=int, default=DEFAULT_MOE_NUM_PRIVATE_EXPERTS)
    parser.add_argument("--topk-private-experts", type=int, default=DEFAULT_MOE_TOPK_PRIVATE_EXPERTS)
    parser.add_argument("--routing-buffer-size", type=int, default=DEFAULT_MOE_ROUTING_BUFFER_SIZE)
    parser.add_argument("--routing-score", type=str, default=DEFAULT_MOE_ROUTING_SCORE, choices=("confidence",))
    parser.add_argument("--router-temperature", type=float, default=DEFAULT_MOE_ROUTER_TEMPERATURE)
    parser.add_argument("--expert-init-std", type=float, default=DEFAULT_MOE_EXPERT_INIT_STD)
    parser.add_argument("--shared-anchor-weight", type=float, default=DEFAULT_MOE_SHARED_ANCHOR_WEIGHT)
    parser.add_argument("--num-visual-prompts", type=int, default=DEFAULT_MOE_VISUAL_NUM_PROMPTS)
    parser.add_argument("--visual-prompt-init-std", type=float, default=DEFAULT_MOE_VISUAL_PROMPT_INIT_STD)
    parser.add_argument("--visual-prompt-scale-init", type=float, default=DEFAULT_MOE_VISUAL_PROMPT_SCALE_INIT)
    parser.add_argument("--num-visual-shared-experts", type=int, default=DEFAULT_MOE_VISUAL_NUM_SHARED_EXPERTS)
    parser.add_argument("--num-visual-private-experts", type=int, default=DEFAULT_MOE_VISUAL_NUM_PRIVATE_EXPERTS)
    parser.add_argument(
        "--visual-topk-private-experts",
        type=int,
        default=DEFAULT_MOE_VISUAL_TOPK_PRIVATE_EXPERTS,
    )
    parser.add_argument(
        "--visual-shared-mix-weight",
        type=float,
        default=DEFAULT_MOE_VISUAL_SHARED_MIX_WEIGHT,
    )
    parser.add_argument(
        "--visual-router-temperature",
        type=float,
        default=DEFAULT_MOE_VISUAL_ROUTER_TEMPERATURE,
    )
    parser.add_argument(
        "--visual-router-source",
        type=str,
        choices=("fixed_zeroshot_text", "current_text_moe"),
        default=DEFAULT_MOE_VISUAL_ROUTER_SOURCE,
    )
    parser.add_argument(
        "--disable-visual-domains",
        nargs="*",
        default=(),
        choices=STANDARD_DOMAINS,
    )
    parser.add_argument(
        "--teacher-supervision",
        type=str,
        choices=("off", "aux_pseudolabel", "distill"),
        default=DEFAULT_TEACHER_SUPERVISION,
    )
    parser.add_argument("--enable-teacher-distill", action="store_true")
    parser.add_argument("--teacher-model-name", type=str, default=DEFAULT_TEACHER_MODEL_NAME)
    parser.add_argument("--teacher-device", type=str, default="auto")
    parser.add_argument("--teacher-template", type=str, default=DEFAULT_TEACHER_TEMPLATE)
    parser.add_argument("--teacher-temperature", type=float, default=DEFAULT_TEACHER_TEMPERATURE)
    parser.add_argument(
        "--teacher-confidence-threshold",
        type=float,
        default=DEFAULT_TEACHER_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--teacher-ce-weight", type=float, default=DEFAULT_TEACHER_CE_WEIGHT)
    parser.add_argument("--distill-kl-weight", type=float, default=DEFAULT_DISTILL_KL_WEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_MOE_SEED)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_output_path(output_arg: Path | None) -> Path:
    if output_arg is not None:
        return output_arg
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"officehome_moe_tta_{timestamp}.json"


def resolve_effective_tta_lr(
    target_domain: str,
    tta_lr: float,
    clipart_tta_lr: float | None,
) -> float:
    if target_domain == "Clipart" and clipart_tta_lr is not None:
        return clipart_tta_lr
    return tta_lr


def print_result_table(results: list[dict]) -> None:
    print("\nMoE TTA Summary")
    print("=" * 120)
    for result in results:
        print(
            f"Target: {result['target_domain']:<10} | "
            f"Images: {result['num_images']:<5} | "
            f"Route: {result['routing_mode']:<14} | "
            f"Shared: {result['num_shared_experts']:<2} | "
            f"Private: {result['num_private_experts']:<2} | "
            f"TopRoute: {str(result['selected_private_expert_ids']):<12} | "
            f"LR: {result['effective_tta_lr']:.2e} | "
            f"Top-1 Acc: {result['top1_accuracy'] * 100:.2f}%"
        )
    print("=" * 120)


def resolve_teacher_supervision(args: argparse.Namespace) -> str:
    if args.teacher_supervision != "off":
        return args.teacher_supervision
    if args.enable_teacher_distill:
        return "distill"
    return "off"


def main() -> int:
    args = parse_args()
    if args.mode == "single" and not args.target_domain:
        raise ValueError("--target-domain is required when --mode single is used.")

    from clip_officehome.clip_utils import load_clip_model, resolve_device
    from clip_officehome.evaluator import save_results
    from clip_officehome.moe_prompt_tta import PersistentMoEPromptTTASession

    device = resolve_device(args.device)
    teacher_device = device if args.teacher_device == "auto" else resolve_device(args.teacher_device)
    output_path = build_output_path(args.output)
    teacher_supervision = resolve_teacher_supervision(args)

    if args.mode == "all":
        domain_specs = build_all_domain_specs(args.split_root, args.raw_root)
    else:
        domain_specs = [
            get_domain_spec_by_target(args.split_root, args.raw_root, args.target_domain)
        ]

    model, preprocess = load_clip_model(args.model_name, device)
    teacher_model = None
    teacher_preprocess = None
    if teacher_supervision != "off":
        teacher_model, teacher_preprocess = load_clip_model(args.teacher_model_name, teacher_device)
    session = PersistentMoEPromptTTASession(
        model=model,
        preprocess=preprocess,
        reference_domain_spec=domain_specs[0],
        num_workers=args.num_workers,
        device=device,
        num_context=args.num_context,
        prompt_init=args.prompt_init,
        num_shared_experts=args.num_shared_experts,
        num_private_experts=args.num_private_experts,
        expert_init_std=args.expert_init_std,
        seed=args.seed,
        num_visual_prompts=args.num_visual_prompts,
        visual_prompt_init_std=args.visual_prompt_init_std,
        visual_prompt_scale_init=args.visual_prompt_scale_init,
        num_visual_shared_experts=args.num_visual_shared_experts,
        num_visual_private_experts=args.num_visual_private_experts,
        visual_router_source=args.visual_router_source,
        disable_visual_domains=args.disable_visual_domains,
        teacher_model=teacher_model,
        teacher_preprocess=teacher_preprocess,
        teacher_device=teacher_device,
        teacher_template=args.teacher_template,
        teacher_temperature=args.teacher_temperature,
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
            routing_buffer_size=args.routing_buffer_size,
            routing_score=args.routing_score,
            router_temperature=args.router_temperature,
            shared_anchor_weight=args.shared_anchor_weight,
            visual_topk_private_experts=args.visual_topk_private_experts,
            visual_shared_mix_weight=args.visual_shared_mix_weight,
            visual_router_temperature=args.visual_router_temperature,
            teacher_supervision=teacher_supervision,
            teacher_ce_weight=args.teacher_ce_weight,
            teacher_confidence_threshold=args.teacher_confidence_threshold,
            distill_kl_weight=args.distill_kl_weight,
        )
        results.append(result.to_dict())
        print(
            f"[Done] {result.target_domain}: "
            f"top-1 accuracy = {result.top1_accuracy * 100:.2f}% "
            f"(lr={result.effective_tta_lr:.2e}, top_routed_private={result.selected_private_expert_ids})"
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
            "method": "moe_prompt_tta",
            "adaptation_protocol": "continual_persistent_expert_bank",
            "routing_mode": "task_fixed",
            "visual_routing_mode": (
                "sample_wise"
                if (args.num_visual_shared_experts + args.num_visual_private_experts) > 0
                else "off"
            ),
            "tta_lr": args.tta_lr,
            "clipart_tta_lr": args.clipart_tta_lr,
            "num_context": args.num_context,
            "prompt_init": args.prompt_init,
            "pseudo_topk": args.pseudo_topk,
            "update_steps": args.update_steps,
            "entropy_weight": args.entropy_weight,
            "confidence_threshold": args.confidence_threshold,
            "shared_mix_weight": args.shared_mix_weight,
            "num_shared_experts": args.num_shared_experts,
            "num_private_experts": args.num_private_experts,
            "topk_private_experts": args.topk_private_experts,
            "routing_buffer_size": args.routing_buffer_size,
            "routing_score": args.routing_score,
            "router_temperature": args.router_temperature,
            "expert_init_std": args.expert_init_std,
            "shared_anchor_weight": args.shared_anchor_weight,
            "num_visual_prompts": args.num_visual_prompts,
            "visual_prompt_init_std": args.visual_prompt_init_std,
            "visual_prompt_scale_init": args.visual_prompt_scale_init,
            "num_visual_shared_experts": args.num_visual_shared_experts,
            "num_visual_private_experts": args.num_visual_private_experts,
            "visual_topk_private_experts": args.visual_topk_private_experts,
            "visual_shared_mix_weight": args.visual_shared_mix_weight,
            "visual_router_temperature": args.visual_router_temperature,
            "visual_router_source": args.visual_router_source,
            "disable_visual_domains": list(args.disable_visual_domains),
            "teacher_supervision": teacher_supervision,
            "teacher_model_name": args.teacher_model_name if teacher_supervision != "off" else None,
            "teacher_device": teacher_device if teacher_supervision != "off" else None,
            "teacher_template": args.teacher_template if teacher_supervision != "off" else None,
            "teacher_temperature": args.teacher_temperature if teacher_supervision != "off" else None,
            "teacher_confidence_threshold": (
                args.teacher_confidence_threshold if teacher_supervision == "distill" else None
            ),
            "teacher_ce_weight": args.teacher_ce_weight if teacher_supervision == "aux_pseudolabel" else None,
            "distill_kl_weight": args.distill_kl_weight if teacher_supervision == "distill" else None,
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
