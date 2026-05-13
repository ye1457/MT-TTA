#!/usr/bin/env python3
"""Evaluate simple prompt-based test-time adaptation for OfficeHome."""

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
    DEFAULT_TTA_CONFIDENCE_THRESHOLD,
    DEFAULT_TTA_CLIPART_LR,
    DEFAULT_TTA_ENTROPY_WEIGHT,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_ROOT,
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
        description="Simple prompt-based test-time adaptation for OfficeHome."
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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_TTA_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_output_path(output_arg: Path | None) -> Path:
    if output_arg is not None:
        return output_arg
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"officehome_tta_{timestamp}.json"


def resolve_effective_tta_lr(
    target_domain: str,
    tta_lr: float,
    clipart_tta_lr: float | None,
) -> float:
    if target_domain == "Clipart" and clipart_tta_lr is not None:
        return clipart_tta_lr
    return tta_lr


def print_result_table(results: list[dict]) -> None:
    print("\nTTA Summary")
    print("=" * 80)
    for result in results:
        print(
            f"Target: {result['target_domain']:<10} | "
            f"Env: {result['env_name']:<22} | "
            f"Classes: {result['num_classes']:<2} | "
            f"Images: {result['num_images']:<5} | "
            f"Selected: {result['num_selected_samples']:<5} | "
            f"Updates: {result['num_updates']:<5} | "
            f"SelRatio: {result['selected_ratio'] * 100:>5.1f}% | "
            f"LR: {result['effective_tta_lr']:.2e} | "
            f"Top-1 Acc: {result['top1_accuracy'] * 100:.2f}%"
        )
    print("=" * 80)


def main() -> int:
    args = parse_args()
    if args.mode == "single" and not args.target_domain:
        raise ValueError("--target-domain is required when --mode single is used.")

    from clip_officehome.clip_utils import load_clip_model, resolve_device
    from clip_officehome.evaluator import save_results
    from clip_officehome.prompt_tta import evaluate_domain_with_prompt_tta

    device = resolve_device(args.device)
    output_path = build_output_path(args.output)

    if args.mode == "all":
        domain_specs = build_all_domain_specs(args.split_root, args.raw_root)
    else:
        domain_specs = [
            get_domain_spec_by_target(args.split_root, args.raw_root, args.target_domain)
        ]

    model, preprocess = load_clip_model(args.model_name, device)

    results = []
    for domain_spec in domain_specs:
        effective_tta_lr = resolve_effective_tta_lr(
            target_domain=domain_spec.target_domain,
            tta_lr=args.tta_lr,
            clipart_tta_lr=args.clipart_tta_lr,
        )
        result = evaluate_domain_with_prompt_tta(
            model=model,
            preprocess=preprocess,
            domain_spec=domain_spec,
            num_workers=args.num_workers,
            device=device,
            tta_lr=effective_tta_lr,
            num_context=args.num_context,
            prompt_init=args.prompt_init,
            pseudo_topk=args.pseudo_topk,
            update_steps=args.update_steps,
            entropy_weight=args.entropy_weight,
            confidence_threshold=args.confidence_threshold,
        )
        result_dict = result.to_dict()
        results.append(result_dict)
        print(
            f"[Done] {result.target_domain}: "
            f"top-1 accuracy = {result.top1_accuracy * 100:.2f}% "
            f"(lr={result.effective_tta_lr:.2e}, {result.num_images} images, "
            f"{result.num_updates} updates, "
            f"{result.num_selected_samples} selected)"
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
            "method": "simple_prompt_tta",
            "tta_lr": args.tta_lr,
            "clipart_tta_lr": args.clipart_tta_lr,
            "num_context": args.num_context,
            "prompt_init": args.prompt_init,
            "pseudo_topk": args.pseudo_topk,
            "update_steps": args.update_steps,
            "entropy_weight": args.entropy_weight,
            "confidence_threshold": args.confidence_threshold,
            "prediction_timing": "pre_update",
            "update_protocol": "online_cumulative",
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
