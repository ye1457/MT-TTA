#!/usr/bin/env python3
"""Evaluate zero-shot CLIP on OfficeHome."""

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
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_ROOT,
    DEFAULT_TEMPLATE,
    STANDARD_DOMAINS,
)
from clip_officehome.officehome import build_all_domain_specs, get_domain_spec_by_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot CLIP evaluation for OfficeHome.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--mode", type=str, choices=("all", "single"), default="all")
    parser.add_argument("--target-domain", type=str, choices=STANDARD_DOMAINS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_output_path(output_arg: Path | None) -> Path:
    if output_arg is not None:
        return output_arg
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"officehome_zeroshot_{timestamp}.json"


def print_result_table(results: list[dict]) -> None:
    print("\nEvaluation Summary")
    print("=" * 80)
    for result in results:
        print(
            f"Target: {result['target_domain']:<10} | "
            f"Env: {result['env_name']:<22} | "
            f"Classes: {result['num_classes']:<2} | "
            f"Images: {result['num_images']:<5} | "
            f"Top-1 Acc: {result['top1_accuracy'] * 100:.2f}%"
        )
    print("=" * 80)


def main() -> int:
    args = parse_args()
    if args.mode == "single" and not args.target_domain:
        raise ValueError("--target-domain is required when --mode single is used.")

    from clip_officehome.clip_utils import load_clip_model, resolve_device
    from clip_officehome.evaluator import evaluate_domain, save_results

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
        result = evaluate_domain(
            model=model,
            preprocess=preprocess,
            domain_spec=domain_spec,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            template=args.template,
            device=device,
        )
        result_dict = result.to_dict()
        results.append(result_dict)
        print(
            f"[Done] {result.target_domain}: "
            f"top-1 accuracy = {result.top1_accuracy * 100:.2f}% "
            f"({result.num_images} images)"
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
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": device,
            "template": args.template,
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
