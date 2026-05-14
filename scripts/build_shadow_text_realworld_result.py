#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Route 1 visual results with a text-only Real_World result."
    )
    parser.add_argument("--visual-result", type=Path, required=True)
    parser.add_argument("--textonly-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visual_payload = json.loads(args.visual_result.read_text(encoding="utf-8"))
    textonly_payload = json.loads(args.textonly_result.read_text(encoding="utf-8"))

    visual_results = {item["target_domain"]: item for item in visual_payload["results"]}
    textonly_results = {item["target_domain"]: item for item in textonly_payload["results"]}

    merged_results = []
    for domain_name in ("Art", "Clipart", "Product"):
        merged_results.append(visual_results[domain_name])
    merged_results.append(textonly_results["Real_World"])

    average_top1_accuracy = sum(item["top1_accuracy"] for item in merged_results) / len(merged_results)

    payload = {
        "config": {
            **visual_payload["config"],
            "method": "shadow_text_realworld_merge",
            "hybrid_visual_result": str(args.visual_result),
            "hybrid_textonly_result": str(args.textonly_result),
            "hybrid_realworld_source": "textonly_shadow_session",
            "output": str(args.output),
        },
        "results": merged_results,
        "average_top1_accuracy": average_top1_accuracy,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged result written to: {args.output}")
    print(f"Average Top-1 Accuracy: {average_top1_accuracy * 100:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
