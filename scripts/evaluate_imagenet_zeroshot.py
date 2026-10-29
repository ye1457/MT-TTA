"""Frozen single-template CLIP-B/16 control on the exact ImageNet OOD stream."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CLIP"))

import argparse
import json
import os
import time
from clip_officehome.imagenet_ood import COUNTS, prepare_imagenet_data
from clip_officehome.protocol import file_sha256, require, require_new_output, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(COUNTS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=Path("models"))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = require_new_output(args.output_dir)
    config, specs, records, inventory = prepare_imagenet_data(args.benchmark, args.data_root, True)
    method = config["baseline_method"]
    model_path = args.model_cache.resolve() / method["student_weights"]
    require(file_sha256(model_path) == method["student_weights_sha256"], "Unexpected student weights")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    from torch.utils.data import DataLoader
    from clip_officehome.clip_utils import build_text_features, freeze_clip_parameters, load_clip_model
    from clip_officehome.evaluator import OfficeHomeImageDataset
    torch.set_num_threads(1)
    torch.manual_seed(0)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One visible CUDA device required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    model, preprocess = load_clip_model(str(model_path), "cuda:0", args.model_cache)
    freeze_clip_parameters(model)
    spec = specs[args.benchmark]
    rows = records[args.benchmark]
    dataset = OfficeHomeImageDataset(spec.examples, preprocess)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    save_json(output / "input_manifest.json", {"inventory": inventory, "records": records})
    save_json(output / "config.json", {"benchmark": args.benchmark, "model": "ViT-B/16", "seed": 0,
        "template": "a photo of a {class}", "class_names": list(spec.prompt_class_names),
        "num_classes": len(spec.class_names), "class_metadata_sha256": config["data"]["class_metadata_sha256"],
        "model_sha256": method["student_weights_sha256"], "logit_scale": 100.0,
        "adaptation": False, "teacher_used": False, "prompt_ensemble": False,
        "inference_batch_size": 64, "order": inventory["order"],
        "script_sha256": file_sha256(Path(__file__))})
    correct = total = 0
    with torch.no_grad(), (output / "predictions.jsonl").open("x") as trace:
        text = build_text_features(model, spec.prompt_class_names, "a photo of a {class}", "cuda:0")
        for images, labels in loader:
            image = model.encode_image(images.cuda(non_blocking=True))
            image = image / image.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * image @ text.t()).softmax(dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            for label, pred, conf in zip(labels.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist(), strict=True):
                row = rows[total]
                require(label == row["label_index"], "Loader/manifest label mismatch")
                trace.write(json.dumps({"index": total, "relative_path": row["relative_path"],
                    "image_sha256": row["sha256"], "label": label, "prediction": pred,
                    "confidence": conf, "prediction_source": "frozen_student_only"}) + "\n")
                correct += label == pred
                total += 1
            if total % 1024 == 0 or total == len(rows):
                trace.flush()
                print(json.dumps({"scored": total, "total": len(rows), "elapsed_seconds": time.time()-started}), flush=True)
    require(total == COUNTS[args.benchmark], "Incomplete zero-shot control")
    checked = [json.loads(line) for line in (output / "predictions.jsonl").open()]
    require(len(checked) == total and sum(r["prediction"] == r["label"] for r in checked) == correct,
            "Control trace recount mismatch")
    summary = {"status": "PASS", "benchmark": args.benchmark, "method": "frozen_CLIP_B16_single_template",
        "num_scored_images": total, "correct": correct, "top1_percent": 100*correct/total,
        "teacher_output_fusion": False, "adaptation_updates": 0, "elapsed_seconds": time.time()-started,
        "ordered_content_sha256": inventory["ordered_content_sha256"]}
    save_json(output / "summary.json", summary)
    save_json(output / "completion.json", {"status": "PASS", **{p: file_sha256(output / p)
        for p in ["summary.json", "config.json", "input_manifest.json", "predictions.jsonl"]}})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
