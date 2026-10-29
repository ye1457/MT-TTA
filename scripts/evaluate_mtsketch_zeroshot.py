"""Frozen CLIP-B/16 control on the complete MT-Sketch4-20K task stream.

The control uses batch size one and the same prepared membership, task-local
class indices, prompt names and image loader as the adaptation experiment.
No teacher, optimizer, prompt ensemble or test-time update is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "CLIP"))

from clip_officehome.officehome import normalize_class_name
from clip_officehome.protocol import (WEIGHTS, canonical_hash, file_sha256,
                                     require, require_new_output, save_json)


TEMPLATE = "a photo of a {class}"
LOGIT_SCALE = 100.0
STREAM = "balanced_round_robin"
DEFINITION_FILES = ("manifest.jsonl", "metadata/classes.json", "streams.json",
                    "metadata/protocol.json", "benchmark.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot(source_hashes):
    observed = dict(source_hashes())
    entry = Path(__file__).resolve().relative_to(WORKSPACE).as_posix()
    digest = file_sha256(Path(__file__))
    require(entry not in observed or observed[entry] == digest,
            "Source provider and entrypoint hash disagree")
    observed[entry] = digest
    return observed


def definition_snapshot(root):
    return {name: file_sha256(Path(root) / name) for name in DEFINITION_FILES}


def control_config(prepared, weight_path, source_start, definitions):
    require(len(prepared.records) == len(prepared.order) == 20000,
            "The control requires the complete 20,000-image benchmark")
    require(sorted(prepared.order) == list(range(20000)),
            "The stream must visit every manifest member exactly once")
    require(len(prepared.classes) == 4 and
            Counter(r["task_id"] for r in prepared.records) == dict.fromkeys(prepared.classes, 5000),
            "Expected four tasks with exactly 5,000 images each")
    mapping = {}
    for task, entries in prepared.classes.items():
        require([entry["class_index"] for entry in entries] == list(range(len(entries))),
                "Task-local class indices must be contiguous and ordered")
        names = [normalize_class_name(entry["class_name"]) for entry in entries]
        require(list(prepared.prompt_names[task]) == names,
                "Control prompt names differ from the shared native-name normalization")
        mapping[task] = [{**entry, "prompt_name": name} for entry, name in zip(entries, names, strict=True)]
    return {
        "schema": "mt_sketch_zeroshot_config_v1",
        "benchmark": prepared.config["benchmark"], "dataset": "MT-Sketch4-20K",
        "method": "frozen_CLIP_B16_single_template", "model": "ViT-B/16",
        "seed": 0, "full_data": True, "num_scored_images": 20000,
        "images_per_task": 5000, "stream": STREAM, "task_known": True,
        "independent_run_initialization": True, "inference_batch_size": 1,
        "template": TEMPLATE, "logit_scale": LOGIT_SCALE,
        "prompt_names": prepared.prompt_names, "class_mapping": mapping,
        "prompt_name_normalization": "normalize_class_name: replace underscores with spaces; strip; retain full native synonyms",
        "prediction_source": "frozen_student_only", "scoring": "pre_update",
        "adaptation": False, "adaptation_updates": 0, "teacher_used": False,
        "teacher_output_fusion": False, "prompt_ensemble": False,
        "previews": 0, "extra_random_augmentation": False,
        "numeric_policy": "unchanged_local_CLIP_CUDA_dtypes; no_added_AMP_or_precision_override",
        "model_path": str(weight_path), "model_sha256": WEIGHTS["student"][1],
        "data_definition_sha256": definitions,
        "shared_method_config_canonical_sha256": canonical_hash(prepared.config),
        "source_hashes": source_start,
    }


def recount_predictions(prepared, path):
    """Validate the full stored stream and call the frozen independent scorer."""
    predictions = []
    correct, totals = Counter(), Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for global_index, (sample_index, line) in enumerate(zip(prepared.order, handle, strict=True)):
            result = json.loads(line)
            record = prepared.records[sample_index]
            task = record["task_id"]
            require(result["global_index"] == global_index and
                    type(result["sample_index"]) is int and result["sample_index"] == sample_index and
                    result["task_id"] == task and result["sample_id"] == record["sample_id"] and
                    result["image_path"] == record["image_path"] and
                    result["image_sha256"] == record["file_sha256"] and
                    result["label"] == record["class_index"], "Control trace/member identity mismatch")
            require(result["schema"] == "mt_sketch_zeroshot_prediction_v1" and
                    result["prediction_source"] == "frozen_student_only" and result["scoring"] == "pre_update" and
                    result["adaptation_updates_before_prediction"] == 0,
                    "Control trace has an invalid model source or update rule")
            logits, probabilities = result["student_logits"], result["student_probabilities"]
            require(len(logits) == len(probabilities) == len(prepared.classes[task]) and
                    all(type(v) in (int, float) and math.isfinite(v) for v in logits + probabilities) and
                    all(0 <= value <= 1 for value in probabilities), "Invalid task-local student output")
            maximum = max(logits)
            exponentials = [math.exp(value - maximum) for value in logits]
            denominator = sum(exponentials)
            require(max(abs(probability - value / denominator)
                        for probability, value in zip(probabilities, exponentials, strict=True)) <= 0.001 and
                    abs(sum(probabilities) - 1.0) <= 0.005,
                    "Stored probabilities do not match the student logits")
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            require(type(result["predicted_class_index"]) is int and
                    result["predicted_class_index"] == predicted and
                    result["confidence"] == max(probabilities), "Student argmax/confidence mismatch")
            predictions.append({"sample_index": sample_index, "task_id": task,
                                "predicted_class_index": predicted})
            totals[task] += 1
            correct[task] += int(predicted == record["class_index"])
    require(len(predictions) == len(prepared.records), "Incomplete control prediction stream")
    independent = prepared.benchmark.score(predictions)
    require(independent["num_scored"] == len(predictions) and
            set(independent["per_task"]) == set(prepared.classes), "Independent scorer task/count mismatch")
    for task, result in independent["per_task"].items():
        require(result["total"] == totals[task] and result["correct"] == correct[task] and
                math.isclose(result["accuracy_percent"], correct[task] / totals[task] * 100, abs_tol=1e-10),
                "Independent scorer disagrees with trace recount")
    macro = sum(correct[task] / totals[task] * 100 for task in prepared.classes) / len(prepared.classes)
    require(math.isclose(independent["macro_accuracy_percent"], macro, abs_tol=1e-10),
            "Independent scorer macro average mismatch")
    return independent


def run(args):
    output = require_new_output(args.output_dir).expanduser().resolve()
    require(str(args.gpu).isdigit(), "--gpu must identify one physical CUDA GPU")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    scored = 0
    metadata = {"schema": "mt_sketch_zeroshot_metadata_v1", "status": "PREPARING",
                "phase": "prepare_data", "pid": os.getpid(), "started_utc": utc_now(),
                "gpu": str(args.gpu), "seed": 0, "teacher_used": False,
                "teacher_output_fusion": False, "adaptation_updates": 0,
                "prediction_rule": "frozen_student_only_pre_update", "inference_batch_size": 1}
    save_json(output / "metadata.json", metadata)
    try:
        # Set visibility before importing the adapter or any torch-dependent helper.
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        from clip_officehome.multitask_sketch import prepare_mtsketch, load_image, source_hashes
        sources = source_snapshot(source_hashes)
        metadata["source_hashes_start"] = sources
        save_json(output / "metadata.json", metadata)
        prepared = prepare_mtsketch(args.data_root, hash_images=True)
        definitions = definition_snapshot(prepared.root)
        model_cache = Path(args.model_cache).expanduser().resolve()
        method = prepared.config["baseline_method"]
        require(method["student"] == "ViT-B/16" and
                (method["student_weights"], method["student_weights_sha256"]) == WEIGHTS["student"],
                "This control requires the declared frozen OpenAI CLIP-B/16 weights")
        weight_path = model_cache / method["student_weights"]
        require(weight_path.is_file(), "Missing cached CLIP-B/16 weights: " + str(weight_path))
        weight_start = file_sha256(weight_path)
        require(weight_start == WEIGHTS["student"][1], "Student weight SHA256 mismatch")
        config = control_config(prepared, weight_path, sources, definitions)
        save_json(output / "config.json", config)
        save_json(output / "input_manifest.json", {
            "inventory": prepared.inventory, "records": prepared.records,
            "order": prepared.order, "classes": prepared.classes,
        })
        metadata.update(phase="load_model", benchmark=config["benchmark"], data=prepared.inventory,
                        config_canonical_sha256=canonical_hash(config),
                        data_definition_hashes_start=definitions,
                        model_path=str(weight_path), model_weights_sha256_start=weight_start)
        save_json(output / "metadata.json", metadata)
        import numpy as np
        import torch
        import torchvision
        import PIL
        from clip_officehome.clip_utils import build_text_features, freeze_clip_parameters, load_clip_model
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.set_num_threads(1)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "The full control requires exactly one visible CUDA GPU; no CPU fallback")
        model, preprocess = load_clip_model(str(weight_path), "cuda:0", model_cache)
        require(file_sha256(weight_path) == weight_start, "Student weights changed during loading")
        freeze_clip_parameters(model)
        require(all(not p.requires_grad for p in model.parameters()) and not model.training,
                "The control model must remain frozen and in evaluation mode")
        require(model.visual.input_resolution == 224, "Unexpected CLIP-B/16 input resolution")
        metadata.update(status="RUNNING", phase="score", gpu_name=torch.cuda.get_device_name(0),
                        versions={"python": sys.version, "torch": torch.__version__,
                                  "torchvision": torchvision.__version__, "pillow": PIL.__version__,
                                  "numpy": np.__version__}, preprocess=str(preprocess),
                        model_dtype=str(model.dtype), num_scored_images=20000)
        save_json(output / "metadata.json", metadata)
        task_visits, online_correct = Counter(), Counter()

        def progress(status="RUNNING"):
            state = {"schema": "mt_sketch_progress_v1", "status": status, "scored": scored,
                     "total": len(prepared.order), "per_task_scored": dict(task_visits),
                     "elapsed_seconds": time.monotonic() - started, "updated_utc": utc_now()}
            save_json(output / "progress.json", state)
            print("PROGRESS " + json.dumps(state), flush=True)

        progress()
        with torch.no_grad(), (output / "predictions.jsonl").open("x", encoding="utf-8") as trace:
            text_features = {task: build_text_features(model, prepared.prompt_names[task], TEMPLATE, "cuda:0")
                             for task in prepared.classes}
            for global_index, sample_index in enumerate(prepared.order):
                record = prepared.records[sample_index]
                task = record["task_id"]
                # Bind the bytes actually about to be read, not only the initial inventory.
                require(file_sha256(prepared.root / record["image_path"]) == record["file_sha256"],
                        "An input image changed after preparation")
                pixels = load_image(prepared, sample_index)
                require(pixels.mode == "RGB", "Shared image loader must return RGB")
                image = preprocess(pixels).unsqueeze(0).to("cuda:0")
                features = model.encode_image(image)
                features = features / features.norm(dim=-1, keepdim=True)
                logits = LOGIT_SCALE * features @ text_features[task].t()
                probabilities = logits.softmax(dim=-1)
                require(tuple(logits.shape) == (1, len(prepared.classes[task])), "Unexpected batch/class shape")
                predicted = int(probabilities.argmax(dim=-1).item())
                confidence = float(probabilities.max(dim=-1).values.item())
                # Ground truth is accessed only after the frozen prediction, for logging/scoring.
                label = record["class_index"]
                trace.write(json.dumps({
                    "schema": "mt_sketch_zeroshot_prediction_v1", "global_index": global_index,
                    "sample_index": sample_index, "task_id": task, "sample_id": record["sample_id"],
                    "image_path": record["image_path"], "image_sha256": record["file_sha256"],
                    "predicted_class_index": predicted, "confidence": confidence, "label": label,
                    "student_logits": logits.detach().cpu()[0].tolist(),
                    "student_probabilities": probabilities.detach().cpu()[0].tolist(),
                    "prediction_source": "frozen_student_only", "scoring": "pre_update",
                    "adaptation_updates_before_prediction": 0,
                }, ensure_ascii=False, allow_nan=False) + "\n")
                scored += 1
                task_visits[task] += 1
                online_correct[task] += int(predicted == label)
                if scored % 1000 == 0 or scored == len(prepared.order):
                    trace.flush()
                    progress()
        metadata.update(phase="verify", scored=scored)
        save_json(output / "metadata.json", metadata)
        require(scored == 20000 and task_visits == dict.fromkeys(prepared.classes, 5000),
                "Incomplete full-data control")
        independent = recount_predictions(prepared, output / "predictions.jsonl")
        require(all(online_correct[task] == result["correct"]
                    for task, result in independent["per_task"].items()), "Online and independently recounted scores disagree")
        require(all(file_sha256(prepared.root / record["image_path"]) == record["file_sha256"]
                    for record in prepared.records), "An input image changed during the control")
        source_end = source_snapshot(source_hashes)
        definition_end = definition_snapshot(prepared.root)
        weight_end = file_sha256(weight_path)
        require(source_end == sources, "Evaluation sources changed during the control")
        require(definition_end == definitions, "Frozen dataset definitions changed during the control")
        require(weight_end == weight_start, "Student weight file changed during the control")
        require(all(not p.requires_grad for p in model.parameters()) and not model.training,
                "Control model no longer frozen/evaluation-only")
        summary = {
            "schema": "mt_sketch_zeroshot_summary_v1", "status": "PASS", "benchmark": config["benchmark"],
            "method": config["method"], "seed": 0, "stream": STREAM, "num_scored_images": scored,
            "per_task": {task: {"num_images": result["total"], "correct": result["correct"],
                                "top1_percent": result["accuracy_percent"]}
                         for task, result in independent["per_task"].items()},
            "macro_top1_percent": independent["macro_accuracy_percent"],
            "micro_top1_percent": sum(r["correct"] for r in independent["per_task"].values()) / scored * 100,
            "teacher_used": False, "teacher_output_fusion": False, "adaptation_updates": 0,
            "inference_batch_size": 1, "prompt_ensemble": False, "preview_image_visits": 0,
            "prediction_rule": "frozen_student_only_pre_update",
            "independent_scorer_result": independent, "all_predictions_recounted": True,
            "source_hashes_unchanged": True, "dataset_definitions_unchanged": True,
            "all_image_hashes_rechecked_after_run": True,
            "model_weights_unchanged": True, "elapsed_seconds": time.monotonic() - started,
        }
        metadata.update(status="COMPLETE", phase="complete", finished_utc=utc_now(),
                        elapsed_seconds=summary["elapsed_seconds"], source_hashes_end=source_end,
                        data_definition_hashes_end=definition_end, model_weights_sha256_end=weight_end,
                        all_image_hashes_rechecked_after_run=True,
                        independent_scorer_verified=True, scored=scored)
        save_json(output / "summary.json", summary)
        save_json(output / "metadata.json", metadata)
        progress("COMPLETE")
        files = ("config.json", "input_manifest.json", "metadata.json", "progress.json",
                 "predictions.jsonl", "summary.json")
        save_json(output / "completion.json", {
            "schema": "mt_sketch_zeroshot_completion_v1", "status": "PASS",
            "files": {name: file_sha256(output / name) for name in files},
            "config_canonical_sha256": canonical_hash(config), "num_scored_images": scored,
            "independent_scorer_verified": True,
        })
        return summary
    except BaseException as exc:
        metadata.update(status="FAILED", failed_utc=utc_now(), error=repr(exc),
                        traceback=traceback.format_exc(), scored=scored,
                        elapsed_seconds=time.monotonic() - started)
        save_json(output / "metadata.json", metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=WORKSPACE / "models")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
