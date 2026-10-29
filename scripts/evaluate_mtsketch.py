"""Full, task-known MT-Sketch4-20K continual evaluation (seed0, student only)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CLIP"))

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import time

from clip_officehome.multitask_sketch import TASKS, FROZEN, prepare_mtsketch, load_image, source_hashes
from clip_officehome.protocol import canonical_hash, file_sha256, require, require_new_output, save_json


def run(args):
    prepared = prepare_mtsketch(args.data_root, hash_images=not args.dry_run)
    config = prepared.config
    if args.dry_run:
        return {"status": "DRY_RUN_PASS_NO_MODEL", "config": config, "inventory": prepared.inventory}
    require(args.output_dir is not None, "--output-dir required for formal evaluation")
    output = require_new_output(args.output_dir)
    require(str(args.gpu).isdigit(), "--gpu must be a single CUDA index")
    method, cache = config["baseline_method"], args.model_cache.expanduser().resolve()
    paths = {role: cache / method[role + "_weights"] for role in ("student", "teacher")}
    for role, path in paths.items():
        require(file_sha256(path) == method[role + "_weights_sha256"], "Unexpected " + role + " weights")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    import torchvision
    from clip_officehome.clip_utils import load_clip_model
    from clip_officehome.evaluation import prompt_hash, session_kwargs
    from clip_officehome.multitask_session import MultiTaskPromptTTASession
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One visible CUDA GPU required")
    frozen_sources = source_hashes()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    save_json(output / "config.json", config)
    save_json(output / "input_manifest.json", {"inventory": prepared.inventory,
        "records": prepared.records, "order": prepared.order, "classes": prepared.classes})
    metadata = {"status": "RUNNING", "benchmark": config["benchmark"], "config_sha256": canonical_hash(config),
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "gpu": str(args.gpu),
        "gpu_name": torch.cuda.get_device_name(0), "source_hashes": frozen_sources,
        "model_hashes": {p.name: method[r + "_weights_sha256"] for r, p in paths.items()},
        "versions": {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__},
        "teacher_output_fusion": False, "teacher_output_calibration": False,
        "output_rule": "pre_update_student_probability_argmax_only", "routes": {},
        "prompt_reset_on_task_switch": False, "optimizer_reset_on_task_switch": False,
        "number_of_prompt_banks": 1, "all_previews_before_adaptation": True}
    save_json(output / "metadata.json", metadata)
    try:
        student, preprocess = load_clip_model(str(paths["student"]), "cuda:0", cache)
        teacher, teacher_preprocess = load_clip_model(str(paths["teacher"]), "cuda:0", cache)
        require(student.visual.input_resolution == teacher.visual.input_resolution == 224, "Unexpected input size")
        session = MultiTaskPromptTTASession(**session_kwargs(config, student, preprocess, teacher,
            teacher_preprocess, prepared.reference_spec(), "primary"), config=config)
        initial_hash = prompt_hash(session)
        metadata["initial_prompt_sha256"] = initial_hash
        metadata["preprocess"] = {"student": str(preprocess), "teacher": str(teacher_preprocess)}
        for task in TASKS:
            session.register_task(task, [str(c["class_key"]) for c in prepared.classes[task]], prepared.prompt_names[task])
        require(prompt_hash(session) == initial_hash, "Class registration changed prompt parameters")
        require(all(not p.requires_grad for model in (student, teacher) for p in model.parameters()), "Backbones trainable")
        for task in TASKS:
            indices = prepared.task_indices(task)[:64]
            images = torch.stack([preprocess(load_image(prepared, i)) for i in indices])
            before = prompt_hash(session)
            route = session.route_task(task, images)
            require(prompt_hash(session) == before == initial_hash, "Preview changed prompt state")
            metadata["routes"][task] = {**route, "sample_indices": indices,
                "before_prompt_sha256": before, "after_prompt_sha256": prompt_hash(session),
                "preview_images": 64, "parameter_updates": 0}
            del images
        metadata["optimizer_initial_state"] = session.optimizer_audit()
        save_json(output / "metadata.json", metadata)
        selected, scored = Counter(), Counter()
        previous_hash = initial_hash
        predictions = []
        total_updates = 0
        with (output / "predictions.jsonl").open("x") as trace, (output / "updates.jsonl").open("x") as updates:
            for global_index, sample_index in enumerate(prepared.order):
                record = prepared.records[sample_index]
                task = record["task_id"]
                image = load_image(prepared, sample_index)
                student_tensor = preprocess(image).unsqueeze(0)
                teacher_tensor = teacher_preprocess(image).unsqueeze(0)
                before = prompt_hash(session)
                require(before == previous_hash, "Prompts changed/reset at task switch")
                captured = []

                def record_prediction(row):
                    # Ground truth is added only to the external audit/scoring
                    # record; it is never an argument or return to the model.
                    require(len(captured) == 0, "Repeated pre-update callback")
                    require(row["updates_before"] == total_updates, "Wrong pre-update global step")
                    require(len(row["student_logits"]) == len(row["student_probabilities"]) == TASKS[task],
                            "Prediction used the wrong task class space")
                    require(row["task_id"] == task and row["prediction_source"] == "student_only" and
                            row["scoring"] == "pre_update", "Invalid prediction source or timing")
                    saved = {**row, "index": global_index, "sample_index": sample_index,
                        "image_sha256": record["file_sha256"], "label": record["class_index"],
                        "prompt_sha256_before": before}
                    trace.write(json.dumps(saved, allow_nan=False) + "\n")
                    captured.append(saved)

                result = session.predict_and_adapt(task, student_tensor, teacher_tensor, record_prediction)
                require(len(captured) == 1, "Prediction was not recorded before adaptation")
                row = captured[0]
                is_selected = row["confidence"] >= config["data"]["student_confidence_threshold"]
                require(row["selected"] is is_selected, "Invalid student confidence gate")
                total_updates += 3 * int(is_selected)
                require(result["updates_after"] == total_updates, "Wrong actual adaptation step count")
                selected[task] += int(is_selected)
                scored[task] += 1
                previous_hash = prompt_hash(session)
                require(is_selected or previous_hash == before, "A skipped image changed prompt parameters")
                updates.write(json.dumps({"index": global_index, "sample_index": sample_index, "task_id": task,
                    "selected": is_selected, "updates_before": row["updates_before"], "updates_after": total_updates,
                    "prompt_sha256_before": before, "prompt_sha256_after": previous_hash}, allow_nan=False) + "\n")
                predictions.append({"sample_index": sample_index, "task_id": task,
                    "predicted_class_index": row["predicted_class_index"]})
                if (global_index + 1) % 100 == 0 or global_index + 1 == 20000:
                    trace.flush()
                    updates.flush()
                    progress = {"status": "RUNNING", "scored": global_index + 1, "total": 20000,
                        "per_task_scored": dict(scored), "updates": total_updates,
                        "elapsed_seconds": time.time() - started}
                    save_json(output / "progress.json", progress)
                    print("PROGRESS " + json.dumps(progress), flush=True)
        require(dict(scored) == dict.fromkeys(TASKS, 5000), "Incomplete full-stream evaluation")
        score = prepared.benchmark.score(predictions)
        # Reopen the file, rather than trusting only in-memory counters.
        persisted = [json.loads(line) for line in (output / "predictions.jsonl").open()]
        require(prepared.benchmark.score(persisted) == score, "Persisted prediction/scorer mismatch")
        require(source_hashes() == frozen_sources, "Runtime source changed during evaluation")
        require(all(file_sha256(prepared.root / p) == h for p, h in FROZEN.items()), "Frozen dataset artifacts changed")
        require(all(file_sha256(prepared.root / r["image_path"]) == r["file_sha256"]
                    for r in prepared.records), "Input images changed during evaluation")
        require(all(file_sha256(p) == method[r + "_weights_sha256"] for r, p in paths.items()), "Model weights changed")
        metadata.update(status="COMPLETE", finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.time() - started, final_prompt_sha256=previous_hash,
            optimizer_final_state=session.optimizer_audit(), total_adaptation_steps=total_updates,
            source_hashes_verified_at_end=True, all_image_hashes_rechecked_after_run=True)
        summary = {"status": "PASS", "benchmark": config["benchmark"], "seed": 0,
            "method": "continual_multitask_primary_student", **score,
            "per_task_selected": dict(selected), "adaptation_updates": total_updates,
            "preview_images": 256, "preview_parameter_updates": 0, "teacher_output_fusion": False,
            "prediction_rule": "pre_update_student_only", "stream": "balanced_round_robin",
            "prompt_reset_on_task_switch": False, "optimizer_reset_on_task_switch": False,
            "elapsed_seconds": metadata["elapsed_seconds"]}
        save_json(output / "metadata.json", metadata)
        save_json(output / "summary.json", summary)
        save_json(output / "progress.json", {"status": "COMPLETE", "scored": 20000, "total": 20000,
            "per_task_scored": dict(scored), "updates": total_updates,
            "elapsed_seconds": metadata["elapsed_seconds"]})
        save_json(output / "completion.json", {"status": "PASS", **{p: file_sha256(output / p)
            for p in ["config.json", "input_manifest.json", "metadata.json", "summary.json", "progress.json", "predictions.jsonl", "updates.jsonl"]}})
        return summary
    except BaseException as error:
        metadata.update(status="FAILED", error=repr(error), elapsed_seconds=time.time() - started)
        save_json(output / "metadata.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=Path("models"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
