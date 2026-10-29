"""Audit both completed MT-Sketch4-20K runs before publishing any results.

This is a CPU-only trace audit, not a model inference entry point. It verifies
completion digests, frozen inputs/configurations, every prediction, prompt
continuity and the persistent AdamW state before creating either report file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
# Importing the frozen benchmark must not add files to its directory.
sys.dont_write_bytecode = True

from clip_officehome.multitask_sketch import FROZEN, TASKS, prepare_mtsketch, source_hashes
from clip_officehome.protocol import WEIGHTS, canonical_hash, file_sha256, read_json, require, save_json

PROBABILITY_ATOL = 0.001  # CUDA fp16 softmax versus CPU double recomputation.
PROBABILITY_SUM_ATOL = 0.005
BENCHMARK = "MT-Sketch4-20K"


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid SHA256 value")
    return value


def same_number(actual, expected, description):
    require(type(actual) in (int, float) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10), description)


def verify_completed(directory, role):
    directory = Path(directory).resolve()
    done = read_json(directory / "completion.json")
    require(done["status"] == "PASS", f"{role} completion is not PASS")
    if role == "tta":
        hashes = {key: value for key, value in done.items() if key != "status"}
        expected = {"config.json", "input_manifest.json", "metadata.json", "summary.json",
                    "predictions.jsonl", "updates.jsonl", "progress.json"}
    else:
        require(done["schema"] == "mt_sketch_zeroshot_completion_v1", "Unexpected control completion schema")
        hashes = done["files"]
        expected = {"config.json", "input_manifest.json", "metadata.json", "summary.json",
                    "predictions.jsonl", "progress.json"}
        require(done["num_scored_images"] == 20000 and done["independent_scorer_verified"] is True,
                "Control completion does not certify the full stream")
    require(set(hashes) == expected, f"{role} completion has missing or unexpected artifacts")
    for name, expected_hash in hashes.items():
        require(file_sha256(directory / name) == digest(expected_hash), f"Changed {role} artifact: {name}")
    metadata, summary = read_json(directory / "metadata.json"), read_json(directory / "summary.json")
    require(metadata["status"] == "COMPLETE" and summary["status"] == "PASS", f"Unfinished {role} run")
    require(metadata["benchmark"] == summary["benchmark"] == BENCHMARK, "Unexpected benchmark")
    return {"path": directory, "completion": done, "hashes": hashes, "metadata": metadata,
            "summary": summary, "config": read_json(directory / "config.json"),
            "inputs": read_json(directory / "input_manifest.json")}


def validate_configurations(prepared, tta, control):
    config, metadata = tta["config"], tta["metadata"]
    require(config == prepared.config, "TTA configuration differs from the complete fixed protocol")
    config_hash = canonical_hash(config)
    require(metadata["config_sha256"] == config_hash, "TTA metadata/config digest mismatch")
    evaluation, data, method = config["evaluation"], config["data"], config["baseline_method"]
    require(evaluation["seed"] == 0 and evaluation["batch_size"] == 1 and
            evaluation["full_data"] is True and evaluation["shuffle"] is False and
            evaluation["stream_order"] == "balanced_round_robin" and evaluation["task_known"] is True and
            evaluation["score_before_update"] is True and evaluation["prediction_source"] == "student_only" and
            evaluation["teacher_output_fusion"] is False and evaluation["teacher_output_calibration"] is False and
            evaluation["persistent_prompts"] is True and evaluation["reset_prompt_on_task_switch"] is False and
            evaluation["reset_optimizer_each_domain"] is False and
            evaluation["reset_optimizer_on_task_switch"] is False and evaluation["optimizer_state_persistent"] is True and
            evaluation["independent_task_initialization"] is False and evaluation["number_of_prompt_banks"] == 1,
            "Invalid full-data, state-persistence or output protocol")
    require(data["student_confidence_threshold"] == 0.815 and data["learning_rate"] == 7.5e-6 and
            data["shadow_text_domains"] == [] and data["domain_learning_rate_overrides"] == {} and
            method["adaptation"]["update_rounds_per_selected_image"] == 3 and
            method["adaptation"]["teacher_ce_weight"] == 0.08 and
            config["transfer_policy"]["new_test_hyperparameter_search"] is False,
            "Method differs from the untuned protocol")
    require(metadata["teacher_output_fusion"] is False and metadata["teacher_output_calibration"] is False and
            metadata["output_rule"] == "pre_update_student_probability_argmax_only" and
            metadata["prompt_reset_on_task_switch"] is False and metadata["optimizer_reset_on_task_switch"] is False and
            metadata["number_of_prompt_banks"] == 1 and metadata["all_previews_before_adaptation"] is True and
            metadata["source_hashes_verified_at_end"] is True and
            metadata["all_image_hashes_rechecked_after_run"] is True, "TTA metadata contradicts the protocol")
    for role, (filename, sha) in WEIGHTS.items():
        require(method[role + "_weights"] == filename and method[role + "_weights_sha256"] == sha,
                "Unexpected configured model weights")
    require(metadata["model_hashes"] == {name: sha for name, sha in WEIGHTS.values()}, "TTA model hash mismatch")
    baseline = control["config"]
    model_path = Path(baseline["model_path"]).expanduser().resolve()
    require(model_path.name == WEIGHTS["student"][0], "Unexpected control model filename")
    mapping = {task: [{**entry, "prompt_name": name} for entry, name in
                     zip(prepared.classes[task], prepared.prompt_names[task], strict=True)] for task in TASKS}
    expected = {
        "schema": "mt_sketch_zeroshot_config_v1", "benchmark": BENCHMARK, "dataset": BENCHMARK,
        "method": "frozen_CLIP_B16_single_template", "model": "ViT-B/16", "seed": 0,
        "full_data": True, "num_scored_images": 20000, "images_per_task": 5000,
        "stream": "balanced_round_robin", "task_known": True, "independent_run_initialization": True,
        "inference_batch_size": 1, "template": "a photo of a {class}", "logit_scale": 100.0,
        "prompt_names": prepared.prompt_names, "class_mapping": mapping,
        "prompt_name_normalization": "normalize_class_name: replace underscores with spaces; strip; retain full native synonyms",
        "prediction_source": "frozen_student_only", "scoring": "pre_update", "adaptation": False,
        "adaptation_updates": 0, "teacher_used": False, "teacher_output_fusion": False,
        "prompt_ensemble": False, "previews": 0, "extra_random_augmentation": False,
        "numeric_policy": "unchanged_local_CLIP_CUDA_dtypes; no_added_AMP_or_precision_override",
        "model_path": str(model_path), "model_sha256": WEIGHTS["student"][1],
        "data_definition_sha256": FROZEN, "shared_method_config_canonical_sha256": config_hash,
        "source_hashes": metadata["source_hashes"],
    }
    require(baseline == expected, "Control configuration/weights/class space differs from the fixed protocol")
    base_meta = control["metadata"]
    require(canonical_hash(baseline) == base_meta["config_canonical_sha256"] ==
            control["completion"]["config_canonical_sha256"], "Control configuration digest mismatch")
    require(base_meta["seed"] == 0 and base_meta["teacher_used"] is False and
            base_meta["teacher_output_fusion"] is False and base_meta["adaptation_updates"] == 0 and
            base_meta["prediction_rule"] == "frozen_student_only_pre_update" and
            base_meta["inference_batch_size"] == 1 and base_meta["scored"] == 20000 and
            base_meta["independent_scorer_verified"] is True, "Invalid control metadata")
    require(base_meta["data_definition_hashes_start"] == base_meta["data_definition_hashes_end"] == FROZEN,
            "Control dataset definition changed")
    require(base_meta["model_path"] == str(model_path) and base_meta["model_weights_sha256_start"] ==
            base_meta["model_weights_sha256_end"] == WEIGHTS["student"][1], "Control weights changed")
    for filename, sha in WEIGHTS.values():
        require(file_sha256(model_path.parent / filename) == sha, "Current cached model bytes differ from the run")
    require(metadata["preprocess"]["student"] == base_meta["preprocess"], "Student/control preprocessing differs")
    sources = metadata["source_hashes"]
    require(sources == base_meta["source_hashes_start"] == base_meta["source_hashes_end"] == source_hashes(),
            "Source hashes differ between runs or from the current checked code")
    for relative, sha in sources.items():
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe source path")
        require(file_sha256(WORKSPACE / path) == digest(sha), "Source changed: " + relative)
    return config_hash


def validate_inputs(prepared, tta, control):
    expected = {"inventory": prepared.inventory, "records": prepared.records,
                "order": prepared.order, "classes": prepared.classes}
    for run in (tta, control):
        require(all(run["inputs"][key] == value for key, value in expected.items()),
                "Run input manifest differs from frozen members, labels, classes, hashes or stream")
    require(control["inputs"]["schema"] == "mt_sketch_input_manifest_v1" and
            control["inputs"]["prompt_names"] == prepared.prompt_names and
            control["inputs"]["stream"] == "balanced_round_robin" and
            control["inputs"]["data_definition_sha256"] == FROZEN,
            "Control input manifest prompt/definition mismatch")
    require(control["metadata"]["data"] == prepared.inventory, "Control inventory metadata mismatch")


def probability_prediction(row, num_classes):
    logits, probabilities = row["student_logits"], row["student_probabilities"]
    require(isinstance(logits, list) and isinstance(probabilities, list) and
            len(logits) == len(probabilities) == num_classes and
            all(type(value) in (int, float) and math.isfinite(value) for value in logits + probabilities) and
            all(0 <= value <= 1 for value in probabilities), "Invalid task-local student logits/probabilities")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    error = max(abs(p - value / denominator) for p, value in zip(probabilities, exponentials, strict=True))
    require(error <= PROBABILITY_ATOL and abs(sum(probabilities) - 1) <= PROBABILITY_SUM_ATOL,
            "Stored probabilities do not match softmax(student logits)")
    predicted = max(range(num_classes), key=probabilities.__getitem__)
    require(type(row["predicted_class_index"]) is int and row["predicted_class_index"] == predicted and
            type(row["confidence"]) in (int, float) and row["confidence"] == max(probabilities),
            "Student argmax/confidence mismatch")
    return predicted, error


def validate_route(route, experts, topk):
    ids, weights, scores = route["selected_ids"], route["selected_weights"], route["routing_scores"]
    require(len(ids) == len(weights) == topk and len(set(ids)) == topk and
            all(type(index) is int and 0 <= index < experts for index in ids) and len(scores) == experts and
            all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                for value in scores + weights), "Invalid expert route")
    # torch.topk may choose either index at a tie; scores, not index order, bind it.
    require([scores[index] for index in ids] == sorted(scores, reverse=True)[:topk], "Route is not score Top-k")
    exponentials = [math.exp(scores[index] - max(scores[index] for index in ids)) for index in ids]
    require(max(abs(weight - value / sum(exponentials)) for weight, value in
                zip(weights, exponentials, strict=True)) <= PROBABILITY_ATOL and
            abs(sum(weights) - 1) <= PROBABILITY_SUM_ATOL, "Route weights differ from temperature-one softmax")
    return {key: route[key] for key in ("selected_ids", "selected_weights", "routing_scores")}


def validate_previews(prepared, metadata):
    initial = digest(metadata["initial_prompt_sha256"])
    require(set(metadata["routes"]) == set(TASKS), "Missing or additional task preview")
    fixed_routes, preview_indices = {}, []
    for task, count in TASKS.items():
        route = metadata["routes"][task]
        indices = prepared.task_indices(task)[:64]
        require(route["task_id"] == task and route["num_classes"] == count and
                route["preview_count"] == route["preview_images"] == 64 and
                route["sample_indices"] == indices and route["parameter_updates"] == 0 and
                route["score_mode"] == "confidence" and
                route["before_prompt_hash"] == route["after_prompt_hash"] ==
                route["before_prompt_sha256"] == route["after_prompt_sha256"] == initial,
                "Preview changed state or used unexpected members/classes")
        fixed_routes[task] = validate_route(route, 8, 2)
        preview_indices.extend(indices)
    require(len(preview_indices) == len(set(preview_indices)) == 256 and
            set(preview_indices) == set(prepared.order[:256]), "Unexpected total task-known lookahead budget")
    return fixed_routes


def recount_control(prepared, control):
    predictions, max_error = [], 0.0
    with (control["path"] / "predictions.jsonl").open(encoding="utf-8") as handle:
        for index, (sample, line) in enumerate(zip(prepared.order, handle, strict=True)):
            row, record = json.loads(line), prepared.records[sample]
            task = record["task_id"]
            require(row["global_index"] == index and type(row["sample_index"]) is int and
                    row["sample_index"] == sample and row["task_id"] == task and
                    row["sample_id"] == record["sample_id"] and row["image_path"] == record["image_path"] and
                    row["image_sha256"] == record["file_sha256"] and
                    type(row["label"]) is int and row["label"] == record["class_index"],
                    "Control trace differs from the frozen per-image identity/label/order")
            require(row["schema"] == "mt_sketch_zeroshot_prediction_v1" and
                    row["prediction_source"] == "frozen_student_only" and row["scoring"] == "pre_update" and
                    row["adaptation_updates_before_prediction"] == 0, "Control source/timing/update mismatch")
            predicted, error = probability_prediction(row, TASKS[task])
            max_error = max(max_error, error)
            predictions.append({"sample_index": sample, "task_id": task, "predicted_class_index": predicted})
    return prepared.benchmark.score(predictions), max_error


def recount_tta(prepared, tta, fixed_routes):
    predictions, visits, selected = [], Counter(), Counter(dict.fromkeys(TASKS, 0))
    text_steps, visual_steps = Counter(), Counter()
    updates, max_error = 0, 0.0
    previous_hash = tta["metadata"]["initial_prompt_sha256"]
    with (tta["path"] / "predictions.jsonl").open(encoding="utf-8") as trace, \
            (tta["path"] / "updates.jsonl").open(encoding="utf-8") as update_trace:
        for index, (sample, line, update_line) in enumerate(zip(prepared.order, trace, update_trace, strict=True)):
            row, update, record = json.loads(line), json.loads(update_line), prepared.records[sample]
            task = record["task_id"]
            require(row["index"] == row["stream_index"] == index and type(row["sample_index"]) is int and
                    row["sample_index"] == sample and row["task_id"] == task and
                    row["task_local_index"] == visits[task] and row["num_classes"] == TASKS[task] and
                    row["image_sha256"] == record["file_sha256"] and
                    type(row["label"]) is int and row["label"] == record["class_index"],
                    "TTA trace differs from the frozen task-local identity/label/order")
            require(row["prediction_source"] == "student_only" and row["scoring"] == "pre_update" and
                    row["updates_before"] == row["optimizer_rounds_before_prediction"] == updates and
                    row["prompt_sha256_before"] == previous_hash and "updates_after" not in row,
                    "TTA prediction is not bound to the pre-update persistent state")
            predicted, error = probability_prediction(row, TASKS[task])
            max_error = max(max_error, error)
            require(row["prediction"] == predicted, "TTA prediction aliases disagree")
            is_selected = row["confidence"] >= 0.815
            require(row["selected"] is is_selected, "TTA gate is not pre-update student confidence >= 0.815")
            require(row["text_route"] == fixed_routes[task], "Task text route changed after preview")
            visual = validate_route(row["visual_route"], 4, 1)
            require(type(row["teacher_pseudolabel"]) is int and
                    0 <= row["teacher_pseudolabel"] < TASKS[task] and
                    type(row["teacher_confidence"]) in (int, float) and
                    math.isfinite(row["teacher_confidence"]) and 0 <= row["teacher_confidence"] <= 1,
                    "Invalid auxiliary teacher metadata")
            after_hash = digest(update["prompt_sha256_after"])
            expected_update = {"index": index, "sample_index": sample, "task_id": task,
                "selected": is_selected, "updates_before": updates, "updates_after": updates + 3 * int(is_selected),
                "prompt_sha256_before": previous_hash, "prompt_sha256_after": after_hash}
            require(update == expected_update and update["selected"] is is_selected,
                    "Update trace identity, step count or prompt continuity mismatch")
            require(is_selected or after_hash == previous_hash, "Skipped image changed prompt parameters")
            if is_selected:
                for expert in fixed_routes[task]["selected_ids"]:
                    text_steps[expert] += 3
                for expert in visual["selected_ids"]:
                    visual_steps[expert] += 3
                selected[task] += 1
                updates += 3
            previous_hash = after_hash
            visits[task] += 1
            predictions.append({"sample_index": sample, "task_id": task, "predicted_class_index": predicted})
    require(visits == dict.fromkeys(TASKS, 5000), "TTA did not score every task member once")
    require(previous_hash == tta["metadata"]["final_prompt_sha256"] and
            updates == tta["metadata"]["total_adaptation_steps"], "TTA final state/step mismatch")
    validate_optimizers(tta["metadata"], updates, selected, text_steps, visual_steps)
    return prepared.benchmark.score(predictions), dict(selected), updates, max_error


def validate_optimizers(metadata, updates, selected, text_steps, visual_steps):
    initial, final = metadata["optimizer_initial_state"], metadata["optimizer_final_state"]
    expected_names = {
        "main": [*[f"shared_bank.contexts.{i}" for i in range(3)],
                 *[f"private_bank.contexts.{i}" for i in range(8)],
                 "visual_scaffold.prompt_positional_embeddings", "visual_scaffold.prompt_scale",
                 *[f"visual_shared_bank.prompts.{i}" for i in range(2)]],
        **{f"visual_private_{i}": [f"visual_private_bank.prompts.{i}"] for i in range(4)},
    }
    for audit, rounds, scored, count in ((initial, 0, 0, 0), (final, updates, 20000, sum(selected.values()))):
        require(audit["optimizer_count"] == audit["optimizer_creations"] == len(audit["optimizers"]) == 5 and
                audit["optimizer_state_persistent"] is True and audit["global_update_rounds"] == rounds and
                audit["scored"] == scored and audit["selected"] == count,
                "Optimizer creation count or cumulative state mismatch")
        require([row["kind"] for row in audit["optimizers"]] == list(expected_names), "Unexpected optimizer kinds")
    for creation, (before, after) in enumerate(zip(initial["optimizers"], final["optimizers"], strict=True)):
        kind = before["kind"]
        require(before["instance_id"] == after["instance_id"] and
                isinstance(before["instance_id"], str) and before["instance_id"].isdigit() and
                before["creation_index"] == after["creation_index"] == creation,
                "Optimizer was recreated/reset between tasks")
        for optimizer in (before, after):
            require(optimizer["parameter_count"] == len(optimizer["parameters"]) == len(expected_names[kind]) and
                    [p["name"] for p in optimizer["parameters"]] == expected_names[kind],
                    "Optimizer parameter membership changed")
            states = sum(p["has_state"] is True for p in optimizer["parameters"])
            require(optimizer["state_parameter_count"] == states and optimizer["empty_state"] is (states == 0),
                    "Optimizer state entry count mismatch")
        require(before["empty_state"] is True and all(p["has_state"] is False and p["step"] == 0
                                                     for p in before["parameters"]), "Preview updated optimizer state")
        for parameter in after["parameters"]:
            name = parameter["name"]
            expected = updates
            if name.startswith("private_bank.contexts."):
                expected = text_steps[int(name.rsplit(".", 1)[1])]
            elif name.startswith("visual_private_bank.prompts."):
                expected = visual_steps[int(name.rsplit(".", 1)[1])]
            require(type(parameter["step"]) is int and parameter["step"] == expected and
                    parameter["has_state"] is (expected > 0), "Per-parameter AdamW steps differ from routed updates")


def validate_scores(tta, control, method_score, baseline_score, selected, updates):
    result, baseline = tta["summary"], control["summary"]
    for key, value in method_score.items():
        require(result[key] == value, "TTA summary differs from independent trace recount: " + key)
    require(result["seed"] == 0 and result["method"] == "continual_multitask_primary_student" and
            result["per_task_selected"] == selected and result["adaptation_updates"] == updates and
            result["preview_images"] == 256 and result["preview_parameter_updates"] == 0 and
            result["teacher_output_fusion"] is False and result["prediction_rule"] == "pre_update_student_only" and
            result["stream"] == "balanced_round_robin" and result["prompt_reset_on_task_switch"] is False and
            result["optimizer_reset_on_task_switch"] is False, "TTA summary protocol mismatch")
    require(baseline["schema"] == "mt_sketch_zeroshot_summary_v1" and baseline["seed"] == 0 and
            baseline["method"] == "frozen_CLIP_B16_single_template" and
            baseline["stream"] == "balanced_round_robin" and baseline["num_scored_images"] == 20000 and
            baseline["teacher_used"] is False and baseline["teacher_output_fusion"] is False and
            baseline["adaptation_updates"] == 0 and baseline["inference_batch_size"] == 1 and
            baseline["prompt_ensemble"] is False and baseline["preview_image_visits"] == 0 and
            baseline["prediction_rule"] == "frozen_student_only_pre_update" and
            baseline["all_predictions_recounted"] is True and baseline["source_hashes_unchanged"] is True and
            baseline["dataset_definitions_unchanged"] is True and baseline["model_weights_unchanged"] is True and
            baseline["independent_scorer_result"] == baseline_score, "Control summary/recount mismatch")
    require(set(baseline["per_task"]) == set(TASKS), "Control summary task mismatch")
    for task, score in baseline_score["per_task"].items():
        require(baseline["per_task"][task] == {"num_images": 5000, "correct": score["correct"],
                                             "top1_percent": score["accuracy_percent"]},
                "Control task score differs from independent recount")
    macro = baseline_score["macro_accuracy_percent"]
    same_number(baseline["macro_top1_percent"], macro, "Wrong control macro accuracy")
    same_number(baseline["micro_top1_percent"], macro, "Wrong control micro accuracy")
    progress = read_json(control["path"] / "progress.json")
    require(progress["status"] == "COMPLETE" and progress["scored"] == progress["total"] == 20000 and
            progress["per_task_scored"] == dict.fromkeys(TASKS, 5000), "Control progress does not certify completion")
    progress = read_json(tta["path"] / "progress.json")
    require(progress["status"] == "COMPLETE" and progress["scored"] == progress["total"] == 20000 and
            progress["per_task_scored"] == dict.fromkeys(TASKS, 5000) and progress["updates"] == updates,
            "TTA progress does not certify completion and cumulative updates")
    for run in (tta, control):
        seconds = run["summary"]["elapsed_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                "Invalid runtime in summary")
        same_number(run["metadata"]["elapsed_seconds"], seconds, "Metadata/summary runtime mismatch")


def render_report(result):
    rows = "\n".join(
        f"| {r['task_id']} | {r['num_classes']:,} | {r['num_images']:,} | "
        f"{r['control_top1_percent']:.3f}% | {r['tta_top1_percent']:.3f}% | "
        f"{r['delta_percentage_points']:+.3f} | {r['selected_images']:,} | {r['adaptation_steps']:,} |"
        for r in result["tasks"])
    return f"""# MT-Sketch4-20K 首轮统一评测结果

两侧全量运行均已完成，completion 摘要、冻结输入及逐图预测审计全部通过。
seed0、每侧20,000张、batch1；本轮没有按该测试结果搜索超参数。

| task | 类别数 | 图像数 | 冻结B/16对照 | 本方法Top-1 | 变化（百分点） | 入选更新图像 | 更新步数 |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

四任务等权平均：本方法 **{result['macro_tta_top1_percent']:.3f}%**，
冻结对照 **{result['macro_control_top1_percent']:.3f}%**，
变化 **{result['macro_delta_percentage_points']:+.3f} 个百分点**。
各任务均5000张，因此本次微平均与宏平均数值相同。

输入按冻结 balanced_round_robin 流，每个任务依次取一张；任务循环为
DomainNet-Sketch → ImageNet-Sketch → TU-Berlin → Sketchy。
任务ID已知，只选择各自345/1000/250/125类的原生类别空间，不做1720类联合分类。
提示类别名仅将下划线转为空格并去除首尾空白，保留原生同义词串。

方法使用单套共享文本/视觉提示参数和持续保留的 AdamW 状态；跨任务不重新初始化、
不保存四套独立提示模型。实现使用一个主优化器及四个私有视觉专家优化器，均只创建一次。
每个任务最初64张在所有更新开始前预览，共256张已知任务的无标签 lookahead；
预览不更新参数，不使用教师或标签，固定各任务文本Top-2路由。
这些图像随后与全部正式输入一样各计分一次，视觉路由逐图Top-1。

学生为冻结OpenAI CLIP ViT-B/16，教师ViT-L/14仅提供权重0.08的辅助硬伪标签CE。
最终输出始终为更新前学生概率argmax，无教师输出融合或校准。
学生置信度≥0.815的图像执行3步AdamW，学习率7.5e-6，无shadow和任务特殊学习率。
对照使用同一B/16权重、同一输入顺序/解码/类别词表，单模板`a photo of a {{class}}`，
batch1，无教师、预览、提示集成或更新；两侧独立启动。

数据仍为冻结 **1.0-review** 版本：保留106张原生标签歧义标记和另外8张待人工审阅图像。
这些标记不等同于已确认错误标签；本轮未依据预测清洗、删图或改标签。
这不是已经专家全面清洗的正式榜单，不能据此声称同协议SOTA或全新未见测试集。
此处ImageNet-Sketch仅为自建benchmark中的5000张子集，且跨任务持续适应；
不能用上轮完整50,889张ImageNet-Sketch的独立测试成绩替代。

审计逐图复算 logits→softmax→预测/置信度，核对门控、更新前全局步数、提示哈希连续性、
256张预览回到正式流、优化器实例ID以及每个参数累计AdamW步数。
CPU双精度softmax与原CUDA低精度记录允许最大逐元素绝对误差{PROBABILITY_ATOL}，
概率和容差{PROBABILITY_SUM_ATOL}；argmax与门控严格使用已保存的学生概率。
日志审计证明其与冻结代码和协议一致，不代表脱离这些代码的密码学执行证明。

正式输出：`{result['tta_output']}`；对照：`{result['control_output']}`。
完整计数、SHA256及审计边界见同目录`mtsketch_seed0_results.json`；
协议说明见`docs/MT_SKETCH4_EVALUATION.md`。
"""


def summarize(tta_output, control_output, data_root, output_dir):
    tta, control = verify_completed(tta_output, "tta"), verify_completed(control_output, "control")
    require(tta["path"] != control["path"], "TTA and control must be separate runs")
    # Rehash every current image; do not trust only the old pre-run inventory.
    prepared = prepare_mtsketch(data_root, hash_images=True)
    config_hash = validate_configurations(prepared, tta, control)
    validate_inputs(prepared, tta, control)
    routes = validate_previews(prepared, tta["metadata"])
    baseline_score, baseline_error = recount_control(prepared, control)
    method_score, selected, updates, method_error = recount_tta(prepared, tta, routes)
    validate_scores(tta, control, method_score, baseline_score, selected, updates)
    rows = []
    for task, count in TASKS.items():
        method, baseline = method_score["per_task"][task], baseline_score["per_task"][task]
        rows.append({"task_id": task, "num_images": 5000, "num_classes": count,
            "tta_correct": method["correct"], "tta_top1_percent": method["accuracy_percent"],
            "control_correct": baseline["correct"], "control_top1_percent": baseline["accuracy_percent"],
            "delta_percentage_points": method["accuracy_percent"] - baseline["accuracy_percent"],
            "selected_images": selected[task], "adaptation_steps": 3 * selected[task], "text_route": routes[task]})
    result = {
        "schema": "mt_sketch_completed_pair_audit_v1", "status": "PASS", "benchmark": BENCHMARK,
        "audited_utc": datetime.now(timezone.utc).isoformat(), "dataset_version": "1.0-review",
        "seed": 0, "num_images_per_run": 20000, "inference_batch_size_both_runs": 1,
        "stream": "balanced_round_robin", "task_known": True, "tasks": rows,
        "macro_tta_top1_percent": method_score["macro_accuracy_percent"],
        "macro_control_top1_percent": baseline_score["macro_accuracy_percent"],
        "macro_delta_percentage_points": method_score["macro_accuracy_percent"] - baseline_score["macro_accuracy_percent"],
        "micro_tta_top1_percent": sum(r["tta_correct"] for r in rows) / 20000 * 100,
        "micro_control_top1_percent": sum(r["control_correct"] for r in rows) / 20000 * 100,
        "adaptation_steps": updates, "selected_images": sum(selected.values()),
        "student_only_pre_update": True, "teacher_output_fusion": False,
        "teacher_role": "frozen_auxiliary_hard_pseudolabel_CE_only", "teacher_ce_weight": 0.08,
        "prompt_banks": 1, "prompt_reset_on_task_switch": False, "optimizer_reset_on_task_switch": False,
        "persistent_optimizer_instances": 5, "preview_images": 256, "preview_parameter_updates": 0,
        "preview_images_each_scored_once_later": True, "new_test_hyperparameter_search": False,
        "native_label_conflict_images_retained": 106, "additional_visual_review_images_retained": 8,
        "fully_expert_semantically_cleaned": False, "official_leaderboard_or_sota_claim": False,
        "fresh_unseen_test_claim": False,
        "tta_output": str(tta["path"]), "control_output": str(control["path"]),
        "data_root": str(prepared.root), "tta_config_canonical_sha256": config_hash,
        "control_config_canonical_sha256": canonical_hash(control["config"]),
        "frozen_artifacts": FROZEN, "ordered_index_sha256": prepared.inventory["ordered_index_sha256"],
        "ordered_content_sha256": prepared.inventory["ordered_content_sha256"],
        "source_hashes": tta["metadata"]["source_hashes"], "model_hashes": tta["metadata"]["model_hashes"],
        "initial_prompt_sha256": tta["metadata"]["initial_prompt_sha256"],
        "final_prompt_sha256": tta["metadata"]["final_prompt_sha256"],
        "run_artifact_sha256": {"tta": tta["hashes"], "control": control["hashes"]},
        "completion_sha256": {role: file_sha256(run["path"] / "completion.json")
                              for role, run in (("tta", tta), ("control", control))},
        "summarizer_sha256": file_sha256(Path(__file__)),
        "audit": {"all_completion_digests_checked": True, "both_runs_complete_pass": True,
            "current_image_bytes_rehashed": True, "same_frozen_inputs_classes_order": True,
            "all_40000_predictions_recounted": True, "student_logit_softmax_argmax_verified": True,
            "probability_absolute_tolerance": PROBABILITY_ATOL, "probability_sum_tolerance": PROBABILITY_SUM_ATOL,
            "maximum_softmax_absolute_error": {"tta": method_error, "control": baseline_error},
            "gate_and_global_step_counts_verified": True, "prompt_hash_chain_verified": True,
            "preview_members_and_no_updates_verified": True, "optimizer_instance_ids_unchanged": True,
            "per_parameter_optimizer_steps_match_routed_updates": True,
            "source_weights_and_exact_configs_bound": True,
            "limitation": "Trace consistency plus frozen implementation; not an independent cryptographic proof of execution."},
    }
    # Recheck the completion-bound files after the potentially long trace scan.
    for role, run in (("tta", tta), ("control", control)):
        observed = verify_completed(run["path"], role)
        require(observed["completion"] == run["completion"], "Run artifacts changed during summary audit")
    require(source_hashes() == tta["metadata"]["source_hashes"], "Source changed during summary audit")
    output = Path(output_dir).expanduser().resolve()
    require(all(run["path"] not in (output, *output.parents) for run in (tta, control)) and
            prepared.root not in (output, *output.parents),
            "Reports must not be written into frozen data or run outputs")
    output.mkdir(parents=True, exist_ok=True)
    json_path, text_path = output / "mtsketch_seed0_results.json", output / "mtsketch_seed0_results_zh.md"
    require(not json_path.exists() and not text_path.exists(), "Refusing to overwrite an existing result report")
    save_json(json_path, result)
    text_path.write_text(render_report(result), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tta-output", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.tta_output, args.control_output, args.data_root, args.output_dir),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
