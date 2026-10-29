"""Audit and summarize complete ImageNet-A/V2/R/Sketch runs without loading models.

The manifest selects completed TTA/control directories, never partial traces.
Both prediction streams are recounted before any final report is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_officehome.evaluation import recount_trace
from clip_officehome.imagenet_ood import benchmark_config
from clip_officehome.protocol import canonical_hash, file_sha256, read_json, require, save_json
from summarize_imagenet_pair import validate_control_mapping


EXPECTED = {"imagenet_a": (7500, 200), "imagenet_v2": (10000, 1000),
            "imagenet_r": (30000, 200), "imagenet_sketch": (50889, 1000)}
TOTAL_IMAGES = 98389
JSON_NAME = "imagenet_ood_seed0_results.json"
MARKDOWN_NAME = "imagenet_ood_seed0_results_zh.md"


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def close(actual, expected, message, tolerance=1e-10):
    require(isinstance(actual, (float, int)) and math.isfinite(actual) and
            abs(actual - expected) <= tolerance, message)


def load_manifest(path):
    manifest = read_json(path)
    require(manifest["schema"] == "imagenet_ood_four_benchmarks_v1" and manifest["seed"] == 0 and
            manifest["expected_total_images"] == TOTAL_IMAGES, "Wrong four-benchmark manifest protocol")
    entries = manifest["benchmarks"]
    require(len(entries) == 4 and {entry["benchmark"] for entry in entries} == set(EXPECTED),
            "Manifest must contain exactly one A, V2, R and Sketch run")
    by_name = {entry["benchmark"]: entry for entry in entries}
    entries = [by_name[name] for name in EXPECTED]
    paths = []
    for entry in entries:
        for key in ("tta_output", "control_output"):
            path = Path(entry[key])
            require(path.is_absolute(), f"Manifest {key} must be an absolute path")
            paths.append(path.resolve())
    require(len(set(paths)) == 8, "Run directories must be distinct")
    return manifest, entries


def require_all_complete(entries):
    """Fail quickly on a running/failed suite before reading large traces."""
    for entry in entries:
        name = entry["benchmark"]
        for kind in ("tta_output", "control_output"):
            directory = Path(entry[kind])
            require((directory / "completion.json").is_file(), f"{name} {kind} is not complete: {directory}")
            require(read_json(directory / "completion.json")["status"] == "PASS" and
                    read_json(directory / "summary.json")["status"] == "PASS", f"{name} {kind} did not pass")
        require(read_json(Path(entry["tta_output"]) / "metadata.json")["status"] == "COMPLETE",
                f"{name} TTA metadata is not complete")


def current_runtime_sources():
    sources = {"clip_officehome/" + p.name: p for p in (ROOT / "src/clip_officehome").glob("*.py")}
    sources.update({"clip/" + p.name: p for p in (ROOT / "CLIP/clip").iterdir()
                    if p.suffix in {".py", ".gz"}})
    return {name: file_sha256(path) for name, path in sources.items()}


def validate_protocol(name, config, metadata, completion):
    # This binds every prompt/optimizer/preprocessing option as well as the
    # official 200/1000-class masks and the non-deduplicated TPT class names.
    require(config == benchmark_config(name), f"{name}: config differs from the fixed ImageNet protocol")
    digest = canonical_hash(config)
    require(config == metadata["config"] and digest == metadata["config_sha256"] == completion["config_sha256"],
            f"{name}: configuration is not bound to the completed metadata")
    evaluation, data, method = config["evaluation"], config["data"], config["baseline_method"]
    text, adaptation = method["text_prompt"], method["adaptation"]
    # Keep fixed statements in the human report explicit even if the config
    # factory is changed in a later development branch.
    require(evaluation["seed"] == 0 and evaluation["batch_size"] == 1 and
            evaluation["full_data"] is True and evaluation["shuffle"] is False and
            evaluation["score_before_update"] is True and evaluation["prediction_source"] == "student_only" and
            evaluation["teacher_output_fusion"] is False and evaluation["teacher_output_calibration"] is False and
            evaluation["independent_benchmark_initialization"] is True and
            evaluation["benchmark_state_transfer"] is False and evaluation["persistent_prompts"] is True,
            f"{name}: invalid seed/state/prediction protocol")
    require(metadata["benchmark"] == name and metadata["teacher_output_fusion"] is False and
            metadata["teacher_output_calibration"] is False and
            metadata["output_rule"] == "student_probability_argmax_only", f"{name}: invalid recorded output rule")
    require(method["student"] == "ViT-B/16" and method["teacher"] == "ViT-L/14" and
            adaptation["teacher_supervision"] == "aux_pseudolabel" and adaptation["teacher_ce_weight"] == 0.08 and
            adaptation["student_pseudo_topk"] == 1 and adaptation["teacher_confidence_gate"] is False and
            adaptation["update_rounds_per_selected_image"] == 3 and adaptation["optimizer"] == "AdamW" and
            data["student_confidence_threshold"] == 0.815 and data["learning_rate"] == 7.5e-6 and
            data["domain_learning_rate_overrides"] == {} and data["shadow_text_domains"] == [],
            f"{name}: invalid model/teacher/update protocol")
    require(text["preview_size"] == 64 and text["private_topk"] == 2 and text["preview_parameter_updates"] == 0 and
            text["preview_uses_ground_truth_or_teacher"] is False and text["preview_images_scored_later"] is True and
            text["post_preview_scoring_start_index"] == 0 and text["route_ids_and_weights_fixed_within_domain"] is True and
            config["transfer_policy"]["new_test_hyperparameter_search"] is False,
            f"{name}: invalid preview/no-new-search protocol")
    for role in ("student", "teacher"):
        require(metadata["model_hashes"][method[role + "_weights"]] == method[role + "_weights_sha256"],
                f"{name}: recorded {role} weights differ from the configuration")
    return digest


def validate_inputs(name, config, metadata, baseline, tta_input, control_input):
    expected, classes = EXPECTED[name]
    require(tta_input == control_input, f"{name}: TTA and control input manifests differ")
    require(set(tta_input["records"]) == {name}, f"{name}: unexpected input domain")
    records, inventory = tta_input["records"][name], tta_input["inventory"]
    data = config["data"]
    require(len(records) == expected and inventory == metadata["data"] and
            inventory["benchmark"] == name and inventory["num_scored_images"] == expected and
            inventory["num_classes"] == classes and inventory["image_bytes_hashed"] is True and
            inventory["order"] == config["evaluation"]["stream_order"] and
            inventory["class_metadata_sha256"] == data["class_metadata_sha256"] and
            inventory["domains"] == [{"domain": name, "num_images": expected, "num_classes": classes}],
            f"{name}: wrong full input inventory")
    seen, previous = set(), None
    for record in records:
        label = record["label_index"]
        require(type(label) is int and 0 <= label < classes, f"{name}: invalid label")
        require(record["wnid"] == data["class_names"][label] and
                record["class_name"] == data["prompt_class_names"][label] and
                record["imagenet_index"] == data["imagenet_indices"][label],
                f"{name}: record does not match the official class mapping")
        relative = PurePosixPath(record["relative_path"])
        folder = str(record["imagenet_index"]) if name == "imagenet_v2" else record["wnid"]
        require(not relative.is_absolute() and ".." not in relative.parts and len(relative.parts) >= 2 and
                relative.parts[0] == folder and relative.as_posix() == record["relative_path"] and
                record["relative_path"] not in seen and valid_digest(record["sha256"]),
                f"{name}: repeated/invalid image identity")
        seen.add(record["relative_path"])
        key = (record["class_name"], record["wnid"], record["relative_path"])
        require(previous is None or previous < key, f"{name}: stream is not class/WNID/path sorted")
        previous = key
    counts = dict(sorted(Counter(r["wnid"] for r in records).items()))
    require(counts == inventory["class_counts"] and set(counts) == set(data["class_names"]),
            f"{name}: incomplete class coverage")
    if name == "imagenet_v2":
        require(set(counts.values()) == {10}, "V2 must contain all ten matched-frequency images per class")
    index_hash = canonical_hash([[r["relative_path"], r["label_index"], r["imagenet_index"]] for r in records])
    content_hash = canonical_hash([[r["relative_path"], r["sha256"]] for r in records])
    require(index_hash == inventory["ordered_index_sha256"] and
            content_hash == inventory["ordered_content_sha256"] == baseline["ordered_content_sha256"],
            f"{name}: input identity/label digests disagree")
    return records


def validate_state(name, config, metadata, result, domain_results, records):
    expected, classes = EXPECTED[name]
    require(len(metadata["domain_states"]) == len(result["domains"]) == len(domain_results) == 1 and
            result["branch_results"] == result["domains"], f"{name}: unexpected extra/reported branch")
    state, domain, core_row = metadata["domain_states"][0], result["domains"][0], domain_results[0]
    for row in (state, domain, core_row):
        require(row["branch"] == "primary" and row["domain"] == name and row["is_reported"] is True,
                f"{name}: wrong reported primary branch")
    require(result["benchmark"] == name and result["seed"] == 0 and result["num_scored_images"] == expected and
            result["prediction_rule"] == "pre_update_student_only" and result["teacher_output_fusion"] is False and
            result["all_student_predictions_recounted"] is True and result["preview_parameter_updates"] == 0 and
            result["independent_benchmark_initialization"] is True and result["benchmark_state_transfer"] is False and
            result["fresh_unseen_test_claim"] is False and result["prompt_state_persistent"] is True and
            result["fresh_optimizers_each_domain"] is True and
            result["formal_branch_image_visits_excluding_prerouting"] == expected and result["preview_image_visits"] == 64,
            f"{name}: summary protocol differs")
    require(state["num_scored"] == domain["num_images"] == expected and
            all(state[k] == domain[k] for k in ("correct", "selected", "update_rounds")) and
            type(domain["correct"]) is int and 0 <= domain["correct"] <= expected and
            type(domain["selected"]) is int and 0 <= domain["selected"] <= expected and
            domain["update_rounds"] == 3 * domain["selected"] == result["branch_update_rounds"] and
            domain["selected"] == result["branch_selected_samples"], f"{name}: inconsistent TTA counts")
    accuracy = 100 * domain["correct"] / expected
    for value in (domain["top1_percent"], result["macro_top1_percent"], result["micro_top1_percent"]):
        close(value, accuracy, f"{name}: wrong accuracy denominator")
    initial = metadata["initial_prompt_hashes"]
    require(set(initial) == {"primary"} and valid_digest(initial["primary"]) and
            state["initial_prompt_hash"] == initial["primary"] and valid_digest(state["final_prompt_hash"]),
            f"{name}: initial/final prompt hashes are not bound")
    route = state["route_audit"]
    require(route["preview_count"] == 64 and route["preview_relative_paths"] ==
            [r["relative_path"] for r in records[:64]] and
            route["before_prompt_hash"] == route["after_prompt_hash"] == initial["primary"] and
            route["score_mode"] == "confidence" and route["routing_score_dtype"] == "torch.float16",
            f"{name}: preview changed parameters or used the wrong images/mode")
    ids, weights, scores = route["selected_ids"], route["selected_weights"], route["routing_scores"]
    require(len(ids) == len(set(ids)) == len(weights) == 2 and len(scores) == 8 and
            all(type(i) is int and 0 <= i < 8 for i in ids) and
            all(math.isfinite(v) and 0 <= v <= 1 for v in weights + scores) and
            min(scores[i] for i in ids) >= max(scores[i] for i in range(8) if i not in ids),
            f"{name}: invalid text Top-2 route")
    exps = [math.exp(scores[i] - max(scores[j] for j in ids)) for i in ids]
    require(max(abs(w - e / sum(exps)) for w, e in zip(weights, exps)) <= 0.001,
            f"{name}: text route weights do not match the scores")
    core = core_row["result"]
    require(core["num_images"] == expected and core["num_classes"] == classes and
            core["num_selected_samples"] == domain["selected"] and core["num_updates"] == domain["update_rounds"] and
            core["num_skipped_samples"] == expected - domain["selected"] and
            core["effective_confidence_threshold"] == 0.815 and core["effective_tta_lr"] == 7.5e-6 and
            core["teacher_supervision_mode"] == "aux_pseudolabel" and
            core["num_teacher_selected_samples"] == domain["selected"] and
            core["num_teacher_guided_updates"] == domain["update_rounds"] and
            core["routing_buffer_size"] == 64 and core["selected_private_expert_ids"] == ids and
            core["selected_private_expert_weights"] == weights and core["routing_scores"] == scores and
            core["visual_domain_enabled"] is True and core["domain_updates_enabled"] is True,
            f"{name}: core results do not match the report/route")
    close(core["top1_accuracy"], domain["correct"] / expected, f"{name}: core accuracy differs")
    optimizers = metadata["optimizer_creations"]
    expected_kinds = {"main"} | {f"visual_private_{i}" for i in range(4)}
    require(len(optimizers) == result["optimizer_instances"] == 5 and
            {o["kind"] for o in optimizers} == expected_kinds, f"{name}: wrong optimizer instances")
    adaptation = config["baseline_method"]["adaptation"]
    for optimizer in optimizers:
        require(optimizer["branch"] == "primary" and optimizer["domain"] == name and
                optimizer["initial_state_entries"] == 0 and optimizer["lr"] == 7.5e-6 and
                all(optimizer[k] == adaptation[k] for k in ("weight_decay", "betas", "eps")),
                f"{name}: optimizer was not independently initialized with the fixed settings")
    require(state["trace_file"] == f"primary.{name}.predictions.jsonl" and valid_digest(state["trace_sha256"]),
            f"{name}: unexpected trace identity")
    close(result["elapsed_seconds"], metadata["elapsed_seconds"], f"{name}: elapsed-time mismatch")
    return state, domain


def recount_control(path, records, num_classes):
    """Control logs record labels/predictions/confidence, but not full logits."""
    correct = count = 0
    with path.open() as handle:
        for index, (line, record) in enumerate(zip(handle, records, strict=True)):
            row = json.loads(line)
            require(row["index"] == index and row["relative_path"] == record["relative_path"] and
                    row["image_sha256"] == record["sha256"] and row["label"] == record["label_index"] and
                    row["prediction_source"] == "frozen_student_only", "Control trace identity/source mismatch")
            require(type(row["prediction"]) is int and 0 <= row["prediction"] < num_classes and
                    math.isfinite(row["confidence"]) and 0 <= row["confidence"] <= 1,
                    "Control prediction/confidence is invalid")
            count += 1
            correct += row["prediction"] == record["label_index"]
    return {"num_images": count, "correct": correct}


def audit_benchmark(entry, runtime_sources):
    name = entry["benchmark"]
    expected, classes = EXPECTED[name]
    tta, control = Path(entry["tta_output"]), Path(entry["control_output"])
    done, control_done = read_json(tta / "completion.json"), read_json(control / "completion.json")
    for filename in ("summary", "metadata", "domain_results", "input_manifest"):
        require(done[filename + "_sha256"] == file_sha256(tta / (filename + ".json")),
                f"{name}: changed TTA {filename}")
    for filename in ("summary.json", "config.json", "input_manifest.json", "predictions.jsonl"):
        require(control_done[filename] == file_sha256(control / filename), f"{name}: changed control {filename}")
    metadata, result, config = [read_json(tta / f) for f in ("metadata.json", "summary.json", "config.json")]
    baseline, control_config = [read_json(control / f) for f in ("summary.json", "config.json")]
    require(done["status"] == control_done["status"] == result["status"] == baseline["status"] == "PASS" and
            metadata["status"] == "COMPLETE", f"{name}: run is not complete")
    config_digest = validate_protocol(name, config, metadata, done)
    require(metadata["source_hashes"] == runtime_sources, f"{name}: runtime source hashes differ from frozen sources")
    validate_control_mapping(config, metadata, control_config)
    require(control_config["script_sha256"] == file_sha256(ROOT / "scripts/evaluate_imagenet_zeroshot.py"),
            f"{name}: frozen-control implementation differs")
    require(baseline["benchmark"] == name and baseline["method"] == "frozen_CLIP_B16_single_template" and
            baseline["num_scored_images"] == expected and baseline["adaptation_updates"] == 0 and
            baseline["teacher_output_fusion"] is False and type(baseline["correct"]) is int and
            0 <= baseline["correct"] <= expected, f"{name}: invalid complete control result")
    close(baseline["top1_percent"], 100 * baseline["correct"] / expected, f"{name}: wrong control denominator")
    records = validate_inputs(name, config, metadata, baseline, read_json(tta / "input_manifest.json"),
                              read_json(control / "input_manifest.json"))
    state, domain = validate_state(name, config, metadata, result, read_json(tta / "domain_results.json"), records)
    trace = tta / state["trace_file"]
    require(file_sha256(trace) == state["trace_sha256"], f"{name}: changed TTA trace")
    stats = recount_trace(trace, records, "primary", name, classes, 0.815, 3)
    require(all(stats[k] == domain[k] for k in stats), f"{name}: TTA trace recount differs from reported counts")
    control_stats = recount_control(control / "predictions.jsonl", records, classes)
    require(control_stats == {"num_images": expected, "correct": baseline["correct"]},
            f"{name}: control trace recount differs from reported counts")
    # Recheck the completion documents after all reads; completed outputs must
    # remain immutable throughout the audit.
    require(read_json(tta / "completion.json") == done and read_json(control / "completion.json") == control_done,
            f"{name}: completion evidence changed during the audit")
    return {"benchmark": name, "images": expected, "classes": classes,
            "tta_config_sha256": config_digest, "initial_prompt_sha256": state["initial_prompt_hash"],
            "final_prompt_sha256": state["final_prompt_hash"],
            "zero_shot_correct": baseline["correct"], "zero_shot_top1_percent": baseline["top1_percent"],
            "tta_correct": domain["correct"], "tta_top1_percent": domain["top1_percent"],
            "delta_percentage_points": domain["top1_percent"] - baseline["top1_percent"],
            "selected_images": domain["selected"], "update_rounds": domain["update_rounds"],
            "text_experts": state["route_audit"]["selected_ids"], "elapsed_seconds": result["elapsed_seconds"],
            "zero_shot_elapsed_seconds": baseline["elapsed_seconds"],
            "tta_output": str(tta.resolve()), "control_output": str(control.resolve()),
            "ordered_index_sha256": metadata["data"]["ordered_index_sha256"],
            "ordered_content_sha256": metadata["data"]["ordered_content_sha256"],
            "tta_trace_sha256": state["trace_sha256"], "control_trace_sha256": control_done["predictions.jsonl"],
            "tta_completion_sha256": file_sha256(tta / "completion.json"),
            "control_completion_sha256": file_sha256(control / "completion.json"),
            "tta_trace_recounted": True, "control_trace_recounted": True}


def render_markdown(summary):
    rows = summary["benchmarks"]
    table = "\n".join(
        f"| {r['benchmark']} | {r['images']:,} | {r['classes']} | {r['zero_shot_top1_percent']:.3f}% | "
        f"{r['tta_top1_percent']:.3f}% | {r['delta_percentage_points']:+.3f} | {r['selected_images']:,} | {r['update_rounds']:,} |"
        for r in rows)
    return f'''# ImageNet 自然分布偏移四件套测试结果

ImageNet-A、V2 matched-frequency、R、Sketch 均完成原版全量测试，共 **{summary['num_images']:,} 张**。八个方法/对照运行均为 COMPLETE/PASS；已复核配置、官方类别映射、输入/输出摘要，并逐图重算两组预测的正确数。未完成或失败尝试不计入结果。

| benchmark | 图像数 | 类别数 | 冻结 B/16 对照 | 本方法 Top-1 | 变化（百分点） | 入选更新图像 | 更新轮数 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

四件套 **等权 OOD Average**：本方法 **{summary['macro_tta_top1_percent']:.3f}%**，冻结对照 **{summary['macro_zero_shot_top1_percent']:.3f}%**，变化 **{summary['macro_delta_percentage_points']:+.3f} 个百分点**。先计算各 benchmark 的 Top-1，再取四者算术平均，每个 benchmark 权重为 1/4。

按全部 {summary['num_images']:,} 张图像加权的总准确率：本方法 **{summary['micro_tta_top1_percent']:.3f}%**，冻结对照 **{summary['micro_zero_shot_top1_percent']:.3f}%**；该值与上述等权 OOD Average 分开报告。

本轮固定协议：seed0；每个 benchmark 从相同提示初值、空优化器状态独立启动；单个 benchmark 内提示参数持续更新。冻结 OpenAI CLIP ViT-B/16 学生与 ViT-L/14 教师，仅主学生分支、batch1、逐图更新前计分，学生概率 argmax 决定输出；**不使用教师输出融合或校准，不使用 shadow**。教师仅提供权重 0.08 的硬伪标签辅助 CE；学生自身硬伪标签 CE 权重 1。学生置信度门限 0.815，学习率 7.5e-6，入选图像执行 3 步 AdamW。

各 benchmark 前 64 张无标签预览只用于固定文本 Top-2 路由，提示参数保持不变；随后从第 0 张开始，每张图仅正式计分一次。图像按 TPT 提示类别名、WNID、原相对路径排序。A/R 使用各自作者的 200 类 mask，V2/Sketch 使用 ImageNet 1000 类；重复提示名称不合并标签，V2 每类 10 张，所有原版成员均保留。A 原始文件名含收集模型分数字样，本轮按固定原路径排序，未读取当前模型分数选择成员或顺序。

冻结对照使用相同 B/16 权重、完整类别映射和输入流，单模板 `a photo of a {{class}}`，无适应、无教师、无提示集成。**对照推理 batch64，方法 batch1**；fp16 批形状可能带来数值差异，耗时也不作为同等批量下的速度对比。

四件套组合遵循常见自然分布偏移评测范围，但本轮的类别排序、前 64 张预览和当前图更新前计分，与部分论文的随机输入流、当前图适应后计分或不同模板设置存在差异。因此这些结果不能直接作为同协议 SOTA 复现或排名；本轮沿用固定配置，未使用这些测试结果搜索新超参数。

审计记录：四次方法运行采用相同运行源码哈希及相同初始提示哈希 `{summary['shared_initial_prompt_sha256']}`。逐图复核学生 logits/probabilities、argmax、门控和更新前累计步数；对照日志只记录预测/置信度，复核其标签与计分，不声称对照完整 logits 已被审计。输入图像字节哈希由正式运行记录，本次汇总校验清单与 trace 的绑定，不重新解码/重哈希原图。详细配置、来源目录与摘要见同目录 `{JSON_NAME}`；此前 A/V2 二件套报告保留。
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the JSON and Chinese Markdown reports")
    args = parser.parse_args(argv)
    manifest, entries = load_manifest(args.manifest)
    require_all_complete(entries)
    manifest_digest = file_sha256(args.manifest)
    sources = current_runtime_sources()
    rows = []
    for entry in entries:
        print(f"AUDIT_START {entry['benchmark']}", flush=True)
        rows.append(audit_benchmark(entry, sources))
        print(f"AUDIT_PASS {entry['benchmark']}", flush=True)
    initial_hashes = {r["initial_prompt_sha256"] for r in rows}
    require(len(initial_hashes) == 1, "Four benchmarks did not start with identical prompt parameters")
    total = sum(r["images"] for r in rows)
    require(total == TOTAL_IMAGES == manifest["expected_total_images"], "Wrong full-suite denominator")
    require(file_sha256(args.manifest) == manifest_digest and current_runtime_sources() == sources,
            "Manifest/runtime source changed during the audit")
    macro_tta = sum(r["tta_top1_percent"] for r in rows) / len(rows)
    macro_control = sum(r["zero_shot_top1_percent"] for r in rows) / len(rows)
    summary = {"schema": "imagenet_ood_four_benchmarks_results_v1", "status": "PASS", "seed": 0,
               "created_utc": datetime.now(timezone.utc).isoformat(), "num_images": total,
               "scope": "Full ImageNet-A / ImageNet-V2 matched-frequency / ImageNet-R / ImageNet-Sketch",
               "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_digest,
               "summarizer_sha256": file_sha256(Path(__file__)),
               "control_validation_helper_sha256": file_sha256(ROOT / "scripts/summarize_imagenet_pair.py"),
               "control_script_sha256": file_sha256(ROOT / "scripts/evaluate_imagenet_zeroshot.py"),
               "shared_runtime_source_hashes": sources,
               "shared_initial_prompt_sha256": next(iter(initial_hashes)),
               "independent_benchmark_initialization": True, "benchmark_state_transfer": False,
               "student_only_pre_update": True, "teacher_output_fusion": False, "teacher_output_calibration": False,
               "teacher_role": "frozen_auxiliary_hard_pseudolabel_loss_only", "teacher_ce_weight": 0.08,
               "student_confidence_threshold": 0.815, "learning_rate": 7.5e-6,
               "update_rounds_per_selected_image": 3, "preview_images_per_benchmark": 64,
               "preview_parameter_updates": 0, "tta_inference_batch_size": 1, "control_inference_batch_size": 64,
               "all_tta_and_control_predictions_recounted": True, "input_image_bytes_rehashed_by_summarizer": False,
               "new_test_hyperparameter_search": False, "literature_same_protocol_claim": False,
               "benchmarks": rows, "macro_definition": "arithmetic mean of four per-benchmark Top-1 percentages",
               "macro_tta_top1_percent": macro_tta, "macro_zero_shot_top1_percent": macro_control,
               "macro_delta_percentage_points": macro_tta - macro_control,
               "micro_definition": "total correct predictions divided by 98389 images",
               "micro_tta_top1_percent": 100 * sum(r["tta_correct"] for r in rows) / total,
               "micro_zero_shot_top1_percent": 100 * sum(r["zero_shot_correct"] for r in rows) / total}
    args.output.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary)
    save_json(args.output / JSON_NAME, summary)
    target = args.output / MARKDOWN_NAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"status": "PASS", "reports": [str(args.output / JSON_NAME), str(target)],
                      "macro_tta_top1_percent": macro_tta, "macro_zero_shot_top1_percent": macro_control},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
