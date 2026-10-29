"""Summarize completed A/V2 experiments only after checking their audit digests."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import json
from clip_officehome.protocol import canonical_hash, file_sha256, read_json, require, save_json


def validate_report_protocol(name, config, metadata, completion=None):
    """Bind the readable configuration and every fixed statement in the report."""
    expected_images = {"imagenet_a": 7500, "imagenet_v2": 10000}[name]
    expected_classes = {"imagenet_a": 200, "imagenet_v2": 1000}[name]
    digest = canonical_hash(config)
    require(config == metadata["config"] and digest == metadata["config_sha256"],
            "TTA config is not the configuration recorded in metadata")
    if completion is not None:
        require(digest == completion["config_sha256"], "TTA config differs from the completed run")
    require(config["benchmark"] == metadata["benchmark"] == name and
            config["schema"] == "mt_tta_imagenet_ood_v1", "Unexpected benchmark/schema")
    evaluation, data, method = config["evaluation"], config["data"], config["baseline_method"]
    text, adaptation = method["text_prompt"], method["adaptation"]
    require(evaluation["seed"] == 0 and evaluation["batch_size"] == 1 and
            evaluation["full_data"] is True and evaluation["shuffle"] is False and
            evaluation["score_before_update"] is True and evaluation["prediction_source"] == "student_only" and
            evaluation["teacher_output_fusion"] is False and evaluation["teacher_output_calibration"] is False,
            "Report requires full seed0, batch1, pre-update student-only predictions")
    require(evaluation["independent_benchmark_initialization"] is True and
            evaluation["benchmark_state_transfer"] is False, "Wrong benchmark state-transfer protocol")
    require(evaluation["stream_order"] == "TPT_prompt_name_then_WNID_then_relative_file_path_sorted",
            "Input ordering differs from the report")
    require(data["domain_order"] == [name] and data["domain_counts"] == {name: expected_images} and
            data["num_scored_images"] == expected_images and data["num_classes"] == expected_classes and
            data["branch_domain_schedule"] == [["primary", name]] and data["reported_branch"] == {name: "primary"} and
            data["shadow_text_domains"] == [], "Report requires a single full primary branch without shadow")
    require(data["student_confidence_threshold"] == 0.815 and data["learning_rate"] == 7.5e-6 and
            data["domain_learning_rate_overrides"] == {} and adaptation["update_rounds_per_selected_image"] == 3 and
            adaptation["optimizer"] == "AdamW", "Adaptation hyperparameters differ from the report")
    require(method["student"] == "ViT-B/16" and method["teacher"] == "ViT-L/14" and
            method["logit_scale"] == 100.0 and adaptation["teacher_supervision"] == "aux_pseudolabel" and
            adaptation["teacher_ce_weight"] == 0.08 and adaptation["student_pseudo_topk"] == 1 and
            adaptation["teacher_confidence_gate"] is False,
            "Model/teacher role differs from the report")
    require(metadata["teacher_output_fusion"] is False and metadata["teacher_output_calibration"] is False and
            metadata["output_rule"] == "student_probability_argmax_only", "Invalid recorded output rule")
    require(text["preview_size"] == 64 and text["private_topk"] == 2 and
            text["preview_parameter_updates"] == 0 and text["preview_uses_ground_truth_or_teacher"] is False and
            text["preview_images_scored_later"] is True and text["post_preview_scoring_start_index"] == 0 and
            text["route_ids_and_weights_fixed_within_domain"] is True,
            "Preview/routing protocol differs from the report")
    require(config["transfer_policy"]["new_test_hyperparameter_search"] is False,
            "Report cannot claim no new-test hyperparameter search")
    for role in ("student", "teacher"):
        require(metadata["model_hashes"][method[role + "_weights"]] == method[role + "_weights_sha256"],
                "Recorded model weights differ from the configuration")
    return digest


def validate_control_mapping(config, metadata, control_config):
    """Require the control to use exactly the same student and class space."""
    data, method = config["data"], config["baseline_method"]
    require(control_config["benchmark"] == config["benchmark"] and control_config["model"] == method["student"] and
            control_config["seed"] == 0 and control_config["template"] == "a photo of a {class}" and
            control_config["logit_scale"] == method["logit_scale"] == 100.0 and
            control_config["adaptation"] is False and control_config["teacher_used"] is False and
            control_config["prompt_ensemble"] is False and control_config["inference_batch_size"] == 64,
            "Frozen control differs from the protocol described in the report")
    require(control_config["model_sha256"] == method["student_weights_sha256"] ==
            metadata["model_hashes"][method["student_weights"]], "TTA/control student weights differ")
    require(control_config["class_metadata_sha256"] == data["class_metadata_sha256"] and
            control_config["class_names"] == data["prompt_class_names"] and
            control_config["num_classes"] == data["num_classes"] and
            control_config["order"] == config["evaluation"]["stream_order"],
            "TTA/control class mapping, prompt order or stream order differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tta-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, initial_prompt_hashes = [], []
    for name, expected in [("imagenet_a", 7500), ("imagenet_v2", 10000)]:
        tta, control = args.tta_root / name, args.control_root / name
        done = read_json(tta / "completion.json")
        metadata, result = read_json(tta / "metadata.json"), read_json(tta / "summary.json")
        require(done["status"] == result["status"] == "PASS" and metadata["status"] == "COMPLETE", "Unfinished TTA run")
        for filename in ["summary", "metadata", "domain_results", "input_manifest"]:
            require(done[filename + "_sha256"] == file_sha256(tta / (filename + ".json")), "Changed TTA artifact")
        for state in metadata["domain_states"]:
            require(file_sha256(tta / state["trace_file"]) == state["trace_sha256"], "Changed TTA predictions")
        config = read_json(tta / "config.json")
        config_digest = validate_report_protocol(name, config, metadata, done)
        require(result["num_scored_images"] == expected and result["all_student_predictions_recounted"], "Incomplete scoring")
        require(result["prediction_rule"] == "pre_update_student_only" and not result["teacher_output_fusion"], "Invalid prediction rule")
        control_done, baseline = read_json(control / "completion.json"), read_json(control / "summary.json")
        require(control_done["status"] == baseline["status"] == "PASS", "Unfinished control")
        for filename in ["summary.json", "config.json", "input_manifest.json", "predictions.jsonl"]:
            require(control_done[filename] == file_sha256(control / filename), "Changed control artifact")
        control_config = read_json(control / "config.json")
        validate_control_mapping(config, metadata, control_config)
        require(baseline["num_scored_images"] == expected, "Incomplete control")
        require(metadata["data"]["ordered_content_sha256"] == baseline["ordered_content_sha256"], "TTA/control input streams differ")
        tta_inputs, control_inputs = read_json(tta / "input_manifest.json"), read_json(control / "input_manifest.json")
        require(tta_inputs["records"][name] == control_inputs["records"][name] and
                tta_inputs["inventory"]["ordered_index_sha256"] == control_inputs["inventory"]["ordered_index_sha256"],
                "TTA/control per-image labels or class identities differ")
        require(result["benchmark"] == baseline["benchmark"] == name and
                baseline["method"] == "frozen_CLIP_B16_single_template" and baseline["adaptation_updates"] == 0 and
                baseline["teacher_output_fusion"] is False, "Unexpected result benchmark/control method")
        require(len(result["domains"]) == 1 and len(metadata["domain_states"]) == 1, "Unexpected reported domains/states")
        domain = result["domains"][0]
        state = metadata["domain_states"][0]
        require(domain["domain"] == state["domain"] == name and domain["branch"] == state["branch"] == "primary" and
                domain["num_images"] == state["num_scored"] == expected and domain["is_reported"] and state["is_reported"],
                "Unexpected reported primary branch")
        require(0 <= domain["correct"] <= expected and 0 <= baseline["correct"] <= expected and
                abs(domain["top1_percent"] - 100 * domain["correct"] / expected) < 1e-10 and
                abs(baseline["top1_percent"] - 100 * baseline["correct"] / expected) < 1e-10 and
                abs(result["macro_top1_percent"] - domain["top1_percent"]) < 1e-10 and
                0 <= domain["selected"] <= expected and domain["update_rounds"] == 3 * domain["selected"],
                "Result counts, denominators or update rounds disagree")
        initial = metadata["initial_prompt_hashes"]["primary"]
        require(isinstance(initial, str) and len(initial) == 64 and state["initial_prompt_hash"] == initial,
                "Initial prompt state is not bound to the reported branch")
        initial_prompt_hashes.append(initial)
        rows.append({"benchmark": name, "images": expected, "classes": config["data"]["num_classes"],
            "tta_config_sha256": config_digest, "initial_prompt_sha256": initial,
            "zero_shot_correct": baseline["correct"], "zero_shot_top1_percent": baseline["top1_percent"],
            "tta_correct": domain["correct"], "tta_top1_percent": domain["top1_percent"],
            "delta_percentage_points": domain["top1_percent"] - baseline["top1_percent"],
            "selected_images": domain["selected"], "update_rounds": domain["update_rounds"],
            "text_experts": metadata["domain_states"][0]["route_audit"]["selected_ids"],
            "elapsed_seconds": result["elapsed_seconds"],
            "tta_output": str(tta.resolve()), "control_output": str(control.resolve())})
    require(len(set(initial_prompt_hashes)) == 1, "The two benchmarks did not start with identical prompt parameters")
    summary = {"status": "PASS", "seed": 0, "num_images": 17500,
        "scope": "Two selected natural-shift benchmarks; not the four-benchmark OOD Average",
        "independent_benchmark_initialization": True, "student_only_pre_update": True,
        "teacher_role": "frozen_auxiliary_hard_pseudolabel_loss_only", "teacher_ce_weight": 0.08,
        "shared_initial_prompt_sha256": initial_prompt_hashes[0],
        "summarizer_sha256": file_sha256(Path(__file__)),
        "benchmarks": rows, "macro_tta_top1_percent": sum(r["tta_top1_percent"] for r in rows) / 2,
        "macro_zero_shot_top1_percent": sum(r["zero_shot_top1_percent"] for r in rows) / 2,
        "micro_tta_top1_percent": 100 * sum(r["tta_correct"] for r in rows) / 17500}
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "imagenet_a_v2_seed0_results.json", summary)
    table = "\n".join(f"| {r['benchmark']} | {r['images']:,} | {r['zero_shot_top1_percent']:.2f}% | {r['tta_top1_percent']:.2f}% | {r['delta_percentage_points']:+.2f} | {r['selected_images']:,} | {r['update_rounds']:,} |" for r in rows)
    report = f'''# ImageNet-A / V2 首轮测试结果

两组原版全量测试均完成，逐图预测已复核。seed0；各benchmark从相同初始状态独立运行。本轮未按测试结果搜索超参数。

| benchmark | 图像数 | 冻结B/16对照 | 本方法Top-1 | 变化（百分点） | 入选更新图像 | 更新轮数 |
|---|---:|---:|---:|---:|---:|---:|
{table}

两benchmark等权平均：本方法 **{summary['macro_tta_top1_percent']:.3f}%**，冻结对照 **{summary['macro_zero_shot_top1_percent']:.3f}%**。该数值不是完整四件套的OOD Average。

方法：冻结OpenAI CLIP ViT-B/16学生、ViT-L/14辅助硬伪标签教师（CE权重0.08）；主学生分支输出；逐图更新前计分；门限0.815、学习率7.5e-6、入选图3步AdamW。前64张无标签预览固定文本Top-2路由，无预览更新，随后每张图仅正式计分一次。所有数据按提示类别名、WNID、原路径排序。无教师输出融合、无shadow。

对照使用同一B/16权重、同一完整类别映射/输入流和单模板`a photo of a {{class}}`，无适应、无教师；推理batch64，方法batch1，存在fp16批形状的数值差异可能。

ImageNet-A采用作者200类mask，保留7500个官方成员（含原包重复）；V2采用matched-frequency的1000类、每类10张。显示名称采用固定TPT词表，重复名称不合并标签。A部分原始文件名带收集模型分数字样；本次固定原路径排序，不读取当前模型分数改变成员或顺序。

该协议含预览、类别排序和更新前计分，与当前图适应后计分的论文数字不同，不能直接宣称复现其同协议SOTA。未完成的资源失败尝试均保留为失败证据，不计入本结果；正式完成记录和清单摘要见结果JSON中的输出目录。

仅修复重复文本计算与跨图计算图生命周期，类别、损失、更新步数和数值精度未因显存问题改变。资源修复的数值/梯度对照和多图验证记录另存于outputs。
'''
    (args.output / "imagenet_a_v2_seed0_results_zh.md").write_text(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
