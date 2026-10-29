"""Portable full-data PACS/OfficeHome protocol and input validation.

No historical output, ordered-input table, private data root or model import is
required. Class-directory order and full POSIX path order are generated from
the requested data root and recorded for each new run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .officehome import OfficeHomeDomain, collect_domain_examples, normalize_class_name


DOMAIN_COUNTS = {
    "pacs": {"art_painting": 2048, "cartoon": 2344, "photo": 1670, "sketch": 3929},
    "officehome": {"Art": 2427, "Clipart": 4365, "Product": 4439, "Real_World": 4357},
}
CLASS_COUNTS = {"pacs": 7, "officehome": 65}
WEIGHTS = {
    "student": ("ViT-B-16.pt", "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"),
    "teacher": ("ViT-L-14.pt", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"),
}
EVALUATION = {
    "seed": 0, "batch_size": 1, "stream_order": "class_name_then_full_file_path_sorted",
    "full_data": True, "shuffle": False, "score_before_update": True,
    "prediction_source": "student_only", "teacher_output_fusion": False,
    "teacher_output_calibration": False, "persistent_prompts": True,
    "reset_optimizer_each_domain": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_config_path(benchmark):
    require(benchmark in DOMAIN_COUNTS, "Only PACS and OfficeHome are supported")
    path = Path(__file__).resolve().parent / "configs" / f"{benchmark}.json"
    if path.is_file():
        return path
    raise FileNotFoundError("Default configuration is not installed; pass --config explicitly")


def validate_config(config, benchmark=None):
    require(config.get("schema") == "mt_tta_evaluation_v1", "Unsupported configuration schema")
    name = config.get("benchmark")
    require(name in DOMAIN_COUNTS and (benchmark is None or name == benchmark), "Benchmark/config mismatch")
    require(config.get("evaluation") == EVALUATION, "Evaluation rules are fixed: full data, seed0, persistent prompts and pre-update student output")
    data, method = config["data"], config["baseline_method"]
    counts = DOMAIN_COUNTS[name]
    require(data["domain_order"] == list(counts) and data["domain_counts"] == counts and
            data["num_scored_images"] == sum(counts.values()), "Full benchmark domain order/counts cannot be changed")
    classes = data["class_names"]
    require(data["num_classes"] == CLASS_COUNTS[name] == len(classes) and
            classes == sorted(set(classes)) and all(isinstance(c, str) and c for c in classes), "Invalid class mapping")
    expected_schedule = ([["primary", d] for d in counts] if name == "pacs" else
                         [[b, d] for d in list(counts)[:3] for b in ("text_shadow", "primary")] + [["text_shadow", "Real_World"]])
    expected_reported = {d: "text_shadow" if name == "officehome" and d == "Real_World" else "primary" for d in counts}
    require(data["branch_domain_schedule"] == expected_schedule and data["reported_branch"] == expected_reported and
            data["shadow_text_domains"] == (["Real_World"] if name == "officehome" else []), "The branch schedule and reported branch are fixed")
    require(data["formal_branch_image_visits_excluding_prerouting"] == sum(counts[d] for _, d in expected_schedule), "Wrong branch compute count")
    require(method["student"] == "ViT-B/16" and method["teacher"] == "ViT-L/14" and method["logit_scale"] == 100.0,
            "This implementation uses frozen CLIP ViT-B/16 and ViT-L/14 with logit scale 100")
    require(method["preprocessing"] == {"resize": "bicubic_short_side_224", "crop": "center_crop_224x224",
            "color": "RGB", "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
            "normalize_std": [0.26862954, 0.26130258, 0.27577711], "extra_random_augmentation": False, "multi_view": False},
            "Only deterministic OpenAI CLIP preprocessing is supported")
    require(method["bank_rng_seed_offsets"] == {"text_shared": 0, "text_private": 10000, "visual_shared": 20000, "visual_private": 30000},
            "Prompt bank seed offsets are fixed by the core implementation")
    require(method["numeric_policy"] == "unchanged_local_CLIP_CUDA_dtypes; external_prompt_parameters_fp32; no_added_AMP_or_precision_override",
            "Numeric policy must preserve the measured implementation")
    for role in ("student", "teacher"):
        filename = method[role + "_weights"]
        digest = method[role + "_weights_sha256"]
        require((filename, digest) == WEIGHTS[role], "The selected frozen OpenAI CLIP weights and cache filenames are fixed")
    text, visual, adaptation = method["text_prompt"], method["visual_prompt"], method["adaptation"]
    def number(value, name, low=0, high=None, integer=False, strict_low=False):
        allowed_types = (int,) if integer else (int, float)
        require(type(value) in allowed_types and math.isfinite(value) and
                (value > low if strict_low else value >= low) and (high is None or value <= high),
                "Invalid numeric parameter: " + name)
    for key in ("num_context", "shared_experts", "private_experts"):
        number(text[key], "text." + key, low=1, integer=True)
    number(text["private_topk"], "text.private_topk", low=1, high=text["private_experts"], integer=True)
    number(text["preview_size"], "text.preview_size", low=1, integer=True)
    number(text["shared_mix_weight"], "text.shared_mix_weight", high=1)
    number(text["expert_init_std"], "text.expert_init_std")
    number(text["router_temperature"], "text.router_temperature", strict_low=True)
    for key in ("num_tokens", "shared_experts", "private_experts"):
        number(visual[key], "visual." + key, low=1, integer=True)
    number(visual["private_topk"], "visual.private_topk", low=1, high=visual["private_experts"], integer=True)
    number(visual["shared_mix_weight"], "visual.shared_mix_weight", high=1)
    for key in ("scale_init", "expert_init_std", "position_extra_noise_std"):
        number(visual[key], "visual." + key)
    number(visual["router_temperature"], "visual.router_temperature", strict_low=True)
    number(adaptation["teacher_temperature"], "teacher_temperature", strict_low=True)
    number(adaptation["teacher_ce_weight"], "teacher_ce_weight")
    number(adaptation["update_rounds_per_selected_image"], "update_rounds_per_selected_image", low=1, integer=True)
    require(text["router_score"] in {"mean_max_class_probability_across_preview_images", "mean_probability_margin", "mean_normalized_neg_entropy"}, "Unsupported text router")
    require(type(text["preview_size"]) is int and text["preview_size"] >= 1 and
            data["preview_image_visits"] == sum(min(text["preview_size"], counts[d]) for _, d in expected_schedule), "Invalid preview count")
    require(text["preview_members"] == f"first_min_{text['preview_size']}_images_of_this_domain_sorted_stream" and
            text["preview_image_features"] == "frozen_student_image_encoder_without_visual_prompts",
            "Preview descriptions must match the configured size and implemented student features")
    require(text["preview_parameter_updates"] == 0 and text["preview_uses_ground_truth_or_teacher"] is False and
            text["preview_images_scored_later"] is True and text["post_preview_scoring_start_index"] == 0 and
            text["route_ids_and_weights_fixed_within_domain"] is True, "Preview cannot update parameters or omit formal images")
    require(visual["expert_init_std"] == text["expert_init_std"], "Text and visual banks share the constructor's initialization std")
    require(visual["router_source"] in {"current_text_moe", "fixed_zeroshot_text"}, "Unsupported visual router")
    require(adaptation["teacher_supervision"] == "aux_pseudolabel" and adaptation["student_pseudo_topk"] == 1 and
            adaptation["teacher_confidence_gate"] is False and adaptation["optimizer"] == "AdamW" and
            adaptation["weight_decay"] == 0.0 and adaptation["betas"] == [0.9, 0.999] and adaptation["eps"] == 1e-8 and
            adaptation["entropy_weight"] == adaptation["shared_anchor_weight"] == 0.0,
            "Only the declared student-gated hard pseudo-label AdamW method is exposed")
    require(adaptation["loss"] == f"student_hard_pseudolabel_CE_plus_{adaptation['teacher_ce_weight']}_teacher_hard_pseudolabel_CE" and
            adaptation["sample_gate"] == "preupdate_student_max_class_probability_greater_equal_benchmark_threshold" and
            adaptation["targets_gate_routes_fixed_during_image_updates"] is True,
            "Adaptation descriptions must match the configured teacher weight and implemented gate/update semantics")
    require(all(adaptation[key] is False for key in ("distill_kl_enabled", "orthogonalization_enabled", "ema_teacher_enabled", "replay_enabled")), "Unsupported adaptation extension")
    require(type(adaptation["update_rounds_per_selected_image"]) is int and adaptation["update_rounds_per_selected_image"] >= 1,
            "Each selected image must receive at least one update")
    require(set(data["domain_learning_rate_overrides"]) <= set(counts), "Unknown learning-rate domain")
    for value in [data["learning_rate"], *data["domain_learning_rate_overrides"].values()]:
        number(value, "learning_rate", strict_low=True)
    number(data["student_confidence_threshold"], "student_confidence_threshold", high=1)
    require(data["student_confidence_threshold"] < 1, "A threshold of one disables updates in the underlying core")
    return config


def load_config(benchmark, path=None):
    path = Path(path) if path is not None else default_config_path(benchmark)
    return validate_config(read_json(path), benchmark)


def scan_domain(domain_path, class_names, expected_count, hash_images=False):
    """Validate one complete domain and return its deterministic relative index."""
    domain_path = Path(domain_path)
    classes, examples = collect_domain_examples(domain_path)
    require(classes == tuple(class_names), f"{domain_path.name}: class directories differ from the declared mapping")
    require(len(examples) == expected_count, f"{domain_path.name}: expected all {expected_count} images, found {len(examples)}")
    records = [{"relative_path": e.image_path.relative_to(domain_path).as_posix(),
                "label_index": e.label_index, "class_name": e.class_name,
                **({"sha256": file_sha256(e.image_path)} if hash_images else {})} for e in examples]
    require(len({r["relative_path"] for r in records}) == len(records), "Repeated relative image path")
    return examples, records


def prepare_data(config, data_root, hash_images=False):
    validate_config(config)
    root = Path(data_root).expanduser().resolve()
    require(root.is_dir(), "Data root must be an existing directory")
    data = config["data"]
    actual_domains = sorted(p.name for p in root.iterdir() if p.is_dir())
    require(actual_domains == sorted(data["domain_order"]), "Data root must contain exactly the four declared domain directories")
    specs, records, domains = {}, {}, []
    classes = tuple(data["class_names"])
    for domain in data["domain_order"]:
        path = root / domain
        examples, rows = scan_domain(path, classes, data["domain_counts"][domain], hash_images)
        records[domain] = rows
        specs[domain] = OfficeHomeDomain(config["benchmark"] + "_" + domain, root, (), domain,
                                        path, classes, tuple(normalize_class_name(c) for c in classes), examples)
        domains.append({"domain": domain, "num_images": len(rows), "num_classes": len(classes),
                        "ordered_index_sha256": canonical_hash([[r["relative_path"], r["label_index"], r["class_name"]] for r in rows]),
                        "ordered_content_sha256": canonical_hash([[r["relative_path"], r["sha256"]] for r in rows]) if hash_images else None})
    return specs, records, {"benchmark": config["benchmark"], "data_root": str(root),
                             "num_scored_images": sum(len(rows) for rows in records.values()),
                             "image_bytes_hashed": hash_images, "domains": domains,
                             "order": EVALUATION["stream_order"]}


def require_new_output(path):
    output = Path(path).expanduser().resolve()
    require(not output.exists(), "Refusing to overwrite an existing output directory: " + str(output))
    return output
