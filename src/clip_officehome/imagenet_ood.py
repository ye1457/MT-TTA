"""Full ImageNet natural-shift evaluation with independent benchmark state.

The dataset adapter changes class metadata and full-set validation only. Model
adaptation and trace auditing are shared with the PACS/OfficeHome evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .constants import IMAGE_EXTENSIONS
from .officehome import ImageExample, OfficeHomeDomain
from .protocol import canonical_hash, file_sha256, load_config, read_json, require


COUNTS = {"imagenet_a": 7500, "imagenet_v2": 10000,
          "imagenet_r": 30000, "imagenet_sketch": 50889}
CLASS_COUNTS = {"imagenet_a": 200, "imagenet_v2": 1000,
                "imagenet_r": 200, "imagenet_sketch": 1000}
CLASS_METADATA_SHA256 = "4f8e2b5c4fdcb12ac2494e322018c75809627c69f638dc2e8584a526972be00d"
ORDER = "TPT_prompt_name_then_WNID_then_relative_file_path_sorted"


def class_metadata():
    path = Path(__file__).resolve().parent / "configs/imagenet_classes.json"
    require(file_sha256(path) == CLASS_METADATA_SHA256, "ImageNet class metadata changed")
    metadata = read_json(path)
    classes = metadata["classes"]
    require([c["imagenet_index"] for c in classes] == list(range(1000)), "Invalid ImageNet index order")
    require(len({c["wnid"] for c in classes}) == 1000, "Non-unique ImageNet WNIDs")
    for benchmark in COUNTS:
        indices = metadata[benchmark + "_indices"]
        require(indices == sorted(set(indices)) and len(indices) == CLASS_COUNTS[benchmark]
                and all(type(i) is int and 0 <= i < 1000 for i in indices), "Invalid benchmark class mask")
    return metadata


def benchmark_config(benchmark):
    require(benchmark in COUNTS, "Unknown ImageNet natural-shift benchmark")
    labels = class_metadata()
    indices = labels[benchmark + "_indices"]
    entries = [labels["classes"][i] for i in indices]
    # Transfer the already declared OfficeHome primary method without tuning on
    # the new test data. Its shadow branch and Clipart LR exception do not apply.
    config = deepcopy(load_config("officehome"))
    config.pop("historical_reference", None)
    config.update(schema="mt_tta_imagenet_ood_v1", benchmark=benchmark)
    config["evaluation"].update(stream_order=ORDER,
        independent_benchmark_initialization=True,
        benchmark_state_transfer=False)
    config["data"] = {
        "display_name": benchmark, "num_classes": len(indices),
        "domain_order": [benchmark], "domain_counts": {benchmark: COUNTS[benchmark]},
        "num_scored_images": COUNTS[benchmark], "student_confidence_threshold": 0.815,
        "learning_rate": 7.5e-6, "domain_learning_rate_overrides": {},
        "shadow_text_domains": [], "reported_branch": {benchmark: "primary"},
        "branch_domain_schedule": [["primary", benchmark]],
        "formal_branch_image_visits_excluding_prerouting": COUNTS[benchmark],
        "preview_image_visits": 64,
        "class_names": [entry["wnid"] for entry in entries],
        "prompt_class_names": [entry["class_name"] for entry in entries],
        "imagenet_indices": indices, "class_metadata_sha256": CLASS_METADATA_SHA256,
        "class_name_source": labels["source_repository"],
        "class_name_source_revision": labels["source_revision"],
        "class_name_duplicates_preserved": labels["duplicate_prompt_names"],
    }
    config["transfer_policy"] = {
        "source": "officehome primary branch configuration",
        "new_test_hyperparameter_search": False,
        "benchmark_boundary": "fresh seed0 session for each benchmark",
        "preview": "first64 unlabeled lookahead, no parameter updates; all preview images scored later once",
        "literature_comparability": "Current-image pre-update prediction and class-sorted stream differ from methods that adapt the current image before scoring or shuffle the stream.",
    }
    return config


def _scan(root, benchmark, entries, expected_count, hash_images):
    """Map numeric/WNID directories without equating display names with labels."""
    root = Path(root).expanduser().resolve()
    require(root.is_dir(), "Data root must exist")
    folder_to_entry = {
        str(entry["imagenet_index"]) if benchmark == "imagenet_v2" else entry["wnid"]:
            (local_index, entry)
        for local_index, entry in enumerate(entries)
    }
    folders = {p.name for p in root.iterdir() if p.is_dir()}
    require(folders == set(folder_to_entry), "Class directories differ from the official benchmark mask")
    rows = []
    for folder, (local_index, entry) in folder_to_entry.items():
        files = sorted(p for p in (root / folder).rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        require(files, "Empty class: " + folder)
        if benchmark == "imagenet_v2":
            require(len(files) == 10, "Matched-frequency ImageNet-V2 needs exactly 10 images per canonical class")
        for path in files:
            rows.append({"relative_path": path.relative_to(root).as_posix(),
                         "label_index": local_index, "class_name": entry["class_name"],
                         "wnid": entry["wnid"], "imagenet_index": entry["imagenet_index"],
                         **({"sha256": file_sha256(path)} if hash_images else {})})
    require(len(rows) == expected_count, f"Expected the full {expected_count} images, found {len(rows)}")
    require(len({r["relative_path"] for r in rows}) == len(rows), "Repeated input path")
    rows.sort(key=lambda r: (r["class_name"], r["wnid"], r["relative_path"]))
    examples = tuple(ImageExample(root / r["relative_path"], r["label_index"], r["wnid"]) for r in rows)
    return root, examples, rows


def prepare_imagenet_data(benchmark, root, hash_images=False):
    config = benchmark_config(benchmark)
    labels = class_metadata()
    entries = [labels["classes"][i] for i in config["data"]["imagenet_indices"]]
    root, examples, rows = _scan(root, benchmark, entries, COUNTS[benchmark], hash_images)
    spec = OfficeHomeDomain(benchmark, root, (), benchmark, root,
                           tuple(config["data"]["class_names"]),
                           tuple(config["data"]["prompt_class_names"]), examples)
    inventory = {
        "benchmark": benchmark, "data_root": str(root), "num_scored_images": len(rows),
        "num_classes": len(entries), "order": ORDER, "image_bytes_hashed": hash_images,
        "class_metadata_sha256": CLASS_METADATA_SHA256,
        "class_space": "official 200-class subset" if len(entries) == 200 else "ImageNet canonical 1000 classes",
        "class_counts": dict(sorted(Counter(r["wnid"] for r in rows).items())),
        "ordered_index_sha256": canonical_hash([[r["relative_path"], r["label_index"], r["imagenet_index"]] for r in rows]),
        "ordered_content_sha256": canonical_hash([[r["relative_path"], r["sha256"]] for r in rows]) if hash_images else None,
        "domains": [{"domain": benchmark, "num_images": len(rows), "num_classes": len(entries)}],
    }
    return config, {benchmark: spec}, {benchmark: rows}, inventory


def main(argv=None):
    import json
    from .evaluation import run_prepared
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(COUNTS), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Class folders directly below this path")
    parser.add_argument("--model-cache", type=Path, default=Path.home() / ".cache/clip")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config, specs, records, inventory = prepare_imagenet_data(args.benchmark, args.data_root, not args.dry_run)
    if args.dry_run:
        result = {"status": "DRY_RUN_PASS_NO_MODEL", "config": config, "data": inventory}
    else:
        require(args.output_dir is not None, "--output-dir is required for formal evaluation")
        result = run_prepared(config, specs, records, inventory, args.model_cache, args.gpu, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
