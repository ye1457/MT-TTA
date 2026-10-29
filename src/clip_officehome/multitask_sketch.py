"""Frozen MT-Sketch4-20K inputs and the first, untuned continual protocol.

Task identity selects a native class space; it never selects a separate prompt
bank. The data object's labels are used only by the external scorer.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path, PurePosixPath

from .officehome import OfficeHomeDomain, normalize_class_name
from .protocol import canonical_hash, file_sha256, load_config, read_json, require


TASKS = {"domainnet_sketch": 345, "imagenet_sketch": 1000, "tuberlin": 250, "sketchy": 125}
STREAM = "balanced_round_robin"
FROZEN = {
    "manifest.jsonl": "89f07bf643e621ecaef70010d51d2044d0750213aac19782324b27a615343570",
    "metadata/classes.json": "b0be1d01dc78230ea56a51836c19443a2a3fbd681fd63b65c182124f9b96b19e",
    "metadata/protocol.json": "a1e2793193f4b4b20844d318e25c65643d1f93fde45e0fc3bd8af4878be0c124",
    "streams.json": "a2b020b296ff32afa622e24b780ca794838cb5ebb3fc55f1bbfdae38a7693d71",
    "benchmark.py": "ae80c683c941bb24fd256f24566c8960eba5f8beea056262014db3ec26bbaade",
}


@dataclass
class PreparedSketch:
    root: Path
    benchmark: object
    config: dict
    inventory: dict
    records: list
    classes: dict
    order: list
    prompt_names: dict

    def task_indices(self, task):
        return [i for i in self.order if self.records[i]["task_id"] == task]

    def reference_spec(self):
        task = next(iter(TASKS))
        return OfficeHomeDomain("MT-Sketch4-20K", self.root, (), task, self.root,
            tuple(str(c["class_key"]) for c in self.classes[task]),
            tuple(self.prompt_names[task]), ())


def fixed_config(classes, prompt_names):
    config = deepcopy(load_config("officehome"))
    config.pop("historical_reference", None)
    config.update(schema="mt_tta_mtsketch_v1", benchmark="MT-Sketch4-20K")
    config["evaluation"].update(stream_order=STREAM, task_known=True,
        reset_optimizer_each_domain=False, reset_optimizer_on_task_switch=False,
        reset_prompt_on_task_switch=False, independent_task_initialization=False,
        number_of_prompt_banks=1, optimizer_state_persistent=True)
    text = config["baseline_method"]["text_prompt"]
    text.update(preview_members="first_64_within_each_task_of_frozen_round_robin_stream",
        preview_timing="all_four_task_routes_before_any_adaptation",
        total_preview_images=256, route_ids_and_weights_fixed_within_task=True)
    text.pop("route_ids_and_weights_fixed_within_domain", None)
    config["baseline_method"]["preprocessing"]["decode"] = "EXIF_transpose_alpha_on_white_then_RGB"
    config["data"] = {
        "task_order": list(TASKS), "task_class_counts": TASKS.copy(),
        "task_image_counts": dict.fromkeys(TASKS, 5000), "num_scored_images": 20000,
        "student_confidence_threshold": 0.815, "learning_rate": 7.5e-6,
        "domain_learning_rate_overrides": {}, "shadow_text_domains": [],
        "preview_image_visits": 256, "formal_image_visits": 20000,
        "class_space": "task_known_native_class_indices",
        "classes": classes, "prompt_class_names": prompt_names,
        "prompt_name_normalization": "replace_underscore_with_space_then_strip; preserve_native_synonyms",
        "frozen_artifacts": FROZEN.copy(),
        "primary_metric": "equal_mean_of_four_task_top1_percentages",
    }
    config["transfer_policy"] = {
        "source": "fixed OfficeHome primary / ImageNet OOD method, no shadow or LR exception",
        "new_test_hyperparameter_search": False,
        "state": "one shared session; prompts and per-parameter AdamW state persist across task switches",
        "routing": "one frozen Top2 text route per task, all from identical initial prompts; imagewise visual Top1",
        "labels": "ground truth absent from adaptation/session interface; external scorer only",
        "dataset_status": "frozen 1.0-review; native label conflicts and review flags retained",
        "fresh_unseen_test_claim": False,
    }
    return config


def validate_structure(records, classes, order):
    require(list(classes) == list(TASKS), "Task order differs from frozen protocol")
    require(len(records) == 20000 and Counter(r["task_id"] for r in records) == dict.fromkeys(TASKS, 5000),
            "Expected all four tasks, 5000 images each")
    require(len(order) == 20000 and sorted(order) == list(range(20000)), "Invalid full stream")
    require([records[i]["task_id"] for i in order] == list(TASKS) * 5000,
            "Stream is not balanced task round-robin")
    all_paths, all_ids = set(), set()
    for task, count in TASKS.items():
        entries = classes[task]
        require(len(entries) == count and [c["class_index"] for c in entries] == list(range(count)),
                "Invalid task-local class mapping")
        require(len({str(c["class_key"]) for c in entries}) == count, "Repeated native class key")
        indices = [i for i in order if records[i]["task_id"] == task]
        expected = sorted(indices, key=lambda i: (records[i]["class_name"], records[i]["relative_path"]))
        require(indices == expected, "Within-task order differs from class/path sorting")
        require({records[i]["class_index"] for i in indices} == set(range(count)), "Missing native class")
    for i, r in enumerate(records):
        task, label = r["task_id"], r["class_index"]
        require(r["manifest_index"] == i and type(label) is int and 0 <= label < TASKS[task], "Bad manifest index")
        c = classes[task][label]
        require(c["class_name"] == r["class_name"] and c["native_class_id"] == r["class_id"], "Native label mismatch")
        p = PurePosixPath(r["image_path"])
        require(not p.is_absolute() and ".." not in p.parts and p.parts[:2] == ("images", task), "Unsafe image path")
        require(r["image_path"] not in all_paths and r["sample_id"] not in all_ids, "Repeated image identity")
        all_paths.add(r["image_path"])
        all_ids.add(r["sample_id"])


def prepare_mtsketch(data_root, hash_images=True):
    root = Path(data_root).expanduser().resolve()
    for relative, expected in FROZEN.items():
        require(file_sha256(root / relative) == expected, "Frozen benchmark artifact changed: " + relative)
    spec = importlib.util.spec_from_file_location("mt_sketch4_frozen_benchmark", root / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    benchmark = module.Benchmark(root, stream=STREAM)
    records, classes, order = benchmark.records, benchmark.classes, benchmark.order
    validate_structure(records, classes, order)
    paths = set()
    for r in records:
        p = root / r["image_path"]
        require(p.is_file() and not p.is_symlink() and root in p.resolve().parents, "Missing/nonlocal frozen image")
        require(p.stat().st_size == r["file_bytes"], "Image size changed: " + r["image_path"])
        if hash_images:
            require(file_sha256(p) == r["file_sha256"], "Image bytes changed: " + r["image_path"])
        paths.add(p.relative_to(root).as_posix())
    require({p.relative_to(root).as_posix() for p in (root / "images").rglob("*") if p.is_file()} == paths,
            "Image tree differs from frozen membership")
    prompt_names = {t: [normalize_class_name(c["class_name"]) for c in entries] for t, entries in classes.items()}
    config = fixed_config(classes, prompt_names)
    inventory = {
        "benchmark": "MT-Sketch4-20K", "version": "1.0-review", "data_root": str(root),
        "num_images": 20000, "task_image_counts": dict.fromkeys(TASKS, 5000), "task_class_counts": TASKS.copy(),
        "order": STREAM, "image_bytes_hashed": hash_images, "frozen_artifacts": FROZEN.copy(),
        "ordered_index_sha256": canonical_hash(order),
        "ordered_content_sha256": canonical_hash([[i, records[i]["file_sha256"]] for i in order]),
        "task_known": True, "primary_metric": "equal_mean_of_four_task_top1_percentages",
        "native_label_conflict_images": 106, "visual_review_needed_images": 8,
        "flagged_images_retained": True,
    }
    return PreparedSketch(root, benchmark, config, inventory, records, classes, order, prompt_names)


def load_image(prepared, index):
    """Match the frozen Benchmark.image_stream decode, including PNG alpha."""
    from PIL import Image, ImageOps
    with Image.open(prepared.root / prepared.records[index]["image_path"]) as original:
        image = ImageOps.exif_transpose(original)
        if "A" in image.getbands() or image.mode == "P" and "transparency" in image.info:
            image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
        return image.convert("RGB").copy()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    files = list((root / "src/clip_officehome").glob("*.py"))
    files += [p for p in (root / "CLIP/clip").iterdir() if p.suffix in {".py", ".gz"}]
    files += [root / "src/clip_officehome/configs/officehome.json"]
    files += [p for p in (root / "scripts").glob("*mtsketch*.py")]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(files)}
