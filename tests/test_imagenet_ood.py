"""ImageNet adapter regressions with tiny images and no model/GPU imports."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from clip_officehome import evaluation
from clip_officehome import imagenet_ood as ood
from clip_officehome.protocol import file_sha256, load_config


def write_images(root, folder, count):
    directory = root / str(folder)
    directory.mkdir(parents=True, exist_ok=True)
    # Reverse creation order to ensure filesystem order is not a label source.
    for index in reversed(range(count)):
        Image.new("RGB", (3, 2), (index, 17, 31)).save(directory / f"image_{index:02d}.png")


class ImageNetMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = ood.class_metadata()

    def test_v2_numeric_directories_keep_canonical_identity(self):
        canonical_ids = [0, 1, 2, 10]
        entries = [self.metadata["classes"][i] for i in canonical_ids]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in [10, 2, 1, 0]:
                write_images(root, index, 10)
            resolved, examples, records = ood._scan(root, "imagenet_v2", entries, 40, True)
            self.assertEqual(resolved, root.resolve())
            self.assertEqual(len(examples), 40)
            self.assertEqual(Counter(r["imagenet_index"] for r in records), dict.fromkeys(canonical_ids, 10))
            expected_local = {str(index): local for local, index in enumerate(canonical_ids)}
            for example, record in zip(examples, records, strict=True):
                folder = record["relative_path"].split("/")[0]
                self.assertEqual(record["imagenet_index"], int(folder))
                self.assertEqual(record["label_index"], expected_local[folder])
                self.assertEqual(example.label_index, expected_local[folder])
                self.assertEqual(example.class_name, entries[expected_local[folder]]["wnid"])
                self.assertEqual(record["sha256"], file_sha256(root / record["relative_path"]))
            # Folder '10' sorts before '2' lexically, but its fixture label is 3.
            self.assertEqual({r["label_index"] for r in records if r["imagenet_index"] == 10}, {3})
            self.assertEqual({r["label_index"] for r in records if r["imagenet_index"] == 2}, {2})
            self.assertEqual(records, sorted(records, key=lambda r: (r["class_name"], r["wnid"], r["relative_path"])))

    def test_duplicate_prompt_names_do_not_merge_wnid_classes(self):
        duplicate_indices = self.metadata["duplicate_prompt_names"]["missile"]
        entries = [self.metadata["classes"][i] for i in duplicate_indices]
        self.assertEqual(len(entries), 2)
        self.assertEqual(len({entry["class_name"] for entry in entries}), 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for entry in reversed(entries):
                write_images(root, entry["wnid"], 1)
            _, examples, records = ood._scan(root, "imagenet_sketch", entries, 2, False)
            self.assertEqual({r["label_index"] for r in records}, {0, 1})
            self.assertEqual({e.class_name for e in examples}, {entry["wnid"] for entry in entries})
            self.assertEqual({r["class_name"] for r in records}, {"missile"})
            self.assertTrue(all("sha256" not in row for row in records))

    def test_a_subset_labels_are_compact_not_canonical_imagenet_indices(self):
        mask = self.metadata["imagenet_a_indices"]
        entries = [self.metadata["classes"][mask[position]] for position in [0, 99, 199]]
        self.assertTrue(all(entry["imagenet_index"] > 2 for entry in entries))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for entry in reversed(entries):
                write_images(root, entry["wnid"], 1)
            _, examples, records = ood._scan(root, "imagenet_a", entries, 3, False)
            expected = {entry["wnid"]: (local, entry["imagenet_index"]) for local, entry in enumerate(entries)}
            self.assertEqual({record["label_index"] for record in records}, {0, 1, 2})
            for example, record in zip(examples, records, strict=True):
                local, canonical = expected[record["wnid"]]
                self.assertEqual((record["label_index"], record["imagenet_index"]), (local, canonical))
                self.assertEqual(example.label_index, local)

    def test_wrong_missing_extra_or_noncanonical_class_directories_rejected(self):
        entries = [self.metadata["classes"][i] for i in [0, 1]]
        for alteration in ["missing", "wrong", "extra", "numeric_alias"]:
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_images(root, "0", 10)
                if alteration != "missing":
                    write_images(root, "1", 10)
                if alteration == "wrong":
                    (root / "1").rename(root / "wrong_class")
                elif alteration == "extra":
                    (root / "extra_class").mkdir()
                elif alteration == "numeric_alias":
                    (root / "0").rename(root / "00")
                with self.assertRaisesRegex(ValueError, "Class directories differ"):
                    ood._scan(root, "imagenet_v2", entries, 20, False)

    def test_empty_class_with_only_nonimage_files_is_rejected(self):
        entry = self.metadata["classes"][6]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / entry["wnid"]
            directory.mkdir()
            (directory / "README.txt").write_text("This is not an image.")
            with self.assertRaisesRegex(ValueError, "Empty class"):
                ood._scan(root, "imagenet_a", [entry], 1, False)

    def test_v2_per_class_counts_are_checked_even_when_total_matches(self):
        entries = [self.metadata["classes"][i] for i in [0, 1]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_images(root, "0", 9)
            write_images(root, "1", 11)
            with self.assertRaisesRegex(ValueError, "exactly 10 images"):
                ood._scan(root, "imagenet_v2", entries, 20, False)

    def test_public_a_adapter_rejects_truncated_data_with_all_classes_present(self):
        entries = [self.metadata["classes"][i] for i in self.metadata["imagenet_a_indices"]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for entry in entries:
                write_images(root, entry["wnid"], 1)
            with self.assertRaisesRegex(ValueError, "Expected the full 7500 images, found 200"):
                ood.prepare_imagenet_data("imagenet_a", root, hash_images=False)


class ImageNetConfigurationTests(unittest.TestCase):
    def test_every_benchmark_keeps_independent_student_preupdate_protocol(self):
        metadata = ood.class_metadata()
        for benchmark, expected_count in ood.COUNTS.items():
            with self.subTest(benchmark=benchmark):
                config = ood.benchmark_config(benchmark)
                rules, data = config["evaluation"], config["data"]
                self.assertEqual(rules["seed"], 0)
                self.assertEqual(rules["batch_size"], 1)
                self.assertTrue(rules["full_data"])
                self.assertFalse(rules["shuffle"])
                self.assertTrue(rules["score_before_update"])
                self.assertEqual(rules["prediction_source"], "student_only")
                self.assertFalse(rules["teacher_output_fusion"])
                self.assertFalse(rules["teacher_output_calibration"])
                self.assertTrue(rules["persistent_prompts"])
                self.assertTrue(rules["independent_benchmark_initialization"])
                self.assertFalse(rules["benchmark_state_transfer"])
                self.assertEqual(data["shadow_text_domains"], [])
                self.assertEqual(data["branch_domain_schedule"], [["primary", benchmark]])
                self.assertEqual(data["reported_branch"], {benchmark: "primary"})
                self.assertEqual(data["domain_order"], [benchmark])
                self.assertEqual(data["domain_counts"], {benchmark: expected_count})
                self.assertEqual(data["num_scored_images"], expected_count)
                self.assertEqual(data["formal_branch_image_visits_excluding_prerouting"], expected_count)
                self.assertEqual(data["preview_image_visits"], 64)
                self.assertEqual(data["domain_learning_rate_overrides"], {})
                self.assertEqual(data["learning_rate"], 7.5e-6)
                self.assertEqual(data["student_confidence_threshold"], 0.815)
                mask = metadata[benchmark + "_indices"]
                self.assertEqual(data["imagenet_indices"], mask)
                self.assertEqual(data["class_names"], [metadata["classes"][i]["wnid"] for i in mask])
                self.assertEqual(data["prompt_class_names"], [metadata["classes"][i]["class_name"] for i in mask])
                self.assertEqual(data["num_classes"], ood.CLASS_COUNTS[benchmark])
                self.assertNotIn("historical_reference", config)
                spec = SimpleNamespace(target_domain=benchmark)
                kwargs = evaluation.session_kwargs(config, None, None, None, None, spec, "primary")
                self.assertEqual(kwargs["seed"], 0)
                self.assertEqual(kwargs["num_visual_prompts"], 2)
                self.assertEqual(evaluation.evaluation_kwargs(config, spec)["tta_lr"], 7.5e-6)

    def test_configs_do_not_share_mutable_state_across_benchmark_sessions(self):
        first = ood.benchmark_config("imagenet_a")
        independent = ood.benchmark_config("imagenet_v2")
        original = deepcopy(independent)
        first["baseline_method"]["text_prompt"]["num_context"] = 999
        first["data"]["class_names"][0] = "changed"
        first["evaluation"]["seed"] = 42
        self.assertEqual(independent, original)
        fresh = ood.benchmark_config("imagenet_a")
        self.assertEqual(fresh["evaluation"]["seed"], 0)
        self.assertEqual(fresh["baseline_method"]["text_prompt"]["num_context"], 4)
        self.assertNotEqual(fresh["data"]["class_names"][0], "changed")
        self.assertEqual(load_config("officehome")["baseline_method"]["text_prompt"]["num_context"], 4)


class ImageNetSummaryTests(unittest.TestCase):
    def test_single_benchmark_macro_is_not_divided_by_four(self):
        config = ood.benchmark_config("imagenet_a")
        benchmark, nclasses = config["benchmark"], config["data"]["num_classes"]
        # A small, internally consistent trace exercises the real recount and
        # summarizer. This is not passed off as a full benchmark evaluation.
        config["data"].update(num_scored_images=3, domain_counts={benchmark: 3},
                              formal_branch_image_visits_excluding_prerouting=3,
                              preview_image_visits=3)
        records, trace = [], []
        for index, label in enumerate([0, 0, 1]):
            record = {"relative_path": f"fixture/image_{index}.png", "label_index": label,
                      "sha256": hashlib.sha256(str(index).encode()).hexdigest()}
            records.append(record)
            trace.append({"branch": "primary", "domain": benchmark, "index": index,
                          "relative_path": record["relative_path"], "image_sha256": record["sha256"],
                          "label": label, "student_logits": [0.] * nclasses,
                          "student_probabilities": [1. / nclasses] * nclasses,
                          "prediction": 0, "confidence": 1. / nclasses, "selected": False,
                          "optimizer_rounds_before_prediction": 0,
                          "prediction_source": "student_only", "scoring": "pre_update"})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "primary_imagenet_a.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in trace))
            state = {"branch": "primary", "domain": benchmark, "is_reported": True,
                     "trace_file": path.name, "trace_sha256": file_sha256(path),
                     "num_scored": 3, "correct": 2, "selected": 0, "update_rounds": 0,
                     "route_audit": {"preview_count": 3}}
            metadata = {"domain_states": [state], "optimizer_creations": [], "elapsed_seconds": 0.1}
            summary = evaluation.summarize(output, config, metadata, {benchmark: records})
        self.assertAlmostEqual(summary["macro_top1_percent"], 200. / 3)
        self.assertAlmostEqual(summary["micro_top1_percent"], 200. / 3)
        self.assertEqual(summary["num_scored_images"], 3)
        self.assertEqual(summary["prediction_rule"], "pre_update_student_only")
        self.assertFalse(summary["teacher_output_fusion"])


if __name__ == "__main__":
    unittest.main()
