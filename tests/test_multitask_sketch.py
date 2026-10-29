import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clip_officehome.multitask_sketch import TASKS, fixed_config, load_image, validate_structure


def fixture():
    records, classes = [], {}
    for task, count in TASKS.items():
        classes[task] = [{"class_index": j, "class_name": f"class{j:04d}",
                         "class_key": str(j), "native_class_id": 1000 + j} for j in range(count)]
        for j in range(5000):
            label = j * count // 5000
            name = f"class{label:04d}"
            records.append({"manifest_index": len(records), "task_id": task, "class_index": label,
                "class_id": 1000 + label, "class_name": name, "relative_path": f"{name}/{j:04d}.png",
                "image_path": f"images/{task}/{name}/{j:04d}.png", "sample_id": f"{task}:{j}"})
    order = [task * 5000 + j for j in range(5000) for task in range(4)]
    return records, classes, order


class MultiTaskDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture()

    def test_native_indices_and_full_round_robin(self):
        validate_structure(*self.base)

    def test_rejects_native_id_used_as_prediction_index(self):
        rows = copy.deepcopy(self.base[0])
        rows[0]["class_index"] = rows[0]["class_id"]
        with self.assertRaises(ValueError):
            validate_structure(rows, self.base[1], self.base[2])

    def test_rejects_blocks_and_duplicate_samples(self):
        for order in (list(range(20000)), self.base[2][:-1] + [self.base[2][0]]):
            with self.assertRaises(ValueError):
                validate_structure(self.base[0], self.base[1], order)

    def test_rejects_within_task_reordering(self):
        order = self.base[2].copy()
        order[0], order[4] = order[4], order[0]
        with self.assertRaises(ValueError):
            validate_structure(self.base[0], self.base[1], order)

    def test_rejects_outside_image_path(self):
        rows = copy.deepcopy(self.base[0])
        rows[0]["image_path"] = "images/domainnet_sketch/../../outside.png"
        with self.assertRaises(ValueError):
            validate_structure(rows, self.base[1], self.base[2])

    def test_alpha_composites_on_white(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGBA", (2, 1))
            image.putdata([(255, 0, 0, 0), (0, 0, 0, 255)])
            image.save(root / "alpha.png")
            prepared = SimpleNamespace(root=root, records=[{"image_path": "alpha.png"}])
            decoded = load_image(prepared, 0)
            self.assertEqual(decoded.mode, "RGB")
            self.assertEqual([decoded.getpixel((0, 0)), decoded.getpixel((1, 0))],
                             [(255, 255, 255), (0, 0, 0)])

    def test_fixed_continual_protocol(self):
        classes = self.base[1]
        prompts = {t: [c["class_name"] for c in v] for t, v in classes.items()}
        config = fixed_config(classes, prompts)
        self.assertEqual(config["data"]["num_scored_images"], 20000)
        self.assertEqual(config["data"]["preview_image_visits"], 256)
        self.assertEqual(config["evaluation"]["seed"], 0)
        self.assertFalse(config["evaluation"]["reset_optimizer_on_task_switch"])
        self.assertFalse(config["evaluation"]["reset_prompt_on_task_switch"])
        self.assertFalse(config["evaluation"]["teacher_output_fusion"])
        self.assertEqual(config["data"]["shadow_text_domains"], [])
        self.assertEqual(config["data"]["domain_learning_rate_overrides"], {})


if __name__ == "__main__":
    unittest.main()
