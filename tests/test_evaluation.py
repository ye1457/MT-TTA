"""Protocol and routing tests; no pretrained weights, GPU or image downloads."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from clip_officehome import evaluation
from clip_officehome.protocol import (load_config, prepare_data, require_new_output,
                                     scan_domain, validate_config)
from clip_officehome.routing import score_probabilities


class ProtocolTests(unittest.TestCase):
    def test_selected_configs_and_parameter_mapping(self):
        pacs, officehome = load_config("pacs"), load_config("officehome")
        self.assertEqual(pacs["data"]["num_scored_images"], 9991)
        self.assertEqual(officehome["data"]["num_scored_images"], 15588)
        self.assertEqual(pacs["baseline_method"]["text_prompt"]["router_score"], "mean_probability_margin")
        self.assertEqual(officehome["baseline_method"]["text_prompt"]["router_score"], "mean_max_class_probability_across_preview_images")
        spec = SimpleNamespace(target_domain="Clipart")
        kwargs = evaluation.evaluation_kwargs(officehome, spec)
        self.assertEqual(kwargs["tta_lr"], 2e-6)
        self.assertEqual(kwargs["confidence_threshold"], 0.815)
        self.assertEqual(kwargs["routing_score"], "confidence")
        self.assertEqual(kwargs["teacher_ce_weight"], 0.08)
        self.assertEqual(kwargs["update_steps"], 3)
        shadow = evaluation.session_kwargs(officehome, None, None, None, None, spec, "text_shadow")
        self.assertEqual(shadow["num_visual_prompts"], 0)
        self.assertEqual(shadow["seed"], 0)
        primary = evaluation.session_kwargs(pacs, None, None, None, None, spec, "primary")
        self.assertEqual(primary["num_visual_prompts"], 2)
        self.assertEqual(primary["visual_prompt_scale_init"], 0.25)
        self.assertEqual(primary["num_workers"], 0)

    def test_forbidden_evaluation_changes(self):
        for key, value in [("teacher_output_fusion", True), ("seed", 1), ("persistent_prompts", False),
                           ("score_before_update", False), ("full_data", False), ("shuffle", True)]:
            with self.subTest(key=key):
                cfg = deepcopy(load_config("pacs"))
                cfg["evaluation"][key] = value
                with self.assertRaises(ValueError): validate_config(cfg)

    def test_truncation_cannot_be_authorized_by_changing_counts(self):
        cfg = deepcopy(load_config("pacs"))
        cfg["data"]["domain_counts"]["photo"] = 100
        cfg["data"]["num_scored_images"] = 8421
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_invalid_method_numbers_and_silent_topk_clipping_rejected(self):
        invalid = [("text_prompt", "preview_size", 0), ("text_prompt", "private_topk", 9),
                   ("text_prompt", "shared_mix_weight", 1.1), ("text_prompt", "router_temperature", 0),
                   ("text_prompt", "expert_init_std", float("nan")),
                   ("visual_prompt", "private_topk", 5), ("visual_prompt", "shared_mix_weight", -0.1),
                   ("visual_prompt", "scale_init", float("inf")),
                   ("adaptation", "teacher_ce_weight", float("inf")),
                   ("adaptation", "update_rounds_per_selected_image", True)]
        for section, key, value in invalid:
            with self.subTest(section=section, key=key):
                cfg = deepcopy(load_config("pacs"))
                cfg["baseline_method"][section][key] = value
                with self.assertRaises(ValueError): validate_config(cfg)

    def test_missing_full_data_rejected_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = load_config("pacs")
            for domain in cfg["data"]["domain_order"]:
                for classname in cfg["data"]["class_names"]:
                    folder = root / domain / classname
                    folder.mkdir(parents=True)
                    (folder / "one.jpg").write_bytes(b"not decoded by metadata-only dry-run")
            with self.assertRaisesRegex(ValueError, "expected all 2048 images"):
                evaluation.main(["--benchmark", "pacs", "--data-root", str(root), "--dry-run"])
            self.assertFalse((root / "summary.json").exists())

    def test_tuned_parameters_require_matching_metadata_descriptions(self):
        cfg = deepcopy(load_config("pacs"))
        adaptation = cfg["baseline_method"]["adaptation"]
        adaptation["teacher_ce_weight"] = 0.2
        with self.assertRaisesRegex(ValueError, "Adaptation descriptions"):
            validate_config(cfg)
        adaptation["loss"] = "student_hard_pseudolabel_CE_plus_0.2_teacher_hard_pseudolabel_CE"
        text = cfg["baseline_method"]["text_prompt"]
        text["preview_size"] = 32
        cfg["data"]["preview_image_visits"] = 128
        with self.assertRaisesRegex(ValueError, "Preview descriptions"):
            validate_config(cfg)
        text["preview_members"] = "first_min_32_images_of_this_domain_sorted_stream"
        validate_config(cfg)
        kwargs = evaluation.evaluation_kwargs(cfg, SimpleNamespace(target_domain="photo"))
        self.assertEqual(kwargs["teacher_ce_weight"], 0.2)
        self.assertEqual(kwargs["routing_buffer_size"], 32)

    def test_fixed_semantics_cannot_be_misrepresented_in_metadata(self):
        invalid = [("text_prompt", "preview_image_features", "teacher_image_features"),
                   ("adaptation", "sample_gate", "teacher_confidence"),
                   ("adaptation", "targets_gate_routes_fixed_during_image_updates", False)]
        for section, key, value in invalid:
            with self.subTest(key=key):
                cfg = deepcopy(load_config("pacs"))
                cfg["baseline_method"][section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(cfg)

    def test_sorted_full_membership_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ["zebra/z.jpg", "ant/c.jpg", "ant/a.jpg", "zebra/b.jpg"]:
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(relative.encode())
            _, rows = scan_domain(root, ["ant", "zebra"], 4, hash_images=True)
            self.assertEqual([r["relative_path"] for r in rows], ["ant/a.jpg", "ant/c.jpg", "zebra/b.jpg", "zebra/z.jpg"])
            self.assertEqual([r["label_index"] for r in rows], [0, 0, 1, 1])
            self.assertTrue(all(len(r["sha256"]) == 64 for r in rows))
            with self.assertRaises(ValueError): scan_domain(root, ["ant", "zebra"], 5)
            with self.assertRaises(ValueError): scan_domain(root, ["ant", "horse"], 4)

    def test_output_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                require_new_output(temporary)
            self.assertEqual(require_new_output(Path(temporary) / "new"), Path(temporary) / "new")

    def test_routing_fixture_uses_probability_margin_not_confidence(self):
        import torch
        probs = torch.tensor([[0.75, 0.20, 0.05], [0.50, 0.25, 0.25]], dtype=torch.float16)
        expected = ((probs[0, 0].float() - probs[0, 1].float()) + (probs[1, 0].float() - probs[1, 1].float())) / 2
        self.assertTrue(torch.equal(score_probabilities(probs, "margin"), expected))
        self.assertEqual(score_probabilities(probs, "margin").dtype, torch.float32)
        self.assertNotEqual(float(score_probabilities(probs, "margin")), float(score_probabilities(probs, "confidence")))
        with self.assertRaises(ValueError): score_probabilities(probs, "teacher_margin")

    def test_trace_recount_rejects_wrong_gate_and_teacher_output(self):
        record = {"relative_path": "dog/a.jpg", "sha256": "a" * 64, "label_index": 0}
        row = {"branch": "primary", "domain": "photo", "index": 0, "relative_path": record["relative_path"],
               "image_sha256": record["sha256"], "label": 0, "student_logits": [0.0, 0.0],
               "student_probabilities": [0.5, 0.5], "prediction": 0, "confidence": 0.5, "selected": True,
               "optimizer_rounds_before_prediction": 0, "prediction_source": "student_only", "scoring": "pre_update"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n")
            counts = evaluation.recount_trace(path, [record], "primary", "photo", 2, 0.5, 3)
            self.assertEqual(counts, {"num_images": 1, "correct": 1, "selected": 1, "update_rounds": 3})
            for key, value in [("selected", False), ("prediction_source", "teacher_student_fusion"), ("optimizer_rounds_before_prediction", 3)]:
                bad = dict(row, **{key: value})
                path.write_text(json.dumps(bad) + "\n")
                with self.assertRaises(ValueError):
                    evaluation.recount_trace(path, [record], "primary", "photo", 2, 0.5, 3)

    def test_four_domain_schedule_preserves_prompts_and_recreates_optimizers(self):
        import torch
        config = load_config("pacs")
        metadata = {"optimizer_creations": [], "domain_states": []}
        core = SimpleNamespace()
        def build_main(*args, **kwargs):
            return torch.optim.AdamW(session.shared_bank.parameters(), lr=kwargs["tta_lr"], weight_decay=0.0)
        def build_private(*args, **kwargs):
            return {i: torch.optim.AdamW([p], lr=kwargs["tta_lr"], weight_decay=0.0) for i, p in enumerate(session.visual_private_bank.parameters())}
        core.build_optimizer_for_active_experts, core.build_private_visual_optimizers = build_main, build_private
        observations = []
        class FakeSession:
            def __init__(self):
                self.shared_bank = torch.nn.Linear(1, 1, bias=False)
                self.shared_bank.weight.data.fill_(1)
                self.private_bank = self.visual_scaffold = self.visual_shared_bank = None
                self.visual_private_bank = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(4)])
                self.branch = "primary"
            def evaluate_domain(self, **kwargs):
                observations.append(float(self.shared_bank.weight.item()))
                main = core.build_optimizer_for_active_experts(tta_lr=kwargs["tta_lr"])
                core.build_private_visual_optimizers(tta_lr=kwargs["tta_lr"])
                self.scored = self.correct = self.selected = 1
                self.trace.write("{}\n")
                self.route_audit = {"preview_count": 1}
                for _ in range(3):
                    main.zero_grad()
                    self.shared_bank.weight.square().sum().backward()
                    main.step()
                result = SimpleNamespace(num_images=1, top1_accuracy=1.0, num_selected_samples=1,
                                         num_updates=3, routing_buffer_size=1, teacher_supervision_mode="aux_pseudolabel")
                result.to_dict = lambda: {"num_images": 1, "top1_accuracy": 1.0}
                return result
        session = FakeSession()
        metadata["initial_prompt_hashes"] = {"primary": evaluation.prompt_hash(session)}
        specs = {d: SimpleNamespace(target_domain=d) for d in config["data"]["domain_order"]}
        records = {d: [{}] for d in specs}
        with tempfile.TemporaryDirectory() as temporary, evaluation.OptimizerMonitor(core, config, metadata) as monitor:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.execute_schedule(config, specs, records, {"primary": session}, monitor, Path(temporary), metadata)
        self.assertEqual(len(metadata["optimizer_creations"]), 20)
        self.assertTrue(all(o["initial_state_entries"] == 0 for o in metadata["optimizer_creations"]))
        self.assertTrue(all(a > b for a, b in zip(observations, observations[1:])))
        self.assertTrue(all(a["final_prompt_hash"] == b["initial_prompt_hash"] for a, b in zip(metadata["domain_states"], metadata["domain_states"][1:])))
        self.assertIs(core.build_optimizer_for_active_experts, build_main)


if __name__ == "__main__":
    unittest.main()
