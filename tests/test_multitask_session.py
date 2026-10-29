"""CPU tests of task switching, persistent AdamW and audited update timing."""
from copy import deepcopy
import gc
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from clip_officehome import moe_prompt_tta as core
from clip_officehome.multitask_session import MultiTaskPromptTTASession
from clip_officehome.visual_moe_prompt_tta import VisualPromptExpertBank


class FakeModel(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def encode_image(self, images):
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack((torch.ones_like(mean), mean * 0.1), dim=1)


class FakeScaffold(nn.Module):
    def __init__(self, model, class_names, prompt_init, num_context, device):
        super().__init__()
        self.register_buffer("base_context", torch.ones(num_context, 3, device=device))
        self.register_buffer("classes", torch.linspace(-1, 1, len(class_names), device=device))

    def encode_text_features(self, model, context):
        features = torch.stack((self.classes + context.mean() * 0.1,
                                torch.ones_like(self.classes)), dim=1)
        return features / features.norm(dim=1, keepdim=True)


class FakeVisualScaffold(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_positional_embeddings = nn.Parameter(torch.ones(2, 3))
        self.prompt_scale = nn.Parameter(torch.tensor(0.25))


def fake_teacher_features(model, class_names, template, device):
    features = torch.zeros(len(class_names), 2, device=device)
    features[:, 1] = 1
    features[1] = torch.tensor([1.0, 0.0], device=device)
    return features


def fake_core_init(self, **kwargs):
    self.model = kwargs["model"]
    self.teacher_model = kwargs["teacher_model"]
    self.device = self.teacher_device = "cpu"
    self.use_cuda = self.teacher_use_cuda = False
    self.teacher_enabled = self.visual_enabled = True
    self.visual_router_source = "current_text_moe"
    self.class_names = tuple(kwargs["reference_domain_spec"].class_names)
    self.prompt_class_names = tuple(kwargs["reference_domain_spec"].prompt_class_names)
    self.scaffold = FakeScaffold(self.model, self.prompt_class_names, "a photo of a", 4, "cpu")
    self.teacher_text_features = fake_teacher_features(self.teacher_model, self.prompt_class_names, "", "cpu")
    self.shared_bank = core.PromptExpertBank(self.scaffold.base_context, 3, 0.02, 0, "cpu")
    self.private_bank = core.PromptExpertBank(self.scaffold.base_context, 8, 0.02, 10000, "cpu")
    self.visual_scaffold = FakeVisualScaffold()
    self.visual_shared_bank = VisualPromptExpertBank(2, 2, 3, 0.02, 20000, "cpu")
    self.visual_private_bank = VisualPromptExpertBank(4, 2, 3, 0.02, 30000, "cpu")


class FakeSession(MultiTaskPromptTTASession):
    def _prepare_student_forward(self, images, routed_private_experts, **kwargs):
        expert = ord(self.active_task_id) - ord("a")
        route = core.RoutedPrivateVisualExperts([expert], torch.ones(1), [0.5] * 4)
        logits = self._recompute_student_logits(
            base_tokens=images.detach(), routed_private_experts=routed_private_experts,
            routed_visual_experts=route)
        self.logit_weakrefs.append(weakref.ref(logits))
        return logits, images.detach(), route

    def _recompute_student_logits(self, base_tokens, routed_private_experts, routed_visual_experts, **kwargs):
        contexts = core.build_active_context(self.shared_bank, self.private_bank,
            routed_private_experts.selected_ids, routed_private_experts.selected_weights, 0.5)
        signal = contexts.mean() + self.visual_scaffold.prompt_scale
        signal = signal + self.visual_scaffold.prompt_positional_embeddings.mean()
        signal = signal + torch.stack(self.visual_shared_bank.get_prompts()).mean()
        signal = signal + self.visual_private_bank.get_prompt(routed_visual_experts.selected_ids[0]).mean()
        count = len(self.class_names)
        high = float(base_tokens.mean().item()) > 0.5
        base = torch.full((count,), -1.0 if high else 0.0)
        base[0] = 2.4 if high else 0.0
        direction = torch.ones(count)
        direction[0] = -1.0
        return (base + signal * direction * (0.01 if high else 0.0)).unsqueeze(0)


class MultiTaskSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        for target, replacement in [("__init__", fake_core_init)]:
            patcher = patch.object(core.PersistentMoEPromptTTASession, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in [("PromptScaffold", FakeScaffold), ("build_text_features", fake_teacher_features)]:
            patcher = patch.object(core, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def session(self, route=True):
        names = ("first", "second")
        session = FakeSession(model=FakeModel(), preprocess=lambda x: x,
            reference_domain_spec=SimpleNamespace(class_names=("a0", "a1"), prompt_class_names=names),
            num_workers=0, device="cpu", num_context=4, prompt_init="a photo of a",
            num_shared_experts=3, num_private_experts=8, expert_init_std=0.02, seed=0,
            num_visual_prompts=2, visual_prompt_init_std=0.0, visual_prompt_scale_init=0.25,
            num_visual_shared_experts=2, num_visual_private_experts=4, visual_router_source="current_text_moe",
            teacher_model=FakeModel(), teacher_preprocess=lambda x: x, teacher_device="cpu",
            teacher_template="a photo of a {class}", teacher_temperature=1.0)
        session.logit_weakrefs = []
        for task, count in zip("abcd", (2, 3, 4, 5)):
            session.register_task(task, [f"{task}{i}" for i in range(count)],
                                  names if task == "a" else [f"class {i}" for i in range(count)])
        if route:
            for task in "abcd":
                session.route_task(task, torch.ones(64, 3, 2, 2))
        return session

    def force_distinct_routes(self, session):
        # Test-only routes isolate inactive expert parameters and AdamW moments.
        for offset, task in enumerate("abcd"):
            session.task_states[task].route = core.RoutedPrivateExperts(
                [offset * 2, offset * 2 + 1], torch.tensor([0.5, 0.5]), [0.5] * 8)

    def test_routes_match_original_confidence_and_never_change_prompts(self):
        session = self.session(route=False)
        before = session.prompt_state_sha256()
        preview = torch.ones(64, 3, 2, 2)
        for task in "abcd":
            state = session.task_states[task]
            with torch.no_grad():
                features = session.model.encode_image(preview)
                features = features / features.norm(dim=-1, keepdim=True)
                ids, weights, scores = core.route_private_experts_for_features(session.model, state.scaffold,
                    session.shared_bank, session.private_bank, features, 2, "confidence", 1.0, 0.5)
            route = session.route_task(task, preview)
            self.assertEqual(route["selected_ids"], ids)
            self.assertEqual(route["selected_weights"], weights.tolist())
            self.assertEqual(route["routing_scores"], scores)
            self.assertEqual(route["before_prompt_hash"], before)
            self.assertEqual(route["after_prompt_hash"], before)
            self.assertFalse(state.route.selected_weights.requires_grad)
            self.assertFalse(state.teacher_text_features.requires_grad)
        self.assertTrue(all(row["empty_state"] for row in session.optimizer_audit()["optimizers"]))
        self.assertEqual(session.scored, 0)

    def test_task_switch_keeps_one_bank_optimizer_and_only_active_moments_advance(self):
        session = self.session()
        self.force_distinct_routes(session)
        initial = session.optimizer_audit()
        bank_ids = [id(getattr(session, name)) for name in
                    ("shared_bank", "private_bank", "visual_scaffold", "visual_shared_bank", "visual_private_bank")]
        image = torch.ones(1, 3, 2, 2)
        session.predict_and_adapt("a", image, image)
        private_a = session.private_bank.get_context(0)
        value_a = private_a.detach().clone()
        state_a = deepcopy(session.main_optimizer.state[private_a])
        row = session.predict_and_adapt("b", image, image)
        self.assertEqual((row["updates_before"], row["updates_after"]), (3, 6))
        self.assertEqual(row["num_classes"], 3)
        self.assertTrue(torch.equal(private_a, value_a))
        for key in state_a:
            self.assertTrue(torch.equal(state_a[key], session.main_optimizer.state[private_a][key]))
        session.predict_and_adapt("a", image, image)
        self.assertEqual(session.optimizer_rounds, 9)
        self.assertEqual(int(session.main_optimizer.state[private_a]["step"]), 6)
        self.assertEqual(int(session.main_optimizer.state[session.shared_bank.get_context(0)]["step"]), 9)
        self.assertEqual(int(session.visual_private_optimizers[0].state[session.visual_private_bank.get_prompt(0)]["step"]), 6)
        self.assertEqual(int(session.visual_private_optimizers[1].state[session.visual_private_bank.get_prompt(1)]["step"]), 3)
        self.assertEqual(bank_ids, [id(getattr(session, name)) for name in
            ("shared_bank", "private_bank", "visual_scaffold", "visual_shared_bank", "visual_private_bank")])
        final = session.optimizer_audit()
        self.assertEqual([o["instance_id"] for o in initial["optimizers"]], [o["instance_id"] for o in final["optimizers"]])
        self.assertEqual(session.task_scored, {"a": 2, "b": 1, "c": 0, "d": 0})
        self.assertTrue(all(p.grad is None for o in [session.main_optimizer, *session.visual_private_optimizers.values()]
                            for group in o.param_groups for p in group["params"]))
        json.dumps(final, allow_nan=False)

    def test_preupdate_callback_precedes_every_backward_and_step(self):
        session = self.session()
        events = []
        parameter = session.shared_bank.get_context(0)
        hook = parameter.register_hook(lambda gradient: events.append("backward"))
        self.addCleanup(hook.remove)
        original_step = session.main_optimizer.step
        def step(*args, **kwargs):
            events.append("step")
            return original_step(*args, **kwargs)
        session.main_optimizer.step = step
        before = session.prompt_state_sha256()
        callback_copy = []
        def callback(row):
            events.append("callback")
            self.assertEqual(session.prompt_state_sha256(), before)
            self.assertEqual(session.optimizer_rounds, 0)
            self.assertFalse(session.main_optimizer.state)
            self.assertNotIn("updates_after", row)
            self.assertEqual(row["prediction"], 0)
            self.assertEqual(row["teacher_pseudolabel"], 1)
            callback_copy.append(deepcopy(row))
            row["student_logits"][0] = 999.0  # Cannot alter actual model output.
        image = torch.ones(1, 3, 2, 2)
        result = session.predict_and_adapt("a", image, image, callback)
        self.assertEqual(events, ["callback", "backward", "step", "backward", "step", "backward", "step"])
        self.assertEqual(result["student_logits"], callback_copy[0]["student_logits"])
        self.assertEqual(result["predicted_class_index"], 0)
        self.assertEqual(result["updates_after"], 3)
        json.dumps(result, allow_nan=False)

    def test_skips_switches_and_updates_do_not_retain_graphs(self):
        session = self.session()
        zero, one = torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2)
        initial = session.prompt_state_sha256()
        results = []
        for task, image in [("a", zero), ("b", zero), ("c", one), ("d", zero), ("a", one)]:
            results.append(session.predict_and_adapt(task, image, one))
            gc.collect()
            self.assertTrue(all(ref() is None for ref in session.logit_weakrefs))
        self.assertEqual([r["selected"] for r in results], [False, False, True, False, True])
        self.assertEqual([r["updates_before"] for r in results], [0, 0, 0, 3, 3])
        self.assertEqual([r["updates_after"] for r in results], [0, 0, 3, 3, 6])
        self.assertEqual([len(r["student_logits"]) for r in results], [2, 3, 4, 5, 2])
        self.assertNotEqual(session.prompt_state_sha256(), initial)

    def test_preview_registration_and_callback_failure_guards(self):
        session = self.session(route=False)
        image = torch.ones(1, 3, 2, 2)
        with self.assertRaises(ValueError): session.predict_and_adapt("a", image, image)
        with self.assertRaises(ValueError): session.route_task("a", torch.ones(63, 3, 2, 2))
        for task in "abcd": session.route_task(task, torch.ones(64, 3, 2, 2))
        with self.assertRaises(ValueError): session.route_task("a", torch.ones(64, 3, 2, 2))
        before = session.prompt_state_sha256()
        def reject(row): raise RuntimeError("External audit rejected prediction")
        with self.assertRaises(RuntimeError): session.predict_and_adapt("a", image, image, reject)
        self.assertEqual(session.prompt_state_sha256(), before)
        self.assertEqual(session.optimizer_rounds, 0)
        self.assertFalse(session.main_optimizer.state)
        with self.assertRaises(ValueError): session.register_task("e", ["x", "y"], ["x", "y"])
        with self.assertRaises(ValueError): session.route_task("b", torch.ones(64, 3, 2, 2))
        with torch.no_grad(), self.assertRaises(ValueError): session.predict_and_adapt("a", image, image)

    def test_optimizer_replacement_is_detected(self):
        session = self.session()
        session.main_optimizer = torch.optim.AdamW(session.shared_bank.parameters(), lr=7.5e-6)
        with self.assertRaisesRegex(ValueError, "replaced"):
            session.optimizer_audit()


if __name__ == "__main__":
    unittest.main()
