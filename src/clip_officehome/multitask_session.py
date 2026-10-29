"""Task-known online prompt adaptation with one persistent prompt/optimizer state.

Only class-specific, frozen scaffold buffers and teacher features are cached.
The public adaptation API accepts tensors and task IDs, never labels or paths.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from . import moe_prompt_tta as core
from .evaluation import prompt_hash
from .protocol import require
from .routing import routing_mode


METHOD_PROTOCOL = {
    "seed": 0, "num_tasks": 4, "batch_size": 1, "preview_size": 64,
    "confidence_threshold": 0.815, "learning_rate": 7.5e-6, "update_steps": 3,
    "teacher_ce_weight": 0.08, "text_shared_mix_weight": 0.5,
    "text_topk": 2, "text_router_temperature": 1.0,
    "visual_shared_mix_weight": 0.44, "visual_topk": 1,
    "visual_router_temperature": 1.0, "weight_decay": 0.0,
}


@dataclass
class TaskState:
    class_ids: tuple[str, ...]
    prompt_names: tuple[str, ...]
    scaffold: core.PromptScaffold
    teacher_text_features: torch.Tensor
    route: core.RoutedPrivateExperts | None = None
    route_audit: dict | None = None


def _route_record(route):
    return {"selected_ids": list(route.selected_ids),
            "selected_weights": route.selected_weights.detach().cpu().tolist(),
            "routing_scores": list(route.routing_scores)}


def validate_method_config(config):
    """Bind fixed session math to the runner's declared configuration."""
    if config is None:
        return
    evaluation, data, method = config["evaluation"], config["data"], config["baseline_method"]
    text, visual, adaptation = method["text_prompt"], method["visual_prompt"], method["adaptation"]
    require(evaluation["seed"] == 0 and evaluation["batch_size"] == 1 and
            evaluation["prediction_source"] == "student_only" and evaluation["score_before_update"] is True and
            evaluation["teacher_output_fusion"] is False and evaluation["teacher_output_calibration"] is False and
            evaluation["persistent_prompts"] is True and evaluation["reset_optimizer_on_task_switch"] is False and
            evaluation["optimizer_state_persistent"] is True, "Wrong multitask state/output protocol")
    require(data["student_confidence_threshold"] == 0.815 and data["learning_rate"] == 7.5e-6 and
            data["domain_learning_rate_overrides"] == {} and data["shadow_text_domains"] == [],
            "Wrong multitask threshold/learning rate/branch")
    require(method["student"] == "ViT-B/16" and method["teacher"] == "ViT-L/14" and method["logit_scale"] == 100.0,
            "Wrong student/teacher/logit scale")
    require(text["preview_size"] == 64 and text["private_topk"] == 2 and text["shared_mix_weight"] == 0.5 and
            text["router_temperature"] == 1.0 and text["preview_parameter_updates"] == 0 and
            text["preview_uses_ground_truth_or_teacher"] is False and
            routing_mode(text["router_score"]) == "confidence", "Wrong multitask text route")
    require(visual["private_topk"] == 1 and visual["shared_mix_weight"] == 0.44 and
            visual["router_temperature"] == 1.0 and visual["router_source"] == "current_text_moe",
            "Wrong multitask visual route")
    require(adaptation["student_pseudo_topk"] == 1 and adaptation["teacher_supervision"] == "aux_pseudolabel" and
            adaptation["teacher_ce_weight"] == 0.08 and adaptation["teacher_confidence_gate"] is False and
            adaptation["update_rounds_per_selected_image"] == 3 and adaptation["optimizer"] == "AdamW" and
            adaptation["weight_decay"] == 0.0 and adaptation["betas"] == [0.9, 0.999] and
            adaptation["eps"] == 1e-8 and adaptation["entropy_weight"] == adaptation["shared_anchor_weight"] == 0.0,
            "Wrong multitask adaptation loss/optimizer")


class MultiTaskPromptTTASession(core.PersistentMoEPromptTTASession):
    """Four task class spaces sharing all trainable prompts and AdamW moments.

    Construct with ``evaluation.session_kwargs(...)`` and optional ``config``.
    Register all tasks, route each from its 64 preprocessed preview tensors,
    then call ``predict_and_adapt`` once for every image in the declared stream.
    """

    def __init__(self, *, config=None, **kwargs):
        validate_method_config(config)
        required = {"seed": 0, "num_context": 4, "prompt_init": "a photo of a",
                    "num_shared_experts": 3, "num_private_experts": 8, "expert_init_std": 0.02,
                    "num_visual_prompts": 2, "num_visual_shared_experts": 2,
                    "num_visual_private_experts": 4, "visual_router_source": "current_text_moe",
                    "visual_prompt_init_std": 0.0, "visual_prompt_scale_init": 0.25}
        require(all(kwargs.get(key) == value for key, value in required.items()),
                "Session initialization differs from the fixed multitask method")
        require(kwargs.get("teacher_model") is not None and kwargs.get("teacher_preprocess") is not None and
                kwargs.get("teacher_temperature", 1.0) == 1.0 and
                kwargs.get("teacher_template", "a photo of a {class}") == "a photo of a {class}",
                "The fixed auxiliary teacher is required")
        require(not kwargs.get("disable_visual_domains") and not kwargs.get("disable_update_domains") and
                not kwargs.get("domain_confidence_threshold_overrides"), "Task-specific overrides are unsupported")
        self._prompt_init = kwargs["prompt_init"]
        self._num_context = kwargs["num_context"]
        self._teacher_template = kwargs.get("teacher_template", "a photo of a {class}")
        super().__init__(**kwargs)
        self.task_states: dict[str, TaskState] = {}
        self.task_scored: dict[str, int] = {}
        self.task_selected: dict[str, int] = {}
        self.active_task_id = None
        self.scored = self.selected = self.optimizer_rounds = 0
        self._started = self._routing_started = self._busy = False
        # Every text expert has a parameter slot. Inactive experts receive
        # grad=None, so AdamW neither changes their values nor advances moments.
        self.main_optimizer = core.build_optimizer_for_active_experts(
            shared_bank=self.shared_bank, private_bank=self.private_bank,
            selected_private_expert_ids=list(range(len(self.private_bank))),
            tta_lr=7.5e-6, weight_decay=0.0,
            visual_scaffold=self.visual_scaffold, visual_shared_bank=self.visual_shared_bank)
        self.visual_private_optimizers = core.build_private_visual_optimizers(
            private_bank=self.visual_private_bank, tta_lr=7.5e-6, weight_decay=0.0)
        optimizers = [("main", self.main_optimizer)] + [
            (f"visual_private_{i}", optimizer) for i, optimizer in self.visual_private_optimizers.items()]
        self.optimizer_initialization = []
        self._optimizer_instances = {kind: optimizer for kind, optimizer in optimizers}
        for kind, optimizer in optimizers:
            require(not optimizer.state, "Initial optimizer state is not empty")
            for group in optimizer.param_groups:
                require(group["lr"] == 7.5e-6 and group["weight_decay"] == 0.0 and
                        group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8, "Wrong AdamW defaults")
            self.optimizer_initialization.append({"kind": kind, "initial_state_entries": 0,
                "parameter_count": sum(len(g["params"]) for g in optimizer.param_groups),
                "lr": 7.5e-6, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8})
        self.initial_prompt_sha256 = self.prompt_state_sha256()

    def prompt_state_sha256(self):
        return prompt_hash(self)

    def optimizer_audit(self):
        """Expose stable creation/parameter identities and persistent AdamW steps."""
        current = {"main": self.main_optimizer, **{
            f"visual_private_{i}": optimizer for i, optimizer in self.visual_private_optimizers.items()}}
        require(current.keys() == self._optimizer_instances.keys() and
                all(current[key] is optimizer for key, optimizer in self._optimizer_instances.items()),
                "An optimizer was replaced after initialization")
        names = {id(parameter): f"{bank}.{name}"
                 for bank in ("shared_bank", "private_bank", "visual_scaffold", "visual_shared_bank", "visual_private_bank")
                 for name, parameter in getattr(self, bank).named_parameters()}
        rows = []
        for creation_index, (kind, optimizer) in enumerate(current.items()):
            parameters = []
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    require(id(parameter) in names, "Optimizer contains a foreign/replaced parameter")
                    state = optimizer.state.get(parameter, {})
                    step = state.get("step", 0)
                    if isinstance(step, torch.Tensor):
                        step = step.item()
                    parameters.append({"name": names[id(parameter)], "has_state": bool(state), "step": int(step)})
            rows.append({"kind": kind, "creation_index": creation_index, "instance_id": str(id(optimizer)),
                         "parameter_count": len(parameters), "state_parameter_count": len(optimizer.state),
                         "empty_state": not bool(optimizer.state), "parameters": parameters})
        return {"optimizer_count": len(rows), "optimizer_creations": len(self._optimizer_instances),
                "optimizer_state_persistent": True, "optimizers": rows,
                "global_update_rounds": self.optimizer_rounds, "scored": self.scored, "selected": self.selected}

    def register_task(self, task_id: str, class_ids: Sequence[str], prompt_names: Sequence[str]):
        require(not self._routing_started and not self._started and not self._busy,
                "All tasks must be registered before any routing or prediction")
        require(isinstance(task_id, str) and task_id and task_id not in self.task_states, "Invalid/repeated task ID")
        class_ids, prompt_names = tuple(class_ids), tuple(prompt_names)
        require(2 <= len(class_ids) == len(prompt_names) and len(set(class_ids)) == len(class_ids) and
                all(isinstance(x, str) and x for x in class_ids + prompt_names), "Invalid task class space")
        require(len(self.task_states) < 4, "The protocol contains exactly four tasks")
        # Display-name duplicates are deliberately allowed; class IDs are the
        # unique task-local identities, and their supplied order is preserved.
        if class_ids == self.class_names and prompt_names == self.prompt_class_names:
            scaffold, teacher_features = self.scaffold, self.teacher_text_features
        else:
            with torch.no_grad():
                scaffold = core.PromptScaffold(self.model, prompt_names, self._prompt_init,
                                                self._num_context, self.device).to(self.device)
                teacher_features = core.build_text_features(
                    model=self.teacher_model, class_names=prompt_names,
                    template=self._teacher_template, device=self.teacher_device)
        require(not list(scaffold.parameters()) and not teacher_features.requires_grad,
                "Class caches must not contain trainable parameters or autograd features")
        require(teacher_features.shape[0] == len(class_ids), "Teacher feature/class size mismatch")
        self.task_states[task_id] = TaskState(class_ids, prompt_names, scaffold, teacher_features.detach())
        self.task_scored[task_id] = self.task_selected[task_id] = 0

    def _activate_task(self, task_id):
        require(task_id in self.task_states, "Unknown task ID")
        task = self.task_states[task_id]
        self.class_names, self.prompt_class_names = task.class_ids, task.prompt_names
        self.scaffold, self.teacher_text_features = task.scaffold, task.teacher_text_features
        self.active_task_id = task_id
        return task

    @torch.no_grad()
    def route_task(self, task_id: str, preprocessed_images: torch.Tensor):
        require(not self._started and not self._busy and len(self.task_states) == 4,
                "Route all four tasks before any prediction/adaptation")
        require(isinstance(preprocessed_images, torch.Tensor) and preprocessed_images.ndim == 4 and
                preprocessed_images.shape[:2] == (64, 3) and preprocessed_images.is_floating_point(),
                "Each task requires exactly 64 preprocessed RGB images")
        task = self._activate_task(task_id)
        require(task.route is None, "Task route is already fixed")
        self._routing_started = True
        before = self.prompt_state_sha256()
        require(before == self.initial_prompt_sha256 and self.optimizer_rounds == self.scored == 0,
                "All previews must use the same initial prompt state")
        image_features = self.model.encode_image(preprocessed_images.to(self.device, non_blocking=self.use_cuda))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # routing.py explicitly delegates confidence routing to this original
        # core implementation. No alternate precision or score formula is used.
        ids, weights, scores = core.route_private_experts_for_features(
            model=self.model, scaffold=self.scaffold, shared_bank=self.shared_bank,
            private_bank=self.private_bank, image_features=image_features.detach(),
            topk_private_experts=2,
            routing_score=routing_mode("mean_max_class_probability_across_preview_images"),
            router_temperature=1.0, shared_mix_weight=0.5)
        task.route = core.RoutedPrivateExperts(ids, weights.detach(), scores)
        after = self.prompt_state_sha256()
        require(after == before and not any(o.state for o in
                [self.main_optimizer, *self.visual_private_optimizers.values()]), "Preview changed prompts/optimizers")
        task.route_audit = {"task_id": task_id, "num_classes": len(task.class_ids),
            "preview_count": 64, "before_prompt_hash": before, "after_prompt_hash": after,
            "score_mode": "confidence", "routing_score_dtype": str(self.model.dtype),
            **_route_record(task.route)}
        return deepcopy(task.route_audit)

    def _zero_all_grad(self):
        self.main_optimizer.zero_grad(set_to_none=True)
        for optimizer in self.visual_private_optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def predict_and_adapt(self, task_id: str, student_tensor: torch.Tensor, teacher_tensor: torch.Tensor,
                          before_update_callback: Callable[[dict], None] | None = None):
        require(not self._busy, "The session is not reentrant")
        require(len(self.task_states) == 4 and all(t.route is not None for t in self.task_states.values()),
                "All four task previews must finish before the first prediction")
        for tensor in (student_tensor, teacher_tensor):
            require(isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.shape[:2] == (1, 3) and
                    tensor.is_floating_point() and not tensor.requires_grad, "Expected a batch1 preprocessed RGB tensor")
        task = self._activate_task(task_id)
        require(torch.is_grad_enabled(), "Prompt adaptation requires gradients; do not wrap the call in no_grad")
        self._started = self._busy = True
        self._zero_all_grad()
        try:
            student_images = student_tensor.to(self.device, non_blocking=self.use_cuda)
            teacher_images = teacher_tensor.to(self.teacher_device, non_blocking=self.teacher_use_cuda)
            with torch.no_grad():
                teacher_features = self.teacher_model.encode_image(teacher_images)
                teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                teacher_logits = core.CLIP_LOGIT_SCALE * teacher_features @ self.teacher_text_features.t()
                teacher_labels = teacher_logits.argmax(dim=-1).to(self.device)
                teacher_confidence = float(teacher_logits.softmax(dim=-1).max().item())
            logits, cached_student_state, visual_route = self._prepare_student_forward(
                images=student_images, routed_private_experts=task.route, shared_mix_weight=0.5,
                visual_topk_private_experts=1, visual_router_temperature=1.0,
                visual_shared_mix_weight=0.44, use_visual=True)
            probabilities = logits.softmax(dim=-1)
            require(logits.shape == probabilities.shape == (1, len(task.class_ids)) and
                    bool(torch.isfinite(logits).all()) and bool(torch.isfinite(probabilities).all()) and
                    bool(torch.isfinite(teacher_logits).all()), "Invalid task-local student/teacher output")
            prediction = int(probabilities.argmax(dim=-1).item())
            confidence = float(probabilities.max().item())
            selected = confidence >= 0.815
            before = self.optimizer_rounds
            trace = {"task_id": task_id, "stream_index": self.scored,
                "task_local_index": self.task_scored[task_id], "num_classes": len(task.class_ids),
                "student_logits": logits.detach().cpu()[0].tolist(),
                "student_probabilities": probabilities.detach().cpu()[0].tolist(),
                "predicted_class_index": prediction, "prediction": prediction,
                "confidence": confidence, "selected": selected,
                "prediction_source": "student_only", "scoring": "pre_update",
                "updates_before": before, "optimizer_rounds_before_prediction": before,
                "text_route": _route_record(task.route), "visual_route": _route_record(visual_route),
                "teacher_pseudolabel": int(teacher_labels.item()), "teacher_confidence": teacher_confidence}
            if before_update_callback is not None:
                before_update_callback(deepcopy(trace))
            if selected:
                for step in range(3):
                    step_logits = logits if step == 0 else self._recompute_student_logits(
                        cached_image_features=None, base_tokens=cached_student_state,
                        routed_private_experts=task.route, shared_mix_weight=0.5,
                        routed_visual_experts=visual_route, visual_shared_mix_weight=0.44, use_visual=True)
                    self_loss = core.compute_soft_target_cross_entropy(
                        step_logits, core.build_topk_soft_targets(probabilities, pseudo_topk=1))
                    teacher_loss = F.cross_entropy(step_logits, teacher_labels)
                    loss = self_loss + 0.08 * teacher_loss
                    require(bool(torch.isfinite(loss)), "Nonfinite adaptation loss")
                    self._zero_all_grad()
                    loss.backward()
                    self.main_optimizer.step()
                    for expert in visual_route.selected_ids:
                        self.visual_private_optimizers[expert].step()
                    self.optimizer_rounds += 1
                self.selected += 1
                self.task_selected[task_id] += 1
            self.scored += 1
            self.task_scored[task_id] += 1
            require(self.optimizer_rounds == before + (3 if selected else 0) == self.selected * 3,
                    "Global update counter differs from the selected-image count")
            trace["updates_after"] = self.optimizer_rounds
            return trace
        finally:
            # No logits, probabilities or cached text autograd graph are ever
            # stored on the session/task cache. Function locals die on return,
            # including the graph of an unselected sample.
            self._zero_all_grad()
            self._busy = False
