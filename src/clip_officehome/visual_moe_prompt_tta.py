"""Visual-only mixture-of-experts prompt-based test-time adaptation for CLIP ViTs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .clip_utils import build_text_features, freeze_clip_parameters, is_cuda_device
from .evaluator import OfficeHomeImageDataset
from .officehome import OfficeHomeDomain
from .prompt_tta import (
    build_topk_soft_targets,
    compute_entropy_loss,
    compute_soft_target_cross_entropy,
)

CLIP_LOGIT_SCALE = 100.0


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RoutedPrivateVisualExperts:
    selected_ids: list[int]
    selected_weights: torch.Tensor
    routing_scores: list[float]


@dataclass
class VisualMoETTADomainEvalResult:
    env_name: str
    target_domain: str
    source_domains: list[str]
    num_classes: int
    num_images: int
    effective_tta_lr: float
    num_visual_prompts: int
    num_selected_samples: int
    num_skipped_samples: int
    num_updates: int
    selected_ratio: float
    mean_confidence: float
    mean_selected_confidence: float
    top1_accuracy: float
    num_shared_experts: int
    num_private_experts: int
    topk_private_experts: int
    shared_mix_weight: float
    routing_mode: str
    most_selected_private_expert_ids: list[int]
    private_expert_selection_counts: list[int]
    private_expert_selection_ratios: list[float]
    mean_private_selected_weights: list[float]
    mean_routing_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_vit_visual_backbone(model) -> None:
    visual = getattr(model, "visual", None)
    required_attributes = (
        "conv1",
        "class_embedding",
        "positional_embedding",
        "ln_pre",
        "transformer",
        "ln_post",
        "proj",
    )
    if visual is None or any(not hasattr(visual, name) for name in required_attributes):
        raise TypeError("Visual prompt MoE TTA currently supports CLIP VisionTransformer backbones only.")

    if getattr(visual.conv1, "kernel_size", None) != getattr(visual.conv1, "stride", None):
        raise TypeError("Visual prompt MoE TTA currently supports CLIP VisionTransformer backbones only.")


class VisualPromptScaffold(nn.Module):
    """A shallow visual prompt scaffold that wraps a frozen CLIP VisionTransformer."""

    def __init__(
        self,
        model,
        num_visual_prompts: int,
        prompt_init_std: float,
        prompt_scale_init: float,
        device: str,
    ) -> None:
        super().__init__()
        _require_vit_visual_backbone(model)
        if num_visual_prompts < 1:
            raise ValueError(f"num_visual_prompts must be >= 1, got {num_visual_prompts}.")
        if prompt_init_std < 0.0:
            raise ValueError(f"prompt_init_std must be >= 0.0, got {prompt_init_std}.")
        if prompt_scale_init < 0.0:
            raise ValueError(f"prompt_scale_init must be >= 0.0, got {prompt_scale_init}.")

        visual = model.visual
        width = int(visual.class_embedding.shape[0])
        patch_positions = visual.positional_embedding[1:].float().clone()
        if patch_positions.shape[0] >= num_visual_prompts:
            initial_prompt_positions = patch_positions[:num_visual_prompts]
        else:
            repeat_factor = (num_visual_prompts + patch_positions.shape[0] - 1) // patch_positions.shape[0]
            initial_prompt_positions = patch_positions.repeat(repeat_factor, 1)[:num_visual_prompts]
        initial_prompt_positions = initial_prompt_positions.to(device=device, dtype=torch.float32)
        if prompt_init_std > 0.0:
            noise = torch.empty_like(initial_prompt_positions)
            torch.nn.init.normal_(noise, std=prompt_init_std)
            initial_prompt_positions = initial_prompt_positions + noise

        self.num_visual_prompts = num_visual_prompts
        self.width = width
        self.register_buffer(
            "initial_prompt_positional_embeddings",
            initial_prompt_positions.clone(),
        )
        self.prompt_positional_embeddings = nn.Parameter(initial_prompt_positions.clone())
        self.register_buffer(
            "initial_prompt_scale",
            torch.tensor(float(prompt_scale_init), device=device, dtype=torch.float32),
        )
        self.prompt_scale = nn.Parameter(self.initial_prompt_scale.clone())

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.prompt_positional_embeddings.copy_(self.initial_prompt_positional_embeddings)
            self.prompt_scale.copy_(self.initial_prompt_scale)

    @torch.no_grad()
    def prepare_base_tokens(self, model, images: torch.Tensor) -> torch.Tensor:
        visual = model.visual
        x = visual.conv1(images.type(model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_tokens = visual.class_embedding.to(x.dtype)
        cls_tokens = cls_tokens + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        return x.detach()

    def encode_image_from_base_tokens(
        self,
        model,
        base_tokens: torch.Tensor,
        visual_prompt: torch.Tensor,
    ) -> torch.Tensor:
        visual = model.visual
        prompt_tokens = visual_prompt.to(dtype=base_tokens.dtype)
        prompt_tokens = prompt_tokens + self.prompt_positional_embeddings.to(dtype=base_tokens.dtype)
        prompt_tokens = self.prompt_scale.to(dtype=base_tokens.dtype) * prompt_tokens
        prompt_tokens = prompt_tokens.unsqueeze(0).expand(base_tokens.shape[0], -1, -1)

        x = torch.cat(
            [base_tokens[:, :1, :], prompt_tokens, base_tokens[:, 1:, :]],
            dim=1,
        )
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        return x


class VisualPromptExpertBank(nn.Module):
    """A bank of learnable shallow visual prompt experts."""

    def __init__(
        self,
        num_experts: int,
        num_visual_prompts: int,
        width: int,
        expert_init_std: float,
        seed: int,
        device: str,
    ) -> None:
        super().__init__()
        if num_experts < 0:
            raise ValueError(f"num_experts must be >= 0, got {num_experts}.")
        if expert_init_std < 0.0:
            raise ValueError(f"expert_init_std must be >= 0.0, got {expert_init_std}.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        base_prompt = torch.zeros(num_visual_prompts, width, dtype=torch.float32)
        initial_prompts: list[torch.Tensor] = []

        for expert_index in range(num_experts):
            prompt = base_prompt.clone()
            if expert_index > 0 and expert_init_std > 0.0:
                noise = torch.randn(prompt.shape, generator=generator, dtype=prompt.dtype)
                prompt = prompt + expert_init_std * noise
            initial_prompts.append(prompt)

        if initial_prompts:
            initial_stack = torch.stack(initial_prompts, dim=0).to(device)
        else:
            initial_stack = torch.empty(
                0,
                num_visual_prompts,
                width,
                device=device,
                dtype=torch.float32,
            )

        self.register_buffer("initial_prompts", initial_stack)
        self.prompts = nn.ParameterList(
            [nn.Parameter(prompt.clone()) for prompt in initial_stack]
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def reset_parameters(self) -> None:
        for expert_index, parameter in enumerate(self.prompts):
            with torch.no_grad():
                parameter.copy_(self.initial_prompts[expert_index])

    def get_prompt(self, expert_index: int) -> nn.Parameter:
        return self.prompts[expert_index]

    def get_prompts(self, indices: Sequence[int] | None = None) -> list[nn.Parameter]:
        if indices is None:
            return list(self.prompts)
        return [self.prompts[index] for index in indices]


def assert_visual_prompt_only_training(
    model,
    scaffold: VisualPromptScaffold,
    shared_bank: VisualPromptExpertBank,
    private_bank: VisualPromptExpertBank,
) -> None:
    unexpected_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if unexpected_trainable:
        raise RuntimeError(
            f"CLIP parameters must be frozen during visual prompt TTA: {unexpected_trainable[:5]}"
        )

    scaffold_trainable = [name for name, param in scaffold.named_parameters() if param.requires_grad]
    if scaffold_trainable != ["prompt_positional_embeddings", "prompt_scale"]:
        raise RuntimeError(
            "Visual prompt scaffold expects only prompt_positional_embeddings/prompt_scale to be trainable, "
            f"got: {scaffold_trainable}"
        )

    expected_shared = [f"prompts.{index}" for index in range(len(shared_bank))]
    expected_private = [f"prompts.{index}" for index in range(len(private_bank))]
    shared_trainable = [name for name, param in shared_bank.named_parameters() if param.requires_grad]
    private_trainable = [name for name, param in private_bank.named_parameters() if param.requires_grad]
    if shared_trainable != expected_shared:
        raise RuntimeError(
            "All shared visual prompt experts must remain trainable, "
            f"got: {shared_trainable}, expected: {expected_shared}"
        )
    if private_trainable != expected_private:
        raise RuntimeError(
            "All private visual prompt experts must remain trainable, "
            f"got: {private_trainable}, expected: {expected_private}"
        )


def compose_mean_prompt(prompts: Sequence[torch.Tensor]) -> torch.Tensor | None:
    if not prompts:
        return None
    return torch.stack(list(prompts), dim=0).mean(dim=0)


def compose_weighted_prompt(
    prompts: Sequence[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not prompts:
        return None
    stacked = torch.stack(list(prompts), dim=0)
    if weights is None:
        return stacked.mean(dim=0)
    normalized_weights = weights.to(device=stacked.device, dtype=stacked.dtype)
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-12)
    return (stacked * normalized_weights.view(-1, 1, 1)).sum(dim=0)


def compose_effective_prompt(
    shared_prompt: torch.Tensor | None,
    private_prompt: torch.Tensor | None,
    shared_mix_weight: float,
) -> torch.Tensor:
    if shared_prompt is None and private_prompt is None:
        raise ValueError("At least one of shared/private visual prompts must be available.")
    if shared_prompt is None:
        return private_prompt
    if private_prompt is None:
        return shared_prompt
    private_mix_weight = 1.0 - shared_mix_weight
    return shared_mix_weight * shared_prompt + private_mix_weight * private_prompt


def build_active_visual_prompt(
    shared_bank: VisualPromptExpertBank,
    private_bank: VisualPromptExpertBank,
    selected_private_expert_ids: Sequence[int],
    selected_private_expert_weights: torch.Tensor,
    shared_mix_weight: float,
) -> torch.Tensor:
    shared_prompt = compose_mean_prompt(shared_bank.get_prompts())
    private_prompt = compose_weighted_prompt(
        private_bank.get_prompts(selected_private_expert_ids),
        selected_private_expert_weights,
    )
    return compose_effective_prompt(
        shared_prompt,
        private_prompt,
        shared_mix_weight=shared_mix_weight,
    )


@torch.no_grad()
def route_private_visual_experts_for_sample(
    model,
    scaffold: VisualPromptScaffold,
    shared_bank: VisualPromptExpertBank,
    private_bank: VisualPromptExpertBank,
    base_tokens: torch.Tensor,
    text_features: torch.Tensor,
    topk_private_experts: int,
    router_temperature: float,
    shared_mix_weight: float,
) -> RoutedPrivateVisualExperts:
    if len(private_bank) == 0:
        empty_weights = torch.empty(0, device=base_tokens.device, dtype=model.dtype)
        return RoutedPrivateVisualExperts([], empty_weights, [])

    shared_prompt = compose_mean_prompt(shared_bank.get_prompts())
    scores: list[float] = []
    for expert_index in range(len(private_bank)):
        private_prompt = private_bank.get_prompt(expert_index)
        active_prompt = compose_effective_prompt(
            shared_prompt,
            private_prompt,
            shared_mix_weight=shared_mix_weight,
        )
        image_features = scaffold.encode_image_from_base_tokens(model, base_tokens, active_prompt)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = CLIP_LOGIT_SCALE * image_features @ text_features.t()
        probabilities = logits.softmax(dim=-1)
        scores.append(float(probabilities.max(dim=-1).values.mean().item()))

    score_tensor = torch.tensor(scores, device=base_tokens.device, dtype=model.dtype)
    actual_topk = min(max(int(topk_private_experts), 1), len(private_bank))
    selected_scores, selected_indices = torch.topk(
        score_tensor,
        k=actual_topk,
        largest=True,
        sorted=True,
    )
    selected_weights = torch.softmax(selected_scores / router_temperature, dim=0)
    return RoutedPrivateVisualExperts(
        selected_ids=selected_indices.tolist(),
        selected_weights=selected_weights,
        routing_scores=scores,
    )


def build_common_optimizer(
    scaffold: VisualPromptScaffold,
    shared_bank: VisualPromptExpertBank,
    tta_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    common_parameters: list[nn.Parameter] = [
        scaffold.prompt_positional_embeddings,
        scaffold.prompt_scale,
    ]
    common_parameters.extend(shared_bank.get_prompts())
    return torch.optim.AdamW(common_parameters, lr=tta_lr, weight_decay=weight_decay)


def build_private_optimizers(
    private_bank: VisualPromptExpertBank,
    tta_lr: float,
    weight_decay: float,
) -> dict[int, torch.optim.Optimizer]:
    return {
        expert_index: torch.optim.AdamW(
            [private_bank.get_prompt(expert_index)],
            lr=tta_lr,
            weight_decay=weight_decay,
        )
        for expert_index in range(len(private_bank))
    }


class PersistentVisualMoEPromptTTASession:
    """A visual-only prompt MoE TTA session with persistent shared/private experts."""

    def __init__(
        self,
        model,
        preprocess,
        reference_domain_spec: OfficeHomeDomain,
        num_workers: int,
        device: str,
        template: str,
        num_visual_prompts: int,
        num_shared_experts: int,
        num_private_experts: int,
        prompt_init_std: float,
        prompt_scale_init: float,
        expert_init_std: float,
        seed: int,
    ) -> None:
        if num_shared_experts < 0:
            raise ValueError(f"num_shared_experts must be >= 0, got {num_shared_experts}.")
        if num_private_experts < 0:
            raise ValueError(f"num_private_experts must be >= 0, got {num_private_experts}.")
        if num_shared_experts == 0 and num_private_experts == 0:
            raise ValueError("At least one shared/private visual prompt expert is required.")

        self.model = model
        self.preprocess = preprocess
        self.num_workers = num_workers
        self.device = device
        self.prompt_class_names = tuple(reference_domain_spec.prompt_class_names)
        self.class_names = tuple(reference_domain_spec.class_names)
        self.template = template
        self.num_visual_prompts = num_visual_prompts

        seed_everything(seed)
        _require_vit_visual_backbone(model)
        freeze_clip_parameters(model)
        self.use_cuda = is_cuda_device(device)

        self.text_features = build_text_features(
            model=model,
            class_names=self.prompt_class_names,
            template=template,
            device=device,
        )
        self.scaffold = VisualPromptScaffold(
            model=model,
            num_visual_prompts=num_visual_prompts,
            prompt_init_std=prompt_init_std,
            prompt_scale_init=prompt_scale_init,
            device=device,
        ).to(device)
        self.shared_bank = VisualPromptExpertBank(
            num_experts=num_shared_experts,
            num_visual_prompts=num_visual_prompts,
            width=self.scaffold.width,
            expert_init_std=expert_init_std,
            seed=seed,
            device=device,
        ).to(device)
        self.private_bank = VisualPromptExpertBank(
            num_experts=num_private_experts,
            num_visual_prompts=num_visual_prompts,
            width=self.scaffold.width,
            expert_init_std=expert_init_std,
            seed=seed + 10_000,
            device=device,
        ).to(device)
        assert_visual_prompt_only_training(
            model=model,
            scaffold=self.scaffold,
            shared_bank=self.shared_bank,
            private_bank=self.private_bank,
        )

    def _assert_compatible_domain(self, domain_spec: OfficeHomeDomain) -> None:
        if tuple(domain_spec.prompt_class_names) != self.prompt_class_names:
            raise ValueError(
                "All visual MoE TTA tasks must share the same prompt class order. "
                f"Expected {self.prompt_class_names[:5]}, got {tuple(domain_spec.prompt_class_names)[:5]}."
            )
        if tuple(domain_spec.class_names) != self.class_names:
            raise ValueError(
                "All visual MoE TTA tasks must share the same class order. "
                f"Expected {self.class_names[:5]}, got {tuple(domain_spec.class_names)[:5]}."
            )

    def evaluate_domain(
        self,
        domain_spec: OfficeHomeDomain,
        tta_lr: float,
        pseudo_topk: int,
        update_steps: int,
        entropy_weight: float,
        confidence_threshold: float,
        shared_mix_weight: float,
        topk_private_experts: int,
        router_temperature: float,
        weight_decay: float = 0.0,
    ) -> VisualMoETTADomainEvalResult:
        if pseudo_topk < 1:
            raise ValueError(f"pseudo_topk must be >= 1, got {pseudo_topk}.")
        if update_steps < 1:
            raise ValueError(f"update_steps must be >= 1, got {update_steps}.")
        if entropy_weight < 0.0:
            raise ValueError(f"entropy_weight must be >= 0.0, got {entropy_weight}.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be within [0.0, 1.0], got {confidence_threshold}."
            )
        if not 0.0 <= shared_mix_weight <= 1.0:
            raise ValueError(f"shared_mix_weight must be within [0.0, 1.0], got {shared_mix_weight}.")
        if topk_private_experts < 1 and len(self.private_bank) > 0:
            raise ValueError(
                "topk_private_experts must be >= 1 when private experts exist, "
                f"got {topk_private_experts}."
            )
        if router_temperature <= 0.0:
            raise ValueError(f"router_temperature must be > 0.0, got {router_temperature}.")

        self._assert_compatible_domain(domain_spec)
        dataset = OfficeHomeImageDataset(domain_spec.examples, self.preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
        )

        common_optimizer = build_common_optimizer(
            scaffold=self.scaffold,
            shared_bank=self.shared_bank,
            tta_lr=tta_lr,
            weight_decay=weight_decay,
        )
        private_optimizers = build_private_optimizers(
            private_bank=self.private_bank,
            tta_lr=tta_lr,
            weight_decay=weight_decay,
        )

        total = 0
        correct = 0
        num_selected_samples = 0
        num_skipped_samples = 0
        num_updates = 0
        confidence_sum = 0.0
        selected_confidence_sum = 0.0
        private_expert_selection_counts = [0 for _ in range(len(self.private_bank))]
        private_expert_selected_weight_sums = [0.0 for _ in range(len(self.private_bank))]
        private_expert_routing_score_sums = [0.0 for _ in range(len(self.private_bank))]

        progress = tqdm(dataloader, desc=f"Visual MoE TTA {domain_spec.target_domain}", leave=False)
        for images, labels in progress:
            images = images.to(self.device, non_blocking=self.use_cuda)
            labels = labels.to(self.device, non_blocking=self.use_cuda)

            base_tokens = self.scaffold.prepare_base_tokens(self.model, images)
            routed_private_experts = route_private_visual_experts_for_sample(
                model=self.model,
                scaffold=self.scaffold,
                shared_bank=self.shared_bank,
                private_bank=self.private_bank,
                base_tokens=base_tokens,
                text_features=self.text_features,
                topk_private_experts=topk_private_experts,
                router_temperature=router_temperature,
                shared_mix_weight=shared_mix_weight,
            )
            for expert_index, score in enumerate(routed_private_experts.routing_scores):
                private_expert_routing_score_sums[expert_index] += score
            for expert_index, weight in zip(
                routed_private_experts.selected_ids,
                routed_private_experts.selected_weights.tolist(),
                strict=False,
            ):
                private_expert_selection_counts[expert_index] += 1
                private_expert_selected_weight_sums[expert_index] += float(weight)

            active_prompt = build_active_visual_prompt(
                shared_bank=self.shared_bank,
                private_bank=self.private_bank,
                selected_private_expert_ids=routed_private_experts.selected_ids,
                selected_private_expert_weights=routed_private_experts.selected_weights,
                shared_mix_weight=shared_mix_weight,
            )
            image_features = self.scaffold.encode_image_from_base_tokens(
                self.model,
                base_tokens,
                active_prompt,
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = CLIP_LOGIT_SCALE * image_features @ self.text_features.t()
            probabilities = logits.softmax(dim=-1)
            predictions = probabilities.argmax(dim=-1)
            confidence = float(probabilities.max(dim=-1).values.item())

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            confidence_sum += confidence

            is_selected = confidence >= confidence_threshold if confidence_threshold < 1.0 else False
            if is_selected:
                num_selected_samples += labels.size(0)
                selected_confidence_sum += confidence
                soft_targets = build_topk_soft_targets(probabilities, pseudo_topk=pseudo_topk)

                for step_index in range(update_steps):
                    if step_index == 0:
                        step_logits = logits
                    else:
                        active_prompt = build_active_visual_prompt(
                            shared_bank=self.shared_bank,
                            private_bank=self.private_bank,
                            selected_private_expert_ids=routed_private_experts.selected_ids,
                            selected_private_expert_weights=routed_private_experts.selected_weights,
                            shared_mix_weight=shared_mix_weight,
                        )
                        image_features = self.scaffold.encode_image_from_base_tokens(
                            self.model,
                            base_tokens,
                            active_prompt,
                        )
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        step_logits = CLIP_LOGIT_SCALE * image_features @ self.text_features.t()

                    ce_loss = compute_soft_target_cross_entropy(step_logits, soft_targets)
                    if entropy_weight > 0.0:
                        entropy_loss = compute_entropy_loss(step_logits)
                        loss = ce_loss + entropy_weight * entropy_loss
                    else:
                        loss = ce_loss

                    common_optimizer.zero_grad(set_to_none=True)
                    for expert_index in routed_private_experts.selected_ids:
                        private_optimizers[expert_index].zero_grad(set_to_none=True)
                    loss.backward()
                    common_optimizer.step()
                    for expert_index in routed_private_experts.selected_ids:
                        private_optimizers[expert_index].step()
                    num_updates += 1
            else:
                num_skipped_samples += labels.size(0)

        accuracy = correct / total if total else 0.0
        selected_ratio = num_selected_samples / total if total else 0.0
        mean_confidence = confidence_sum / total if total else 0.0
        mean_selected_confidence = (
            selected_confidence_sum / num_selected_samples if num_selected_samples else 0.0
        )
        private_expert_selection_ratios = [
            count / total if total else 0.0 for count in private_expert_selection_counts
        ]
        mean_private_selected_weights = [
            (
                private_expert_selected_weight_sums[index] / private_expert_selection_counts[index]
                if private_expert_selection_counts[index] > 0
                else 0.0
            )
            for index in range(len(self.private_bank))
        ]
        mean_routing_scores = [
            (
                private_expert_routing_score_sums[index] / total
                if total
                else 0.0
            )
            for index in range(len(self.private_bank))
        ]
        most_selected_private_expert_ids = sorted(
            range(len(self.private_bank)),
            key=lambda index: private_expert_selection_counts[index],
            reverse=True,
        )

        return VisualMoETTADomainEvalResult(
            env_name=domain_spec.env_name,
            target_domain=domain_spec.target_domain,
            source_domains=list(domain_spec.source_domains),
            num_classes=len(domain_spec.class_names),
            num_images=len(domain_spec.examples),
            effective_tta_lr=tta_lr,
            num_visual_prompts=self.num_visual_prompts,
            num_selected_samples=num_selected_samples,
            num_skipped_samples=num_skipped_samples,
            num_updates=num_updates,
            selected_ratio=selected_ratio,
            mean_confidence=mean_confidence,
            mean_selected_confidence=mean_selected_confidence,
            top1_accuracy=accuracy,
            num_shared_experts=len(self.shared_bank),
            num_private_experts=len(self.private_bank),
            topk_private_experts=min(topk_private_experts, len(self.private_bank))
            if len(self.private_bank) > 0
            else 0,
            shared_mix_weight=shared_mix_weight,
            routing_mode="sample_wise",
            most_selected_private_expert_ids=most_selected_private_expert_ids,
            private_expert_selection_counts=private_expert_selection_counts,
            private_expert_selection_ratios=private_expert_selection_ratios,
            mean_private_selected_weights=mean_private_selected_weights,
            mean_routing_scores=mean_routing_scores,
        )
