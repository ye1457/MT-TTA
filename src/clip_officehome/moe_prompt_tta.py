"""Mixture-of-experts prompt-based test-time adaptation for CLIP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import DEFAULT_MOE_VISUAL_ROUTER_SOURCE, DEFAULT_TEMPLATE
from .clip_utils import (
    build_text_features,
    encode_text_from_embeddings,
    freeze_clip_parameters,
    import_clip,
    is_cuda_device,
)
from .evaluator import OfficeHomeDualImageDataset, OfficeHomeImageDataset
from .officehome import ImageExample, OfficeHomeDomain
from .prompt_tta import (
    build_topk_soft_targets,
    compute_entropy_loss,
    compute_soft_target_cross_entropy,
)
from .visual_moe_prompt_tta import (
    RoutedPrivateVisualExperts,
    VisualPromptExpertBank,
    VisualPromptScaffold,
    assert_visual_prompt_only_training,
    build_active_visual_prompt,
    route_private_visual_experts_for_sample,
)

CLIP_LOGIT_SCALE = 100.0


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PromptScaffold(nn.Module):
    """Shared prompt template scaffold used by all experts."""

    def __init__(
        self,
        model,
        class_names: Sequence[str],
        prompt_init: str,
        num_context: int,
        device: str,
    ) -> None:
        super().__init__()
        clip = import_clip()
        normalized_prompt_init = prompt_init.strip()
        tokenized_prompt_init = clip.tokenize(normalized_prompt_init).to(device)
        prompt_token_count = int(tokenized_prompt_init.argmax(dim=-1).item()) - 1
        if prompt_token_count != num_context:
            raise ValueError(
                f"prompt_init='{prompt_init}' produces {prompt_token_count} tokens, "
                f"but num_context={num_context}."
            )

        prompt_texts = [f"{normalized_prompt_init} {class_name}".strip() for class_name in class_names]
        tokenized_prompts = clip.tokenize(prompt_texts).to(device)

        with torch.no_grad():
            prompt_embeddings = model.token_embedding(tokenized_prompts).type(model.dtype)
            base_context = model.token_embedding(tokenized_prompt_init)[0, 1 : 1 + num_context]

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("prefix_embeddings", prompt_embeddings[:, :1, :])
        self.register_buffer("suffix_embeddings", prompt_embeddings[:, 1 + num_context :, :])
        self.register_buffer("base_context", base_context.float().clone())

    def build_prompt_embeddings(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        repeated_context = context.to(dtype=self.prefix_embeddings.dtype)
        repeated_context = repeated_context.unsqueeze(0).expand(
            self.tokenized_prompts.size(0),
            -1,
            -1,
        )
        prompt_embeddings = torch.cat(
            [self.prefix_embeddings, repeated_context, self.suffix_embeddings],
            dim=1,
        )
        return prompt_embeddings, self.tokenized_prompts

    def encode_text_features(self, model, context: torch.Tensor) -> torch.Tensor:
        prompt_embeddings, tokenized_prompts = self.build_prompt_embeddings(context)
        return encode_text_from_embeddings(model, prompt_embeddings, tokenized_prompts)


class PromptExpertBank(nn.Module):
    """A bank of learnable prompt experts."""

    def __init__(
        self,
        base_context: torch.Tensor,
        num_experts: int,
        expert_init_std: float,
        seed: int,
        device: str,
    ) -> None:
        super().__init__()
        if num_experts < 0:
            raise ValueError(f"num_experts must be >= 0, got {num_experts}.")

        self.device = device
        initial_contexts: list[torch.Tensor] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        for expert_index in range(num_experts):
            context = base_context.float().clone().cpu()
            if expert_index > 0 and expert_init_std > 0.0:
                noise = torch.randn(
                    context.shape,
                    generator=generator,
                    dtype=context.dtype,
                )
                context = context + expert_init_std * noise
            initial_contexts.append(context)

        if initial_contexts:
            initial_stack = torch.stack(initial_contexts, dim=0).to(device)
        else:
            initial_stack = torch.empty(
                0,
                *base_context.shape,
                device=device,
                dtype=torch.float32,
            )

        self.register_buffer("initial_contexts", initial_stack)
        self.contexts = nn.ParameterList(
            [nn.Parameter(context.clone()) for context in initial_stack]
        )

    def __len__(self) -> int:
        return len(self.contexts)

    def reset_parameters(self) -> None:
        for expert_index, parameter in enumerate(self.contexts):
            with torch.no_grad():
                parameter.copy_(self.initial_contexts[expert_index])

    def get_context(self, expert_index: int) -> nn.Parameter:
        return self.contexts[expert_index]

    def get_contexts(self, indices: Sequence[int] | None = None) -> list[nn.Parameter]:
        if indices is None:
            return list(self.contexts)
        return [self.contexts[index] for index in indices]


@dataclass
class MoETTADomainEvalResult:
    env_name: str
    target_domain: str
    source_domains: list[str]
    num_classes: int
    num_images: int
    effective_tta_lr: float
    num_selected_samples: int
    num_skipped_samples: int
    num_updates: int
    selected_ratio: float
    mean_confidence: float
    mean_selected_confidence: float
    top1_accuracy: float
    mean_anchor_loss: float
    num_shared_experts: int
    num_private_experts: int
    shared_mix_weight: float
    routing_mode: str
    selected_private_expert_ids: list[int]
    selected_private_expert_weights: list[float]
    routing_buffer_size: int
    routing_scores: list[float]
    private_expert_selection_counts: list[int]
    private_expert_selection_ratios: list[float]
    mean_private_selected_weights: list[float]
    num_visual_prompts: int
    num_visual_shared_experts: int
    num_visual_private_experts: int
    visual_shared_mix_weight: float
    visual_routing_mode: str
    visual_router_source: str
    visual_domain_enabled: bool
    visual_topk_private_experts: int
    most_selected_visual_private_expert_ids: list[int]
    visual_private_expert_selection_counts: list[int]
    visual_private_expert_selection_ratios: list[float]
    mean_visual_private_selected_weights: list[float]
    mean_visual_routing_scores: list[float]
    mean_teacher_confidence: float = 0.0
    mean_selected_teacher_confidence: float = 0.0
    num_teacher_selected_samples: int = 0
    mean_kl_loss: float = 0.0
    teacher_supervision_mode: str = "off"
    teacher_student_preupdate_agreement: float = 0.0
    mean_teacher_ce_loss: float = 0.0
    num_teacher_guided_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoutedPrivateExperts:
    selected_ids: list[int]
    selected_weights: torch.Tensor
    routing_scores: list[float]


def build_empty_visual_route(device: str, dtype: torch.dtype) -> RoutedPrivateVisualExperts:
    return RoutedPrivateVisualExperts(
        selected_ids=[],
        selected_weights=torch.empty(0, device=device, dtype=dtype),
        routing_scores=[],
    )


def assert_moe_prompt_only_training(
    model,
    shared_bank: PromptExpertBank,
    private_bank: PromptExpertBank,
) -> None:
    unexpected_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if unexpected_trainable:
        raise RuntimeError(f"CLIP parameters must be frozen during TTA: {unexpected_trainable[:5]}")

    expected_shared = [f"contexts.{index}" for index in range(len(shared_bank))]
    expected_private = [f"contexts.{index}" for index in range(len(private_bank))]

    shared_trainable = [name for name, param in shared_bank.named_parameters() if param.requires_grad]
    private_trainable = [name for name, param in private_bank.named_parameters() if param.requires_grad]
    if shared_trainable != expected_shared:
        raise RuntimeError(
            "All shared experts must remain trainable, "
            f"got: {shared_trainable}, expected: {expected_shared}"
        )
    if private_trainable != expected_private:
        raise RuntimeError(
            "All private experts must remain trainable before routing selection, "
            f"got: {private_trainable}, expected: {expected_private}"
        )


def compose_mean_context(contexts: Sequence[torch.Tensor]) -> torch.Tensor | None:
    if not contexts:
        return None
    return torch.stack(list(contexts), dim=0).mean(dim=0)


def compose_weighted_context(
    contexts: Sequence[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not contexts:
        return None
    stacked = torch.stack(list(contexts), dim=0)
    if weights is None:
        return stacked.mean(dim=0)
    normalized_weights = weights.to(device=stacked.device, dtype=stacked.dtype)
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-12)
    return (stacked * normalized_weights.view(-1, 1, 1)).sum(dim=0)


def compose_effective_context(
    shared_context: torch.Tensor | None,
    private_context: torch.Tensor | None,
    shared_mix_weight: float = 0.5,
) -> torch.Tensor:
    if shared_context is None and private_context is None:
        raise ValueError("At least one of shared/private context must be available.")
    if shared_context is None:
        return private_context
    if private_context is None:
        return shared_context
    private_mix_weight = 1.0 - shared_mix_weight
    return shared_mix_weight * shared_context + private_mix_weight * private_context


@torch.no_grad()
def collect_image_features(
    model,
    preprocess,
    examples: Sequence[ImageExample],
    num_workers: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    use_cuda = is_cuda_device(device)
    dataset = OfficeHomeImageDataset(tuple(examples), preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    feature_batches: list[torch.Tensor] = []
    for images, _ in dataloader:
        images = images.to(device, non_blocking=use_cuda)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        feature_batches.append(image_features.detach())
    return torch.cat(feature_batches, dim=0)


@torch.no_grad()
def route_private_experts_for_features(
    model,
    scaffold: PromptScaffold,
    shared_bank: PromptExpertBank,
    private_bank: PromptExpertBank,
    image_features: torch.Tensor,
    topk_private_experts: int,
    routing_score: str,
    router_temperature: float,
    shared_mix_weight: float,
) -> tuple[list[int], torch.Tensor, list[float]]:
    if len(private_bank) == 0:
        empty_weights = torch.empty(0, device=image_features.device, dtype=model.dtype)
        return [], empty_weights, []
    if routing_score != "confidence":
        raise ValueError(f"Unsupported routing_score '{routing_score}'.")

    shared_context = compose_mean_context(shared_bank.get_contexts())
    scores: list[float] = []

    for expert_index in range(len(private_bank)):
        private_context = private_bank.get_context(expert_index)
        effective_context = compose_effective_context(
            shared_context,
            private_context,
            shared_mix_weight=shared_mix_weight,
        )
        text_features = scaffold.encode_text_features(model, effective_context)
        logits = CLIP_LOGIT_SCALE * image_features @ text_features.t()
        probabilities = logits.softmax(dim=-1)
        score = float(probabilities.max(dim=-1).values.mean().item())
        scores.append(score)

    score_tensor = torch.tensor(scores, device=image_features.device, dtype=model.dtype)
    actual_topk = min(max(int(topk_private_experts), 1), len(private_bank))
    selected_scores, selected_indices = torch.topk(score_tensor, k=actual_topk, largest=True, sorted=True)
    selected_weights = torch.softmax(selected_scores / router_temperature, dim=0)
    return selected_indices.tolist(), selected_weights, scores


def build_optimizer_for_active_experts(
    shared_bank: PromptExpertBank,
    private_bank: PromptExpertBank,
    selected_private_expert_ids: Sequence[int],
    tta_lr: float,
    weight_decay: float,
    visual_scaffold: VisualPromptScaffold | None = None,
    visual_shared_bank: VisualPromptExpertBank | None = None,
) -> torch.optim.Optimizer:
    active_parameters: list[nn.Parameter] = []
    active_parameters.extend(shared_bank.get_contexts())
    active_parameters.extend(private_bank.get_contexts(selected_private_expert_ids))
    if visual_scaffold is not None:
        active_parameters.append(visual_scaffold.prompt_positional_embeddings)
        active_parameters.append(visual_scaffold.prompt_scale)
    if visual_shared_bank is not None:
        active_parameters.extend(visual_shared_bank.get_prompts())
    if not active_parameters:
        raise ValueError("No active prompt experts were selected for optimization.")
    return torch.optim.AdamW(active_parameters, lr=tta_lr, weight_decay=weight_decay)


def build_private_visual_optimizers(
    private_bank: VisualPromptExpertBank | None,
    tta_lr: float,
    weight_decay: float,
) -> dict[int, torch.optim.Optimizer]:
    if private_bank is None:
        return {}
    return {
        expert_index: torch.optim.AdamW(
            [private_bank.get_prompt(expert_index)],
            lr=tta_lr,
            weight_decay=weight_decay,
        )
        for expert_index in range(len(private_bank))
    }


def build_active_context(
    shared_bank: PromptExpertBank,
    private_bank: PromptExpertBank,
    selected_private_expert_ids: Sequence[int],
    selected_private_expert_weights: torch.Tensor,
    shared_mix_weight: float,
) -> torch.Tensor:
    shared_context = compose_mean_context(shared_bank.get_contexts())
    private_context = compose_weighted_context(
        private_bank.get_contexts(selected_private_expert_ids),
        selected_private_expert_weights,
    )
    return compose_effective_context(
        shared_context,
        private_context,
        shared_mix_weight=shared_mix_weight,
    )


def compute_shared_anchor_loss(
    shared_bank: PromptExpertBank,
    shared_anchor_snapshots: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(shared_bank) == 0:
        device = shared_anchor_snapshots[0].device if shared_anchor_snapshots else "cpu"
        return torch.zeros((), device=device, dtype=torch.float32)

    losses = []
    for index, parameter in enumerate(shared_bank.get_contexts()):
        anchor_target = shared_anchor_snapshots[index].to(device=parameter.device, dtype=parameter.dtype)
        losses.append(torch.mean((parameter - anchor_target) ** 2))
    return torch.stack(losses, dim=0).mean()


def compute_distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probabilities = F.log_softmax((student_logits / temperature).float(), dim=-1)
    kl_loss = F.kl_div(
        student_log_probabilities,
        teacher_probabilities.float(),
        reduction="batchmean",
    )
    return (temperature ** 2) * kl_loss


class PersistentMoEPromptTTASession:
    """Continual MoE prompt-TTA session with persistent expert banks."""

    def __init__(
        self,
        model,
        preprocess,
        reference_domain_spec: OfficeHomeDomain,
        num_workers: int,
        device: str,
        num_context: int,
        prompt_init: str,
        num_shared_experts: int,
        num_private_experts: int,
        expert_init_std: float,
        seed: int,
        num_visual_prompts: int = 4,
        visual_prompt_init_std: float = 0.0,
        visual_prompt_scale_init: float = 1.0,
        num_visual_shared_experts: int = 0,
        num_visual_private_experts: int = 0,
        visual_router_source: str = DEFAULT_MOE_VISUAL_ROUTER_SOURCE,
        disable_visual_domains: Sequence[str] | None = None,
        teacher_model=None,
        teacher_preprocess=None,
        teacher_device: str | None = None,
        teacher_template: str = "a photo of a {class}",
        teacher_temperature: float = 1.0,
    ) -> None:
        if num_shared_experts < 0:
            raise ValueError(f"num_shared_experts must be >= 0, got {num_shared_experts}.")
        if num_private_experts < 0:
            raise ValueError(f"num_private_experts must be >= 0, got {num_private_experts}.")
        if num_shared_experts == 0 and num_private_experts == 0:
            raise ValueError("At least one shared/private expert is required.")
        if num_visual_shared_experts < 0:
            raise ValueError(
                f"num_visual_shared_experts must be >= 0, got {num_visual_shared_experts}."
            )
        if num_visual_private_experts < 0:
            raise ValueError(
                f"num_visual_private_experts must be >= 0, got {num_visual_private_experts}."
            )
        if visual_router_source not in {"fixed_zeroshot_text", "current_text_moe"}:
            raise ValueError(
                "visual_router_source must be one of {'fixed_zeroshot_text', 'current_text_moe'}, "
                f"got {visual_router_source!r}."
            )
        if teacher_temperature <= 0.0:
            raise ValueError(f"teacher_temperature must be > 0.0, got {teacher_temperature}.")
        if (teacher_model is None) != (teacher_preprocess is None):
            raise ValueError("teacher_model and teacher_preprocess must be provided together.")

        self.model = model
        self.preprocess = preprocess
        self.num_workers = num_workers
        self.device = device
        self.prompt_class_names = tuple(reference_domain_spec.prompt_class_names)
        self.class_names = tuple(reference_domain_spec.class_names)
        self.seed = seed
        self.teacher_model = teacher_model
        self.teacher_preprocess = teacher_preprocess
        self.teacher_device = teacher_device if teacher_device is not None else device
        self.teacher_temperature = teacher_temperature
        self.teacher_enabled = teacher_model is not None
        self.visual_enabled = (num_visual_shared_experts + num_visual_private_experts) > 0
        self.visual_router_source = visual_router_source if self.visual_enabled else "off"
        self.disable_visual_domains = frozenset(disable_visual_domains or ())

        seed_everything(seed)
        freeze_clip_parameters(model)
        self.use_cuda = is_cuda_device(device)
        self.teacher_use_cuda = is_cuda_device(self.teacher_device)
        if self.teacher_enabled:
            freeze_clip_parameters(self.teacher_model)
            self.teacher_text_features = build_text_features(
                model=self.teacher_model,
                class_names=self.prompt_class_names,
                template=teacher_template,
                device=self.teacher_device,
            )
        else:
            self.teacher_text_features = None

        self.scaffold = PromptScaffold(
            model=model,
            class_names=self.prompt_class_names,
            prompt_init=prompt_init,
            num_context=num_context,
            device=device,
        ).to(device)
        self.shared_bank = PromptExpertBank(
            base_context=self.scaffold.base_context,
            num_experts=num_shared_experts,
            expert_init_std=expert_init_std,
            seed=seed,
            device=device,
        ).to(device)
        self.private_bank = PromptExpertBank(
            base_context=self.scaffold.base_context,
            num_experts=num_private_experts,
            expert_init_std=expert_init_std,
            seed=seed + 10_000,
            device=device,
        ).to(device)
        self.num_visual_prompts = num_visual_prompts if self.visual_enabled else 0
        self.visual_scaffold = None
        self.visual_shared_bank = None
        self.visual_private_bank = None
        self.visual_router_text_features = None
        if self.visual_enabled:
            self.visual_scaffold = VisualPromptScaffold(
                model=model,
                num_visual_prompts=num_visual_prompts,
                prompt_init_std=visual_prompt_init_std,
                prompt_scale_init=visual_prompt_scale_init,
                device=device,
            ).to(device)
            self.visual_shared_bank = VisualPromptExpertBank(
                num_experts=num_visual_shared_experts,
                num_visual_prompts=num_visual_prompts,
                width=self.visual_scaffold.width,
                expert_init_std=expert_init_std,
                seed=seed + 20_000,
                device=device,
            ).to(device)
            self.visual_private_bank = VisualPromptExpertBank(
                num_experts=num_visual_private_experts,
                num_visual_prompts=num_visual_prompts,
                width=self.visual_scaffold.width,
                expert_init_std=expert_init_std,
                seed=seed + 30_000,
                device=device,
            ).to(device)
            if self.visual_router_source == "fixed_zeroshot_text":
                self.visual_router_text_features = build_text_features(
                    model=self.model,
                    class_names=self.prompt_class_names,
                    template=DEFAULT_TEMPLATE,
                    device=self.device,
                )
        self.shared_anchor_snapshots = tuple(
            context.detach().clone() for context in self.shared_bank.get_contexts()
        )
        assert_moe_prompt_only_training(model, self.shared_bank, self.private_bank)
        if self.visual_enabled:
            assert_visual_prompt_only_training(
                model=model,
                scaffold=self.visual_scaffold,
                shared_bank=self.visual_shared_bank,
                private_bank=self.visual_private_bank,
            )

    def _assert_compatible_domain(self, domain_spec: OfficeHomeDomain) -> None:
        if tuple(domain_spec.prompt_class_names) != self.prompt_class_names:
            raise ValueError(
                "All continual MoE tasks must share the same prompt class order. "
                f"Expected {self.prompt_class_names[:5]}, got {tuple(domain_spec.prompt_class_names)[:5]}."
            )
        if tuple(domain_spec.class_names) != self.class_names:
            raise ValueError(
                "All continual MoE tasks must share the same class order. "
                f"Expected {self.class_names[:5]}, got {tuple(domain_spec.class_names)[:5]}."
            )

    def _route_private_experts_for_task(
        self,
        domain_spec: OfficeHomeDomain,
        topk_private_experts: int,
        routing_buffer_size: int,
        routing_score: str,
        router_temperature: float,
        shared_mix_weight: float,
    ) -> RoutedPrivateExperts:
        if len(self.private_bank) == 0:
            return RoutedPrivateExperts(
                selected_ids=[],
                selected_weights=torch.empty(0, device=self.device, dtype=self.model.dtype),
                routing_scores=[],
            )
        warmup_examples = domain_spec.examples[: min(routing_buffer_size, len(domain_spec.examples))]
        if not warmup_examples:
            raise ValueError("routing_buffer_size produced an empty routing warmup set.")
        image_features = collect_image_features(
            model=self.model,
            preprocess=self.preprocess,
            examples=warmup_examples,
            num_workers=self.num_workers,
            device=self.device,
            batch_size=min(64, len(warmup_examples)),
        )
        selected_ids, selected_weights, routing_scores = route_private_experts_for_features(
            model=self.model,
            scaffold=self.scaffold,
            shared_bank=self.shared_bank,
            private_bank=self.private_bank,
            image_features=image_features,
            topk_private_experts=topk_private_experts,
            routing_score=routing_score,
            router_temperature=router_temperature,
            shared_mix_weight=shared_mix_weight,
        )
        return RoutedPrivateExperts(
            selected_ids=selected_ids,
            selected_weights=selected_weights,
            routing_scores=routing_scores,
        )

    def _build_text_features(
        self,
        routed_private_experts: RoutedPrivateExperts,
        shared_mix_weight: float,
    ) -> torch.Tensor:
        active_context = build_active_context(
            shared_bank=self.shared_bank,
            private_bank=self.private_bank,
            selected_private_expert_ids=routed_private_experts.selected_ids,
            selected_private_expert_weights=routed_private_experts.selected_weights,
            shared_mix_weight=shared_mix_weight,
        )
        return self.scaffold.encode_text_features(self.model, active_context)

    def _prepare_student_forward(
        self,
        images: torch.Tensor,
        routed_private_experts: RoutedPrivateExperts,
        shared_mix_weight: float,
        visual_topk_private_experts: int,
        visual_router_temperature: float,
        visual_shared_mix_weight: float,
        use_visual: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, RoutedPrivateVisualExperts]:
        text_features = self._build_text_features(routed_private_experts, shared_mix_weight)
        if not use_visual:
            with torch.no_grad():
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.detach()
            logits = CLIP_LOGIT_SCALE * image_features @ text_features.t()
            return logits, image_features, build_empty_visual_route(self.device, self.model.dtype)

        base_tokens = self.visual_scaffold.prepare_base_tokens(self.model, images)
        if self.visual_router_source == "current_text_moe":
            router_text_features = text_features
        else:
            router_text_features = self.visual_router_text_features
        routed_visual_experts = route_private_visual_experts_for_sample(
            model=self.model,
            scaffold=self.visual_scaffold,
            shared_bank=self.visual_shared_bank,
            private_bank=self.visual_private_bank,
            base_tokens=base_tokens,
            text_features=router_text_features,
            topk_private_experts=visual_topk_private_experts,
            router_temperature=visual_router_temperature,
            shared_mix_weight=visual_shared_mix_weight,
        )
        logits = self._recompute_student_logits(
            cached_image_features=None,
            base_tokens=base_tokens,
            routed_private_experts=routed_private_experts,
            shared_mix_weight=shared_mix_weight,
            routed_visual_experts=routed_visual_experts,
            visual_shared_mix_weight=visual_shared_mix_weight,
            use_visual=use_visual,
        )
        return logits, base_tokens, routed_visual_experts

    def _recompute_student_logits(
        self,
        cached_image_features: torch.Tensor | None,
        base_tokens: torch.Tensor | None,
        routed_private_experts: RoutedPrivateExperts,
        shared_mix_weight: float,
        routed_visual_experts: RoutedPrivateVisualExperts,
        visual_shared_mix_weight: float,
        use_visual: bool,
    ) -> torch.Tensor:
        text_features = self._build_text_features(routed_private_experts, shared_mix_weight)
        if not use_visual:
            if cached_image_features is None:
                raise ValueError("cached_image_features is required when visual prompts are disabled.")
            image_features = cached_image_features
        else:
            if base_tokens is None:
                raise ValueError("base_tokens is required when visual prompts are enabled.")
            active_visual_prompt = build_active_visual_prompt(
                shared_bank=self.visual_shared_bank,
                private_bank=self.visual_private_bank,
                selected_private_expert_ids=routed_visual_experts.selected_ids,
                selected_private_expert_weights=routed_visual_experts.selected_weights,
                shared_mix_weight=visual_shared_mix_weight,
            )
            image_features = self.visual_scaffold.encode_image_from_base_tokens(
                self.model,
                base_tokens,
                active_visual_prompt,
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return CLIP_LOGIT_SCALE * image_features @ text_features.t()

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
        routing_buffer_size: int,
        routing_score: str,
        router_temperature: float,
        shared_anchor_weight: float,
        visual_topk_private_experts: int = 1,
        visual_shared_mix_weight: float = 0.5,
        visual_router_temperature: float = 1.0,
        teacher_supervision: str = "off",
        teacher_ce_weight: float = 0.25,
        teacher_confidence_threshold: float = 0.8,
        distill_kl_weight: float = 0.5,
        weight_decay: float = 0.0,
    ) -> MoETTADomainEvalResult:
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
        if routing_buffer_size < 1 and len(self.private_bank) > 0:
            raise ValueError(
                "routing_buffer_size must be >= 1 when private experts exist, "
                f"got {routing_buffer_size}."
            )
        if shared_anchor_weight < 0.0:
            raise ValueError(f"shared_anchor_weight must be >= 0.0, got {shared_anchor_weight}.")
        if self.visual_enabled and visual_topk_private_experts < 1 and len(self.visual_private_bank) > 0:
            raise ValueError(
                "visual_topk_private_experts must be >= 1 when visual private experts exist, "
                f"got {visual_topk_private_experts}."
            )
        if not 0.0 <= visual_shared_mix_weight <= 1.0:
            raise ValueError(
                f"visual_shared_mix_weight must be within [0.0, 1.0], got {visual_shared_mix_weight}."
            )
        if visual_router_temperature <= 0.0:
            raise ValueError(
                f"visual_router_temperature must be > 0.0, got {visual_router_temperature}."
            )
        if teacher_supervision not in {"off", "aux_pseudolabel", "distill"}:
            raise ValueError(
                "teacher_supervision must be one of {'off', 'aux_pseudolabel', 'distill'}, "
                f"got {teacher_supervision!r}."
            )
        if teacher_ce_weight < 0.0:
            raise ValueError(f"teacher_ce_weight must be >= 0.0, got {teacher_ce_weight}.")
        if not 0.0 <= teacher_confidence_threshold <= 1.0:
            raise ValueError(
                "teacher_confidence_threshold must be within [0.0, 1.0], "
                f"got {teacher_confidence_threshold}."
            )
        if distill_kl_weight < 0.0:
            raise ValueError(f"distill_kl_weight must be >= 0.0, got {distill_kl_weight}.")
        if teacher_supervision != "off" and not self.teacher_enabled:
            raise ValueError("Teacher supervision was requested, but no teacher model is loaded.")

        self._assert_compatible_domain(domain_spec)
        domain_visual_enabled = (
            self.visual_enabled and domain_spec.target_domain not in self.disable_visual_domains
        )
        routed_private_experts = self._route_private_experts_for_task(
            domain_spec=domain_spec,
            topk_private_experts=topk_private_experts,
            routing_buffer_size=routing_buffer_size,
            routing_score=routing_score,
            router_temperature=router_temperature,
            shared_mix_weight=shared_mix_weight,
        )
        optimizer = build_optimizer_for_active_experts(
            shared_bank=self.shared_bank,
            private_bank=self.private_bank,
            selected_private_expert_ids=routed_private_experts.selected_ids,
            tta_lr=tta_lr,
            weight_decay=weight_decay,
            visual_scaffold=self.visual_scaffold if domain_visual_enabled else None,
            visual_shared_bank=self.visual_shared_bank if domain_visual_enabled else None,
        )
        visual_private_optimizers = build_private_visual_optimizers(
            private_bank=self.visual_private_bank if domain_visual_enabled else None,
            tta_lr=tta_lr,
            weight_decay=weight_decay,
        )

        teacher_active = teacher_supervision != "off"
        if teacher_active:
            dataset = OfficeHomeDualImageDataset(
                domain_spec.examples,
                self.preprocess,
                self.teacher_preprocess,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.use_cuda or self.teacher_use_cuda,
            )
        else:
            dataset = OfficeHomeImageDataset(domain_spec.examples, self.preprocess)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.use_cuda,
            )

        total = 0
        correct = 0
        num_selected_samples = 0
        num_skipped_samples = 0
        num_updates = 0
        confidence_sum = 0.0
        selected_confidence_sum = 0.0
        anchor_loss_sum = 0.0
        teacher_confidence_sum = 0.0
        selected_teacher_confidence_sum = 0.0
        num_teacher_selected_samples = 0
        kl_loss_sum = 0.0
        teacher_ce_loss_sum = 0.0
        teacher_student_agreement_count = 0
        num_teacher_guided_updates = 0
        private_expert_selection_counts = [0 for _ in range(len(self.private_bank))]
        private_expert_selected_weight_sums = [0.0 for _ in range(len(self.private_bank))]
        private_expert_routing_score_sums = list(routed_private_experts.routing_scores)
        visual_private_expert_selection_counts = (
            [0 for _ in range(len(self.visual_private_bank))]
            if domain_visual_enabled
            else []
        )
        visual_private_expert_selected_weight_sums = (
            [0.0 for _ in range(len(self.visual_private_bank))]
            if domain_visual_enabled
            else []
        )
        visual_private_expert_routing_score_sums = (
            [0.0 for _ in range(len(self.visual_private_bank))]
            if domain_visual_enabled
            else []
        )

        total_examples = len(domain_spec.examples)
        for expert_index, weight in zip(
            routed_private_experts.selected_ids,
            routed_private_experts.selected_weights.tolist(),
            strict=False,
        ):
            private_expert_selection_counts[expert_index] = total_examples
            private_expert_selected_weight_sums[expert_index] = float(weight) * total_examples

        progress = tqdm(dataloader, desc=f"MoE TTA {domain_spec.target_domain}", leave=False)
        if teacher_active:
            for student_images, teacher_images, labels in progress:
                student_images = student_images.to(self.device, non_blocking=self.use_cuda)
                labels = labels.to(self.device, non_blocking=self.use_cuda)
                teacher_images = teacher_images.to(
                    self.teacher_device,
                    non_blocking=self.teacher_use_cuda,
                )

                with torch.no_grad():
                    teacher_image_features = self.teacher_model.encode_image(teacher_images)
                    teacher_image_features = teacher_image_features / teacher_image_features.norm(
                        dim=-1,
                        keepdim=True,
                    )
                    teacher_logits = CLIP_LOGIT_SCALE * teacher_image_features @ self.teacher_text_features.t()
                    teacher_confidence_probabilities = teacher_logits.softmax(dim=-1)
                    teacher_probabilities = (teacher_logits / self.teacher_temperature).softmax(dim=-1)
                    teacher_hard_labels = teacher_logits.argmax(dim=-1).to(self.device)
                logits, cached_student_state, routed_visual_experts = self._prepare_student_forward(
                    images=student_images,
                    routed_private_experts=routed_private_experts,
                    shared_mix_weight=shared_mix_weight,
                    visual_topk_private_experts=visual_topk_private_experts,
                    visual_router_temperature=visual_router_temperature,
                    visual_shared_mix_weight=visual_shared_mix_weight,
                    use_visual=domain_visual_enabled,
                )
                teacher_probabilities_for_student = teacher_probabilities.to(
                    device=self.device,
                    dtype=logits.dtype,
                )
                if domain_visual_enabled:
                    for expert_index, score in enumerate(routed_visual_experts.routing_scores):
                        visual_private_expert_routing_score_sums[expert_index] += score
                    for expert_index, weight in zip(
                        routed_visual_experts.selected_ids,
                        routed_visual_experts.selected_weights.tolist(),
                        strict=False,
                    ):
                        visual_private_expert_selection_counts[expert_index] += 1
                        visual_private_expert_selected_weight_sums[expert_index] += float(weight)

                probabilities = logits.softmax(dim=-1)
                predictions = probabilities.argmax(dim=-1)
                confidence = float(probabilities.max(dim=-1).values.item())
                teacher_confidence = float(teacher_confidence_probabilities.max(dim=-1).values.item())

                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                confidence_sum += confidence
                teacher_confidence_sum += teacher_confidence
                teacher_student_agreement_count += (predictions == teacher_hard_labels).sum().item()

                if teacher_supervision == "distill":
                    is_selected = (
                        teacher_confidence >= teacher_confidence_threshold
                        if teacher_confidence_threshold < 1.0
                        else False
                    )
                else:
                    is_selected = confidence >= confidence_threshold if confidence_threshold < 1.0 else False
                if is_selected:
                    num_selected_samples += labels.size(0)
                    selected_confidence_sum += confidence
                    if teacher_supervision in {"aux_pseudolabel", "distill"}:
                        num_teacher_selected_samples += labels.size(0)
                        selected_teacher_confidence_sum += teacher_confidence

                    for step_index in range(update_steps):
                        if step_index == 0:
                            step_logits = logits
                        else:
                            step_logits = self._recompute_student_logits(
                                cached_image_features=(
                                    None if domain_visual_enabled else cached_student_state
                                ),
                                base_tokens=(
                                    cached_student_state if domain_visual_enabled else None
                                ),
                                routed_private_experts=routed_private_experts,
                                shared_mix_weight=shared_mix_weight,
                                routed_visual_experts=routed_visual_experts,
                                visual_shared_mix_weight=visual_shared_mix_weight,
                                use_visual=domain_visual_enabled,
                            )

                        self_ce_loss = compute_soft_target_cross_entropy(
                            step_logits,
                            build_topk_soft_targets(probabilities, pseudo_topk=pseudo_topk),
                        )
                        teacher_ce_loss = F.cross_entropy(step_logits, teacher_hard_labels)
                        anchor_loss = compute_shared_anchor_loss(
                            shared_bank=self.shared_bank,
                            shared_anchor_snapshots=self.shared_anchor_snapshots,
                        )
                        if teacher_supervision == "aux_pseudolabel":
                            loss = self_ce_loss + teacher_ce_weight * teacher_ce_loss
                            kl_loss = torch.zeros((), device=step_logits.device, dtype=step_logits.dtype)
                        else:
                            kl_loss = compute_distillation_kl_loss(
                                student_logits=step_logits,
                                teacher_probabilities=teacher_probabilities_for_student,
                                temperature=self.teacher_temperature,
                            )
                            loss = teacher_ce_loss + distill_kl_weight * kl_loss
                        if shared_anchor_weight > 0.0:
                            loss = loss + shared_anchor_weight * anchor_loss
                        optimizer.zero_grad(set_to_none=True)
                        for expert_index in routed_visual_experts.selected_ids:
                            visual_private_optimizers[expert_index].zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        for expert_index in routed_visual_experts.selected_ids:
                            visual_private_optimizers[expert_index].step()
                        num_updates += 1
                        num_teacher_guided_updates += 1
                        anchor_loss_sum += float(anchor_loss.item())
                        kl_loss_sum += float(kl_loss.item())
                        teacher_ce_loss_sum += float(teacher_ce_loss.item())
                else:
                    num_skipped_samples += labels.size(0)
        else:
            for images, labels in progress:
                images = images.to(self.device, non_blocking=self.use_cuda)
                labels = labels.to(self.device, non_blocking=self.use_cuda)

                logits, cached_student_state, routed_visual_experts = self._prepare_student_forward(
                    images=images,
                    routed_private_experts=routed_private_experts,
                    shared_mix_weight=shared_mix_weight,
                    visual_topk_private_experts=visual_topk_private_experts,
                    visual_router_temperature=visual_router_temperature,
                    visual_shared_mix_weight=visual_shared_mix_weight,
                    use_visual=domain_visual_enabled,
                )
                if domain_visual_enabled:
                    for expert_index, score in enumerate(routed_visual_experts.routing_scores):
                        visual_private_expert_routing_score_sums[expert_index] += score
                    for expert_index, weight in zip(
                        routed_visual_experts.selected_ids,
                        routed_visual_experts.selected_weights.tolist(),
                        strict=False,
                    ):
                        visual_private_expert_selection_counts[expert_index] += 1
                        visual_private_expert_selected_weight_sums[expert_index] += float(weight)

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
                            step_logits = self._recompute_student_logits(
                                cached_image_features=(
                                    None if domain_visual_enabled else cached_student_state
                                ),
                                base_tokens=(
                                    cached_student_state if domain_visual_enabled else None
                                ),
                                routed_private_experts=routed_private_experts,
                                shared_mix_weight=shared_mix_weight,
                                routed_visual_experts=routed_visual_experts,
                                visual_shared_mix_weight=visual_shared_mix_weight,
                                use_visual=domain_visual_enabled,
                            )

                        ce_loss = compute_soft_target_cross_entropy(step_logits, soft_targets)
                        anchor_loss = compute_shared_anchor_loss(
                            shared_bank=self.shared_bank,
                            shared_anchor_snapshots=self.shared_anchor_snapshots,
                        )
                        if entropy_weight > 0.0:
                            entropy_loss = compute_entropy_loss(step_logits)
                            loss = ce_loss + entropy_weight * entropy_loss
                        else:
                            loss = ce_loss
                        if shared_anchor_weight > 0.0:
                            loss = loss + shared_anchor_weight * anchor_loss
                        optimizer.zero_grad(set_to_none=True)
                        for expert_index in routed_visual_experts.selected_ids:
                            visual_private_optimizers[expert_index].zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        for expert_index in routed_visual_experts.selected_ids:
                            visual_private_optimizers[expert_index].step()
                        num_updates += 1
                        anchor_loss_sum += float(anchor_loss.item())
                else:
                    num_skipped_samples += labels.size(0)

        accuracy = correct / total if total else 0.0
        selected_ratio = num_selected_samples / total if total else 0.0
        mean_confidence = confidence_sum / total if total else 0.0
        mean_selected_confidence = (
            selected_confidence_sum / num_selected_samples if num_selected_samples else 0.0
        )
        mean_anchor_loss = anchor_loss_sum / num_updates if num_updates else 0.0
        mean_teacher_confidence = teacher_confidence_sum / total if total and teacher_active else 0.0
        mean_selected_teacher_confidence = (
            selected_teacher_confidence_sum / num_teacher_selected_samples
            if num_teacher_selected_samples
            else 0.0
        )
        mean_kl_loss = kl_loss_sum / num_updates if num_updates and teacher_supervision == "distill" else 0.0
        mean_teacher_ce_loss = (
            teacher_ce_loss_sum / num_teacher_guided_updates if num_teacher_guided_updates else 0.0
        )
        teacher_student_preupdate_agreement = (
            teacher_student_agreement_count / total if total and teacher_active else 0.0
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
            private_expert_routing_score_sums[index]
            for index in range(len(self.private_bank))
        ]
        visual_private_expert_selection_ratios = [
            count / total if total else 0.0 for count in visual_private_expert_selection_counts
        ]
        mean_visual_private_selected_weights = [
            (
                visual_private_expert_selected_weight_sums[index]
                / visual_private_expert_selection_counts[index]
                if visual_private_expert_selection_counts[index] > 0
                else 0.0
            )
            for index in range(len(visual_private_expert_selection_counts))
        ]
        mean_visual_routing_scores = [
            (
                visual_private_expert_routing_score_sums[index] / total
                if total
                else 0.0
            )
            for index in range(len(visual_private_expert_routing_score_sums))
        ]
        most_selected_visual_private_expert_ids = sorted(
            range(len(visual_private_expert_selection_counts)),
            key=lambda index: visual_private_expert_selection_counts[index],
            reverse=True,
        )
        top_private_expert_ids = routed_private_experts.selected_ids
        return MoETTADomainEvalResult(
            env_name=domain_spec.env_name,
            target_domain=domain_spec.target_domain,
            source_domains=list(domain_spec.source_domains),
            num_classes=len(domain_spec.class_names),
            num_images=len(domain_spec.examples),
            effective_tta_lr=tta_lr,
            num_selected_samples=num_selected_samples,
            num_skipped_samples=num_skipped_samples,
            num_updates=num_updates,
            selected_ratio=selected_ratio,
            mean_confidence=mean_confidence,
            mean_selected_confidence=mean_selected_confidence,
            top1_accuracy=accuracy,
            mean_anchor_loss=mean_anchor_loss,
            num_shared_experts=len(self.shared_bank),
            num_private_experts=len(self.private_bank),
            shared_mix_weight=shared_mix_weight,
            routing_mode="task_fixed",
            selected_private_expert_ids=top_private_expert_ids,
            selected_private_expert_weights=[
                mean_private_selected_weights[index] for index in top_private_expert_ids
            ],
            routing_buffer_size=min(routing_buffer_size, len(domain_spec.examples)),
            routing_scores=mean_routing_scores,
            private_expert_selection_counts=private_expert_selection_counts,
            private_expert_selection_ratios=private_expert_selection_ratios,
            mean_private_selected_weights=mean_private_selected_weights,
            num_visual_prompts=self.num_visual_prompts if domain_visual_enabled else 0,
            num_visual_shared_experts=len(self.visual_shared_bank) if domain_visual_enabled else 0,
            num_visual_private_experts=len(self.visual_private_bank) if domain_visual_enabled else 0,
            visual_shared_mix_weight=visual_shared_mix_weight if domain_visual_enabled else 0.0,
            visual_routing_mode="sample_wise" if domain_visual_enabled else "off",
            visual_router_source=self.visual_router_source if domain_visual_enabled else "off",
            visual_domain_enabled=domain_visual_enabled,
            visual_topk_private_experts=(
                min(visual_topk_private_experts, len(self.visual_private_bank))
                if domain_visual_enabled and len(self.visual_private_bank) > 0
                else 0
            ),
            most_selected_visual_private_expert_ids=most_selected_visual_private_expert_ids,
            visual_private_expert_selection_counts=visual_private_expert_selection_counts,
            visual_private_expert_selection_ratios=visual_private_expert_selection_ratios,
            mean_visual_private_selected_weights=mean_visual_private_selected_weights,
            mean_visual_routing_scores=mean_visual_routing_scores,
            mean_teacher_confidence=mean_teacher_confidence,
            mean_selected_teacher_confidence=mean_selected_teacher_confidence,
            num_teacher_selected_samples=num_teacher_selected_samples,
            mean_kl_loss=mean_kl_loss,
            teacher_supervision_mode=teacher_supervision,
            teacher_student_preupdate_agreement=teacher_student_preupdate_agreement,
            mean_teacher_ce_loss=mean_teacher_ce_loss,
            num_teacher_guided_updates=num_teacher_guided_updates,
        )


def evaluate_domain_with_moe_prompt_tta(
    model,
    preprocess,
    domain_spec: OfficeHomeDomain,
    num_workers: int,
    device: str,
    tta_lr: float,
    num_context: int,
    prompt_init: str,
    pseudo_topk: int,
    update_steps: int,
    entropy_weight: float,
    confidence_threshold: float,
    shared_mix_weight: float,
    num_shared_experts: int,
    num_private_experts: int,
    topk_private_experts: int,
    routing_buffer_size: int,
    routing_score: str,
    router_temperature: float,
    expert_init_std: float,
    shared_anchor_weight: float,
    seed: int,
    num_visual_prompts: int = 4,
    visual_prompt_init_std: float = 0.0,
    visual_prompt_scale_init: float = 1.0,
    num_visual_shared_experts: int = 0,
    num_visual_private_experts: int = 0,
    visual_router_source: str = DEFAULT_MOE_VISUAL_ROUTER_SOURCE,
    disable_visual_domains: Sequence[str] | None = None,
    visual_topk_private_experts: int = 1,
    visual_shared_mix_weight: float = 0.5,
    visual_router_temperature: float = 1.0,
    teacher_supervision: str = "off",
    teacher_ce_weight: float = 0.25,
    teacher_model=None,
    teacher_preprocess=None,
    teacher_device: str | None = None,
    teacher_template: str = "a photo of a {class}",
    teacher_temperature: float = 1.0,
    teacher_confidence_threshold: float = 0.8,
    distill_kl_weight: float = 0.5,
    weight_decay: float = 0.0,
) -> MoETTADomainEvalResult:
    session = PersistentMoEPromptTTASession(
        model=model,
        preprocess=preprocess,
        reference_domain_spec=domain_spec,
        num_workers=num_workers,
        device=device,
        num_context=num_context,
        prompt_init=prompt_init,
        num_shared_experts=num_shared_experts,
        num_private_experts=num_private_experts,
        expert_init_std=expert_init_std,
        seed=seed,
        num_visual_prompts=num_visual_prompts,
        visual_prompt_init_std=visual_prompt_init_std,
        visual_prompt_scale_init=visual_prompt_scale_init,
        num_visual_shared_experts=num_visual_shared_experts,
        num_visual_private_experts=num_visual_private_experts,
        visual_router_source=visual_router_source,
        disable_visual_domains=disable_visual_domains,
        teacher_model=teacher_model,
        teacher_preprocess=teacher_preprocess,
        teacher_device=teacher_device,
        teacher_template=teacher_template,
        teacher_temperature=teacher_temperature,
    )
    return session.evaluate_domain(
        domain_spec=domain_spec,
        tta_lr=tta_lr,
        pseudo_topk=pseudo_topk,
        update_steps=update_steps,
        entropy_weight=entropy_weight,
        confidence_threshold=confidence_threshold,
        shared_mix_weight=shared_mix_weight,
        topk_private_experts=topk_private_experts,
        routing_buffer_size=routing_buffer_size,
        routing_score=routing_score,
        router_temperature=router_temperature,
        shared_anchor_weight=shared_anchor_weight,
        visual_topk_private_experts=visual_topk_private_experts,
        visual_shared_mix_weight=visual_shared_mix_weight,
        visual_router_temperature=visual_router_temperature,
        teacher_supervision=teacher_supervision,
        teacher_ce_weight=teacher_ce_weight,
        teacher_confidence_threshold=teacher_confidence_threshold,
        distill_kl_weight=distill_kl_weight,
        weight_decay=weight_decay,
    )
