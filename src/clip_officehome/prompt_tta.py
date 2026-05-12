"""Simple prompt-based test-time adaptation for CLIP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .clip_utils import (
    encode_text_from_embeddings,
    freeze_clip_parameters,
    import_clip,
    is_cuda_device,
)
from .evaluator import OfficeHomeImageDataset
from .officehome import OfficeHomeDomain

CLIP_LOGIT_SCALE = 100.0


class SharedPromptLearner(nn.Module):
    """Shared learnable context tokens prepended before each class name."""

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
            context_init = model.token_embedding(tokenized_prompt_init)[0, 1 : 1 + num_context]

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("prefix_embeddings", prompt_embeddings[:, :1, :])
        self.register_buffer("suffix_embeddings", prompt_embeddings[:, 1 + num_context :, :])
        self.register_buffer("initial_context", context_init.float().clone())
        self.context = nn.Parameter(self.initial_context.clone())

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.context.copy_(self.initial_context)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        shared_context = self.context.to(dtype=self.prefix_embeddings.dtype)
        shared_context = shared_context.unsqueeze(0).expand(
            self.tokenized_prompts.size(0),
            -1,
            -1,
        )
        prompt_embeddings = torch.cat(
            [self.prefix_embeddings, shared_context, self.suffix_embeddings],
            dim=1,
        )
        return prompt_embeddings, self.tokenized_prompts


@dataclass
class TTADomainEvalResult:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_prompt_only_training(model, prompt_learner: SharedPromptLearner) -> None:
    unexpected_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if unexpected_trainable:
        raise RuntimeError(f"CLIP parameters must be frozen during TTA: {unexpected_trainable[:5]}")

    trainable_prompt_params = [
        name for name, param in prompt_learner.named_parameters() if param.requires_grad
    ]
    if trainable_prompt_params != ["context"]:
        raise RuntimeError(
            "Simple prompt TTA expects only the shared context parameter to be trainable, "
            f"got: {trainable_prompt_params}"
        )


def build_prompt_learner(
    model,
    class_names: Sequence[str],
    prompt_init: str,
    num_context: int,
    device: str,
) -> SharedPromptLearner:
    prompt_learner = SharedPromptLearner(
        model=model,
        class_names=class_names,
        prompt_init=prompt_init,
        num_context=num_context,
        device=device,
    )
    prompt_learner.reset_parameters()
    return prompt_learner


def build_topk_soft_targets(probabilities: torch.Tensor, pseudo_topk: int) -> torch.Tensor:
    topk = min(max(int(pseudo_topk), 1), probabilities.size(-1))
    topk_values, topk_indices = probabilities.detach().topk(topk, dim=-1)

    soft_targets = torch.zeros_like(probabilities)
    soft_targets.scatter_(dim=-1, index=topk_indices, src=topk_values)
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return soft_targets.detach()


def compute_soft_target_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probabilities).sum(dim=-1).mean()


def compute_entropy_loss(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    return -(probabilities * (probabilities + eps).log()).sum(dim=-1).mean()


def evaluate_domain_with_prompt_tta(
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
    weight_decay: float = 0.0,
) -> TTADomainEvalResult:
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

    freeze_clip_parameters(model)
    use_cuda = is_cuda_device(device)

    dataset = OfficeHomeImageDataset(domain_spec.examples, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    prompt_learner = build_prompt_learner(
        model=model,
        class_names=domain_spec.prompt_class_names,
        prompt_init=prompt_init,
        num_context=num_context,
        device=device,
    ).to(device)
    prompt_learner.train()
    assert_prompt_only_training(model, prompt_learner)

    optimizer = torch.optim.AdamW(
        [prompt_learner.context],
        lr=tta_lr,
        weight_decay=weight_decay,
    )

    total = 0
    correct = 0
    num_selected_samples = 0
    num_skipped_samples = 0
    num_updates = 0
    confidence_sum = 0.0
    selected_confidence_sum = 0.0

    progress = tqdm(
        dataloader,
        desc=f"TTA {domain_spec.target_domain}",
        leave=False,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=use_cuda)
        labels = labels.to(device, non_blocking=use_cuda)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        prompt_embeddings, tokenized_prompts = prompt_learner()
        text_features = encode_text_from_embeddings(model, prompt_embeddings, tokenized_prompts)
        logits = CLIP_LOGIT_SCALE * image_features @ text_features.t()
        probabilities = logits.softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        confidences = probabilities.max(dim=-1).values
        confidence = float(confidences.item())

        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        confidence_sum += confidence

        # Treat threshold=1.0 as an explicit no-update sentinel; otherwise a few
        # numerically saturated softmax outputs can appear as exactly 1.0.
        is_selected = confidence >= confidence_threshold if confidence_threshold < 1.0 else False

        if is_selected:
            num_selected_samples += labels.size(0)
            selected_confidence_sum += confidence
            soft_targets = build_topk_soft_targets(probabilities, pseudo_topk=pseudo_topk)

            for step_index in range(update_steps):
                if step_index == 0:
                    step_logits = logits
                else:
                    prompt_embeddings, tokenized_prompts = prompt_learner()
                    text_features = encode_text_from_embeddings(
                        model,
                        prompt_embeddings,
                        tokenized_prompts,
                    )
                    step_logits = CLIP_LOGIT_SCALE * image_features @ text_features.t()

                ce_loss = compute_soft_target_cross_entropy(step_logits, soft_targets)
                if entropy_weight > 0.0:
                    entropy_loss = compute_entropy_loss(step_logits)
                    loss = ce_loss + entropy_weight * entropy_loss
                else:
                    loss = ce_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                num_updates += 1
        else:
            num_skipped_samples += labels.size(0)

    accuracy = correct / total if total else 0.0
    selected_ratio = num_selected_samples / total if total else 0.0
    mean_confidence = confidence_sum / total if total else 0.0
    mean_selected_confidence = (
        selected_confidence_sum / num_selected_samples if num_selected_samples else 0.0
    )
    return TTADomainEvalResult(
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
    )
