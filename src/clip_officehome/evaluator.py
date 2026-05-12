"""Zero-shot evaluation loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .clip_utils import build_text_features, is_cuda_device
from .officehome import ImageExample, OfficeHomeDomain


class OfficeHomeImageDataset(Dataset):
    def __init__(self, examples: tuple[ImageExample, ...], preprocess):
        self.examples = examples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        with Image.open(example.image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.preprocess(image)
        return image_tensor, example.label_index


class OfficeHomeDualImageDataset(Dataset):
    def __init__(self, examples: tuple[ImageExample, ...], student_preprocess, teacher_preprocess):
        self.examples = examples
        self.student_preprocess = student_preprocess
        self.teacher_preprocess = teacher_preprocess

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        with Image.open(example.image_path) as image:
            image = image.convert("RGB")
            student_image_tensor = self.student_preprocess(image)
            teacher_image_tensor = self.teacher_preprocess(image)
        return student_image_tensor, teacher_image_tensor, example.label_index


@dataclass
class DomainEvalResult:
    env_name: str
    target_domain: str
    source_domains: list[str]
    num_classes: int
    num_images: int
    top1_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def evaluate_domain(
    model,
    preprocess,
    domain_spec: OfficeHomeDomain,
    batch_size: int,
    num_workers: int,
    template: str,
    device: str,
) -> DomainEvalResult:
    use_cuda = is_cuda_device(device)
    text_features = build_text_features(
        model=model,
        class_names=domain_spec.prompt_class_names,
        template=template,
        device=device,
    )

    dataset = OfficeHomeImageDataset(domain_spec.examples, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    total = 0
    correct = 0

    progress = tqdm(
        dataloader,
        desc=f"Evaluating {domain_spec.target_domain}",
        leave=False,
    )
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        predictions = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    accuracy = correct / total if total else 0.0
    return DomainEvalResult(
        env_name=domain_spec.env_name,
        target_domain=domain_spec.target_domain,
        source_domains=list(domain_spec.source_domains),
        num_classes=len(domain_spec.class_names),
        num_images=len(domain_spec.examples),
        top1_accuracy=accuracy,
    )


def save_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
