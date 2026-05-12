"""OpenAI CLIP loading and feature helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Sequence

import torch

from .constants import DEFAULT_CLIP_CACHE_DIR


def import_clip():
    try:
        import clip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is not installed. Install the dependencies from requirements.txt "
            "after installing a compatible torch/torchvision build."
        ) from exc
    return clip


def is_cuda_device(device: str) -> bool:
    normalized = str(device).strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def resolve_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if is_cuda_device(normalized) and not torch.cuda.is_available():
        reason = (
            f"the installed torch build ({torch.__version__}) does not include CUDA support"
            if torch.version.cuda is None
            else "CUDA is not available on this machine"
        )
        warnings.warn(
            f"Requested device '{device}', but {reason}. Falling back to 'cpu'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return normalized


def _get_model_cache_path(clip_module, model_name: str, cache_dir: Path) -> Path | None:
    model_registry = _get_model_registry(clip_module)
    if model_registry is None:
        return None
    model_url = model_registry.get(model_name)
    if not model_url:
        return None
    return cache_dir / os.path.basename(model_url)


def _get_model_registry(clip_module):
    model_registry = getattr(clip_module, "_MODELS", None)
    if model_registry is None:
        nested_clip_module = getattr(clip_module, "clip", None)
        model_registry = getattr(nested_clip_module, "_MODELS", None)
    return model_registry


def _get_model_url(clip_module, model_name: str) -> str | None:
    model_registry = _get_model_registry(clip_module)
    if model_registry is None:
        return None
    return model_registry.get(model_name)


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_aria2(model_url: str, cache_dir: Path) -> Path:
    aria2c_path = shutil.which("aria2c")
    if aria2c_path is None:
        raise RuntimeError("aria2c is not available for resilient CLIP model download.")

    download_target = cache_dir / os.path.basename(model_url)
    expected_sha256 = model_url.split("/")[-2]
    if download_target.exists():
        download_target.unlink()

    subprocess.run(
        [
            aria2c_path,
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "-d",
            str(cache_dir),
            "-o",
            download_target.name,
            model_url,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if _sha256_file(download_target) != expected_sha256:
        raise RuntimeError(
            f"aria2c downloaded '{download_target.name}', but the SHA256 checksum did not match."
        )
    return download_target


def load_clip_model(model_name: str, device: str, cache_dir: str | Path | None = None):
    clip = import_clip()
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CLIP_CACHE_DIR
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, preprocess = clip.load(
            model_name,
            device=device,
            download_root=str(resolved_cache_dir),
        )
    except RuntimeError as exc:
        if "SHA256 checksum" not in str(exc):
            raise
        cached_model_path = _get_model_cache_path(clip, model_name, resolved_cache_dir)
        model_url = _get_model_url(clip, model_name)
        if cached_model_path is not None and cached_model_path.exists():
            cached_model_path.unlink()
        if model_url is not None:
            try:
                _download_with_aria2(model_url, resolved_cache_dir)
            except Exception:
                if cached_model_path is None or not cached_model_path.exists():
                    raise
        model, preprocess = clip.load(
            model_name,
            device=device,
            download_root=str(resolved_cache_dir),
        )
    model.eval()
    return model, preprocess


def freeze_clip_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()


@torch.no_grad()
def build_text_features(
    model,
    class_names: Sequence[str],
    template: str,
    device: str,
) -> torch.Tensor:
    clip = import_clip()
    prompts = [template.format(class_name=name, **{"class": name}) for name in class_names]
    tokens = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def encode_text_from_embeddings(
    model,
    prompt_embeddings: torch.Tensor,
    tokenized_prompts: torch.Tensor,
) -> torch.Tensor:
    x = prompt_embeddings.to(dtype=model.dtype)
    x = x + model.positional_embedding.to(x.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    eot_indices = tokenized_prompts.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=tokenized_prompts.device), eot_indices]
    x = x @ model.text_projection
    x = x / x.norm(dim=-1, keepdim=True)
    return x
