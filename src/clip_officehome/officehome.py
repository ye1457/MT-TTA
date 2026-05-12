"""OfficeHome dataset parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .constants import IMAGE_EXTENSIONS, STANDARD_DOMAINS


@dataclass(frozen=True)
class ImageExample:
    image_path: Path
    label_index: int
    class_name: str


@dataclass(frozen=True)
class OfficeHomeDomain:
    env_name: str
    env_path: Path
    source_domains: tuple[str, ...]
    target_domain: str
    domain_path: Path
    class_names: tuple[str, ...]
    prompt_class_names: tuple[str, ...]
    examples: tuple[ImageExample, ...]


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _sorted_dirs(paths: Iterable[Path]) -> list[Path]:
    return sorted((path for path in paths if path.is_dir()), key=lambda item: item.name)


def normalize_class_name(class_name: str) -> str:
    return class_name.replace("_", " ").strip()


def list_env_directories(split_root: Path) -> list[Path]:
    split_root = Path(split_root)
    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")
    if not split_root.is_dir():
        raise NotADirectoryError(f"Split root is not a directory: {split_root}")

    env_dirs = _sorted_dirs(split_root.glob("Officehome_env*_source"))
    if not env_dirs:
        raise FileNotFoundError(
            f"No OfficeHome split directories were found under: {split_root}"
        )
    return env_dirs


def get_source_domains(env_dir: Path) -> tuple[str, ...]:
    env_dir = Path(env_dir)
    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory does not exist: {env_dir}")
    if not env_dir.is_dir():
        raise NotADirectoryError(f"Environment path is not a directory: {env_dir}")

    source_domains = sorted(
        child.name
        for child in env_dir.iterdir()
        if child.name in STANDARD_DOMAINS and (child.is_dir() or child.is_symlink())
    )
    if not source_domains:
        raise ValueError(f"No source domains found under: {env_dir}")
    return tuple(source_domains)


def infer_target_domain(env_dir: Path) -> str:
    source_domains = set(get_source_domains(env_dir))
    missing_domains = sorted(set(STANDARD_DOMAINS) - source_domains)
    if len(missing_domains) != 1:
        raise ValueError(
            f"Expected exactly one target domain for {env_dir}, got: {missing_domains}"
        )
    return missing_domains[0]


def collect_domain_examples(domain_path: Path) -> tuple[tuple[str, ...], tuple[ImageExample, ...]]:
    domain_path = Path(domain_path)
    if not domain_path.exists():
        raise FileNotFoundError(f"Target domain directory does not exist: {domain_path}")
    if not domain_path.is_dir():
        raise NotADirectoryError(f"Target domain path is not a directory: {domain_path}")

    class_dirs = _sorted_dirs(domain_path.iterdir())
    if not class_dirs:
        raise ValueError(f"No class directories found under: {domain_path}")

    class_names = tuple(class_dir.name for class_dir in class_dirs)
    examples: list[ImageExample] = []

    for label_index, class_dir in enumerate(class_dirs):
        image_files = sorted(
            (path for path in class_dir.rglob("*") if _is_image_file(path)),
            key=lambda item: item.as_posix(),
        )
        if not image_files:
            raise ValueError(f"No image files found in class directory: {class_dir}")
        for image_path in image_files:
            examples.append(
                ImageExample(
                    image_path=image_path,
                    label_index=label_index,
                    class_name=class_dir.name,
                )
            )

    return class_names, tuple(examples)


def build_domain_spec(env_dir: Path, raw_root: Path) -> OfficeHomeDomain:
    env_dir = Path(env_dir)
    raw_root = Path(raw_root)

    source_domains = get_source_domains(env_dir)
    target_domain = infer_target_domain(env_dir)
    domain_path = raw_root / target_domain
    class_names, examples = collect_domain_examples(domain_path)

    return OfficeHomeDomain(
        env_name=env_dir.name,
        env_path=env_dir,
        source_domains=source_domains,
        target_domain=target_domain,
        domain_path=domain_path,
        class_names=class_names,
        prompt_class_names=tuple(normalize_class_name(name) for name in class_names),
        examples=examples,
    )


def build_all_domain_specs(split_root: Path, raw_root: Path) -> list[OfficeHomeDomain]:
    return [build_domain_spec(env_dir, raw_root) for env_dir in list_env_directories(split_root)]


def get_domain_spec_by_target(
    split_root: Path,
    raw_root: Path,
    target_domain: str,
) -> OfficeHomeDomain:
    if target_domain not in STANDARD_DOMAINS:
        raise ValueError(
            f"Unsupported target domain '{target_domain}'. Expected one of: {STANDARD_DOMAINS}"
        )

    for spec in build_all_domain_specs(split_root, raw_root):
        if spec.target_domain == target_domain:
            return spec

    raise ValueError(f"Could not find a split mapping for target domain: {target_domain}")
