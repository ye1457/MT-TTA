"""Reconstruct the frozen 20K benchmark from separately obtained source images.

No images are downloaded, sampled, decoded or re-encoded. Every copied image
must match the frozen member's byte count and SHA256. PyArrow is needed only
when using the optional TU-Berlin Parquet input instead of extracted PNGs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clip_officehome.multitask_sketch import FROZEN, TASKS, prepare_mtsketch, validate_structure


DEFAULT_METADATA = Path(__file__).resolve().parents[1] / "benchmarks/mt_sketch4/frozen_1.0-review.tar.gz"


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or "\\" in value or "\0" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe relative path: " + repr(value))
    if not path.parts or ":" in path.parts[0]:
        raise ValueError("Unsafe relative path: " + repr(value))
    return path


def load_metadata(archive):
    """Read only the five pinned regular files; never extract archive paths."""
    payload = {}
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if len(members) != len(FROZEN) or {m.name for m in members} != set(FROZEN):
            raise ValueError("Metadata archive must contain exactly the five frozen files")
        for member in members:
            safe_relative(member.name)
            if not member.isfile() or member.size > 32 * 1024 * 1024:
                raise ValueError("Invalid frozen metadata member: " + member.name)
            content = handle.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != FROZEN[member.name]:
                raise ValueError("Frozen metadata hash mismatch: " + member.name)
            payload[member.name] = content
    records = [json.loads(line) for line in payload["manifest.jsonl"].splitlines() if line.strip()]
    classes = json.loads(payload["metadata/classes.json"])
    order = json.loads(payload["streams.json"])["balanced_round_robin"]
    validate_structure(records, classes, order)
    return payload, records


def source_candidates(record):
    """Accept original archives' roots or their image-bearing subdirectories."""
    relative = record["relative_path"]
    candidates = [relative]
    if record["task_id"] in {"domainnet_sketch", "imagenet_sketch", "tuberlin"}:
        candidates.append(str(PurePosixPath(relative).relative_to(PurePosixPath(relative).parts[0])))
    else:
        original = record["source_relative_path"]
        mirror = record["mirror_archive_member"]
        candidates += [original, mirror]
        for prefix in ("256x256/", "256x256/sketch/", "256x256/sketch/tx_000000000000/"):
            if original.startswith(prefix):
                candidates.append(original[len(prefix):])
        if mirror.startswith("sketchy/"):
            candidates.append(mirror[len("sketchy/"):])
        if relative.startswith("images/"):
            candidates.append(relative[len("images/"):])
    return tuple(dict.fromkeys(str(safe_relative(path)) for path in candidates))


def check_bytes(record, content):
    return (len(content) == record["file_bytes"] and
            hashlib.sha256(content).hexdigest() == record["file_sha256"])


def write_image(destination, record, content):
    if not check_bytes(record, content):
        raise ValueError("Source image bytes differ from frozen membership: " + record["sample_id"])
    path = destination / safe_relative(record["image_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def copy_from_tree(source_root, destination, records):
    source_root = Path(source_root).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError("Source directory does not exist: " + str(source_root))
    for record in records:
        found = False
        for relative in source_candidates(record):
            source = source_root / relative
            if not source.is_file():
                continue
            resolved = source.resolve()
            if source_root not in resolved.parents:
                raise ValueError("Source image resolves outside its supplied root: " + relative)
            if source.stat().st_size != record["file_bytes"]:
                continue
            content = source.read_bytes()
            if check_bytes(record, content):
                write_image(destination, record, content)
                found = True
                break
        if not found:
            raise ValueError("Missing or byte-mismatched source image: " + record["sample_id"] +
                             "; expected one of " + ", ".join(source_candidates(record)))


def copy_tuberlin_parquet(source_root, destination, records):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("TU-Berlin Parquet input needs optional pyarrow; install it or use --tuberlin-root") from error
    source_root = Path(source_root).expanduser().resolve()
    grouped = {}
    for record in records:
        grouped.setdefault(record["source_parquet"], {})[record["source_row"]] = record
    for filename, selected in grouped.items():
        path = source_root / safe_relative(filename)
        if not path.is_file() or source_root not in path.resolve().parents:
            raise ValueError("Missing/nonlocal TU-Berlin Parquet file: " + filename)
        copied, offset = set(), 0
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=128, columns=["image"]):
            for local_index, image in enumerate(batch.column(0).to_pylist()):
                index = offset + local_index
                if index not in selected:
                    continue
                if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
                    raise ValueError("Expected embedded original image bytes in TU-Berlin Parquet")
                write_image(destination, selected[index], image["bytes"])
                copied.add(index)
            offset += batch.num_rows
        if copied != set(selected):
            raise ValueError("TU-Berlin Parquet is missing frozen row indices: " + filename)


def reconstruct(metadata, records, roots, output, tuberlin_parquet=None):
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a new directory: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mt-sketch4-prepare-", dir=output.parent))
    try:
        for name, content in metadata.items():
            path = staging / safe_relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        for task in TASKS:
            selected = [r for r in records if r["task_id"] == task]
            if task == "tuberlin" and tuberlin_parquet is not None:
                copy_tuberlin_parquet(tuberlin_parquet, staging, selected)
            else:
                copy_from_tree(roots[task], staging, selected)
            print(json.dumps({"task": task, "copied_and_hash_verified": len(selected)}), flush=True)
        prepared = prepare_mtsketch(staging, hash_images=True)
        # Reserve the destination without replacing even an empty existing directory.
        output.mkdir()
        try:
            for child in staging.iterdir():
                shutil.move(str(child), output / child.name)
        except BaseException:
            shutil.rmtree(output)
            raise
        return {"status": "PREPARE_PASS", "output": str(output),
                "images": prepared.inventory["num_images"], "frozen_artifacts": FROZEN,
                "ordered_content_sha256": prepared.inventory["ordered_content_sha256"]}
    finally:
        shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--domainnet-root", type=Path, required=True,
                        help="Root containing sketch/<class>/<file> or directly <class>/<file>")
    parser.add_argument("--imagenet-sketch-root", type=Path, required=True,
                        help="Root containing sketch/<WNID>/<file> or directly <WNID>/<file>")
    tuberlin = parser.add_mutually_exclusive_group(required=True)
    tuberlin.add_argument("--tuberlin-root", type=Path, help="Root containing png/<class>/<file> or <class>/<file>")
    tuberlin.add_argument("--tuberlin-parquet-dir", type=Path, help="Directory containing the three pinned mirror Parquet files")
    parser.add_argument("--sketchy-root", type=Path, required=True,
                        help="Extracted author/mirror root, tx_000000000000 directory, or root containing images/<class>/<file>")
    parser.add_argument("--output-dir", type=Path, required=True, help="New frozen bundle directory")
    args = parser.parse_args(argv)
    try:
        metadata, records = load_metadata(args.metadata)
        roots = {"domainnet_sketch": args.domainnet_root, "imagenet_sketch": args.imagenet_sketch_root,
                 "tuberlin": args.tuberlin_root, "sketchy": args.sketchy_root}
        print(json.dumps(reconstruct(metadata, records, roots, args.output_dir, args.tuberlin_parquet_dir), indent=2))
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        parser.exit(2, "Preparation failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
