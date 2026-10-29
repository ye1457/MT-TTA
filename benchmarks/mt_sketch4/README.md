# MT-Sketch4-20K frozen membership

The accompanying `frozen_1.0-review.tar.gz` contains **metadata and a loader only**,
not images. It supplies the exact membership needed by the public evaluation
adapter, without rerunning sampling or choosing new examples. Acquire the source
images separately under their respective terms.

| Frozen file | Purpose |
| --- | --- |
| `manifest.jsonl` | 20,000 source/target identities, native labels, review flags and image hashes |
| `metadata/classes.json` | Four native class spaces and task-local prediction indices |
| `metadata/protocol.json` | Dataset version and stream assumptions |
| `streams.json` | Fixed balanced round-robin and task-block indices |
| `benchmark.py` | Frozen data loader and external scorer |

The five files retain their original bytes and SHA256 values pinned in
[`multitask_sketch.py`](../../src/clip_officehome/multitask_sketch.py). The archive
uses deterministic gzip/tar headers; its SHA256 is
`df440fb595781e4f35da3e34fc488a1b206603de0dfa086f624737fe44db8975`.
It does not include local acquisition logs, private machine paths or model outputs.
The native label ambiguity/review flags remain part of version `1.0-review`.

## Reconstruct from extracted sources

Install the repository, obtain the four sources, then run from the checkout:

```bash
python scripts/prepare_mtsketch.py \
  --domainnet-root /path/to/DomainNet \
  --imagenet-sketch-root /path/to/imagenet-sketch \
  --tuberlin-root /path/to/TU-Berlin \
  --sketchy-root /path/to/Sketchy \
  --output-dir data/MT-Sketch4-20K

python scripts/evaluate_mtsketch.py --data-root data/MT-Sketch4-20K --dry-run
```

Accepted source layouts are:

| Argument | Images beneath that root |
| --- | --- |
| `--domainnet-root` | `sketch/<class>/<filename>` or `<class>/<filename>` |
| `--imagenet-sketch-root` | `sketch/<WNID>/<filename>` or `<WNID>/<filename>` |
| `--tuberlin-root` | `png/<class>/<numeric ID>.png` or `<class>/<numeric ID>.png` |
| `--sketchy-root` | Author `256x256/sketch/tx_000000000000/<class>/<filename>`; mirror `sketchy/sketch/tx_000000000000/<class>/<filename>`; the `tx_000000000000` directory itself; or prepared `images/<class>/<filename>` |

The tool checks each member's original byte count and SHA256 before copying.
Images are not decoded, resized, re-encoded, relabeled or resampled. It creates
independent copies, never hard links, and needs roughly 1.6 GB in addition to the
source downloads. Use a new output directory: existing destinations are refused.
If preparation fails, its temporary staging directory is removed. A successful
run prints `PREPARE_PASS` after verifying all 20,000 images and all five metadata
hashes through the same adapter used for evaluation.

## Source downloads and TU-Berlin Parquet option

- **DomainNet Sketch:** use the original [DomainNet](https://ai.bu.edu/M3SDA/)
  sketch archive. The frozen members come only from the official cleaned test
  list. Keep the original filenames and bytes.
- **ImageNet-Sketch:** use the author-linked
  [fixed mirror revision](https://huggingface.co/datasets/songweig/imagenet_sketch/tree/0d8b8b5380efe50baae075842505918ad43c88fe).
- **Sketchy:** use the [author image archive](https://patsorn.me/projects/sketchy/)
  or the [fixed mirror revision](https://huggingface.co/datasets/DrRORAL/sketchy-dataset/tree/dcbdf00ab09409b4244d9fe717e652274b3fe648).
  Only original geometry `tx_000000000000` is accepted. Frozen file SHA256 checks
  enforce the actual images used; visually similar re-encoded copies will fail.
- **TU-Berlin:** the reference PNG bytes came from the
  [fixed mirror revision](https://huggingface.co/datasets/sdiaeyu6n/tu-berlin/tree/e7c6c0dedcf2aa4ca060878893bca8d77ef699f4).
  Original author PNG archive byte equivalence is not established. To use the
  verified mirror directly, obtain all three files in that revision's `data/`
  directory: `train-00000-of-00001.parquet`, `validation-00000-of-00001.parquet`,
  and `test-00000-of-00001.parquet`. Install optional `pyarrow`, then replace
  `--tuberlin-root` in the command above with
  `--tuberlin-parquet-dir /path/to/the/three/parquet/files`.

```bash
python -m pip install pyarrow==26.0.0  # tested optional TU-Berlin Parquet dependency
```

The Parquet option selects the frozen file/row indices and writes the embedded
original PNG bytes. It does not use Pillow encoding. It validates the selected
bytes against the same per-image hashes; the arbitrary mirror partitions do not
become the benchmark's task splits. No other extra dependency is needed for the
extracted-image preparation route.

## Scope and terms

This is deterministic reconstruction of the already selected benchmark. The
original exploratory selection/review pipeline and image archives are not
distributed here. Dataset composition and scientific caveats are described in
the [dataset guide](../../docs/MT_SKETCH4_DATASET.md); evaluation is described in
the [protocol](../../docs/MT_SKETCH4_EVALUATION.md).

The project's MIT license covers its code and authored assembly metadata. The
source images and their original annotations retain their original terms; this
release grants no new image redistribution rights. Metadata includes native
labels and pseudonymous hashed Sketchy artist IDs for the frozen construction
record. No raw participant identifiers or image pixels are included.
