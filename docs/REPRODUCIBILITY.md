# Reproducing MT-TTA

[Home](../README.md) · [Data and weights](../DATA_DEPENDENCIES.md) · [Reference results](RESULTS.md)

This guide follows the path from a clean source checkout to a completed result. The fixed protocol uses seed **0**, full input streams, frozen CLIP backbones and **student-only predictions before the current image's update**.

## 1. Pin the code and environment

Use the source checkout for the complete workflow, including dataset preparation and report scripts. Installing only the wheel provides the PACS/OfficeHome and ImageNet evaluation commands, but does not install the repository's `scripts/` or benchmark assembly assets.

```bash
git clone --branch v0.1.2 https://github.com/ye1457/MT-TTA.git
cd MT-TTA
git rev-parse HEAD
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-runtime.txt
python -m pip install -e .
python -m pip check
```

The reference machine uses Linux, Python **3.11.15**, PyTorch **2.7.1+cu118**, torchvision **0.22.1+cu118**, and the versions in [`requirements-runtime.txt`](../requirements-runtime.txt). The torch/torchvision pairing follows [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v271). Use a compatible NVIDIA driver. The CUDA wheel includes its runtime; this project does not compile a custom CUDA extension.

The package accepts Python >=3.10, but the pinned reproduction requirements need Python >=3.11. The supported installation range is broader than the numerically checked reference environment. Different GPUs, numerical libraries or package versions can change finite-precision predictions; matching seed0 alone does not guarantee bitwise identity.

<details>
<summary><strong>CPU checks without CUDA</strong></summary>

In a separate Python 3.11 virtual environment, replace the CUDA installation command with:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-runtime.txt
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

These checks exercise software contracts. Full method evaluation requires CUDA and does not silently fall back to the CPU.

</details>

## 2. Prepare the exact inputs

Follow [data preparation](../DATA_DEPENDENCIES.md) for dataset sources, original folder names and the two official CLIP weight files. Use complete PACS, OfficeHome and ImageNet releases. Do not split them into source/target training sets, resize/re-encode images, rename files, or generate a random 80% subset.

| Evaluation | Images | Class space | Data root contains |
| :--- | ---: | :--- | :--- |
| PACS | 9,991 | 7 | `art_painting`, `cartoon`, `photo`, `sketch` |
| OfficeHome | 15,588 | 65 | `Art`, `Clipart`, `Product`, `Real_World` |
| ImageNet-A | 7,500 | 200 | Official WNID folders |
| ImageNet-V2 | 10,000 | 1,000 | Numeric canonical indices `0` through `999`; matched-frequency release |
| ImageNet-R | 30,000 | 200 | Official WNID folders |
| ImageNet-Sketch | 50,889 | 1,000 | Official WNID folders |
| MT-Sketch4-20K | 20,000 | Four native vocabularies | Frozen manifest, metadata and `images/` tree |

For the custom 20K benchmark, use the [frozen membership and assembly tool](../benchmarks/mt_sketch4/README.md). The published metadata defines the selected images and their exact byte hashes; images remain separate downloads under their original terms. Assembly copies those members, without drawing a new subset or changing labels.

Store `ViT-B-16.pt` and `ViT-L-14.pt` in `models/`. Verify downloads before running:

```bash
sha256sum --check <<'SHA256'
5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f  models/ViT-B-16.pt
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836  models/ViT-L-14.pt
SHA256
```

The method requires both weights. A frozen zero-shot control uses only B/16. The evaluator checks cached weights; it does not download missing files as part of a formal run.

## 3. Check the data before allocating GPU time

```bash
mt-tta-evaluate --benchmark pacs --data-root /path/to/PACS --dry-run --hash-images
mt-tta-evaluate --benchmark officehome --data-root /path/to/OfficeHome --dry-run --hash-images
mt-tta-evaluate-imagenet --benchmark imagenet_a --data-root /path/to/imagenet-a --dry-run --hash-images
python scripts/evaluate_mtsketch.py --data-root /path/to/MT-Sketch4-20K --dry-run
```

Repeat the ImageNet command with `imagenet_v2`, `imagenet_r` and `imagenet_sketch` and their own roots. A successful preflight prints `DRY_RUN_PASS_NO_MODEL`. It does not create formal outputs or load model weights.

For PACS/OfficeHome/ImageNet, plain `--dry-run` checks class directories, full counts and order; adding `--hash-images` also reads and hashes every image. Neither option decodes every image or proves a GPU run will finish. Compare the returned ordered index/content hashes with [`reference_inputs.json`](reference_inputs.json). The reference excludes machine-specific absolute paths. A matching count with a different content/order hash is insufficient to claim the same input stream.

The MT-Sketch adapter always checks the five frozen metadata digests. Its ordinary dry run validates membership and paths without hashing every image; the assembly tool verifies the copied image bytes, and formal evaluation verifies them again.

## 4. Run complete streams

Run from this checkout with its virtual environment activated. Replace data paths and select an available physical GPU index. Each command is a separate fresh benchmark session.

```bash
mt-tta-evaluate --benchmark pacs --data-root /path/to/PACS \
  --model-cache models --gpu 0 --output-dir outputs/pacs_seed0

mt-tta-evaluate --benchmark officehome --data-root /path/to/OfficeHome \
  --model-cache models --gpu 0 --output-dir outputs/officehome_seed0
```

Use [the ImageNet commands](IMAGENET_OOD.md) for A/V2/R/Sketch and their frozen controls, or [the multi-task commands](MT_SKETCH4_EVALUATION.md) for the 20K stream. A method/control pair must use the **same data-root path**, not two copies of the same files. The report validators compare the recorded manifests as well as image bytes.

Use a new output directory for every run. The evaluators have no midstream checkpoint/resume facility. Do not modify runtime source, data or model files during evaluation. Keep data and weight paths in place until report generation finishes; the MT-Sketch report rechecks the recorded input and model locations.

The OfficeHome command runs its predeclared primary/shadow schedule automatically. Do not run four domains separately and combine their best scores. ImageNet benchmarks initialize independently; MT-Sketch tasks share prompt and optimizer state. These differences are intentional and documented in their protocols.

## 5. Check completion and read the metric

For PACS or OfficeHome:

```bash
python -m json.tool outputs/pacs_seed0/summary.json
python -m json.tool outputs/pacs_seed0/completion.json
```

| Artifact / field | Meaning |
| :--- | :--- |
| `completion.json` → `status: "PASS"` | Run finished and the evaluator's artifact/trace checks passed |
| `summary.json` → `status: "PASS"` | Recounted summary is complete |
| `summary.json` → `macro_top1_percent` | PACS/OH equal-domain mean; single-benchmark accuracy for ImageNet |
| `metadata.json` | Model/source hashes, environment, routing and prompt/optimizer observations |
| `input_manifest.json` | Exact members, labels for scoring, ordering and image hashes |
| `*.predictions.jsonl` | Pre-update student predictions and cumulative update counts |

MT-Sketch also records a separate `updates.jsonl`; its method summary uses `macro_accuracy_percent`, while its zero-shot summary uses `macro_top1_percent`. Its four tasks are equal-sized, so the macro and pooled accuracies coincide. ImageNet's four-benchmark mean gives each dataset weight 1/4 regardless of its size. Detailed report generation is in [ImageNet reporting](IMAGENET_OOD.md) and [MT-Sketch reporting](MT_SKETCH4_EVALUATION.md).

`PASS` certifies the implemented consistency checks, not equality to a published accuracy. A progress message, partial predictions file or failed/interrupted run is not a completed result. Retain your complete output directory if later independent recounting is required.

## Verification scope

The 2026-10-29 reproduction review checked the real full input collections for all seven evaluation entries, fixed model digests, pinned dependencies, public source installation, CPU tests and command-line interfaces. Public CI additionally builds and installs the package outside the checkout. The input reference records come from byte-level preflight checks, **not a new GPU accuracy measurement**.

The values in [RESULTS.md](RESULTS.md) are historical completed runs. Historical raw predictions and output backups were deleted at the owner's request. This release preserves their method configurations and dataset identities; it does not claim that a documentation/package review reran every benchmark or proves identical accuracy on every platform.

## Common problems

| Symptom | Action |
| :--- | :--- |
| Missing class folders / wrong image counts | Point to the correct complete image root; preserve original folders and filenames. Do not bypass the full-set checks. |
| Content/order hashes differ | Check dataset edition, renamed/re-encoded images and extraction layout against the pinned data references. |
| Missing or mismatched CLIP weights | Download both official files and run the SHA256 check above. |
| Wrong `clip` module | Use a clean environment and install this checkout; the unrelated PyPI package named `clip` is not the vendored OpenAI implementation. |
| CUDA unavailable / out of memory | Check the CUDA torch build, driver and selected GPU's available memory. Formal evaluation requires both backbones and adaptation state. |
| Output directory already exists | Choose a new directory; partial runs are not resumed or silently overwritten. |
| Report source/manifest mismatch | Run method, control and summarizer from the same commit, against the same data path; keep recorded resources in place. |
