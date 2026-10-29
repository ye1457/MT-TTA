# MT-TTA

Continual CLIP test-time adaptation with shared and private text/visual prompt experts. The source tree supports PACS, OfficeHome, the four ImageNet natural-shift benchmarks, and the task-known MT-Sketch4-20K stream. Historical experiments and local research archives are not runtime dependencies.

Version **0.1.1** is the development baseline. Continue development in this source tree; installed packages and exported source archives are snapshots. Local `.venv/`, `models/`, `data/`, `outputs/` and `reports/`, if present, are development assets excluded from distributions.

The student is frozen OpenAI CLIP ViT-B/16. A frozen ViT-L/14 teacher supplies an auxiliary hard-pseudo-label loss. The enabled objective is student hard-pseudo-label CE plus **0.08 × teacher hard-pseudo-label CE**, both applied to student outputs. **Reported predictions are student-only; teacher output fusion and calibration are excluded.**

## Installation

Use Python >=3.10. Reference runs used Python 3.11, torch 2.7.1+cu118 and torchvision 0.22.1+cu118. Start from a source checkout:

```bash
git clone https://github.com/ye1457/MT-TTA.git
cd MT-TTA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a compatible torch/torchvision pair for your CUDA environment. The reference CUDA 11.8 builds can be installed with:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-runtime.txt
python -m pip install -e .
```

`requirements-runtime.txt` records the reference Python 3.11 dependency versions. Its NumPy pin requires Python 3.11; on Python 3.10 omit the requirements-file command and use the package dependency ranges from `pip install -e .`. Python 3.10 dependency compatibility is declared, but this repository's installation and numerical checks were run on Python 3.11.

The vendored OpenAI `clip` module and tokenizer are included; do not install the unrelated PyPI package named `clip`. Formal evaluation requires CUDA. Dry-run validation does not load models. Weights and datasets are obtained separately and are not packaged.

## Evaluation

Data roots use `domain/class/image` folders. No historical split/order JSON or experiment report is needed.

```bash
mt-tta-evaluate --benchmark pacs --data-root /path/to/PACS --dry-run
mt-tta-evaluate --benchmark officehome --data-root /path/to/OfficeHome --dry-run
mt-tta-evaluate --benchmark pacs --data-root /path/to/PACS --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/pacs_seed0
mt-tta-evaluate --benchmark officehome --data-root /path/to/OfficeHome --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/officehome_seed0
```

The equivalent source-checkout command is `python scripts/evaluate.py`. Existing output directories and incorrect full-data counts are rejected. Cache the official model files before formal evaluation; the evaluator checks their digests and does not silently download weights. The default cache is `~/.cache/clip`; retained local weights can be used with `--model-cache models`. See [data preparation](DATA_DEPENDENCIES.md) and [evaluation details](docs/EVALUATION.md).

Use a GPU with enough available memory for both backbones and adaptation; memory requirements vary by benchmark and class count. `--gpu` is one physical CUDA GPU index. CUDA is unnecessary for `--dry-run` and the CPU unit tests. A successful formal run creates `summary.json` and `completion.json` with status `PASS`; see [output records and restart behavior](docs/EVALUATION.md#metrics-and-records). There is no midstream resume support.

| Setting | PACS | OfficeHome |
| --- | --- | --- |
| Full images / classes | 9,991 / 7 | 15,588 / 65 |
| Domain order | art_painting → cartoon → photo → sketch | Art → Clipart → Product → Real_World |
| Fixed text route | First64, mean probability margin, Top-2 | First64, mean maximum class probability, Top-2 |
| Student confidence gate | 0.95 | 0.815 |
| Learning rate | 7.5e-6 | 7.5e-6; Clipart 2e-6 |
| Reported branch | Primary in every domain | Primary first three; text shadow Real_World |
| Audited reference macro Top-1 | 98.997603% | 86.488202% |

Both use seed0, all images sorted by class-directory/path, batch1 and scoring before the current image update. Prompts persist across domains; optimizers restart. Unlabeled previews perform no updates and are included once in formal scoring. This is a known-domain protocol with preview lookahead, not strict no-lookahead online evaluation.

PACS was selected on this same full stream from 24 configurations. A separate seed0 repeat matched all predictions, probability/logit values and domain-boundary prompt states. Completed reference results and their evaluation conditions are documented in [RESULTS.md](docs/RESULTS.md); reorganizing this source tree does not constitute another full benchmark run.

## Layout and checks

```text
src/clip_officehome/  Core prompts, routing, data adapters and evaluator
CLIP/                OpenAI CLIP implementation, tokenizer and license
src/clip_officehome/configs/  PACS/OfficeHome settings and ImageNet class metadata
scripts/             Evaluation, zero-shot controls and result validation
tests/               Configuration, routing and evaluation-contract checks
docs/                Protocols, frozen dataset description and reference results
```

The `clip_officehome` namespace is retained for compatibility across the supported adapters. Text-only and visual-only classes that share helpers with the main method remain in the core modules.

The ImageNet natural-shift adapter evaluates full-set **A/V2/R/Sketch** in independent sessions through `scripts/evaluate_imagenet.py` or `mt-tta-evaluate-imagenet`. See [the fixed protocol and commands](docs/IMAGENET_OOD.md). It reuses the student-only pre-update method with confidence gate 0.815 and learning rate 7.5e-6.

The frozen **MT-Sketch4-20K** adapter uses one continual stream with four known tasks and different native class spaces. Prompts and AdamW state persist while tasks alternate image by image. It uses 64 unlabeled previews per task and batch1 student-only pre-update scoring. The completed seed0 reference scored **60.215%**, versus **57.825%** for frozen CLIP B/16. The adapter is included, but the exact frozen membership, construction pipeline and prepared data bundle are not distributed in this code release. The following commands require that separate bundle; downloading the four original datasets alone is insufficient. See the [dataset description](docs/MT_SKETCH4_DATASET.md), [evaluation protocol](docs/MT_SKETCH4_EVALUATION.md), and [reference results](docs/RESULTS.md).

```bash
python scripts/evaluate_mtsketch.py --data-root /path/to/MT-Sketch4-20K --dry-run
python scripts/evaluate_mtsketch.py --data-root /path/to/MT-Sketch4-20K --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/mtsketch_seed0
```

```bash
python -m unittest discover -s tests -v
```

## Licensing

The project is distributed under the [MIT license](LICENSE). OpenAI CLIP's notice is retained in [CLIP/LICENSE](CLIP/LICENSE), and the TPT class metadata notice is retained in [licenses/TPT_LICENSE](licenses/TPT_LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for their scope. Datasets and weights have separate terms.
