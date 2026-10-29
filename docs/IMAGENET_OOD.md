# ImageNet natural-shift evaluation

This adapter evaluates each original benchmark in a fresh seed0 session. All four natural-shift datasets are supported:

| Benchmark argument | Images | Classes |
| --- | ---: | ---: |
| `imagenet_a` | 7,500 | 200 |
| `imagenet_v2` | 10,000 | 1,000 |
| `imagenet_r` | 30,000 | 200 |
| `imagenet_sketch` | 50,889 | 1,000 |

V2 uses the matched-frequency release, with ten images per class. The mixed [MT-Sketch4-20K](MT_SKETCH4_DATASET.md) dataset is a separate benchmark and is not used here.

```bash
python scripts/evaluate_imagenet.py --benchmark imagenet_v2 --data-root /path/to/imagenetv2-matched-frequency-format-val --dry-run
python scripts/evaluate_imagenet.py --benchmark imagenet_a --data-root /path/to/imagenet-a --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/imagenet_a_seed0
python scripts/evaluate_imagenet.py --benchmark imagenet_v2 --data-root /path/to/imagenetv2-matched-frequency-format-val --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/imagenet_v2_seed0
python scripts/evaluate_imagenet.py --benchmark imagenet_r --data-root /path/to/imagenet-r --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/imagenet_r_seed0
python scripts/evaluate_imagenet.py --benchmark imagenet_sketch --data-root /path/to/imagenet-sketch --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/imagenet_sketch_seed0
```

The method transfers the OfficeHome **primary** branch parameters: CLIP-B/16
student, frozen CLIP-L/14 auxiliary hard pseudo-label teacher (CE weight 0.08),
confidence gate 0.815, AdamW LR 7.5e-6, three update rounds per selected image,
three shared/eight private text experts and two shared/four private visual experts.
There is no shadow branch or domain-specific LR exception for these benchmarks.
All predictions are student-only and scored before adapting the current image.

The first 64 unlabeled images select a fixed Top-2 text route by mean maximum
class probability. The preview makes no parameter updates; those images are
then included once in the complete scored stream. This is known-benchmark
evaluation with preview lookahead, not strict causal no-lookahead TTA.

Class names and A/R masks are pinned to the TPT source revision recorded in
`src/clip_officehome/configs/imagenet_classes.json`. WNID/canonical indices are
the label identities; duplicate human-readable names are preserved. V2 numeric
directories map to canonical ImageNet indices, never lexicographic directory
indices. ImageNet-A predictions use the official compact 200-class space.

Input order is human prompt name, then WNID, then original relative filename.
Every image is included, with a hashed ordered manifest. Some original A
filenames contain scores from dataset collection; fixed filename ordering does
not imply independence from those historical collection scores. No score from
the current evaluation selects members or changes their order.

Report all four benchmark accuracies and their equal mean: `(Acc_A + Acc_V2 + Acc_R + Acc_Sketch) / 4`. A subset mean must be labeled with its included benchmarks and is not the four-benchmark OOD Average. Published methods may use different current-image adaptation/scoring semantics, prompts and stream orders; retain those distinctions when comparing paper-reported numbers.

`scripts/evaluate_imagenet_zeroshot.py` supplies a frozen B/16, single-template
control on the same complete class mapping and stream; its batch64 inference
does not perform adaptation. It never loads a teacher.

For example, run the A control with:

```bash
python scripts/evaluate_imagenet_zeroshot.py --benchmark imagenet_a --data-root /path/to/imagenet-a --model-cache /path/to/clip_weights --gpu 0 --output-dir outputs/imagenet_a_zeroshot_seed0
```

Repeat with the other benchmark arguments and their own data/output directories. After all eight method/control runs finish, `scripts/summarize_imagenet_ood.py` recounts their predictions and validates the shared protocol. Create a JSON manifest using **absolute paths** to those new output directories:

```json
{
  "schema": "imagenet_ood_four_benchmarks_v1",
  "seed": 0,
  "expected_total_images": 98389,
  "benchmarks": [
    {"benchmark": "imagenet_a", "tta_output": "/path/to/outputs/imagenet_a_seed0", "control_output": "/path/to/outputs/imagenet_a_zeroshot_seed0"},
    {"benchmark": "imagenet_v2", "tta_output": "/path/to/outputs/imagenet_v2_seed0", "control_output": "/path/to/outputs/imagenet_v2_zeroshot_seed0"},
    {"benchmark": "imagenet_r", "tta_output": "/path/to/outputs/imagenet_r_seed0", "control_output": "/path/to/outputs/imagenet_r_zeroshot_seed0"},
    {"benchmark": "imagenet_sketch", "tta_output": "/path/to/outputs/imagenet_sketch_seed0", "control_output": "/path/to/outputs/imagenet_sketch_zeroshot_seed0"}
  ]
}
```

Save it as `outputs/imagenet_suite.json`, then run:

```bash
python scripts/summarize_imagenet_ood.py --manifest outputs/imagenet_suite.json --output reports/imagenet_suite
```

The summary includes a machine-readable JSON result and a Chinese Markdown report. All eight runs must use the same runtime source revision; the summarizer checks the current source against each run. Use a new report directory to avoid replacing a previous summary.

Formal results require `completion.json` with passing trace recount. Existing
output directories are not overwritten. The evaluator does not implement
midstream prompt/optimizer checkpoint restoration: an interrupted run must
restart from its initial state in a new output directory. Do not modify recorded
source files during a running evaluation.

The completed seed0 reference and frozen-control results are summarized in [RESULTS.md](RESULTS.md).
