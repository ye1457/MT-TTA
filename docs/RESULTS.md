# Reference results

Top-1 accuracy with the fixed seed0 configurations. Follow the [reproducibility guide](REPRODUCIBILITY.md) to run each evaluation.

## Shared method

All method runs use seed0, frozen OpenAI CLIP ViT-B/16 student and ViT-L/14 auxiliary teacher, batch1, one deterministic view, and **student-only prediction before updating the current image**. The enabled objective is student hard-pseudo-label CE plus **0.08 × teacher hard-pseudo-label CE**, both applied to student logits. Selected images receive three AdamW updates. Teacher output fusion/calibration and entropy/anchor/KL penalties are disabled.

Each protocol declares 64-image unlabeled previews per domain or task for fixed text Top-2 routing. Previews perform no updates and use no labels or teacher outputs; those images are later scored once.

## PACS and OfficeHome

| Benchmark | Full images | Classes | Four-domain macro Top-1 |
| --- | ---: | ---: | ---: |
| PACS | 9,991 | 7 | 98.997603% |
| OfficeHome | 15,588 | 65 | 86.488202% |

Domain sequences are `art_painting → cartoon → photo → sketch` and `Art → Clipart → Product → Real_World`. Prompts persist across domains; optimizers restart at domain boundaries. Images are class-directory/path sorted. Gates are 0.95 for PACS and 0.815 for OfficeHome. Learning rate is 7.5e-6, except OfficeHome Clipart 2e-6.

PACS selects text routes by mean Top1-minus-Top2 probability margin; OfficeHome uses mean maximum class probability. OfficeHome initializes independent primary and text-only shadow branches before testing; both adapt continuously. The predeclared report uses primary for the first three domains and shadow for Real_World, with no branch fusion. Its 15,588 scored images require 26,819 branch visits plus 448 previews; PACS uses 9,991 primary visits plus 256 previews.

PACS uses the `r06_margin64` configuration. See [the exact PACS/OfficeHome protocol](EVALUATION.md).

## ImageNet natural distribution shifts

Each benchmark starts from the same prompt initialization and empty optimizer state; state persists only within that benchmark. Runs transfer the fixed OfficeHome primary configuration: gate 0.815, lr 7.5e-6, no shadow or benchmark-specific LR.

| Benchmark | Images | Classes | Frozen CLIP B/16 | Method Top-1 | Change (percentage points) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet-A | 7,500 | 200 | 47.880% | 53.187% | +5.307 |
| ImageNet-V2 matched-frequency | 10,000 | 1,000 | 60.630% | 60.830% | +0.200 |
| ImageNet-R | 30,000 | 200 | 73.793% | 84.377% | +10.583 |
| ImageNet-Sketch | 50,889 | 1,000 | 45.961% | 49.258% | +3.297 |
| **Equal four-benchmark mean** | **98,389 total** | — | **57.066%** | **61.913%** | **+4.847** |

The mean assigns each benchmark weight 1/4, rather than pooling all 98,389 images. Input order is pinned TPT prompt name, WNID and original relative path, with A/R 200-class spaces and V2/Sketch 1,000-class mappings.

The frozen control uses the same B/16 weights, members and labels, one `a photo of a {class}` template and no teacher/adaptation. Control inference uses batch64 while the method uses batch1. See [the ImageNet protocol](IMAGENET_OOD.md).

## MT-Sketch4-20K

The frozen `1.0-review` [dataset](MT_SKETCH4_DATASET.md) uses one prompt bank and persistent AdamW state across image-by-image task switches. All four text routes are chosen from the same initial prompts before adaptation, with 64 previews per task and 256 total. Gate is 0.815 and LR 7.5e-6, with primary only and no shadow.

| Task | Images | Native classes | Frozen CLIP B/16 | Method Top-1 | Change (percentage points) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DomainNet Sketch | 5,000 | 345 | 64.820% | 66.980% | +2.160 |
| ImageNet-Sketch subset | 5,000 | 1,000 | 44.260% | 44.780% | +0.520 |
| TU-Berlin | 5,000 | 250 | 56.560% | 59.000% | +2.440 |
| Sketchy | 5,000 | 125 | 65.660% | 70.100% | +4.440 |
| **Equal four-task mean** | **20,000 total** | **4 native spaces** | **57.825%** | **60.215%** | **+2.390** |

Tasks alternate DomainNet-Sketch → ImageNet-Sketch → TU-Berlin → Sketchy, one image each per cycle. Task ID selects its native class space; this is not a merged 1,720-class classifier. Both method and control use batch1 with identical decoding and membership. Equal task sizes make the equal-task mean and pooled accuracy identical here.

The ImageNet-Sketch task here is a 5,000-image subset evaluated with cross-task state; the ImageNet table above uses the full 50,889-image benchmark with independent initialization. See [the multi-task evaluation protocol](MT_SKETCH4_EVALUATION.md).
