# MT-Sketch4-20K dataset

**Version: `1.0-review`.** This benchmark combines 20,000 sketch images from four existing datasets into four known classification tasks. Each source contributes 5,000 images and retains its native class space. Sketch/line drawing is their common visual domain; the sources retain distinct acquisition styles and distributions.

Images are obtained separately under the source datasets' terms. The project's MIT license does not cover these images. This code release does not include a public download of the prepared frozen bundle.

## Sources and membership

| Source / task ID | Initial candidates | Eligible after screening | Selected | Native classes | Selected per class |
| --- | ---: | ---: | ---: | ---: | --- |
| DomainNet Sketch / `domainnet_sketch` | 20,916 | 20,836 | 5,000 | 345 | 4–15 |
| ImageNet-Sketch / `imagenet_sketch` | 50,889 | 35,755 | 5,000 | 1,000 | 1–6 |
| TU-Berlin / `tuberlin` | 20,000 | 20,000 | 5,000 | 250 | 20 |
| Sketchy / `sketchy` | 7,063 | 6,588 | 5,000 | 125 | 40 |

All native classes are represented. Candidate pools are those used in this construction, not necessarily the original total size of each source.

- DomainNet uses only the official cleaned **test** list, without train members.
- ImageNet-Sketch starts from its full release; its 5,000 selected images are a subset for this benchmark and do not replace the full 50,889-image natural-shift evaluation.
- TU-Berlin uses the 20,000-image collection, recombined from mirror partitions before selection. This is a newly frozen subset, not an official author-provided test split.
- Sketchy uses original geometry `tx_000000000000` for author test objects. Error/Ambiguous examples are excluded; recognizable WrongPose/Context examples remain. All 1,250 test objects contribute four distinct human sketches each. Geometric variants are not counted as independent samples.

## Selection and review status

Construction uses seed0. Capacity-constrained balanced quotas cover every native class without duplicating images to fill rare classes. Fixed morphology features—ink occupancy, edge density, aspect ratio and grayscale entropy—spread selection over different drawing forms. Sketchy additionally preserves full object coverage and increases within-object artist diversity where available.

Selection does not use CLIP/teacher predictions, confidence or accuracy. Original bytes, source identities, file/pixel hashes, class mappings, exclusions and final membership are recorded. Final images have 20,000 distinct file hashes and decoded-pixel hashes. These checks support integrity and duplicate screening; they do not replace semantic review.

The frozen version retains 106 `native_label_conflict` flags: 3 DomainNet and 103 ImageNet-Sketch images. An additional 8 images have `visual_review_needed` flags, without overlap. These are ambiguity/review flags, not 114 confirmed wrong labels. The reported evaluation retains this membership and native labels. Review evidence remains in the separate data bundle.

## Class spaces and stream

Task ID selects the native 345/1,000/250/125-class vocabulary. The sum 1,720 counts task-local class positions, not merged semantic classes. `class_index` is the continuous prediction index within a task; native `class_id` and ImageNet canonical indices must be mapped through `metadata/classes.json`.

The reference stream is `balanced_round_robin`: one image from DomainNet-Sketch, ImageNet-Sketch, TU-Berlin and Sketchy, repeated 5,000 times. Within each task, order is class name then original relative path. The bundle also defines `task_blocks`, a different evaluation condition. Adaptation receives image, task identity and class candidates; labels are reserved for scoring. Preview budgets are specified in the [evaluation protocol](MT_SKETCH4_EVALUATION.md).

## Required frozen bundle

```text
MT-Sketch4-20K/
  benchmark.py
  manifest.jsonl
  streams.json
  metadata/
    classes.json
    protocol.json
    source_index.json
    sources/
  images/
    domainnet_sketch/
    imagenet_sketch/
    tuberlin/
    sketchy/
```

The bundle also retains per-class counts, checksums and audit records. Original source downloads alone do not reproduce its membership or satisfy the frozen checks. The adapter pins the required artifact digests in `src/clip_officehome/multitask_sketch.py`. Reference manifest SHA256:

```text
89f07bf643e621ecaef70010d51d2044d0750213aac19782324b27a615343570
```

Validate the matching bundle without model inference:

```bash
python scripts/evaluate_mtsketch.py --data-root /path/to/MT-Sketch4-20K --dry-run
```

## Original sources

| Source | Acquisition references | Recorded verification scope |
| --- | --- | --- |
| DomainNet Sketch | [Official cleaned test list](https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt) | Native identities and labels match the official list; local hashes and decoding are recorded. No complete official image-ZIP comparison is claimed. |
| ImageNet-Sketch | [Author-linked fixed revision](https://huggingface.co/datasets/songweig/imagenet_sketch/tree/0d8b8b5380efe50baae075842505918ad43c88fe) | Source member lengths and CRC32 match the fixed original ZIP directory; local SHA256 values are recorded. |
| TU-Berlin | [Original project](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/); [fixed mirror revision](https://huggingface.co/datasets/sdiaeyu6n/tu-berlin/tree/e7c6c0dedcf2aa4ca060878893bca8d77ef699f4) | Mirror matches LFS SHA256; 20,000 native class/image IDs match the research group's SVG archive. No original PNG-package checksum is claimed. |
| Sketchy | [Author project](https://patsorn.me/projects/sketchy/); [fixed mirror revision](https://huggingface.co/datasets/DrRORAL/sketchy-dataset/tree/dcbdf00ab09409b4244d9fe717e652274b3fe648) | Mirror matches LFS SHA256; 7,063 PNG entries match the author 7z directory by length and CRC32. |

Exact source URLs, revisions, hashes and qualifications remain in the bundle's `metadata/source_index.json` and per-source metadata. This repository supplies the description and adapter; image distribution remains separate.
