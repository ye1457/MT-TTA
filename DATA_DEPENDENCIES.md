# External data and model dependencies

No image datasets or model weights are redistributed. Obtain each dataset from its authorized source and pass its location with `--data-root`.

| Benchmark | Required domain folders | Expected images |
| --- | --- | --- |
| PACS | art_painting, cartoon, photo, sketch | 2,048 / 2,344 / 1,670 / 3,929 |
| OfficeHome | Art, Clipart, Product, Real_World | 2,427 / 4,365 / 4,439 / 4,357 |

Use the authors' dataset pages for access and dataset-specific terms:

- [PACS project](https://domaingeneralization.github.io/)
- [OfficeHome project](https://www.hemanthdv.org/officeHomeDataset.html)
- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
- [ImageNet-V2](https://github.com/modestyachts/ImageNetV2), **matched-frequency** variant
- [ImageNet-R](https://github.com/hendrycks/imagenet-r)
- [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

Point `--data-root` directly at the directory containing the domain folders for PACS/OfficeHome, or the class folders for each ImageNet benchmark. For example:

```text
PACS/art_painting/dog/<original filename>
PACS/cartoon/dog/<original filename>
PACS/photo/dog/<original filename>
PACS/sketch/dog/<original filename>
OfficeHome/Art/Alarm_Clock/<original filename>
OfficeHome/Clipart/Alarm_Clock/<original filename>
OfficeHome/Product/Alarm_Clock/<original filename>
OfficeHome/Real_World/Alarm_Clock/<original filename>
imagenet-a/<WNID>/<original filename>
imagenetv2-matched-frequency-format-val/0/<original filename>
imagenet-r/<WNID>/<original filename>
imagenet-sketch/<WNID>/<original filename>
```

These examples show one class per domain; all required classes and images must be present. Use complete releases rather than their usual supervised training/validation partitions. This protocol evaluates every member as an unlabeled target stream; it does not train on three labeled source domains.

The [ImageNet natural-shift adapter](docs/IMAGENET_OOD.md) also accepts the original released ImageNet-A (7,500 images / 200 classes), ImageNet-V2 matched-frequency (10,000 / 1,000), ImageNet-R (30,000 / 200), and ImageNet-Sketch (50,889 / 1,000). Pass each benchmark's extracted image directory separately; V2 uses its numeric canonical class directories, while A/R/Sketch use WNID directories. The packaged class metadata fixes canonical identities, prompt names and A/R masks.

The [MT-Sketch4-20K adapter](docs/MT_SKETCH4_EVALUATION.md) requires the separately prepared **1.0-review** frozen bundle, with four sources contributing 5,000 images each. Its required layout, membership identity, original sources and review status are described in [MT_SKETCH4_DATASET.md](docs/MT_SKETCH4_DATASET.md). This code release does not include a public download of the prepared bundle or redistribute its images. Original source downloads alone do not satisfy the frozen manifest checks.

For PACS/OfficeHome, each domain contains class directories with image files. PACS uses dog, elephant, giraffe, guitar, horse, house, person; OfficeHome uses its standard 65 classes. Class names are case-sensitive and identical across domains. The configurations record expected class names and counts. Dry-run validates these without loading models.

PACS/OfficeHome order is class-directory name followed by full POSIX image path within the class. ImageNet order uses the pinned human prompt name, WNID and original relative path. MT-Sketch4-20K uses its frozen balanced round-robin stream. Keep release filenames and hierarchy to reproduce the applicable reference order. Matching counts alone does not prove identical image bytes; formal runs record ordered input hashes. PACS/OfficeHome and the four ImageNet evaluations use their full releases; the self-built 20K benchmark has its own fixed subset definition.

Supply `ViT-B-16.pt` and `ViT-L-14.pt` in `--model-cache`, or use `~/.cache/clip`. Expected SHA256:

- ViT-B/16: `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`
- ViT-L/14: `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`

Formal evaluation verifies these model digests. Ignored `models/` and `.venv/` may remain on a development machine; they are local dependencies, excluded from source/wheel distributions. Public configurations contain no machine-specific data roots.

To populate a new local cache, download the official URLs recorded by the bundled OpenAI CLIP implementation:

```bash
mkdir -p models
curl --fail --location --retry 3 --output models/ViT-B-16.pt https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
curl --fail --location --retry 3 --output models/ViT-L-14.pt https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
sha256sum models/ViT-B-16.pt models/ViT-L-14.pt
```

Compare the printed digests with those above, then pass `--model-cache models`. Evaluation refuses missing or mismatched weights before model execution. Zero-shot control scripts require only the B/16 file.
