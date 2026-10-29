# Third-party components

`CLIP/clip/` contains OpenAI CLIP's Python implementation and tokenizer vocabulary.
The four Python modules match the upstream copy retained during the original
project import. Upstream: https://github.com/openai/CLIP.

OpenAI CLIP is distributed under the MIT license, reproduced in
[`CLIP/LICENSE`](CLIP/LICENSE), Copyright (c) 2021 OpenAI. The license is also
included in the installed distribution under `share/mt-tta/third_party`.

`src/clip_officehome/configs/imagenet_classes.json` contains ImageNet class names
and subset mappings derived from the TPT project's `data/imagnet_prompts.py`
and `data/imagenet_variants.py`, pinned to revision
`63ecbace79694205d7884e63fdc3137a200f0b0e`:
https://github.com/azshue/TPT/tree/63ecbace79694205d7884e63fdc3137a200f0b0e.
The adapter uses this class metadata; it does not bundle TPT's adaptation method.
TPT is distributed under the MIT license, Copyright (c) 2022 Manli Shu,
reproduced in [`licenses/TPT_LICENSE`](licenses/TPT_LICENSE) and included in the
installed distribution under `share/mt-tta/third_party`.

PyTorch, torchvision, NumPy, Pillow, ftfy, regex and tqdm are dependencies, not
project-owned code. Their respective licenses apply. Model weights and datasets
are obtained separately and are not included in the source or wheel distribution.

The project's own implementation is distributed under the MIT license in the
root `LICENSE`, as selected by the project owner. OpenAI's existing copyright
notice is retained for the vendored CLIP component, and TPT's notice is retained
for the derived ImageNet class metadata.
