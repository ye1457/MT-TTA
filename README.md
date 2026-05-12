<div align="center">

# MT-TTA

**A clean research workspace for CLIP-based OfficeHome adaptation experiments**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu118-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76b900)
![Status](https://img.shields.io/badge/Status-Active-success)

</div>

## Overview

`MT-TTA` is a personal research repository for OfficeHome experiments built around CLIP, zero-shot transfer, and test-time adaptation.

This GitHub repository is intentionally kept clean and lightweight:

- Core source code is tracked with Git.
- Large experiment outputs are excluded.
- Local launcher scripts are excluded.
- GPU environment files are excluded.

## Repository Layout

```text
MT-TTA/
|-- CLIP/                 # Local copy of the CLIP reference implementation
|-- src/clip_officehome/  # Core project code
|-- requirements.txt      # Python dependencies
|-- README.md             # Project overview
`-- .gitignore            # Git exclusion rules
```

## Environment

This project is maintained for a GPU workflow.

```bash
conda create -n MT-TTA python=3.11 -y
conda activate MT-TTA

export PYTHONNOUSERSITE=1

pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 torchvision==0.22.1+cu118

pip install -r requirements.txt
pip install --no-deps git+https://github.com/openai/CLIP.git
```

After the environment is created once, you can re-enter it with:

```bash
conda activate MT-TTA
export PYTHONNOUSERSITE=1
```

## What Is Tracked

The following content is included in Git:

- `src/clip_officehome/`
- `CLIP/`
- `requirements.txt`
- `README.md`
- `.gitignore`

The following content is intentionally not pushed to GitHub:

- `results/`
- `scripts/`
- `.venv-gpu/`
- `.cache/`

## Notes

- The repository name has been aligned to `MT-TTA`.
- The GitHub remote is configured as `git@github.com:ye1457/MT-TTA.git`.
- This repo is currently structured for code management first, experiment artifacts second.
