# Environment Setup

```bash
cd /data/ckpt/yeshengjie/clip_officehome_zeroshot

conda create -n MT-TTA python=3.11 -y
conda activate MT-TTA

export PYTHONNOUSERSITE=1

pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 torchvision==0.22.1+cu118

pip install -r requirements.txt
pip install --no-deps git+https://github.com/openai/CLIP.git
```

如果当前环境安装的是 CPU-only PyTorch（例如 `torch 2.11.0+cpu`），不要传 `--device cuda`。
脚本默认支持 `--device auto`，会在有 CUDA 时使用 GPU，否则自动回退到 CPU。

# Run Test

四域测试（推荐使用 `auto`）：

```bash
cd /data/ckpt/yeshengjie/clip_officehome_zeroshot
conda activate MT-TTA
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=0 python scripts/eval_officehome_zeroshot.py \
  --mode all \
  --device auto \
  --batch-size 256 \
  --num-workers 8 \
  --output results/officehome_all.json
```

单域测试（推荐使用 `auto`）：

```bash
cd /data/ckpt/yeshengjie/clip_officehome_zeroshot
conda activate MT-TTA
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=0 python scripts/eval_officehome_zeroshot.py \
  --mode single \
  --target-domain Art \
  --device auto \
  --batch-size 256 \
  --num-workers 8 \
  --output results/art.json
```

如果你明确知道当前环境只能跑 CPU，也可以直接传：

```bash
python scripts/eval_officehome_zeroshot.py --mode all --device cpu
```
