"""Source-checkout entry point for ImageNet natural-shift benchmarks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clip_officehome.imagenet_ood import main

if __name__ == "__main__":
    main()
