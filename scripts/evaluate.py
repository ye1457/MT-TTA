#!/usr/bin/env python3
"""Source-checkout entry point; installed users can run mt-tta-evaluate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "CLIP")]

from clip_officehome.evaluation import main

if __name__ == "__main__":
    main()
