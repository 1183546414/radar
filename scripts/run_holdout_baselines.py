#!/usr/bin/env python3
"""运行留出集 6 个 baseline 对比（需已训练 VAE + Q）。

用法（项目根目录）:
    python scripts/run_holdout_baselines.py
    python scripts/run_holdout_baselines.py --methods retrieve_q random
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.infer.holdout_baselines import METHODS, run_all_baselines

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="留出集 baseline 对比")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
    )
    parser.add_argument("--n-holdout", type=int, default=500)
    parser.add_argument("--regenerate-holdout", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_all_baselines(
        methods=args.methods,
        n_holdout=args.n_holdout,
        regenerate_holdout=args.regenerate_holdout,
        verbose=not args.quiet,
    )
