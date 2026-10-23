# -*- coding: utf-8 -*-
"""兼容入口：留出评估见 holdout_baselines.py。"""
from radar.infer.holdout_baselines import holdout_eval_main, run_all_baselines

__all__ = ["holdout_eval_main", "run_all_baselines"]

if __name__ == "__main__":
    run_all_baselines()
