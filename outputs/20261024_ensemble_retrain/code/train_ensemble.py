# -*- coding: utf-8 -*-
"""重训 5-seed QRetrievalNet 集成（20261024，新编码器/新管线）。

与 src/radar/train/train_q_retrieval.py 完全同协议（直接复用其训练主函数），
仅按成员覆写 seed 与输出路径；成员保存到
outputs/q_value/latent_q_retrieval_ensemble/seed{S}/，供
radar.infer.q_ensemble.QEnsembleRecommender（部署默认打分器）加载。

z_ctx 缓存复用 outputs/q_value/latent_q_retrieval/ 下的现有缓存（与成员无关）。
热启动权重 = 20261023 重训的基础 Q（q_value/latent_q）。

用法:
    python outputs/20261024_ensemble_retrain/code/train_ensemble.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
for _p in HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        sys.path.insert(0, str(_p / "src"))
        break

import radar.train.train_q_retrieval as tqr
from radar.paths import OUT_Q_VALUE

SEEDS = (42, 1042, 2042, 3042, 4042)   # 与 20261007 轮集成一致的 seed 集
OUT_ROOT = OUT_Q_VALUE / "latent_q_retrieval_ensemble"
REPORT_PATH = ROUND / "train_report.json"


def main() -> None:
    members = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        member_dir = OUT_ROOT / f"seed{seed}"
        metrics_path = member_dir / "metrics.json"
        if metrics_path.is_file():
            print(f"[seed {seed}] 已存在 metrics.json，跳过")
            members.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        print(f"\n===== 集成成员 seed={seed} =====")
        tqr.SEED = seed
        tqr.RETRIEVAL_OUT_DIR = member_dir
        tqr.RETRIEVAL_MODEL_PATH = member_dir / "q_retrieval_best.pt"
        tqr.RETRIEVAL_METRICS_PATH = metrics_path
        t_seed = time.perf_counter()
        tqr.train_q_retrieval_main()
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        m["seed"] = seed
        m["elapsed_min"] = (time.perf_counter() - t_seed) / 60
        metrics_path.write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        members.append(m)
        if hasattr(tqr, "gc"):
            tqr.gc.collect()

    report = {
        "round": "20261024_ensemble_retrain",
        "protocol": "与 train_q_retrieval（20261023 重训版）完全一致，仅换 seed",
        "seeds": list(SEEDS),
        "out_root": str(OUT_ROOT),
        "val_hit1": [m["best_val_retrieval_hit_at_1"] for m in members],
        "val_pearson_std": [m["val_pearson_std"] for m in members],
        "epochs_run": [m["epochs_run"] for m in members],
        "total_min": (time.perf_counter() - t0) / 60,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n全部成员完成: val_hit@1 = {[round(v, 3) for v in report['val_hit1']]}")
    print(f"总耗时 {report['total_min']:.1f} min | 报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
