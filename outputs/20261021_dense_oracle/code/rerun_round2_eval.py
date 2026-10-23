# -*- coding: utf-8 -*-
"""在当前可复现留出集上重跑 Round2 方法评估（输出到本轮目录，不覆盖历史存档）。

背景：历史评估（2026-09/10）在旧项目副本 Desktop\\radar 的留出 CSV 上运行，
该副本已丢失且无法由当前代码重建（K20 均值 17.25 vs 当前 CSV 19.30）。
为与 outputs/20261021_dense_oracle 的稠密 oracle 配对，必须在同一
（当前、确定性可复现）留出集上重跑方法评估。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261021_dense_oracle/code/rerun_round2_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]           # outputs/20261021_dense_oracle
RADAR_ROOT = HERE.parents[3]      # radar 项目根
sys.path.insert(0, str(RADAR_ROOT / "src"))
sys.path.insert(0, str(RADAR_ROOT / "outputs" / "20260909_signal_level" / "code"))

from radar.infer import holdout_baselines as hb  # noqa: E402
from signal_sinr import SignalSinrBackend  # noqa: E402

JSR_DB = 10.0
N_HOLDOUT = 50
ROUND2 = RADAR_ROOT / "outputs" / "20260909_signal_level"


def main() -> None:
    t0 = time.time()
    out_dir = ROUND / "eval_r2_current"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 路径重定向（与 eval_signal_round.py 相同，仅输出目录指向本轮）----
    hb.ART_PATH = ROUND2 / "processed" / "artifacts.pkl"
    hb.A_STD_PATH = ROUND2 / "processed" / "A_std.npy"
    hb.SCENARIO_ID_PATH = ROUND2 / "processed" / "scenario_id.npy"
    hb.MODEL_PATH = ROUND2 / "models" / "latent_q" / "q_latent_best.pt"
    hb.RETRIEVAL_MODEL_PATH = ROUND2 / "models" / "latent_q_retrieval" / "q_retrieval_best.pt"
    # hb.EMB_PATH 保持原值 outputs/state_vae/state_embeddings_mu.npy（与原 Round2 评估一致）
    hb.OUT_DIR = out_dir
    hb.SUMMARY_JSON = out_dir / "summary.json"
    hb.SUMMARY_CSV = out_dir / "summary.csv"
    hb.PER_SCENARIO_JSON = out_dir / "per_scenario_detail.json"

    assert hb.MODEL_PATH.is_file(), f"重训模型不存在: {hb.MODEL_PATH}"
    assert hb.ART_PATH.is_file(), f"processed 不存在: {hb.ART_PATH}"
    assert hb.EMB_PATH.is_file(), f"VAE 潜向量缓存不存在: {hb.EMB_PATH}"

    # artifacts.pkl 内 source_csv 是搬迁前旧路径 -> 回退到 Round2 重标注 CSV
    relabel_csv = ROUND2 / "data" / "simulated_radar_jamming_high_contrast_10k_signal.csv"
    orig_load = hb.load_simulated_csv

    def _load_csv_fallback(path):
        p = Path(path)
        return orig_load(p if p.is_file() else relabel_csv)

    hb.load_simulated_csv = _load_csv_fallback

    backend = SignalSinrBackend(jsr_db=JSR_DB)
    hb.set_sinr_backend(backend.sinr)
    summary = hb.run_all_baselines(n_holdout=N_HOLDOUT, verbose=True)
    summary["backend"] = {
        "name": "radarsimpy_signal_level",
        "jsr_db": JSR_DB,
        "n_holdout": N_HOLDOUT,
        "note": ("在当前可复现留出集（data/v2/processed/holdout_eval_scenarios.csv）上重跑，"
                 "与 outputs/20261021_dense_oracle 稠密 oracle 同一场景集；"
                 "历史存档 eval_retrained 基于已丢失的旧场景集，不可与本轮数值直接混用"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "backend_stats": backend.stats(),
    }
    with open(hb.SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n后端统计: {backend.stats()}")
    print(f"总耗时: {(time.time() - t0) / 60:.1f} 分钟")
    print(f"输出: {out_dir}")


if __name__ == "__main__":
    main()
