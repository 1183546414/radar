# -*- coding: utf-8 -*-
"""Round 20260909 路线 A 第 3 步：重训后模型的信号级留出评估。

与重训前评估（archive/holdout_baselines_radarsimpy_jsr10）严格同条件对照：
  * 同一 holdout CSV 的前 50 个场景
  * 同一信号级真值后端（JSR=10dB）
  * 状态嵌入复用原 outputs/state_vae（S 未变）
唯一区别：Q 模型来自本轮重训（models/），train_df 来自重标注 CSV。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20260909_signal_level/code/eval_signal_round.py
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
sys.path.insert(0, str(HERE.parent))

from radar.infer import holdout_baselines as hb  # noqa: E402
from signal_sinr import SignalSinrBackend  # noqa: E402

JSR_DB = 10.0
N_HOLDOUT = 50


def main() -> None:
    t0 = time.time()
    eval_dir = ROUND / "eval_retrained"
    eval_dir.mkdir(parents=True, exist_ok=True)
    round_processed = ROUND / "processed"
    round_models = ROUND / "models"

    # ---- 路径重定向（原 hb 常量不动，仅在运行期覆盖）----
    hb.ART_PATH = round_processed / "artifacts.pkl"
    hb.A_STD_PATH = round_processed / "A_std.npy"
    hb.SCENARIO_ID_PATH = round_processed / "scenario_id.npy"
    hb.MODEL_PATH = round_models / "latent_q" / "q_latent_best.pt"
    hb.RETRIEVAL_MODEL_PATH = round_models / "latent_q_retrieval" / "q_retrieval_best.pt"
    # hb.EMB_PATH 保持原值（state_embeddings_mu.npy，S 未变）
    hb.OUT_DIR = eval_dir
    hb.SUMMARY_JSON = eval_dir / "summary.json"
    hb.SUMMARY_CSV = eval_dir / "summary.csv"
    hb.PER_SCENARIO_JSON = eval_dir / "per_scenario_detail.json"

    assert hb.MODEL_PATH.is_file(), f"重训模型不存在: {hb.MODEL_PATH}"
    assert hb.ART_PATH.is_file(), f"processed 不存在: {hb.ART_PATH}"

    backend = SignalSinrBackend(jsr_db=JSR_DB)
    hb.set_sinr_backend(backend.sinr)

    summary = hb.run_all_baselines(n_holdout=N_HOLDOUT, verbose=True)
    summary["backend"] = {
        "name": "radarsimpy_signal_level",
        "jsr_db": JSR_DB,
        "n_holdout": N_HOLDOUT,
        "truth_metric": "峰/(底噪中位数+最大杂散峰)",
        "labels": "信号级重标注数据 + 本轮重训模型",
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "backend_stats": backend.stats(),
    }
    with open(hb.SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n后端统计: {backend.stats()}")
    print(f"总耗时: {(time.time() - t0) / 60:.1f} 分钟")
    print(f"输出: {eval_dir}")


if __name__ == "__main__":
    main()
