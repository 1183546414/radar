# -*- coding: utf-8 -*-
"""验证默认推荐模块 radar.infer.q_ensemble 与 20261007 评估结果完全一致。

对 50 个留出场景用 QEnsembleRecommender 重算推荐动作，经信号级真值
（SignalSinrBackend, JSR=10dB）打分后，与存档
eval_fewshot/per_scenario_detail.json 中 ensemble_ctx_pessimistic 的
recommended_sinr_db 逐场景对拍。rng 协议一致（union 池种子 = 2026 + sid），
预期 50/50 精确一致。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261007_fewshot/code/verify_default_ensemble.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
for _p in HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        sys.path.insert(0, str(_p / "src"))
        break
sys.path.insert(0, str(HERE.parents[2] / "20260909_signal_level" / "code"))  # signal_sinr

from radar.infer import holdout_baselines as hb  # noqa: E402
from radar.infer.q_ensemble import (  # noqa: E402
    HOLDOUT_SEED,
    QEnsembleRecommender,
    _patch_hb_paths,
)
from signal_sinr import SignalSinrBackend  # noqa: E402

JSR_DB = 10.0
N_HOLDOUT = 50
ARCHIVE = HERE.parents[1] / "eval_fewshot" / "per_scenario_detail.json"


def main() -> None:
    t0 = time.time()
    backend = SignalSinrBackend(jsr_db=JSR_DB)
    hb.set_sinr_backend(backend.sinr)
    _patch_hb_paths()

    with open(ARCHIVE, encoding="utf-8") as f:
        archive = json.load(f)
    ref = archive["ensemble_ctx_pessimistic"]["recommended_sinr_db"]
    assert len(ref) == N_HOLDOUT, f"存档场景数 {len(ref)} != {N_HOLDOUT}"

    rec = QEnsembleRecommender()
    ctx = rec.ctx
    sids = list(ctx.holdout_sids)[:N_HOLDOUT]
    print(f"对拍 {len(sids)} 个留出场景（存档 ensemble_ctx_pessimistic）...", flush=True)

    diffs = []
    mismatches = []
    for i, sid in enumerate(sids):
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jrow = block.iloc[0]
        d = rec.recommend_for_holdout(sid)
        row_idx = int(d["row_indices"][0])
        r_rec = hb._sinr_on_query(jrow, ctx.train_df, row_idx)
        diff = abs(float(r_rec) - float(ref[i]))
        diffs.append(diff)
        if diff > 1e-9:
            mismatches.append((sid, float(r_rec), float(ref[i])))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sids)} 已对拍 (累计最大差 {max(diffs):.2e})",
                  flush=True)

    max_diff = max(diffs)
    print(f"\n结果: {'PASS' if max_diff <= 1e-9 else 'FAIL'} | "
          f"最大偏差 {max_diff:.2e} | 不一致场景 {len(mismatches)} | "
          f"耗时 {time.time() - t0:.1f}s")
    for sid, got, exp in mismatches[:5]:
        print(f"  sid={sid}: 新模块 {got:.4f} vs 存档 {exp:.4f}")


if __name__ == "__main__":
    main()
