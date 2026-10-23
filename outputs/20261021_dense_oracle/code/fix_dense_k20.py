# -*- coding: utf-8 -*-
"""修正稠密 oracle 产物中的 oracle_k20 字段：解析式(CSV列) -> 信号级重评估。

原因：src/radar/infer/holdout_baselines.py 的 _oracle_k20 在接入信号级后端后，
对留出 K=10 动作做信号级重评估（均值 17.25 dB）；而留出 CSV 的 SINR_dB 列
是解析式标签（均值 19.30 dB），两者不可混用。此前 dense_oracle.py 误用了
CSV 解析列，本脚本以与评估协议一致的方式重算并修正产物。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261021_dense_oracle/code/fix_dense_k20.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
RADAR_ROOT = HERE.parents[3]
sys.path.insert(0, str(RADAR_ROOT / "src"))
sys.path.insert(0, str(RADAR_ROOT / "outputs" / "20260909_signal_level" / "code"))

from signal_sinr import SignalSinrBackend  # noqa: E402

JSR_DB = 10.0
HOLDOUT_CSV = RADAR_ROOT / "data" / "v2" / "processed" / "holdout_eval_scenarios.csv"
RES = ROUND / "results"


def main() -> None:
    doc = json.loads((RES / "dense_oracle_per_scenario.json").read_text(encoding="utf-8"))
    per_sc = doc["per_scenario"]
    holdout_df = pd.read_csv(HOLDOUT_CSV)
    backend = SignalSinrBackend(jsr_db=JSR_DB)

    for p in per_sc:
        block = holdout_df[holdout_df["scenario_id"] == p["sid"]]
        assert len(block) == 10, f"sid={p['sid']}: K={len(block)} != 10"
        sig = float(max(backend.sinr(block.iloc[0], block.iloc[i]) for i in range(len(block))))
        p["oracle_k20_analytic_csv_db"] = round(p.pop("oracle_k20_db"), 4)  # 保留解析值备查
        p["oracle_k20_db"] = round(sig, 4)  # 信号级（与评估协议一致）

    k20 = [p["oracle_k20_db"] for p in per_sc]
    doc["summary"]["reference"] = {
        "oracle_k20_signal_mean_db": round(sum(k20) / len(k20), 3),
        "note": ("K20 为信号级重评估（与 holdout_baselines._oracle_k20 接入后端后一致，"
                 "应等于评估报告的 17.25）；CSV 的 SINR_dB 列为解析式（均值 19.30），"
                 "仅存于 oracle_k20_analytic_csv_db 字段，不可与信号级混用"),
        "oracle_wide_mean_db_reported": "Round2 报告 23.18（100 随机，抽样型非确定）",
    }

    (RES / "dense_oracle_per_scenario.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同步 actions.csv 的 oracle_k20_db 列
    import csv
    csv_path = RES / "dense_oracle_actions.csv"
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_sid = {str(p["sid"]): p for p in per_sc}
    for r in rows:
        p = by_sid[r["scenario_id"]]
        r["oracle_k20_db"] = p["oracle_k20_db"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"已修正 {len(per_sc)} 个场景的 K20 为信号级，均值 {sum(k20)/len(k20):.3f} dB")
    print(f"后端统计: {backend.stats()}")


if __name__ == "__main__":
    main()
