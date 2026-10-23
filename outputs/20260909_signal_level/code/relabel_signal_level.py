# -*- coding: utf-8 -*-
"""Round 20260909 路线 A 第 1 步：全量训练数据重标注为信号级 SINR。

输入: data/simulated/simulated_radar_jamming_high_contrast_10k.csv（解析标签）
输出: outputs/20260909_signal_level/data/
  - simulated_radar_jamming_high_contrast_10k_signal.csv
      SINR_dB          <- 信号级真值（相位编码雷达行保留解析标签）
      SINR_analytic_dB <- 原解析标签（保留对照）
      label_source     <- "signal" | "analytic_fallback(相位编码)"
  - relabel_report.json

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20260909_signal_level/code/relabel_signal_level.py
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from radar.data.high_contrast_sim import calculate_sinr_high_contrast, ghz_to_hz  # noqa: E402
from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV  # noqa: E402
from signal_sinr import SignalSinrBackend, radar_family  # noqa: E402

JSR_DB = 10.0
PROGRESS_EVERY = 5000


def _analytic(row: pd.Series) -> float:
    return calculate_sinr_high_contrast(
        ghz_to_hz(float(row["Radar_CarrierFreq_GHz"])),
        ghz_to_hz(float(row["Radar_Bandwidth_GHz"])),
        float(row["Radar_Gain_dB"]),
        ghz_to_hz(float(row["Jammer_CarrierFreq_GHz"])),
        ghz_to_hz(float(row["Jammer_Bandwidth_GHz"])),
        float(row["Jammer_Gain_dB"]),
        str(row["Jammer_Type"]),
        r_type=str(row["Radar_Type"]),
        r_pw=float(row["Radar_PulseWidth_ns"]),
        j_sweep_step_hz=ghz_to_hz(float(row.get("Jammer_Sweep_Step_GHz", 0.0) or 0.0)),
        j_int_dur_ns=float(row.get("Jammer_IntSample_Dur_ns", 0.0) or 0.0),
        j_int_delay_ns=float(row.get("Jammer_IntSample_Delay_ns", 0.0) or 0.0),
        j_int_intv_ns=float(row.get("Jammer_IntSample_Intv_ns", 0.0) or 0.0),
        j_ft_num=float(row.get("Jammer_FalseTarget_Num", 0.0) or 0.0),
        add_noise=False,
    )


def main() -> None:
    t0 = time.time()
    out_dir = ROUND / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_SIMULATED_HIGH_CONTRAST_CSV)
    print(f"源 CSV: {DATA_SIMULATED_HIGH_CONTRAST_CSV}")
    print(f"行数: {len(df)}, 场景数: {df['scenario_id'].nunique()}")

    backend = SignalSinrBackend(jsr_db=JSR_DB)
    old = df["SINR_dB"].to_numpy(dtype=np.float64)
    new = np.empty(len(df), dtype=np.float64)
    sources = np.empty(len(df), dtype=object)
    n_fallback = 0

    for i in range(len(df)):
        row = df.iloc[i]
        if radar_family(str(row["Radar_Type"])) == "相位编码":
            new[i] = _analytic(row)
            sources[i] = "analytic_fallback"
            n_fallback += 1
        else:
            new[i] = backend.sinr(row, row)
            sources[i] = "signal"
        if i % PROGRESS_EVERY == 0 and i > 0:
            rate = i / (time.time() - t0)
            eta = (len(df) - i) / max(rate, 1e-9) / 60
            print(f"  {i}/{len(df)}  {rate:.0f} 行/秒  ETA {eta:.1f} 分钟")

    df["SINR_analytic_dB"] = np.round(old, 4)
    df["SINR_dB"] = np.round(new, 4)
    df["label_source"] = sources
    out_csv = out_dir / "simulated_radar_jamming_high_contrast_10k_signal.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    mask = sources == "signal"
    old_s, new_s = pd.Series(old[mask]), pd.Series(new[mask])
    report = {
        "source_csv": str(DATA_SIMULATED_HIGH_CONTRAST_CSV),
        "output_csv": str(out_csv),
        "jsr_db": JSR_DB,
        "n_rows": int(len(df)),
        "n_signal": int(mask.sum()),
        "n_analytic_fallback": int(n_fallback),
        "fallback_reason": "相位编码波形无法用 FM 形式建模",
        "pearson_vs_analytic_signal_rows": float(old_s.corr(new_s)),
        "spearman_vs_analytic_signal_rows": float(old_s.corr(new_s, method="spearman")),
        "signal_label_stats": {
            "mean": float(new_s.mean()), "std": float(new_s.std()),
            "min": float(new_s.min()), "max": float(new_s.max()),
        },
        "analytic_label_stats": {
            "mean": float(old_s.mean()), "std": float(old_s.std()),
            "min": float(old_s.min()), "max": float(old_s.max()),
        },
        "backend_stats": backend.stats(),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(out_dir / "relabel_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n输出: {out_csv}")


if __name__ == "__main__":
    main()
