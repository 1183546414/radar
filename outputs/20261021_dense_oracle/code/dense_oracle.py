# -*- coding: utf-8 -*-
"""稠密近似 oracle：为 50 个留出场景逼近动作空间真上界。

背景：此前所有轮次的 regret 参照 Oracle-K20（留出 10 个随机动作的最大值），
它只是弱参照而非真上界。本脚本对每个场景在 6 维混合动作空间上做
best-of-N 随机采样 + 坐标细化，得到近似最优动作 â*（真最优的下界），
对它的 regret 是真 regret 的上界。

与既有评估严格同口径：
  * 同一 holdout CSV 前 50 个场景（unique sid 按出现顺序），jammer_row = block.iloc[0]
  * 同一信号级后端 SignalSinrBackend(jsr=10dB)，含 GHz 舍入（1MHz 网格）
  * 雷达动作行构造与 _oracle_wide 完全一致

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261021_dense_oracle/code/dense_oracle.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_RADAR_ROOT = None
for _p in _HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        _RADAR_ROOT = _p
        sys.path.insert(0, str(_p / "src"))
        break
assert _RADAR_ROOT is not None, "未找到 radar 项目根目录"
sys.path.insert(0, str(_RADAR_ROOT / "src"))
sys.path.insert(0, str(_RADAR_ROOT / "outputs" / "20260909_signal_level" / "code"))  # signal_sinr.py

from radar.data.high_contrast_sim import (  # noqa: E402
    RADAR_CONFIG,
    RADAR_TYPES,
    get_random_100_aligned,
    hz_to_ghz,
    sample_radar_action,
)
from signal_sinr import SignalSinrBackend  # noqa: E402

JSR_DB = 10.0
N_HOLDOUT = 50
N_RANDOM = 5000
MAX_SWEEPS = 8
MIN_IMPROVE_DB = 0.01
GLOBAL_SEED = 20261021

HOLDOUT_CSV = _RADAR_ROOT / "data" / "v2" / "processed" / "holdout_eval_scenarios.csv"
ROUND = _HERE.parents[1]
OUT_DIR = ROUND / "results"


def make_radar_row(act: dict) -> pd.Series:
    return pd.Series({
        "Radar_CarrierFreq_GHz": hz_to_ghz(act["r_freq"]),
        "Radar_Gain_dB": act["r_gain"],
        "Radar_Bandwidth_GHz": hz_to_ghz(act["r_bw"]),
        "Radar_PulseWidth_ns": act["r_pw"],
        "Radar_Type": act["r_type"],
        "Radar_ChirpType": act["r_chirp"],
    })


def snap_action(act: dict) -> dict:
    """把连续维吸附到评估网格（载频/带宽 1MHz、脉宽 100ns、增益整数）。"""
    return {
        "r_freq": float(round(act["r_freq"] / 1e6) * 1e6),
        "r_bw": float(round(act["r_bw"] / 1e6) * 1e6),
        "r_pw": int(round(act["r_pw"] / 100) * 100),
        "r_gain": int(act["r_gain"]),
        "r_type": act["r_type"],
        "r_chirp": act["r_chirp"],
    }


def sample_action_seeded(rng: random.Random) -> dict:
    """sample_radar_action 使用全局 random，这里临时换种子后恢复。"""
    state = random.getstate()
    random.seed(rng.random())
    act = sample_radar_action()
    random.setstate(state)
    return snap_action(act)


def refine(best_act: dict, best_val: float, sinr_fn, jammer_row: pd.Series) -> tuple[dict, float, int]:
    """坐标细化：连续维双向步进（步长逐轮减半）+ 类型离散切换。"""
    n_evals = 0
    steps = {
        "r_freq": [20e6, 10e6, 5e6, 1e6],
        "r_gain": [5, 2, 1],
        "r_bw": [5e6, 2e6, 1e6],
        "r_pw": [1000, 300, 100],
    }
    for sweep in range(MAX_SWEEPS):
        improved = False
        # 类型（离散维）：固定当前连续参数试全部类型
        for t in RADAR_TYPES:
            if t == best_act["r_type"]:
                continue
            cand = {**best_act, "r_type": t}
            v = sinr_fn(jammer_row, make_radar_row(cand))
            n_evals += 1
            if v > best_val + MIN_IMPROVE_DB:
                best_act, best_val, improved = cand, v, True
        # 连续维
        for dim, levels in steps.items():
            step = levels[min(sweep, len(levels) - 1)]
            for sign in (+1, -1):
                cand = dict(best_act)
                if dim == "r_freq":
                    cand["r_freq"] = float(np.clip(best_act[dim] + sign * step,
                                                   RADAR_CONFIG["freq_min"], RADAR_CONFIG["freq_max"]))
                elif dim == "r_bw":
                    cand["r_bw"] = float(np.clip(best_act[dim] + sign * step,
                                                 RADAR_CONFIG["bw_min"], RADAR_CONFIG["bw_max"]))
                elif dim == "r_pw":
                    cand["r_pw"] = int(np.clip(best_act[dim] + sign * step,
                                               RADAR_CONFIG["pw_min"], RADAR_CONFIG["pw_max"]))
                else:
                    cand["r_gain"] = int(np.clip(best_act[dim] + sign * step,
                                                 RADAR_CONFIG["gain_min"], RADAR_CONFIG["gain_max"]))
                cand = snap_action(cand)
                if cand == best_act:
                    continue
                v = sinr_fn(jammer_row, make_radar_row(cand))
                n_evals += 1
                if v > best_val + MIN_IMPROVE_DB:
                    best_act, best_val, improved = cand, v, True
        if not improved:
            break
    return best_act, best_val, n_evals


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df = pd.read_csv(HOLDOUT_CSV)
    sids = holdout_df["scenario_id"].unique()[:N_HOLDOUT]
    print(f"留出场景: {len(sids)} 个, CSV={HOLDOUT_CSV.name}")

    backend = SignalSinrBackend(jsr_db=JSR_DB)
    sinr_fn = backend.sinr

    per_scenario = []
    rows = []
    for si, sid in enumerate(sids):
        block = holdout_df[holdout_df["scenario_id"] == sid]
        jammer_row = block.iloc[0]
        oracle_k20 = float(block["SINR_dB"].max())  # 与既有评估同源，做交叉校验

        rng = random.Random(GLOBAL_SEED + int(sid))
        # ---- 阶段1：best-of-N 随机 ----
        vals = np.empty(N_RANDOM)
        acts = [sample_action_seeded(rng) for _ in range(N_RANDOM)]
        for i, act in enumerate(acts):
            vals[i] = sinr_fn(jammer_row, make_radar_row(act))
        i_best = int(vals.argmax())
        best_act, best_val = acts[i_best], float(vals[i_best])
        rand_best, rand_mean = best_val, float(vals.mean())

        # ---- 阶段2：坐标细化 ----
        best_act, best_val, n_refine = refine(best_act, best_val, sinr_fn, jammer_row)

        per_scenario.append({
            "sid": int(sid),
            "jammer_type": str(jammer_row["Jammer_Type"]),
            "oracle_k20_db": oracle_k20,
            "random_best_db": rand_best,
            "random_mean_db": rand_mean,
            "dense_oracle_db": best_val,
            "refine_gain_db": best_val - rand_best,
            "n_random": N_RANDOM,
            "n_refine_evals": n_refine,
            "best_action": best_act,
        })
        rows.append({
            "scenario_id": int(sid),
            "jammer_type": str(jammer_row["Jammer_Type"]),
            "dense_oracle_db": round(best_val, 4),
            "random_best_db": round(rand_best, 4),
            "oracle_k20_db": round(oracle_k20, 4),
            "refine_gain_db": round(best_val - rand_best, 4),
            "r_freq_GHz": hz_to_ghz(best_act["r_freq"]),
            "r_gain_dB": best_act["r_gain"],
            "r_bw_GHz": hz_to_ghz(best_act["r_bw"]),
            "r_pw_ns": best_act["r_pw"],
            "r_type": best_act["r_type"],
        })
        d = per_scenario[-1]
        print(f"[{si + 1:2d}/{len(sids)}] sid={sid} {d['jammer_type']:<28s} "
              f"K20={oracle_k20:7.2f}  rand_best={rand_best:7.2f}  "
              f"dense={best_val:7.2f}  (+{best_val - rand_best:.2f} 细化)")

    dense = np.array([p["dense_oracle_db"] for p in per_scenario])
    k20 = np.array([p["oracle_k20_db"] for p in per_scenario])
    summary = {
        "protocol": {
            "n_holdout": N_HOLDOUT, "n_random": N_RANDOM, "max_sweeps": MAX_SWEEPS,
            "seed": GLOBAL_SEED, "jsr_db": JSR_DB, "backend": "SignalSinrBackend",
            "grid": "载频/带宽 1MHz、脉宽 100ns、增益 1dB、类型 4 选 1（与既有评估同口径）",
            "holdout_csv": str(HOLDOUT_CSV),
        },
        "dense_oracle": {
            "mean_db": float(dense.mean()), "median_db": float(np.median(dense)),
            "min_db": float(dense.min()), "max_db": float(dense.max()),
        },
        "reference": {
            "oracle_k20_mean_db": float(k20.mean()),
            "oracle_k20_check_note": "应与 Round2 报告的 17.25 dB 一致（同 CSV 同列）",
            "oracle_wide_mean_db_reported": "Round2 报告 23.18（100 随机，供对照）",
        },
        "backend_stats": backend.stats(),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (OUT_DIR / "dense_oracle_per_scenario.json").write_text(
        json.dumps({"summary": summary, "per_scenario": per_scenario},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "dense_oracle_actions.csv",
                              index=False, encoding="utf-8-sig")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
