# -*- coding: utf-8 -*-
"""预实验（go/no-go）：干扰参数扰动下最优动作的 SINR 敏感性。

目的：验证「对抗鲁棒决策」章节的立论根基——若干扰参数存在测量误差/对抗性偏移，
在观测参数点上选出的最优动作会退化多少。

方法（只读旧数据，不改动任何旧文件）：
  1. 加载 Round2 信号级 CSV（5000 场景 × 20 动作），用解析仿真器确定性复算
     全部 10 万行 SINR（add_noise=False），与 SINR_analytic_dB 列做一致性校验
  2. 每场景取解析最优动作 a*
  3. 扰动半径 ρ ∈ {2%, 5%, 10%, 20%}，每场景采 n_dir=6 个随机扰动方向：
     - 载频/带宽/扫频步进/中断采样时长/延迟/假目标数：乘性因子 U(1-ρ, 1+ρ)，裁剪到物理范围
     - 中断采样周期 Intv 重算为 dur' + delay' + 200（保持公式一致性）
     - 干扰增益：加性 U(-10ρ, +10ρ) dB（增益全范围 20 dB，10% 半径 ≈ ±2 dB），裁剪 [20, 40]
     - 仅扰动激活参数（>0），未激活参数保持 0
  4. 扰动后重算：
     - 退化 = SINR(基线, a*) − SINR(扰动, a*)（正值 = 变差）
     - 决策翻转率 = 扰动后 20 动作重排后最优动作 ≠ 原最优动作 的比例
     - 静态 regret = max_a SINR(扰动, a) − SINR(扰动, a*)（≥0，鲁棒优化可压缩的量）
  5. 输出：按半径汇总表、按干扰类型分解表、退化曲线图、summary.json

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261021_robust_pre/code/pre_perturb_sensitivity.py
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

import numpy as np
import pandas as pd

from radar.data.high_contrast_sim import calculate_sinr_high_contrast, ghz_to_hz

CSV_PATH = (
    HERE.parents[2] / "20260909_signal_level" / "data"
    / "simulated_radar_jamming_high_contrast_10k_signal.csv"
)
RESULTS_DIR = ROUND / "results"

SEED = 2026
RADII = (0.02, 0.05, 0.10, 0.20)
N_DIR = 6

J_FREQ_LO, J_FREQ_HI = 3.0e9, 5.0e9
J_GAIN_LO, J_GAIN_HI = 20.0, 40.0
J_BW_LO, J_BW_HI = 5e6, 500e6
SWEEP_LO, SWEEP_HI = 1e6, 50e6
DUR_LO, DUR_HI = 2000.0, 20000.0
DELAY_LO, DELAY_HI = 1000.0, 10000.0
FT_LO, FT_HI = 1.0, 20.0

R_BANDWIDTH = {
    "jammer_swept": (200e6, 400e6),
    "jammer_wide_band": (200e6, 500e6),
    "jammer_narrow_band": (10e6, 30e6),
    "jammer_InterruptedSampling": (20e6, 20e6),
    "jammer_multi_false_targets": (20e6, 20e6),
    "jammer_close": (60e6, 60e6),
    "jammer_RANDOM": (10e6, 200e6),
}


def sinr_of(row, j):
    """按 CSV 行结构调用原仿真器（确定性）。row: 雷达动作参数 dict，j: 干扰参数 dict。"""
    return calculate_sinr_high_contrast(
        row["r_freq"], row["r_bw"], row["r_gain"],
        j["j_freq"], j["j_bw"], j["j_gain"], j["j_type"],
        r_type=row["r_type"], r_pw=row["r_pw"],
        j_sweep_step_hz=j["j_sweep_step_hz"],
        j_int_dur_ns=j["j_int_dur_ns"],
        j_int_delay_ns=j["j_int_delay_ns"],
        j_int_intv_ns=j["j_int_intv_ns"],
        j_ft_num=j["j_ft_num"],
        add_noise=False,
    )


def load_scenarios():
    df = pd.read_csv(CSV_PATH)
    n = len(df)
    assert n % 20 == 0, f"行数 {n} 不是 20 的倍数"

    # 一致性校验：抽样重算解析 SINR
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(n, size=3000, replace=False)
    diffs = []
    for i in sample_idx:
        r = df.iloc[i]
        s = calculate_sinr_high_contrast(
            ghz_to_hz(r["Radar_CarrierFreq_GHz"]),
            ghz_to_hz(r["Radar_Bandwidth_GHz"]),
            float(r["Radar_Gain_dB"]),
            ghz_to_hz(r["Jammer_CarrierFreq_GHz"]),
            ghz_to_hz(r["Jammer_Bandwidth_GHz"]),
            float(r["Jammer_Gain_dB"]),
            str(r["Jammer_Type"]),
            r_type=str(r["Radar_Type"]),
            r_pw=float(r["Radar_PulseWidth_ns"]),
            j_sweep_step_hz=ghz_to_hz(float(r["Jammer_Sweep_Step_GHz"])),
            j_int_dur_ns=float(r["Jammer_IntSample_Dur_ns"]),
            j_int_delay_ns=float(r["Jammer_IntSample_Delay_ns"]),
            j_int_intv_ns=float(r["Jammer_IntSample_Intv_ns"]),
            j_ft_num=float(r["Jammer_FalseTarget_Num"]),
            add_noise=False,
        )
        diffs.append(abs(s - float(r["SINR_analytic_dB"])))
    sanity = {"n_checked": int(len(diffs)), "max_abs_diff_db": float(np.max(diffs)),
              "mean_abs_diff_db": float(np.mean(diffs))}
    print(f"[sanity] 解析复算 vs SINR_analytic_dB: max|Δ|={sanity['max_abs_diff_db']:.4f} dB")

    # 组装场景结构：干扰参数（场景内恒定）+ 20 个动作
    scenarios = []
    for sid, g in df.groupby("scenario_id", sort=True):
        g0 = g.iloc[0]
        jam = {
            "j_type": str(g0["Jammer_Type"]),
            "j_freq": ghz_to_hz(float(g0["Jammer_CarrierFreq_GHz"])),
            "j_bw": ghz_to_hz(float(g0["Jammer_Bandwidth_GHz"])),
            "j_gain": float(g0["Jammer_Gain_dB"]),
            "j_sweep_step_hz": ghz_to_hz(float(g0["Jammer_Sweep_Step_GHz"])),
            "j_int_dur_ns": float(g0["Jammer_IntSample_Dur_ns"]),
            "j_int_delay_ns": float(g0["Jammer_IntSample_Delay_ns"]),
            "j_int_intv_ns": float(g0["Jammer_IntSample_Intv_ns"]),
            "j_ft_num": float(g0["Jammer_FalseTarget_Num"]),
        }
        actions = [
            {
                "r_freq": ghz_to_hz(float(r["Radar_CarrierFreq_GHz"])),
                "r_bw": ghz_to_hz(float(r["Radar_Bandwidth_GHz"])),
                "r_gain": float(r["Radar_Gain_dB"]),
                "r_pw": float(r["Radar_PulseWidth_ns"]),
                "r_type": str(r["Radar_Type"]),
            }
            for _, r in g.iterrows()
        ]
        base = np.array([sinr_of(a, jam) for a in actions])
        best = int(np.argmax(base))
        scenarios.append({"sid": int(sid), "jtype": jam["j_type"],
                          "jam": jam, "actions": actions,
                          "base": base, "best": best})
    return scenarios, sanity


def perturb_jam(jam, rho, rng):
    """按半径 ρ 生成一个扰动副本（仅激活参数被扰动）。"""
    j = dict(jam)
    j["j_freq"] = float(np.clip(jam["j_freq"] * rng.uniform(1 - rho, 1 + rho), J_FREQ_LO, J_FREQ_HI))
    j["j_gain"] = float(np.clip(jam["j_gain"] + rng.uniform(-10 * rho, 10 * rho), J_GAIN_LO, J_GAIN_HI))
    if jam["j_bw"] > 0:
        lo, hi = R_BANDWIDTH.get(jam["j_type"], (J_BW_LO, J_BW_HI))
        j["j_bw"] = float(np.clip(jam["j_bw"] * rng.uniform(1 - rho, 1 + rho), max(lo, J_BW_LO), min(hi, J_BW_HI)))
    if jam["j_sweep_step_hz"] > 0:
        j["j_sweep_step_hz"] = float(np.clip(jam["j_sweep_step_hz"] * rng.uniform(1 - rho, 1 + rho), SWEEP_LO, SWEEP_HI))
    if jam["j_int_dur_ns"] > 0:
        j["j_int_dur_ns"] = float(np.clip(jam["j_int_dur_ns"] * rng.uniform(1 - rho, 1 + rho), DUR_LO, DUR_HI))
    if jam["j_int_delay_ns"] > 0:
        j["j_int_delay_ns"] = float(np.clip(jam["j_int_delay_ns"] * rng.uniform(1 - rho, 1 + rho), DELAY_LO, DELAY_HI))
    if jam["j_int_dur_ns"] > 0 or jam["j_int_delay_ns"] > 0:
        j["j_int_intv_ns"] = j["j_int_dur_ns"] + j["j_int_delay_ns"] + 200.0
    if jam["j_ft_num"] > 0:
        j["j_ft_num"] = float(np.clip(round(jam["j_ft_num"] * rng.uniform(1 - rho, 1 + rho)), FT_LO, FT_HI))
    return j


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    scenarios, sanity = load_scenarios()
    n_scen = len(scenarios)
    print(f"[data] 场景数={n_scen}，加载耗时 {time.time() - t0:.1f}s")

    rows_out = []          # 每个 (sid, rho, dir) 一行
    for rho in RADII:
        t_rho = time.time()
        for sid_i, sc in enumerate(scenarios):
            for d in range(N_DIR):
                jp = perturb_jam(sc["jam"], rho, rng)
                a_old = sc["actions"][sc["best"]]
                s_old = sinr_of(a_old, jp)
                pool = np.array([sinr_of(a, jp) for a in sc["actions"]])
                new_best = int(np.argmax(pool))
                rows_out.append({
                    "scenario_id": sc["sid"],
                    "jammer_type": sc["jtype"],
                    "rho": rho,
                    "dir": d,
                    "base_sinr": float(sc["base"][sc["best"]]),
                    "pert_sinr_best": float(s_old),
                    "pert_oracle": float(pool.max()),
                    "new_best_idx": new_best,
                    "flip": int(new_best != sc["best"]),
                    "degradation": float(sc["base"][sc["best"]] - s_old),
                    "static_regret": float(pool.max() - s_old),
                })
        print(f"[rho={rho:.0%}] 完成，耗时 {time.time() - t_rho:.1f}s")

    detail = pd.DataFrame(rows_out)
    detail.to_csv(RESULTS_DIR / "perturb_detail.csv", index=False)

    # ---- 按半径汇总（含最坏方向视角）----
    agg = detail.groupby("rho").agg(
        mean_degradation=("degradation", "mean"),
        median_degradation=("degradation", "median"),
        p95_degradation=("degradation", lambda x: np.percentile(x, 95)),
        mean_static_regret=("static_regret", "mean"),
        median_static_regret=("static_regret", "median"),
        flip_rate=("flip", "mean"),
    ).reset_index()
    worst = detail.groupby(["scenario_id", "rho"]).agg(
        worst_degradation=("degradation", "max"),
        worst_static_regret=("static_regret", "max"),
    ).reset_index().groupby("rho").agg(
        mean_worst_degradation=("worst_degradation", "mean"),
        mean_worst_static_regret=("worst_static_regret", "mean"),
    ).reset_index()
    by_rho = agg.merge(worst, on="rho")
    by_rho.to_csv(RESULTS_DIR / "sensitivity_by_radius.csv", index=False)

    # ---- 按干扰类型分解（ρ=10% 代表性半径）----
    rep = detail[detail["rho"] == 0.10]
    by_type = rep.groupby("jammer_type").agg(
        mean_degradation=("degradation", "mean"),
        median_degradation=("degradation", "median"),
        p95_degradation=("degradation", lambda x: np.percentile(x, 95)),
        mean_static_regret=("static_regret", "mean"),
        flip_rate=("flip", "mean"),
        n=("degradation", "size"),
    ).reset_index().sort_values("mean_degradation", ascending=False)
    by_type.to_csv(RESULTS_DIR / "sensitivity_by_type_rho10.csv", index=False)

    summary = {
        "seed": SEED,
        "n_scenarios": n_scen,
        "n_dir": N_DIR,
        "radii": list(RADII),
        "sanity": sanity,
        "by_radius": by_rho.to_dict(orient="records"),
        "by_type_rho10": by_type.to_dict(orient="records"),
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        x = by_rho["rho"] * 100

        axes[0].plot(x, by_rho["mean_degradation"], "o-", label="mean")
        axes[0].plot(x, by_rho["median_degradation"], "s-", label="median")
        axes[0].plot(x, by_rho["p95_degradation"], "^--", label="p95")
        axes[0].plot(x, by_rho["mean_worst_degradation"], "v--", label="mean worst-dir")
        axes[0].set_xlabel("扰动半径 ρ (%)")
        axes[0].set_ylabel("最优动作 SINR 退化 (dB)")
        axes[0].set_title("SINR 退化 vs 扰动半径")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(x, by_rho["flip_rate"], "o-")
        axes[1].set_xlabel("扰动半径 ρ (%)")
        axes[1].set_ylabel("决策翻转率")
        axes[1].set_title("最优动作易主比例 vs 扰动半径")
        axes[1].grid(alpha=0.3)

        axes[2].plot(x, by_rho["mean_static_regret"], "o-", label="mean")
        axes[2].plot(x, by_rho["median_static_regret"], "s-", label="median")
        axes[2].plot(x, by_rho["mean_worst_static_regret"], "v--", label="mean worst-dir")
        axes[2].set_xlabel("扰动半径 ρ (%)")
        axes[2].set_ylabel("静态动作 regret (dB)")
        axes[2].set_title("静态最优动作的 regret vs 扰动半径")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "degradation_curves.png", dpi=150)
        print("[plot] 已保存 degradation_curves.png")
    except Exception as e:  # 画图失败不影响结论
        print(f"[plot] 跳过: {e}")

    print(json.dumps(summary["by_radius"], ensure_ascii=False, indent=2))
    print(f"[done] 总耗时 {time.time() - t0:.1f}s，结果目录: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
