# -*- coding: utf-8 -*-
"""第二轮验证（修正版）：同场景动作排序一致性。

相对初版的关键修正：
  * 干扰功率 = 场景内固定绝对功率（按场景各动作回波功率的中位数标定 JSR），
    而非随各动作回波浮动——真实干扰机发射功率固定，雷达增益高则回波强、SINR 好，
    解析公式同样奖励高增益，二者才有可比性。
  * 干扰形状为零（频偏完全出带）时跳过标定，避免 NaN。
  * Radar_Gain_dB 接入 tx_power（回波功率随雷达增益变化）。
  * 前端信道滤波带 15% BW 滚降过渡带。

对每个抽样场景（固定干扰状态 S，≤20 个动作 A）：
  - 每动作算解析 SINR（add_noise=False）与信号级 SINR（JSR 10/20 dB 两档）
  - 场景内 Spearman(解析, 信号级)、Top-1 命中、Top-5 交集

运行:
    D:\conda\envs\radar\python.exe radar/scripts/validate_round2_scenario_ranking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_radarsimpy_labels as v1  # noqa: E402
from validate_radarsimpy_labels import (  # noqa: E402
    C, CSV_PATH, OUT_DIR, SEED, SKIP_FAMILIES, TARGET_RCS_DBSM, TARGET_RANGE,
    damage_metrics, make_ref, matched_filter, radar_family,
)

sinr_from_csv_row = v1.sinr_from_csv_row
from radarsimpy import sim_radar  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

N_SCEN_PER_TYPE = 3
JSR_LEVELS_DB = (10.0, 20.0)
ROLLOFF = 0.15           # 过渡带占 BW 比例

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def soft_channel_filter(x_fine: np.ndarray, fs_fine: float, fs: float,
                        n_samp: int, bw: float) -> np.ndarray:
    """雷达信道滤波：通带 ±BW/2，15% BW 过渡带滚降，再降采样。"""
    spec = np.fft.fft(x_fine)
    freqs = np.fft.fftfreq(len(x_fine), 1 / fs_fine)
    af = np.abs(freqs)
    edge = bw / 2
    trans = ROLLOFF * bw
    rise = np.clip((af - edge) / trans, 0, 1)
    denom = np.clip(1.0 - rise, 1e-9, None)
    gain = 1.0 / (1.0 + (rise / denom) ** 4)
    gain[af > edge + trans] = 0.0
    spec *= gain
    return np.fft.ifft(spec)[: n_samp * 4 : 4]


def build_radar_gain(row: pd.Series, fs: float):
    """tx_power 接入 Radar_Gain_dB，使回波功率随雷达增益变化。"""
    fc = float(row["Radar_CarrierFreq_GHz"]) * 1e9
    bw = float(row["Radar_Bandwidth_GHz"]) * 1e9
    pw = float(row["Radar_PulseWidth_ns"]) * 1e-9
    fam = radar_family(row["Radar_Type"])
    if fam == "线性调频":
        tx_f, tx_t = [fc - bw / 2, fc + bw / 2], None
    else:
        t_tx = np.linspace(0, pw, 2048)
        if fam == "非线性调频":
            tx_f, tx_t = fc + v1.f_rel_nlfm(t_tx, bw), t_tx
        else:
            tx_f, tx_t = (fc + v1.f_rel_freqcoded(t_tx, bw, seed=int(row["scenario_id"])),
                          t_tx)
    g = float(row["Radar_Gain_dB"])
    tx = (v1.Transmitter(f=tx_f, t=pw, tx_power=g, pulses=1)
          if tx_t is None else v1.Transmitter(f=tx_f, t=tx_t, tx_power=g, pulses=1))
    rx = v1.Receiver(fs=fs, noise_figure=8, rf_gain=30, baseband_gain=40)
    return v1.Radar(transmitter=tx, receiver=rx, seed=SEED)


def prepare_action(row: pd.Series) -> dict:
    """单动作：仿真回波 + 干扰形状（单位功率）+ 解析 SINR。"""
    bw = float(row["Radar_Bandwidth_GHz"]) * 1e9
    pw = float(row["Radar_PulseWidth_ns"]) * 1e-9
    fs = 2 * bw

    radar = build_radar_gain(row, fs)
    data = sim_radar(
        radar,
        [{"location": (TARGET_RANGE, 0, 0), "rcs": TARGET_RCS_DBSM,
          "speed": (0, 0, 0), "phase": 0}],
        density=1, device="cpu",
    )
    bb_raw = np.asarray(data["baseband"])[0, 0]
    n = min(int(np.ceil(pw * fs)), len(bb_raw))
    bb = bb_raw[:n]
    ref_bb = make_ref(row, fs, n)
    delay_bin = int(round(2 * TARGET_RANGE / C * fs))

    v1._frontend_filter = lambda x, fF, f, ns: soft_channel_filter(x, fF, f, ns, bw)
    np.random.seed(SEED + int(row["action_idx"]))
    jam_unit = v1.synth_jamming(row, fs, n)
    p_jam = float(np.mean(np.abs(jam_unit) ** 2)) if np.abs(jam_unit).max() > 0 else 0.0

    return {
        "bb": bb, "ref": ref_bb, "delay_bin": delay_bin,
        "clean_sinr": damage_metrics(matched_filter(bb, ref_bb), delay_bin)["sinr_db"],
        "jam_unit": jam_unit if p_jam > 0 else None,
        "p_echo": float(np.mean(np.abs(bb[max(0, delay_bin - 50): delay_bin + 50]) ** 2)),
        "analytic": sinr_from_csv_row(row, row, add_noise=False),
        "bw": bw,
    }


def eval_at(act: dict, jam_power: float) -> tuple[float, float]:
    """在给定干扰绝对功率下评估，返回 (sinr_eff_db, sinr_jam_db)。"""
    if act["jam_unit"] is None or jam_power <= 0:
        return act["clean_sinr"], act["clean_sinr"]
    jam = act["jam_unit"] / np.sqrt(np.mean(np.abs(act["jam_unit"]) ** 2)) * np.sqrt(jam_power)
    m = damage_metrics(matched_filter(act["bb"] + jam, act["ref"]), act["delay_bin"])
    return m["sinr_eff_db"], m["sinr_db"]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["fam"] = df["Radar_Type"].map(radar_family)
    usable = df[~df["fam"].isin(SKIP_FAMILIES)]

    parts = [
        g.sample(min(N_SCEN_PER_TYPE, g["scenario_id"].nunique()),
                 random_state=SEED)["scenario_id"].unique()
        for _, g in usable.groupby("Jammer_Type")
    ]
    scen_ids = np.concatenate(parts)
    print(f"抽样 {len(scen_ids)} 个场景 × ≤20 动作, JSR 档位 {JSR_LEVELS_DB} dB"
          f"（场景内固定绝对干扰功率）")

    rows = []
    for k, sid in enumerate(scen_ids):
        grp = usable[usable["scenario_id"] == sid]
        jtype = str(grp["Jammer_Type"].iloc[0])
        acts = [prepare_action(r) for _, r in grp.iterrows()]
        p_ref = float(np.median([a["p_echo"] for a in acts]))
        for jsr in JSR_LEVELS_DB:
            jam_power = p_ref * 10 ** (jsr / 10)
            recs = [eval_at(a, jam_power) for a in acts]
            r = pd.DataFrame(
                {"analytic": [a["analytic"] for a in acts],
                 "eff": [x[0] for x in recs],
                 "jam": [x[1] for x in recs],
                 "clean": [a["clean_sinr"] for a in acts]}
            )
            sp = float(spearmanr(r["analytic"], r["eff"]).statistic)
            top1_a, top1_s = int(r["analytic"].idxmax()), int(r["eff"].idxmax())
            top5_a = set(r["analytic"].nlargest(5).index)
            top5_s = set(r["eff"].nlargest(5).index)
            rows.append({
                "scenario_id": int(sid), "jammer_type": jtype, "jsr_db": jsr,
                "n_actions": len(r), "spearman": sp,
                "top1_match": int(top1_a == top1_s),
                "top5_overlap": len(top5_a & top5_s) / 5,
            })
        print(f"[{k+1}/{len(scen_ids)}] 场景 {sid} ({jtype}) 完成")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "round2_scenario_ranking.csv", index=False)

    print("\n=== 场景内排序一致性（JSR=10dB 主档, 场景内固定干扰功率） ===")
    main_lvl = res[res["jsr_db"] == JSR_LEVELS_DB[0]]
    for jtype, g in main_lvl.groupby("jammer_type"):
        print(f"{jtype:32s} 场景={len(g)}  Spearman均值={g['spearman'].mean():+.3f}  "
              f"Top-1命中={g['top1_match'].mean()*100:.0f}%  "
              f"Top-5重合={g['top5_overlap'].mean()*100:.0f}%")
    m_sp, m_t1 = main_lvl["spearman"].mean(), main_lvl["top1_match"].mean()
    print(f"\n总体(主档): Spearman均值={m_sp:+.3f}  Top-1命中={m_t1*100:.0f}%  "
          f"Top-5重合={main_lvl['top5_overlap'].mean()*100:.0f}%")
    lvl20 = res[res["jsr_db"] == JSR_LEVELS_DB[1]]
    print(f"参照(JSR=20dB): Spearman均值={lvl20['spearman'].mean():+.3f}  "
          f"Top-1命中={lvl20['top1_match'].mean()*100:.0f}%")
    verdict = ("标签可用：排序结构基本一致" if (m_sp >= 0.7 and m_t1 >= 0.6) else
               "部分可信：粗排序可用、细排序不可" if (m_sp >= 0.4) else
               "应替换标签：解析排序与信号级排序系统性不符")
    print(f"结论: {verdict}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    types = main_lvl["jammer_type"].unique()
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(types):
        g = main_lvl[main_lvl["jammer_type"] == t]
        axes[0].scatter([t] * len(g), g["spearman"], color=cmap(i % 10), s=60,
                        alpha=0.8, label=t)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axhline(0.7, color="g", ls="--", alpha=0.6, label="0.7 可用线")
    axes[0].axhline(0.4, color="orange", ls="--", alpha=0.6, label="0.4 及格线")
    axes[0].set_ylabel("场景内 Spearman(解析, 信号级)")
    axes[0].set_title("每场景 20 动作排序一致性 (JSR=10dB)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    ex = main_lvl.sort_values("spearman").iloc[-1]
    ex_rows = usable[usable["scenario_id"] == ex["scenario_id"]]
    acts = [prepare_action(r) for _, r in ex_rows.iterrows()]
    jam_power = float(np.median([a["p_echo"] for a in acts])) * 10 ** (JSR_LEVELS_DB[0] / 10)
    pts = []
    for a in acts:
        eff, _ = eval_at(a, jam_power)
        pts.append((a["analytic"], eff))
    pts = np.array(pts)
    axes[1].scatter(pts[:, 0], pts[:, 1], c="steelblue", s=50)
    for j in range(len(pts)):
        axes[1].annotate(str(j), pts[j], fontsize=7, alpha=0.7)
    axes[1].set_xlabel("解析 SINR (dB)")
    axes[1].set_ylabel("信号级 SINR (dB)")
    axes[1].set_title(f"示例场景 {int(ex['scenario_id'])} ({ex['jammer_type']}, "
                      f"Spearman={ex['spearman']:+.2f})")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "round2_summary.png", dpi=140)
    print(f"\n产物: {OUT_DIR / 'round2_scenario_ranking.csv'}")
    print(f"      {OUT_DIR / 'round2_summary.png'}")


if __name__ == "__main__":
    main()
