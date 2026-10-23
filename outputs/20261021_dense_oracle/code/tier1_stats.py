# -*- coding: utf-8 -*-
"""第一梯队补充统计（20261021）：
  quick: 消融表真口径重算（4.1/4.2/4.3）+ 主表 bootstrap 95% CI  —— 纯重算，秒级
  wide : Oracle-Wide 非确定性量化 —— 每场景 1000 随机动作按 10 组×100 切分，
         给出 best-of-100 的 run-to-run 波动与 best-of-1000 更紧下界（~5 万次仿真）

所有口径：true_regret = 稠密近似 oracle − 推荐动作 SINR（上界口径）。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261021_dense_oracle/code/tier1_stats.py quick
    D:\\conda\\envs\\radar\\python.exe outputs/20261021_dense_oracle/code/tier1_stats.py wide
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]           # outputs/20261021_dense_oracle
RADAR_ROOT = HERE.parents[3]
sys.path.insert(0, str(RADAR_ROOT / "src"))
sys.path.insert(0, str(RADAR_ROOT / "outputs" / "20260909_signal_level" / "code"))

RES = ROUND / "results"
R2_DETAIL = RADAR_ROOT / "outputs" / "20260909_signal_level" / "eval_retrained" / "per_scenario_detail.json"
FS_DETAIL = RADAR_ROOT / "outputs" / "20261007_fewshot" / "eval_fewshot" / "per_scenario_detail.json"
FS_SUMMARY = RADAR_ROOT / "outputs" / "20261007_fewshot" / "eval_fewshot" / "summary.json"
ICDT_SUMMARY = ROUND / "eval_icdt_current" / "summary.json"
N_BOOT = 2000
BOOT_SEED = 20261021
WIDE_SEED = 20261022
N_WIDE_TOTAL, N_GROUPS = 1000, 10


def load_rec(path: Path, key: str) -> np.ndarray:
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(d[key]["recommended_sinr_db"], dtype=float)


def true_stats(rec: np.ndarray, dense: np.ndarray, q_only: np.ndarray) -> dict:
    reg = dense - rec
    return {
        "mean": round(float(reg.mean()), 3),
        "median": round(float(np.median(reg)), 3),
        "p95": round(float(np.percentile(reg, 95)), 3),
        "frac_at_oracle": round(float((reg <= 1e-6).mean()), 3),
        "win_vs_q_only": round(float((rec > q_only + 1e-9).mean()), 3),
        "n": int(len(reg)),
    }


def fmt_row(label: str, s: dict) -> str:
    return (f"| {label} | {s['mean']:.2f} | {s['median']:.2f} | {s['p95']:.2f} "
            f"| {s['frac_at_oracle']:.2f} | {s['win_vs_q_only']:.2f} |")


def load_context() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    doc = json.loads((RES / "dense_oracle_per_scenario.json").read_text(encoding="utf-8"))
    per_sc = doc["per_scenario"]
    sids = [p["sid"] for p in per_sc]
    dense = np.array([p["dense_oracle_db"] for p in per_sc])
    q_only = load_rec(R2_DETAIL, "q_only")
    return dense, q_only, np.array(sids), sids


def quick() -> None:
    dense, q_only, _, sids = load_context()
    fs_sum = json.loads(FS_SUMMARY.read_text(encoding="utf-8"))
    meta = fs_sum["per_scenario_meta"]
    med_sim = fs_sum["bucket_median_top1_sim"]
    med_nt = fs_sum["bucket_median_same_type_count"]
    top1 = np.asarray(meta["top1_cosine_sim"], dtype=float)
    ntype = np.asarray(meta["same_type_bank_count"], dtype=float)
    masks = {
        "rare_type": ntype <= med_nt, "common_type": ntype > med_nt,
        "low_sim": top1 < med_sim, "high_sim": top1 >= med_sim,
        "hard": (ntype <= med_nt) & (top1 < med_sim),
    }

    out: dict = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "ablations": {}, "main_ci": []}

    # ---- 4.1 悲观打分消融（真口径）----
    ab41 = {}
    for label, path, key in [
        ("Retrieve+Q-Union（单模型，无集成）", R2_DETAIL, "retrieve_q_union"),
        ("① Ensemble-Mean（集成均值，无悲观项）", FS_DETAIL, "ensemble_ctx_mean"),
        ("① Ensemble 悲观 μ−1.0σ", FS_DETAIL, "ensemble_ctx_pessimistic"),
    ]:
        ab41[label] = true_stats(load_rec(path, key), dense, q_only)
    out["ablations"]["pessimistic_scoring"] = ab41

    # ---- 4.2 上下文表达消融（真口径，含 hard 桶）----
    ab42 = {}
    for label, path, key in [
        ("Retrieve+Q（无上下文，Top-5）", R2_DETAIL, "retrieve_q"),
        ("Retrieve+Q-Ctx（z_ctx 加权均值）", R2_DETAIL, "retrieve_q_ctx"),
        ("Retrieve+Q-Union（基线，无 ctx）", R2_DETAIL, "retrieve_q_union"),
        ("② Set-Attention 单模型", FS_DETAIL, "attention_ctx"),
        ("①+② Attention-Ensemble 悲观", FS_DETAIL, "attention_ens_pessimistic"),
    ]:
        rec = load_rec(path, key)
        s = true_stats(rec, dense, q_only)
        s["hard_bucket_mean"] = round(float((dense - rec)[masks["hard"]].mean()), 3)
        ab42[label] = s
    out["ablations"]["context_expr"] = ab42

    # ---- 4.3 小样本分桶（真口径）----
    ab43 = {}
    for label, path, key in [
        ("q_only", R2_DETAIL, "q_only"),
        ("Retrieve+Q-Union", R2_DETAIL, "retrieve_q_union"),
        ("① Ensemble-Mean", FS_DETAIL, "ensemble_ctx_mean"),
        ("① Ensemble 悲观", FS_DETAIL, "ensemble_ctx_pessimistic"),
        ("② Set-Attention 单模型", FS_DETAIL, "attention_ctx"),
    ]:
        reg = dense - load_rec(path, key)
        ab43[label] = {b: {"n": int(m.sum()),
                           "mean": round(float(reg[m].mean()), 3),
                           "median": round(float(np.median(reg[m])), 3),
                           "win_vs_q_only": round(float((load_rec(path, key)[m] > q_only[m] + 1e-9).mean()), 3)}
                       for b, m in masks.items()}
    out["ablations"]["small_sample_buckets"] = ab43

    # ---- 主表 bootstrap 95% CI（mean 真 regret）----
    main_methods = [
        ("random（候选池内随机）", R2_DETAIL, "random"),
        ("q_only（无检索，随机 5A）", R2_DETAIL, "q_only"),
        ("retrieval_max_r（仅检索）", R2_DETAIL, "retrieval_max_r"),
        ("Retrieve+Q（Top-5）", R2_DETAIL, "retrieve_q"),
        ("Retrieve+Q-Union", R2_DETAIL, "retrieve_q_union"),
        ("① Ensemble-Mean", FS_DETAIL, "ensemble_ctx_mean"),
        ("① Ensemble 悲观 μ−1.0σ", FS_DETAIL, "ensemble_ctx_pessimistic"),
        ("② Set-Attention 单模型", FS_DETAIL, "attention_ctx"),
        ("①+② Attention-Ensemble 悲观", FS_DETAIL, "attention_ens_pessimistic"),
    ]
    if ICDT_SUMMARY.is_file():
        icdt = json.loads(ICDT_SUMMARY.read_text(encoding="utf-8"))["per_scenario"]
        main_methods.append(("ICDT 零观测（5-seed）", None, "icdt_m0"))
    rng = np.random.default_rng(BOOT_SEED)
    ci_rows = []
    for label, path, key in main_methods:
        rec = (np.asarray(icdt[key]["recommended_sinr_db"], dtype=float)
               if path is None else load_rec(path, key))
        reg = dense - rec
        boots = np.array([reg[rng.integers(0, len(reg), len(reg))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci_rows.append({"method": label, "key": key, "mean_true_regret": round(float(reg.mean()), 3),
                        "ci95_lo": round(float(lo), 3), "ci95_hi": round(float(hi), 3)})
    out["main_ci"] = ci_rows

    (RES / "tier1_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print("## 4.1 悲观打分消融（真口径）\n")
    print("| 变体 | 真regret均值 | 中位 | p95 | 达oracle | 胜率vs q_only |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k, v in ab41.items():
        print(fmt_row(k, v))
    print("\n## 4.2 上下文表达消融（真口径，含 hard 桶 n=9）\n")
    print("| 变体 | 真regret均值 | 中位 | hard桶均值 | 胜率vs q_only |")
    print("| --- | --- | --- | --- | --- |")
    for k, v in ab42.items():
        print(f"| {k} | {v['mean']:.2f} | {v['median']:.2f} | {v['hard_bucket_mean']:.2f} | {v['win_vs_q_only']:.2f} |")
    print("\n## 4.3 小样本分桶（真口径 mean）\n")
    print("| 方法 | rare_type(n=25) | common_type(n=25) | low_sim(n=25) | high_sim(n=25) | hard(n=9) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k, v in ab43.items():
        print(f"| {k} | {v['rare_type']['mean']:.2f} | {v['common_type']['mean']:.2f} "
              f"| {v['low_sim']['mean']:.2f} | {v['high_sim']['mean']:.2f} | {v['hard']['mean']:.2f} |")
    print(f"\n## 主表 bootstrap 95% CI（mean 真 regret, {N_BOOT} 次）\n")
    print("| 方法 | mean | 95% CI |")
    print("| --- | --- | --- |")
    for r in ci_rows:
        print(f"| {r['method']} | {r['mean_true_regret']:.2f} | [{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}] |")


def wide() -> None:
    from radar.data.high_contrast_sim import (RADAR_CONFIG, get_random_100_aligned,
                                              hz_to_ghz, sample_radar_action)
    from signal_sinr import SignalSinrBackend

    dense, _, _, sids = load_context()
    holdout_df = pd.read_csv(RADAR_ROOT / "data" / "v2" / "processed" / "holdout_eval_scenarios.csv")
    backend = SignalSinrBackend(jsr_db=10.0)

    def make_row(act: dict) -> pd.Series:
        return pd.Series({
            "Radar_CarrierFreq_GHz": hz_to_ghz(act["r_freq"]),
            "Radar_Gain_dB": act["r_gain"],
            "Radar_Bandwidth_GHz": hz_to_ghz(act["r_bw"]),
            "Radar_PulseWidth_ns": act["r_pw"],
            "Radar_Type": act["r_type"],
            "Radar_ChirpType": act["r_chirp"],
        })

    group_max = np.zeros((len(sids), N_GROUPS))
    best1000 = np.zeros(len(sids))
    for si, sid in enumerate(sids):
        jrow = holdout_df[holdout_df["scenario_id"] == sid].iloc[0]
        rng = random.Random(WIDE_SEED + int(sid))
        vals = np.empty(N_WIDE_TOTAL)
        for i in range(N_WIDE_TOTAL):
            st = random.getstate()
            random.seed(rng.random())
            act = sample_radar_action()
            random.setstate(st)
            vals[i] = backend.sinr(jrow, make_row(act))
        g = vals.reshape(N_GROUPS, N_WIDE_TOTAL // N_GROUPS)
        group_max[si] = g.max(axis=1)
        best1000[si] = vals.max()
        print(f"[{si + 1:2d}/{len(sids)}] sid={sid} best1000={best1000[si]:.2f} "
              f"group100 range=[{group_max[si].min():.2f}, {group_max[si].max():.2f}]", flush=True)

    run_means = group_max.mean(axis=0)  # 10 次"best-of-100 全场景均值"的实现
    summary = {
        "protocol": {"n_holdout": len(sids), "n_actions_per_scenario": N_WIDE_TOTAL,
                     "n_groups": N_GROUPS, "group_size": N_WIDE_TOTAL // N_GROUPS,
                     "seed": WIDE_SEED, "jsr_db": 10.0},
        "wide100_run_means": [round(float(x), 3) for x in run_means],
        "wide100_mean_of_runs": round(float(run_means.mean()), 3),
        "wide100_run_to_run_std": round(float(run_means.std(ddof=1)), 3),
        "wide100_range_across_runs": [round(float(run_means.min()), 3), round(float(run_means.max()), 3)],
        "best_of_1000_mean": round(float(best1000.mean()), 3),
        "note": ("wide100 口径的非确定性：10 次独立抽样的全场景均值 std；"
                 "best-of-1000 是稠密 oracle 随机阶段的更紧下界（细化前）"),
        "backend_stats": backend.stats(),
    }
    (RES / "oracle_wide_variability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    (wide if mode == "wide" else quick)()
