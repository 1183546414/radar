# -*- coding: utf-8 -*-
"""雷达抗干扰项目 — 实验结果图汇总（2026-10-23）。

从各轮实验产物（JSON/CSV）生成系列结果图，保存到 outputs/20261023_results_figures/。
运行: PYTHONPATH= 无需；直接 python make_result_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OUT.mkdir(parents=True, exist_ok=True)

R22 = ROOT / "outputs" / "20261022_comprehensive" / "results"
HOLD_NEW = ROOT / "outputs" / "q_value" / "holdout_baselines"
HOLD_OLD = ROOT / "outputs" / "archive_pre_retrain_20261022" / "holdout_baselines"

# ---------- 全局风格 ----------
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.size": 10.5,
})

C = {
    "random": "#B8B8B8",
    "q_only": "#8C8C8C",
    "retrieval_max_r": "#8172B3",
    "retrieve_q": "#4C72B0",
    "retrieve_q_union": "#DD8452",
    "ensemble_ctx_mean": "#55A868",
    "ensemble_ctx_pessimistic": "#C44E52",
    "oracle_wide": "#31708E",
}
ZH = {
    "random": "Random",
    "q_only": "Q-only",
    "retrieval_max_r": "Retrieval-only",
    "retrieve_q": "Retrieve+Q",
    "retrieve_q_union": "Retrieve+Q-Union",
    "ensemble_ctx_mean": "Ensemble-Mean",
    "ensemble_ctx_pessimistic": "Ensemble-悲观",
    "oracle_wide": "Oracle-Wide",
}


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name: str, note: str):
    fig.text(0.01, 0.005, note, fontsize=7, color="#777777", ha="left")
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------- 图1：fresh450 核心方法对比（弱参照 regret） ----------
def fig01():
    d = load_json(R22 / "core_500" / "summary.json")
    methods = [m for m in d["methods"]]
    names = [m["method"] for m in methods]
    labels = [ZH[n] for n in names]
    stats = ["mean", "median", "p95", "cvar10"]
    stat_zh = ["均值", "中位数", "p95", "CVaR-10%"]
    x = np.arange(len(methods))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, (s, sz) in enumerate(zip(stats, stat_zh)):
        vals = [m["fresh450"][s] for m in methods]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=sz, color=plt.cm.viridis(0.15 + i * 0.22))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.35 if v >= 0 else -1.1),
                    f"{v:.2f}", ha="center", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("弱参照 regret（dB，vs Oracle-K10）")
    ax.set_title("核心方法对比 — 450 个未用于历史报告的留出场景（20261022 全面实验）")
    ax.legend(ncol=4, fontsize=9)
    ax.set_ylim(min(-6.2, ax.get_ylim()[0]), max(27, ax.get_ylim()[1]))
    savefig(fig, "fig01_核心方法对比_fresh450.png",
            "数据: outputs/20261022_comprehensive/results/core_500/summary.json | 评估: 450 新留出场景, JSR=10dB, regret 相对留出随机 10 动作弱参照; 负值表示超过该参照")


# ---------- 图2：250 场景真 regret（稠密 oracle） ----------
def fig02():
    d = load_json(R22 / "final_summary.json")["true_regret_fresh250"]
    stats = ["mean_true_regret", "median_true_regret", "p90", "p95", "cvar10"]
    stat_zh = ["均值", "中位数", "p90", "p95", "CVaR-10%"]
    x = np.arange(len(d))
    w = 0.16
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, (s, sz) in enumerate(zip(stats, stat_zh)):
        vals = [m[s] for m in d]
        bars = ax.bar(x + (i - 2) * w, vals, w, label=sz, color=plt.cm.viridis(0.1 + i * 0.2))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [ZH[m["method"]] for m in d])
    ax.set_ylabel("真 regret（dB，vs 稠密近似 oracle）")
    ax.set_title("真 regret — 250 个新场景稠密 oracle（best-of-5000 + 8 轮坐标细化，94 万次仿真）")
    ax.legend(ncol=5, fontsize=9)
    savefig(fig, "fig02_真regret_稠密oracle250.png",
            "数据: outputs/20261022_comprehensive/results/final_summary.json | 稠密 oracle 为真最优下界, 故真 regret 为严格上界; Ensemble-Mean 均值最低但与 Union 差仅 0.14 dB")


# ---------- 图3：配对检验森林图（vs Union） ----------
def fig03():
    d = load_json(R22 / "final_summary.json")["paired"]["results"]
    d = [m for m in d if m["method"] != "retrieve_q_union"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ys = np.arange(len(d))[::-1]
    ax.set_ylim(-0.55, len(d) - 0.45)
    fig.subplots_adjust(bottom=0.18)
    for y, m in zip(ys, d):
        lo, hi = m["delta_ci95"]
        mean = m["mean_delta_sinr_vs_union"]
        color = C.get(m["method"], "#666666")
        ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(mean, y, "o", color=color, ms=8)
        star = "***" if m["permutation_p_holm"] < 0.001 else ("*" if m["permutation_p_holm"] < 0.05 else "n.s.")
        ax.text(hi + 0.15, y, f"{star} (Holm p={m['permutation_p_holm']:.3g})",
                va="center", fontsize=8.5)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(ys, [ZH[m["method"]] for m in d])
    ax.set_xlabel("相对 Retrieve+Q-Union 的平均 SINR 差（dB，95% bootstrap CI）")
    ax.set_title("配对统计检验 — 450 个新场景，参照 = Retrieve+Q-Union")
    lo_all = min(m["delta_ci95"][0] for m in d) - 1.2
    ax.set_xlim(lo_all, max(m["delta_ci95"][1] for m in d) + 3.4)
    savefig(fig, "fig03_配对检验_forest_vs_Union.png",
            "数据: outputs/20261022_comprehensive/results/final_summary.json | 10000 bootstrap + 100000 符号置换 + Wilcoxon, Holm 校正; Ensemble 均值差异不显著")


# ---------- 图4：超参扫描（λ / Top-K / 随机池） ----------
def fig04():
    rows = list(csv.DictReader(open(R22 / "sweeps" / "summary.csv", encoding="utf-8-sig")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    panels = [
        ("lambda", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0], "悲观系数 λ", (0, 0.5)),
        ("top_k", [1, 3, 5, 10, 20], "检索 Top-K", (3, 1)),
        ("pool", [5, 20, 50, 100, 500], "随机候选池大小", (4, 1)),
    ]
    for ax, (sweep, vals, title, star) in zip(axes, panels):
        sub = [r for r in rows if r["sweep"] == sweep]
        xs = [float(r[sweep]) for r in sub]
        mean = [float(r["fresh450.mean"]) for r in sub]
        p95 = [float(r["fresh450.p95"]) for r in sub]
        ax.plot(xs, mean, "o-", color="#4C72B0", lw=2, label="mean regret")
        ax.axvline(xs[star[0]], color="#DD8452", ls="--", lw=1.2, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(xs, p95, "s--", color="#55A868", lw=1.5, label="p95 regret")
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
        ax2.set_ylabel("p95（dB）", color="#55A868")
        ax.set_xscale("log" if sweep == "pool" else "linear")
        ax.set_xticks(xs, [f"{v:g}" for v in vals], fontsize=8.5,
                      rotation=15 if sweep == "lambda" else 0,
                      ha="right" if sweep == "lambda" else "center")
        ax.set_xlabel(title)
        ax.set_ylabel("fresh450 mean regret（dB）", color="#4C72B0")
        ax.set_title(title)
        ax.annotate("最优", (xs[star[0]], mean[star[0]]), xytext=(0, 12),
                    textcoords="offset points", ha="center", color="#DD8452", fontsize=9, weight="bold")
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("超参数扫描 — 冻结 5-seed Ensemble，450 个新场景（20261022）", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig04_超参扫描_lambda_topk_pool.png",
            "数据: outputs/20261022_comprehensive/results/sweeps/summary.csv | 最优: λ=0.5(-4.27dB), Top-K=20(-4.77dB), 随机池=500(-5.01dB); 历史默认 Top-K=5/池=100/λ=1 非最优")


# ---------- 图5：JSR 跨条件测试 ----------
def fig05():
    rows = list(csv.DictReader(open(R22 / "sweeps" / "summary.csv", encoding="utf-8-sig")))
    jsr_rows = [r for r in rows if r["sweep"] == "jsr"]
    base = [r for r in rows if r["sweep"] == "lambda" and float(r["lambda"]) == 1.0][0]
    jsr = [0.0, 5.0, 10.0, 15.0, 20.0]
    data = {float(r["jsr_db"]): r for r in jsr_rows}
    data[10.0] = base
    mean = [float(data[j]["fresh450.mean"]) for j in jsr]
    p95 = [float(data[j]["fresh450.p95"]) for j in jsr]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(jsr, mean, "o-", color="#4C72B0", lw=2, label="mean regret")
    for xx, yy in zip(jsr, mean):
        ax.text(xx, yy + 0.006, f"{yy:.2f}", ha="center", va="bottom", fontsize=8.5, color="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(jsr, p95, "s--", color="#C44E52", lw=1.8, label="p95 regret")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.set_ylabel("p95（dB）", color="#C44E52")
    ax.set_ylim(-4.335, -4.135)
    ax2.set_ylim(1.6, 2.35)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right")
    ax.set_xlabel("干信比 JSR（dB）")
    ax.set_ylabel("fresh450 mean regret（dB）", color="#4C72B0")
    ax.set_title("JSR 跨条件测试 — 冻结策略（Top-K=5, 池=100, λ=1）")
    ax.set_xticks(jsr)
    savefig(fig, "fig05_JSR跨条件测试.png",
            "数据: outputs/20261022_comprehensive/results/sweeps/summary.csv | 冻结策略在 JSR 0-20dB 内相对优势稳定(-4.17~-4.29dB); 跨条件测试, 非多 JSR 联合训练")


# ---------- 图6：鲁棒性 ----------
def fig06():
    rows = list(csv.DictReader(open(R22 / "robust_formal" / "summary_by_radius.csv", encoding="utf-8-sig")))
    rho = [float(r["rho"]) * 100 for r in rows]
    mean_d = [float(r["mean_worst_degradation"]) for r in rows]
    p95_d = [float(r["p95_worst_degradation"]) for r in rows]
    cvar = [float(r["cvar10_worst_degradation"]) for r in rows]
    flip = [float(r["any_flip_rate"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(rho, mean_d, "o-", color="#4C72B0", lw=2, label="最坏方向平均退化")
    ax.plot(rho, p95_d, "s-", color="#C44E52", lw=2, label="p95 退化")
    ax.plot(rho, cvar, "^-", color="#8172B3", lw=2, label="CVaR-10% 退化")
    for xx, yy in zip(rho, mean_d):
        ax.text(xx, yy + 0.25, f"{yy:.2f}", ha="center", fontsize=8.5, color="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(rho, flip, "d--", color="#55A868", lw=2, label="动作翻转率")
    for xx, yy in zip(rho, flip):
        ax2.text(xx, yy + 1.5, f"{yy:.1f}%", ha="center", fontsize=8.5, color="#55A868")
    ax2.set_ylabel("至少一次最优动作翻转率（%）", color="#55A868")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_xlabel("干扰参数扰动半径 ρ（%）")
    ax.set_ylabel("最坏方向性能退化（dB）")
    ax.set_title("正式鲁棒性评估 — 5000 训练场景，20 随机方向 + 坐标边界搜索")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    savefig(fig, "fig06_鲁棒性_扰动退化.png",
            "数据: outputs/20261022_comprehensive/results/robust_formal/summary_by_radius.csv | 10% 扰动: 平均退化 1.83dB, p95 8.18dB, 57% 场景最优动作翻转 → 鲁棒决策为主要后续方向")


# ---------- 图7：按干扰类型热图 ----------
def fig07():
    rows = list(csv.DictReader(open(R22 / "core_500" / "by_jammer_type.csv", encoding="utf-8-sig")))
    methods = ["q_only", "retrieve_q", "retrieve_q_union", "ensemble_ctx_mean"]
    types = sorted({r["jammer_type"] for r in rows})
    type_zh = {t: t.replace("jammer_", "") for t in types}
    rows = [r for r in rows if r["method"] in methods]
    M = np.zeros((len(methods), len(types)))
    for r in rows:
        M[methods.index(r["method"]), types.index(r["jammer_type"])] = float(r["mean"])
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    fig.subplots_adjust(bottom=0.24)
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(types)), [type_zh[t] for t in types], rotation=20, ha="right")
    ax.set_yticks(range(len(methods)), [ZH[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(types)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(M[i, j]) > vmax * 0.55 else "black")
    ax.set_title("按干扰类型的弱参照 regret（dB）— 500 留出场景（20261022）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean regret（dB）")
    savefig(fig, "fig07_按干扰类型_regret热图.png",
            "数据: outputs/20261022_comprehensive/results/core_500/by_jammer_type.csv | 红色=regret 高(差), 蓝色=超过弱参照; close(近距离压制) 类型对所有方法最难")


# ---------- 图8：修正结构重训前后对比（同协议 500 场景） ----------
def fig08():
    new_rows = list(csv.DictReader(open(HOLD_NEW / "summary.csv", encoding="utf-8-sig")))
    old_rows = list(csv.DictReader(open(HOLD_OLD / "summary.csv", encoding="utf-8-sig")))
    order = ["random", "retrieval_max_r", "q_only", "retrieve_q_union", "oracle_wide"]
    new_m = {r["method"]: r for r in new_rows}
    old_m = {r["method"]: r for r in old_rows}
    new_j = load_json(HOLD_NEW / "summary.json")
    old_j = load_json(HOLD_OLD / "summary.json")
    new_wr = {k: v["win_rate"] * 100 for k, v in new_j["paired_vs_q_only"]["results"].items()}
    old_wr = {k: v["win_rate"] * 100 for k, v in old_j["paired_vs_q_only"]["results"].items()}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    x = np.arange(len(order))
    w = 0.38
    panels = [
        ("mean_recommended_sinr_db", "推荐 SINR（dB）", 1),
        ("mean_regret_vs_k20_db", "mean regret vs Reference-K10（dB）", 0),
    ]
    for ax, (col, title, higher_better) in zip(axes[:2], panels):
        ov = [float(old_m[m][col]) for m in order]
        nv = [float(new_m[m][col]) for m in order]
        b1 = ax.bar(x - w / 2, ov, w, label="修正前结构（20261022 归档）", color="#B8B8B8")
        b2 = ax.bar(x + w / 2, nv, w, label="修正后重训（20261023）", color="#DD8452")
        for bars in (b1, b2):
            for b in bars:
                v = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, v + (0.08 if v >= 0 else -0.3),
                        f"{v:.2f}", ha="center", fontsize=7.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [ZH[m] for m in order], rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylim(min(ov + nv) - 1.2, max(ov + nv) + 1.2)
    axes[0].legend(fontsize=8.5)
    ax = axes[2]
    methods_wr = ["random", "retrieval_max_r", "retrieve_q_union", "oracle_wide"]
    x2 = np.arange(len(methods_wr))
    ov = [old_wr[m] for m in methods_wr]
    nv = [new_wr[m] for m in methods_wr]
    b1 = ax.bar(x2 - w / 2, ov, w, color="#B8B8B8")
    b2 = ax.bar(x2 + w / 2, nv, w, color="#DD8452")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(x2, [ZH[m] for m in methods_wr], rotation=15, ha="right")
    ax.set_ylabel("配对胜率（%）")
    ax.set_title("对 Q-only 配对胜率")
    ax.set_ylim(0, 108)
    fig.suptitle("算法结构修正前后留出评估对比 — 同协议 500 场景 × K=10（20261023 重训）", y=1.03)
    fig.tight_layout()
    savefig(fig, "fig08_修正结构_重训前后对比.png",
            "数据: outputs/q_value/holdout_baselines vs outputs/archive_pre_retrain_20261022/holdout_baselines | 主方法 Union: 推荐 SINR 19.33→19.54dB, regret -0.738→-0.943dB, 胜率 81.8%→86.2%")


# ---------- 图9：组件收益阶梯（修正后结构，同协议） ----------
def fig09():
    rows = list(csv.DictReader(open(HOLD_NEW / "summary.csv", encoding="utf-8-sig")))
    m = {r["method"]: r for r in rows}
    ladder = [
        ("random", "Random\n(池内随机)"),
        ("q_only", "Q-only\n(无检索, 随机5A)"),
        ("retrieval_max_r", "Retrieval-only\n(训练R最高)"),
        ("retrieve_q_union", "Retrieve+Q-Union\n(检索池∪随机100A)"),
        ("oracle_wide", "Oracle-Wide\n(100A 上界)"),
    ]
    vals = [float(m[k]["mean_recommended_sinr_db"]) for k, _ in ladder]
    colors = [C[k] for k, _ in ladder]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(ladder))
    ax.step(x, vals, where="mid", color="#999999", lw=1.2, zorder=1)
    bars = ax.bar(x, vals, 0.55, color=colors, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=9.5, weight="bold")
    for i in range(1, len(vals)):
        ax.annotate(f"+{vals[i]-vals[i-1]:.2f}", (x[i] - 0.5, (vals[i] + vals[i - 1]) / 2 + 0.35),
                    ha="center", fontsize=8.5, color="#555555")
    ax.set_xticks(x, [lbl for _, lbl in ladder], fontsize=9)
    ax.set_ylabel("推荐动作信号级 SINR（dB）")
    ax.set_title("各组件收益阶梯 — 修正后结构留出评估（500 场景 × K=10，20261023）")
    ax.set_ylim(0, 22.5)
    savefig(fig, "fig09_组件收益阶梯.png",
            "数据: outputs/q_value/holdout_baselines/summary.csv | 检索(+1.60dB)与大候选池是最大收益来源; Oracle-Wide(100A)非全局最优, 仅为弱上界")


ICDT_DIR = ROOT / "outputs" / "20261013_icdt" / "eval_icdt"


def fig10():
    import csv as _csv
    rows = list(_csv.DictReader(open(ICDT_DIR / "comparison_table.csv", encoding="utf-8-sig")))
    sel = [
        ("q_only", "Q-only（无检索）", C["q_only"]),
        ("retrieve_q_union", "Retrieve+Q-Union\n（Round2 基线）", C["retrieve_q_union"]),
        ("ens_pessimistic", "Q-Ensemble 悲观\n（20261007 默认）", C["ensemble_ctx_pessimistic"]),
        ("icdt_m0", "ICDT 零观测\n（0 次真值查询）", "#31708E"),
        ("icdt_rand_m5", "ICDT 随机观测 m=5\n（250 次真值查询）", "#7FA6C9"),
        ("icdt_act_m5", "ICDT 主动观测 m=5\n（250 次真值查询）", "#26456E"),
        ("oracle_wide", "Oracle-Wide\n（100A 上界）", C["oracle_wide"]),
    ]
    m = {r["method"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(sel))
    vals, cols, queries = [], [], []
    for k, _, c in sel:
        r = m.get(k)
        if r is None:
            alt = [kk for kk in m if kk.startswith(k)] or [kk for kk in m if k in kk]
            r = m[alt[0]] if alt else None
        vals.append(float(r["mean_regret_vs_k20_db"]))
        cols.append(c)
        q = r.get("total_truth_queries", "")
        queries.append(int(q) if str(q).strip().isdigit() else 0)
    bars = ax.bar(x, vals, 0.6, color=cols)
    for b, v, q in zip(bars, vals, queries):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.12 if v >= 0 else -0.32),
                f"{v:.2f}", ha="center", fontsize=9, weight="bold")
        if q:
            ax.text(b.get_x() + b.get_width() / 2, 5.15, f"{q} 次查询",
                    ha="center", fontsize=8, color="#555555")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(vals[1], color=C["retrieve_q_union"], ls=":", lw=1.2, alpha=0.9)
    ax.set_ylim(-6.4, 6.0)
    ax.set_xticks(x, [lbl for _, lbl, _ in sel], fontsize=8.8)
    ax.set_ylabel("mean regret vs Oracle-K20（dB）")
    ax.set_title("ICDT 上下文决策 Transformer：观测预算与收益 — 前 50 留出场景，旧结构协议（20261013 轮）")
    fig.tight_layout()
    savefig(fig, "fig10_ICDT观测模式对比.png",
            "数据: outputs/20261013_icdt/eval_icdt/comparison_table.csv | 独立轮协议(50场景×K10, 旧编码器模型), 与 500 场景轮数字不可直接比较 | 结论: 零观测即达基线水平, 花真值查询(主动/随机)几乎无额外收益")


ICDT_NEW = ROOT / "outputs" / "20261023_icdt_retrain" / "eval_icdt_holdout"


def fig11():
    """新协议（500 场景）下重训 ICDT 与主方法对比。"""
    if not (ICDT_NEW / "summary.json").is_file():
        print("fig11 跳过：ICDT 新协议评估结果尚未生成")
        return
    d = load_json(ICDT_NEW / "summary.json")
    icdt = {m["method"]: m for m in d["methods"]}
    base_rows = list(csv.DictReader(open(HOLD_NEW / "summary.csv", encoding="utf-8-sig")))
    base = {r["method"]: r for r in base_rows}

    order = [
        ("q_only", "Q-only", "q_only", C["q_only"]),
        ("retrieve_q_union", "Retrieve+Q-Union\n（主方法）", "retrieve_q_union", C["retrieve_q_union"]),
        ("icdt_m0", "ICDT 零观测\n(0 次查询)", "icdt_m0", "#31708E"),
        ("icdt_rand_m5", "ICDT 随机观测 m=5\n(5 次查询)", "icdt_rand_m5", "#7FA6C9"),
        ("icdt_act_m5", "ICDT 主动观测 m=5\n(5 次查询)", "icdt_act_m5", "#26456E"),
        ("oracle_wide", "Oracle-Wide\n(100A 上界)", "oracle_wide", C["oracle_wide"]),
    ]
    vals = []
    for key, *_rest in order:
        src = base if key in base else icdt
        vals.append(float(src[key]["mean_recommended_sinr_db"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    x = np.arange(len(order))
    bars = axes[0].bar(x, vals, 0.6, color=[c for *_, c in order])
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}",
                     ha="center", fontsize=9, weight="bold")
    axes[0].set_xticks(x, [lbl for _, lbl, _, _ in order], fontsize=8,
                       rotation=14, ha="right")
    axes[0].set_ylabel("推荐动作信号级 SINR（dB）")
    axes[0].set_title("推荐 SINR — 同候选池（检索池∪随机100A）同协议")
    axes[0].set_ylim(0, max(vals) + 1.4)

    wr = d["paired_win_rate"]["retrieve_q_union"]
    methods_wr = ["icdt_m0", "icdt_rand_m5", "icdt_act_m5"]
    xs = np.arange(len(methods_wr))
    wv = [wr[m]["win_rate"] * 100 for m in methods_wr]
    bars2 = axes[1].bar(xs, wv, 0.5, color=["#31708E", "#7FA6C9", "#26456E"])
    for b, v in zip(bars2, wv):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%",
                     ha="center", fontsize=9.5, weight="bold")
    axes[1].axhline(50, color="#C44E52", ls="--", lw=1.3)
    axes[1].text(2.42, 50.8, "50%（持平）", color="#C44E52", fontsize=8.5, ha="right")
    axes[1].set_xticks(xs, ["ICDT 零观测", "ICDT 随机 m=5", "ICDT 主动 m=5"], fontsize=9)
    axes[1].set_ylabel("对 Retrieve+Q-Union 配对胜率（%）")
    axes[1].set_title("配对胜率 vs 主方法（500 场景）")
    axes[1].set_ylim(0, 100)

    reg = {m["method"]: m["mean_regret_vs_k_db"] for m in d["methods"]}
    fig.suptitle("ICDT（新编码器重训）vs Q 系主方法 — 新留出协议 500 场景 × K=10（20261023）", y=1.02)
    fig.tight_layout()
    note = (f"数据: outputs/20261023_icdt_retrain/eval_icdt_holdout + q_value/holdout_baselines | "
            f"ICDT 悲观 μ−1.0σ, 5-seed; regret vs Reference-K: "
            + ", ".join(f"{m} {reg[m]:+.2f}" for m in ("icdt_m0", "icdt_rand_m5", "icdt_act_m5"))
            + " | 旧轮结论(主动观测无增益)在新结构下是否复现, 看零/随机/主动三柱是否单调改善")
    savefig(fig, "fig11_ICDT重训_新协议对比.png", note)


if __name__ == "__main__":
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    fig08()
    fig09()
    fig10()
    fig11()
    print("全部完成 →", OUT)
