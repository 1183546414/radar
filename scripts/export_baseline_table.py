#!/usr/bin/env python3
"""从 holdout baseline 结果生成详细对比表（CSV + Markdown，含线性功率 / 增益回收率）。"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar.infer.holdout_baselines import OUR_METHOD, REPORT_METHODS

SUMMARY_JSON = ROOT / "outputs/q_value/holdout_baselines/summary.json"
Q_METRICS = ROOT / "outputs/q_value/latent_q/metrics.json"
OUT_DIR = ROOT / "outputs/q_value/holdout_baselines"
OUT_CSV = OUT_DIR / "comparison_table_detailed.csv"
OUT_MD = OUT_DIR / "comparison_table.md"

METHOD_META = {
    "random": {
        "type": "Baseline",
        "uses_retrieval": "是",
        "uses_q": "否",
        "candidate_desc": "Top-5 相似训练场景动作并集",
        "candidate_size": "~100",
        "is_ours": False,
    },
    "retrieval_max_r": {
        "type": "Baseline",
        "uses_retrieval": "是",
        "uses_q": "否",
        "candidate_desc": "Top-5 相似训练场景动作并集",
        "candidate_size": "~100",
        "is_ours": False,
    },
    "q_only": {
        "type": "Baseline",
        "uses_retrieval": "否",
        "uses_q": "是",
        "candidate_desc": "meta-train 池随机抽样",
        "candidate_size": "5",
        "is_ours": False,
    },
    "retrieve_q": {
        "type": "Ablation",
        "uses_retrieval": "是",
        "uses_q": "是",
        "candidate_desc": "Top-5 相似训练场景动作并集",
        "candidate_size": "~100",
        "is_ours": False,
    },
    "retrieve_q_union": {
        "type": "Ours",
        "uses_retrieval": "是",
        "uses_q": "是",
        "candidate_desc": "Top-5 检索动作 ∪ meta-train 随机 100 A",
        "candidate_size": "~200",
        "is_ours": True,
    },
    "retrieve_q_local": {
        "type": "Ours-A",
        "uses_retrieval": "是",
        "uses_q": "是",
        "candidate_desc": "检索池 + Top-Q 种子局部扰动",
        "candidate_size": "~200",
        "is_ours": True,
    },
    "retrieve_q_gen": {
        "type": "Ours-A",
        "uses_retrieval": "是",
        "uses_q": "是",
        "candidate_desc": "检索池 + 新采样雷达 A",
        "candidate_size": "~200",
        "is_ours": True,
    },
    "retrieve_q_ctx": {
        "type": "Ours-B",
        "uses_retrieval": "是",
        "uses_q": "是（检索条件）",
        "candidate_desc": "Top-5 检索池 + z_ctx 邻居上下文",
        "candidate_size": "~100",
        "is_ours": True,
    },
    "oracle_k20": {
        "type": "Upper bound",
        "uses_retrieval": "—",
        "uses_q": "—",
        "candidate_desc": "留出场景预生成 K 个随机 A",
        "candidate_size": "20",
        "is_ours": False,
    },
    "oracle_wide": {
        "type": "Upper bound",
        "uses_retrieval": "—",
        "uses_q": "—",
        "candidate_desc": "对查询 S 现场随机采样 A",
        "candidate_size": "100",
        "is_ours": False,
    },
}


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _f(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def db_to_linear_power(sinr_db: float) -> float:
    """SINR(dB) → 线性功率比 P/P_noise（相对单位，非 dB）。"""
    return float(10 ** (sinr_db / 10.0))


def build_rows(summary: dict) -> list[dict]:
    report_ids = set(REPORT_METHODS)
    methods = [m for m in summary["methods"] if m["method"] in report_ids]
    random_m = next(m for m in methods if m["method"] == "random")
    random_regret_db = random_m["mean_regret_vs_k20_db"]
    random_rec_db = random_m["mean_recommended_sinr_db"]
    p_random = db_to_linear_power(random_rec_db)

    paired = summary.get("paired_vs_q_only", {}).get("results", {})

    rows = []
    for m in methods:
        name = m["method"]
        meta = METHOD_META[name]
        regret_k20_db = m["mean_regret_vs_k20_db"]
        regret_wide = m["mean_regret_vs_wide_db"]
        rec_db = m["mean_recommended_sinr_db"]
        oracle_k20_db = m["mean_oracle_k20_db"]
        oracle_wide_db = m["mean_oracle_wide_db"]

        p_rec = db_to_linear_power(rec_db)
        p_oracle = db_to_linear_power(oracle_k20_db)
        recoverable = p_oracle - p_random
        oracle_recovery_pct = (
            (p_rec - p_random) / recoverable * 100.0 if recoverable > 1e-12 else float("nan")
        )
        linear_regret = p_oracle - p_rec
        power_loss_pct = (1.0 - p_rec / p_oracle) * 100.0 if p_oracle > 1e-12 else float("nan")
        vs_random_power_index = p_rec / p_random * 100.0 if p_random > 1e-12 else float("nan")
        vs_random_linear_gain_pct = (p_rec / p_random - 1.0) * 100.0 if p_random > 1e-12 else float("nan")

        pair = paired.get(name, {})
        win_rate_vs_q = pair.get("win_rate")
        wins_vs_q = pair.get("wins")

        rows.append(
            {
                "method_id": name,
                "method_name": m["label"],
                "method_type": meta["type"],
                "is_ours": "★" if meta["is_ours"] else "",
                "uses_vae_retrieval": meta["uses_retrieval"],
                "uses_q_network": meta["uses_q"],
                "candidate_pool": meta["candidate_desc"],
                "candidate_size": meta["candidate_size"],
                "n_eval_scenarios": m["n_evaluated"],
                # dB 指标（保留）
                "mean_recommended_sinr_db": round(rec_db, 4),
                "mean_oracle_k20_sinr_db": round(oracle_k20_db, 4),
                "mean_regret_vs_oracle_k20_db": round(regret_k20_db, 4),
                "median_regret_vs_oracle_k20_db": round(m["median_regret_vs_k20_db"], 4),
                "hit_at_1_vs_oracle_k20_pct": _pct(m["frac_zero_regret_vs_k20"]),
                # 线性功率指标（主展示）
                "mean_linear_power": round(p_rec, 2),
                "linear_regret_vs_oracle": round(linear_regret, 2),
                "oracle_recovery_pct": round(oracle_recovery_pct, 2),
                "power_loss_vs_oracle_pct": round(power_loss_pct, 2),
                "vs_random_power_index": round(vs_random_power_index, 2),
                "vs_random_linear_gain_pct": round(vs_random_linear_gain_pct, 2),
                # 配对胜率 vs Q-only
                "win_rate_vs_q_only_pct": round(win_rate_vs_q * 100, 1)
                if win_rate_vs_q is not None
                else float("nan"),
                "wins_vs_q_only": wins_vs_q if wins_vs_q is not None else "",
                # 其它
                "regret_reduction_vs_random_pct": round(
                    (1.0 - regret_k20_db / random_regret_db) * 100.0, 1
                )
                if random_regret_db > 1e-9
                else 0.0,
                "mean_top1_cosine_sim": round(m.get("mean_top1_cosine_sim", float("nan")), 4),
                "mean_candidate_pool_size": round(
                    m.get("mean_candidate_pool_size", float("nan")), 1
                ),
            }
        )

    rows.sort(key=lambda r: r["oracle_recovery_pct"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank_by_oracle_recovery"] = i
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_md(summary: dict, rows: list[dict], q_metrics: dict | None) -> None:
    random_row = next(r for r in rows if r["method_id"] == "random")

    lines = [
        "# 留出集 Baseline 对比实验结果",
        "",
        "## 1. 实验设置",
        "",
        "| 项目 | 配置 |",
        "|------|------|",
        f"| 留出干扰场景数 | {summary['n_holdout_scenarios']} |",
        f"| 每场景预生成动作数 K | {summary['actions_per_holdout_scenario']} |",
        f"| 主对比指标 | **Oracle 增益回收率**、**线性功率**（见下文） |",
        f"| 辅助指标 | regret@K10 (dB) |",
        "",
        "> **为何不用 dB 做主指标？** dB 是对数刻度，0.5 dB 的差异在线性功率上约等于 "
        "10% 左右的功率损失；用线性功率与增益回收率衡量，方法间差距更直观。",
        "",
        "**换算**：线性功率 \\(P = 10^{\\mathrm{SINR/dB}/10}\\)（相对噪声功率比）；"
        "Random 功率指数 = **100**（基准）。",
        "",
        f"**本文方法**：Retrieve+Q（检索池 ∪ 随机 100 A，argmax Q）。",
        "",
        "## 2. 主表：线性功率视角（按 Oracle 增益回收率↑排序）",
        "",
        "| 排名 | 方法 | 增益回收率↑ | 功率指数↑ | 相对Random提升 | 线性regret↓ | 功率损失率↓ |",
        "|------|------|:-----------:|:---------:|:--------------:|:-----------:|:-----------:|",
    ]

    for r in rows:
        lines.append(
            f"| {r['rank_by_oracle_recovery']} | {r['is_ours']}{r['method_name']} | "
            f"**{r['oracle_recovery_pct']:.1f}%** | "
            f"**{r['vs_random_power_index']:.1f}** | "
            f"{r['vs_random_linear_gain_pct']:.1f}% | "
            f"{r['linear_regret_vs_oracle']:.0f} | "
            f"{r['power_loss_vs_oracle_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "**主指标说明**",
            "",
            f"- **增益回收率↑**：\\((P_{{方法}} - P_{{Random}}) / (P_{{Oracle}} - P_{{Random}})\\)，"
            f"Random={random_row['oracle_recovery_pct']:.0f}%；"
            f"Oracle-K10（留出 K 动作内最优，不列入表中）=100%",
            "- **功率指数↑**：\\(P_{方法} / P_{Random} \\times 100\\)，Random 固定为 100",
            "- **相对 Random 提升**：线性功率提升百分比 \\((P/P_{Random} - 1)\\)",
            "- **线性 regret↓**：\\(P_{Oracle} - P_{方法}\\)（线性功率单位，越大越差）",
            "- **功率损失率↓**：\\((1 - P/P_{Oracle}) \\times 100\\%\\)",
            "",
            "## 3. 辅表：dB 视角（传统电信单位）",
            "",
            "| 方法 | 推荐 SINR (dB) | Oracle-K10 (dB) | regret (dB) | 中位 regret (dB) |",
            "|------|----------------|-----------------|-------------|------------------|",
        ]
    )

    db_sorted = sorted(rows, key=lambda r: r["mean_regret_vs_oracle_k20_db"])
    for r in db_sorted:
        lines.append(
            f"| {r['is_ours']}{r['method_name']} | "
            f"{r['mean_recommended_sinr_db']:.2f} | "
            f"{r['mean_oracle_k20_sinr_db']:.2f} | "
            f"{r['mean_regret_vs_oracle_k20_db']:.3f} | "
            f"{r['median_regret_vs_oracle_k20_db']:.3f} |"
        )

    if summary.get("paired_vs_q_only"):
        report_ids = set(REPORT_METHODS)
        lines.extend(["", "## 4. 逐场景配对胜率（相对 Q-only）", ""])
        lines.append("| 方法 | 胜 | 平 | 负 | 胜率 |")
        lines.append("|------|-----|-----|-----|------|")
        pr = summary["paired_vs_q_only"]
        n = pr["n_scenarios"]
        for mid, st in sorted(
            pr["results"].items(), key=lambda x: -x[1]["win_rate"]
        ):
            if mid not in report_ids or mid == "q_only":
                continue
            label = METHOD_META.get(mid, {}).get("type", mid)
            if mid in METHOD_META:
                name = next(
                    (r["method_name"] for r in rows if r["method_id"] == mid), mid
                )
            else:
                name = mid
            lines.append(
                f"| {name} | {st['wins']} | {st['ties']} | {st['losses']} | "
                f"**{100*st['win_rate']:.1f}%** |"
            )
        lines.append(f"| *基准* | *Q-only* | — | — | 50%（定义） |")

    if q_metrics and "per_scenario_action_selection" in q_metrics:
        ps = q_metrics["per_scenario_action_selection"]
        lines.extend(
            [
                "",
                "## 5. 分布内验证（meta-val，K=20）",
                "",
                "| 指标 | 数值 |",
                "|------|------|",
                f"| Q 回归 val Pearson | {q_metrics.get('val_pearson_std', 0):.4f} |",
                f"| Hit@1 | {_pct(ps['oracle_hit_at_1'])} |",
                f"| mean regret (dB) | {ps['mean_regret_sinr_db']:.4f} |",
                "",
            ]
        )

    lines.append(f"*生成自 `{SUMMARY_JSON.name}`*")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SUMMARY_JSON.is_file():
        print(f"未找到 {SUMMARY_JSON}，请先运行: python scripts/run_holdout_baselines.py")
        sys.exit(1)

    with open(SUMMARY_JSON, encoding="utf-8") as f:
        summary = json.load(f)

    if "paired_vs_q_only" not in summary:
        print("summary 缺少配对胜率，正在计算（约 3–5 分钟）...")
        from radar.infer.holdout_baselines import compute_paired_vs_reference, load_holdout_context

        ctx = load_holdout_context()
        methods = [m["method"] for m in summary["methods"]]
        summary["paired_vs_q_only"] = compute_paired_vs_reference(
            ctx, reference="q_only", methods=methods
        )
        with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    q_metrics = None
    if Q_METRICS.is_file():
        with open(Q_METRICS, encoding="utf-8") as f:
            q_metrics = json.load(f)

    rows = build_rows(summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    write_md(summary, rows, q_metrics)
    print(f"已写入: {OUT_CSV}")
    print(f"已写入: {OUT_MD}")


if __name__ == "__main__":
    main()
