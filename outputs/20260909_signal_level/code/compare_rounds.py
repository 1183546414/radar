# -*- coding: utf-8 -*-
"""Round 20260909 第 4 步：重训前后信号级留出评估对比报告。

对照双方（同 50 holdout 场景、同信号级真值 JSR=10dB）：
  * Round1（重训前）: archive/holdout_baselines_radarsimpy_jsr10/
      原模型（解析标签训练） + 信号级真值评估
  * Round2（重训后）: eval_retrained/
      信号级重标注数据重训模型 + 信号级真值评估

输出:
  * comparison_table.csv   按方法逐行对比
  * comparison_report.md   结论报告
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
R1_DIR = ROUND / "archive" / "holdout_baselines_radarsimpy_jsr10"
R2_DIR = ROUND / "eval_retrained"
OUT_CSV = ROUND / "comparison_table.csv"
OUT_MD = ROUND / "comparison_report.md"

METRIC_KEYS = (
    "mean_recommended_sinr_db",
    "mean_regret_vs_k20_db",
    "median_regret_vs_k20_db",
    "mean_oracle_wide_db",
    "frac_zero_regret_vs_k20",
)
METRIC_LABELS = {
    "mean_recommended_sinr_db": "推荐动作信号级SINR (dB)",
    "mean_regret_vs_k20_db": "Regret vs K20 (dB)",
    "median_regret_vs_k20_db": "Regret中位数 (dB)",
    "mean_oracle_wide_db": "Oracle-Wide (dB)",
    "frac_zero_regret_vs_k20": "零Regret占比",
}


def _load(d: Path) -> dict:
    with open(d / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _per_scenario(d: Path) -> dict:
    with open(d / "per_scenario_detail.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "-" if f != f else f"{f:.2f}"


def main() -> None:
    r1, r2 = _load(R1_DIR), _load(R2_DIR)
    m1 = {row["method"]: row for row in r1["methods"]}
    m2 = {row["method"]: row for row in r2["methods"]}
    methods = [row["method"] for row in r2["methods"]]

    # ---- 汇总表 ----
    lines = ["method,metric,round1_pretrain,round2_retrained,delta"]
    for m in methods:
        for k in METRIC_KEYS:
            v1, v2 = m1.get(m, {}).get(k), m2.get(m, {}).get(k)
            d = (
                (v2 - v1)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 == v1 and v2 == v2
                else ""
            )
            lines.append(f"{m},{k},{_fmt(v1)},{_fmt(v2)},{_fmt(d)}")
    for m in methods:
        w1 = r1.get("paired_vs_q_only", {}).get("results", {}).get(m, {}).get("win_rate")
        w2 = r2.get("paired_vs_q_only", {}).get("results", {}).get(m, {}).get("win_rate")
        if w1 is not None and w2 is not None:
            lines.append(f"{m},win_rate_vs_q_only,{_fmt(w1)},{_fmt(w2)},{_fmt(w2 - w1)}")
    OUT_CSV.write_text("\n".join(lines), encoding="utf-8")

    # ---- 逐场景配对：Round2 vs Round1 每方法推荐 SINR ----
    ps1, ps2 = _per_scenario(R1_DIR), _per_scenario(R2_DIR)
    paired_lines = ["method,n_better,n_tie,n_worse,mean_delta_db"]
    paired_rows = {}
    for m in methods:
        a = ps1.get(m, {}).get("recommended_sinr_db", [])
        b = ps2.get(m, {}).get("recommended_sinr_db", [])
        if not a or not b or len(a) != len(b):
            continue
        nb = nt = nw = 0
        deltas = []
        for x, y in zip(a, b):
            d = y - x
            deltas.append(d)
            if d > 1e-9:
                nb += 1
            elif d < -1e-9:
                nw += 1
            else:
                nt += 1
        mean_d = sum(deltas) / len(deltas)
        paired_rows[m] = (nb, nt, nw, mean_d)
        paired_lines.append(f"{m},{nb},{nt},{nw},{mean_d:.3f}")
    (ROUND / "comparison_paired_scenarios.csv").write_text(
        "\n".join(paired_lines), encoding="utf-8"
    )

    # ---- Markdown 报告 ----
    md = ["# 重训前后对比（信号级真值，JSR=10dB，50 holdout 场景）", ""]
    md.append("## 1. 设置")
    md.append("")
    md.append("| 项目 | Round1（重训前） | Round2（重训后） |")
    md.append("| --- | --- | --- |")
    md.append("| Q 模型 | 原模型（解析标签训练） | 本轮重训（信号级重标注 + 信号级 ranking oracle） |")
    md.append("| 训练标签 | 解析公式 SINR | 信号级 SINR（74,860 仿真 + 25,140 相位编码解析回退） |")
    md.append("| ranking/早停 oracle | 解析公式 | RadarSimPy 信号级（与标签同源） |")
    md.append("| 评估真值 | 信号级（RadarSimPy） | 信号级（RadarSimPy，同后端同 JSR） |")
    md.append("| holdout 场景 | 前 50 个（seed=2026） | 同左（严格同场景） |")
    md.append("")
    md.append("## 2. 各方法对比（真值 = 信号级 SINR）")
    md.append("")
    md.append("| 方法 | 指标 | Round1 | Round2 | Δ |")
    md.append("| --- | --- | --- | --- | --- |")
    for m in methods:
        for k in METRIC_KEYS:
            v1, v2 = m1.get(m, {}).get(k), m2.get(m, {}).get(k)
            d = (
                v2 - v1
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 == v1 and v2 == v2
                else None
            )
            label = f"{m} — {METRIC_LABELS[k]}"
            md.append(f"| {label} | {_fmt(v1)} | {_fmt(v2)} | {_fmt(d)} |")
    md.append("")
    md.append("## 3. 配对胜率 vs Q-only（逐场景）")
    md.append("")
    md.append("| 方法 | Round1 胜率 | Round2 胜率 | Δ |")
    md.append("| --- | --- | --- | --- |")
    for m in methods:
        w1 = r1.get("paired_vs_q_only", {}).get("results", {}).get(m, {}).get("win_rate")
        w2 = r2.get("paired_vs_q_only", {}).get("results", {}).get(m, {}).get("win_rate")
        if w1 is None or w2 is None:
            continue
        md.append(f"| {m} | {100 * w1:.0f}% | {100 * w2:.0f}% | {100 * (w2 - w1):+.0f}pct |")
    md.append("")
    md.append("## 4. 逐场景配对：Round2 vs Round1（同方法同场景推荐 SINR）")
    md.append("")
    md.append("| 方法 | Round2 更优 | 持平 | Round1 更优 | 平均 ΔSINR (dB) |")
    md.append("| --- | --- | --- | --- | --- |")
    for m, (nb, nt, nw, md_) in paired_rows.items():
        md.append(f"| {m} | {nb} | {nt} | {nw} | {md_:+.2f} |")
    md.append("")
    md.append("## 5. 参考基线（真值上界）")
    md.append("")
    ok1 = m1.get("oracle_k20", {}).get("mean_oracle_k20_db")
    ok2 = m2.get("oracle_k20", {}).get("mean_oracle_k20_db")
    ow1 = m1.get("oracle_wide", {}).get("mean_recommended_sinr_db")
    ow2 = m2.get("oracle_wide", {}).get("mean_recommended_sinr_db")
    md.append(f"- Oracle-K20 均值: R1 {_fmt(ok1)} dB → R2 {_fmt(ok2)} dB（同 K=20 随机动作池）")
    md.append(f"- Oracle-Wide 均值: R1 {_fmt(ow1)} dB → R2 {_fmt(ow2)} dB（100 随机动作上界）")
    md.append("")
    md.append("## 6. 结论要点")
    md.append("")
    for m in ("retrieve_q", "retrieve_q_union", "retrieve_q_ctx_union"):
        if m not in m1 or m not in m2:
            continue
        d = m2[m]["mean_recommended_sinr_db"] - m1[m]["mean_recommended_sinr_db"]
        dr = m2[m]["mean_regret_vs_k20_db"] - m1[m]["mean_regret_vs_k20_db"]
        md.append(
            f"- **{m}**: 推荐SINR {m1[m]['mean_recommended_sinr_db']:.2f} → "
            f"{m2[m]['mean_recommended_sinr_db']:.2f} dB（{d:+.2f}），"
            f"regret {m1[m]['mean_regret_vs_k20_db']:.2f} → {m2[m]['mean_regret_vs_k20_db']:.2f} dB（{dr:+.2f}）"
        )
    q1 = m1.get("q_only", {}).get("mean_recommended_sinr_db")
    q2 = m2.get("q_only", {}).get("mean_recommended_sinr_db")
    if q1 is not None and q2 is not None:
        md.append(f"- **q_only**（对照，无检索）: {q1:.2f} → {q2:.2f} dB")
    md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"对比表: {OUT_CSV}")
    print(f"逐场景配对: {ROUND / 'comparison_paired_scenarios.csv'}")
    print(f"报告: {OUT_MD}")


if __name__ == "__main__":
    main()
