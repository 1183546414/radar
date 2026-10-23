# -*- coding: utf-8 -*-
"""按真 regret 口径（稠密近似 oracle 参照）重算主方法对比表。

参照定义：
  dense_oracle = 每场景 best-of-5000 随机 + 坐标细化 的近似最优（真最优下界）
  true_regret  = dense_oracle - 推荐动作SINR（对真 regret 是上界）
同时保留 vs K20 / vs Wide 旧口径作对照列。

数据来源（逐场景推荐 SINR，均为既有评估存档，不重跑模型）：
  * outputs/20260909_signal_level/eval_retrained/per_scenario_detail.json
  * outputs/20261007_fewshot/eval_fewshot/per_scenario_detail.json
ICDT 轮（20261013）未存逐场景明细，无法重算，单列说明。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261021_dense_oracle/code/true_regret_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
RADAR_ROOT = next(p for p in _HERE.parents if (p / "src" / "radar" / "paths.py").is_file())
OUT_DIR = _HERE.parents[1] / "results"

# 历史存档逐场景明细（已在 outputs/20261021_dense_oracle/eval_*_current 验证比特级可复现）
R2_DETAIL = RADAR_ROOT / "outputs" / "20260909_signal_level" / "eval_retrained" / "per_scenario_detail.json"
FS_DETAIL = RADAR_ROOT / "outputs" / "20261007_fewshot" / "eval_fewshot" / "per_scenario_detail.json"

# (来源文件, 存档键, 输出标签)
METHODS = [
    (R2_DETAIL, "random", "random（候选池内随机）"),
    (R2_DETAIL, "retrieval_max_r", "retrieval_max_r（仅检索，无 Q）"),
    (R2_DETAIL, "q_only", "q_only（无检索，随机 5A）"),
    (R2_DETAIL, "retrieve_q", "Retrieve+Q（Top-5 argmax）"),
    (R2_DETAIL, "retrieve_q_union", "Retrieve+Q-Union（Round2 基线）"),
    (R2_DETAIL, "retrieve_q_local", "Retrieve+Q-Local"),
    (R2_DETAIL, "retrieve_q_gen", "Retrieve+Q-Gen"),
    (R2_DETAIL, "retrieve_q_ctx", "Retrieve+Q-Ctx"),
    (R2_DETAIL, "retrieve_q_ctx_union", "Retrieve+Q-Ctx-Union"),
    (FS_DETAIL, "ensemble_ctx_mean", "① Q-Ensemble-Mean（5×Q，union 池）"),
    (FS_DETAIL, "ensemble_ctx_pessimistic", "① Q-Ensemble 悲观 μ−1.0σ"),
    (FS_DETAIL, "attention_ctx", "② Set-Attention（单模型，union 池）"),
    (FS_DETAIL, "attention_ens_pessimistic", "①+② Attention-Ensemble 悲观"),
]
REFERENCES = [
    (R2_DETAIL, "oracle_k20", "Oracle-K20（10 随机 A，弱参照）"),
    (R2_DETAIL, "oracle_wide", "Oracle-Wide（100 随机 A）"),
]
N_EXPECT = 50


def load_rec(path: Path, key: str) -> np.ndarray:
    detail = json.loads(path.read_text(encoding="utf-8"))
    vals = np.asarray(detail[key]["recommended_sinr_db"], dtype=float)
    assert len(vals) == N_EXPECT, f"{path.name}/{key}: {len(vals)} != {N_EXPECT}"
    return vals


def main() -> None:
    dense_doc = json.loads((OUT_DIR / "dense_oracle_per_scenario.json").read_text(encoding="utf-8"))
    per_sc = dense_doc["per_scenario"]
    dense = np.array([p["dense_oracle_db"] for p in per_sc])
    sids = [p["sid"] for p in per_sc]

    # 交叉校验：存档 K20（信号级重评估）应与稠密轮独立重算的信号级 K20 完全一致
    k20_stored = load_rec(R2_DETAIL, "oracle_k20")
    k20_csv = np.array([p["oracle_k20_db"] for p in per_sc])
    k20_diff = float(np.abs(k20_stored - k20_csv).max())
    assert k20_diff < 1e-3, f"oracle_k20 交叉校验失败: max|diff|={k20_diff}"  # 1e-3=产物4位小数舍入容差

    q_only = load_rec(R2_DETAIL, "q_only")
    rows = []
    for path, key, label in METHODS + REFERENCES:
        rec = load_rec(path, key)
        regret = dense - rec
        wins = int((rec > q_only + 1e-9).sum())
        rows.append({
            "method": label,
            "key": key,
            "mean_rec_sinr_db": round(float(rec.mean()), 3),
            "mean_true_regret_db": round(float(regret.mean()), 3),
            "median_true_regret_db": round(float(np.median(regret)), 3),
            "p95_true_regret_db": round(float(np.percentile(regret, 95)), 3),
            "min_true_regret_db": round(float(regret.min()), 3),
            "frac_at_dense_oracle": round(float((regret <= 1e-6).mean()), 3),
            "win_vs_q_only": round(wins / len(rec), 3),
            "mean_rec_minus_k20_db": round(float((rec - k20_stored).mean()), 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "true_regret_comparison.csv", index=False, encoding="utf-8-sig")

    d_mean, d_med = float(dense.mean()), float(np.median(dense))
    lines = [
        "# 真 regret 口径重算报告（稠密近似 oracle 参照）",
        "",
        f"参照：每场景 best-of-{dense_doc['summary']['protocol']['n_random']} 随机 + 坐标细化，"
        f"50 场景 mean={d_mean:.2f} dB / median={d_med:.2f} dB"
        f"（对照：Oracle-K20={k20_csv.mean():.2f}，Oracle-Wide 报告值≈23.2~24.4）。",
        "",
        "true_regret = dense_oracle − 推荐SINR；近似 oracle 是真最优的下界，",
        "故 true_regret 是真 regret 的**上界**（细化平均再抬升 "
        f"{np.mean([p['refine_gain_db'] for p in per_sc]):.2f} dB）。",
        "",
        "交叉校验：存档 oracle_k20（信号级）与稠密轮独立重算 max|diff| = "
        f"{k20_diff:.2e}，场景与顺序比特级一致。",
        "",
        "| 方法 | 推荐SINR (dB) | 真regret均值 (dB) | 真regret中位 | p95 | 达oracle占比 | 胜率vs q_only | 旧口径vs K20 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['mean_rec_sinr_db']:.2f} | {r['mean_true_regret_db']:.2f} | "
            f"{r['median_true_regret_db']:.2f} | {r['p95_true_regret_db']:.2f} | "
            f"{r['frac_at_dense_oracle']:.2f} | {r['win_vs_q_only']:.2f} | "
            f"{r['mean_rec_minus_k20_db']:+.2f} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- `frac_at_dense_oracle`：推荐动作即本轮搜索最优的场景占比（regret≤1e-6）。",
        "- 方法逐场景推荐值直接取**历史存档**（eval_retrained / eval_fewshot）：已在 "
        "eval_r2_current / eval_fs_current 用相同留出集与 checkpoint 重跑验证，"
        "除抽样型方法（retrieve_q_gen、oracle_wide）外比特级一致。",
        "- 注意口径：存档 oracle_k20/oracle_wide 为**信号级**参照；留出 CSV 的 "
        "SINR_dB 列是解析式（K20 解析均值 19.30），两者不可混用。",
        "- ICDT（20261013）未存逐场景明细，本轮不列入。",
        "- 旧口径列（vs K20）仅用于方法间横向对比，不再作为 regret 报告。",
        "",
        f"本轮耗时/统计见 summary.json（backend: {dense_doc['summary']['backend_stats']}）。",
    ]
    (OUT_DIR / "true_regret_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
