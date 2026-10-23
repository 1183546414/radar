# -*- coding: utf-8 -*-
"""留出集主报图：ΔSINR vs Q-only、配对胜率、regret 分位数。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radar.paths import OUT_FIGURES, OUT_Q_VALUE

from radar.infer.holdout_baselines import OUR_METHOD, REPORT_METHODS

OUT_DIR = OUT_FIGURES / "eval"
DETAIL_JSON = OUT_Q_VALUE / "holdout_baselines" / "per_scenario_detail.json"
SUMMARY_JSON = OUT_Q_VALUE / "holdout_baselines" / "summary.json"

PLOT_ORDER = [
    "random",
    "retrieval_max_r",
    OUR_METHOD,
]

SHORT_LABELS = {
    "random": "Random",
    "retrieval_max_r": "Retrieval",
    "q_only": "Q-only",
    OUR_METHOD: "Retrieve+Q",
    "oracle_wide": "Oracle-Wide",
}

OURS = {OUR_METHOD}


def _collect_per_scenario(methods: list[str] | None = None) -> dict[str, dict]:
    from radar.infer.holdout_baselines import REPORT_METHODS, evaluate_method, load_holdout_context

    methods = list(methods or REPORT_METHODS)
    ctx = load_holdout_context()
    detail: dict[str, dict] = {}
    for m in methods:
        r = evaluate_method(ctx, m, verbose=False)
        detail[m] = {
            "recommended_sinr_db": r["per_scenario_recommended_sinr_db"],
            "regret_vs_k20_db": r["per_scenario_regret_vs_k20_db"],
        }
    return detail


def _load_or_collect(detail_path: Path, refresh: bool = False) -> dict[str, dict]:
    if not refresh and detail_path.exists():
        with open(detail_path, encoding="utf-8") as f:
            return json.load(f)
    detail = _collect_per_scenario()
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    return detail


def _load_paired(summary_path: Path) -> dict:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    return summary.get("paired_vs_q_only", {}).get("results", {})


def _plot_delta_sinr(detail: dict, out_path: Path) -> None:
    q = np.array(detail["q_only"]["recommended_sinr_db"], dtype=np.float64)
    methods = [m for m in PLOT_ORDER if m in detail and m != "q_only"]
    labels, means, p25s, p75s, colors = [], [], [], [], []

    for m in methods:
        rec = np.array(detail[m]["recommended_sinr_db"], dtype=np.float64)
        delta = rec - q
        labels.append(SHORT_LABELS.get(m, m))
        means.append(float(delta.mean()))
        p25s.append(float(np.percentile(delta, 25)))
        p75s.append(float(np.percentile(delta, 75)))
        colors.append("#2ca02c" if m in OURS else "#1f77b4")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(labels))))
    ax.barh(y, means, color=colors, edgecolor="white", height=0.55)
    for yi, m, lo, hi in zip(y, means, p25s, p75s):
        ax.plot([lo, hi], [yi, yi], color="black", lw=1.2)
        ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color="black", lw=1.2)
        ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color="black", lw=1.2)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("ΔSINR vs Q-only (dB)  [bar=mean, line=p25–p75]")
    ax.set_title("Holdout: per-scenario SINR gain over Q-only (500 scenarios)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_win_rate(paired: dict, out_path: Path) -> None:
    methods = [m for m in PLOT_ORDER if m in paired and m != "q_only"]
    labels = [SHORT_LABELS.get(m, m) for m in methods]
    rates = [paired[m]["win_rate"] * 100 for m in methods]
    colors = ["#2ca02c" if m in OURS else "#1f77b4" for m in methods]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(methods))))
    y = np.arange(len(methods))
    ax.barh(y, rates, color=colors, edgecolor="white", height=0.55)
    ax.axvline(50, color="gray", ls="--", lw=1, label="50%")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Win rate vs Q-only (%)")
    ax.set_title("Paired comparison: fraction of scenarios with higher SINR")
    ax.grid(True, axis="x", alpha=0.3)
    for yi, r in zip(y, rates):
        ax.text(r + 1, yi, f"{r:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_regret_quantiles(detail: dict, out_path: Path) -> None:
    methods = [m for m in PLOT_ORDER if m in detail]
    labels = [SHORT_LABELS.get(m, m) for m in methods]
    data = [np.array(detail[m]["regret_vs_k20_db"], dtype=np.float64) for m in methods]
    colors = ["#2ca02c" if m in OURS else "#1f77b4" for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        vert=True,
        patch_artist=True,
        showfliers=False,
        whis=[10, 90],
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Regret vs Oracle-K10 (dB)  [↓ better]")
    ax.set_title("Holdout regret distribution (box=p25–p75, whiskers=p10–p90)")
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    # 分位数折线补充图
    quant_path = out_path.with_name("presentation_regret_quantile_lines.png")
    qs = [10, 25, 50, 75, 90]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m, lab, c in zip(methods, labels, colors):
        arr = np.array(detail[m]["regret_vs_k20_db"], dtype=np.float64)
        vals = [float(np.percentile(arr, q)) for q in qs]
        ax.plot(qs, vals, marker="o", lw=1.8, label=lab, color=c)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Regret vs Oracle-K10 (dB)")
    ax.set_title("Regret quantile curves")
    ax.set_xticks(qs)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(quant_path, dpi=160)
    plt.close(fig)


def plot_holdout_presentation(
    outdir: Path | str = OUT_DIR,
    detail_path: Path | str = DETAIL_JSON,
    summary_path: Path | str = SUMMARY_JSON,
    refresh: bool = False,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail = _load_or_collect(Path(detail_path), refresh=refresh)
    paired = _load_paired(Path(summary_path))

    p1 = outdir / "presentation_delta_sinr_vs_qonly.png"
    p2 = outdir / "presentation_win_rate_vs_qonly.png"
    p3 = outdir / "presentation_regret_quantiles.png"

    _plot_delta_sinr(detail, p1)
    _plot_win_rate(paired, p2)
    _plot_regret_quantiles(detail, p3)

    print(f"[OK] {p1.resolve()}")
    print(f"[OK] {p2.resolve()}")
    print(f"[OK] {p3.resolve()}")
    print(f"[OK] {(outdir / 'presentation_regret_quantile_lines.png').resolve()}")
    return outdir


def plot_holdout_presentation_main(refresh: bool = False) -> None:
    plot_holdout_presentation(refresh=refresh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout presentation figures.")
    parser.add_argument("--outdir", type=str, default=str(OUT_DIR))
    parser.add_argument("--refresh", action="store_true", help="重新跑留出评估收集逐场景数据")
    args = parser.parse_args()
    plot_holdout_presentation(outdir=args.outdir, refresh=args.refresh)


if __name__ == "__main__":
    main()
