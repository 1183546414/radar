# -*- coding: utf-8 -*-
"""最终训练效果可视化：VAE / Q 训练曲线 + 留出集 baseline 对比图。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radar.paths import OUT_FIGURES, OUT_Q_VALUE, OUT_STATE_VAE

from radar.infer.holdout_baselines import OUR_METHOD, REPORT_METHODS

OUT_DIR = OUT_FIGURES / "eval"
BASELINE_JSON = OUT_Q_VALUE / "holdout_baselines" / "summary.json"
VAE_METRICS = OUT_STATE_VAE / "metrics.json"
Q_LATENT_METRICS = OUT_Q_VALUE / "latent_q" / "metrics.json"
Q_RETRIEVAL_METRICS = OUT_Q_VALUE / "latent_q_retrieval" / "metrics.json"

# 主流程方法展示顺序（不含 oracle 上界）
BASELINE_ORDER = [m for m in REPORT_METHODS if m != "oracle_wide"]

SHORT_LABELS = {
    "random": "Random",
    "retrieval_max_r": "Retrieval",
    "q_only": "Q-only",
    OUR_METHOD: "Retrieve+Q",
    "oracle_wide": "Oracle-Wide",
}

OURS_METHODS = {OUR_METHOD}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"[WARN] 跳过，文件不存在: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _db_to_power(sinr_db: float) -> float:
    return float(10 ** (sinr_db / 10.0))


def _method_map(summary: dict) -> dict[str, dict]:
    return {m["method"]: m for m in summary["methods"]}


def _oracle_recovery_pct(rec_db: float, oracle_db: float, random_db: float) -> float:
    p_rec = _db_to_power(rec_db)
    p_oracle = _db_to_power(oracle_db)
    p_random = _db_to_power(random_db)
    recoverable = p_oracle - p_random
    if recoverable <= 1e-12:
        return float("nan")
    return (p_rec - p_random) / recoverable * 100.0


def _save_vae_curve(data: dict, out_path: Path) -> None:
    history = data.get("history", [])
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    val_rmse = [h.get("val_rec_rmse", math.sqrt(h.get("val_rec_mse", 0))) for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, val_rmse, marker="o", ms=3, lw=1.5, label="Val RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction RMSE")
    ax.set_title("State AE: validation reconstruction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_q_curves(
    q_latent: dict | None,
    q_retrieval: dict | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, data, title in (
        (axes[0], q_latent, "Q(z,A) — last epochs"),
        (axes[1], q_retrieval, "Q(z,z_ctx,A) — last epochs"),
    ):
        if data is None or not data.get("history"):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        hist = data["history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["val_mse"] for h in hist], marker="o", ms=4, label="Val MSE")
        ax2 = ax.twinx()
        ax2.plot(epochs, [h["val_rank_nll"] for h in hist], color="C1", marker="s", ms=4, label="Val rank NLL")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val MSE", color="C0")
        ax2.set_ylabel("Val rank NLL", color="C1")
        pearson = data.get("val_pearson_std")
        subtitle = f"Pearson={pearson:.3f}" if pearson is not None else ""
        ax.set_title(f"{title}\n{subtitle}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_q_selection_bars(q_latent: dict | None, out_path: Path) -> None:
    if q_latent is None:
        return
    k20 = q_latent.get("per_scenario_k20_selection", {})
    pool = q_latent.get("retrieval_pool_selection", {})
    if not k20 and not pool:
        return

    labels = ["K=20\n(Q-only pool)", "Retrieval\npool (~100)"]
    hit = [
        k20.get("oracle_hit_at_1", 0) * 100,
        pool.get("oracle_hit_at_1", 0) * 100,
    ]
    regret = [
        k20.get("mean_regret_sinr_db", 0),
        pool.get("mean_regret_sinr_db", 0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, hit, color=["steelblue", "darkorange"], edgecolor="white")
    axes[0].set_ylabel("Hit@1 vs oracle (%)")
    axes[0].set_title("Q argmax: oracle hit rate")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, regret, color=["steelblue", "darkorange"], edgecolor="white")
    axes[1].set_ylabel("Mean regret (dB)")
    axes[1].set_title("Q argmax: mean regret vs oracle")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Q network offline selection (500 holdout scenarios)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_baseline_bars(summary: dict, outdir: Path) -> None:
    methods = _method_map(summary)
    random_m = methods["random"]
    paired = summary.get("paired_vs_q_only", {}).get("results", {})

    labels, regrets, recoveries, win_rates, colors = [], [], [], [], []
    for mid in BASELINE_ORDER:
        m = methods.get(mid)
        if m is None:
            continue
        labels.append(SHORT_LABELS.get(mid, mid))
        regrets.append(m["mean_regret_vs_k20_db"])
        recoveries.append(
            _oracle_recovery_pct(
                m["mean_recommended_sinr_db"],
                m["mean_oracle_k20_db"],
                random_m["mean_recommended_sinr_db"],
            )
        )
        wr = paired.get(mid, {}).get("win_rate")
        win_rates.append(wr * 100 if wr is not None else float("nan"))
        colors.append("#2ca02c" if mid in OURS_METHODS else "#1f77b4")

    # --- regret ---
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    ax.barh(y, regrets, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean regret vs Oracle-K10 (dB) ↓")
    ax.set_title("Holdout baselines: regret@K10 (500 scenarios)")
    ax.axvline(methods["q_only"]["mean_regret_vs_k20_db"], color="gray", ls="--", lw=1, label="Q-only")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "baseline_regret_db.png", dpi=160)
    plt.close(fig)

    # --- oracle recovery ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y, recoveries, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Oracle gain recovery (%) ↑")
    ax.set_title("Holdout baselines: linear power recovery")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "baseline_oracle_recovery.png", dpi=160)
    plt.close(fig)

    # --- win rate vs Q-only ---
    fig, ax = plt.subplots(figsize=(9, 5))
    wr_labels = [lb for lb, wr in zip(labels, win_rates) if not math.isnan(wr)]
    wr_vals = [wr for wr in win_rates if not math.isnan(wr)]
    wr_colors = [c for c, wr in zip(colors, win_rates) if not math.isnan(wr)]
    ax.barh(np.arange(len(wr_labels)), wr_vals, color=wr_colors, edgecolor="white")
    ax.set_yticks(np.arange(len(wr_labels)))
    ax.set_yticklabels(wr_labels)
    ax.invert_yaxis()
    ax.axvline(50, color="gray", ls="--", lw=1)
    ax.set_xlabel("Win rate vs Q-only (%)")
    ax.set_title("Paired comparison: SINR beats Q-only")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "baseline_win_rate_vs_qonly.png", dpi=160)
    plt.close(fig)


def _save_dashboard(
    vae_data: dict | None,
    q_latent: dict | None,
    summary: dict | None,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    if vae_data and vae_data.get("history"):
        hist = vae_data["history"]
        ax0.plot([h["epoch"] for h in hist], [h.get("val_rec_rmse", 0) for h in hist], lw=1.5)
        ax0.set_title("State AE val RMSE")
        ax0.set_xlabel("Epoch")
        ax0.grid(True, alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax0.transAxes)

    ax1 = fig.add_subplot(gs[0, 1])
    if q_latent and q_latent.get("history"):
        hist = q_latent["history"]
        ax1.plot([h["epoch"] for h in hist], [h["val_mse"] for h in hist], lw=1.5, label="MSE")
        ax1.set_title(f"Q(z,A) val MSE (Pearson={q_latent.get('val_pearson_std', 0):.2f})")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax1.transAxes)

    ax2 = fig.add_subplot(gs[1, 0])
    if summary:
        methods = _method_map(summary)
        random_m = methods["random"]
        mids = [m for m in BASELINE_ORDER if m in methods]
        regrets = [methods[m]["mean_regret_vs_k20_db"] for m in mids]
        ax2.barh([SHORT_LABELS[m] for m in mids], regrets, color="#1f77b4")
        ax2.invert_yaxis()
        ax2.axvline(methods["q_only"]["mean_regret_vs_k20_db"], color="gray", ls="--", lw=1)
        ax2.set_xlabel("Regret (dB)")
        ax2.set_title("Holdout regret@K10")
        ax2.grid(True, axis="x", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    if summary:
        paired = summary.get("paired_vs_q_only", {}).get("results", {})
        mids = [m for m in BASELINE_ORDER if m in paired and m != "q_only"]
        wr = [paired[m]["win_rate"] * 100 for m in mids]
        ax3.barh([SHORT_LABELS[m] for m in mids], wr, color="#2ca02c")
        ax3.invert_yaxis()
        ax3.axvline(50, color="gray", ls="--", lw=1)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("Win % vs Q-only")
        ax3.set_title("Paired win rate")
        ax3.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Radar jamming: training & holdout summary", fontsize=13)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_summary(
    outdir: Path | str = OUT_DIR,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vae_raw = _load_json(VAE_METRICS)
    vae_data = vae_raw if vae_raw else None
    q_latent = _load_json(Q_LATENT_METRICS)
    q_retrieval = _load_json(Q_RETRIEVAL_METRICS)
    summary = _load_json(BASELINE_JSON)

    if vae_data:
        _save_vae_curve(vae_data, outdir / "vae_val_rmse.png")
        print(f"[OK] {outdir / 'vae_val_rmse.png'}")

    _save_q_curves(q_latent, q_retrieval, outdir / "q_training_curves.png")
    print(f"[OK] {outdir / 'q_training_curves.png'}")

    _save_q_selection_bars(q_latent, outdir / "q_selection_metrics.png")
    print(f"[OK] {outdir / 'q_selection_metrics.png'}")

    if summary:
        methods = _method_map(summary)
        if "random" in methods and "q_only" in methods:
            _save_baseline_bars(summary, outdir)
            print(f"[OK] {outdir / 'baseline_regret_db.png'}")
            print(f"[OK] {outdir / 'baseline_oracle_recovery.png'}")
            print(f"[OK] {outdir / 'baseline_win_rate_vs_qonly.png'}")
        else:
            print("[WARN] summary 方法不全，跳过 baseline 对比图")

    _save_dashboard(vae_data, q_latent, summary, outdir / "training_dashboard.png")
    print(f"[OK] {outdir / 'training_dashboard.png'}")

    return outdir


def plot_training_summary_main() -> None:
    plot_training_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training & holdout summary figures.")
    parser.add_argument("--outdir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    plot_training_summary(outdir=args.outdir)


if __name__ == "__main__":
    main()
