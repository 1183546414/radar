# -*- coding: utf-8 -*-
"""潜向量 μ 的 t-SNE 可视化（按干扰类型上色 + 密度图）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from radar.paths import DATA_V2_PROCESSED, OUT_FIGURES, OUT_STATE_VAE

DEFAULT_EMB = OUT_STATE_VAE / "state_embeddings_mu.npy"
DEFAULT_SCENARIO_IDS = DATA_V2_PROCESSED / "scenario_id.npy"
DEFAULT_JAMMER_LABELS = DATA_V2_PROCESSED / "jammer_type_labels.json"
DEFAULT_OUTDIR = OUT_FIGURES / "latent"
SEED = 42
TSNE_ITER = 1000


def auto_perplexity(n_samples: int) -> int:
    if n_samples <= 10:
        return max(2, n_samples // 3) or 2
    return max(5, min(50, (n_samples - 1) // 3))


def _load_scenario_ids(n: int) -> np.ndarray:
    path = DEFAULT_SCENARIO_IDS
    if not path.exists():
        raise FileNotFoundError(f"未找到 scenario_id.npy: {path}")
    scenario_ids = np.load(path)
    if len(scenario_ids) != n:
        raise ValueError(f"scenario_id 行数 ({len(scenario_ids)}) != 潜向量行数 ({n})")
    return scenario_ids.astype(np.int64)


def _load_jammer_labels(n: int) -> np.ndarray:
    path = DEFAULT_JAMMER_LABELS
    if not path.exists():
        raise FileNotFoundError(f"未找到 jammer_type_labels.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if len(raw) != n:
        raise ValueError(f"jammer 标签数 ({len(raw)}) != 潜向量行数 ({n})")
    return np.array([str(x) for x in raw])


def _dedupe_by_scenario(
    Z: np.ndarray,
    scenario_ids: np.ndarray,
    row_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """每个 scenario 只保留一行 μ（同场景 20 个动作共享同一 S → 同一 μ）。"""
    _, first_idx = np.unique(scenario_ids, return_index=True)
    first_idx = np.sort(first_idx)
    labels_out = row_labels[first_idx] if row_labels is not None else None
    return Z[first_idx], scenario_ids[first_idx], labels_out


def _build_tsne(**kwargs) -> TSNE:
    if "n_iter" in TSNE.__init__.__code__.co_varnames:
        return TSNE(**kwargs)
    if "n_iter" in kwargs:
        kwargs["max_iter"] = kwargs.pop("n_iter")
    return TSNE(**kwargs)


def _compute_intra_scenario_stats(Z: np.ndarray, scenario_ids: np.ndarray) -> dict[str, float]:
    max_spread = []
    for sid in np.unique(scenario_ids):
        rows = Z[scenario_ids == sid]
        if len(rows) <= 1:
            max_spread.append(0.0)
            continue
        centroid = rows.mean(axis=0, keepdims=True)
        spread = np.linalg.norm(rows - centroid, axis=1)
        max_spread.append(float(spread.max()))
    max_spread = np.array(max_spread)
    return {
        "n_scenarios": int(len(max_spread)),
        "intra_max_dist_mean": float(max_spread.mean()),
        "intra_max_dist_max": float(max_spread.max()),
        "n_scenarios_with_spread_gt_1e-6": int((max_spread > 1e-6).sum()),
    }


def _save_intra_scenario_hist(
    Z: np.ndarray,
    scenario_ids: np.ndarray,
    out_path: Path,
) -> None:
    max_spread = []
    for sid in np.unique(scenario_ids):
        rows = Z[scenario_ids == sid]
        if len(rows) <= 1:
            max_spread.append(0.0)
            continue
        centroid = rows.mean(axis=0, keepdims=True)
        max_spread.append(float(np.linalg.norm(rows - centroid, axis=1).max()))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(max_spread, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Max L2 distance to scenario centroid (128-d μ)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Intra-scenario μ consistency (expect ≈ 0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_tsne_by_category(
    Z_tsne: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_points: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in sorted(np.unique(labels), key=str):
        mask = labels == lab
        short = str(lab).replace("jammer_", "")
        ax.scatter(
            Z_tsne[mask, 0],
            Z_tsne[mask, 1],
            s=12,
            alpha=0.7,
            label=short,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=7, markerscale=1.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_tsne_density(Z_tsne: np.ndarray, out_path: Path, n_points: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(Z_tsne[:, 0], Z_tsne[:, 1], gridsize=40, cmap="Blues", mincnt=1, linewidths=0)
    ax.set_title(f"t-SNE density (n={n_points} scenarios)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latent_viz(
    emb_path: Path | str = DEFAULT_EMB,
    outdir: Path | str = DEFAULT_OUTDIR,
    seed: int = SEED,
    tsne_perplexity: float | None = None,
    tsne_iter: int = TSNE_ITER,
) -> Path:
    """对去重后的 scenario μ 做 t-SNE，按干扰类型离散着色。"""
    emb_path = Path(emb_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Z = np.load(emb_path).astype(np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {Z.shape}")

    scenario_ids = _load_scenario_ids(len(Z))
    jammer_labels = _load_jammer_labels(len(Z))
    mask_finite = np.isfinite(Z).all(axis=1)
    if not mask_finite.all():
        print(f"[WARN] Dropping {int((~mask_finite).sum())} rows containing NaN/Inf")
        Z = Z[mask_finite]
        scenario_ids = scenario_ids[mask_finite]
        jammer_labels = jammer_labels[mask_finite]

    stats = _compute_intra_scenario_stats(Z, scenario_ids)
    print(
        f"[INFO] Loaded embeddings: {Z.shape}, scenarios={stats['n_scenarios']}, "
        f"intra_max_dist_mean={stats['intra_max_dist_mean']:.2e}"
    )

    stats_path = outdir / "scenario_mu_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}={v}\n")
    print(f"[OK] 场景 μ 一致性统计: {stats_path.resolve()}")

    hist_path = outdir / "scenario_mu_consistency.png"
    _save_intra_scenario_hist(Z, scenario_ids, hist_path)
    print(f"[OK] 场景内 μ 一致性直方图: {hist_path.resolve()}")

    Z_unique, sid_unique, jammer_unique = _dedupe_by_scenario(Z, scenario_ids, jammer_labels)
    n_tsne = len(Z_unique)
    n_types = len(np.unique(jammer_unique))
    perplexity = tsne_perplexity if tsne_perplexity is not None else auto_perplexity(n_tsne)
    if perplexity >= n_tsne:
        perplexity = max(2, n_tsne // 3)
    print(f"[INFO] t-SNE: n={n_tsne} scenarios, {n_types} jammer types, perplexity={perplexity}")

    Z_tsne = _build_tsne(
        n_components=2,
        init="random",
        perplexity=perplexity,
        random_state=seed,
        learning_rate="auto",
        early_exaggeration=12.0,
        verbose=1,
        n_iter=tsne_iter,
    ).fit_transform(Z_unique)
    np.save(outdir / "Z_tsne.npy", Z_tsne)
    np.save(outdir / "tsne_scenario_ids.npy", sid_unique)
    np.save(outdir / "tsne_jammer_labels.npy", jammer_unique)

    tsne_png = outdir / "tsne_scatter.png"
    _save_tsne_by_category(
        Z_tsne,
        jammer_unique,
        tsne_png,
        n_tsne,
        title=f"t-SNE of μ by jammer type (n={n_tsne} scenarios)",
    )
    print(f"[OK] t-SNE 散点图已保存: {tsne_png.resolve()}")

    density_png = outdir / "tsne_density.png"
    _save_tsne_density(Z_tsne, density_png, n_tsne)
    print(f"[OK] t-SNE 密度图已保存: {density_png.resolve()}")

    return outdir


def plot_latent_viz_main() -> None:
    plot_latent_viz()


def vis_lat_main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE visualization for latent μ.")
    parser.add_argument("--npy", type=str, default=str(DEFAULT_EMB))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tsne_perplexity", type=float, default=None)
    parser.add_argument("--tsne_iter", type=int, default=TSNE_ITER)
    args = parser.parse_args()
    plot_latent_viz(
        emb_path=args.npy,
        outdir=args.outdir,
        seed=args.seed,
        tsne_perplexity=args.tsne_perplexity,
        tsne_iter=args.tsne_iter,
    )


if __name__ == "__main__":
    vis_lat_main()
