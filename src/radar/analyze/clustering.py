# -*- coding: utf-8 -*-
"""
无标签潜空间聚类评估：
- 扫 k=K_MIN..K_MAX 的 KMeans（或 MiniBatchKMeans）
- 计算 Silhouette（越高越好）与 Davies–Bouldin（越低越好）
- 保存指标曲线与 summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from radar.paths import OUT_FIGURES, OUT_STATE_VAE


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n


def run_eval(
    emb_path: Path,
    outdir: Path,
    k_min: int = 2,
    k_max: int = 20,
    normalize: bool = True,
    subsample_n: Optional[int] = 20000,
    use_minibatch: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    X = np.load(emb_path)
    assert X.ndim == 2, f"Embeddings must be 2D, got {X.shape}"

    if subsample_n and len(X) > subsample_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=subsample_n, replace=False)
        X = X[idx]

    if normalize:
        X = l2_normalize(X)

    rows = []
    for k in range(k_min, k_max + 1):
        if use_minibatch:
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=1024,
                max_iter=50,
                n_init=3,
            )
        else:
            km = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
                max_iter=300,
                algorithm="elkan",
            )

        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, metric="euclidean")
        dbi = davies_bouldin_score(X, labels)
        rows.append({"k": int(k), "silhouette": float(sil), "davies_bouldin": float(dbi)})

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_sil = df.loc[df["silhouette"].idxmax()]
    best_dbi = df.loc[df["davies_bouldin"].idxmin()]

    print("\n=== Clustering Summary ===")
    print(f"Embeddings: {emb_path}  |  n={len(X)}, dim={X.shape[1]}")
    print(f"Best Silhouette:      k={int(best_sil.k)}  score={best_sil.silhouette:.4f}  (higher=better)")
    print(f"Best Davies–Bouldin:  k={int(best_dbi.k)}  score={best_dbi.davies_bouldin:.4f}  (lower=better)")
    print()
    print(df.to_string(index=False, formatters={
        "silhouette": lambda v: f"{v:.4f}",
        "davies_bouldin": lambda v: f"{v:.4f}",
    }))

    df.to_csv(outdir / "kmeans_metrics.csv", index=False)
    summary = {
        "emb_path": str(emb_path),
        "n_samples": int(len(X)),
        "dim": int(X.shape[1]),
        "k_min": k_min,
        "k_max": k_max,
        "normalize_l2": normalize,
        "best_silhouette_k": int(best_sil.k),
        "best_silhouette": float(best_sil.silhouette),
        "best_dbi_k": int(best_dbi.k),
        "best_dbi": float(best_dbi.davies_bouldin),
    }
    with open(outdir / "kmeans_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    k_np = df["k"].to_numpy()
    sil_np = df["silhouette"].to_numpy()
    dbi_np = df["davies_bouldin"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_np, sil_np, marker="o")
    ax.axvline(best_sil.k, color="gray", ls="--", lw=0.8, label=f"best k={int(best_sil.k)}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette (higher=better)")
    ax.set_title("Silhouette vs k")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "silhouette_vs_k.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_np, dbi_np, marker="o", color="C1")
    ax.axvline(best_dbi.k, color="gray", ls="--", lw=0.8, label=f"best k={int(best_dbi.k)}")
    ax.set_xlabel("k")
    ax.set_ylabel("Davies–Bouldin (lower=better)")
    ax.set_title("Davies–Bouldin vs k")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "davies_bouldin_vs_k.png", dpi=160)
    plt.close(fig)

    print(f"[OK] 聚类曲线已保存: {outdir.resolve()}")
    return df


def clustering_main(
    emb_path: Path | str | None = None,
    outdir: Path | str | None = None,
    k_min: int = 2,
    k_max: int = 20,
    normalize: bool = True,
    subsample_n: Optional[int] = 20000,
    use_minibatch: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    emb_path = Path(emb_path or OUT_STATE_VAE / "state_embeddings_mu.npy")
    outdir = Path(outdir or OUT_FIGURES / "clustering")
    return run_eval(
        emb_path=emb_path,
        outdir=outdir,
        k_min=k_min,
        k_max=k_max,
        normalize=normalize,
        subsample_n=subsample_n,
        use_minibatch=use_minibatch,
        random_state=random_state,
    )


def kmeans_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, default=str(OUT_STATE_VAE / "state_embeddings_mu.npy"))
    ap.add_argument("--outdir", type=str, default=str(OUT_FIGURES / "clustering"))
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--subsample", type=int, default=20000)
    ap.add_argument("--minibatch", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clustering_main(
        emb_path=args.emb,
        outdir=args.outdir,
        k_min=args.k_min,
        k_max=args.k_max,
        normalize=not args.no_norm,
        subsample_n=None if args.subsample == 0 else args.subsample,
        use_minibatch=args.minibatch,
        random_state=args.seed,
    )


if __name__ == "__main__":
    kmeans_main()
