# -*- coding: utf-8 -*-
"""
无标签潜空间聚类评估：
- 扫 k=K_MIN..K_MAX 的 KMeans（或 MiniBatchKMeans）
- 计算 Silhouette（越高越好）与 Davies–Bouldin（越低越好）
- 画出指标随 k 的曲线，并打印最佳 k

用法示例：
python analyze_clustering.py --emb result/state_vae_runs/state_embeddings_mu.npy
python analyze_clustering.py --emb result/state_vae_runs/state_embeddings_mu.npy --minibatch --subsample 5000 --k-max 12
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

from typing import Optional  # 放在文件顶部 import 区

def run_eval(
    emb_path: Path,
    k_min: int = 2,
    k_max: int = 20,
    normalize: bool = True,
    subsample_n: Optional[int] = 20000, 
    use_minibatch: bool = False,
    random_state: int = 42,
):
    # 1) 加载 μ
    X = np.load(emb_path)
    assert X.ndim == 2, f"Embeddings must be 2D, got {X.shape}"

    # 2) 可选子采样（大数据时用于加速）
    if subsample_n and len(X) > subsample_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=subsample_n, replace=False)
        X = X[idx]

    # 3) 可选 L2 归一化（常对余弦结构更稳）
    if normalize:
        X = l2_normalize(X)

    # 4) 扫 k
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

    # 5) 打印表格与摘要
    print("\n=== Summary ===")
    print(f"Embeddings: {emb_path}  |  n={len(X)}, dim={X.shape[1]}")
    print(f"Best Silhouette:      k={int(best_sil.k)}  score={best_sil.silhouette:.4f}  (higher=better)")
    print(f"Best Davies–Bouldin:  k={int(best_dbi.k)}  score={best_dbi.davies_bouldin:.4f}  (lower=better)")
    print()
    print(df.to_string(index=False, formatters={
        "silhouette": lambda v: f"{v:.4f}",
        "davies_bouldin": lambda v: f"{v:.4f}",
    }))

    # 6) 画图（将 pandas.Series 转 numpy，避免 pandas 2.2+ 多维切片报错）
    k_np = df["k"].to_numpy()
    sil_np = df["silhouette"].to_numpy()
    dbi_np = df["davies_bouldin"].to_numpy()

    plt.figure()
    plt.plot(k_np, sil_np, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette (higher=better)")
    plt.title("Silhouette vs k")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(k_np, dbi_np, marker="o")
    plt.xlabel("k")
    plt.ylabel("Davies–Bouldin (lower=better)")
    plt.title("Davies–Bouldin vs k")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def kmeans_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, default="result/state_vae_runs/state_embeddings_mu.npy",
                    help="路径到 state_embeddings_mu.npy")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--no-norm", action="store_true",
                    help="关闭 L2 归一化")
    ap.add_argument("--subsample", type=int, default=20000,
                    help="子采样大小（大数据加速），0 表示不用子采样")
    ap.add_argument("--minibatch", action="store_true",
                    help="使用 MiniBatchKMeans（更快）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_eval(
        emb_path=Path(args.emb),
        k_min=args.k_min,
        k_max=args.k_max,
        normalize=not args.no_norm,
        subsample_n=None if args.subsample == 0 else args.subsample,
        use_minibatch=args.minibatch,
        random_state=args.seed,
    )


if __name__ == "__main__":
    kmeans_main()
