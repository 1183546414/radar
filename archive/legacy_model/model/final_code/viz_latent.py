# viz_latent.py
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def auto_perplexity(n_samples: int) -> int:
    # t-SNE 要求 perplexity < n_samples；经验上 5~50 较稳
    if n_samples <= 10:
        return max(2, n_samples // 3) or 2
    return max(5, min(30, (n_samples - 1) // 3))

def maybe_downsample(X: np.ndarray, labels: np.ndarray | None, max_points: int, seed: int = 42):
    n = X.shape[0]
    if n <= max_points:
        return X, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return X[idx], (labels[idx] if labels is not None else None)

def scatter2d(ax, Z2, labels=None, title=""):
    if labels is None:
        ax.scatter(Z2[:, 0], Z2[:, 1], s=6, alpha=0.7)
    else:
        # 若提供标签，则按类别上色
        labs = np.array(labels)
        for lab in np.unique(labs):
            mask = labs == lab
            ax.scatter(Z2[mask, 0], Z2[mask, 1], s=6, alpha=0.7, label=str(lab))
        ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

def vis_lat_main():
    parser = argparse.ArgumentParser(description="PCA & t-SNE visualization for latent embeddings (mu).")
    parser.add_argument("--npy", type=str, default= "./state_vae_runs/state_embeddings_mu.npy", help="Path to state_embeddings_mu.npy")
    parser.add_argument("--labels", type=str, default=None,
                        help="Optional path to labels file (txt/csv, one label per line)")
    parser.add_argument("--outdir", type=str, default="latent_viz", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE and sampling")
    parser.add_argument("--max_tsne_points", type=int, default=20000, help="Max points used for t-SNE to避免超大数据过慢")
    parser.add_argument("--tsne_perplexity", type=float, default=None, help="Manually set t-SNE perplexity")
    parser.add_argument("--tsne_iter", type=int, default=1000, help="t-SNE n_iter")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 加载潜向量
    Z = np.load(args.npy)
    if Z.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {Z.shape}")
    n, d = Z.shape
    print(f"[INFO] Loaded embeddings: {Z.shape}")

    # 2) 读可选标签
    labs = None
    if args.labels is not None:
        with open(args.labels, "r", encoding="utf-8") as f:
            labs = [line.strip() for line in f if line.strip() != ""]
        if len(labs) != n:
            raise ValueError(f"Labels length ({len(labs)}) != embeddings rows ({n})")
        labs = np.array(labs)
        print(f"[INFO] Loaded labels: {len(labs)}")

    # 3) 处理 NaN/Inf
    mask_finite = np.isfinite(Z).all(axis=1)
    if not mask_finite.all():
        kept = mask_finite.sum()
        print(f"[WARN] Dropping {n - kept} rows containing NaN/Inf")
        Z = Z[mask_finite]
        if labs is not None:
            labs = labs[mask_finite]
        n = Z.shape[0]

    # 4) PCA（自动中心化；无需标准化）
    pca = PCA(n_components=2, random_state=args.seed)
    Z_pca = pca.fit_transform(Z)
    var_ratio = pca.explained_variance_ratio_
    print(f"[INFO] PCA explained variance ratio: PC1={var_ratio[0]:.4f}, PC2={var_ratio[1]:.4f}")

    # 保存 PCA 结果
    np.save(os.path.join(args.outdir, "Z_pca.npy"), Z_pca)
    with open(os.path.join(args.outdir, "pca_variance.txt"), "w", encoding="utf-8") as f:
        f.write(f"PC1={var_ratio[0]:.6f}\nPC2={var_ratio[1]:.6f}\n")

    # 绘制 PCA 散点
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter2d(ax, Z_pca, labels=labs, title=f"PCA (PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%})")
    pca_png = os.path.join(args.outdir, "pca_scatter.png")
    plt.tight_layout()
    plt.savefig(pca_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved PCA scatter -> {pca_png}")

    # 5) t-SNE（对超大样本自动降采样；自动 perplexity）
    Z_tsne_in, labs_tsne = maybe_downsample(Z, labs, max_points=args.max_tsne_points, seed=args.seed)
    n_tsne = Z_tsne_in.shape[0]
    perplexity = args.tsne_perplexity if args.tsne_perplexity is not None else auto_perplexity(n_tsne)
    if perplexity >= n_tsne:
        perplexity = max(2, n_tsne // 3)
    print(f"[INFO] t-SNE using n={n_tsne}, perplexity={perplexity}, n_iter={args.tsne_iter}, seed={args.seed}")

    tsne = TSNE(
        n_components=2,
        init="random",
        perplexity=perplexity,
        n_iter=args.tsne_iter,
        random_state=args.seed,
        learning_rate="auto",
        early_exaggeration=12.0,
        verbose=1
    )
    Z_tsne = tsne.fit_transform(Z_tsne_in)

    # 保存 t-SNE 结果
    np.save(os.path.join(args.outdir, "Z_tsne.npy"), Z_tsne)

    # 绘制 t-SNE 散点
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter2d(ax, Z_tsne, labels=labs_tsne, title="t-SNE")
    tsne_png = os.path.join(args.outdir, "tsne_scatter.png")
    plt.tight_layout()
    plt.savefig(tsne_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved t-SNE scatter -> {tsne_png}")

    print("[DONE] Outputs:")
    print(f" - {pca_png}")
    print(f" - {tsne_png}")
    print(f" - {os.path.join(args.outdir, 'Z_pca.npy')}")
    print(f" - {os.path.join(args.outdir, 'Z_tsne.npy')}")
    print(f" - {os.path.join(args.outdir, 'pca_variance.txt')}")

if __name__ == "__main__":
    vis_lat_main()
