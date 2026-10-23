# viz_reconstruction.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def inv_std(X, mean, std):
    return X * std + mean

def ensure_2d(name, X):
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"{name} should be 2D, got shape {X.shape}")
    return X

def compute_metrics(orig, recon):
    residual = orig - recon
    mae = np.mean(np.abs(residual), axis=0)          # per-dim
    rmse = np.sqrt(np.mean(residual**2, axis=0))     # per-dim
    overall_mae = float(np.mean(np.abs(residual)))
    overall_rmse = float(np.sqrt(np.mean(residual**2)))
    return residual, mae, rmse, overall_mae, overall_rmse

def maybe_downsample(X, Y, max_points=20000, seed=42):
    n = X.shape[0]
    if n <= max_points:
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return X[idx], Y[idx]

def vis_recon_main():
    from radar.paths import DATA_V2_PROCESSED, OUT_FIGURES
    parser = argparse.ArgumentParser(description="Visualize differences between original and reconstructed vectors.")
    parser.add_argument("--orig", default=str(DATA_V2_PROCESSED / "S_std.npy"), type=str, help="Path to original vectors (.npy), shape [N, D]")
    parser.add_argument("--recon", default=str(DATA_V2_PROCESSED / "S_rec_std.npy"), type=str, help="Path to reconstructed vectors (.npy), shape [N, D]")
    parser.add_argument("--mean", type=str, default=None, help="Optional mean (.npy) for inverse-standardization")
    parser.add_argument("--std", type=str, default=None, help="Optional std (.npy) for inverse-standardization")
    parser.add_argument("--outdir", type=str, default=str(OUT_FIGURES / "recon"), help="Output directory")
    parser.add_argument("--max_scatter", type=int, default=20000, help="Max points in orig-vs-recon scatter")
    parser.add_argument("--sample_idx", type=int, default=None, help="Optional sample index for per-dim line plot")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 加载
    orig = np.load(args.orig)
    recon = np.load(args.recon)
    orig = ensure_2d("orig", orig)
    recon = ensure_2d("recon", recon)

    if orig.shape != recon.shape:
        raise ValueError(f"orig shape {orig.shape} != recon shape {recon.shape}")

    # 2) 可选反标准化
    if (args.mean is None) ^ (args.std is None):
        raise ValueError("If using inverse-standardization, provide BOTH --mean and --std.")
    if args.mean is not None and args.std is not None:
        mean = np.load(args.mean)
        std = np.load(args.std)
        # 广播兼容 [D] 或 [1, D]
        orig = inv_std(orig, mean, std)
        recon = inv_std(recon, mean, std)

    N, D = orig.shape
    print(f"[INFO] Data loaded: N={N}, D={D}")

    # 3) 指标
    residual, mae, rmse, overall_mae, overall_rmse = compute_metrics(orig, recon)
    print(f"[METRIC] Overall MAE = {overall_mae:.6f}, Overall RMSE = {overall_rmse:.6f}")

    # 保存指标
    np.save(os.path.join(args.outdir, "residual.npy"), residual)
    np.save(os.path.join(args.outdir, "per_dim_mae.npy"), mae)
    np.save(os.path.join(args.outdir, "per_dim_rmse.npy"), rmse)
    with open(os.path.join(args.outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall_MAE={overall_mae:.8f}\nOverall_RMSE={overall_rmse:.8f}\n")

    # 4) orig vs recon 散点（降采样 + 摊平成 1D）
    x = orig.reshape(-1, 1)
    y = recon.reshape(-1, 1)
    x_ds, y_ds = maybe_downsample(x, y, max_points=args.max_scatter, seed=42)

    plt.figure(figsize=(6, 5))
    plt.scatter(x_ds.ravel(), y_ds.ravel(), s=6, alpha=0.6)
    # 参考线 y = x
    xy_min = float(min(x_ds.min(), y_ds.min()))
    xy_max = float(max(x_ds.max(), y_ds.max()))
    plt.plot([xy_min, xy_max], [xy_min, xy_max])
    plt.xlabel("Original")
    plt.ylabel("Reconstruction")
    plt.title(f"Original vs Reconstruction (overall RMSE={overall_rmse:.4f})")
    plt.tight_layout()
    out_scatter = os.path.join(args.outdir, "orig_vs_recon_scatter.png")
    plt.savefig(out_scatter, dpi=200)
    plt.close()
    print(f"[OK] Saved -> {out_scatter}")

    # 5) 残差直方图
    plt.figure(figsize=(6, 5))
    plt.hist(residual.ravel(), bins=60)
    plt.xlabel("Residual = Original - Recon")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.tight_layout()
    out_hist = os.path.join(args.outdir, "residual_hist.png")
    plt.savefig(out_hist, dpi=200)
    plt.close()
    print(f"[OK] Saved -> {out_hist}")

    # 6) 逐维误差条形图（RMSE & MAE）
    idx = np.arange(D)
    width = 0.45
    plt.figure(figsize=(max(6, D * 0.3), 5))
    plt.bar(idx - width/2, rmse, width, label="RMSE")
    plt.bar(idx + width/2, mae,  width, label="MAE")
    plt.xlabel("Dimension")
    plt.ylabel("Error")
    plt.title("Per-dimension Error (RMSE & MAE)")
    plt.legend()
    plt.tight_layout()
    out_bar = os.path.join(args.outdir, "per_dim_errors.png")
    plt.savefig(out_bar, dpi=200)
    plt.close()
    print(f"[OK] Saved -> {out_bar}")

    # 7) 残差热力图（子样本）
    max_rows = min(500, N)  # 控制图像大小
    sel = np.linspace(0, N-1, max_rows, dtype=int) if N > max_rows else np.arange(N)
    plt.figure(figsize=(max(6, D * 0.3), 6))
    plt.imshow(residual[sel], aspect="auto", interpolation="nearest")
    plt.colorbar(label="Residual (orig - recon)")
    plt.xlabel("Dimension")
    plt.ylabel("Sample (subset)")
    plt.title("Residual Heatmap (subset)")
    plt.tight_layout()
    out_heat = os.path.join(args.outdir, "residual_heatmap.png")
    plt.savefig(out_heat, dpi=200)
    plt.close()
    print(f"[OK] Saved -> {out_heat}")

    # 8) 单样本维度曲线（可选）
    if args.sample_idx is not None:
        i = int(args.sample_idx)
        if not (0 <= i < N):
            raise ValueError(f"sample_idx must be in [0, {N-1}]")
        plt.figure(figsize=(max(6, D * 0.3), 4.5))
        plt.plot(orig[i], marker="o")
        plt.plot(recon[i], marker="x")
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.title(f"Sample #{i}: Original vs Reconstruction")
        plt.legend(["Original", "Reconstruction"])
        plt.tight_layout()
        out_line = os.path.join(args.outdir, f"sample_{i}_orig_vs_recon.png")
        plt.savefig(out_line, dpi=200)
        plt.close()
        print(f"[OK] Saved -> {out_line}")

    print("[DONE] Outputs saved in:", args.outdir)

if __name__ == "__main__":
    vis_recon_main()
