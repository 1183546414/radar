# -*- coding: utf-8 -*-
"""State VAE 重建可视化：5×4 对比图与 |original - reconstructed| 误差曲线。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from radar.frozen_vae import load_frozen_vae_encoder
from radar.paths import DATA_V2_PROCESSED, OUT_FIGURES

DEFAULT_COMPARE_OUT = OUT_FIGURES / "reconstructed_samples.png"
DEFAULT_ABS_ERR_OUT = OUT_FIGURES / "reconstruction_abs_error.png"
N_GRID_SAMPLES = 20
NROWS = 5
NCOLS = 4
SEED = 42


def _pick_sample_indices(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    if n_samples != NROWS * NCOLS:
        raise ValueError(f"n_samples 应为 {NROWS * NCOLS}（5×4 网格），收到 {n_samples}")
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_samples, replace=False)


@torch.no_grad()
def _reconstruct_samples(
    s_std: np.ndarray,
    sample_indices: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model, _, _ = load_frozen_vae_encoder(device)
    originals = []
    reconstructions = []
    for row_idx in sample_indices:
        s_vec = s_std[int(row_idx)]
        s_t = torch.from_numpy(s_vec).unsqueeze(0).to(device)
        mu, _ = model.encode(s_t)
        s_rec = model.dec(mu).squeeze(0).cpu().numpy()
        originals.append(s_vec)
        reconstructions.append(s_rec)
    return np.stack(originals), np.stack(reconstructions)


def _save_compare_grid(
    out_path: Path,
    sample_indices: np.ndarray,
    originals: np.ndarray,
    reconstructions: np.ndarray,
) -> None:
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(10, 12))
    for ax, row_idx, s_vec, s_rec in zip(axes.ravel(), sample_indices, originals, reconstructions):
        x = np.arange(len(s_vec))
        ax.plot(x, s_vec, label="Original (numeric)", linewidth=1.5)
        ax.plot(x, s_rec, label="Reconstructed (numeric)", linewidth=1.5)
        ax.set_title(f"Sample #{int(row_idx)}", fontsize=9)
        ax.set_xlabel("Feature Index", fontsize=8)
        ax.set_ylabel("Standardized value", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_abs_error_grid(
    out_path: Path,
    sample_indices: np.ndarray,
    originals: np.ndarray,
    reconstructions: np.ndarray,
) -> None:
    abs_errors = np.abs(originals - reconstructions)
    ymax = float(abs_errors.max()) * 1.1 or 1.0

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(10, 12))
    for ax, row_idx, err in zip(axes.ravel(), sample_indices, abs_errors):
        x = np.arange(len(err))
        ax.plot(x, err, color="C3", linewidth=1.5, label="|Original - Reconstructed|")
        ax.set_title(f"Sample #{int(row_idx)}", fontsize=9)
        ax.set_xlabel("Feature Index", fontsize=8)
        ax.set_ylabel("Absolute error", fontsize=8)
        ax.set_ylim(0.0, ymax)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_reconstructed_samples_grid(
    out_path: Path | str = DEFAULT_COMPARE_OUT,
    n_samples: int = N_GRID_SAMPLES,
    seed: int = SEED,
    device: torch.device | None = None,
) -> Path:
    """随机抽 n_samples 条 S_std，绘制 5×4 Original vs Reconstructed 对比图。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_std = np.load(DATA_V2_PROCESSED / "S_std.npy").astype(np.float32)
    sample_indices = _pick_sample_indices(len(s_std), n_samples, seed)
    originals, reconstructions = _reconstruct_samples(s_std, sample_indices, device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_compare_grid(out_path, sample_indices, originals, reconstructions)
    print(f"[OK] 重建对比图已保存: {out_path.resolve()}")
    return out_path


@torch.no_grad()
def plot_reconstruction_abs_error_grid(
    out_path: Path | str = DEFAULT_ABS_ERR_OUT,
    n_samples: int = N_GRID_SAMPLES,
    seed: int = SEED,
    device: torch.device | None = None,
) -> Path:
    """随机抽 n_samples 条 S_std，绘制 5×4 |original - reconstructed| 误差曲线。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_std = np.load(DATA_V2_PROCESSED / "S_std.npy").astype(np.float32)
    sample_indices = _pick_sample_indices(len(s_std), n_samples, seed)
    originals, reconstructions = _reconstruct_samples(s_std, sample_indices, device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_abs_error_grid(out_path, sample_indices, originals, reconstructions)
    print(f"[OK] 重建绝对误差图已保存: {out_path.resolve()}")
    return out_path


def plot_reconstructed_samples_main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_std = np.load(DATA_V2_PROCESSED / "S_std.npy").astype(np.float32)
    sample_indices = _pick_sample_indices(len(s_std), N_GRID_SAMPLES, SEED)
    originals, reconstructions = _reconstruct_samples(s_std, sample_indices, device)

    compare_out = Path(DEFAULT_COMPARE_OUT)
    err_out = Path(DEFAULT_ABS_ERR_OUT)
    compare_out.parent.mkdir(parents=True, exist_ok=True)

    _save_compare_grid(compare_out, sample_indices, originals, reconstructions)
    print(f"[OK] 重建对比图已保存: {compare_out.resolve()}")

    _save_abs_error_grid(err_out, sample_indices, originals, reconstructions)
    print(f"[OK] 重建绝对误差图已保存: {err_out.resolve()}")


if __name__ == "__main__":
    plot_reconstructed_samples_main()
