# -*- coding: utf-8 -*-
"""
方向 B：检索条件 Q 网络 Q(z_query, z_ctx, A) -> R

z_ctx = 同类型 Top-K 相似训练场景潜向量的相似度加权均值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

DROPOUT = 0.1

# 余弦相似度 softmax 温度（越小越偏向最相似邻居）
Z_CTX_TEMPERATURE = 0.07


class QRetrievalNet(nn.Module):
    """Q(z_query, z_ctx, a) -> 标量奖励（标准化空间）。"""

    def __init__(self, z_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = z_dim * 2 + a_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        z_query: torch.Tensor,
        a: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([z_query, z_ctx, a], dim=-1))


def aggregate_neighbor_z(
    z_neighbors: np.ndarray,
    sims: Union[np.ndarray, None] = None,
    temperature: float = Z_CTX_TEMPERATURE,
) -> np.ndarray:
    """[K, z_dim] + 余弦相似度 -> 加权均值上下文；无 sims 时退化为算术均值。"""
    if len(z_neighbors) == 0:
        raise ValueError("empty neighbors")
    z_neighbors = np.asarray(z_neighbors, dtype=np.float32)
    if sims is None or len(sims) != len(z_neighbors):
        return z_neighbors.mean(axis=0).astype(np.float32)
    w = np.asarray(sims, dtype=np.float64)
    w = w / max(float(temperature), 1e-8)
    w = np.exp(w - w.max())
    w = w / (w.sum() + 1e-8)
    return (z_neighbors * w[:, None]).sum(axis=0).astype(np.float32)


def warm_start_from_q_latent(
    model: QRetrievalNet,
    q_latent_ckpt_path: Path,
    device: torch.device,
) -> None:
    """用 QLatentNet 权重初始化 QRetrievalNet（首层拆分 z_ctx 列，其余层直接拷贝）。"""
    ckpt = torch.load(q_latent_ckpt_path, map_location=device, weights_only=False)
    latent_sd = ckpt["model"]
    z_dim = int(ckpt["z_dim"])
    a_dim = int(ckpt["a_dim"])

    w_latent = latent_sd["net.0.weight"]
    b_latent = latent_sd["net.0.bias"]
    hidden = w_latent.shape[0]

    w_new = torch.zeros(hidden, z_dim * 2 + a_dim, device=device)
    w_new[:, :z_dim] = w_latent[:, :z_dim]
    w_new[:, z_dim : 2 * z_dim] = w_latent[:, :z_dim] * 0.5
    w_new[:, 2 * z_dim :] = w_latent[:, z_dim:]

    state = model.state_dict()
    state["net.0.weight"] = w_new.cpu()
    state["net.0.bias"] = b_latent.cpu()
    for key, val in latent_sd.items():
        if key.startswith("net.") and not key.startswith("net.0.") and key in state:
            state[key] = val.cpu()
    model.load_state_dict(state)


@torch.no_grad()
def predict_q_retrieval(
    model: QRetrievalNet,
    z_query: np.ndarray,
    a: np.ndarray,
    z_ctx: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    z_query = np.asarray(z_query, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    z_ctx = np.asarray(z_ctx, dtype=np.float32)
    if z_query.ndim == 1:
        z_query = z_query.reshape(1, -1)
    if z_ctx.ndim == 1:
        z_ctx = z_ctx.reshape(1, -1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = max(len(z_query), len(a), len(z_ctx))
    if len(z_query) == 1 and n > 1:
        z_query = np.repeat(z_query, n, axis=0)
    if len(z_ctx) == 1 and n > 1:
        z_ctx = np.repeat(z_ctx, n, axis=0)
    if len(a) == 1 and n > 1:
        a = np.repeat(a, n, axis=0)
    assert len(z_query) == len(a) == len(z_ctx)

    preds = []
    for i in range(0, n, batch_size):
        zq = torch.from_numpy(z_query[i : i + batch_size]).to(device)
        zc = torch.from_numpy(z_ctx[i : i + batch_size]).to(device)
        ab = torch.from_numpy(a[i : i + batch_size]).to(device)
        preds.append(model(zq, ab, zc).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)
