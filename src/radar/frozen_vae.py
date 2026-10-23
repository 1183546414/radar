# -*- coding: utf-8 -*-
"""冻结 State VAE，仅作特征提取器。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from radar.paths import DATA_V2_PROCESSED, OUT_STATE_VAE


def _load_latent_dim_from_metrics() -> int | None:
    metrics_path = OUT_STATE_VAE / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("summary", {}).get("latent_dim", 0)) or None


def _validate_encoder_metadata() -> None:
    """拒绝静默加载带输入直连/行级切分的旧状态编码器。"""
    metrics_path = OUT_STATE_VAE / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少状态编码器元数据，请重新训练: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        summary = json.load(f).get("summary", {})
    if summary.get("decoder_input_skip") is not False or summary.get("split_unit") != "scenario":
        raise RuntimeError(
            "检测到旧版状态编码器（输入直连或非场景级切分）。"
            "请先运行 python -m radar.train.state_vae_contrastive 重新训练，"
            "然后重训下游 Q 模型。"
        )


def _resolve_vae_checkpoint() -> Path:
    for name in ("state_ae_or_vae.pt", "state_vae.pt"):
        p = OUT_STATE_VAE / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"未找到 VAE 权重，请先运行主流程训练 State VAE: {OUT_STATE_VAE}"
    )


def load_frozen_vae_encoder(device: torch.device | None = None) -> Tuple[nn.Module, int, int]:
    """
    加载训练好的 State VAE，冻结全部参数，仅暴露 encode(s)->mu。

    Returns:
        model: 完整 VAE（eval + requires_grad=False）
        s_dim: 状态维度
        latent_dim: 潜向量维度
    """
    from radar.train.state_vae_contrastive import Model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _validate_encoder_metadata()
    ckpt_path = _resolve_vae_checkpoint()
    S_std = np.load(DATA_V2_PROCESSED / "S_std.npy")
    s_dim = int(S_std.shape[1])

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "enc.mu.weight" in state:
        latent_dim = int(state["enc.mu.weight"].shape[0])
    else:
        latent_dim = _load_latent_dim_from_metrics() or 64

    model = Model(s_dim, latent_dim).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, s_dim, latent_dim


@torch.no_grad()
def encode_states(
    model: nn.Module,
    S: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """将标准化状态 S_std 批量编码为潜向量 μ。"""
    model.eval()
    feats = []
    S = np.asarray(S, dtype=np.float32)
    for i in range(0, len(S), batch_size):
        batch = torch.from_numpy(S[i : i + batch_size]).to(device)
        mu, _ = model.encode(batch)
        feats.append(mu.cpu().numpy())
    return np.concatenate(feats, axis=0)
