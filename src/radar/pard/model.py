"""Token interaction model predicting a conditional SINR quantile distribution."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .config import ACTION_CATEGORICAL, ACTION_NUMERIC, STATE_CATEGORICAL, STATE_NUMERIC


class NumericToken(nn.Module):
    def __init__(self, d_model: int, bands: int):
        super().__init__()
        self.freq = nn.Parameter(torch.arange(1, bands + 1, dtype=torch.float32), requires_grad=False)
        self.project = nn.Sequential(nn.Linear(2 + 2 * bands, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        phase = x.unsqueeze(-1) * self.freq.to(x)
        raw = torch.cat([x.unsqueeze(-1), torch.log1p(x.abs()).unsqueeze(-1), phase.sin(), phase.cos()], dim=-1)
        return self.project(raw) * present.unsqueeze(-1)


class PARDModel(nn.Module):
    """New raw-field encoder. It has no dependency on legacy project models."""
    def __init__(self, vocab_sizes: Sequence[int], quantiles: Sequence[float], d_model: int = 96, n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1, fourier_bands: int = 4):
        super().__init__()
        self.quantiles = list(quantiles)
        self.n_numeric = len(STATE_NUMERIC) + len(ACTION_NUMERIC)
        self.numeric = NumericToken(d_model, fourier_bands)
        self.category_embeddings = nn.ModuleList([nn.Embedding(int(n), d_model) for n in vocab_sizes])
        n_tokens = self.n_numeric + len(vocab_sizes)
        self.field_embedding = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        roles = [0] * len(STATE_NUMERIC) + [1] * len(ACTION_NUMERIC) + [0] * len(STATE_CATEGORICAL) + [1] * len(ACTION_CATEGORICAL)
        self.register_buffer("role_ids", torch.tensor(roles, dtype=torch.long), persistent=False)
        self.role_embedding = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 3, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.physics = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, len(quantiles)))

    def forward(self, numeric: torch.Tensor, present: torch.Tensor, categorical: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        n_tokens = self.numeric(numeric, present)
        c_tokens = torch.stack([emb(categorical[:, i]) for i, emb in enumerate(self.category_embeddings)], dim=1)
        tokens = torch.cat([n_tokens, c_tokens], dim=1)
        tokens = tokens + self.field_embedding.unsqueeze(0) + self.role_embedding(self.role_ids).unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(torch.cat([pooled, self.physics(physics)], dim=-1))

    def risk_score(self, quantile_prediction: torch.Tensor, beta: float = 0.2, cvar_weight: float = 0.65) -> torch.Tensor:
        """Higher is safer: lower-tail CVaR blended with median minus spread."""
        q = torch.as_tensor(self.quantiles, device=quantile_prediction.device)
        lower = quantile_prediction[:, q <= 0.10].mean(dim=1)
        median = quantile_prediction[:, torch.argmin((q - 0.50).abs())]
        spread = quantile_prediction[:, -1] - quantile_prediction[:, 0]
        return cvar_weight * lower + (1.0 - cvar_weight) * (median - beta * spread)
