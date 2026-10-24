"""Distributional, within-scenario ranking and adversarial consistency losses."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    err = target.unsqueeze(1) - pred
    return torch.maximum(quantiles * err, (quantiles - 1.0) * err).mean()


def scenario_pair_ranking(score: torch.Tensor, reward: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
    """Pairwise ordering only within the same scenario; no cross-scene leakage."""
    same = scenario[:, None].eq(scenario[None, :])
    better = reward[:, None] > reward[None, :]
    mask = torch.triu(same & better, diagonal=1)
    if not mask.any():
        return score.new_zeros(())
    margin = score[:, None] - score[None, :]
    return F.softplus(-margin[mask]).mean()


def top_action_weight(reward: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
    """Weight high-reward actions more without accessing oracle labels at inference."""
    weight = torch.ones_like(reward)
    for sid in scenario.unique():
        idx = scenario.eq(sid)
        y = reward[idx]
        threshold = torch.quantile(y, 0.80)
        weight[idx] = 1.0 + (y >= threshold).float()
    return weight
