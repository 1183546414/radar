"""One-command PARD inference for a single jammer state.

Example:
  PYTHONPATH=src python -m radar.pard.recommend --checkpoint outputs/pard/pard_best.pt --state jammer.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

from .config import PARDConfig, STATE_NUMERIC
from .features import RawFeatureSpec
from .model import PARDModel
from .robust import RobustCEMOptimizer


def load_pard(checkpoint: Path, device: torch.device | None = None) -> Tuple[PARDModel, RawFeatureSpec, PARDConfig, torch.device]:
    """Loads every inference dependency solely from a PARD checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = PARDConfig(**ckpt["config"])
    spec = RawFeatureSpec.from_dict(ckpt["feature_spec"])
    model = PARDModel(ckpt["vocab_sizes"], cfg.quantiles, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout, cfg.fourier_bands).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, spec, cfg, device


def normalize_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Ensures inactive jammer mechanism fields have explicit zero values."""
    result = dict(state)
    if "Jammer_Type" not in result:
        raise ValueError("state requires Jammer_Type")
    for col in STATE_NUMERIC:
        result[col] = float(result.get(col, 0.0))
    return result


def recommend(checkpoint: Path, state: Mapping[str, Any], seed: int = 20261024) -> Dict[str, Any]:
    model, spec, cfg, device = load_pard(checkpoint)
    optimizer = RobustCEMOptimizer(
        model, spec, device, population=cfg.cem_population,
        elite_fraction=cfg.cem_elite_fraction, iterations=cfg.cem_iterations,
        inner_perturbations=cfg.inner_perturbations, radius=cfg.adversarial_radius,
        beta=cfg.risk_beta, cvar_weight=cfg.cvar_weight, seed=seed,
    )
    action, score = optimizer.optimize(normalize_state(state))
    return {"action": {k: (float(v) if hasattr(v, "item") else v) for k, v in action.items()}, "robust_score": float(score), "checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend one robust radar action from a PARD checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="JSON file containing the jammer state.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path; stdout is always printed.")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = recommend(args.checkpoint, state, args.seed)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
