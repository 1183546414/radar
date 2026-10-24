"""Batch PARD evaluation using the same whole-scenario holdout protocol.

The default evaluator uses each held-out scenario's K recorded actions as a
reproducible local oracle.  `--dense-oracle` may point to an externally
generated per-scenario oracle table (e.g. RadarSimPy best-of-5000 results) to
report the project's primary true-regret metric.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ACTION_CATEGORICAL, ACTION_NUMERIC, PARDConfig, SCENARIO, STATE_CATEGORICAL, STATE_NUMERIC, TARGET
from .recommend import load_pard, normalize_state
from .robust import RobustCEMOptimizer
from .train import _read_csv, split_by_scenario


def _action_columns() -> List[str]:
    return ACTION_CATEGORICAL + ACTION_NUMERIC


def _state_columns() -> List[str]:
    return STATE_CATEGORICAL + STATE_NUMERIC


def _load_scorer(import_path: str):
    """Loads `package.module:function(state_dict, action_dict) -> float` explicitly."""
    module_name, sep, function_name = import_path.partition(":")
    if not sep:
        raise ValueError("scorer must use package.module:function format")
    fn = getattr(importlib.import_module(module_name), function_name)
    return fn


def evaluate(checkpoint: Path, csv: Path, output_dir: Path, dense_oracle: Path | None = None, max_scenarios: int | None = None, action_mode: str = "recorded", scorer_path: str | None = None) -> Dict:
    model, spec, cfg, device = load_pard(checkpoint)
    frame = _read_csv(csv)
    _, _, test = split_by_scenario(frame, cfg)
    sids = sorted(test[SCENARIO].unique().tolist())
    if max_scenarios is not None:
        sids = sids[:max_scenarios]
    oracle_map: Dict[int, float] = {}
    if dense_oracle:
        table = pd.read_csv(dense_oracle)
        value_col = next((c for c in ("dense_oracle_sinr_db", "oracle_sinr_db", "SINR_dB") if c in table), None)
        if value_col is None or SCENARIO not in table:
            raise ValueError("dense oracle CSV must contain scenario_id and a SINR column")
        oracle_map = dict(zip(table[SCENARIO].astype(int), table[value_col].astype(float)))
    scorer = _load_scorer(scorer_path) if scorer_path else None
    if action_mode == "cem" and scorer is None:
        raise ValueError("CEM action evaluation needs --scorer package.module:function, typically a RadarSimPy adapter.")
    optimizer = RobustCEMOptimizer(model, spec, device, population=cfg.cem_population, elite_fraction=cfg.cem_elite_fraction, iterations=cfg.cem_iterations, inner_perturbations=cfg.inner_perturbations, radius=cfg.adversarial_radius, beta=cfg.risk_beta, cvar_weight=cfg.cvar_weight, seed=cfg.seed)
    rows = []
    for sid in sids:
        group = test[test[SCENARIO] == sid].reset_index(drop=True)
        state = normalize_state(group.loc[0, _state_columns()].to_dict())
        action, robust_score = optimizer.optimize(state)
        # `recorded` isolates learned ranking on identical K candidates. `cem`
        # evaluates the actual continuous robust action through the supplied
        # signal-level backend, so it is the end-to-end comparison setting.
        if action_mode == "recorded":
            action_frame = group[_action_columns()].copy()
            candidates = pd.concat([pd.DataFrame([state] * len(group)), action_frame], axis=1)
            predicted = optimizer._predict(candidates)
            chosen = int(np.argmax(predicted))
            chosen_true = float(group.loc[chosen, TARGET])
            reported_action = group.loc[chosen, _action_columns()].to_dict()
        else:
            chosen_true = float(scorer(state, action))
            reported_action = action
        local_oracle = float(group[TARGET].max())
        dense = oracle_map.get(int(sid), np.nan)
        rows.append({
            SCENARIO: int(sid), "recommended_sinr_db": chosen_true,
            "oracle_k_sinr_db": local_oracle,
            "regret_vs_oracle_k_db": local_oracle - chosen_true,
            "dense_oracle_sinr_db": dense,
            "true_regret_db": dense - chosen_true if np.isfinite(dense) else np.nan,
            "hit_at_1_k": int(chosen_true == local_oracle), "robust_score": float(robust_score),
            "action_mode": action_mode,
            **{f"recommended_{col}": reported_action[col] for col in _action_columns()},
        })
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "pard_per_scenario.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_scenarios": len(result), "checkpoint": str(checkpoint), "csv": str(csv),
        "mean_recommended_sinr_db": float(result.recommended_sinr_db.mean()),
        "mean_regret_vs_oracle_k_db": float(result.regret_vs_oracle_k_db.mean()),
        "median_regret_vs_oracle_k_db": float(result.regret_vs_oracle_k_db.median()),
        "p95_regret_vs_oracle_k_db": float(result.regret_vs_oracle_k_db.quantile(0.95)),
        "hit_at_1_k": float(result.hit_at_1_k.mean()),
    }
    if result.true_regret_db.notna().any():
        valid = result.true_regret_db.dropna()
        summary.update({"mean_true_regret_db": float(valid.mean()), "median_true_regret_db": float(valid.median()), "p95_true_regret_db": float(valid.quantile(0.95)), "n_dense_oracle_matched": int(valid.size)})
    (output_dir / "pard_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PARD on deterministic held-out scenarios.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/pard_eval"))
    parser.add_argument("--dense-oracle", type=Path, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--action-mode", choices=("recorded", "cem"), default="recorded")
    parser.add_argument("--scorer", type=str, default=None, help="Required for CEM: package.module:function(state, action) -> SINR dB")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.csv, args.output, args.dense_oracle, args.max_scenarios, args.action_mode, args.scorer), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
