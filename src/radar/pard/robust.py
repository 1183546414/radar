"""Adversarial state perturbation and CEM action search for PARD inference."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import ACTION_BOUNDS
from .features import RawFeatureSpec


PERTURBABLE_STATE_COLUMNS = [
    "Jammer_CarrierFreq_GHz", "Jammer_Gain_dB", "Jammer_Bandwidth_GHz",
    "Jammer_Sweep_Step_GHz", "Jammer_IntSample_Dur_ns",
    "Jammer_IntSample_Delay_ns", "Jammer_IntSample_Intv_ns", "Jammer_FalseTarget_Num",
]


def perturb_state_rows(state: Mapping, n: int, radius: float, rng: np.random.Generator) -> pd.DataFrame:
    """Samples admissible sensor/parameter error; categorical jammer type is fixed."""
    rows = [dict(state) for _ in range(n)]
    out = pd.DataFrame(rows)
    for col in PERTURBABLE_STATE_COLUMNS:
        if col not in out:
            continue
        x = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(float)
        # A zero means a type-inactive parameter: preserve it and its semantics.
        active = np.abs(x) > 1e-12
        x[active] *= 1.0 + rng.uniform(-radius, radius, active.sum())
        out[col] = x
    return out


class RobustCEMOptimizer:
    def __init__(self, model, spec: RawFeatureSpec, device: torch.device, *, population: int = 512, elite_fraction: float = 0.10, iterations: int = 6, inner_perturbations: int = 12, radius: float = 0.10, beta: float = 0.20, cvar_weight: float = 0.65, seed: int = 20261024):
        self.model, self.spec, self.device = model, spec, device
        self.population, self.elite_fraction, self.iterations = population, elite_fraction, iterations
        self.inner_perturbations, self.radius = inner_perturbations, radius
        self.beta, self.cvar_weight = beta, cvar_weight
        self.rng = np.random.default_rng(seed)

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.spec.transform(frame)
        with torch.no_grad():
            q = self.model(**{k: torch.as_tensor(v, device=self.device) for k, v in x.items()})
            return self.model.risk_score(q, self.beta, self.cvar_weight).cpu().numpy()

    def _worst_case_score(self, state: Mapping, actions: pd.DataFrame) -> np.ndarray:
        states = perturb_state_rows(state, self.inner_perturbations, self.radius, self.rng)
        tiled_actions = pd.concat([actions] * len(states), ignore_index=True)
        tiled_states = states.loc[states.index.repeat(len(actions))].reset_index(drop=True)
        score = self._predict(pd.concat([tiled_states.reset_index(drop=True), tiled_actions.reset_index(drop=True)], axis=1))
        return score.reshape(len(states), len(actions)).min(axis=0)

    def optimize(self, state: Mapping) -> Tuple[Dict, float]:
        """Mixed discrete/continuous CEM; returns an action without simulator calls."""
        categories = {c: [v for v in self.spec.vocabularies[c] if v != "<unknown>"] for c in ("Radar_Type", "Radar_ChirpType")}
        means = np.array([(lo + hi) / 2 for lo, hi in ACTION_BOUNDS.values()], dtype=float)
        stds = np.array([(hi - lo) / 2 for lo, hi in ACTION_BOUNDS.values()], dtype=float)
        names, bounds = list(ACTION_BOUNDS), np.asarray(list(ACTION_BOUNDS.values()), dtype=float)
        best_action, best_score = None, -np.inf
        for _ in range(self.iterations):
            values = self.rng.normal(means, stds, size=(self.population, len(names)))
            values = np.clip(values, bounds[:, 0], bounds[:, 1])
            actions = pd.DataFrame(values, columns=names)
            for c, choices in categories.items():
                actions[c] = self.rng.choice(choices, size=len(actions))
            score = self._worst_case_score(state, actions)
            elite = actions.iloc[np.argsort(score)[-max(2, int(self.population * self.elite_fraction)):]]
            means = elite[names].to_numpy().mean(axis=0)
            stds = np.maximum(elite[names].to_numpy().std(axis=0), (bounds[:, 1] - bounds[:, 0]) * 0.02)
            idx = int(np.argmax(score))
            if score[idx] > best_score:
                best_action, best_score = actions.iloc[idx].to_dict(), float(score[idx])
        return best_action, best_score
