"""Raw-schema preprocessing and physically meaningful interaction features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ACTION_CATEGORICAL, ACTION_NUMERIC, STATE_CATEGORICAL, STATE_NUMERIC


ALL_NUMERIC = STATE_NUMERIC + ACTION_NUMERIC
ALL_CATEGORICAL = STATE_CATEGORICAL + ACTION_CATEGORICAL


def _safe_numeric(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)


def physical_features(frame: pd.DataFrame) -> np.ndarray:
    """Interactions supplied as inductive bias, never as a simulator shortcut."""
    rf, jf = _safe_numeric(frame, "Radar_CarrierFreq_GHz"), _safe_numeric(frame, "Jammer_CarrierFreq_GHz")
    rbw, jbw = _safe_numeric(frame, "Radar_Bandwidth_GHz"), _safe_numeric(frame, "Jammer_Bandwidth_GHz")
    rg, jg = _safe_numeric(frame, "Radar_Gain_dB"), _safe_numeric(frame, "Jammer_Gain_dB")
    pw, dur = _safe_numeric(frame, "Radar_PulseWidth_ns"), _safe_numeric(frame, "Jammer_IntSample_Dur_ns")
    interval, step = _safe_numeric(frame, "Jammer_IntSample_Intv_ns"), _safe_numeric(frame, "Jammer_Sweep_Step_GHz")
    eps = 1e-6
    left = np.maximum(rf - rbw / 2.0, jf - jbw / 2.0)
    right = np.minimum(rf + rbw / 2.0, jf + jbw / 2.0)
    overlap = np.maximum(right - left, 0.0) / np.maximum(rbw, eps)
    return np.column_stack([
        np.abs(rf - jf), overlap, rbw / np.maximum(jbw, eps), rg - jg,
        pw / np.maximum(dur, 1.0), dur / np.maximum(interval, 1.0),
        rbw / np.maximum(step, eps),
    ]).astype(np.float32)


@dataclass
class RawFeatureSpec:
    """Training-only statistics/vocabularies; persisted with the checkpoint."""
    medians: Dict[str, float]
    means: Dict[str, float]
    scales: Dict[str, float]
    vocabularies: Dict[str, Dict[str, int]]
    physics_mean: List[float]
    physics_scale: List[float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "RawFeatureSpec":
        medians, means, scales = {}, {}, {}
        for col in ALL_NUMERIC:
            x = _safe_numeric(frame, col)
            median = float(np.nanmedian(x))
            x = np.nan_to_num(x, nan=median, posinf=median, neginf=median)
            medians[col] = median
            means[col] = float(x.mean())
            scales[col] = float(max(x.std(), 1e-6))
        vocabs: Dict[str, Dict[str, int]] = {}
        for col in ALL_CATEGORICAL:
            values = sorted(frame[col].fillna("<missing>").astype(str).unique().tolist())
            vocabs[col] = {"<unknown>": 0, **{v: i + 1 for i, v in enumerate(values)}}
        p = physical_features(frame)
        return cls(medians, means, scales, vocabs, p.mean(axis=0).tolist(), np.maximum(p.std(axis=0), 1e-6).tolist())

    def transform(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        numeric, present, categorical = [], [], []
        for col in ALL_NUMERIC:
            x = _safe_numeric(frame, col)
            present.append(np.isfinite(x).astype(np.float32))
            x = np.nan_to_num(x, nan=self.medians[col], posinf=self.medians[col], neginf=self.medians[col])
            numeric.append((x - self.means[col]) / self.scales[col])
        for col in ALL_CATEGORICAL:
            vocab = self.vocabularies[col]
            categorical.append(np.asarray([vocab.get(v, 0) for v in frame[col].fillna("<missing>").astype(str)], dtype=np.int64))
        p = physical_features(frame)
        p = (p - np.asarray(self.physics_mean, dtype=np.float32)) / np.asarray(self.physics_scale, dtype=np.float32)
        return {
            "numeric": np.stack(numeric, axis=1).astype(np.float32),
            "present": np.stack(present, axis=1).astype(np.float32),
            "categorical": np.stack(categorical, axis=1),
            "physics": p.astype(np.float32),
        }

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Mapping) -> "RawFeatureSpec":
        return cls(**dict(data))
