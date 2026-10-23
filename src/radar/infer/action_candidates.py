# -*- coding: utf-8 -*-
"""方向 A：检索候选扩展（局部扰动 / 新采样雷达动作）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from radar.data.build_dataset import action_vectors_to_std, row_to_action_std
from radar.data.high_contrast_sim import (
    RADAR_CONFIG,
    ghz_to_hz,
    hz_to_ghz,
    sample_radar_action,
    sinr_from_csv_row,
)

LOCAL_SEED_TOP = 10
LOCAL_PERTURB_PER_SEED = 10
GEN_NEW_ACTIONS = 100


@dataclass
class ScoredAction:
    """统一表示 train 行或合成雷达动作。"""

    a_std: np.ndarray
    radar_row: pd.Series
    train_row_idx: Optional[int] = None


def radar_dict_to_series(radar: dict) -> pd.Series:
    return pd.Series(
        {
            "Radar_CarrierFreq_GHz": hz_to_ghz(radar["r_freq"]),
            "Radar_Gain_dB": radar["r_gain"],
            "Radar_Bandwidth_GHz": hz_to_ghz(radar["r_bw"]),
            "Radar_PulseWidth_ns": radar["r_pw"],
            "Radar_Type": radar["r_type"],
            "Radar_ChirpType": radar["r_chirp"],
        }
    )


def train_row_to_radar_series(train_df: pd.DataFrame, row_idx: int) -> pd.Series:
    row = train_df.iloc[row_idx]
    return pd.Series(
        {
            "Radar_CarrierFreq_GHz": row["Radar_CarrierFreq_GHz"],
            "Radar_Gain_dB": row["Radar_Gain_dB"],
            "Radar_Bandwidth_GHz": row["Radar_Bandwidth_GHz"],
            "Radar_PulseWidth_ns": row["Radar_PulseWidth_ns"],
            "Radar_Type": row["Radar_Type"],
            "Radar_ChirpType": row["Radar_ChirpType"],
        }
    )


def perturb_radar_series(base: pd.Series, rng: np.random.Generator) -> pd.Series:
    """在雷达参数空间做局部扰动（频率/增益/带宽/脉宽）。"""
    freq_hz = ghz_to_hz(float(base["Radar_CarrierFreq_GHz"]))
    freq_hz += rng.integers(-5, 6) * 10e6
    freq_hz = float(np.clip(freq_hz, RADAR_CONFIG["freq_min"], RADAR_CONFIG["freq_max"]))

    gain = float(base["Radar_Gain_dB"]) + rng.integers(-2, 3)
    gain = float(np.clip(gain, RADAR_CONFIG["gain_min"], RADAR_CONFIG["gain_max"]))

    bw_hz = ghz_to_hz(float(base["Radar_Bandwidth_GHz"]))
    bw_hz *= float(rng.uniform(0.85, 1.15))
    bw_hz = float(np.clip(bw_hz, RADAR_CONFIG["bw_min"], RADAR_CONFIG["bw_max"]))

    pw = int(base["Radar_PulseWidth_ns"]) + int(rng.integers(-3, 4) * 1000)
    pw = int(np.clip(pw, RADAR_CONFIG["pw_min"], RADAR_CONFIG["pw_max"]))

    return pd.Series(
        {
            "Radar_CarrierFreq_GHz": hz_to_ghz(freq_hz),
            "Radar_Gain_dB": gain,
            "Radar_Bandwidth_GHz": hz_to_ghz(bw_hz),
            "Radar_PulseWidth_ns": pw,
            "Radar_Type": base["Radar_Type"],
            "Radar_ChirpType": base["Radar_ChirpType"],
        }
    )


def build_retrieval_train_actions(
    train_df: pd.DataFrame,
    artifacts: dict,
    retrieval_rows: np.ndarray,
) -> List[ScoredAction]:
    out: List[ScoredAction] = []
    for ri in retrieval_rows:
        ri = int(ri)
        radar = train_row_to_radar_series(train_df, ri)
        out.append(
            ScoredAction(
                a_std=row_to_action_std(radar, artifacts),
                radar_row=radar,
                train_row_idx=ri,
            )
        )
    return out


def build_local_perturbation_actions(
    train_df: pd.DataFrame,
    artifacts: dict,
    retrieval_rows: np.ndarray,
    seed_top_q_indices: np.ndarray,
    rng: np.random.Generator,
    n_per_seed: int = LOCAL_PERTURB_PER_SEED,
) -> List[ScoredAction]:
    """对检索池中 Q 值最高的若干种子动作做局部扰动。"""
    out: List[ScoredAction] = []
    for local_i in seed_top_q_indices:
        base = train_row_to_radar_series(train_df, int(retrieval_rows[int(local_i)]))
        for _ in range(n_per_seed):
            pert = perturb_radar_series(base, rng)
            out.append(
                ScoredAction(
                    a_std=row_to_action_std(pert, artifacts),
                    radar_row=pert,
                    train_row_idx=None,
                )
            )
    return out


def build_generated_radar_actions(
    artifacts: dict,
    n: int,
    rng: np.random.Generator,
) -> List[ScoredAction]:
    out: List[ScoredAction] = []
    for _ in range(n):
        radar = radar_dict_to_series(sample_radar_action())
        out.append(
            ScoredAction(
                a_std=row_to_action_std(radar, artifacts),
                radar_row=radar,
                train_row_idx=None,
            )
        )
    return out


def sinr_for_action(jammer_row: pd.Series, action: ScoredAction) -> float:
    return sinr_from_csv_row(jammer_row, action.radar_row)


def stack_a_std(actions: List[ScoredAction]) -> np.ndarray:
    if not actions:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([a.a_std for a in actions], axis=0).astype(np.float32)
