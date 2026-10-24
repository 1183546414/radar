"""Schema and hyper-parameters for the standalone PARD method."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


STATE_NUMERIC = [
    "Jammer_CarrierFreq_GHz", "Jammer_Gain_dB", "Jammer_Bandwidth_GHz",
    "Jammer_Sweep_Step_GHz", "Jammer_IntSample_Dur_ns",
    "Jammer_IntSample_Delay_ns", "Jammer_IntSample_Intv_ns",
    "Jammer_FalseTarget_Num",
]
STATE_CATEGORICAL = ["Jammer_Type"]
ACTION_NUMERIC = [
    "Radar_CarrierFreq_GHz", "Radar_Gain_dB", "Radar_Bandwidth_GHz",
    "Radar_PulseWidth_ns",
]
ACTION_CATEGORICAL = ["Radar_Type", "Radar_ChirpType"]
TARGET = "SINR_dB"
SCENARIO = "scenario_id"

# The search bounds deliberately follow the simulator configuration, not a
# learned decoder.  They can be tightened from a deployment safety envelope.
ACTION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Radar_CarrierFreq_GHz": (3.0, 5.0),
    "Radar_Gain_dB": (20.0, 40.0),
    "Radar_Bandwidth_GHz": (0.005, 0.025),
    "Radar_PulseWidth_ns": (20000.0, 40000.0),
}


@dataclass
class PARDConfig:
    seed: int = 20261024
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.10
    fourier_bands: int = 4
    batch_size: int = 256
    epochs: int = 120
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 18
    quantiles: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    ranking_weight: float = 0.40
    top_weight: float = 0.35
    robust_weight: float = 0.20
    adversarial_radius: float = 0.10
    adversarial_steps: int = 2
    risk_beta: float = 0.20
    cvar_alpha: float = 0.10
    cvar_weight: float = 0.65
    cem_population: int = 512
    cem_elite_fraction: float = 0.10
    cem_iterations: int = 6
    inner_perturbations: int = 12

    def to_dict(self) -> dict:
        return asdict(self)
