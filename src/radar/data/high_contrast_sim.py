# -*- coding: utf-8 -*-
"""雷达/干扰仿真（连续 SINR + 雷达频率对齐采样，任务难度更高）。"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# 雷达频率以该概率采在干扰频段附近（提高频谱重叠、放大动作间 SINR 差异）
RADAR_JAMMER_COLOCATE_PROB = 0.5

RADAR_TYPES: Tuple[str, ...] = (
    "线性调频0",
    "非线性调频1",
    "相位编码3",
    "频率编码2",
)

JAMMER_TYPES: Tuple[str, ...] = (
    "jammer_swept",
    "jammer_multi_false_targets",
    "jammer_InterruptedSampling",
    "jammer_narrow_band",
    "jammer_wide_band",
    "jammer_close",
    "jammer_RANDOM",
)

RADAR_CONFIG = {
    "freq_min": 3.0e9,
    "freq_max": 5.0e9,
    "gain_min": 20,
    "gain_max": 40,
    "bw_min": 5e6,
    "bw_max": 25e6,
    "pw_min": 20000,
    "pw_max": 40000,
    "radar_types": list(RADAR_TYPES),
    "chirp_type": "NegativeTriangle",
}

JAMMER_CONFIG = {
    "types": list(JAMMER_TYPES),
    "freq_min": 3.0e9,
    "freq_max": 5.0e9,
    "gain_min": 20,
    "gain_max": 40,
}

# Radar_Type × Jammer_Type 交互系数 (dB)
# 物理直觉：频率编码抗扫频、相位编码抗中断采样、线性调频通用但近距弱等
RADAR_JAMMER_COEFF: Dict[Tuple[str, str], float] = {
    ("线性调频0", "jammer_swept"): -0.8,
    ("线性调频0", "jammer_multi_false_targets"): 0.6,
    ("线性调频0", "jammer_InterruptedSampling"): 1.2,
    ("线性调频0", "jammer_narrow_band"): 1.0,
    ("线性调频0", "jammer_wide_band"): -0.6,
    ("线性调频0", "jammer_close"): -1.2,
    ("线性调频0", "jammer_RANDOM"): 0.0,
    ("非线性调频1", "jammer_swept"): -0.3,
    ("非线性调频1", "jammer_multi_false_targets"): 1.0,
    ("非线性调频1", "jammer_InterruptedSampling"): 1.8,
    ("非线性调频1", "jammer_narrow_band"): 0.6,
    ("非线性调频1", "jammer_wide_band"): -0.2,
    ("非线性调频1", "jammer_close"): -0.8,
    ("非线性调频1", "jammer_RANDOM"): 0.2,
    ("相位编码3", "jammer_swept"): -1.8,
    ("相位编码3", "jammer_multi_false_targets"): -0.8,
    ("相位编码3", "jammer_InterruptedSampling"): 3.2,
    ("相位编码3", "jammer_narrow_band"): 0.2,
    ("相位编码3", "jammer_wide_band"): -1.2,
    ("相位编码3", "jammer_close"): -2.2,
    ("相位编码3", "jammer_RANDOM"): -0.3,
    ("频率编码2", "jammer_swept"): 2.8,
    ("频率编码2", "jammer_multi_false_targets"): 0.4,
    ("频率编码2", "jammer_InterruptedSampling"): 0.6,
    ("频率编码2", "jammer_narrow_band"): 1.6,
    ("频率编码2", "jammer_wide_band"): 0.4,
    ("频率编码2", "jammer_close"): -1.6,
    ("频率编码2", "jammer_RANDOM"): 0.5,
}

PULSE_WIDTH_REF_NS = 30000.0

# 训练集 scenario_id 范围 [0, num_states-1]，留出 ID 从此起始
HOLDOUT_SCENARIO_ID_START = 1_000_000


def get_random_100_aligned(min_val: float, max_val: float) -> int:
    return random.randint(int(min_val // 100), int(max_val // 100)) * 100


def hz_to_ghz(hz_value: float) -> float:
    return round(hz_value / 1e9, 3) if hz_value > 0 else 0.0


def ghz_to_hz(ghz: float) -> float:
    return float(ghz) * 1e9


def random_freq_hz(min_f: float, max_f: float) -> float:
    steps = int((max_f - min_f) / 10e6)
    return min_f + random.randint(0, steps) * 10e6


def _overlap_factor(r_freq: float, r_bw: float, j_freq: float, j_bw: float) -> float:
    real_j_bw = j_bw if j_bw > 0 else 10e6
    freq_dist = abs(r_freq - j_freq)
    collision_radius = max((r_bw / 2) + (real_j_bw / 2), 1e6)
    return float(max(0.0, 1.0 - freq_dist / collision_radius))


def _pulse_compression_gain(r_pw: float, r_type: str) -> float:
    """脉宽影响脉压增益；调频类雷达受益更大。"""
    ratio = max(r_pw, 1.0) / PULSE_WIDTH_REF_NS
    base = 1.2 * float(np.log10(ratio))
    if r_type in ("线性调频0", "非线性调频1"):
        base *= 1.35
    elif r_type == "相位编码3":
        base *= 1.15
    elif r_type == "频率编码2":
        base *= 1.05
    return base


@dataclass(frozen=True)
class _SinrInputs:
    r_freq: float
    r_bw: float
    r_gain: float
    r_type: str
    r_pw: float
    j_freq: float
    j_bw: float
    j_gain: float
    j_type: str
    j_sweep_step_hz: float
    j_int_dur_ns: float
    j_int_delay_ns: float
    j_int_intv_ns: float
    j_ft_num: float


def _baseline_clear_sinr(inp: _SinrInputs) -> float:
    bw_term = 2.0 * np.log10(max(inp.r_bw, 1e6) / 1e6)
    pw_term = _pulse_compression_gain(inp.r_pw, inp.r_type)
    clear = 28.0 + 0.30 * (inp.r_gain - inp.j_gain) + bw_term + pw_term
    return float(np.clip(clear, 22.0, 38.0))


def _overlap(inp: _SinrInputs) -> float:
    return _overlap_factor(inp.r_freq, inp.r_bw, inp.j_freq, inp.j_bw)


def _freq_sep_hz(inp: _SinrInputs) -> float:
    return abs(inp.r_freq - inp.j_freq)


def _spectral_tail(inp: _SinrInputs, scale_hz: float) -> float:
    """频谱旁瓣/距离耦合：即使未完全重叠，窄谱干扰仍可能落入接收机。"""
    sep = _freq_sep_hz(inp)
    denom = max(scale_hz, inp.j_bw, inp.r_bw, 1e6)
    return float(np.exp(-0.5 * (sep / denom) ** 2))


def _type_interaction_bonus(inp: _SinrInputs, coupling: float) -> float:
    coeff = RADAR_JAMMER_COEFF.get((inp.r_type, inp.j_type), 0.0)
    return (0.20 + 0.80 * coupling) * coeff


def _resolve_random_subtype(inp: _SinrInputs) -> str:
    if inp.j_sweep_step_hz > 0:
        return "jammer_swept"
    if inp.j_int_dur_ns > 0 and inp.j_int_intv_ns > 0:
        return "jammer_InterruptedSampling"
    if inp.j_ft_num > 0:
        return "jammer_multi_false_targets"
    if inp.j_bw >= 100e6:
        return "jammer_wide_band"
    if inp.j_bw > 0:
        return "jammer_narrow_band"
    return "jammer_wide_band"


def _sinr_narrow_band(inp: _SinrInputs) -> float:
    """窄带噪声：谱密度高，频率对准 + 旁瓣落入均会压制。"""
    overlap = _overlap(inp)
    tail = _spectral_tail(inp, inp.j_bw * 0.6)
    coupling = float(np.clip(max(overlap, 0.55 * tail), 0.0, 1.0))
    j_bw_mhz = max(inp.j_bw, 1e6) / 1e6
    psd_penalty = 6.0 + 4.5 * float(np.log10(30.0 / j_bw_mhz))
    gain_penalty = 0.42 * inp.j_gain - 0.14 * inp.r_gain
    penalty = coupling * (psd_penalty + gain_penalty)
    return _baseline_clear_sinr(inp) - penalty + _type_interaction_bonus(inp, coupling)


def _sinr_wide_band(inp: _SinrInputs) -> float:
    """宽带阻塞：覆盖带宽比例决定压制强度。"""
    overlap = _overlap(inp)
    real_j_bw = max(inp.j_bw, 10e6)
    coverage = float(np.clip(min(inp.r_bw, real_j_bw) / max(inp.r_bw, 1e6), 0.0, 1.0))
    coupling = float(np.clip(0.35 * overlap + 0.65 * coverage, 0.0, 1.0))
    barrage = 7.5 + 0.35 * inp.j_gain - 0.10 * inp.r_gain
    penalty = coupling * barrage + 0.25 * coverage * (inp.j_gain - 20.0)
    return _baseline_clear_sinr(inp) - penalty + _type_interaction_bonus(inp, coupling)


def _sinr_swept(inp: _SinrInputs) -> float:
    """扫频干扰：扫频步进决定驻留时间，不完全依赖静态重叠。"""
    overlap = _overlap(inp)
    step_hz = max(inp.j_sweep_step_hz, 2e6)
    dwell = float(np.clip(inp.r_bw / step_hz, 0.05, 1.0))
    sweep_rate_penalty = min(6.0, 24.0 / max(step_hz / 1e6, 2.0))
    coupling = float(np.clip(0.40 * overlap + 0.60 * dwell, 0.0, 1.0))
    penalty = (
        3.5
        + 7.0 * dwell
        + 2.5 * overlap
        + sweep_rate_penalty
        + 0.28 * inp.j_gain
        - 0.10 * inp.r_gain
    )
    return _baseline_clear_sinr(inp) - penalty + _type_interaction_bonus(inp, coupling)


def _sinr_close(inp: _SinrInputs) -> float:
    """近距离压制：功率 + 相干转发，频偏影响弱于其它类型。"""
    overlap = _overlap(inp)
    tail = _spectral_tail(inp, max(inp.j_bw, 40e6))
    proximity = float(np.clip(0.55 + 0.25 * tail + 0.20 * overlap, 0.0, 1.0))
    penalty = proximity * (11.0 + 0.55 * inp.j_gain - 0.08 * inp.r_gain)
    return _baseline_clear_sinr(inp) - penalty + _type_interaction_bonus(inp, proximity)


def _sinr_interrupted_sampling(inp: _SinrInputs) -> float:
    """中断采样：时序占空比 + 脉宽对齐，频谱重叠仅作辅助项。"""
    overlap = _overlap(inp)
    if inp.j_int_dur_ns <= 0 or inp.j_int_intv_ns <= 0:
        duty = 0.0
        sync = 0.0
    else:
        duty = float(np.clip(inp.j_int_dur_ns / inp.j_int_intv_ns, 0.0, 1.0))
        sync = float(np.exp(-abs(inp.r_pw - inp.j_int_dur_ns) / 5000.0))
    delay_boost = 1.0 + 0.7 * (5000.0 / max(inp.j_int_delay_ns, 500.0))
    timing = float(np.clip(0.25 + 0.45 * duty + 0.30 * sync, 0.0, 1.0))
    coupling = float(np.clip(0.65 * timing + 0.35 * overlap, 0.0, 1.0))
    penalty = (
        4.0
        + 8.5 * timing * min(delay_boost, 2.2)
        + 1.5 * overlap
        + 0.22 * inp.j_gain
    )
    return _baseline_clear_sinr(inp) - penalty + _type_interaction_bonus(inp, coupling)


def _sinr_false_targets(inp: _SinrInputs) -> float:
    """假目标：欺骗式干扰，主要受假目标数量与雷达波形影响。"""
    overlap = _overlap(inp)
    ft_num = max(inp.j_ft_num, 1.0) if inp.j_ft_num > 0 else 1.0
    deception = float(np.clip(0.35 + 0.08 * ft_num, 0.0, 1.0))
    coupling = float(np.clip(0.70 * deception + 0.30 * overlap, 0.0, 1.0))
    penalty = 5.0 + 6.5 * deception + 1.2 * overlap + 0.15 * inp.j_gain
    return _baseline_clear_sinr(inp) - penalty + _type_interaction_bonus(inp, coupling)


def _sinr_random(inp: _SinrInputs) -> float:
    """随机干扰：按激活的子机制分派到对应类型公式。"""
    subtype = _resolve_random_subtype(inp)
    calc = _SINR_CALCULATORS.get(subtype, _sinr_wide_band)
    base = calc(inp)
    jitter = 0.8 * _overlap(inp)
    return base - jitter


_SINR_CALCULATORS: Dict[str, Callable[[_SinrInputs], float]] = {
    "jammer_narrow_band": _sinr_narrow_band,
    "jammer_wide_band": _sinr_wide_band,
    "jammer_swept": _sinr_swept,
    "jammer_close": _sinr_close,
    "jammer_InterruptedSampling": _sinr_interrupted_sampling,
    "jammer_multi_false_targets": _sinr_false_targets,
    "jammer_RANDOM": _sinr_random,
}


def calculate_sinr_high_contrast(
    r_freq: float,
    r_bw: float,
    r_gain: float,
    j_freq: float,
    j_bw: float,
    j_gain: float,
    j_type: str,
    *,
    r_type: str = "线性调频0",
    r_pw: float = PULSE_WIDTH_REF_NS,
    j_sweep_step_hz: float = 0.0,
    j_int_dur_ns: float = 0.0,
    j_int_delay_ns: float = 0.0,
    j_int_intv_ns: float = 0.0,
    j_ft_num: float = 0.0,
    add_noise: bool = True,
) -> float:
    """
    分干扰类型的 SINR 模型：
    - 每种 Jammer_Type 使用独立公式（见 _SINR_CALCULATORS）
    - 重叠仅作为耦合因子之一，而非唯一主导项
    - 扫频→驻留、窄带→谱密度、假目标→欺骗强度、中断采样→时序对齐
    - Radar_Type × Jammer_Type 交互系数仍保留
    """
    inp = _SinrInputs(
        r_freq=r_freq,
        r_bw=r_bw,
        r_gain=r_gain,
        r_type=r_type,
        r_pw=r_pw,
        j_freq=j_freq,
        j_bw=j_bw,
        j_gain=j_gain,
        j_type=j_type,
        j_sweep_step_hz=j_sweep_step_hz,
        j_int_dur_ns=j_int_dur_ns,
        j_int_delay_ns=j_int_delay_ns,
        j_int_intv_ns=j_int_intv_ns,
        j_ft_num=j_ft_num,
    )
    calc = _SINR_CALCULATORS.get(j_type, _sinr_wide_band)
    final_sinr = float(calc(inp))

    if add_noise:
        final_sinr += float(np.random.normal(0, 0.35))
    return round(final_sinr, 4)


def sinr_from_csv_row(
    jammer_row: pd.Series,
    radar_row: pd.Series,
    *,
    add_noise: bool = True,
) -> float:
    """用 CSV 行中的 GHz/dB 列计算 SINR（与仿真公式一致）。"""
    return calculate_sinr_high_contrast(
        ghz_to_hz(float(radar_row["Radar_CarrierFreq_GHz"])),
        ghz_to_hz(float(radar_row["Radar_Bandwidth_GHz"])),
        float(radar_row["Radar_Gain_dB"]),
        ghz_to_hz(float(jammer_row["Jammer_CarrierFreq_GHz"])),
        ghz_to_hz(float(jammer_row["Jammer_Bandwidth_GHz"])),
        float(jammer_row["Jammer_Gain_dB"]),
        str(jammer_row["Jammer_Type"]),
        r_type=str(radar_row["Radar_Type"]),
        r_pw=float(radar_row["Radar_PulseWidth_ns"]),
        j_sweep_step_hz=ghz_to_hz(float(jammer_row.get("Jammer_Sweep_Step_GHz", 0.0))),
        j_int_dur_ns=float(jammer_row.get("Jammer_IntSample_Dur_ns", 0.0)),
        j_int_delay_ns=float(jammer_row.get("Jammer_IntSample_Delay_ns", 0.0)),
        j_int_intv_ns=float(jammer_row.get("Jammer_IntSample_Intv_ns", 0.0)),
        j_ft_num=float(jammer_row.get("Jammer_FalseTarget_Num", 0.0)),
        add_noise=add_noise,
    )


def _apply_jammer_type_params(
    j_type: str,
    j_params: Dict[str, float],
) -> float:
    """按干扰类型设置带宽与专属参数，返回 j_bw。"""
    if j_type == "jammer_swept":
        j_bw = float(random.choice([200e6, 300e6, 400e6]))
        j_params["Sweep_Step"] = float(random.choice([5e6, 10e6]))
        j_params["Sweep_Power"] = random.randint(20, 50)
    elif j_type == "jammer_wide_band":
        j_bw = float(random.uniform(200e6, 500e6))
    elif j_type == "jammer_narrow_band":
        j_bw = float(random.uniform(10e6, 30e6))
    elif j_type == "jammer_InterruptedSampling":
        j_bw = 20e6
        j_params["IntDur"] = float(get_random_100_aligned(2000, 20000))
        j_params["IntDelay"] = float(get_random_100_aligned(1000, 10000))
        j_params["IntIntv"] = j_params["IntDur"] + j_params["IntDelay"] + 200
    elif j_type == "jammer_multi_false_targets":
        j_bw = 20e6
        j_params["FT_Num"] = random.randint(5, 15)
        j_params["FT_Intv"] = random.randint(1000, 3000)
        j_params["FT_Delay"] = 33000
    elif j_type == "jammer_close":
        j_bw = 60e6
    else:
        j_bw = 0.0
    return j_bw


def _sample_jammer_random(j_params: Dict[str, float]) -> float:
    """jammer_RANDOM：随机带宽 + 随机激活一种子机制。"""
    j_bw = float(random.uniform(10e6, 200e6))
    mechanism = random.choice(["sweep", "int_sample", "false_target", "plain"])
    if mechanism == "sweep":
        j_params["Sweep_Step"] = float(random.choice([5e6, 10e6, 20e6]))
    elif mechanism == "int_sample":
        j_params["IntDur"] = float(get_random_100_aligned(2000, 20000))
        j_params["IntDelay"] = float(get_random_100_aligned(1000, 10000))
        j_params["IntIntv"] = j_params["IntDur"] + j_params["IntDelay"] + 200
    elif mechanism == "false_target":
        j_params["FT_Num"] = float(random.randint(3, 12))
    return j_bw


def sample_jammer_state() -> Dict[str, Any]:
    j_type = random.choice(JAMMER_CONFIG["types"])
    j_freq = random_freq_hz(JAMMER_CONFIG["freq_min"], JAMMER_CONFIG["freq_max"])
    j_gain = random.randint(JAMMER_CONFIG["gain_min"], JAMMER_CONFIG["gain_max"])
    j_params = {
        k: 0.0
        for k in [
            "Sweep_Step",
            "Sweep_Power",
            "IntDur",
            "IntDelay",
            "IntIntv",
            "FT_Num",
            "FT_Intv",
            "FT_Delay",
        ]
    }

    if j_type == "jammer_RANDOM":
        j_bw = _sample_jammer_random(j_params)
    else:
        j_bw = _apply_jammer_type_params(j_type, j_params)

    if j_type == "jammer_close":
        j_gain = 40

    return {
        "j_type": j_type,
        "j_freq": j_freq,
        "j_bw": j_bw,
        "j_gain": j_gain,
        "j_params": j_params,
    }


def sample_radar_action(jammer: Dict[str, Any] | None = None) -> Dict[str, Any]:
    r_bw = float(int(random.uniform(RADAR_CONFIG["bw_min"], RADAR_CONFIG["bw_max"])))
    r_gain = random.randint(RADAR_CONFIG["gain_min"], RADAR_CONFIG["gain_max"])

    if jammer is not None and random.random() < RADAR_JAMMER_COLOCATE_PROB:
        j_freq = jammer["j_freq"]
        j_bw = jammer["j_bw"] if jammer["j_bw"] > 0 else 10e6
        collision_radius = (r_bw / 2) + (j_bw / 2)
        offset = random.uniform(-collision_radius * 1.5, collision_radius * 1.5)
        r_freq = float(np.clip(
            j_freq + offset,
            RADAR_CONFIG["freq_min"],
            RADAR_CONFIG["freq_max"],
        ))
    else:
        r_freq = random_freq_hz(RADAR_CONFIG["freq_min"], RADAR_CONFIG["freq_max"])

    return {
        "r_freq": r_freq,
        "r_bw": r_bw,
        "r_pw": get_random_100_aligned(RADAR_CONFIG["pw_min"], RADAR_CONFIG["pw_max"]),
        "r_gain": r_gain,
        "r_type": random.choice(RADAR_CONFIG["radar_types"]),
        "r_chirp": RADAR_CONFIG["chirp_type"],
    }


def build_row(scenario_id: int, action_idx: int, jammer: Dict[str, Any], radar: Dict[str, Any]) -> dict:
    jp = jammer["j_params"]
    sinr = calculate_sinr_high_contrast(
        radar["r_freq"],
        radar["r_bw"],
        radar["r_gain"],
        jammer["j_freq"],
        jammer["j_bw"],
        jammer["j_gain"],
        jammer["j_type"],
        r_type=radar["r_type"],
        r_pw=float(radar["r_pw"]),
        j_sweep_step_hz=float(jp["Sweep_Step"]),
        j_int_dur_ns=float(jp["IntDur"]),
        j_int_delay_ns=float(jp["IntDelay"]),
        j_int_intv_ns=float(jp["IntIntv"]),
        j_ft_num=float(jp["FT_Num"]),
    )
    return {
        "scenario_id": scenario_id,
        "action_idx": action_idx,
        "Jammer_Type": jammer["j_type"],
        "Jammer_CarrierFreq_GHz": hz_to_ghz(jammer["j_freq"]),
        "Jammer_Gain_dB": jammer["j_gain"],
        "Jammer_Bandwidth_GHz": hz_to_ghz(jammer["j_bw"]),
        "Radar_CarrierFreq_GHz": hz_to_ghz(radar["r_freq"]),
        "Radar_Gain_dB": radar["r_gain"],
        "Radar_Bandwidth_GHz": hz_to_ghz(radar["r_bw"]),
        "Radar_PulseWidth_ns": radar["r_pw"],
        "Radar_Type": radar["r_type"],
        "SINR_dB": sinr,
        "Jammer_Sweep_Step_GHz": hz_to_ghz(jp["Sweep_Step"]),
        "Jammer_IntSample_Dur_ns": jp["IntDur"],
        "Jammer_IntSample_Delay_ns": jp["IntDelay"],
        "Jammer_IntSample_Intv_ns": jp["IntIntv"],
        "Jammer_FalseTarget_Num": jp["FT_Num"],
        "Radar_ChirpType": radar["r_chirp"],
    }


def generate_scenario_dataset(
    num_states: int,
    actions_per_state: int,
    scenario_id_start: int = 0,
) -> pd.DataFrame:
    rows: List[dict] = []
    for i in range(num_states):
        sid = scenario_id_start + i
        jammer = sample_jammer_state()
        for aid in range(actions_per_state):
            radar = sample_radar_action(jammer)
            rows.append(build_row(sid, aid, jammer, radar))
    return pd.DataFrame(rows)


def generate_holdout_dataset(
    n_scenarios: int = 500,
    actions_per_state: int = 10,
    seed: int = 2026,
) -> pd.DataFrame:
    """生成与训练集不重叠的 scenario_id 的留出数据。"""
    random.seed(seed)
    np.random.seed(seed)
    return generate_scenario_dataset(
        n_scenarios,
        actions_per_state,
        scenario_id_start=HOLDOUT_SCENARIO_ID_START,
    )
