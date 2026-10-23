# -*- coding: utf-8 -*-
"""信号级 SINR 后端：用 RadarSimPy 为留出评估/重标注提供信号级真值。

设计要点（与两轮验证结论一致）：
  * 干扰功率为场景级固定绝对功率，仅由干扰参数决定（j_gain），
    与雷达动作解耦——物理上干扰机发射功率固定；
    标定基准 P0 为参考动作（30dB 增益 / 10MHz BW / 30us PW / LFM）的回波功率：
        jam_power = P0 * 10^((j_gain - REF_GAIN + JSR_DB) / 10)
  * 真值取综合有效 SINR = 目标峰 / (底噪中位数 + 最大杂散峰)，
    同时覆盖压制通道与欺骗通道（幻峰）。
  * 帧编码（相位编码）波形无法用 FM 形式建模 → 回退解析公式并计数。
  * 结果按 (干扰参数, 雷达参数) 缓存，跨方法/跨行共享。
  * 随机种子用 zlib.crc32 稳定哈希，跨进程/跨运行可复现。
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# 兼容两种存放位置：src/radar/infer/（原位）或 outputs/<round>/code/（本轮目录）
_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        _src = _p / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        break

from radar.data.high_contrast_sim import (  # noqa: E402
    calculate_sinr_high_contrast,
    ghz_to_hz,
)

C = 3e8
TARGET_RANGE = 1000.0
TARGET_RCS_DBSM = 5.0
SEED = 2026
GUARD = 60
ROLLOFF = 0.15
FT_SPACING_S = 2e-6
REF_GAIN = 30.0

_SKIP_FAMILIES = ("相位编码",)
_warned_families: set[str] = set()


def radar_family(r_type: str) -> str:
    for fam in ("线性调频", "非线性调频", "频率编码", "相位编码"):
        if str(r_type).startswith(fam):
            return fam
    return "线性调频"


def _f_rel_lfm(t: np.ndarray, bw: float) -> np.ndarray:
    return -bw / 2 + bw * t / t[-1]


def _f_rel_nlfm(t: np.ndarray, bw: float) -> np.ndarray:
    u = t / t[-1]
    return -bw / 2 + bw * (u - np.sin(2 * np.pi * u) / (2 * np.pi))


def _f_rel_freqcoded(t: np.ndarray, bw: float, seed: int, hops: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    levels = -bw / 2 + bw * np.arange(hops) / (hops - 1)
    order = rng.permutation(hops)
    idx = np.clip(np.floor(t / t[-1] * hops).astype(int), 0, hops - 1)
    return levels[order[idx]]


def _make_ref(fam: str, bw: float, fs: float, n: int, seed: int) -> np.ndarray:
    t_grid = np.arange(n) / fs
    if fam == "线性调频":
        f_rel = _f_rel_lfm(t_grid, bw)
    elif fam == "非线性调频":
        f_rel = _f_rel_nlfm(t_grid, bw)
    else:
        f_rel = _f_rel_freqcoded(t_grid, bw, seed=seed)
    return np.exp(1j * 2 * np.pi * np.cumsum(f_rel) / fs)


def _frontend_filter(x_fine: np.ndarray, fs_fine: float, fs: float,
                     n: int, bw: float) -> np.ndarray:
    """信道滤波：通带 ±BW/2，15% BW 滚降，4x 过采样降采样。"""
    spec = np.fft.fft(x_fine)
    freqs = np.fft.fftfreq(len(x_fine), 1 / fs_fine)
    af = np.abs(freqs)
    edge, trans = bw / 2, ROLLOFF * bw
    rise = np.clip((af - edge) / trans, 0, 1)
    denom = np.clip(1.0 - rise, 1e-9, None)
    gain = 1.0 / (1.0 + (rise / denom) ** 4)
    gain[af > edge + trans] = 0.0
    spec *= gain
    return np.fft.ifft(spec)[: n * 4 : 4]


def _synth_jamming(j: dict, fam: str, bw: float, fs: float, n: int) -> np.ndarray | None:
    """按干扰参数合成基带干扰（单位功率）；完全出带返回 None。"""
    fs_fine = 4 * fs
    n_fine = n * 4
    t_fine = np.arange(n_fine) / fs_fine
    df = j["jf"] - j["rf"]
    jbw, jtype, pw = j["jbw"], j["jtype"], j["rpw"] * 1e-9
    if not jbw > 0 or not pw > 0:
        return None

    if jtype == "jammer_RANDOM":
        if j["jss"] > 0:
            jtype = "jammer_swept"
        elif j["jdur"] > 0:
            jtype = "jammer_InterruptedSampling"
        elif j["jft"] > 0:
            jtype = "jammer_multi_false_targets"
        else:
            jtype = "jammer_wide_band" if jbw >= 100e6 else "jammer_narrow_band"
    if jtype not in (
        "jammer_narrow_band", "jammer_wide_band", "jammer_swept", "jammer_close",
        "jammer_InterruptedSampling", "jammer_multi_false_targets",
    ):
        jtype = "jammer_wide_band"

    if jtype in ("jammer_narrow_band", "jammer_wide_band"):
        noise = (np.random.randn(n_fine) + 1j * np.random.randn(n_fine)) / np.sqrt(2)
        spec = np.fft.fft(noise)
        freqs = np.fft.fftfreq(n_fine, 1 / fs_fine)
        sigma = max(jbw / 4, 1e3)
        spec *= np.exp(-((freqs - df) ** 2) / (2 * sigma**2))
        return _frontend_filter(np.fft.ifft(spec), fs_fine, fs, n, bw)

    if jtype == "jammer_swept":
        step = min(j["jss"], jbw) if j["jss"] > 0 else jbw / 8
        dwell = pw * step / jbw
        f_j = df - jbw / 2 + (np.floor(t_fine / dwell) * step) % jbw
        phase = 2 * np.pi * np.cumsum(f_j) / fs_fine
        return _frontend_filter(np.exp(1j * phase), fs_fine, fs, n, bw)

    # 转发类：近距压制 / ISRJ / 假目标 —— 雷达波形延迟副本 + 频移
    if fam == "线性调频":
        f_rel = _f_rel_lfm(t_fine, bw)
    elif fam == "非线性调频":
        f_rel = _f_rel_nlfm(t_fine, bw)
    else:
        f_rel = _f_rel_freqcoded(t_fine, bw, seed=int(j["sid"]))
    chirp = np.exp(1j * 2 * np.pi * np.cumsum(f_rel) / fs_fine)
    rng = np.random.default_rng(int(j["sid"]) + 1)

    x = np.zeros(n_fine, dtype=complex)
    if jtype == "jammer_close":
        # 近距压制：DRFM 相干转发，小延迟强副本（重新辐射雷达自身频率，频偏影响弱）
        d = 0.3e-6 + np.random.uniform(0.0, 2.4e-6)
        idx = t_fine - d
        valid = (idx >= 0) & (idx < pw)
        x[valid] = chirp[np.round(idx[valid] * fs_fine).astype(int)]
        return _frontend_filter(x, fs_fine, fs, n, bw)
    if jtype == "jammer_InterruptedSampling":
        dur, delay = j["jdur"] * 1e-9, j["jdelay"] * 1e-9
        intv = j["jintv"] * 1e-9 if j["jintv"] > 0 else dur + delay + 200e-9
        k = 0
        while k * intv < pw:
            d = k * intv + delay
            idx = t_fine - d
            valid = (idx >= 0) & (idx < pw)
            x[valid] += chirp[np.round(idx[valid] * fs_fine).astype(int)]
            k += 1
    else:  # jammer_multi_false_targets
        for k in range(1, int(j["jft"]) + 1):
            idx = t_fine - (3e-6 + k * FT_SPACING_S)
            valid = (idx >= 0) & (idx < pw)
            if valid.any():
                x[valid] += np.exp(1j * rng.uniform(0, 2 * np.pi)) * chirp[
                    np.round(idx[valid] * fs_fine).astype(int)
                ]
    x *= np.exp(1j * 2 * np.pi * df * t_fine)
    return _frontend_filter(x, fs_fine, fs, n, bw)


def _matched_filter(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n = len(x)
    n_fft = 4 * n
    y = np.fft.ifft(np.fft.fft(x, n_fft) * np.conj(np.fft.fft(ref, n_fft)))
    return y[:n]


def _damage(y: np.ndarray, delay_bin: int) -> float:
    lo, hi = max(0, delay_bin - GUARD), delay_bin + GUARD
    p_peak = float((np.abs(y[lo:hi]) ** 2).max())
    peak_idx = lo + int((np.abs(y[lo:hi]) ** 2).argmax())
    mask = np.ones(len(y), dtype=bool)
    mask[max(0, peak_idx - GUARD) : peak_idx + GUARD] = False
    p_others = np.abs(y[mask]) ** 2
    floor = float(np.median(p_others))
    spur = float(p_others.max()) if len(p_others) else 0.0
    return 10 * np.log10(max(p_peak / max(floor + spur, 1e-30), 1e-30))


def _num(row: pd.Series, key: str, scale: float = 1.0) -> float:
    """安全取数：缺失/非数值/NaN 一律按 0 处理（未启用的类型参数在 CSV 中为 NaN）。"""
    try:
        v = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return v * scale if np.isfinite(v) else 0.0


def _radar_fields(radar_row: pd.Series) -> dict:
    return {
        "rf": _num(radar_row, "Radar_CarrierFreq_GHz", 1e9),
        "rg": _num(radar_row, "Radar_Gain_dB"),
        "rbw": _num(radar_row, "Radar_Bandwidth_GHz", 1e9),
        "rpw": _num(radar_row, "Radar_PulseWidth_ns"),
        "rtype": str(radar_row["Radar_Type"]),
        "sid": 0,
    }


def _jammer_fields(jammer_row: pd.Series, sid: int = 0) -> dict:
    return {
        "jtype": str(jammer_row["Jammer_Type"]),
        "jf": _num(jammer_row, "Jammer_CarrierFreq_GHz", 1e9),
        "jg": _num(jammer_row, "Jammer_Gain_dB"),
        "jbw": _num(jammer_row, "Jammer_Bandwidth_GHz", 1e9),
        "jss": _num(jammer_row, "Jammer_Sweep_Step_GHz", 1e9),
        "jdur": _num(jammer_row, "Jammer_IntSample_Dur_ns"),
        "jdelay": _num(jammer_row, "Jammer_IntSample_Delay_ns"),
        "jintv": _num(jammer_row, "Jammer_IntSample_Intv_ns"),
        "jft": _num(jammer_row, "Jammer_FalseTarget_Num"),
        "rpw": _num(jammer_row, "Radar_PulseWidth_ns"),
        "rf": _num(jammer_row, "Radar_CarrierFreq_GHz", 1e9),
        "sid": sid,
    }


class SignalSinrBackend:
    """sinr(jammer_row, radar_row) -> 信号级综合 SINR (dB)，带缓存。"""

    def __init__(self, jsr_db: float = 10.0):
        from radarsimpy import Radar, Receiver, Transmitter, sim_radar

        self.jsr_db = float(jsr_db)
        self._sim = sim_radar
        self._Radar = Radar
        self._Receiver = Receiver
        self._Transmitter = Transmitter
        self._cache: dict[tuple, float] = {}
        self.n_fallback_analytic = 0
        self.n_sim = 0

        # 参考动作回波功率 P0（一次仿真）
        ref_row = pd.Series({
            "Radar_CarrierFreq_GHz": 4.0, "Radar_Gain_dB": REF_GAIN,
            "Radar_Bandwidth_GHz": 0.01, "Radar_PulseWidth_ns": 30000,
            "Radar_Type": "线性调频0", "Radar_ChirpType": "UpChirp",
        })
        self._p0 = self._simulate_echo_power(_radar_fields(ref_row))
        print(f"[SignalSinrBackend] JSR={self.jsr_db}dB, 参考回波功率 P0={self._p0:.3e}")

    def _simulate_echo_power(self, r: dict) -> float:
        fam = radar_family(r["rtype"])
        bw, pw = r["rbw"], r["rpw"] * 1e-9
        fs = 2 * bw
        radar = self._build_radar(r, fam, fs, pw)
        data = self._sim(
            radar,
            [{"location": (TARGET_RANGE, 0, 0), "rcs": TARGET_RCS_DBSM,
              "speed": (0, 0, 0), "phase": 0}],
            density=1, device="cpu",
        )
        bb = np.asarray(data["baseband"])[0, 0]
        n = min(int(np.ceil(pw * fs)), len(bb))
        delay_bin = int(round(2 * TARGET_RANGE / C * fs))
        win = bb[:n][max(0, delay_bin - 50): delay_bin + 50]
        return float(np.mean(np.abs(win) ** 2))

    def _build_radar(self, r: dict, fam: str, fs: float, pw: float):
        fc = r["rf"]
        if fam == "线性调频":
            tx_f, tx_t = [fc - r["rbw"] / 2, fc + r["rbw"] / 2], None
        else:
            t_tx = np.linspace(0, pw, 2048)
            if fam == "非线性调频":
                tx_f, tx_t = fc + _f_rel_nlfm(t_tx, r["rbw"]), t_tx
            else:
                tx_f, tx_t = fc + _f_rel_freqcoded(t_tx, r["rbw"], seed=int(r["sid"])), t_tx
        tx = (self._Transmitter(f=tx_f, t=pw, tx_power=r["rg"], pulses=1)
              if tx_t is None else self._Transmitter(f=tx_f, t=tx_t, tx_power=r["rg"], pulses=1))
        rx = self._Receiver(fs=fs, noise_figure=8, rf_gain=30, baseband_gain=40)
        return self._Radar(transmitter=tx, receiver=rx, seed=SEED)

    @staticmethod
    def _key(j: dict, r: dict) -> tuple:
        return (
            j["jtype"], round(j["jf"], 3), round(j["jg"], 1), round(j["jbw"], 6),
            round(j["jss"], 6), round(j["jdur"], 1), round(j["jdelay"], 1),
            round(j["jintv"], 1), int(j["jft"]), int(j["sid"]),
            round(r["rf"], 3), round(r["rg"], 1), round(r["rbw"], 6),
            int(r["rpw"]), r["rtype"],
        )

    def sinr(self, jammer_row: pd.Series, radar_row: pd.Series,
             add_noise: bool = True, **_) -> float:
        """与 sinr_from_csv_row 同签名（add_noise 忽略，真值确定性）。"""
        fam = radar_family(str(radar_row["Radar_Type"]))
        if fam in _SKIP_FAMILIES:
            self.n_fallback_analytic += 1
            if fam not in _warned_families:
                _warned_families.add(fam)
                print(f"[SignalSinrBackend] {fam} 波形无法建模，回退解析公式")
            return self._analytic(jammer_row, radar_row)

        j = _jammer_fields(jammer_row)
        r = _radar_fields(radar_row)
        key = self._key(j, r)
        if key in self._cache:
            return self._cache[key]

        bw, pw = r["rbw"], r["rpw"] * 1e-9
        fs = 2 * bw
        radar = self._build_radar(r, fam, fs, pw)
        data = self._sim(
            radar,
            [{"location": (TARGET_RANGE, 0, 0), "rcs": TARGET_RCS_DBSM,
              "speed": (0, 0, 0), "phase": 0}],
            density=1, device="cpu",
        )
        bb_raw = np.asarray(data["baseband"])[0, 0]
        n = min(int(np.ceil(pw * fs)), len(bb_raw))
        bb = bb_raw[:n]
        ref = _make_ref(fam, bw, fs, n, seed=int(j["sid"]))
        delay_bin = int(round(2 * TARGET_RANGE / C * fs))

        # 稳定种子：zlib.crc32 跨进程/跨运行一致（Python 内置 hash 有随机盐）
        np.random.seed(SEED + (zlib.crc32(repr(key).encode("utf-8")) % (2**31)))
        jam_unit = _synth_jamming(j, fam, bw, fs, n)
        if jam_unit is not None and not np.all(np.isfinite(jam_unit)):
            # 合成结果非有限 → 不产出误导性分数，回退解析公式
            self.n_fallback_analytic += 1
            print(f"[SignalSinrBackend] 干扰合成异常，回退解析: {j['jtype']}")
            return self._analytic(jammer_row, radar_row)
        if jam_unit is None or np.abs(jam_unit).max() == 0:
            # 无带内有效干扰（完全出带或滤波后为零）
            val = _damage(self._matched(bb, ref), delay_bin)
        else:
            p_j = float(np.mean(np.abs(jam_unit) ** 2))
            if not np.isfinite(p_j) or p_j <= 0.0:
                # 残留功率下溢（强失配 + 滤波）→ 等效无干扰
                val = _damage(self._matched(bb, ref), delay_bin)
            else:
                jam_power = self._p0 * 10 ** ((j["jg"] - REF_GAIN + self.jsr_db) / 10)
                jam = jam_unit / np.sqrt(p_j) * np.sqrt(jam_power)
                val = _damage(self._matched(bb + jam, ref), delay_bin)
        if not np.isfinite(val):
            # 兜底：任何路径产出的非有限值一律回退解析，严禁污染缓存
            self.n_fallback_analytic += 1
            return self._analytic(jammer_row, radar_row)

        self._cache[key] = val
        self.n_sim += 1
        return val

    @staticmethod
    def _matched(bb: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return _matched_filter(bb, ref)

    @staticmethod
    def _analytic(jammer_row: pd.Series, radar_row: pd.Series) -> float:
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
            j_sweep_step_hz=ghz_to_hz(_num(jammer_row, "Jammer_Sweep_Step_GHz")),
            j_int_dur_ns=_num(jammer_row, "Jammer_IntSample_Dur_ns"),
            j_int_delay_ns=_num(jammer_row, "Jammer_IntSample_Delay_ns"),
            j_int_intv_ns=_num(jammer_row, "Jammer_IntSample_Intv_ns"),
            j_ft_num=_num(jammer_row, "Jammer_FalseTarget_Num"),
            add_noise=False,
        )

    def stats(self) -> str:
        return (f"仿真次数={self.n_sim}, 缓存命中={len(self._cache)}, "
                f"相位编码回退={self.n_fallback_analytic}")
