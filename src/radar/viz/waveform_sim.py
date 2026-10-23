# -*- coding: utf-8 -*-
"""根据雷达/干扰参数生成受扰波形示意图（基带仿真，用于演示）。"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radar.data.high_contrast_sim import _overlap_factor, ghz_to_hz, sinr_from_csv_row
from radar.infer.demo_recommend import jammer_dict_to_series, radar_series_to_dict

# 演示用固定随机种子，保证同一组参数波形可复现
_WF_SEED = 7
_MAX_SAMPLES = 2048

_FIG_BG = "#ffffff"
_AX_BG = "#ffffff"
_GRID = "#cccccc"
_TEXT = "#222222"
_TICK = "#444444"
_C_RADAR = "#0d9488"
_C_JAM = "#d97706"
_C_RX = "#0369a1"
_C_BAND_R = "#0d948833"
_C_BAND_J = "#d9770633"

_FS_TITLE = 12
_FS_LABEL = 10
_FS_TICK = 9
_FS_LEGEND = 10
_FS_SUBTITLE = 11

_BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64)


def _radar_row_from_dict(radar: Dict[str, Any]) -> pd.Series:
    return pd.Series(
        {
            "Radar_Type": str(radar.get("radar_type", "线性调频0")),
            "Radar_ChirpType": str(radar.get("chirp_type", "NegativeTriangle")),
            "Radar_CarrierFreq_GHz": float(radar["carrier_freq_ghz"]),
            "Radar_Gain_dB": float(radar["gain_db"]),
            "Radar_Bandwidth_GHz": float(radar["bandwidth_ghz"]),
            "Radar_PulseWidth_ns": float(radar.get("pulse_width_ns", 30000)),
        }
    )


def _time_axis_ns(pw_ns: float, n: int) -> Tuple[np.ndarray, float]:
    """返回时间轴（ns）与采样间隔 dt（s）。"""
    t_ns = np.linspace(0.0, pw_ns, n, endpoint=False)
    dt = (pw_ns * 1e-9) / max(n, 1)
    return t_ns, dt


def _radar_baseband(
    t_s: np.ndarray,
    radar_type: str,
    bw_hz: float,
    pw_ns: float,
) -> np.ndarray:
    """生成雷达发射脉冲的基带复包络（归一化功率）。"""
    n = len(t_s)
    T = max(pw_ns * 1e-9, 1e-9)
    t0 = t_s - T / 2.0
    win = np.hanning(n)

    if radar_type == "相位编码3":
        seg = max(n // len(_BARKER13), 1)
        code = np.ones(n)
        for i, c in enumerate(_BARKER13):
            lo, hi = i * seg, min((i + 1) * seg, n)
            code[lo:hi] = c
        phase = np.pi * (code < 0).astype(float)
        s = np.exp(1j * phase) * win
    elif radar_type == "频率编码2":
        nstep = 8
        step_len = max(n // nstep, 1)
        s = np.zeros(n, dtype=np.complex128)
        for k in range(nstep):
            lo, hi = k * step_len, min((k + 1) * step_len, n)
            f_off = (k - nstep / 2) * bw_hz / nstep
            tt = t0[lo:hi]
            s[lo:hi] = np.exp(1j * 2 * np.pi * f_off * tt) * win[lo:hi]
    elif radar_type == "非线性调频1":
        K = bw_hz / T
        nl = 0.35 * np.sin(2 * np.pi * 3 * t0 / T)
        phase = np.pi * K * t0**2 + np.pi * bw_hz * nl * t0
        s = np.exp(1j * phase) * win
    else:
        K = bw_hz / T
        phase = np.pi * K * t0**2
        s = np.exp(1j * phase) * win

    return _normalize_power(s)


def _nyquist_hz(dt: float) -> float:
    return 0.5 / max(dt, 1e-12)


def _viz_jammer_band(
    df_hz: float,
    dt: float,
    r_bw_hz: float,
    j_bw_hz: float,
    overlap: float,
) -> Tuple[float, float]:
    """
    脉内 FFT 可观测带宽有限（~1/脉冲宽度）。
    载频差过大时，将干扰映射为雷达基带内的等效耦合带宽（用于演示可视化）。
    """
    f_nyq = _nyquist_hz(dt)
    if abs(df_hz) <= f_nyq * 0.95:
        return df_hz, j_bw_hz
    eff_bw = max(min(r_bw_hz, j_bw_hz) * max(overlap, 0.08), 1e5)
    return 0.0, eff_bw


def _bandpass_noise(
    rng: np.random.Generator,
    n: int,
    dt: float,
    center_hz: float,
    bw_hz: float,
) -> np.ndarray:
    """频域带限复噪声。"""
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    freqs = np.fft.fftfreq(n, dt)
    f_nyq = _nyquist_hz(dt)
    center_hz = float(np.clip(center_hz, -f_nyq * 0.95, f_nyq * 0.95))
    mask = np.abs(freqs - center_hz) <= max(bw_hz / 2, 1e5)
    spec = np.fft.fft(noise)
    spec[~mask] = 0
    out = np.fft.ifft(spec)
    if float(np.mean(np.abs(out) ** 2)) < 1e-12:
        # 兜底：在基带中心生成窄带噪声，避免干扰图成一条直线
        mask = np.abs(freqs) <= max(bw_hz / 2, 1e5)
        spec = np.fft.fft(noise)
        spec[~mask] = 0
        out = np.fft.ifft(spec)
    return _normalize_power(out)


def _resolve_random_mechanism(jammer: Dict[str, Any]) -> str:
    if float(jammer.get("sweep_step_ghz", 0)) > 0:
        return "swept"
    if float(jammer.get("int_sample_dur_ns", 0)) > 0:
        return "int_sample"
    if float(jammer.get("false_target_num", 0)) > 0:
        return "false_target"
    return "barrage"


def _jammer_baseband(
    t_s: np.ndarray,
    dt: float,
    jammer: Dict[str, Any],
    radar_signal: np.ndarray,
    df_hz: float,
    r_bw_hz: float,
    overlap: float,
) -> np.ndarray:
    """生成干扰基带信号（归一化功率）。"""
    n = len(t_s)
    rng = np.random.default_rng(_WF_SEED)
    j_type = str(jammer["jammer_type"])
    j_bw_hz = max(ghz_to_hz(float(jammer["bandwidth_ghz"])), 1e6)
    T = max(t_s[-1] - t_s[0], 1e-9)
    j_center, j_eff_bw = _viz_jammer_band(df_hz, dt, r_bw_hz, j_bw_hz, overlap)
    f_nyq = _nyquist_hz(dt)
    phase_df = float(np.clip(df_hz, -f_nyq * 0.95, f_nyq * 0.95))

    if j_type == "jammer_RANDOM":
        mech = _resolve_random_mechanism(jammer)
        if mech == "swept":
            j_type = "jammer_swept"
        elif mech == "int_sample":
            j_type = "jammer_InterruptedSampling"
        elif mech == "false_target":
            j_type = "jammer_multi_false_targets"
        else:
            j_type = "jammer_wide_band"

    if j_type == "jammer_swept":
        step_hz = ghz_to_hz(float(jammer.get("sweep_step_ghz", 0.01)))
        sweep_rate = step_hz / max(T, 1e-9)
        phase = 2 * np.pi * (phase_df * t_s + 0.5 * sweep_rate * t_s**2)
        j = np.exp(1j * phase)
        j *= 0.6 + 0.4 * rng.standard_normal(n)
    elif j_type == "jammer_narrow_band":
        j = _bandpass_noise(rng, n, dt, j_center, min(j_eff_bw, 30e6))
    elif j_type == "jammer_wide_band":
        j = _bandpass_noise(rng, n, dt, j_center, j_eff_bw)
    elif j_type == "jammer_close":
        j = _bandpass_noise(rng, n, dt, j_center, max(j_eff_bw, 80e6))
        j += 0.5 * np.exp(1j * 2 * np.pi * phase_df * t_s)
    elif j_type == "jammer_InterruptedSampling":
        dur_ns = float(jammer.get("int_sample_dur_ns", 5000))
        delay_ns = float(jammer.get("int_sample_delay_ns", 2000))
        intv_ns = float(jammer.get("int_sample_intv_ns", dur_ns + delay_ns + 200))
        dur_s = dur_ns * 1e-9
        delay_s = delay_ns * 1e-9
        intv_s = max(intv_ns * 1e-9, dur_s + 1e-9)
        noise = _bandpass_noise(rng, n, dt, j_center, j_eff_bw)
        gate = np.zeros(n)
        t = t_s[0]
        while t < t_s[-1]:
            on_start = t + delay_s
            on_end = on_start + dur_s
            gate[(t_s >= on_start) & (t_s < on_end)] = 1.0
            t += intv_s
        j = noise * gate
    elif j_type == "jammer_multi_false_targets":
        ft_num = int(max(float(jammer.get("false_target_num", 5)), 1))
        spacing = max(n // (ft_num + 2), 8)
        j = np.zeros(n, dtype=np.complex128)
        for k in range(1, ft_num + 1):
            shift = k * spacing
            amp = 0.55 / np.sqrt(k)
            j += amp * np.roll(radar_signal, shift)
        j += 0.15 * _bandpass_noise(rng, n, dt, j_center, j_eff_bw)
    else:
        j = _bandpass_noise(rng, n, dt, j_center, j_eff_bw)

    return _normalize_power(j)


def _normalize_power(x: np.ndarray) -> np.ndarray:
    p = float(np.mean(np.abs(x) ** 2))
    if p < 1e-12:
        return x.astype(np.complex128)
    return (x / np.sqrt(p)).astype(np.complex128)


def _scale_to_sinr(
    radar: np.ndarray,
    jammer: np.ndarray,
    sinr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按目标 SINR 缩放信号与干扰，并加入弱热噪声。"""
    ps = 1.0
    pn = 10 ** (-42 / 10)
    pj = max(ps / (10 ** (sinr_db / 10)) - pn, 1e-6)
    noise = _normalize_power(
        (rng.standard_normal(len(radar)) + 1j * rng.standard_normal(len(radar))) / np.sqrt(2)
    ) * np.sqrt(pn)
    s = radar * np.sqrt(ps)
    j = jammer * np.sqrt(pj)
    return s, j, noise


def _fig_to_data_url(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _style_axes(ax: plt.Axes, *, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(_AX_BG)
    ax.grid(True, color=_GRID, alpha=0.55, lw=0.6)
    ax.tick_params(colors=_TICK, labelsize=_FS_TICK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_FS_LABEL, color=_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_FS_LABEL, color=_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)


def _plot_time_domain(
    t_us: np.ndarray,
    s: np.ndarray,
    j_shape: np.ndarray,
    y: np.ndarray,
    sinr_db: float,
    jammer_type: str,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.0), sharex=True)
    fig.patch.set_facecolor(_FIG_BG)
    panels = [
        (np.abs(s), "Radar echo (SINR-scaled)", _C_RADAR),
        (np.abs(j_shape), "Jammer envelope (normalized shape)", _C_JAM),
        (np.abs(y), "Jammed RX (SINR-scaled)", _C_RX),
    ]
    for ax, (env, title, color) in zip(axes, panels):
        ax.plot(t_us, env, color=color, lw=1.1, alpha=0.95)
        _style_axes(ax, ylabel="Amplitude")
        ax.set_title(title, fontsize=_FS_SUBTITLE, color=_TEXT, loc="left", pad=6)
    _style_axes(axes[-1], xlabel="Time (us)")
    fig.suptitle(
        f"Baseband time domain | {jammer_type} | SINR {sinr_db:.1f} dB",
        fontsize=_FS_TITLE,
        color=_TEXT,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _annotate_bw_spans(
    ax: plt.Axes,
    half_r_mhz: float,
    df_mhz: float,
    half_j_mhz: float,
    *,
    show_legend: bool,
) -> None:
    ax.axvspan(-half_r_mhz, half_r_mhz, color=_C_BAND_R, label="Radar BW" if show_legend else None)
    ax.axvspan(
        df_mhz - half_j_mhz,
        df_mhz + half_j_mhz,
        color=_C_BAND_J,
        label="Jammer BW" if show_legend else None,
    )


def _plot_spectrum(
    dt: float,
    s: np.ndarray,
    j: np.ndarray,
    y: np.ndarray,
    r_bw_hz: float,
    j_bw_hz: float,
    df_hz: float,
    overlap: float,
) -> plt.Figure:
    n = len(y)
    freqs_mhz = np.fft.fftshift(np.fft.fftfreq(n, dt)) / 1e6
    mag_s = np.abs(np.fft.fftshift(np.fft.fft(s)))
    mag_j = np.abs(np.fft.fftshift(np.fft.fft(j)))
    mag_y = np.abs(np.fft.fftshift(np.fft.fft(y)))

    spec_s = 20 * np.log10(mag_s + 1e-9)
    spec_j = 20 * np.log10(mag_j + 1e-9)
    spec_y = 20 * np.log10(mag_y + 1e-9)
    spec_delta = 10 * np.log10(np.maximum(mag_y**2 - mag_s**2, 1e-18))

    j_center, j_eff_bw = _viz_jammer_band(df_hz, dt, r_bw_hz, j_bw_hz, overlap)
    f_nyq_mhz = _nyquist_hz(dt) / 1e6
    half_r = r_bw_hz / 2e6
    half_j = j_eff_bw / 2e6
    df_mhz = j_center / 1e6

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.6), sharex=True)
    fig.patch.set_facecolor(_FIG_BG)

    ax0 = axes[0]
    _annotate_bw_spans(ax0, half_r, df_mhz, half_j, show_legend=True)
    ax0.plot(freqs_mhz, spec_s, color=_C_RADAR, lw=1.4, label="Radar")
    ax0.plot(freqs_mhz, spec_j, color=_C_JAM, lw=1.4, label="Jammer")
    ax0.set_xlim(-f_nyq_mhz, f_nyq_mhz)
    _style_axes(ax0, ylabel="Magnitude (dB)")
    ax0.set_title("Signal components (Radar vs Jammer)", fontsize=_FS_SUBTITLE, color=_TEXT, loc="left", pad=6)
    ax0.legend(
        fontsize=_FS_LEGEND,
        loc="upper right",
        framealpha=0.95,
        edgecolor=_GRID,
        facecolor=_FIG_BG,
        labelcolor=_TEXT,
        handlelength=2.2,
        borderpad=0.6,
    )

    ax1 = axes[1]
    _annotate_bw_spans(ax1, half_r, df_mhz, half_j, show_legend=False)
    ax1.plot(freqs_mhz, spec_s, color=_C_RADAR, lw=1.0, ls="--", alpha=0.85, label="Radar (clean)")
    ax1.plot(freqs_mhz, spec_y, color=_C_RX, lw=1.6, label="Jammed (RX)")
    ax1.fill_between(
        freqs_mhz,
        spec_s,
        spec_y,
        where=spec_y >= spec_s,
        color=_C_JAM,
        alpha=0.3,
        label="Spectral lift (dB gap)",
    )
    _style_axes(ax1, ylabel="Magnitude (dB)")
    ax1.set_title(
        "Received vs clean (solid = Jammed, dashed = Radar; orange fill = lift)",
        fontsize=_FS_SUBTITLE,
        color=_TEXT,
        loc="left",
        pad=6,
    )
    ax1.legend(
        fontsize=_FS_LEGEND,
        loc="upper right",
        framealpha=0.95,
        edgecolor=_GRID,
        facecolor=_FIG_BG,
        labelcolor=_TEXT,
        handlelength=2.2,
        borderpad=0.6,
    )

    ax2 = axes[2]
    _annotate_bw_spans(ax2, half_r, df_mhz, half_j, show_legend=False)
    ax2.plot(freqs_mhz, spec_delta, color="#b45309", lw=1.5, label="Excess power (Jammed − Radar)")
    fill_floor = float(np.percentile(spec_delta, 2)) - 3.0
    ax2.fill_between(freqs_mhz, fill_floor, spec_delta, color=_C_JAM, alpha=0.25)
    delta_lo = float(np.percentile(spec_delta, 2))
    delta_hi = float(np.percentile(spec_delta, 99.8))
    pad = max(3.0, (delta_hi - delta_lo) * 0.15)
    ax2.set_ylim(delta_lo - pad, delta_hi + pad)
    _style_axes(
        ax2,
        xlabel="Baseband freq offset (MHz, ref. radar carrier)",
        ylabel="Excess (dB)",
    )
    ax2.set_title(
        "Interference excess only (auto-scaled — highlights jammer-affected bands)",
        fontsize=_FS_SUBTITLE,
        color=_TEXT,
        loc="left",
        pad=6,
    )
    ax2.legend(
        fontsize=_FS_LEGEND,
        loc="upper right",
        framealpha=0.95,
        edgecolor=_GRID,
        facecolor=_FIG_BG,
        labelcolor=_TEXT,
        handlelength=2.2,
        borderpad=0.6,
    )

    fig.suptitle("Intra-pulse spectrum (FFT)", fontsize=_FS_TITLE, color=_TEXT, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def generate_jammed_waveform_figures(
    jammer: Dict[str, Any],
    radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据干扰/雷达参数合成受扰波形图。

    采用基带复包络近似：LFM/编码脉冲 + 类型化干扰 + 按解析 SINR 定标。
    """
    jammer_row = jammer_dict_to_series(jammer)
    radar_row = _radar_row_from_dict(radar)
    sinr_db = float(sinr_from_csv_row(jammer_row, radar_row, add_noise=False))

    r_freq = ghz_to_hz(float(radar["carrier_freq_ghz"]))
    j_freq = ghz_to_hz(float(jammer["carrier_freq_ghz"]))
    r_bw = ghz_to_hz(float(radar["bandwidth_ghz"]))
    j_bw = max(ghz_to_hz(float(jammer["bandwidth_ghz"])), 1e6)
    pw_ns = float(radar.get("pulse_width_ns", 30000))
    df_hz = j_freq - r_freq
    overlap = _overlap_factor(r_freq, r_bw, j_freq, j_bw)

    n = min(_MAX_SAMPLES, max(int(r_bw * pw_ns * 1e-9 * 2.5), 512))
    t_ns, dt = _time_axis_ns(pw_ns, n)
    t_s = t_ns * 1e-9
    t_us = t_ns * 1e-3

    radar_bb = _radar_baseband(t_s, str(radar_row["Radar_Type"]), r_bw, pw_ns)
    jammer_bb = _jammer_baseband(t_s, dt, jammer, radar_bb, df_hz, r_bw, overlap)

    rng = np.random.default_rng(_WF_SEED)
    s, j, noise = _scale_to_sinr(radar_bb, jammer_bb, sinr_db, rng)
    y = s + j + noise

    time_fig = _plot_time_domain(
        t_us, s, jammer_bb, y, sinr_db, str(jammer["jammer_type"])
    )
    spec_fig = _plot_spectrum(dt, s, j, y, r_bw, j_bw, df_hz, overlap)

    df_note = (
        f"频偏 {df_hz/1e6:.1f} MHz"
        if abs(df_hz) <= _nyquist_hz(dt) * 0.95
        else f"RF 频偏 {df_hz/1e6:.1f} MHz（脉内窗展示等效耦合带宽）"
    )

    return {
        "sinr_db": round(sinr_db, 2),
        "overlap": round(overlap, 3),
        "freq_offset_mhz": round(df_hz / 1e6, 2),
        "radar": radar_series_to_dict(radar_row),
        "description": (
            f"基带仿真：雷达 {radar_row['Radar_Type']} 脉冲内，"
            f"干扰 {jammer['jammer_type']}，{df_note}，"
            f"谱重叠 {overlap:.0%}；中图为归一化干扰形状，上下图为 SINR 定标。"
        ),
        "images": {
            "time_domain": _fig_to_data_url(time_fig),
            "spectrum": _fig_to_data_url(spec_fig),
        },
    }
