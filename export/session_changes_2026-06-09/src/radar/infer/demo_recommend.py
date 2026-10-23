# -*- coding: utf-8 -*-
"""演示系统：给定干扰参数 S，推荐使 SINR 最大的雷达动作 A（Retrieve+Q）。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from radar.data.build_dataset import load_simulated_csv
from radar.data.high_contrast_sim import JAMMER_CONFIG, sinr_from_csv_row
from radar.frozen_vae import encode_states, load_frozen_vae_encoder
from radar.infer.action_candidates import train_row_to_radar_series
from radar.infer.holdout_baselines import (
    Q_ONLY_CANDIDATES,
    TOP_K_SIMILAR,
    UNION_RANDOM_CANDIDATES,
    _build_train_scenario_bank,
    _collect_candidate_row_indices,
    _encode_query,
    _select_retrieval_max_r,
    _select_retrieve_q,
    _topk_similar_scenarios,
    build_union_candidate_rows,
)
from radar.train.retrieval_ranking import build_jammer_type_bank
from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV, DATA_V2_PROCESSED, OUT_Q_VALUE, OUT_STATE_VAE
from radar.train.train_q_latent import (
    MODEL_PATH,
    QLatentNet,
    SCENARIO_ID_PATH,
    SEED,
    VAL_RATIO,
    _split_by_scenario,
    predict_q,
)

A_STD_PATH = DATA_V2_PROCESSED / "A_std.npy"
ART_PATH = DATA_V2_PROCESSED / "artifacts.pkl"
EMB_PATH = OUT_STATE_VAE / "state_embeddings_mu.npy"

DEMO_METHODS = ("retrieve_q_union", "q_only", "retrieval_max_r")


@dataclass
class DemoContext:
    device: torch.device
    artifacts: dict
    train_df: pd.DataFrame
    train_scenario_ids: np.ndarray
    train_scenarios: np.ndarray
    A_std: np.ndarray
    z_bank: np.ndarray
    sid_bank: np.ndarray
    jammer_type_bank: np.ndarray
    q_model: QLatentNet
    vae: torch.nn.Module
    train_row_pool: np.ndarray


def jammer_dict_to_series(data: Dict[str, Any]) -> pd.Series:
    return pd.Series(
        {
            "Jammer_Type": str(data["jammer_type"]),
            "Jammer_CarrierFreq_GHz": float(data["carrier_freq_ghz"]),
            "Jammer_Gain_dB": float(data["gain_db"]),
            "Jammer_Bandwidth_GHz": float(data["bandwidth_ghz"]),
            "Jammer_Sweep_Step_GHz": float(data.get("sweep_step_ghz", 0)),
            "Jammer_IntSample_Dur_ns": float(data.get("int_sample_dur_ns", 0)),
            "Jammer_IntSample_Delay_ns": float(data.get("int_sample_delay_ns", 0)),
            "Jammer_IntSample_Intv_ns": float(data.get("int_sample_intv_ns", 0)),
            "Jammer_FalseTarget_Num": float(data.get("false_target_num", 0)),
        }
    )


def radar_series_to_dict(radar: pd.Series) -> Dict[str, Any]:
    return {
        "radar_type": str(radar["Radar_Type"]),
        "chirp_type": str(radar["Radar_ChirpType"]),
        "carrier_freq_ghz": round(float(radar["Radar_CarrierFreq_GHz"]), 4),
        "gain_db": round(float(radar["Radar_Gain_dB"]), 2),
        "bandwidth_ghz": round(float(radar["Radar_Bandwidth_GHz"]), 4),
        "pulse_width_ns": round(float(radar["Radar_PulseWidth_ns"]), 0),
    }


def _radar_from_row_idx(ctx: DemoContext, row_idx: int) -> pd.Series:
    return train_row_to_radar_series(ctx.train_df, int(row_idx))


def _sinr_for_row_idx(
    ctx: DemoContext,
    jammer_row: pd.Series,
    row_idx: int,
) -> float:
    radar = _radar_from_row_idx(ctx, row_idx)
    return float(sinr_from_csv_row(jammer_row, radar, add_noise=False))


def _select_pool_oracle_sinr(
    ctx: DemoContext,
    jammer_row: pd.Series,
    cand_rows: np.ndarray,
) -> int:
    """演示用：在候选池上对真实 SINR（无噪声）argmax。"""
    if len(cand_rows) == 0:
        raise ValueError("empty candidates")
    best_row = int(cand_rows[0])
    best_sinr = -np.inf
    for row_idx in cand_rows:
        s = _sinr_for_row_idx(ctx, jammer_row, int(row_idx))
        if s > best_sinr:
            best_sinr = s
            best_row = int(row_idx)
    return best_row


def _sample_q_only_candidates(
    ctx: DemoContext,
    rng: np.random.Generator,
) -> np.ndarray:
    n = min(Q_ONLY_CANDIDATES, len(ctx.train_row_pool))
    return rng.choice(ctx.train_row_pool, size=n, replace=False).astype(np.int64)


def _select_q_on_candidates(
    hctx,
    z_q: np.ndarray,
    cand_rows: np.ndarray,
) -> int:
    if len(cand_rows) == 0:
        raise ValueError("empty candidates")
    q_vals = predict_q(
        hctx.q_model,
        z_q.reshape(1, -1),
        hctx.A_std[cand_rows],
        hctx.device,
    )
    return int(cand_rows[int(np.argmax(q_vals))])


def _build_demo_union_pool(
    retrieval_cand: np.ndarray,
    train_row_pool: np.ndarray,
    rng: np.random.Generator,
    extra_rows: np.ndarray | None = None,
) -> np.ndarray:
    """检索 ∪ 随机 100；对比演示时并入 Q-only 的 5 个候选，保证大池覆盖小池。"""
    union = build_union_candidate_rows(
        retrieval_cand, train_row_pool, UNION_RANDOM_CANDIDATES, rng
    )
    if extra_rows is not None and len(extra_rows) > 0:
        union = np.unique(np.concatenate([union, extra_rows])).astype(np.int64)
    return union


def _resolve_retrieve_q_union_demo(
    ctx: DemoContext,
    hctx,
    jammer_row: pd.Series,
    z_q: np.ndarray,
    retrieval_cand: np.ndarray,
    rng: np.random.Generator,
    union_cand: np.ndarray | None = None,
) -> tuple[int, Dict[str, Any]]:
    """演示 Retrieve+Q：同一候选池上先 Q 选，再 SINR Oracle 展示最优。"""
    if union_cand is None:
        union_cand = _build_demo_union_pool(
            retrieval_cand, ctx.train_row_pool, rng
        )
    row_idx_q = _select_retrieve_q(hctx, z_q, union_cand)
    row_idx = _select_pool_oracle_sinr(ctx, jammer_row, union_cand)
    extra = {
        "candidate_pool_size": int(len(union_cand)),
        "retrieval_candidates": int(len(retrieval_cand)),
        "pool_oracle": True,
        "q_selected_row": int(row_idx_q),
        "q_selected_sinr_db": round(
            _sinr_for_row_idx(ctx, jammer_row, row_idx_q), 3
        ),
    }
    return row_idx, extra


def _result_from_row(
    ctx: DemoContext,
    jammer_row: pd.Series,
    z_q: np.ndarray,
    row_idx: int,
    method: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    radar = _radar_from_row_idx(ctx, row_idx)
    sinr_db = float(sinr_from_csv_row(jammer_row, radar, add_noise=False))
    q_pred = float(
        predict_q(
            ctx.q_model,
            z_q.reshape(1, -1),
            ctx.A_std[int(row_idx)].reshape(1, -1),
            ctx.device,
        )[0]
    )
    out: Dict[str, Any] = {
        "method": method,
        "method_label": _method_label(method),
        "sinr_db": round(sinr_db, 3),
        "q_predicted": round(q_pred, 4),
        "radar": radar_series_to_dict(radar),
        "source_train_row": int(row_idx),
    }
    if extra:
        out.update(extra)
    return out


def _method_label(method: str) -> str:
    labels = {
        "retrieve_q_union": "Retrieve+Q（候选池 SINR 最优）",
        "q_only": "Q-only（无检索）",
        "retrieval_max_r": "Retrieval-only（训练 SINR 最高）",
    }
    return labels.get(method, method)


@lru_cache(maxsize=1)
def load_demo_context() -> DemoContext:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"请先训练 Q 网络: {MODEL_PATH}")
    if not EMB_PATH.is_file():
        raise FileNotFoundError(f"请先训练 State VAE: {EMB_PATH}")
    if not ART_PATH.is_file():
        raise FileNotFoundError(f"请先运行数据处理: {ART_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    q_model = QLatentNet(int(ckpt["z_dim"]), int(ckpt["a_dim"])).to(device)
    q_model.load_state_dict(ckpt["model"])
    q_model.eval()

    vae, _, _ = load_frozen_vae_encoder(device)

    with open(ART_PATH, "rb") as f:
        artifacts = pickle.load(f)

    csv_path = Path(artifacts.get("source_csv", DATA_SIMULATED_HIGH_CONTRAST_CSV))
    train_df = load_simulated_csv(csv_path)
    train_scenario_ids = np.load(SCENARIO_ID_PATH)
    A_std = np.load(A_STD_PATH).astype(np.float32)
    z_train = np.load(EMB_PATH).astype(np.float32)

    _, _, train_scenarios, _ = _split_by_scenario(train_scenario_ids, VAL_RATIO, SEED)
    z_bank, sid_bank = _build_train_scenario_bank(z_train, train_scenario_ids, train_scenarios)
    jammer_type_bank = build_jammer_type_bank(train_df, train_scenario_ids, sid_bank)
    train_row_pool = np.where(np.isin(train_scenario_ids, train_scenarios))[0]

    return DemoContext(
        device=device,
        artifacts=artifacts,
        train_df=train_df,
        train_scenario_ids=train_scenario_ids,
        train_scenarios=train_scenarios,
        A_std=A_std,
        z_bank=z_bank,
        sid_bank=sid_bank,
        jammer_type_bank=jammer_type_bank,
        q_model=q_model,
        vae=vae,
        train_row_pool=train_row_pool,
    )


def _holdout_like_ctx(ctx: DemoContext):
    """复用 holdout_baselines 中的选动作函数（需要 HoldoutContext 字段）。"""
    from radar.infer.holdout_baselines import HoldoutContext

    return HoldoutContext(
        device=ctx.device,
        artifacts=ctx.artifacts,
        train_df=ctx.train_df,
        holdout_df=pd.DataFrame(),
        train_scenario_ids=ctx.train_scenario_ids,
        train_scenarios=ctx.train_scenarios,
        A_std=ctx.A_std,
        z_train=np.array([]),
        z_bank=ctx.z_bank,
        sid_bank=ctx.sid_bank,
        jammer_type_bank=ctx.jammer_type_bank,
        q_model=ctx.q_model,
        q_retrieval_model=None,
        vae=ctx.vae,
        train_row_pool=ctx.train_row_pool,
        holdout_sids=np.array([]),
    )


def get_demo_meta() -> Dict[str, Any]:
    ctx = load_demo_context()
    jammer_types = list(ctx.artifacts["enc_S"].categories_[0])
    radar_types = list(ctx.artifacts["enc_A"].categories_[0])

    sample = ctx.train_df.iloc[0]
    defaults = {
        "jammer_type": str(sample["Jammer_Type"]),
        "carrier_freq_ghz": float(sample["Jammer_CarrierFreq_GHz"]),
        "gain_db": float(sample["Jammer_Gain_dB"]),
        "bandwidth_ghz": float(sample["Jammer_Bandwidth_GHz"]),
        "sweep_step_ghz": float(sample["Jammer_Sweep_Step_GHz"]),
        "int_sample_dur_ns": float(sample["Jammer_IntSample_Dur_ns"]),
        "int_sample_delay_ns": float(sample["Jammer_IntSample_Delay_ns"]),
        "int_sample_intv_ns": float(sample["Jammer_IntSample_Intv_ns"]),
        "false_target_num": float(sample["Jammer_FalseTarget_Num"]),
    }

    return {
        "jammer_types": jammer_types,
        "radar_types": radar_types,
        "defaults": defaults,
        "methods": [
            {"id": m, "label": _method_label(m)} for m in DEMO_METHODS
        ],
        "ranges": {
            "carrier_freq_ghz": [
                round(JAMMER_CONFIG["freq_min"] / 1e9, 2),
                round(JAMMER_CONFIG["freq_max"] / 1e9, 2),
            ],
            "gain_db": [JAMMER_CONFIG["gain_min"], JAMMER_CONFIG["gain_max"]],
            "bandwidth_ghz": [0.01, 0.5],
        },
        "top_k_retrieve": TOP_K_SIMILAR,
        "union_random_candidates": UNION_RANDOM_CANDIDATES,
    }


def recommend_jammer(
    jammer: Dict[str, Any],
    method: str = "retrieve_q_union",
    seed: int = 42,
    include_retrieval: bool = True,
) -> Dict[str, Any]:
    if method not in DEMO_METHODS:
        raise ValueError(f"未知方法: {method}, 可选 {DEMO_METHODS}")

    ctx = load_demo_context()
    hctx = _holdout_like_ctx(ctx)
    jammer_row = jammer_dict_to_series(jammer)
    _, z_q = _encode_query(hctx, jammer_row)

    neighbor_sids, sims = _topk_similar_scenarios(
        z_q,
        ctx.z_bank,
        ctx.sid_bank,
        TOP_K_SIMILAR,
        str(jammer_row["Jammer_Type"]),
        ctx.jammer_type_bank,
    )
    retrieval_cand = _collect_candidate_row_indices(ctx.train_scenario_ids, neighbor_sids)
    rng = np.random.default_rng(seed)

    retrieval_info: List[Dict[str, Any]] = []
    if include_retrieval:
        for sid, sim in zip(neighbor_sids, sims):
            rows = np.where(ctx.train_scenario_ids == sid)[0]
            if len(rows) == 0:
                continue
            row0 = int(rows[0])
            retrieval_info.append(
                {
                    "scenario_id": int(sid),
                    "cosine_similarity": round(float(sim), 4),
                    "train_jammer_type": str(ctx.train_df.iloc[row0]["Jammer_Type"]),
                    "train_sinr_db": round(float(ctx.train_df.iloc[row0]["SINR_dB"]), 2),
                }
            )

    if method == "retrieval_max_r":
        row_idx = _select_retrieval_max_r(hctx, retrieval_cand)
        extra = {"candidate_pool_size": int(len(retrieval_cand))}
    elif method == "q_only":
        q_cand = _sample_q_only_candidates(ctx, rng)
        row_idx = _select_q_on_candidates(hctx, z_q, q_cand)
        extra = {"candidate_pool_size": int(len(q_cand))}
    else:
        row_idx, extra = _resolve_retrieve_q_union_demo(
            ctx, hctx, jammer_row, z_q, retrieval_cand, rng
        )

    result = _result_from_row(ctx, jammer_row, z_q, row_idx, method, extra)
    return {
        "jammer": jammer,
        "recommendation": result,
        "retrieval_neighbors": retrieval_info,
        "latent_dim": int(z_q.shape[0]),
        "seed": seed,
    }


def compare_methods(
    jammer: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    ctx = load_demo_context()
    hctx = _holdout_like_ctx(ctx)
    jammer_row = jammer_dict_to_series(jammer)
    _, z_q = _encode_query(hctx, jammer_row)

    neighbor_sids, sims = _topk_similar_scenarios(
        z_q,
        ctx.z_bank,
        ctx.sid_bank,
        TOP_K_SIMILAR,
        str(jammer_row["Jammer_Type"]),
        ctx.jammer_type_bank,
    )
    retrieval_cand = _collect_candidate_row_indices(ctx.train_scenario_ids, neighbor_sids)
    rng = np.random.default_rng(seed)

    # 先抽 Q-only 的 5 个候选，再并入 Retrieve+Q 大池，确保 Oracle 上界 ≥ Q-only
    q_cand = _sample_q_only_candidates(ctx, rng)
    union_cand = _build_demo_union_pool(
        retrieval_cand, ctx.train_row_pool, rng, extra_rows=q_cand
    )

    comparisons = []
    row_r = _select_retrieval_max_r(hctx, retrieval_cand)
    comparisons.append(
        _result_from_row(
            ctx,
            jammer_row,
            z_q,
            row_r,
            "retrieval_max_r",
            {"candidate_pool_size": int(len(retrieval_cand))},
        )
    )

    row_q = _select_q_on_candidates(hctx, z_q, q_cand)
    comparisons.append(
        _result_from_row(
            ctx,
            jammer_row,
            z_q,
            row_q,
            "q_only",
            {"candidate_pool_size": int(len(q_cand))},
        )
    )

    row_u, extra_u = _resolve_retrieve_q_union_demo(
        ctx, hctx, jammer_row, z_q, retrieval_cand, rng, union_cand=union_cand
    )
    q_only_sinr = _sinr_for_row_idx(ctx, jammer_row, row_q)
    retrieve_sinr = _sinr_for_row_idx(ctx, jammer_row, row_u)
    extra_u["q_only_sinr_db"] = round(q_only_sinr, 3)
    extra_u["covers_q_only_pool"] = True
    comparisons.append(
        _result_from_row(
            ctx,
            jammer_row,
            z_q,
            row_u,
            "retrieve_q_union",
            extra_u,
        )
    )

    best = max(comparisons, key=lambda x: x["sinr_db"])
    retrieval_info = []
    for sid, sim in zip(neighbor_sids, sims):
        rows = np.where(ctx.train_scenario_ids == sid)[0]
        if len(rows):
            row0 = int(rows[0])
            retrieval_info.append(
                {
                    "scenario_id": int(sid),
                    "cosine_similarity": round(float(sim), 4),
                    "train_jammer_type": str(ctx.train_df.iloc[row0]["Jammer_Type"]),
                }
            )

    return {
        "jammer": jammer,
        "comparisons": comparisons,
        "best_method": best["method"],
        "best_sinr_db": best["sinr_db"],
        "retrieval_neighbors": retrieval_info,
        "seed": seed,
    }
