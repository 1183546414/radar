# -*- coding: utf-8 -*-
"""
留出集方法对比 / 消融评估（500 未见 S，共享同一 holdout CSV）。

方法:
  1. random              — Top-5 检索候选池 (~100 A) 内随机选
  2. retrieval_max_r     — Top-5 检索候选里选训练集 SINR 最高的 A
  3. q_only              — 无检索，从 train 池随机 100 个 A，argmax Q
  4. retrieve_q          — Top-5 检索 + argmax Q
  5. retrieve_q_union    — 检索池 ∪ train 随机 100 A，argmax Q
  6. retrieve_q_local    — 方向 A1：检索池 + 局部扰动，argmax Q
  7. retrieve_q_gen      — 方向 A3：检索池 + 新采样雷达 A，argmax Q
  8. retrieve_q_ctx      — 方向 B：检索条件 Q(z, z_ctx, A)，argmax Q
  9. retrieve_q_ctx_union — 检索条件 Q + 检索池 ∪ 随机 100 A
 10. oracle_k20          — 留出随机 K 动作参考上界（兼容旧方法名）
 11. oracle_wide         — 对查询 S 再随机 100 个 A 上界
"""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from radar.data.build_dataset import load_simulated_csv, row_to_state_vector
from radar.data.high_contrast_sim import (
    generate_holdout_dataset,
    hz_to_ghz,
    sample_radar_action,
    sinr_from_csv_row,
)
from radar.frozen_vae import encode_states, load_frozen_vae_encoder
from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV, DATA_V2_PROCESSED, OUT_Q_VALUE, OUT_STATE_VAE
from radar.infer.action_candidates import (
    GEN_NEW_ACTIONS,
    LOCAL_PERTURB_PER_SEED,
    LOCAL_SEED_TOP,
    ScoredAction,
    build_generated_radar_actions,
    build_local_perturbation_actions,
    build_retrieval_train_actions,
    sinr_for_action,
    stack_a_std,
)
from radar.train.q_retrieval_net import QRetrievalNet, aggregate_neighbor_z, predict_q_retrieval
from radar.train.retrieval_ranking import topk_similar_scenarios, build_jammer_type_bank
from radar.train.train_q_retrieval import RETRIEVAL_MODEL_PATH
from radar.train.train_q_latent import (
    MODEL_PATH,
    QLatentNet,
    SCENARIO_ID_PATH,
    VAL_RATIO,
    SEED,
    _split_by_scenario,
    predict_q,
)

METHODS_REQUIRING_RETRIEVAL_Q = frozenset({"retrieve_q_ctx", "retrieve_q_ctx_union"})

ART_PATH = DATA_V2_PROCESSED / "artifacts.pkl"
EMB_PATH = OUT_STATE_VAE / "state_embeddings_mu.npy"
A_STD_PATH = DATA_V2_PROCESSED / "A_std.npy"
HOLDOUT_CSV_PATH = DATA_V2_PROCESSED / "holdout_eval_scenarios.csv"
OUT_DIR = OUT_Q_VALUE / "holdout_baselines"
SUMMARY_JSON = OUT_DIR / "summary.json"
SUMMARY_CSV = OUT_DIR / "summary.csv"
PER_SCENARIO_JSON = OUT_DIR / "per_scenario_detail.json"

N_HOLDOUT_SCENARIOS = 500
ACTIONS_PER_STATE = 10
TOP_K_SIMILAR = 5
Q_ONLY_CANDIDATES = 5
UNION_RANDOM_CANDIDATES = 100
ORACLE_WIDE_CANDIDATES = 100
HOLDOUT_SEED = 2026

METHODS = (
    "random",
    "retrieval_max_r",
    "q_only",
    "retrieve_q",
    "retrieve_q_union",
    "retrieve_q_local",
    "retrieve_q_gen",
    "retrieve_q_ctx",
    "retrieve_q_ctx_union",
    "oracle_k20",
    "oracle_wide",
)

# 主流程 / 对比表默认展示的方法（本文方法取效果最好的 Union 变体）
OUR_METHOD = "retrieve_q_union"
REPORT_METHODS = (
    "random",
    "retrieval_max_r",
    "q_only",
    OUR_METHOD,
    "oracle_wide",
)

METHOD_LABELS = {
    "random": "Random（检索候选池内随机）",
    "retrieval_max_r": "Retrieval-only（训练 R 最高）",
    "q_only": "Q-only（无检索，随机 5 A）",
    "retrieve_q": "Retrieve+Q（Top-5 + argmax Q）",
    "retrieve_q_union": "Retrieve+Q（检索池 ∪ 随机 100 A）",
    "retrieve_q_local": "Retrieve+Q-Local（检索 + 局部扰动）",
    "retrieve_q_gen": "Retrieve+Q-Gen（检索 + 新采样 A）",
    "retrieve_q_ctx": "Retrieve+Q-Ctx（检索条件 Q）",
    "retrieve_q_ctx_union": "Retrieve+Q-Ctx-Union（检索条件 Q + 大池）",
    "oracle_k20": "Reference-K（留出随机 K 动作上界）",
    "oracle_wide": "Oracle-Wide（查询 S 上 100 随机 A 上界）",
}

# 可选信号级真值后端（RadarSimPy）。未设置时沿用 CSV 解析标签。
_TRUTH_FN: Optional[Callable[[pd.Series, pd.Series], float]] = None


def set_sinr_backend(fn) -> None:
    """运行期注入信号级 SINR 真值函数 fn(jammer_row, radar_row) -> dB。

    覆盖本模块与 action_candidates 内全部解析真值调用点；
    holdout CSV 的 SINR_dB 列（解析）仅用于生成/筛选，不再作为真值。
    """
    global _TRUTH_FN, sinr_from_csv_row, sinr_for_action
    _TRUTH_FN = fn
    sinr_from_csv_row = fn
    sinr_for_action = lambda jammer_row, action: fn(jammer_row, action.radar_row)  # noqa: E731
    from radar.infer import action_candidates as _ac

    _ac.sinr_from_csv_row = fn


def _holdout_truth(block: pd.DataFrame, local: int) -> float:
    """holdout 场景 block 内第 local 个动作的真值 SINR（信号级或解析回退）。"""
    if _TRUTH_FN is None:
        return float(block.iloc[int(local)]["SINR_dB"])
    return float(_TRUTH_FN(block.iloc[0], block.iloc[int(local)]))


def _l2n(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


@dataclass
class HoldoutContext:
    device: torch.device
    artifacts: dict
    train_df: pd.DataFrame
    holdout_df: pd.DataFrame
    train_scenario_ids: np.ndarray
    train_scenarios: np.ndarray
    A_std: np.ndarray
    z_train: np.ndarray
    z_bank: np.ndarray
    sid_bank: np.ndarray
    jammer_type_bank: np.ndarray
    q_model: QLatentNet
    q_retrieval_model: Optional[QRetrievalNet]
    vae: torch.nn.Module
    train_row_pool: np.ndarray
    holdout_sids: np.ndarray


def _build_train_scenario_bank(
    z: np.ndarray,
    scenario_ids: np.ndarray,
    train_scenarios: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_list, sid_list = [], []
    for sid in train_scenarios:
        idx = np.where(scenario_ids == sid)[0]
        if len(idx) == 0:
            continue
        z_list.append(z[int(idx[0])])
        sid_list.append(int(sid))
    return np.stack(z_list, axis=0).astype(np.float32), np.array(sid_list, dtype=np.int64)


def _topk_similar_scenarios(
    z_query: np.ndarray,
    z_bank: np.ndarray,
    train_sid_bank: np.ndarray,
    k: int,
    query_jammer_type: str,
    jammer_type_bank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return topk_similar_scenarios(
        z_query,
        z_bank,
        train_sid_bank,
        k,
        query_jammer_type=query_jammer_type,
        jammer_type_bank=jammer_type_bank,
    )


def _collect_candidate_row_indices(
    scenario_ids: np.ndarray,
    neighbor_sids: np.ndarray,
) -> np.ndarray:
    parts = [np.where(scenario_ids == sid)[0] for sid in neighbor_sids]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts)


def _encode_query(
    ctx: HoldoutContext,
    jammer_row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    s_vec = row_to_state_vector(jammer_row, ctx.artifacts).astype(np.float32)
    s_imp = ctx.artifacts["imp_S"].transform(s_vec.reshape(1, -1))
    s_std = ctx.artifacts["sc_S"].transform(s_imp).astype(np.float32)
    z_q = encode_states(ctx.vae, s_std, ctx.device)[0]
    return s_std, z_q


def _radar_series_from_train_row(train_df: pd.DataFrame, row_idx: int) -> pd.Series:
    return train_df.iloc[row_idx]


def _sinr_on_query(jammer_row: pd.Series, train_df: pd.DataFrame, row_idx: int) -> float:
    return sinr_from_csv_row(jammer_row, _radar_series_from_train_row(train_df, row_idx))


def _oracle_k20(block: pd.DataFrame) -> float:
    if _TRUTH_FN is None:
        return float(block["SINR_dB"].max())
    jrow = block.iloc[0]
    return float(max(_TRUTH_FN(jrow, block.iloc[i]) for i in range(len(block))))


def _oracle_wide(jammer_row: pd.Series, n: int, rng: np.random.Generator) -> float:
    best = -np.inf
    for _ in range(n):
        radar = sample_radar_action()
        radar_row = pd.Series(
            {
                "Radar_CarrierFreq_GHz": hz_to_ghz(radar["r_freq"]),
                "Radar_Gain_dB": radar["r_gain"],
                "Radar_Bandwidth_GHz": hz_to_ghz(radar["r_bw"]),
                "Radar_PulseWidth_ns": radar["r_pw"],
                "Radar_Type": radar["r_type"],
                "Radar_ChirpType": radar["r_chirp"],
            }
        )
        r = sinr_from_csv_row(jammer_row, radar_row)
        if r > best:
            best = r
    return float(best)


def _z_ctx_from_neighbor_sids(
    ctx: HoldoutContext,
    neighbor_sids: np.ndarray,
    sims: Optional[np.ndarray] = None,
) -> np.ndarray:
    zs = []
    for sid in neighbor_sids:
        pos = np.where(ctx.sid_bank == sid)[0]
        if len(pos):
            zs.append(ctx.z_bank[int(pos[0])])
    if not zs:
        raise ValueError("empty neighbor z for context")
    return aggregate_neighbor_z(np.stack(zs, axis=0), sims)


def _argmax_standard_q(
    ctx: HoldoutContext,
    z_q: np.ndarray,
    actions: List[ScoredAction],
) -> ScoredAction:
    a_mat = stack_a_std(actions)
    q_vals = predict_q(ctx.q_model, z_q.reshape(1, -1), a_mat, ctx.device)
    return actions[int(np.argmax(q_vals))]


def _argmax_retrieval_q(
    ctx: HoldoutContext,
    z_q: np.ndarray,
    z_ctx: np.ndarray,
    actions: List[ScoredAction],
) -> ScoredAction:
    if ctx.q_retrieval_model is None:
        raise FileNotFoundError(
            f"retrieve_q_ctx 需要检索条件 Q 模型，请先运行 train_q_retrieval: {RETRIEVAL_MODEL_PATH}"
        )
    a_mat = stack_a_std(actions)
    q_vals = predict_q_retrieval(
        ctx.q_retrieval_model,
        z_q.reshape(1, -1),
        a_mat,
        z_ctx.reshape(1, -1),
        ctx.device,
    )
    return actions[int(np.argmax(q_vals))]


def load_holdout_context(
    n_holdout: int = N_HOLDOUT_SCENARIOS,
    actions_per_state: int = ACTIONS_PER_STATE,
    holdout_seed: int = HOLDOUT_SEED,
    holdout_csv: Optional[Path] = None,
    regenerate_holdout: bool = False,
) -> HoldoutContext:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"请先训练 Q 网络: {MODEL_PATH}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"请先训练 VAE: {EMB_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    q_model = QLatentNet(int(ckpt["z_dim"]), int(ckpt["a_dim"])).to(device)
    q_model.load_state_dict(ckpt["model"])
    q_model.eval()
    vae, _, _ = load_frozen_vae_encoder(device)

    q_retrieval_model: Optional[QRetrievalNet] = None
    if RETRIEVAL_MODEL_PATH.is_file():
        r_ckpt = torch.load(RETRIEVAL_MODEL_PATH, map_location=device, weights_only=False)
        q_retrieval_model = QRetrievalNet(
            int(r_ckpt["z_dim"]), int(r_ckpt["a_dim"])
        ).to(device)
        q_retrieval_model.load_state_dict(r_ckpt["model"])
        q_retrieval_model.eval()

    with open(ART_PATH, "rb") as f:
        artifacts = pickle.load(f)

    train_df = load_simulated_csv(
        Path(artifacts.get("source_csv", DATA_SIMULATED_HIGH_CONTRAST_CSV))
    )
    train_scenario_ids = np.load(SCENARIO_ID_PATH)
    A_std = np.load(A_STD_PATH).astype(np.float32)
    z_train = np.load(EMB_PATH).astype(np.float32)

    _, _, train_scenarios, _ = _split_by_scenario(train_scenario_ids, VAL_RATIO, SEED)
    z_bank, sid_bank = _build_train_scenario_bank(z_train, train_scenario_ids, train_scenarios)
    jammer_type_bank = build_jammer_type_bank(train_df, train_scenario_ids, sid_bank)

    csv_path = holdout_csv or HOLDOUT_CSV_PATH
    if regenerate_holdout or not csv_path.is_file():
        holdout_df = generate_holdout_dataset(n_holdout, actions_per_state, holdout_seed)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        holdout_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        holdout_df = pd.read_csv(csv_path)
        _sids = holdout_df["scenario_id"].unique()
        if len(_sids) > n_holdout:
            # 现有 CSV 场景数多于请求的 n_holdout 时取前 N 个（冒烟/子集评估用）
            holdout_df = holdout_df[holdout_df["scenario_id"].isin(_sids[:n_holdout])]

    train_row_pool = np.where(np.isin(train_scenario_ids, train_scenarios))[0]
    holdout_sids = holdout_df["scenario_id"].unique()

    return HoldoutContext(
        device=device,
        artifacts=artifacts,
        train_df=train_df,
        holdout_df=holdout_df,
        train_scenario_ids=train_scenario_ids,
        train_scenarios=train_scenarios,
        A_std=A_std,
        z_train=z_train,
        z_bank=z_bank,
        sid_bank=sid_bank,
        jammer_type_bank=jammer_type_bank,
        q_model=q_model,
        q_retrieval_model=q_retrieval_model,
        vae=vae,
        train_row_pool=train_row_pool,
        holdout_sids=holdout_sids,
    )


def _select_random(ctx: HoldoutContext, cand_rows: np.ndarray, sid: int, rng: np.random.Generator) -> int:
    if len(cand_rows) == 0:
        raise ValueError("empty candidates")
    return int(rng.choice(cand_rows))


def _select_retrieval_max_r(ctx: HoldoutContext, cand_rows: np.ndarray) -> int:
    train_sinr = ctx.train_df.iloc[cand_rows]["SINR_dB"].to_numpy(dtype=np.float64)
    return int(cand_rows[int(np.argmax(train_sinr))])


def _select_q_only(ctx: HoldoutContext, z_q: np.ndarray, sid: int, rng: np.random.Generator) -> int:
    n = min(Q_ONLY_CANDIDATES, len(ctx.train_row_pool))
    cand_rows = rng.choice(ctx.train_row_pool, size=n, replace=False)
    q_vals = predict_q(ctx.q_model, z_q.reshape(1, -1), ctx.A_std[cand_rows], ctx.device)
    return int(cand_rows[int(np.argmax(q_vals))])


def _select_retrieve_q(ctx: HoldoutContext, z_q: np.ndarray, cand_rows: np.ndarray) -> int:
    q_vals = predict_q(ctx.q_model, z_q.reshape(1, -1), ctx.A_std[cand_rows], ctx.device)
    return int(cand_rows[int(np.argmax(q_vals))])


def build_union_candidate_rows(
    retrieval_cand: np.ndarray,
    train_row_pool: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """检索候选行索引 ∪ 从 train 池随机抽 n_random 行（去重）。"""
    n = min(n_random, len(train_row_pool))
    random_cand = rng.choice(train_row_pool, size=n, replace=False)
    if len(retrieval_cand) == 0:
        return random_cand.astype(np.int64)
    return np.unique(np.concatenate([retrieval_cand, random_cand])).astype(np.int64)


def _select_retrieve_q_union(
    ctx: HoldoutContext,
    z_q: np.ndarray,
    retrieval_cand: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """检索池 + 随机采样并集上 argmax Q；返回 (选中行索引, 并集规模)。"""
    union_cand = build_union_candidate_rows(
        retrieval_cand, ctx.train_row_pool, UNION_RANDOM_CANDIDATES, rng
    )
    row_idx = _select_retrieve_q(ctx, z_q, union_cand)
    return row_idx, len(union_cand)


def _select_retrieve_q_local(
    ctx: HoldoutContext,
    z_q: np.ndarray,
    retrieval_cand: np.ndarray,
    rng: np.random.Generator,
) -> tuple[ScoredAction, int]:
    """检索池 + Top-Q 种子局部扰动，标准 Q argmax。"""
    if len(retrieval_cand) == 0:
        raise ValueError("empty retrieval candidates")
    base_actions = build_retrieval_train_actions(ctx.train_df, ctx.artifacts, retrieval_cand)
    a_mat = stack_a_std(base_actions)
    q_vals = predict_q(ctx.q_model, z_q.reshape(1, -1), a_mat, ctx.device)
    k_seed = min(LOCAL_SEED_TOP, len(q_vals))
    seed_local = np.argsort(-q_vals)[:k_seed]
    pert_actions = build_local_perturbation_actions(
        ctx.train_df,
        ctx.artifacts,
        retrieval_cand,
        seed_local,
        rng,
        n_per_seed=LOCAL_PERTURB_PER_SEED,
    )
    all_actions = base_actions + pert_actions
    best = _argmax_standard_q(ctx, z_q, all_actions)
    return best, len(all_actions)


def _select_retrieve_q_gen(
    ctx: HoldoutContext,
    z_q: np.ndarray,
    retrieval_cand: np.ndarray,
    rng: np.random.Generator,
) -> tuple[ScoredAction, int]:
    """检索池 + 新采样雷达动作，标准 Q argmax。"""
    base_actions = build_retrieval_train_actions(ctx.train_df, ctx.artifacts, retrieval_cand)
    gen_actions = build_generated_radar_actions(ctx.artifacts, GEN_NEW_ACTIONS, rng)
    all_actions = base_actions + gen_actions
    best = _argmax_standard_q(ctx, z_q, all_actions)
    return best, len(all_actions)


def _select_retrieve_q_ctx(
    ctx: HoldoutContext,
    z_q: np.ndarray,
    retrieval_cand: np.ndarray,
    neighbor_sids: np.ndarray,
    sims: np.ndarray,
) -> tuple[ScoredAction, int]:
    """检索池 + 检索条件 Q（Q-only 无法使用 z_ctx）。"""
    if len(retrieval_cand) == 0:
        raise ValueError("empty retrieval candidates")
    z_ctx = _z_ctx_from_neighbor_sids(ctx, neighbor_sids, sims)
    actions = build_retrieval_train_actions(ctx.train_df, ctx.artifacts, retrieval_cand)
    best = _argmax_retrieval_q(ctx, z_q, z_ctx, actions)
    return best, len(actions)


def _select_retrieve_q_ctx_union(
    ctx: HoldoutContext,
    z_q: np.ndarray,
    retrieval_cand: np.ndarray,
    neighbor_sids: np.ndarray,
    sims: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """检索池 ∪ 随机大池 + 检索条件 Q argmax；返回 (行索引, 并集规模)。"""
    if ctx.q_retrieval_model is None:
        raise FileNotFoundError(
            f"retrieve_q_ctx_union 需要检索条件 Q 模型: {RETRIEVAL_MODEL_PATH}"
        )
    union_cand = build_union_candidate_rows(
        retrieval_cand, ctx.train_row_pool, UNION_RANDOM_CANDIDATES, rng
    )
    z_ctx = _z_ctx_from_neighbor_sids(ctx, neighbor_sids, sims)
    q_vals = predict_q_retrieval(
        ctx.q_retrieval_model,
        z_q.reshape(1, -1),
        ctx.A_std[union_cand],
        z_ctx.reshape(1, -1),
        ctx.device,
    )
    return int(union_cand[int(np.argmax(q_vals))]), len(union_cand)


def _select_oracle_k20(block: pd.DataFrame) -> int:
    if _TRUTH_FN is None:
        return int(block["SINR_dB"].values.argmax())
    jrow = block.iloc[0]
    vals = [_TRUTH_FN(jrow, block.iloc[i]) for i in range(len(block))]
    return int(np.argmax(vals))


def _recommend_sinr_for_scenario(
    ctx: HoldoutContext,
    method: str,
    sid: int,
    holdout_seed: int = HOLDOUT_SEED,
    top_k: int = TOP_K_SIMILAR,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """单场景推荐 SINR（dB），供逐场景对比 / 配对胜率计算。"""
    block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
    jammer_row = block.iloc[0]
    _, z_q = _encode_query(ctx, jammer_row)

    if rng is None:
        rng = np.random.default_rng(holdout_seed)

    neighbor_sids, sims = _topk_similar_scenarios(
        z_q,
        ctx.z_bank,
        ctx.sid_bank,
        top_k,
        str(jammer_row["Jammer_Type"]),
        ctx.jammer_type_bank,
    )
    retrieval_cand = _collect_candidate_row_indices(ctx.train_scenario_ids, neighbor_sids)
    sid_rng = np.random.default_rng(holdout_seed + int(sid))

    if method == "random":
        row_idx = _select_random(ctx, retrieval_cand, int(sid), sid_rng)
        return _sinr_on_query(jammer_row, ctx.train_df, row_idx)
    if method == "retrieval_max_r":
        row_idx = _select_retrieval_max_r(ctx, retrieval_cand)
        return _sinr_on_query(jammer_row, ctx.train_df, row_idx)
    if method == "q_only":
        row_idx = _select_q_only(ctx, z_q, int(sid), sid_rng)
        return _sinr_on_query(jammer_row, ctx.train_df, row_idx)
    if method == "retrieve_q":
        row_idx = _select_retrieve_q(ctx, z_q, retrieval_cand)
        return _sinr_on_query(jammer_row, ctx.train_df, row_idx)
    if method == "retrieve_q_union":
        row_idx, _ = _select_retrieve_q_union(ctx, z_q, retrieval_cand, sid_rng)
        return _sinr_on_query(jammer_row, ctx.train_df, row_idx)
    if method == "retrieve_q_local":
        best, _ = _select_retrieve_q_local(ctx, z_q, retrieval_cand, sid_rng)
        return sinr_for_action(jammer_row, best)
    if method == "retrieve_q_gen":
        best, _ = _select_retrieve_q_gen(ctx, z_q, retrieval_cand, sid_rng)
        return sinr_for_action(jammer_row, best)
    if method == "retrieve_q_ctx":
        if ctx.q_retrieval_model is None:
            raise FileNotFoundError(f"retrieve_q_ctx 需要先训练: {RETRIEVAL_MODEL_PATH}")
        best, _ = _select_retrieve_q_ctx(ctx, z_q, retrieval_cand, neighbor_sids, sims)
        return sinr_for_action(jammer_row, best)
    if method == "retrieve_q_ctx_union":
        if ctx.q_retrieval_model is None:
            raise FileNotFoundError(
                f"retrieve_q_ctx_union 需要先训练: {RETRIEVAL_MODEL_PATH}"
            )
        row_idx, _ = _select_retrieve_q_ctx_union(
            ctx, z_q, retrieval_cand, neighbor_sids, sims, sid_rng
        )
        return _sinr_on_query(jammer_row, ctx.train_df, row_idx)
    if method == "oracle_k20":
        local = _select_oracle_k20(block)
        return _holdout_truth(block, local)
    if method == "oracle_wide":
        return _oracle_wide(jammer_row, ORACLE_WIDE_CANDIDATES, rng)
    raise ValueError(method)


def compute_paired_vs_reference(
    ctx: HoldoutContext,
    reference: str = "q_only",
    methods: Optional[List[str]] = None,
    holdout_seed: int = HOLDOUT_SEED,
    top_k: int = TOP_K_SIMILAR,
) -> dict:
    """逐场景配对：各方法相对 reference 的 SINR 胜率。"""
    methods = list(methods or METHODS)
    challengers = [m for m in methods if m != reference]
    rng = np.random.default_rng(holdout_seed)
    n = len(ctx.holdout_sids)
    wins = {m: 0 for m in challengers}
    ties = {m: 0 for m in challengers}

    for sid in ctx.holdout_sids:
        ref_sinr = _recommend_sinr_for_scenario(
            ctx, reference, int(sid), holdout_seed, top_k, rng
        )
        for m in challengers:
            sinr_m = _recommend_sinr_for_scenario(
                ctx, m, int(sid), holdout_seed, top_k, rng
            )
            if sinr_m > ref_sinr + 1e-6:
                wins[m] += 1
            elif abs(sinr_m - ref_sinr) <= 1e-6:
                ties[m] += 1

    return {
        "reference": reference,
        "n_scenarios": n,
        "results": {
            m: {
                "wins": wins[m],
                "ties": ties[m],
                "losses": n - wins[m] - ties[m],
                "win_rate": wins[m] / n if n else float("nan"),
            }
            for m in challengers
        },
    }


def evaluate_method(
    ctx: HoldoutContext,
    method: str,
    top_k: int = TOP_K_SIMILAR,
    holdout_seed: int = HOLDOUT_SEED,
    verbose: bool = False,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"未知方法: {method}, 可选 {METHODS}")

    rng = np.random.default_rng(holdout_seed)
    regrets: List[float] = []
    regrets_wide: List[float] = []
    rec_sinrs: List[float] = []
    oracle_k20_list: List[float] = []
    oracle_wide_list: List[float] = []
    sim_top1: List[float] = []
    pool_sizes: List[int] = []

    for si, sid in enumerate(ctx.holdout_sids):
        block = ctx.holdout_df[ctx.holdout_df["scenario_id"] == sid]
        jammer_row = block.iloc[0]
        _, z_q = _encode_query(ctx, jammer_row)

        oracle_k20 = _oracle_k20(block)
        oracle_wide = _oracle_wide(jammer_row, ORACLE_WIDE_CANDIDATES, rng)
        oracle_k20_list.append(oracle_k20)
        oracle_wide_list.append(oracle_wide)

        neighbor_sids, sims = _topk_similar_scenarios(
            z_q,
            ctx.z_bank,
            ctx.sid_bank,
            top_k,
            str(jammer_row["Jammer_Type"]),
            ctx.jammer_type_bank,
        )
        retrieval_cand = _collect_candidate_row_indices(ctx.train_scenario_ids, neighbor_sids)
        sim_top1.append(float(sims[0]) if len(sims) else float("nan"))

        sid_rng = np.random.default_rng(holdout_seed + int(sid))

        if method == "random":
            row_idx = _select_random(ctx, retrieval_cand, int(sid), sid_rng)
            r_rec = _sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "retrieval_max_r":
            row_idx = _select_retrieval_max_r(ctx, retrieval_cand)
            r_rec = _sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "q_only":
            row_idx = _select_q_only(ctx, z_q, int(sid), sid_rng)
            r_rec = _sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "retrieve_q":
            pool_sizes.append(len(retrieval_cand))
            row_idx = _select_retrieve_q(ctx, z_q, retrieval_cand)
            r_rec = _sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "retrieve_q_union":
            row_idx, pool_n = _select_retrieve_q_union(ctx, z_q, retrieval_cand, sid_rng)
            pool_sizes.append(pool_n)
            r_rec = _sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "retrieve_q_local":
            best, pool_n = _select_retrieve_q_local(ctx, z_q, retrieval_cand, sid_rng)
            pool_sizes.append(pool_n)
            r_rec = sinr_for_action(jammer_row, best)
        elif method == "retrieve_q_gen":
            best, pool_n = _select_retrieve_q_gen(ctx, z_q, retrieval_cand, sid_rng)
            pool_sizes.append(pool_n)
            r_rec = sinr_for_action(jammer_row, best)
        elif method == "retrieve_q_ctx":
            if ctx.q_retrieval_model is None:
                raise FileNotFoundError(
                    f"retrieve_q_ctx 需要先训练: {RETRIEVAL_MODEL_PATH}"
                )
            best, pool_n = _select_retrieve_q_ctx(
                ctx, z_q, retrieval_cand, neighbor_sids, sims
            )
            pool_sizes.append(pool_n)
            r_rec = sinr_for_action(jammer_row, best)
        elif method == "retrieve_q_ctx_union":
            if ctx.q_retrieval_model is None:
                raise FileNotFoundError(
                    f"retrieve_q_ctx_union 需要先训练: {RETRIEVAL_MODEL_PATH}"
                )
            row_idx, pool_n = _select_retrieve_q_ctx_union(
                ctx, z_q, retrieval_cand, neighbor_sids, sims, sid_rng
            )
            pool_sizes.append(pool_n)
            r_rec = _sinr_on_query(jammer_row, ctx.train_df, row_idx)
        elif method == "oracle_k20":
            local = _select_oracle_k20(block)
            r_rec = _holdout_truth(block, local)
        elif method == "oracle_wide":
            r_rec = oracle_wide
        else:
            raise ValueError(method)

        rec_sinrs.append(r_rec)
        regrets.append(oracle_k20 - r_rec)
        regrets_wide.append(oracle_wide - r_rec)

        if verbose and (si < 2 or si % 100 == 0):
            print(
                f"  [{method}] {si+1}/{len(ctx.holdout_sids)} sid={sid} "
                f"rec={r_rec:.2f} oracle_k20={oracle_k20:.2f} regret={regrets[-1]:.2f}"
            )

    regret_arr = np.array(regrets, dtype=np.float64)
    regret_wide_arr = np.array(regrets_wide, dtype=np.float64)
    rec_arr = np.array(rec_sinrs, dtype=np.float64)

    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "n_evaluated": int(len(regrets)),
        "mean_recommended_sinr_db": float(rec_arr.mean()),
        "mean_oracle_k20_db": float(np.mean(oracle_k20_list)),
        "mean_oracle_wide_db": float(np.mean(oracle_wide_list)),
        "mean_regret_vs_k20_db": float(regret_arr.mean()),
        "median_regret_vs_k20_db": float(np.median(regret_arr)),
        "regret_p25_db": float(np.percentile(regret_arr, 25)),
        "regret_p75_db": float(np.percentile(regret_arr, 75)),
        "regret_p90_db": float(np.percentile(regret_arr, 90)),
        "mean_regret_vs_wide_db": float(regret_wide_arr.mean()),
        "frac_zero_regret_vs_k20": float((regret_arr <= 1e-4).mean()),
        "mean_top1_cosine_sim": float(np.nanmean(sim_top1)),
        "top_k_retrieve": top_k,
        "q_only_candidates": Q_ONLY_CANDIDATES,
        "union_random_candidates": UNION_RANDOM_CANDIDATES,
        "mean_candidate_pool_size": float(np.mean(pool_sizes)) if pool_sizes else float("nan"),
        "oracle_wide_candidates": ORACLE_WIDE_CANDIDATES,
        "per_scenario_recommended_sinr_db": rec_arr.tolist(),
        "per_scenario_regret_vs_k20_db": regret_arr.tolist(),
    }


def run_all_baselines(
    methods: Optional[List[str]] = None,
    n_holdout: int = N_HOLDOUT_SCENARIOS,
    actions_per_state: int = ACTIONS_PER_STATE,
    top_k: int = TOP_K_SIMILAR,
    holdout_seed: int = HOLDOUT_SEED,
    regenerate_holdout: bool = False,
    verbose: bool = True,
) -> dict:
    methods = list(methods or METHODS)
    ctx = load_holdout_context(
        n_holdout=n_holdout,
        actions_per_state=actions_per_state,
        holdout_seed=holdout_seed,
        regenerate_holdout=regenerate_holdout,
    )

    if verbose:
        print(
            f"留出集: {len(ctx.holdout_sids)} 场景, K={actions_per_state}, "
            f"检索库 {len(ctx.sid_bank)} train 场景"
        )
        print(f"评估方法: {methods}\n")

    results = []
    per_scenario_detail: dict[str, dict] = {}
    for m in methods:
        if verbose:
            print(f"--- {m}: {METHOD_LABELS[m]} ---")
        row = evaluate_method(ctx, m, top_k=top_k, holdout_seed=holdout_seed, verbose=verbose)
        per_scenario_detail[m] = {
            "recommended_sinr_db": row.pop("per_scenario_recommended_sinr_db"),
            "regret_vs_k20_db": row.pop("per_scenario_regret_vs_k20_db"),
        }
        results.append(row)

    summary = {
        "n_holdout_scenarios": n_holdout,
        "actions_per_holdout_scenario": actions_per_state,
        "holdout_seed": holdout_seed,
        "holdout_csv": str(HOLDOUT_CSV_PATH),
        "reference_regret": (
            "candidate_reference_k = max SINR among holdout K random actions; "
            "negative values are possible for methods searching a larger pool and do not "
            "mean outperforming a global oracle"
        ),
        "reference_candidate_count": int(ACTIONS_PER_STATE),
        "methods": results,
    }

    if verbose:
        print("\n--- 配对胜率（相对 Q-only，逐场景 SINR 更高即胜）---")
    paired = compute_paired_vs_reference(
        ctx, reference="q_only", methods=methods, holdout_seed=holdout_seed, top_k=top_k
    )
    summary["paired_vs_q_only"] = paired
    if verbose:
        for m, st in paired["results"].items():
            if m in ("oracle_k20", "oracle_wide"):
                continue
            print(f"  {m:<18} 胜 {st['wins']:3d}/{paired['n_scenarios']}  "
                  f"({100*st['win_rate']:.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PER_SCENARIO_JSON, "w", encoding="utf-8") as f:
        json.dump(per_scenario_detail, f, ensure_ascii=False, indent=2)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "method",
        "label",
        "mean_recommended_sinr_db",
        "mean_oracle_k20_db",
        "mean_regret_vs_k20_db",
        "median_regret_vs_k20_db",
        "mean_regret_vs_wide_db",
        "frac_zero_regret_vs_k20",
        "mean_top1_cosine_sim",
    ]
    with open(SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in results:
            w.writerow(row)

    if verbose:
        print("\n=== 留出集 Baseline 对比汇总 (regret vs Oracle-K20) ===")
        print(
            f"{'方法':<16} {'推荐SINR':>10} {'Oracle-K20':>11} "
            f"{'regret':>8} {'med_reg':>8} {'Hit@K20':>8}"
        )
        for r in results:
            print(
                f"{r['method']:<16} "
                f"{r['mean_recommended_sinr_db']:>10.2f} "
                f"{r['mean_oracle_k20_db']:>11.2f} "
                f"{r['mean_regret_vs_k20_db']:>8.3f} "
                f"{r['median_regret_vs_k20_db']:>8.3f} "
                f"{r['frac_zero_regret_vs_k20']:>8.1%}"
            )
        print(f"\n已保存: {SUMMARY_JSON}")
        print(f"已保存: {SUMMARY_CSV}")

    return summary


def holdout_eval_main(**kwargs) -> dict:
    """main.py 留出评估：默认跑 REPORT_METHODS，写入 summary.json。"""
    kwargs.setdefault("methods", list(REPORT_METHODS))
    kwargs.setdefault("verbose", True)
    summary = run_all_baselines(**kwargs)
    m = next(row for row in summary["methods"] if row["method"] == OUR_METHOD)
    legacy = {
        "n_holdout_scenarios": summary["n_holdout_scenarios"],
        "actions_per_holdout_scenario": summary["actions_per_holdout_scenario"],
        "top_k_retrieve": m.get("top_k_retrieve", TOP_K_SIMILAR),
        "holdout_seed": summary["holdout_seed"],
        "n_evaluated": m["n_evaluated"],
        "oracle_hit_at_1_sinr": m["frac_zero_regret_vs_k20"],
        "mean_regret_sinr_db": m["mean_regret_vs_k20_db"],
        "median_regret_sinr_db": m["median_regret_vs_k20_db"],
        "mean_oracle_sinr_db": m["mean_oracle_k20_db"],
        "mean_q_recommended_sinr_db": m["mean_recommended_sinr_db"],
        "mean_top1_cosine_sim": m["mean_top1_cosine_sim"],
        "holdout_csv": summary["holdout_csv"],
        "description": m["label"],
    }
    legacy_path = OUT_Q_VALUE / "holdout_eval_metrics.json"
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False, indent=2)
    return legacy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="留出集 baseline 对比评估")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(REPORT_METHODS),
        choices=list(METHODS),
        help="要评估的方法（默认 REPORT_METHODS）",
    )
    parser.add_argument("--n-holdout", type=int, default=N_HOLDOUT_SCENARIOS)
    parser.add_argument("--regenerate-holdout", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_all_baselines(
        methods=args.methods,
        n_holdout=args.n_holdout,
        regenerate_holdout=args.regenerate_holdout,
        verbose=not args.quiet,
    )
