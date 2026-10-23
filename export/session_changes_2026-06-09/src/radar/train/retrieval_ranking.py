# -*- coding: utf-8 -*-
"""
检索候选池 ranking loss / 评估（与 holdout Retrieve+Q 推理对齐）。

训练场景 s 作查询时：从 train 库同 Jammer_Type 内 Top-K 相似场景（排除 s）取 ~100 个 A，
用仿真公式算 R(s, a)，对 Q 做 listwise softmax，使 argmax Q = argmax R。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from radar.data.build_dataset import load_simulated_csv
from radar.data.high_contrast_sim import sinr_from_csv_row
from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV, DATA_V2_PROCESSED


def _l2n(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def build_jammer_type_bank(
    train_df: pd.DataFrame,
    scenario_ids: np.ndarray,
    sid_bank: np.ndarray,
) -> np.ndarray:
    """与 sid_bank 对齐：每个训练场景 scenario_id 对应的 Jammer_Type。"""
    types: list[str] = []
    for sid in sid_bank:
        rows = np.where(scenario_ids == sid)[0]
        types.append(str(train_df.iloc[int(rows[0])]["Jammer_Type"]))
    return np.array(types, dtype=object)


def topk_similar_scenarios(
    z_query: np.ndarray,
    z_bank: np.ndarray,
    sid_bank: np.ndarray,
    k: int,
    query_jammer_type: str,
    jammer_type_bank: np.ndarray,
    exclude_sid: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (neighbor_sids, cosine_sims)，按相似度降序；仅在同 Jammer_Type 内检索。"""
    if len(jammer_type_bank) != len(sid_bank):
        raise ValueError("jammer_type_bank 长度须与 sid_bank 一致")
    z_bank_n = _l2n(z_bank)
    q_n = z_query / (np.linalg.norm(z_query) + 1e-8)
    sims = z_bank_n @ q_n
    valid = jammer_type_bank == str(query_jammer_type)
    if exclude_sid is not None:
        valid &= sid_bank != exclude_sid
    if not valid.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    idx_valid = np.where(valid)[0]
    sims_v = sims[idx_valid]
    kk = min(k, len(sims_v))
    top_local = np.argpartition(-sims_v, kth=kk - 1)[:kk]
    top_local = top_local[np.argsort(-sims_v[top_local])]
    top_global = idx_valid[top_local]
    return sid_bank[top_global], sims[top_global].astype(np.float32)


def collect_candidate_row_indices(
    scenario_ids: np.ndarray,
    neighbor_sids: np.ndarray,
) -> np.ndarray:
    parts = [np.where(scenario_ids == sid)[0] for sid in neighbor_sids]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts)


def listwise_ranking_loss(q_scores: torch.Tensor, oracle_idx: int) -> torch.Tensor:
    """Listwise softmax：真实最优动作的 log-prob 取负。"""
    log_probs = torch.log_softmax(q_scores, dim=0)
    return -log_probs[oracle_idx]


@dataclass
class RetrievalRankingContext:
    train_df: pd.DataFrame
    scenario_ids: np.ndarray
    a_std: np.ndarray
    z: np.ndarray
    train_scenarios: np.ndarray
    z_bank: np.ndarray
    sid_bank: np.ndarray
    jammer_type_bank: np.ndarray
    sid_to_rows: dict[int, np.ndarray]
    jammer_by_sid: dict[int, pd.Series]

    @classmethod
    def build(
        cls,
        z: np.ndarray,
        scenario_ids: np.ndarray,
        a_std: np.ndarray,
        train_scenarios: np.ndarray,
        csv_path: Optional[Path] = None,
    ) -> "RetrievalRankingContext":
        art_path = DATA_V2_PROCESSED / "artifacts.pkl"
        with open(art_path, "rb") as f:
            artifacts = pickle.load(f)
        src = csv_path or Path(
            artifacts.get("source_csv", DATA_SIMULATED_HIGH_CONTRAST_CSV)
        )
        train_df = load_simulated_csv(src)

        sid_to_rows: dict[int, np.ndarray] = {}
        jammer_by_sid: dict[int, pd.Series] = {}
        for sid in np.unique(scenario_ids):
            rows = np.where(scenario_ids == sid)[0]
            sid_to_rows[int(sid)] = rows
            jammer_by_sid[int(sid)] = train_df.iloc[int(rows[0])]

        z_list, sid_list = [], []
        for sid in train_scenarios:
            rows = sid_to_rows.get(int(sid))
            if rows is None or len(rows) == 0:
                continue
            z_list.append(z[int(rows[0])])
            sid_list.append(int(sid))
        z_bank = np.stack(z_list, axis=0).astype(np.float32)
        sid_bank = np.array(sid_list, dtype=np.int64)
        jammer_type_bank = build_jammer_type_bank(train_df, scenario_ids, sid_bank)

        return cls(
            train_df=train_df,
            scenario_ids=scenario_ids,
            a_std=a_std,
            z=z,
            train_scenarios=train_scenarios,
            z_bank=z_bank,
            sid_bank=sid_bank,
            jammer_type_bank=jammer_type_bank,
            sid_to_rows=sid_to_rows,
            jammer_by_sid=jammer_by_sid,
        )

    def candidate_rows_for_query(
        self,
        query_sid: int,
        top_k: int,
        exclude_self: bool,
    ) -> np.ndarray:
        rows = self.sid_to_rows.get(int(query_sid))
        if rows is None or len(rows) == 0:
            return np.array([], dtype=np.int64)
        z_q = self.z[int(rows[0])]
        query_type = str(self.jammer_by_sid[int(query_sid)]["Jammer_Type"])
        exclude = int(query_sid) if exclude_self else None
        neighbor_sids, _ = topk_similar_scenarios(
            z_q,
            self.z_bank,
            self.sid_bank,
            top_k,
            query_jammer_type=query_type,
            jammer_type_bank=self.jammer_type_bank,
            exclude_sid=exclude,
        )
        return collect_candidate_row_indices(self.scenario_ids, neighbor_sids)

    def true_sinr_on_query(self, query_sid: int, cand_rows: np.ndarray) -> np.ndarray:
        jammer = self.jammer_by_sid[int(query_sid)]
        out = np.empty(len(cand_rows), dtype=np.float64)
        for i, row_idx in enumerate(cand_rows):
            out[i] = sinr_from_csv_row(jammer, self.train_df.iloc[int(row_idx)])
        return out


def ranking_loss_for_scenario(
    model: nn.Module,
    ctx: RetrievalRankingContext,
    query_sid: int,
    device: torch.device,
    top_k: int,
    exclude_self: bool,
) -> Optional[torch.Tensor]:
    """单场景 listwise ranking loss；无候选时返回 None。"""
    cand_rows = ctx.candidate_rows_for_query(query_sid, top_k, exclude_self)
    if len(cand_rows) == 0:
        return None

    rows = ctx.sid_to_rows[int(query_sid)]
    z_q = ctx.z[int(rows[0])]
    r_true = ctx.true_sinr_on_query(query_sid, cand_rows)
    oracle_idx = int(np.argmax(r_true))

    a_cand = torch.from_numpy(ctx.a_std[cand_rows]).to(device)
    z_t = torch.from_numpy(z_q).to(device).unsqueeze(0).expand(len(cand_rows), -1)
    q_scores = model(z_t, a_cand).squeeze(-1)
    return listwise_ranking_loss(q_scores, oracle_idx)


def ranking_loss_batch(
    model: nn.Module,
    ctx: RetrievalRankingContext,
    query_sids: np.ndarray,
    device: torch.device,
    top_k: int,
    exclude_self: bool,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for sid in query_sids:
        loss = ranking_loss_for_scenario(
            model, ctx, int(sid), device, top_k, exclude_self
        )
        if loss is not None:
            losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


@torch.no_grad()
def eval_retrieval_pool_selection(
    model: nn.Module,
    ctx: RetrievalRankingContext,
    eval_scenarios: np.ndarray,
    device: torch.device,
    top_k: int,
    exclude_self: bool,
) -> dict:
    """
    检索候选池 argmax Q vs oracle argmax R（与 holdout retrieve_q 一致）。
    train 场景作 query 时 exclude_self=True；val/holdout 场景 exclude_self=False。
    """
    model.eval()
    hits: list[bool] = []
    regrets_db: list[float] = []
    rank_losses: list[float] = []
    pool_sizes: list[int] = []

    for sid in eval_scenarios:
        cand_rows = ctx.candidate_rows_for_query(int(sid), top_k, exclude_self)
        if len(cand_rows) == 0:
            continue
        pool_sizes.append(len(cand_rows))

        rows = ctx.sid_to_rows[int(sid)]
        z_q = ctx.z[int(rows[0])]
        r_true = ctx.true_sinr_on_query(int(sid), cand_rows)
        oracle_idx = int(np.argmax(r_true))

        a_cand = torch.from_numpy(ctx.a_std[cand_rows]).to(device)
        z_t = torch.from_numpy(z_q).to(device).unsqueeze(0).expand(len(cand_rows), -1)
        q_scores = model(z_t, a_cand).squeeze(-1)
        pred_idx = int(torch.argmax(q_scores).item())

        hits.append(pred_idx == oracle_idx)
        regrets_db.append(float(r_true[oracle_idx] - r_true[pred_idx]))
        rank_losses.append(float(listwise_ranking_loss(q_scores, oracle_idx).item()))

    if not hits:
        nan = float("nan")
        return {
            "n_eval_scenarios": 0,
            "candidates_per_scenario_mean": 0.0,
            "oracle_hit_at_1": nan,
            "mean_regret_sinr_db": nan,
            "median_regret_sinr_db": nan,
            "frac_zero_regret": nan,
            "mean_rank_nll": nan,
        }

    regret_arr = np.array(regrets_db, dtype=np.float64)
    return {
        "n_eval_scenarios": int(len(hits)),
        "candidates_per_scenario_mean": float(np.mean(pool_sizes)),
        "oracle_hit_at_1": float(np.mean(hits)),
        "mean_regret_sinr_db": float(regret_arr.mean()),
        "median_regret_sinr_db": float(np.median(regret_arr)),
        "frac_zero_regret": float((regret_arr <= 1e-6).mean()),
        "mean_rank_nll": float(np.mean(rank_losses)),
    }
