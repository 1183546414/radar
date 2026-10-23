# -*- coding: utf-8 -*-
"""
训练检索条件 Q 网络 Q(z_query, z_ctx, A) -> R（方向 B）。

z_ctx = 同类型 Top-K 相似 train 场景潜向量相似度加权均值。
早停：验证集检索池 Oracle Hit@1。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from radar.paths import DATA_V2_PROCESSED, OUT_Q_VALUE
from radar.train.q_retrieval_net import (
    QRetrievalNet,
    aggregate_neighbor_z,
    warm_start_from_q_latent,
)
from radar.train.retrieval_ranking import (
    RetrievalRankingContext,
    build_jammer_type_bank,
    listwise_ranking_loss,
    topk_similar_scenarios,
)
from radar.train.train_q_latent import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    MODEL_PATH as Q_LATENT_MODEL_PATH,
    PATIENCE,
    RANK_LAMBDA,
    RANK_SCENARIOS_PER_STEP,
    RANK_STEPS_PER_EPOCH,
    SCENARIO_ID_PATH,
    SEED,
    TOP_K_RETRIEVE,
    VAL_RANK_MAX_SCENARIOS,
    VAL_RATIO,
    WEIGHT_DECAY,
    _load_or_encode_latent,
    _split_by_scenario,
)

RETRIEVAL_OUT_DIR = OUT_Q_VALUE / "latent_q_retrieval"
RETRIEVAL_MODEL_PATH = RETRIEVAL_OUT_DIR / "q_retrieval_best.pt"
RETRIEVAL_METRICS_PATH = RETRIEVAL_OUT_DIR / "metrics.json"
Z_CTX_CACHE = RETRIEVAL_OUT_DIR / "z_ctx_per_row_same_type_weighted.npy"


def _build_scenario_z_bank(
    z: np.ndarray,
    scenario_ids: np.ndarray,
    scenarios: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_list, sid_list = [], []
    for sid in scenarios:
        idx = np.where(scenario_ids == sid)[0]
        if len(idx) == 0:
            continue
        z_list.append(z[int(idx[0])])
        sid_list.append(int(sid))
    return np.stack(z_list, axis=0).astype(np.float32), np.array(sid_list, dtype=np.int64)


def _z_ctx_for_scenario(
    z_q: np.ndarray,
    z_bank: np.ndarray,
    sid_bank: np.ndarray,
    jammer_type_bank: np.ndarray,
    scenario_ids: np.ndarray,
    z: np.ndarray,
    query_sid: int,
    query_type: str,
    train_sid_set: set[int],
    top_k: int,
) -> np.ndarray:
    exclude = int(query_sid) if int(query_sid) in train_sid_set else None
    neighbor_sids, sims = topk_similar_scenarios(
        z_q,
        z_bank,
        sid_bank,
        top_k,
        query_jammer_type=query_type,
        jammer_type_bank=jammer_type_bank,
        exclude_sid=exclude,
    )
    if len(neighbor_sids) == 0:
        return z_q.astype(np.float32)
    neighbor_z = []
    for ns in neighbor_sids:
        nrows = np.where(scenario_ids == ns)[0]
        if len(nrows):
            neighbor_z.append(z[int(nrows[0])])
    if not neighbor_z:
        return z_q.astype(np.float32)
    return aggregate_neighbor_z(np.stack(neighbor_z), sims)


def precompute_z_ctx_per_row(
    z: np.ndarray,
    scenario_ids: np.ndarray,
    train_scenarios: np.ndarray,
    train_df: pd.DataFrame,
    top_k: int = TOP_K_RETRIEVE,
) -> np.ndarray:
    """全量场景 z_ctx：train 排除自身，val 从 train 库同类型检索。"""
    z_bank, sid_bank = _build_scenario_z_bank(z, scenario_ids, train_scenarios)
    jammer_type_bank = build_jammer_type_bank(train_df, scenario_ids, sid_bank)
    train_sid_set = {int(s) for s in train_scenarios}

    jammer_type_by_sid: dict[int, str] = {}
    for sid in np.unique(scenario_ids):
        rows = np.where(scenario_ids == sid)[0]
        if len(rows):
            jammer_type_by_sid[int(sid)] = str(train_df.iloc[int(rows[0])]["Jammer_Type"])

    sid_to_ctx: dict[int, np.ndarray] = {}
    for sid in np.unique(scenario_ids):
        sid = int(sid)
        rows = np.where(scenario_ids == sid)[0]
        if len(rows) == 0:
            continue
        z_q = z[int(rows[0])]
        sid_to_ctx[sid] = _z_ctx_for_scenario(
            z_q,
            z_bank,
            sid_bank,
            jammer_type_bank,
            scenario_ids,
            z,
            sid,
            jammer_type_by_sid[sid],
            train_sid_set,
            top_k,
        )

    z_ctx = np.zeros_like(z, dtype=np.float32)
    for i, sid in enumerate(scenario_ids):
        z_ctx[i] = sid_to_ctx.get(int(sid), z[i])
    return z_ctx


def _load_or_compute_z_ctx(
    z: np.ndarray,
    scenario_ids: np.ndarray,
    train_scenarios: np.ndarray,
) -> np.ndarray:
    import pickle

    from radar.data.build_dataset import load_simulated_csv
    from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV

    RETRIEVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if Z_CTX_CACHE.exists():
        cached = np.load(Z_CTX_CACHE)
        if cached.shape == z.shape:
            print(f"加载 z_ctx 缓存: {Z_CTX_CACHE} {cached.shape}")
            return cached
    print("预计算每行 z_ctx（同类型 Top-K 相似度加权）...")
    with open(DATA_V2_PROCESSED / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    train_df = load_simulated_csv(
        Path(artifacts.get("source_csv", DATA_SIMULATED_HIGH_CONTRAST_CSV))
    )
    z_ctx = precompute_z_ctx_per_row(z, scenario_ids, train_scenarios, train_df)
    np.save(Z_CTX_CACHE, z_ctx)
    return z_ctx


def _ranking_loss_batch_retrieval(
    model: QRetrievalNet,
    ctx: RetrievalRankingContext,
    query_sids: np.ndarray,
    device: torch.device,
    top_k: int,
) -> torch.Tensor:
    losses = []
    for sid in query_sids:
        cand_rows = ctx.candidate_rows_for_query(int(sid), top_k, exclude_self=True)
        if len(cand_rows) == 0:
            continue
        rows = ctx.sid_to_rows[int(sid)]
        z_q = ctx.z[int(rows[0])]
        query_type = str(ctx.jammer_by_sid[int(sid)]["Jammer_Type"])
        neighbor_sids, sims = topk_similar_scenarios(
            z_q,
            ctx.z_bank,
            ctx.sid_bank,
            top_k,
            query_jammer_type=query_type,
            jammer_type_bank=ctx.jammer_type_bank,
            exclude_sid=int(sid),
        )
        if len(neighbor_sids) == 0:
            z_ctx = z_q
        else:
            neighbor_z = []
            for ns in neighbor_sids:
                nr = ctx.sid_to_rows.get(int(ns))
                if nr is not None and len(nr):
                    neighbor_z.append(ctx.z[int(nr[0])])
            z_ctx = (
                aggregate_neighbor_z(np.stack(neighbor_z), sims)
                if neighbor_z
                else z_q
            )

        r_true = ctx.true_sinr_on_query(int(sid), cand_rows)
        oracle_idx = int(np.argmax(r_true))
        a_cand = torch.from_numpy(ctx.a_std[cand_rows]).to(device)
        n = len(cand_rows)
        zq_t = torch.from_numpy(z_q).to(device).unsqueeze(0).expand(n, -1)
        zc_t = torch.from_numpy(z_ctx).to(device).unsqueeze(0).expand(n, -1)
        q_scores = model(zq_t, a_cand, zc_t).squeeze(-1)
        log_probs = torch.log_softmax(q_scores, dim=0)
        losses.append(-log_probs[oracle_idx])
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


@torch.no_grad()
def eval_retrieval_pool_selection_ctx(
    model: QRetrievalNet,
    ctx: RetrievalRankingContext,
    eval_scenarios: np.ndarray,
    device: torch.device,
    top_k: int,
    exclude_self: bool,
) -> dict:
    """检索候选池 argmax QRetrievalNet(z, z_ctx, A) vs oracle（对齐 retrieve_q_ctx）。"""
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
        query_type = str(ctx.jammer_by_sid[int(sid)]["Jammer_Type"])
        exclude = int(sid) if exclude_self else None
        neighbor_sids, sims = topk_similar_scenarios(
            z_q,
            ctx.z_bank,
            ctx.sid_bank,
            top_k,
            query_jammer_type=query_type,
            jammer_type_bank=ctx.jammer_type_bank,
            exclude_sid=exclude,
        )
        if len(neighbor_sids) == 0:
            z_ctx = z_q.astype(np.float32)
        else:
            neighbor_z = []
            for ns in neighbor_sids:
                nr = ctx.sid_to_rows.get(int(ns))
                if nr is not None and len(nr):
                    neighbor_z.append(ctx.z[int(nr[0])])
            z_ctx = (
                aggregate_neighbor_z(np.stack(neighbor_z), sims)
                if neighbor_z
                else z_q.astype(np.float32)
            )

        r_true = ctx.true_sinr_on_query(int(sid), cand_rows)
        oracle_idx = int(np.argmax(r_true))

        a_cand = torch.from_numpy(ctx.a_std[cand_rows]).to(device)
        n = len(cand_rows)
        zq_t = torch.from_numpy(z_q).to(device).unsqueeze(0).expand(n, -1)
        zc_t = torch.from_numpy(z_ctx).to(device).unsqueeze(0).expand(n, -1)
        q_scores = model(zq_t, a_cand, zc_t).squeeze(-1)
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


def train_q_retrieval_main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    A_std = np.load(DATA_V2_PROCESSED / "A_std.npy").astype(np.float32)
    R_std = np.load(DATA_V2_PROCESSED / "R_std.npy").reshape(-1).astype(np.float32)
    scenario_ids = np.load(SCENARIO_ID_PATH)

    z = _load_or_encode_latent(device)
    z_dim, a_dim = z.shape[1], A_std.shape[1]

    tr_idx, val_idx, tr_scenarios, val_scenarios = _split_by_scenario(
        scenario_ids, VAL_RATIO, SEED
    )
    z_ctx_all = _load_or_compute_z_ctx(z, scenario_ids, tr_scenarios)

    rank_ctx = RetrievalRankingContext.build(
        z=z, scenario_ids=scenario_ids, a_std=A_std, train_scenarios=tr_scenarios
    )
    rank_rng = np.random.default_rng(SEED)
    val_rank_scenarios = val_scenarios[
        : min(VAL_RANK_MAX_SCENARIOS, len(val_scenarios))
    ]

    z_tr = z[tr_idx]
    z_ctx_tr = z_ctx_all[tr_idx]
    a_tr = A_std[tr_idx]
    r_tr = R_std[tr_idx]

    z_val = z[val_idx]
    z_ctx_val = z_ctx_all[val_idx]
    a_val = A_std[val_idx]
    r_val = R_std[val_idx]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_tr),
            torch.from_numpy(z_ctx_tr),
            torch.from_numpy(a_tr),
            torch.from_numpy(r_tr),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = QRetrievalNet(z_dim, a_dim).to(device)
    if Q_LATENT_MODEL_PATH.is_file():
        warm_start_from_q_latent(model, Q_LATENT_MODEL_PATH, device)
        print(f"QLatentNet 热启动: {Q_LATENT_MODEL_PATH}")
    else:
        print(f"未找到 QLatentNet 权重，从头训练: {Q_LATENT_MODEL_PATH}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    z_val_t = torch.from_numpy(z_val).to(device)
    z_ctx_val_t = torch.from_numpy(z_ctx_val).to(device)
    a_val_t = torch.from_numpy(a_val).to(device)
    r_val_t = torch.from_numpy(r_val).to(device)

    best_hit1 = -1.0
    wait = 0
    history = []

    print(
        f"训练 QRetrievalNet: Top-{TOP_K_RETRIEVE} 加权 ctx, "
        f"MSE + λ={RANK_LAMBDA} ranking, 早停=检索池 Hit@1"
    )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_mse_losses: list[float] = []
        train_rank_losses: list[float] = []

        for zq, zc, ab, rb in train_loader:
            zq, zc, ab, rb = zq.to(device), zc.to(device), ab.to(device), rb.to(device)
            pred = model(zq, ab, zc).squeeze(-1)
            loss_mse = loss_fn(pred, rb)
            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()
            train_mse_losses.append(loss_mse.item())

        for _ in range(RANK_STEPS_PER_EPOCH):
            batch_sids = rank_rng.choice(
                tr_scenarios,
                size=min(RANK_SCENARIOS_PER_STEP, len(tr_scenarios)),
                replace=False,
            )
            loss_rank = _ranking_loss_batch_retrieval(
                model, rank_ctx, batch_sids, device, TOP_K_RETRIEVE
            )
            optimizer.zero_grad()
            (RANK_LAMBDA * loss_rank).backward()
            optimizer.step()
            train_rank_losses.append(float(loss_rank.item()))

        model.eval()
        with torch.no_grad():
            pred_val = model(z_val_t, a_val_t, z_ctx_val_t).squeeze(-1)
            val_mse = loss_fn(pred_val, r_val_t).item()
            val_rank_nll = float(
                _ranking_loss_batch_retrieval(
                    model, rank_ctx, val_rank_scenarios, device, TOP_K_RETRIEVE
                ).item()
            )
            val_retrieval = eval_retrieval_pool_selection_ctx(
                model,
                rank_ctx,
                val_rank_scenarios,
                device,
                TOP_K_RETRIEVE,
                exclude_self=False,
            )
            val_hit1 = float(val_retrieval["oracle_hit_at_1"])

        tr_mse = float(np.mean(train_mse_losses))
        tr_rank = float(np.mean(train_rank_losses)) if train_rank_losses else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_mse": tr_mse,
                "train_rank_nll": tr_rank,
                "val_mse": val_mse,
                "val_rank_nll": val_rank_nll,
                "val_retrieval_hit_at_1": val_hit1,
                "val_retrieval_regret_db": val_retrieval.get("mean_regret_sinr_db"),
            }
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | train_mse={tr_mse:.5f} rank={tr_rank:.4f} | "
                f"val_mse={val_mse:.5f} val_rank={val_rank_nll:.4f} "
                f"val_hit@1={val_hit1:.3f}"
            )

        if val_hit1 > best_hit1 + 1e-6:
            best_hit1 = val_hit1
            wait = 0
            RETRIEVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "z_dim": z_dim,
                    "a_dim": a_dim,
                    "top_k_retrieve": TOP_K_RETRIEVE,
                    "warm_start": Q_LATENT_MODEL_PATH.is_file(),
                },
                RETRIEVAL_MODEL_PATH,
            )
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"早停 @ epoch {epoch} (val_retrieval_hit@1={val_hit1:.3f})")
                break

    ckpt = torch.load(RETRIEVAL_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        y_pred = model(z_val_t, a_val_t, z_ctx_val_t).squeeze(-1).cpu().numpy()

    mse = float(np.mean((r_val - y_pred) ** 2))
    rec_retrieval = eval_retrieval_pool_selection_ctx(
        model,
        rank_ctx,
        val_scenarios,
        device,
        TOP_K_RETRIEVE,
        exclude_self=False,
    )
    metrics = {
        "val_rmse_std": float(math.sqrt(mse)),
        "val_mae_std": float(mean_absolute_error(r_val, y_pred)),
        "val_r2_std": float(r2_score(r_val, y_pred)),
        "val_pearson_std": float(np.corrcoef(r_val, y_pred)[0, 1])
        if y_pred.std() > 1e-12
        else float("nan"),
        "best_val_retrieval_hit_at_1": float(best_hit1),
        "top_k_retrieve": TOP_K_RETRIEVE,
        "epochs_run": len(history),
        "warm_start_from_q_latent": Q_LATENT_MODEL_PATH.is_file(),
        "z_ctx_cache": str(Z_CTX_CACHE),
        "model_path": str(RETRIEVAL_MODEL_PATH),
        "retrieval_pool_selection": rec_retrieval,
        "history": history[-10:],
    }

    print("\n--- QRetrievalNet 验证 ---")
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

    with open(RETRIEVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存: {RETRIEVAL_MODEL_PATH}")


if __name__ == "__main__":
    train_q_retrieval_main()
