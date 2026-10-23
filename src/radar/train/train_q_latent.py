# -*- coding: utf-8 -*-
"""
监督 Q 网络：Q(z, A) -> R

- z: 冻结 State VAE 对 S_std 编码的潜向量 μ
- A: 标准化后的雷达动作
- R: 标准化后的 SINR 奖励

训练：MSE 回归 + 检索候选池 listwise ranking loss（对齐 Retrieve+Q 推理）。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from radar.frozen_vae import encode_states, load_frozen_vae_encoder
from radar.paths import DATA_V2_PROCESSED, OUT_Q_VALUE
from radar.train.retrieval_ranking import (
    RetrievalRankingContext,
    eval_retrieval_pool_selection,
    ranking_loss_batch,
)

OUT_DIR = OUT_Q_VALUE / "latent_q"
LATENT_CACHE = OUT_DIR / "state_latent_mu.npy"
MODEL_PATH = OUT_DIR / "q_latent_best.pt"
METRICS_PATH = OUT_DIR / "metrics.json"

SEED = 42
EPOCHS = 300
BATCH_SIZE = int(os.environ.get("RADAR_Q_BATCH_SIZE", "256"))
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
PATIENCE = 25
VAL_RATIO = 0.2
SCENARIO_ID_PATH = DATA_V2_PROCESSED / "scenario_id.npy"
R_RAW_PATH = DATA_V2_PROCESSED / "R_raw.npy"

# 检索 ranking loss（与 holdout Top-5 检索 + argmax Q 对齐）
TOP_K_RETRIEVE = 5
RANK_LAMBDA = 0.5
RANK_SCENARIOS_PER_STEP = 4
RANK_STEPS_PER_EPOCH = 32
VAL_RANK_MAX_SCENARIOS = 100


class QLatentNet(nn.Module):
    """Q(z, a) -> 标量奖励（标准化空间）。"""

    def __init__(self, z_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = z_dim + a_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


def _load_or_encode_latent(device: torch.device) -> np.ndarray:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    S_std = np.load(DATA_V2_PROCESSED / "S_std.npy").astype(np.float32)
    if LATENT_CACHE.exists():
        z = np.load(LATENT_CACHE)
        if z.shape[0] == S_std.shape[0]:
            print(f"加载缓存潜向量: {LATENT_CACHE} {z.shape}")
            return z
    vae, _, _ = load_frozen_vae_encoder(device)
    print(f"冻结 VAE 编码 S_std -> μ, N={len(S_std)}")
    z = encode_states(vae, S_std, device)
    np.save(LATENT_CACHE, z)
    return z


@torch.no_grad()
def predict_q(
    model: QLatentNet,
    z: np.ndarray,
    a: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    z = np.asarray(z, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    n = max(len(z), len(a))
    if len(z) == 1 and n > 1:
        z = np.repeat(z, n, axis=0)
    if len(a) == 1 and n > 1:
        a = np.repeat(a, n, axis=0)
    assert len(z) == len(a)
    preds = []
    for i in range(0, n, batch_size):
        zb = torch.from_numpy(z[i : i + batch_size]).to(device)
        ab = torch.from_numpy(a[i : i + batch_size]).to(device)
        preds.append(model(zb, ab).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _split_by_scenario(
    scenario_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 scenario_id 划分 train/val 行索引（同一场景的全部 K 行在同一侧）。"""
    unique = np.unique(scenario_ids)
    tr_s, val_s = train_test_split(unique, test_size=val_ratio, random_state=seed)
    tr_mask = np.isin(scenario_ids, tr_s)
    val_mask = np.isin(scenario_ids, val_s)
    tr_idx = np.where(tr_mask)[0]
    val_idx = np.where(val_mask)[0]
    return tr_idx, val_idx, tr_s, val_s


def _sample_scenarios(scenarios: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    n = min(n, len(scenarios))
    if n <= 0:
        return np.array([], dtype=scenarios.dtype)
    return rng.choice(scenarios, size=n, replace=False)


def _eval_per_scenario_argmax(
    model: QLatentNet,
    z: np.ndarray,
    a_std: np.ndarray,
    r_raw: np.ndarray,
    scenario_ids: np.ndarray,
    eval_scenarios: np.ndarray,
    device: torch.device,
) -> dict:
    """
    对每个验证场景：在该场景 K 个 A 上 argmax Q(z,a)，
    与真实 argmax R 的动作对比（Oracle Hit@1、SINR regret dB）。
    """
    hits: list[bool] = []
    regrets_db: list[float] = []
    rank_oracle: list[int] = []
    k_list: list[int] = []

    for sid in eval_scenarios:
        idx = np.where(scenario_ids == sid)[0]
        if len(idx) == 0:
            continue
        k_list.append(len(idx))
        z_i = z[idx[0:1]]
        a_cand = a_std[idx]
        r_cand = r_raw[idx].reshape(-1)

        q_pred = predict_q(model, np.repeat(z_i, len(idx), axis=0), a_cand, device)
        pred_local = int(np.argmax(q_pred))
        true_local = int(np.argmax(r_cand))

        hits.append(pred_local == true_local)
        regrets_db.append(float(r_cand[true_local] - r_cand[pred_local]))
        # 真实最优动作在 Q 排序中的名次；此前误用了预测最优动作，结果恒为 1。
        rank_oracle.append(int((q_pred >= q_pred[true_local] - 1e-9).sum()))

    hits_arr = np.array(hits, dtype=bool)
    regret_arr = np.array(regrets_db, dtype=np.float64)
    return {
        "n_eval_scenarios": int(len(hits)),
        "actions_per_scenario_mean": float(np.mean(k_list)) if k_list else 0.0,
        "oracle_hit_at_1": float(hits_arr.mean()) if len(hits_arr) else float("nan"),
        "mean_regret_sinr_db": float(regret_arr.mean()) if len(regret_arr) else float("nan"),
        "median_regret_sinr_db": float(np.median(regret_arr)) if len(regret_arr) else float("nan"),
        "frac_zero_regret": float((regret_arr <= 1e-6).mean()) if len(regret_arr) else float("nan"),
        "mean_rank_of_oracle_action_by_q": float(np.mean(rank_oracle)) if rank_oracle else float("nan"),
    }


def train_q_latent_main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    S_std = np.load(DATA_V2_PROCESSED / "S_std.npy")
    A_std = np.load(DATA_V2_PROCESSED / "A_std.npy").astype(np.float32)
    R_std = np.load(DATA_V2_PROCESSED / "R_std.npy").reshape(-1).astype(np.float32)

    if not SCENARIO_ID_PATH.exists():
        raise FileNotFoundError(
            f"缺少 {SCENARIO_ID_PATH}，请先运行 build_dataset（需含 scenario_id 的 CSV）"
        )
    scenario_ids = np.load(SCENARIO_ID_PATH)
    r_raw = np.load(R_RAW_PATH).reshape(-1).astype(np.float64)
    if len(scenario_ids) != len(S_std):
        raise ValueError(
            f"scenario_id 长度 {len(scenario_ids)} != S_std 行数 {len(S_std)}"
        )

    z = _load_or_encode_latent(device)
    z_dim, a_dim = z.shape[1], A_std.shape[1]
    if len(scenario_ids) != len(z):
        raise ValueError(
            f"scenario_id 长度 {len(scenario_ids)} != 样本数 {len(z)}"
        )
    n_scenarios = int(np.unique(scenario_ids).size)
    k_per = int(np.bincount(scenario_ids.astype(int)).max())
    print(
        f"数据: z{z.shape}, A{A_std.shape}, R{R_std.shape}, "
        f"scenarios={n_scenarios}, K≈{k_per}/场景"
    )

    tr_idx, val_idx, tr_scenarios, val_scenarios = _split_by_scenario(
        scenario_ids, VAL_RATIO, SEED
    )
    print(f"划分: train {len(tr_scenarios)} 场景 / val {len(val_scenarios)} 场景")

    rank_ctx = RetrievalRankingContext.build(
        z=z,
        scenario_ids=scenario_ids,
        a_std=A_std,
        train_scenarios=tr_scenarios,
    )
    print(
        f"检索 ranking: Top-{TOP_K_RETRIEVE}, λ={RANK_LAMBDA}, "
        f"每 epoch {RANK_STEPS_PER_EPOCH}×{RANK_SCENARIOS_PER_STEP} 场景"
    )

    rank_rng = np.random.default_rng(SEED)
    val_rank_scenarios = _sample_scenarios(
        val_scenarios, VAL_RANK_MAX_SCENARIOS, rank_rng
    )

    z_tr, a_tr, r_tr = z[tr_idx], A_std[tr_idx], R_std[tr_idx]
    z_val, a_val, r_val = z[val_idx], A_std[val_idx], R_std[val_idx]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_tr),
            torch.from_numpy(a_tr),
            torch.from_numpy(r_tr),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = QLatentNet(z_dim, a_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    z_val_t = torch.from_numpy(z_val).to(device)
    a_val_t = torch.from_numpy(a_val).to(device)
    r_val_t = torch.from_numpy(r_val).to(device)

    best_score = float("inf")
    wait = 0
    history = []

    print(f"开始训练 QLatentNet: epochs<={EPOCHS}, patience={PATIENCE}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_mse_losses: list[float] = []
        train_rank_losses: list[float] = []

        for zb, ab, rb in train_loader:
            zb, ab, rb = zb.to(device), ab.to(device), rb.to(device)
            pred = model(zb, ab).squeeze(-1)
            loss_mse = loss_fn(pred, rb)
            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()
            train_mse_losses.append(loss_mse.item())

        for _ in range(RANK_STEPS_PER_EPOCH):
            batch_sids = _sample_scenarios(
                tr_scenarios, RANK_SCENARIOS_PER_STEP, rank_rng
            )
            if len(batch_sids) == 0:
                continue
            loss_rank = ranking_loss_batch(
                model,
                rank_ctx,
                batch_sids,
                device,
                top_k=TOP_K_RETRIEVE,
                exclude_self=True,
            )
            optimizer.zero_grad()
            (RANK_LAMBDA * loss_rank).backward()
            optimizer.step()
            train_rank_losses.append(float(loss_rank.item()))

        model.eval()
        with torch.no_grad():
            pred_val = model(z_val_t, a_val_t).squeeze(-1)
            val_mse = loss_fn(pred_val, r_val_t).item()
            val_rank_nll = float(
                ranking_loss_batch(
                    model,
                    rank_ctx,
                    val_rank_scenarios,
                    device,
                    top_k=TOP_K_RETRIEVE,
                    exclude_self=False,
                ).item()
            )

        tr_mse = float(np.mean(train_mse_losses))
        tr_rank = float(np.mean(train_rank_losses)) if train_rank_losses else 0.0
        combined_score = val_mse + RANK_LAMBDA * val_rank_nll
        history.append(
            {
                "epoch": epoch,
                "train_mse": tr_mse,
                "train_rank_nll": tr_rank,
                "val_mse": val_mse,
                "val_rank_nll": val_rank_nll,
                "val_combined": combined_score,
            }
        )

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | train_mse={tr_mse:.5f} rank_nll={tr_rank:.4f} | "
                f"val_mse={val_mse:.5f} val_rank_nll={val_rank_nll:.4f} "
                f"combined={combined_score:.5f}"
            )

        if combined_score < best_score - 1e-6:
            best_score = combined_score
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "z_dim": z_dim,
                    "a_dim": a_dim,
                },
                MODEL_PATH,
            )
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"早停 @ epoch {epoch} (combined=val_mse+{RANK_LAMBDA}*val_rank_nll)")
                break

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        y_pred = model(z_val_t, a_val_t).squeeze(-1).cpu().numpy()

    mse = float(np.mean((r_val - y_pred) ** 2))
    metrics = {
        "val_rmse_std": float(math.sqrt(mse)),
        "val_mae_std": float(mean_absolute_error(r_val, y_pred)),
        "val_r2_std": float(r2_score(r_val, y_pred)),
        "val_pearson_std": float(np.corrcoef(r_val, y_pred)[0, 1])
        if y_pred.std() > 1e-12
        else float("nan"),
        "best_val_combined": float(best_score),
        "rank_lambda": RANK_LAMBDA,
        "top_k_retrieve": TOP_K_RETRIEVE,
        "epochs_run": len(history),
        "z_dim": z_dim,
        "a_dim": a_dim,
        "latent_cache": str(LATENT_CACHE),
        "model_path": str(MODEL_PATH),
        "history": history[-10:],
    }

    print("\n--- Q(z,A) 回归（验证集，R 为标准化 SINR）---")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "history"},
        indent=2,
        ensure_ascii=False,
    ))

    rec_k20 = _eval_per_scenario_argmax(
        model,
        z,
        A_std,
        r_raw,
        scenario_ids,
        val_scenarios,
        device,
    )
    metrics["per_scenario_k20_selection"] = rec_k20
    metrics["per_scenario_action_selection"] = rec_k20  # 兼容旧脚本
    print("\n--- 动作推荐（每场景 K=20 A 内 argmax Q）---")
    print(json.dumps(rec_k20, indent=2, ensure_ascii=False))

    rec_retrieval = eval_retrieval_pool_selection(
        model,
        rank_ctx,
        val_scenarios,
        device,
        top_k=TOP_K_RETRIEVE,
        exclude_self=False,
    )
    metrics["retrieval_pool_selection"] = rec_retrieval
    print(
        f"\n--- 检索候选池推荐（Top-{TOP_K_RETRIEVE} train 场景, ~"
        f"{rec_retrieval.get('candidates_per_scenario_mean', 0):.0f} A, 对齐 Retrieve+Q）---"
    )
    print(json.dumps(rec_retrieval, indent=2, ensure_ascii=False))

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存: {MODEL_PATH}")
    print(f"指标已保存: {METRICS_PATH}")


if __name__ == "__main__":
    train_q_latent_main()
