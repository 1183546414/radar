# -*- coding: utf-8 -*-
"""ICDT（In-Context Decision Transformer）正式训练脚本。

架构与序列构造沿用冒烟验证过的 smoke_icdt.py（复用其数据加载/邻居构建/模型类），
补齐正式训练要素：
  * 按场景 80/20 split（复用 train_q_latent._split_by_scenario, SEED=42，与 Round2 同协议）
  * 向量化序列构造（每步动态重采样观测组合，m ∈ 0..8 模拟 k-shot）
  * 早停：固定 val 集 MSE（PATIENCE=25, MAX_EPOCHS=300）
  * 5 seeds 集成（42/1042/2042/3042/4042，与前章 Q-Ensemble 一致），悲观打分在评估侧做
  * 断点续训：metrics.json 已存在的成员跳过

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261013_icdt/code/train_icdt.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
for _p in HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        sys.path.insert(0, str(_p / "src"))
        break
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch
import torch.nn as nn

from smoke_icdt import (  # 复用冒烟验证过的组件
    A_DIM, D_MODEL, FEAT_DIM, ICDTSmoke, K_NEIGHBORS, N_OBS_SLOTS, SEQ_LEN,
    TYPE_CAND, TYPE_NB, TYPE_OBS, TYPE_QUERY, Z_DIM,
    build_neighbors, load_data,
)
from radar.train.train_q_latent import EPOCHS, PATIENCE, VAL_RATIO, _split_by_scenario

OUT_DIR = ROUND / "models" / "icdt_ensemble"
REPORT_PATH = ROUND / "train_report.json"

SEEDS = (42, 1042, 2042, 3042, 4042)
SPLIT_SEED = 42          # 与 Round2 / 20261007 轮相同的场景 split
VAL_SET_SEED = 2026      # 固定 val 集构造（跨成员可比）
BATCH = 256
STEPS_PER_EPOCH = 200    # ≈51k 样本/epoch（数据 10 万行的一半）
LR = 3e-4
VAL_BATCH = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_rows_mat(sid_to_rows, all_sids):
    """场景 → 行索引矩阵 [n_sid, 20] 与 场景id→块号 查找表。"""
    rows_mat = np.stack([sid_to_rows[int(s)] for s in all_sids])
    n_sid = len(all_sids)
    sid2block = np.full(int(all_sids.max()) + 1, -1, dtype=np.int64)
    sid2block[all_sids] = np.arange(n_sid)
    return rows_mat, sid2block


def sample_batch_vec(rng, sids_pool, data, batch=BATCH):
    """向量化构造 ICDT 序列批（动态：场景/查询行/观测组合每步重采样）。

    序列布局（与冒烟一致）：[obs×8槽 | nb×5 | query_z | cand_a]，长度 15。
    观测无放回采样技巧：rand keys 把查询列置最大值 → argsort 后查询列排最后，
    perm 前 8 列必不含查询行；槽位有效性由 obs_valid 控制经 pad mask 屏蔽。
    """
    z, A_std, R_std, rows_mat, sid2block, nb_a, nb_r, nb_sim = data
    B = batch
    sids = rng.choice(sids_pool, size=B)
    blk = sid2block[sids]
    rm = rows_mat[blk]                                  # [B, 20]
    q = rng.integers(0, 20, B)
    target_row = rm[np.arange(B), q]                    # [B]
    m = rng.integers(0, N_OBS_SLOTS + 1, B)             # 观测数 0..8

    keys = rng.random((B, 20))
    keys[np.arange(B), q] = 2.0                         # 查询列排到最后
    perm = np.argsort(keys, axis=1)
    obs_rows = rm[np.arange(B)[:, None], perm[:, :N_OBS_SLOTS]]   # [B, 8]
    obs_valid = np.arange(N_OBS_SLOTS)[None, :] < m[:, None]      # [B, 8]

    toks = np.zeros((B, SEQ_LEN, FEAT_DIM), dtype=np.float32)
    p_obs, p_nb = 0, N_OBS_SLOTS
    p_q, p_c = N_OBS_SLOTS + K_NEIGHBORS, N_OBS_SLOTS + K_NEIGHBORS + 1
    t_base = A_DIM + 2 + Z_DIM                          # 类型 one-hot 基址 = 139

    # obs tokens: [a(9); r; 0; 0; onehot(OBS)]
    toks[:, p_obs:p_obs + N_OBS_SLOTS, :A_DIM] = A_std[obs_rows]
    toks[:, p_obs:p_obs + N_OBS_SLOTS, A_DIM] = R_std[obs_rows]
    toks[:, p_obs:p_obs + N_OBS_SLOTS, t_base + TYPE_OBS] = 1.0
    # nb tokens: [a(9); r; sim; 0; onehot(NB)]
    toks[:, p_nb:p_nb + K_NEIGHBORS, :A_DIM] = nb_a[sids]
    toks[:, p_nb:p_nb + K_NEIGHBORS, A_DIM] = nb_r[sids]
    toks[:, p_nb:p_nb + K_NEIGHBORS, A_DIM + 1] = nb_sim[sids]
    toks[:, p_nb:p_nb + K_NEIGHBORS, t_base + TYPE_NB] = 1.0
    # query token: [0; 0; 0; z(128); onehot(QUERY)]（同场景 z 相同，取块首行）
    toks[:, p_q, A_DIM + 2:A_DIM + 2 + Z_DIM] = z[rm[:, 0]]
    toks[:, p_q, t_base + TYPE_QUERY] = 1.0
    # cand token: [a(9); 0; 0; 0; onehot(CAND)]
    toks[:, p_c, :A_DIM] = A_std[target_row]
    toks[:, p_c, t_base + TYPE_CAND] = 1.0

    pad = np.ones((B, SEQ_LEN), dtype=bool)
    pad[:, p_obs:p_obs + N_OBS_SLOTS] = ~obs_valid
    pad[:, p_nb:] = False                               # nb×5 + query + cand 恒有效
    return (torch.from_numpy(toks).to(DEVICE),
            torch.from_numpy(pad).to(DEVICE),
            torch.from_numpy(R_std[target_row]).to(DEVICE))


@torch.no_grad()
def eval_val_mse(model, val_set):
    model.eval()
    toks, pad, y = val_set
    se, n = 0.0, len(y)
    for i in range(0, n, VAL_BATCH):
        pred = model(toks[i:i + VAL_BATCH], pad[i:i + VAL_BATCH])
        se += float(((pred - y[i:i + VAL_BATCH]) ** 2).sum().item())
    return se / n


def train_one(seed: int, data, tr_sids, val_set) -> dict:
    out_dir = OUT_DIR / f"seed{seed}"
    ckpt_path = out_dir / "icdt_best.pt"
    metrics_path = out_dir / "metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path.is_file():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"[seed {seed}] 已存在 metrics.json，跳过（best_val_mse={m['best_val_mse']:.4f}）")
        return m

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = ICDTSmoke().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_mse, best_epoch, t0 = float("inf"), -1, time.perf_counter()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for _ in range(STEPS_PER_EPOCH):
            toks, pad, y = sample_batch_vec(np_rng, tr_sids, data)
            pred = model(toks, pad)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(y)
            ep_n += len(y)
        val_mse = eval_val_mse(model, val_set)
        history.append({"epoch": epoch, "train_mse": ep_loss / ep_n, "val_mse": val_mse})
        if val_mse < best_mse - 1e-5:
            best_mse, best_epoch = val_mse, epoch
            torch.save(model.state_dict(), ckpt_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[seed {seed}] epoch {epoch:3d} | train {ep_loss/ep_n:.4f} "
                  f"| val {val_mse:.4f} | best {best_mse:.4f} @ {best_epoch}")
        if epoch - best_epoch >= PATIENCE:
            print(f"[seed {seed}] 早停于 epoch {epoch}（best {best_mse:.4f} @ {best_epoch}）")
            break

    elapsed = time.perf_counter() - t0
    metrics = {
        "seed": seed, "best_val_mse": best_mse, "best_epoch": best_epoch,
        "stopped_epoch": epoch, "elapsed_min": elapsed / 60,
        "n_params_k": sum(p.numel() for p in model.parameters()) / 1e3,
        "config": {"batch": BATCH, "steps_per_epoch": STEPS_PER_EPOCH, "lr": LR,
                   "max_epochs": EPOCHS, "patience": PATIENCE,
                   "seq_len": SEQ_LEN, "feat_dim": FEAT_DIM},
        "history_tail": history[-5:],
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    del model, opt
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main():
    t0 = time.perf_counter()
    z, A_std, R_std, sid_to_rows, all_sids, jammer_by_sid = load_data()
    nb_a, nb_r, nb_sim = build_neighbors(z, A_std, R_std, sid_to_rows, all_sids,
                                         jammer_by_sid)
    rows_mat, sid2block = _build_rows_mat(sid_to_rows, all_sids)
    data = (z, A_std, R_std, rows_mat, sid2block, nb_a, nb_r, nb_sim)

    # 场景级 80/20 split（SEED=42，与 Round2 同协议）
    _, _, tr_sids, val_sids = _split_by_scenario(
        np.load(ROUND2_SCENARIO_PATH := HERE.parents[3] / "outputs"
                / "20260909_signal_level" / "processed" / "scenario_id.npy"),
        VAL_RATIO, SPLIT_SEED)
    print(f"[split] train {len(tr_sids)} 场景 / val {len(val_sids)} 场景")

    # 固定 val 集（跨成员可比，VAL_SET_SEED 构造一次）
    val_set = sample_batch_vec(np.random.default_rng(VAL_SET_SEED), val_sids, data,
                               batch=4096)
    print(f"[val] 固定 val 集 4096 样本 | device={DEVICE}")

    all_metrics = []
    for seed in SEEDS:
        all_metrics.append(train_one(seed, data, tr_sids, val_set))

    report = {
        "round": "20261023_icdt_retrain", "seeds": list(SEEDS),
        "encoder": "outputs/state_vae (20261023 修正结构重训)",
        "split": {"val_ratio": VAL_RATIO, "seed": SPLIT_SEED},
        "members": all_metrics,
        "val_mses": [m["best_val_mse"] for m in all_metrics],
        "total_min": (time.perf_counter() - t0) / 60,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n全部成员完成: val_mse = "
          f"{[round(m['best_val_mse'], 4) for m in all_metrics]}")
    print(f"总耗时 {report['total_min']:.1f} min | 报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
