# -*- coding: utf-8 -*-
"""ICDT 数据加载与序列构造（20261023 修正结构重训版）。

与 outputs/20261013_icdt/code/smoke_icdt.py 唯一差异：
  z 潜向量数据源改为新编码器 outputs/state_vae/state_embeddings_mu.npy
  （旧版读 20260909_signal_level/models/latent_q/state_latent_mu.npy，属旧结构产物）。
其余常量 / 序列布局 / 邻居构造与旧版逐行一致，保证训练协议可比。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
for _p in HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        sys.path.insert(0, str(_p / "src"))
        break

import numpy as np
import pandas as pd
import torch.nn as nn

from radar.paths import OUT_STATE_VAE

# ---- Round2 处理后数据路径（与 20261007 轮相同套路：只读不改）----
ROUND2 = HERE.parents[3] / "outputs" / "20260909_signal_level"
ROUND2_PROCESSED = ROUND2 / "processed"
ROUND2_CSV = ROUND2 / "data" / "simulated_radar_jamming_high_contrast_10k_signal.csv"

# ---- 序列 / 模型常量 ----
A_DIM = 9          # 雷达动作维度（标准化；A_raw 9 列，含离散与连续分量）
Z_DIM = 128        # 干扰状态潜向量维度
N_OBS_SLOTS = 8    # 场景内观测槽位（m ∈ 0..8，不足补零）
K_NEIGHBORS = 5    # 邻居经验 token 数
TYPE_OBS, TYPE_NB, TYPE_QUERY, TYPE_CAND = 0, 1, 2, 3
N_TYPES = 4
FEAT_DIM = A_DIM + 1 + 1 + Z_DIM + N_TYPES  # 140
SEQ_LEN = N_OBS_SLOTS + K_NEIGHBORS + 2     # 15
D_MODEL, N_HEADS, N_LAYERS, FF_DIM = 128, 4, 4, 256

BATCH = 256
STEPS = 600
LR = 3e-4
SEED = 42


def load_data():
    # ★ 与旧版唯一差异：z 来自修正后重训的状态编码器（潜向量与动作行一一对应）
    z = np.load(OUT_STATE_VAE / "state_embeddings_mu.npy").astype(np.float32)
    A_std = np.load(ROUND2_PROCESSED / "A_std.npy").astype(np.float32)
    R_std = np.load(ROUND2_PROCESSED / "R_std.npy").reshape(-1).astype(np.float32)
    scenario_ids = np.load(ROUND2_PROCESSED / "scenario_id.npy")
    assert z.shape[1] == Z_DIM, f"z 维度不符: {z.shape}"
    assert len(z) == len(A_std) == len(R_std) == len(scenario_ids)

    df = pd.read_csv(ROUND2_CSV, usecols=["scenario_id", "Jammer_Type"])
    df = df.drop_duplicates(subset=["scenario_id"])
    jammer_by_sid = dict(zip(df["scenario_id"].astype(int), df["Jammer_Type"].astype(str)))

    # 场景 → 行索引（5000 场景 × 每场景 20 动作；同场景共享同一干扰状态，z 取首行）
    sid_to_rows: dict[int, np.ndarray] = {}
    for sid in np.unique(scenario_ids):
        rows = np.where(scenario_ids == sid)[0]
        assert len(rows) == 20, f"场景 {sid} 行数 {len(rows)} != 20"
        sid_to_rows[int(sid)] = rows
    all_sids = np.array(sorted(sid_to_rows.keys()))
    return z, A_std, R_std, sid_to_rows, all_sids, jammer_by_sid


def build_neighbors(z, A_std, R_std, sid_to_rows, all_sids, jammer_by_sid, top_k=K_NEIGHBORS):
    """按干扰类型分组，组内 z 余弦 top-k 邻居（排除自身）。

    返回 nb_a/nb_r/nb_sim: [n_sid, top_k, ...]，邻居经验 = 邻居场景内 R_std 最高的动作。
    """
    z_sid = np.stack([z[sid_to_rows[int(s)][0]] for s in all_sids])  # [n_sid, 128]
    z_norm = z_sid / (np.linalg.norm(z_sid, axis=1, keepdims=True) + 1e-8)
    type_of = np.array([jammer_by_sid[int(s)] for s in all_sids])

    # 每场景的最优动作与 R（邻居经验的"这么打最有效"信息）
    best_a = np.stack([A_std[sid_to_rows[int(s)][int(np.argmax(R_std[sid_to_rows[int(s)]]))]]
                       for s in all_sids])  # [n_sid, 6]
    best_r = np.array([R_std[sid_to_rows[int(s)][int(np.argmax(R_std[sid_to_rows[int(s)]]))]]
                       for s in all_sids], dtype=np.float32)  # [n_sid]

    n = len(all_sids)
    nb_a = np.zeros((n, top_k, A_DIM), dtype=np.float32)
    nb_r = np.zeros((n, top_k), dtype=np.float32)
    nb_sim = np.zeros((n, top_k), dtype=np.float32)
    for t in np.unique(type_of):
        idx = np.where(type_of == t)[0]
        if len(idx) <= top_k + 1:
            continue
        sims = z_norm[idx] @ z_norm[idx].T  # 组内余弦 [g, g]
        for i, row_i in enumerate(idx):
            sims_i = sims[i].copy()
            sims_i[i] = -2.0  # 排除自身
            nb = idx[np.argsort(-sims_i)[:top_k]]
            nb_a[row_i] = best_a[nb]
            nb_r[row_i] = best_r[nb]
            nb_sim[row_i] = sims_i[np.argsort(-sims_i)[:top_k]]
    assert (nb_sim > -1).all(), "存在未填充的邻居（类型组过小）"
    return nb_a, nb_r, nb_sim


def make_token(a, r, sim, z_vec, ttype):
    """统一 token 特征 [a;r;sim;z] 缺失补零 + 类型 one-hot。"""
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    feat[:A_DIM] = a
    if r is not None:
        feat[A_DIM] = r
    if sim is not None:
        feat[A_DIM + 1] = sim
    if z_vec is not None:
        feat[A_DIM + 2: A_DIM + 2 + Z_DIM] = z_vec
    feat[A_DIM + 2 + Z_DIM + ttype] = 1.0
    return feat


class ICDTSmoke(nn.Module):
    """bidirectional self-attention 小 Transformer（集合推理，非自回归）。

    手动堆叠 encoder 层：规避 nn.TransformerEncoder eval 模式融合内核
    与 src_key_padding_mask 的 NaN 兼容问题（与旧版一致）。
    """

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(FEAT_DIM, D_MODEL)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                D_MODEL, N_HEADS, FF_DIM, dropout=0.1,
                batch_first=True, norm_first=True)
            for _ in range(N_LAYERS)
        ])
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, toks, pad_mask):
        h = self.in_proj(toks)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=pad_mask)
        return self.head(h[:, -1]).squeeze(-1)  # 候选 token 在序列末位
