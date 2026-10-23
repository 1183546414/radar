# -*- coding: utf-8 -*-
"""改进②：set-attention 检索条件 Q 网络（QSetAttentionNet）。

与 QRetrievalNet（z_ctx = 邻居加权均值向量）不同，本网络把同类型 Top-K
检索邻居当作一个"集合"处理：
  * 邻居潜向量 + 检索相似度 -> key/value tokens [B, K, D]
  * (z_query, 候选动作 A) -> query token（A 参与 query）
  * cross-attention 聚合邻居集合信息 -> Q 值

关键设计：邻居集合在 forward 时作为输入注入（而非训练时固定成单向量），
因此推理阶段可以把检索库替换为任意邻居库（例如将来的 k-shot 实测样本库），
网络零改动复用。neighbor_set_for_query() 即为"任意库注入"的统一入口。

输入约定：
  z_query [B, z_dim]  查询状态潜向量
  a       [B, a_dim]  候选动作（标准化）
  nb_z    [B, K, z_dim] 邻居潜向量集合（不足 K 个补零）
  nb_sim  [B, K]      每个邻居的余弦相似度
  nb_mask [B, K]      bool，True=有效邻居
输出：Q 值 [B, 1]（标准化奖励空间）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

DROPOUT = 0.1
D_MODEL = 128
N_HEADS = 4
N_BLOCKS = 2


class CrossAttentionBlock(nn.Module):
    """Pre-Norm cross-attention 残差块（query 序列长 1）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,          # [B, 1, D]
        kv: torch.Tensor,         # [B, K, D]
        key_padding_mask: torch.Tensor,  # [B, K] True=padding(不参与注意力)
    ) -> torch.Tensor:
        q_n, kv_n = self.norm_q(q), self.norm_kv(kv)
        attn_out, _ = self.mha(q_n, kv_n, kv_n, key_padding_mask=key_padding_mask)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return self.norm_out(q)


class QSetAttentionNet(nn.Module):
    """Q(z_query, {z_neighbors}, a) -> 标量奖励（标准化空间）。

    候选动作 A 与 z_query 一起构成 query token，使每个候选动作都能
    "提问"邻居集合中与自己相关的先验信息。
    """

    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_blocks: int = N_BLOCKS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.z_dim, self.a_dim = z_dim, a_dim
        self.d_model, self.n_heads, self.n_blocks = d_model, n_heads, n_blocks
        self.z_proj = nn.Linear(z_dim, d_model)
        self.a_proj = nn.Linear(a_dim, d_model)
        # 邻居 token = Linear([z_nb ; sim_nb])
        self.nb_proj = nn.Linear(z_dim + 1, d_model)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        z_query: torch.Tensor,   # [B, z_dim]
        a: torch.Tensor,         # [B, a_dim]
        nb_z: torch.Tensor,      # [B, K, z_dim]
        nb_sim: torch.Tensor,    # [B, K]
        nb_mask: torch.Tensor,   # [B, K] bool
    ) -> torch.Tensor:
        nb_feat = torch.cat([nb_z, nb_sim.unsqueeze(-1)], dim=-1)  # [B,K,dz+1]
        kv = self.nb_proj(nb_feat)                                  # [B,K,D]
        q0 = self.z_proj(z_query) + self.a_proj(a)                  # [B,D]
        key_padding = ~nb_mask
        q = q0.unsqueeze(1)                                         # [B,1,D]
        for blk in self.blocks:
            q = blk(q, kv, key_padding)
        return self.head(torch.cat([q0, q.squeeze(1)], dim=-1))     # [B,1]


def neighbor_set_for_query(
    z_query: np.ndarray,
    z_bank: np.ndarray,
    sid_bank: np.ndarray,
    jammer_type_bank: np.ndarray,
    query_jammer_type: str,
    k: int,
    exclude_sid: Optional[int] = None,
    fallback_z: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """任意邻居库注入接口：库(z_bank, sid_bank, type_bank) -> 定长邻居集合。

    返回 (nb_z [k,dz], nb_sim [k], mask [k])；同类型 Top-K 余弦检索，
    无有效邻居时回退 fallback_z（默认 z_query 自身，sim=1）。
    将来做 k-shot 实验时，把 z_bank 换成实测样本潜向量库即可。
    """
    from radar.train.retrieval_ranking import topk_similar_scenarios

    z_query = np.asarray(z_query, dtype=np.float32)
    if fallback_z is None:
        fallback_z = z_query
    neighbor_sids, sims = topk_similar_scenarios(
        z_query,
        z_bank,
        sid_bank,
        k,
        query_jammer_type=str(query_jammer_type),
        jammer_type_bank=jammer_type_bank,
        exclude_sid=exclude_sid,
    )
    dz = int(z_bank.shape[1]) if len(z_bank) else int(len(fallback_z))
    nb_z = np.zeros((k, dz), dtype=np.float32)
    nb_sim = np.zeros(k, dtype=np.float32)
    mask = np.zeros(k, dtype=bool)
    filled = 0
    for i, sid in enumerate(neighbor_sids):
        pos = np.where(sid_bank == sid)[0]
        if len(pos):
            nb_z[filled] = z_bank[int(pos[0])]
            nb_sim[filled] = float(sims[i])
            mask[filled] = True
            filled += 1
    if filled == 0:
        nb_z[0] = np.asarray(fallback_z, dtype=np.float32)
        nb_sim[0] = 1.0
        mask[0] = True
    return nb_z, nb_sim, mask


@torch.no_grad()
def predict_q_attention(
    model: QSetAttentionNet,
    z_query: np.ndarray,
    a: np.ndarray,
    nb_z: np.ndarray,
    nb_sim: np.ndarray,
    nb_mask: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """QSetAttentionNet 批量推理。

    支持两种用法：
      * 1 个查询 + N 个候选动作 + 1 组共享邻居集合（nb_* 传 [K,...]）
      * N 行一一对应（nb_* 传 [N,K,...]）
    """
    model.eval()
    z_query = np.asarray(z_query, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    nb_z = np.asarray(nb_z, dtype=np.float32)
    nb_sim = np.asarray(nb_sim, dtype=np.float32)
    nb_mask = np.asarray(nb_mask, dtype=bool)
    if z_query.ndim == 1:
        z_query = z_query.reshape(1, -1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    single_set = nb_z.ndim == 2
    if single_set:
        nb_z, nb_sim, nb_mask = nb_z[None], nb_sim[None], nb_mask[None]
    n = max(len(z_query), len(a))
    if len(z_query) == 1 and n > 1:
        z_query = np.repeat(z_query, n, axis=0)
    if len(a) == 1 and n > 1:
        a = np.repeat(a, n, axis=0)
    if single_set and n > 1:
        nb_z = np.repeat(nb_z, n, axis=0)
        nb_sim = np.repeat(nb_sim, n, axis=0)
        nb_mask = np.repeat(nb_mask, n, axis=0)
    assert len(z_query) == len(a) == len(nb_z)

    preds = []
    for i in range(0, n, batch_size):
        zq = torch.from_numpy(z_query[i : i + batch_size]).to(device)
        ab = torch.from_numpy(a[i : i + batch_size]).to(device)
        zb = torch.from_numpy(nb_z[i : i + batch_size]).to(device)
        sb = torch.from_numpy(nb_sim[i : i + batch_size]).to(device)
        mb = torch.from_numpy(nb_mask[i : i + batch_size]).to(device)
        preds.append(model(zq, ab, zb, sb, mb).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)
