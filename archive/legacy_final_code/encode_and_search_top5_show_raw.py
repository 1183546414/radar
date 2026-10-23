# -*- coding: utf-8 -*-
"""
对一条“新的原始输入”：
1) 按训练时的流程做中位数填充 + 标准化（从 artifacts.pkl 读取）
2) 用训练好的模型权重编码为潜向量 μ
3) 在已有 latent 集合 state_embeddings_mu.npy 中做 Top-5 余弦相似度检索
4) 输出 Top-5 的索引、相似度，以及它们在 S_raw.npy 中对应的“原始数据行”

用法：直接运行本文件（内部写死了一个示例 new_input_raw；你可按需替换）
"""

from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 路径配置（按你的工程组织修改） =================
DATA_DIR   = Path("./dataset/data_2/json/processed_dataset")
MODEL_PATH = Path("./result/state_vae_runs/state_ae_or_vae.pt")
EMB_PATH   = Path("./result/state_vae_runs/state_embeddings_mu.npy")

S_RAW_PATH = DATA_DIR / "S_raw.npy"      # 原始特征
ART_PATH   = DATA_DIR / "artifacts.pkl"  # 内含 imp_S / sc_S
S_STD_PATH = DATA_DIR / "S_std.npy"      # 仅用于校验维度

# ================== 与训练一致的编码器（只需 Encoder） ==================
DROPOUT = 0.05
USE_LAYERNORM = True
USE_RESIDUAL = True
ENC_HIDDEN = [1024, 1024, 1024, 768]

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, layernorm=True, residual=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.ln = nn.LayerNorm(out_dim) if layernorm else nn.Identity()
        self.use_residual = residual and (in_dim == out_dim)
    def forward(self, x):
        y = F.gelu(self.fc(x)); y = self.dropout(y); y = self.ln(y)
        return y + x if self.use_residual else y

class StackedMLP(nn.Module):
    def __init__(self, in_dim, layers, dropout=0.0, layernorm=True, residual=True):
        super().__init__()
        seq, last = [], in_dim
        for h in layers:
            seq.append(MLPBlock(last, h, dropout, layernorm, residual)); last = h
        self.net = nn.Sequential(*seq)
    def forward(self, x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, s_dim, latent_dim):
        super().__init__()
        self.trunk = StackedMLP(s_dim, ENC_HIDDEN, dropout=DROPOUT,
                                layernorm=USE_LAYERNORM, residual=USE_RESIDUAL)
        h = ENC_HIDDEN[-1]
        self.mu = nn.Linear(h, latent_dim)
        nn.init.xavier_uniform_(self.mu.weight); nn.init.zeros_(self.mu.bias)
    def forward(self, s):
        h = self.trunk(s)
        mu = self.mu(h)
        return mu, None

class ModelForEncode(nn.Module):
    def __init__(self, s_dim, latent_dim):
        super().__init__()
        self.enc = Encoder(s_dim, latent_dim)
    def encode(self, s):
        return self.enc(s)

# ============== 工具函数 ==============
def l2n(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

def search_top5_main():
    # ============== 0) 读取基础数据 ==============
    S_raw = np.load(S_RAW_PATH)            # [N, s_dim] 原始特征
    S_std = np.load(S_STD_PATH)            # [N, s_dim] 标准化后的特征（校验列数一致）
    assert S_raw.shape == S_std.shape, f"S_raw{S_raw.shape} 与 S_std{S_std.shape} 形状不一致！"
    s_dim = S_raw.shape[1]

    # ============== 1) 加载预处理工件（imputer + scaler） ==============
    with open(ART_PATH, "rb") as f:
        artifacts = pickle.load(f)
    imp_S = artifacts["imp_S"]   # SimpleImputer(strategy="median")
    sc_S  = artifacts["sc_S"]    # StandardScaler()

    # ============== 2) 准备“新的原始输入” ==============
    # 你给的示例：1, 4000000000, 30, 1, 1, -30000000, 5000000, 0, 80, 0, 0, 0, 0
    # 注意：必须与 S_raw 的列顺序完全一致，长度 = s_dim
    new_input_raw = np.array(
        [0, 4000000000, 30, 1, 1, -30000000, 5000000, 0, 80, 0, 0, 0, 0],
        dtype=S_raw.dtype  # 与 S_raw 相同 dtype，避免精度不一致
    )
    assert new_input_raw.shape[0] == s_dim, f"new_input_raw 维度({new_input_raw.shape[0]}) != s_dim({s_dim})"

    # ============== 3) 预处理：中位数填充 + 标准化（与训练一致） ==============
    # 如果 new_input_raw 本身没有缺失，这里 imp_S 只是透传；保持流程一致
    new_input_imp = imp_S.transform(new_input_raw.reshape(1, -1))
    new_input_std = sc_S.transform(new_input_imp).astype(np.float32)  # [1, s_dim]

    # ============== 4) 加载模型并编码为 μ ==============
    emb_all = np.load(EMB_PATH)           # [N, latent_dim]
    latent_dim = emb_all.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelForEncode(s_dim, latent_dim).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=False)   # 兼容包含 decoder 的 state_dict
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(new_input_std).to(device)  # [1, s_dim]
        mu, _ = model.encode(x)
        mu_new = mu.cpu().numpy().squeeze(0)           # [latent_dim]

    # ============== 5) 在已有 latent 集合中检索 Top-5，输出对应“原始数据行” ==============
    emb_n = l2n(emb_all)
    mu_n  = mu_new / (np.linalg.norm(mu_new) + 1e-8)

    sims = emb_n @ mu_n  # 余弦相似度，[N]
    k = 5 if len(sims) >= 5 else len(sims)
    topk_idx = np.argpartition(-sims, kth=k-1)[:k]
    topk_idx = topk_idx[np.argsort(-sims[topk_idx])]  # 降序

    print("\n=== 新原始输入 ===")
    print(new_input_raw.tolist())
    print("\n=== Top-5 相似结果（索引、相似度、对应的原始数据行）===")
    for rank, idx in enumerate(topk_idx, start=1):
        print(f"[{rank}] index={idx}, sim={sims[idx]:.6f}")
        print("raw:", S_raw[idx].tolist())
        print("-"*70)

# 可选：把结果保存为 CSV
# import pandas as pd
# rows = [{"rank": r+1, "index": int(i), "sim": float(sims[i]), **{f"raw_{j}": float(S_raw[i, j]) for j in range(s_dim)}}
#         for r, i in enumerate(topk_idx)]
# pd.DataFrame(rows).to_csv("top5_sim_raw_rows.csv", index=False, encoding="utf-8-sig")
# print("已保存到 top5_sim_raw_rows.csv")


if __name__ == "__main__":
    search_top5_main()