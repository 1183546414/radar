# -*- coding: utf-8 -*-
"""
对一个“新的原始输入”先做与训练一致的预处理（中位数填充 + 标准化），
用已训练权重编码为潜向量 μ，并在已有的 latent 集合中检索 Top-5 最相似向量。
"""

from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 路径配置（按需修改） =================
DATA_DIR   = Path("./dataset/data_2/json/processed_dataset")
MODEL_PATH = Path("./result/state_vae_runs/state_ae_or_vae.pt")
EMB_PATH   = Path("./result/state_vae_runs/state_embeddings_mu.npy")

# ================== 与训练一致的编码器结构 ==================
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

# ============== 1) 加载标准化工件（imputer + scaler） ==============
art_path = DATA_DIR / "artifacts.pkl"
with open(art_path, "rb") as f:
    artifacts = pickle.load(f)

imp_S = artifacts["imp_S"]   # SimpleImputer(strategy="median")
sc_S  = artifacts["sc_S"]    # StandardScaler()

# 也可从 S_std.npy 读到 s_dim；与 scaler 的 n_features_in_ 应一致
S_STD = np.load(DATA_DIR / "S_std.npy")
s_dim = S_STD.shape[1]

# ============== 2) 假定一个“新的原始输入” ==============
# 注意：这里是“原始特征”的一行，维度必须与 S_raw 的列一致（等于 s_dim）。
# S_raw 的列顺序与构造时一致：[ejt_vec, jcf_arr, jp_arr] 的拼接。
# 你可以把下面这行替换成**真实的原始输入**（未标准化、未填充的数值），长度= s_dim。
new_input_raw = np.array([1.0, 4.0e9, 30.0, 1.0, 1.0, -3.0e7, 5.0e6, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
#new_input_raw = np.random.rand(s_dim).astype(np.float32)  # TODO: 替换为你的真实原始输入

# ============== 3) 预处理：中位数填充 + 标准化（与训练一致） ==============
# reshape 成 (1, s_dim) 后用 fit 好的 imputer/scaler 做 transform
new_input_imp = imp_S.transform(new_input_raw.reshape(1, -1))
new_input_std = sc_S.transform(new_input_imp)  # 得到与 S_std 同分布的标准化特征

# ============== 4) 加载模型并编码为 μ ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从已有 embeddings 文件推断 latent_dim，避免手填出错
emb_all = np.load(EMB_PATH)            # [N, latent_dim]
latent_dim = emb_all.shape[1]

model = ModelForEncode(s_dim, latent_dim).to(device)
state = torch.load(MODEL_PATH, map_location=device)
# 只需要加载 encoder 的权重；strict=False 以兼容包含 decoder 的 state_dict
model.load_state_dict(state, strict=False)
model.eval()

with torch.no_grad():
    x = torch.from_numpy(new_input_std.astype(np.float32)).to(device)  # [1, s_dim]
    mu, _ = model.encode(x)
    mu_new = mu.cpu().numpy().squeeze(0)  # [latent_dim]

print("新样本的标准化特征：", new_input_std.squeeze(0))
print("新样本编码得到的潜向量 μ：", mu_new)

# ============== 5) 在已有 latent 集合中检索 Top-5 最相似向量 ==============
emb_n = l2n(emb_all)
mu_n  = mu_new / (np.linalg.norm(mu_new) + 1e-8)

sims = emb_n @ mu_n  # 余弦相似度，[N]
k = 5
topk_idx = np.argpartition(-sims, kth=k-1)[:k]
topk_idx = topk_idx[np.argsort(-sims[topk_idx])]  # 按相似度降序

print("\nTop-5 最相似隐向量：")
for rank, idx in enumerate(topk_idx, start=1):
    print(f"[{rank}] index={idx}, sim={sims[idx]:.6f}")
    print(emb_all[idx], "\n")
