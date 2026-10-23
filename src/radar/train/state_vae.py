# -*- coding: utf-8 -*-
"""
StateVAE: 训练与验证
- 从 .npy 加载标准化后的 S/A/R
- 使用 StateVAE 重构状态 S
- 80/20 训练/验证划分 + 早停
- 导出最佳模型和全量样本的潜表示 mu (Z)
"""

import json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ==================== 配置区 ====================
# 把下面路径改为你本地的 processed_dataset 路径


from radar.paths import DATA_V2_PROCESSED, OUT_STATE_VAE

DATA_DIR = DATA_V2_PROCESSED
print("StateVAE数据路径为：", DATA_DIR)


# 训练超参数与路径：
# DATA_DIR 指向你保存的 S_std.npy / A_std.npy / R_std.npy。
# LATENT_DIM 潜空间维度 z 的大小。
# HIDDEN 隐层宽度。
# LR 学习率；BATCH_SIZE 批大小；MAX_EPOCHS 最大轮次。
# PATIENCE 早停耐心值；验证集指标连续 PATIENCE 轮不变好就停止。
# BETA 为 β-VAE 的 KL 权重（MSE + β*KLD）。
# VAL_RATIO 验证集比例；SEED 随机种子，保证可复现。
LATENT_DIM = 32
HIDDEN = 128
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 200
PATIENCE = 10         # 多少轮验证集不提升就早停
BETA = 1.0            # KL 权重
VAL_RATIO = 0.2
SEED = 42


OUT_DIR = OUT_STATE_VAE
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "state_vae.pt"
EMB_PATH   = OUT_DIR / "state_embeddings_mu.npy"
METRICS_PATH = OUT_DIR / "metrics.json"

# ==== 新增：更宽/更深网络与可选超参 ====
# 编码器/解码器的隐藏层配置（更宽更深）
ENC_HIDDEN = [512, 512, 512, 256]
DEC_HIDDEN = [256, 512, 512, 512]
# 正则与归一化/残差
DROPOUT = 0.10
USE_LAYERNORM = True
USE_RESIDUAL  = True   # 仅在层的 in_dim == out_dim 时生效
# （可选）使用 β 退火：前期重构为主，后期逐步加强 KL
USE_BETA_COSINE_ANNEAL = True




# ==================== 工具函数 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==================== 数据集封装 ====================
class SAROfflineDataset(Dataset):
    """
    一个用于离线强化学习的 (S, A, R) 数据集
    - S: [N, s_dim] (标准化后的状态)
    - A: [N, a_dim] (标准化后的动作) —— VAE不使用  但保留以便后续扩展
    - R: [N, 1]     (标准化后的奖励) —— VAE不使用  但保留以便后续扩展
    """

    #把 NumPy 数组转成 float32 的 Tensor，长度必须一致
    def __init__(self, S, A, R):
        self.S = torch.as_tensor(S, dtype=torch.float32)
        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)
        assert len(self.S) == len(self.A) == len(self.R), "S/A/R 数量不一致"

    def __len__(self): return len(self.S)

    def __getitem__(self, i):
        return self.S[i], self.A[i], self.R[i]



# ==================== 模型与损失 ====================
class StateVAE(nn.Module):
    def __init__(self, s_dim, latent_dim=16, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, s_dim),
        )

    def encode(self, s):
        h = self.enc(s)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, s):
        mu, logvar = self.encode(s)
        z = self.reparam(mu, logvar)
        s_rec = self.dec(z)
        return s_rec, mu, logvar, z


def vae_loss(s, s_rec, mu, logvar, beta=1.0):
    # 重构误差用 MSE；KLD 使用标准公式
    rec = F.mse_loss(s_rec, s, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld, {"rec": rec.item(), "kld": kld.item()}





# ==================== 主流程 ====================
def train_vae_main():
    # 1) 加载数据，读取标准化后的数据
    S = np.load(DATA_DIR / "S_std.npy")
    A = np.load(DATA_DIR / "A_std.npy")
    R = np.load(DATA_DIR / "R_std.npy")

    #封装数据集，并按 80/20 切分训练/验证；用固定种子保证可复现
    dataset = SAROfflineDataset(S, A, R)
    N = len(dataset)
    val_size = int(N * VAL_RATIO)
    train_size = N - val_size

    # 固定划分，保证可复现
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    #构建 DataLoader；训练集随机打乱，验证集不打乱
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


    # 2) 定义模型/优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_dim = S.shape[1]
    model = StateVAE(s_dim=s_dim, latent_dim=LATENT_DIM, hidden=HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) 训练 + 验证 + 早停
    best_val = float("inf")
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- train ----
        model.train()
        tr_cnt = 0
        tr_rec_sum, tr_kld_sum = 0.0, 0.0
        for S_b, _, _ in train_loader:
            S_b = S_b.to(device)
            S_rec, mu, logvar, _ = model(S_b)
            loss, parts = vae_loss(S_b, S_rec, mu, logvar, beta=BETA)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = S_b.size(0)
            tr_cnt += bs
            tr_rec_sum += parts["rec"] * bs
            tr_kld_sum += parts["kld"] * bs

        tr_rec = tr_rec_sum / tr_cnt
        tr_kld = tr_kld_sum / tr_cnt

        # ---- val ----
        model.eval()
        va_cnt = 0
        va_rec_sum, va_kld_sum = 0.0, 0.0
        with torch.no_grad():
            for S_b, _, _ in val_loader:
                S_b = S_b.to(device)
                S_rec, mu, logvar, _ = model(S_b)
                _, parts = vae_loss(S_b, S_rec, mu, logvar, beta=BETA)
                bs = S_b.size(0)
                va_cnt += bs
                va_rec_sum += parts["rec"] * bs
                va_kld_sum += parts["kld"] * bs

        va_rec = va_rec_sum / va_cnt
        va_kld = va_kld_sum / va_cnt
        va_total = va_rec + BETA * va_kld
        va_rmse = float(np.sqrt(va_rec))

        row = {
            "epoch": epoch,
            "train_rec_mse": tr_rec,
            "train_kld": tr_kld,
            "val_rec_mse": va_rec,
            "val_rec_rmse": va_rmse,
            "val_kld": va_kld,
            "val_total": va_total,
        }
        history.append(row)

        print(f"Epoch {epoch:03d} | "
              f"train_rec={tr_rec:.6f}  train_kld={tr_kld:.6f} | "
              f"val_rec={va_rec:.6f} (RMSE={va_rmse:.6f})  val_kld={va_kld:.6f}  total={va_total:.6f}")

        # 早停
        if va_total < best_val - 1e-6:
            best_val = va_total
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val total={best_val:.6f}")
                break

    # 4) 载入最好模型，导出全量潜表示 mu
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    all_mu = []
    with torch.no_grad():
        for i in range(0, N, 2048):
            S_chunk = torch.as_tensor(S[i:i+2048], dtype=torch.float32, device=device)
            mu, _ = model.encode(S_chunk)
            all_mu.append(mu.cpu().numpy())
    Z = np.vstack(all_mu)
    np.save(EMB_PATH, Z)

    summary = {
        "s_dim": int(s_dim),
        "latent_dim": LATENT_DIM,
        "epochs_run": len(history),
        "best_val_total": float(best_val),
        "best_model_path": str(MODEL_PATH),
        "embeddings_mu_path": str(EMB_PATH),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"history": history, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))
    # state_vae_runs\state_vae.pt
    # 含义：训练好的 VAE 模型权重（PyTorch 的 state_dict）。
    # 用途：后续推理/再训练时加载模型，做重构、编码到潜空间、或接下游任务。

    # state_vae_runs\state_embeddings_mu.npy
    # 含义：把训练/验证集样本送入编码器后得到的潜变量的均值向量 μ（per-sample），形状一般是 [N, latent_dim]。
    # 用途：快速做可视化（t-SNE/UMAP）、聚类、下游简单模型（线性/MLP 头），或作为压缩后的状态表示供 RL/BC 使用。

if __name__ == "__main__":
    train_vae_main()
