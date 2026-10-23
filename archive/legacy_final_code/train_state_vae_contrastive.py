# -*- coding: utf-8 -*-
"""
Train State VAE with Contrastive Head / AE mode
目标：
1) 把重构 MSE 尽量压低（逼近 1e-3）
2) 同时提升潜向量的可分性（不同输入在 μ 上更有区分度）

做法：
- 提供两种模式：MODEL_TYPE = "VAE" 或 "AE"
  - AE：无 KL、无采样，重构精度最佳；潜向量取 encoder 输出
  - VAE：保留 KL，但 β 上限降到 0.1（可继续降到 0.05/0.01），验证/导出统一用 μ（确定性）
- 更大容量：enc=[1024,1024,768,512] dec=[512,768,1024,1024]；latent_dim=64
- 解码器输出做**残差重构**：pred = skip_coef * x + residual_head(h)
- 加入**对比学习（NT-Xent）**到 μ 上，权重 CONTRAST_W
- 可选：契合 Denoising-AE 思路的小噪声输入（AUG_NOISE_STD_FOR_REC）

使用：
- 直接运行本脚本；超参在顶部可切换/调节
- 如只关心“把误差打到 1e-3”：设 MODEL_TYPE="AE"，把 β 无视；适度增大容量与训练轮数
"""

import json, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ================= 基础设置 =================
DATA_DIR = Path("./dataset/data_2/json/processed_dataset")
print("数据路径：", DATA_DIR)

# ---- 任务偏好开关 ----
MODEL_TYPE = "AE"          # 用 AE 去掉采样噪声；潜空间更干净
USE_CONTRAST = True        # 保持开启

# ---- 模型结构与超参 ----
LATENT_DIM = 128           # ↑ 给判别留空间（96/128皆可，先用128）
ENC_HIDDEN = [1024, 1024, 1024, 768]
DEC_HIDDEN = [768, 1024, 1024, 1024]
DROPOUT = 0.05
USE_LAYERNORM = True
USE_RESIDUAL  = True
SKIP_COEF = 1.0

# ---- 训练超参 ----
LR = 1e-3
BATCH_SIZE = 512           # ↑ 批越大，批内负样本越多；看显存可用 256/384/512
MAX_EPOCHS = 300
PATIENCE = 20
VAL_RATIO = 0.2
SEED = 42

# ---- VAE 相关（若日后想切回 VAE 再用）----
BETA_FIXED = 0.05          # 仅在 MODEL_TYPE="VAE" 时生效
USE_BETA_COSINE = True

# ---- 对比学习（NT-Xent） ----
CONTRAST_W = 3.0           # ↑ 更强的判别约束
NTXENT_TAU = 0.1           # ↓ 温度更低，拉开更猛
AUG_NOISE_STD = 0.07       # 视图噪声稍强
AUG_DROP_PROB = 0.15

# ---- Denoising-AE（只作用到重构分支输入） ----
AUG_NOISE_STD_FOR_REC = 0.0  # 先关；若过拟合再开到 0.01~0.02

# ---- 输出路径 ----
OUT_DIR = Path("./result/state_vae_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "state_ae_or_vae.pt"
EMB_PATH   = OUT_DIR / "state_embeddings_mu.npy"
METRICS_PATH = OUT_DIR / "metrics.json"

# ================= 工具函数 =================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SAROfflineDataset(Dataset):
    def __init__(self, S, A, R):
        self.S = torch.as_tensor(S, dtype=torch.float32)
        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)
    def __len__(self): return len(self.S)
    def __getitem__(self, i): return self.S[i], self.A[i], self.R[i]

# ================= 模型定义 =================
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, layernorm=True, residual=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.ln = nn.LayerNorm(out_dim) if layernorm else nn.Identity()
        self.use_residual = residual and (in_dim==out_dim)
    def forward(self, x):
        y = F.gelu(self.fc(x)); y = self.dropout(y); y = self.ln(y)
        return y + x if self.use_residual else y

class StackedMLP(nn.Module):
    def __init__(self, in_dim, layers, dropout=0.0, layernorm=True, residual=True):
        super().__init__()
        seq=[]; last=in_dim
        for h in layers:
            seq.append(MLPBlock(last,h,dropout=dropout,layernorm=layernorm,residual=residual))
            last=h
        self.net = nn.Sequential(*seq)
    def forward(self,x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, s_dim, latent_dim):
        super().__init__()
        self.trunk = StackedMLP(s_dim, ENC_HIDDEN, dropout=DROPOUT, layernorm=USE_LAYERNORM, residual=USE_RESIDUAL)
        h = ENC_HIDDEN[-1]
        self.mu = nn.Linear(h, latent_dim)
        nn.init.xavier_uniform_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        if MODEL_TYPE=="VAE":
            self.logvar = nn.Linear(h, latent_dim)
            nn.init.xavier_uniform_(self.logvar.weight); nn.init.zeros_(self.logvar.bias)
        else:
            self.logvar = None
    def forward(self, s):
        h = self.trunk(s)
        mu = self.mu(h)
        if self.logvar is None:
            return mu, None
        logvar = self.logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, s_dim, latent_dim):
        super().__init__()
        self.inp = nn.Linear(latent_dim, DEC_HIDDEN[0])
        nn.init.xavier_uniform_(self.inp.weight); nn.init.zeros_(self.inp.bias)
        self.trunk = StackedMLP(DEC_HIDDEN[0], DEC_HIDDEN[1:], dropout=DROPOUT, layernorm=USE_LAYERNORM, residual=USE_RESIDUAL)
        self.residual_head = nn.Linear(DEC_HIDDEN[-1], s_dim)
        nn.init.xavier_uniform_(self.residual_head.weight); nn.init.zeros_(self.residual_head.bias)
    def forward(self, z, x_skip=None):
        h = F.gelu(self.inp(z)); h = self.trunk(h)
        residual = self.residual_head(h)
        if x_skip is None:
            return residual
        return SKIP_COEF * x_skip + residual   # 残差重构

class Model(nn.Module):
    def __init__(self, s_dim, latent_dim):
        super().__init__()
        self.enc = Encoder(s_dim, latent_dim)
        self.dec = Decoder(s_dim, latent_dim)
        self.s_dim = s_dim
    def reparam(self, mu, logvar):
        if MODEL_TYPE=="AE" or logvar is None:
            return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, s):
        mu, logvar = self.enc(s)
        z = self.reparam(mu, logvar)
        s_rec = self.dec(z, x_skip=s)  # 残差重构
        return s_rec, mu, logvar, z
    def encode(self, s):
        return self.enc(s)

# ================= Losses =================
def vae_parts(s, s_rec, mu, logvar, beta):
    rec = F.mse_loss(s_rec, s, reduction="mean")
    if MODEL_TYPE=="AE" or logvar is None:
        kld = torch.tensor(0.0, device=s.device)
    else:
        kld = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec, kld, rec + beta * kld

def cosine_beta(step:int, total:int, beta_max:float) -> float:
    if MODEL_TYPE=="AE": return 0.0
    if (not USE_BETA_COSINE) or total<=1: return beta_max
    t = step/max(total-1,1)
    return 0.5*beta_max*(1 - math.cos(math.pi*t))

# ------- Contrastive (NT-Xent on μ) -------
def augment_state(x: torch.Tensor) -> torch.Tensor:
    if AUG_NOISE_STD>0:
        x = x + AUG_NOISE_STD*torch.randn_like(x)
    if AUG_DROP_PROB>0:
        mask = (torch.rand_like(x) > AUG_DROP_PROB).float()
        x = x * mask
    return x

def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = NTXENT_TAU) -> torch.Tensor:
    z1 = l2_normalize(z1); z2 = l2_normalize(z2)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # 2N x D
    sim = torch.matmul(z, z.t())            # 2N x 2N
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float('-inf'))
    pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    logits = sim / temperature
    labels = pos
    return F.cross_entropy(logits, labels)

# ================= 训练流程 =================
def train_vae_main():
    set_seed(SEED)
    S = np.load(DATA_DIR/"S_std.npy"); A = np.load(DATA_DIR/"A_std.npy"); R = np.load(DATA_DIR/"R_std.npy")
    dataset = SAROfflineDataset(S, A, R)
    val_size = int(len(dataset)*VAL_RATIO); train_size=len(dataset)-val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_dim = S.shape[1]
    model = Model(s_dim, LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float('inf'); wait=0; history=[]
    total_steps = len(train_loader)*MAX_EPOCHS
    global_step = 0

    for epoch in range(1, MAX_EPOCHS+1):
        model.train(); tr_cnt=0
        tr_rec_sum=0.0; tr_kld_sum=0.0
        for S_b,_,_ in train_loader:
            S_b = S_b.to(device)

            # （可选）Denoising 正则：仅给重构分支加很小噪声
            S_b_rec = S_b
            if AUG_NOISE_STD_FOR_REC>0:
                S_b_rec = S_b + AUG_NOISE_STD_FOR_REC*torch.randn_like(S_b)

            S_rec, mu, logvar, z = model(S_b_rec)
            beta_now = cosine_beta(global_step, total_steps, BETA_FIXED)
            rec, kld, loss = vae_parts(S_b, S_rec, mu, logvar, beta_now)  # 注意 rec 仍然对 S_b 评估

            # Contrastive on μ（两视图）
            if USE_CONTRAST:
                with torch.no_grad():
                    S1 = augment_state(S_b)
                    S2 = augment_state(S_b)
                mu1, _ = model.encode(S1)
                mu2, _ = model.encode(S2)
                c_loss = nt_xent_loss(mu1, mu2)
                loss = loss + CONTRAST_W * c_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); global_step += 1

            bs = S_b.size(0); tr_cnt += bs
            tr_rec_sum += rec.item()*bs; tr_kld_sum += kld.item()*bs

        tr_rec = tr_rec_sum/tr_cnt; tr_kld = tr_kld_sum/tr_cnt

        # ---- validate（确定性评估）----
        model.eval(); va_cnt=0; va_rec_sum=0.0; va_kld_sum=0.0
        with torch.no_grad():
            for S_b,_,_ in val_loader:
                S_b = S_b.to(device)
                mu, logvar = model.encode(S_b)
                # 确定性：用 μ 直接解码
                S_rec = model.dec(mu, x_skip=S_b)
                rec, kld, _ = vae_parts(S_b, S_rec, mu, logvar, beta=BETA_FIXED if MODEL_TYPE=="VAE" else 0.0)
                bs = S_b.size(0); va_cnt += bs
                va_rec_sum += rec.item()*bs; va_kld_sum += kld.item()*bs
        va_rec = va_rec_sum/va_cnt; va_kld = va_kld_sum/va_cnt
        va_total = va_rec + (BETA_FIXED if MODEL_TYPE=="VAE" else 0.0)*va_kld
        va_rmse = float(np.sqrt(va_rec))

        print(f"Epoch {epoch:03d} | train_rec={tr_rec:.6f}  train_kld={tr_kld:.6f} | "
              f"val_rec={va_rec:.6f} (RMSE={va_rmse:.6f})  val_kld={va_kld:.6f}  total={va_total:.6f}")
        history.append({"epoch":epoch,"train_rec_mse":tr_rec,"train_kld":tr_kld,"val_rec_mse":va_rec,
                        "val_rec_rmse":va_rmse,"val_kld":va_kld,"val_total":va_total})

        if va_total < best_val - 1e-9:
            best_val = va_total; wait=0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            wait += 1
            if wait>=PATIENCE: break

    # ---- 导出全量 μ（确定性）----
    best = Model(s_dim, LATENT_DIM).to(device)
    best.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    best.eval()
    with torch.no_grad():
        mu_all=[]
        loader_full = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for S_b,_,_ in loader_full:
            S_b = S_b.to(device)
            mu, _ = best.encode(S_b)
            mu_all.append(mu.cpu().numpy())
        mu_all = np.concatenate(mu_all, axis=0)
        np.save(EMB_PATH, mu_all)

    summary = {"s_dim": int(s_dim), "latent_dim": int(LATENT_DIM), "epochs_run": len(history),
               "best_val_total": float(best_val), "best_model_path": str(MODEL_PATH).replace("/","\\"),
               "embeddings_mu_path": str(EMB_PATH).replace("/","\\"),
               "model_type": MODEL_TYPE, "contrast_w": CONTRAST_W}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"history":history, "summary":summary}, f, ensure_ascii=False, indent=2)
    print("Done.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    train_vae_main()
