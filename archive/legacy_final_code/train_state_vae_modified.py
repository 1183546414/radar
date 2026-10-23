# -*- coding: utf-8 -*-
"""
StateVAE: 训练与验证（更宽/更深 MLP 版）
- 从 .npy 加载标准化后的 S/A/R
- 使用更宽/更深的 StateVAE 重构状态 S
- 80/20 训练/验证划分 + 早停
- 导出最佳模型和全量样本的潜表示 mu (Z)

【说明】
- 本文件可**直接替换**你现有的 train_state_vae.py 使用。
- 我已用注释标出和原版相比的**关键改动**：# [CHANGED] / # [NEW]
- 训练/日志/输出接口保持兼容。
"""

# ====== 基础导入 ======
import json, random, math  # [CHANGED] 加入 math 以支持 β 退火
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ====== 数据路径与超参 ======
DATA_DIR = Path("./dataset/data_2/json/processed_dataset")
print("StateVAE数据路径为：")
print(DATA_DIR)

# ---- 训练超参（保持你的原风格命名；仅少量调整） ----
LATENT_DIM = 32                 # [CHANGED] 潜空间更大（原=16）。可试 32/48/64
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 200               # [CHANGED] 可跑更久，但有早停
PATIENCE = 10                  # 验证集不提升就早停
BETA = 1.0                     # KL 权重（若不开启退火则使用这个）
VAL_RATIO = 0.2
SEED = 42

OUT_DIR = Path("./result/state_vae_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "state_vae.pt"
EMB_PATH   = OUT_DIR / "state_embeddings_mu.npy"
METRICS_PATH = OUT_DIR / "metrics.json"

# ====== [NEW] 更宽/更深网络配置 & 正则 ======
ENC_HIDDEN = [512, 512, 512, 256]   # [NEW] 编码器隐藏层
DEC_HIDDEN = [256, 512, 512, 512]   # [NEW] 解码器隐藏层
DROPOUT = 0.10                      # [NEW]
USE_LAYERNORM = True                # [NEW]
USE_RESIDUAL  = True                # [NEW] 仅在 in_dim == out_dim 的层内生效
USE_BETA_COSINE_ANNEAL = True       # [NEW] 使用 β 余弦退火（0→1），前期重构主导

# ====== 工具函数 ======
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====== 数据加载 ======
def load_std_arrays(data_dir: Path):
    S = np.load(data_dir / "S_std.npy")  # [N, s_dim]
    A = np.load(data_dir / "A_std.npy")  # [N, a_dim]
    R = np.load(data_dir / "R_std.npy")  # [N, 1]
    return S, A, R

# ====== 数据集封装 ======
class SAROfflineDataset(Dataset):
    """
    一个用于离线强化学习的 (S, A, R) 数据集
    - S: [N, s_dim] (标准化后的状态)
    - A: [N, a_dim] (标准化后的动作)
    - R: [N, 1]     (标准化后的奖励)
    训练本 VAE 只用 S，保留 A/R 以便后续扩展。
    """
    def __init__(self, S, A, R):
        self.S = torch.as_tensor(S, dtype=torch.float32)
        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)
        assert len(self.S) == len(self.A) == len(self.R)
    def __len__(self):
        return len(self.S)
    def __getitem__(self, i):
        return self.S[i], self.A[i], self.R[i]

# ====== [NEW] 更宽/更深的 MLP 组件 ======
class MLPBlock(nn.Module):
    """Linear -> GELU -> (Dropout) -> (LayerNorm) -> (+Residual)"""
    def __init__(self, in_dim, out_dim, dropout=0.0, layernorm=True, residual=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(out_dim) if layernorm else nn.Identity()
        self.use_residual = residual and (in_dim == out_dim)
    def forward(self, x):
        y = self.fc(x)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.ln(y)
        if self.use_residual:
            y = y + x
        return y

class StackedMLP(nn.Module):
    def __init__(self, in_dim, layer_sizes, dropout=0.1, layernorm=True, residual=True):
        super().__init__()
        layers = []
        last = in_dim
        for h in layer_sizes:
            layers.append(MLPBlock(last, h, dropout=dropout, layernorm=layernorm, residual=residual))
            last = h
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ====== [CHANGED] StateVAE：更宽/更深的编码器/解码器 ======
class StateVAE(nn.Module):
    def __init__(self, s_dim, latent_dim=LATENT_DIM,
                 enc_hidden=ENC_HIDDEN, dec_hidden=DEC_HIDDEN,
                 dropout=DROPOUT, layernorm=USE_LAYERNORM, residual=USE_RESIDUAL):
        super().__init__()
        self.s_dim = s_dim
        self.latent_dim = latent_dim
        # encoder
        self.encoder = StackedMLP(s_dim, enc_hidden, dropout=dropout, layernorm=layernorm, residual=residual)
        enc_out = enc_hidden[-1]
        self.mu = nn.Linear(enc_out, latent_dim)
        self.logvar = nn.Linear(enc_out, latent_dim)
        nn.init.xavier_uniform_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.xavier_uniform_(self.logvar.weight); nn.init.zeros_(self.logvar.bias)
        # decoder
        self.dec_in = nn.Linear(latent_dim, dec_hidden[0])
        nn.init.xavier_uniform_(self.dec_in.weight); nn.init.zeros_(self.dec_in.bias)
        self.decoder = StackedMLP(dec_hidden[0], dec_hidden[1:], dropout=dropout, layernorm=layernorm, residual=residual)
        self.dec_out = nn.Linear(dec_hidden[-1], s_dim)
        nn.init.xavier_uniform_(self.dec_out.weight); nn.init.zeros_(self.dec_out.bias)
    def encode(self, s):
        h = self.encoder(s)
        return self.mu(h), self.logvar(h)
    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps
    def forward(self, s):
        mu, logvar = self.encode(s)
        z = self.reparam(mu, logvar)
        h = F.gelu(self.dec_in(z))
        h = self.decoder(h)
        s_rec = self.dec_out(h)
        return s_rec, mu, logvar, z

# ====== 损失函数（接口不变） ======
def vae_loss(s, s_rec, mu, logvar, beta=1.0):
    rec = F.mse_loss(s_rec, s, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld, {"rec": rec.item(), "kld": kld.item()}

# ====== [NEW] 余弦 β 退火（0 -> 1） ======
def cosine_beta(step: int, total_steps: int, beta_max: float = 1.0, beta_min: float = 0.0) -> float:
    if total_steps <= 1:
        return beta_max
    t = step / (total_steps - 1)
    return beta_min + 0.5 * (beta_max - beta_min) * (1 - math.cos(math.pi * t))

# ====== 训练主流程 ======
def train_vae_main():
    set_seed(SEED)

    # 1) 加载数据
    S, A, R = load_std_arrays(DATA_DIR)
    dataset = SAROfflineDataset(S, A, R)
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 2) 定义模型/优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_dim = S.shape[1]
    model = StateVAE(
        s_dim=s_dim,
        latent_dim=LATENT_DIM,              # [CHANGED]
        enc_hidden=ENC_HIDDEN,              # [NEW]
        dec_hidden=DEC_HIDDEN,              # [NEW]
        dropout=DROPOUT,                    # [NEW]
        layernorm=USE_LAYERNORM,            # [NEW]
        residual=USE_RESIDUAL,              # [NEW]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) 训练 + 验证 + 早停
    best_val = float("inf")
    wait = 0
    history = []

    # [NEW] β 退火步数统计
    total_steps = (len(train_loader) * MAX_EPOCHS) if USE_BETA_COSINE_ANNEAL else 1
    global_step = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- train ----
        model.train()
        tr_cnt = 0
        tr_rec_sum, tr_kld_sum = 0.0, 0.0
        for S_b, _, _ in train_loader:
            S_b = S_b.to(device)
            S_rec, mu, logvar, _ = model(S_b)

            # [CHANGED] 支持 β 退火
            beta_now = cosine_beta(global_step, total_steps, 1.0, 0.0) if USE_BETA_COSINE_ANNEAL else BETA
            loss, parts = vae_loss(S_b, S_rec, mu, logvar, beta=beta_now)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            global_step += 1

            bs = S_b.size(0)
            tr_cnt += bs
            tr_rec_sum += parts["rec"] * bs
            tr_kld_sum += parts["kld"] * bs

        tr_rec = tr_rec_sum / tr_cnt
        tr_kld = tr_kld_sum / tr_cnt

        # ---- validate ----
        model.eval()
        va_cnt = 0
        va_rec_sum, va_kld_sum = 0.0, 0.0
        with torch.no_grad():
            for S_b, _, _ in val_loader:
                S_b = S_b.to(device)
                S_rec, mu, logvar, _ = model(S_b)
                beta_eval = 1.0 if USE_BETA_COSINE_ANNEAL else BETA
                _, parts = vae_loss(S_b, S_rec, mu, logvar, beta=beta_eval)
                bs = S_b.size(0)
                va_cnt += bs
                va_rec_sum += parts["rec"] * bs
                va_kld_sum += parts["kld"] * bs

        va_rec = va_rec_sum / va_cnt
        va_kld = va_kld_sum / va_cnt
        va_total = va_rec + (1.0 if USE_BETA_COSINE_ANNEAL else BETA) * va_kld  # [CHANGED]
        va_rmse = float(np.sqrt(va_rec))

        # 日志打印（保持你原有格式）
        print(
            f"Epoch {epoch:03d} | "
            f"train_rec={tr_rec:.6f}  train_kld={tr_kld:.6f} | "
            f"val_rec={va_rec:.6f} (RMSE={va_rmse:.6f})  val_kld={va_kld:.6f}  total={va_total:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_rec_mse": tr_rec,
            "train_kld": tr_kld,
            "val_rec_mse": va_rec,
            "val_rec_rmse": va_rmse,
            "val_kld": va_kld,
            "val_total": va_total,
        })

        # ---- early stopping ----
        if va_total < best_val - 1e-8:
            best_val = va_total
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # 4) 载入最佳模型 & 导出全量样本的 μ
    best = StateVAE(s_dim=S.shape[1], latent_dim=LATENT_DIM).to(device)
    best.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    best.eval()

    with torch.no_grad():
        mu_all = []
        loader_full = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for S_b, _, _ in loader_full:
            S_b = S_b.to(device)
            mu, logvar = best.encode(S_b)
            mu_all.append(mu.cpu().numpy())
        mu_all = np.concatenate(mu_all, axis=0)
        np.save(EMB_PATH, mu_all)

    # 5) 写 metrics.json
    summary = {
        "s_dim": int(S.shape[1]),
        "latent_dim": int(LATENT_DIM),
        "epochs_run": int(len(history)),
        "best_val_total": float(best_val),
        "best_model_path": str(MODEL_PATH).replace("/", "\\"),
        "embeddings_mu_path": str(EMB_PATH).replace("/", "\\"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"history": history, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    train_vae_main()
