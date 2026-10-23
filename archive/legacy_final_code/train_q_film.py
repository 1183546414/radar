# -*- coding: utf-8 -*-
"""
改进版双塔 + FiLM 网络训练脚本
增强方向：
  1. 显式统计特征：范数 / 余弦 / 比值 / 差分均值
  2. 显式交互：乘积(S*A)、差分(|S-A|)
  3. Bottleneck + LayerNorm 提升稳定性
  4. 训练时轻度噪声增强
"""
import os, json, math, random, pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------- Config --------------------
SEED = 42
VAL_RATIO = 0.2
BATCH_SIZE = 256
LR = 1e-3
MAX_EPOCHS = 400
PATIENCE = 40
WEIGHT_DECAY = 3e-4
DROPOUT = 0.2

S_HIDDEN = [512, 512, 256]
A_HIDDEN = [256, 256]
FUSION_HIDDEN = [256, 128]

DATA_DIR = Path("../dataset/data_2/json/processed_dataset")
OUT_DIR = Path("../result/film_advanced")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Utils --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def huber_loss(pred, target, delta=1.0):
    diff = torch.abs(pred - target)
    return torch.where(diff < delta, 0.5 * diff**2, delta*(diff - 0.5*delta))

def evaluate(y_true, y_pred, sc_R=None):
    rmse_std = math.sqrt(np.mean((y_true - y_pred)**2))
    mae_std = mean_absolute_error(y_true, y_pred)
    r2_std = r2_score(y_true, y_pred)
    pearson_std = np.corrcoef(y_true, y_pred)[0,1]
    metrics = dict(rmse_std=rmse_std, mae_std=mae_std, r2_std=r2_std, pearson_std=pearson_std)
    if sc_R:
        y_true_orig = sc_R.inverse_transform(y_true.reshape(-1,1)).reshape(-1)
        y_pred_orig = sc_R.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)
        rmse = math.sqrt(np.mean((y_true_orig - y_pred_orig)**2))
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        r2 = r2_score(y_true_orig, y_pred_orig)
        pearson = np.corrcoef(y_true_orig, y_pred_orig)[0,1]
        metrics.update(dict(rmse_orig=rmse, mae_orig=mae, r2_orig=r2, pearson_orig=pearson))
    return metrics

# -------------------- Feature Enhancement --------------------
def build_extra_features(S, A):
    """构造显式统计特征"""
    s_norm = np.linalg.norm(S, axis=1, keepdims=True)
    a_norm = np.linalg.norm(A, axis=1, keepdims=True)
    cosine = np.sum(S * A.mean(1, keepdims=True), axis=1, keepdims=True) / (s_norm * a_norm + 1e-8)
    diff = np.abs(S - A.mean(1, keepdims=True))
    diff_mean = diff.mean(1, keepdims=True)
    ratio = a_norm / (s_norm + 1e-8)
    return np.concatenate([s_norm, a_norm, cosine, diff_mean, ratio], axis=1)

# -------------------- Dataset --------------------
class SAROfflineDataset(Dataset):
    def __init__(self, S, A, R, sc_R=None, train=True):
        self.S, self.A, self.R = S, A, R
        self.sc_R = sc_R
        self.train = train
        self.extra = build_extra_features(S, A)
    def __len__(self): return len(self.R)
    def __getitem__(self, idx):
        s = torch.tensor(self.S[idx], dtype=torch.float32)
        a = torch.tensor(self.A[idx], dtype=torch.float32)
        e = torch.tensor(self.extra[idx], dtype=torch.float32)
        r = torch.tensor(self.R[idx], dtype=torch.float32)
        if self.train:
            s += 0.05 * torch.randn_like(s)
            a += 0.05 * torch.randn_like(a)
        return s, a, e, r

# -------------------- MLP + FiLM --------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class FiLM(nn.Module):
    def __init__(self, a_dim, s_dim):
        super().__init__()
        self.gamma = nn.Linear(a_dim, s_dim)
        self.beta = nn.Linear(a_dim, s_dim)
    def forward(self, s, a):
        gamma, beta = self.gamma(a), self.beta(a)
        return gamma * s + beta

# -------------------- Q Network --------------------
class QNet(nn.Module):
    def __init__(self, s_dim, a_dim, extra_dim=5):
        super().__init__()
        self.s_enc = MLP(s_dim, S_HIDDEN, dropout=DROPOUT)
        self.a_enc = MLP(a_dim, A_HIDDEN, dropout=DROPOUT)
        self.extra_enc = MLP(extra_dim, [64, 32], dropout=DROPOUT)
        self.film = FiLM(A_HIDDEN[-1], S_HIDDEN[-1])
        self.bottleneck = nn.Linear(S_HIDDEN[-1], 128)
        self.ln = nn.LayerNorm(128)

        # 🔧 修复处：输入维度改为 928
        self.fuse = MLP(928, FUSION_HIDDEN, dropout=DROPOUT)
        self.head = nn.Linear(FUSION_HIDDEN[-1], 1)

    def forward(self, s, a, e):
        h_s = self.s_enc(s)
        h_a = self.a_enc(a)
        h_film = self.film(h_s, h_a)
        h_film = self.ln(self.bottleneck(h_film))
        h_mul = h_s * h_a
        h_diff = torch.abs(h_s - h_a)
        h_extra = self.extra_enc(e)
        h = torch.cat([h_film, h_mul, h_diff, h_a, h_extra], dim=-1)
        return self.head(self.fuse(h)).squeeze(-1)

# -------------------- Training --------------------
def main():
    set_seed(SEED)
    S = np.load(DATA_DIR/"S_std.npy")
    A = np.load(DATA_DIR/"A_std.npy")
    R = np.load(DATA_DIR/"R_std.npy").reshape(-1)
    art = DATA_DIR/"artifacts.pkl"
    sc_R = None
    if art.exists():
        with open(art, "rb") as f:
            sc_R = pickle.load(f).get("sc_R", None)
        if sc_R is not None:
            print("Loaded sc_R for original-scale metrics.")

    idx = np.random.permutation(len(R))
    val_sz = int(len(R)*VAL_RATIO)
    val_idx, tr_idx = idx[:val_sz], idx[val_sz:]
    tr_set = SAROfflineDataset(S[tr_idx], A[tr_idx], R[tr_idx], sc_R, train=True)
    va_set = SAROfflineDataset(S[val_idx], A[val_idx], R[val_idx], sc_R, train=False)
    tr_loader = DataLoader(tr_set, BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_set, BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QNet(S.shape[1], A.shape[1], extra_dim=5).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    best_rmse, best_state, wait = 1e9, None, 0

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        losses = []
        for s, a, e, r in tr_loader:
            s, a, e, r = s.to(device), a.to(device), e.to(device), r.to(device)
            pred = model(s, a, e)
            loss = huber_loss(pred, r).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())

        # validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for s, a, e, r in va_loader:
                pred = model(s.to(device), a.to(device), e.to(device)).cpu().numpy()
                preds.append(pred); gts.append(r.numpy())
        y_true, y_pred = np.concatenate(gts), np.concatenate(preds)
        metrics = evaluate(y_true, y_pred, sc_R)
        rmse = metrics["rmse_std"]
        sched.step(rmse)
        print(f"Epoch {epoch:03d} | train={np.mean(losses):.4f} | val_rmse={rmse:.4f} | r2={metrics['r2_std']:.3f} pearson={metrics['pearson_std']:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse; best_state = model.state_dict(); wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    torch.save(best_state, OUT_DIR/"best_model.pt")
    final_metrics = evaluate(y_true, y_pred, sc_R)



    print("Final metrics:", json.dumps(final_metrics, indent=2, ensure_ascii=False))



    
    with open(OUT_DIR/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
