# -*- coding: utf-8 -*-
"""
Train State VAE with Contrastive Head / AE mode (Modified for CSV Input)
目标：
1) 读取 simulated_radar_jamming_flat_GHz_10k.csv 作为输入
2) 将干扰机参数 (Jammer_*) 处理为状态 S
3) 训练 VAE/AE 以提取隐向量
"""

import json
import random
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib  # 用于保存 scaler

# ====== 1. 配置参数 ======
# 输入数据路径
CSV_PATH = "radar-10.11/radar/dataset/data_4/simulated_radar_jamming_flat_GHz_10k.csv"
OUTPUT_DIR = Path("./radar-10.11/radar/dataset/data_4/vae_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练超参
MODEL_TYPE = "VAE"  # "VAE" (with KL) or "AE" (Determininstic)
LATENT_DIM = 64     # 隐向量维度
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 100    # 可以根据需要增加
PATIENCE = 15       # 早停忍耐轮数

# 对比学习/损失权重
CONTRAST_W = 0.1             # 对比损失权重
TEMP = 0.1                   # 温度系数
skip_coef = 1.0              # 残差连接系数
AUG_NOISE_STD_FOR_REC = 0.05 # 输入加噪增强鲁棒性

# Beta-VAE Schedule (KL 权重退火)
BETA_START = 0.0
BETA_STOP = 0.1     # VAE模式下建议较小 (0.01~0.1) 以保证重构质量
BETA_CYCLES = 1
BETA_RATIO = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== 2. 数据加载与预处理 (新增部分) ======

def load_and_process_data(csv_path, output_dir):
    print(f"正在读取数据: {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # --- 1. 处理状态 S (干扰机参数) ---
    # 分离类别特征和数值特征
    s_cat_cols = ['Jammer_Type']
    s_num_cols = [
        'Jammer_CarrierFreq_GHz', 'Jammer_Gain_dB', 'Jammer_Bandwidth_GHz',
        'Jammer_Sweep_Step_GHz', 'Jammer_Sweep_Power',
        'Jammer_IntSample_Dur_ns', 'Jammer_IntSample_Delay_ns', 'Jammer_IntSample_Intv_ns',
        'Jammer_FalseTarget_Num', 'Jammer_FalseTarget_Intv_m', 'Jammer_FalseTarget_Delay_m'
    ]
    
    # 处理类别特征 (One-Hot)
    enc_s = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    s_cat_encoded = enc_s.fit_transform(df[s_cat_cols])
    
    # 处理数值特征 (StandardScaler)
    scaler_s = StandardScaler()
    s_num_scaled = scaler_s.fit_transform(df[s_num_cols])
    
    # 合并为 S
    S = np.hstack([s_cat_encoded, s_num_scaled]).astype(np.float32)
    
    # --- 2. 处理动作 A (雷达参数) - 仅做简单处理用于Dataset占位 ---
    # 同样区分类别和数值
    a_cat_cols = ['Radar_Type', 'Radar_ChirpType']
    a_num_cols = ['Radar_CarrierFreq_GHz', 'Radar_Gain_dB', 'Radar_Bandwidth_GHz', 'Radar_PulseWidth_ns']
    
    enc_a = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    a_cat_encoded = enc_a.fit_transform(df[a_cat_cols])
    
    scaler_a = StandardScaler()
    a_num_scaled = scaler_a.fit_transform(df[a_num_cols])
    
    A = np.hstack([a_cat_encoded, a_num_scaled]).astype(np.float32)
    
    # --- 3. 处理奖励 R ---
    R = df[['SINR_dB']].values.astype(np.float32)
    # R 通常不需要标准化，或者只除以一个常数
    
    print(f"数据处理完成:")
    print(f"S shape: {S.shape} (Input to VAE)")
    print(f"A shape: {A.shape}")
    print(f"R shape: {R.shape}")
    
    # 保存预处理器，供 RL Agent 推理使用
    joblib.dump(scaler_s, output_dir / "scaler_s.joblib")
    joblib.dump(enc_s, output_dir / "encoder_s.joblib")
    joblib.dump(scaler_a, output_dir / "scaler_a.joblib") # 动作通常由Agent直接输出，但也可能需要逆变换
    print(f"预处理器已保存至 {output_dir}")
    
    return S, A, R

# 加载数据
S_raw, A_raw, R_raw = load_and_process_data(CSV_PATH, OUTPUT_DIR)
s_dim = S_raw.shape[1]

# ====== 3. 模型定义 (保持原有强力结构) ======

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x):
        return x + self.fc(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[1024, 1024, 768, 512]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(0.05)
            ))
            layers.append(ResidualBlock(h_d)) # 加强深度特征提取
            in_d = h_d
        self.extract = nn.Sequential(*layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.extract(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[512, 768, 1024, 1024]):
        super().__init__()
        layers = []
        in_d = latent_dim
        for h_d in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(0.05)
            ))
            layers.append(ResidualBlock(h_d))
            in_d = h_d
        self.decode_net = nn.Sequential(*layers)
        
        # 最后的残差预测头
        self.residual_head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.decode_net(z)
        return self.residual_head(h)

class StateVAE(nn.Module):
    def __init__(self, s_dim, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(s_dim, latent_dim)
        self.decoder = Decoder(latent_dim, s_dim)
        
        # 投影头 (用于对比学习 loss)
        self.proj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 32) # project to lower dim for contrastive loss
        )

    def reparameterize(self, mu, logvar):
        if MODEL_TYPE == "AE":
            return mu # AE 模式下直接通过
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # 残差重构: pred = x + delta
        # 但如果是标准化数据，直接预测 x 可能更好。
        # 这里使用 skip_coef * x + residual 的方式
        delta = self.decoder(z)
        recon_x = skip_coef * x + delta
        
        return recon_x, mu, logvar, z

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

# ====== 4. 工具函数与 Dataset ======

def nt_xent_loss(z1, z2, temperature=0.1):
    """
    SimCLR 风格的对比损失：
    z1, z2 是同一个样本经过不同 augment (dropout/noise) 得到的 latent
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0) # 2N, dim
    sim = torch.mm(z, z.t()) / temperature # 2N, 2N
    
    # Mask out self-similarity
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    
    positives = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2*N, 1)
    
    # Negative mask
    mask = torch.eye(2*N, device=z.device).bool()
    sim = sim.masked_fill(mask, -float('inf'))
    
    # InfoNCE
    logits = torch.cat([positives, sim], dim=1) # The positives are the first column? No.
    # 标准 InfoNCE 实现比较繁琐，这里简化：
    # nominator = exp(sim_pos)
    # denominator = sum(exp(sim_all))
    labels = torch.arange(2 * N, device=z.device)
    # 实际上 sim 矩阵里，对于 idx i (0~N-1), 正样本是 i+N
    # 对于 idx i (N~2N-1), 正样本是 i-N
    
    # 重新构建 label
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)], dim=0).to(z.device)
    
    loss = F.cross_entropy(sim, labels)
    return loss

class OfflineDataset(Dataset):
    def __init__(self, S, A, R):
        self.S = torch.FloatTensor(S)
        self.A = torch.FloatTensor(A)
        self.R = torch.FloatTensor(R)
    def __len__(self):
        return len(self.S)
    def __getitem__(self, idx):
        return self.S[idx], self.A[idx], self.R[idx]

def cyclcal_beta(step, total_steps, cycles=4, ratio=0.5, start=0.0, stop=1.0):
    # 简单的周期性退火
    period = total_steps // cycles
    step_in_period = step % period
    fraction = step_in_period / (period * ratio)
    if fraction > 1.0:
        return stop
    else:
        return start + (stop - start) * fraction

# ====== 5. 训练循环 ======

def train():
    # 划分数据集
    dataset = OfflineDataset(S_raw, A_raw, R_raw)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = StateVAE(s_dim, LATENT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    wait = 0
    history = []
    
    MODEL_PATH = OUTPUT_DIR / "best_vae_model.pth"

    total_steps = MAX_EPOCHS * len(train_loader)
    global_step = 0

    print(f"\nStart Training... Model Type: {MODEL_TYPE}")
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss_total = 0
        train_rec = 0
        train_kld = 0
        train_cont = 0

        for batch_S, _, _ in train_loader:
            batch_S = batch_S.to(device)
            
            # --- Data Augmentation for Contrastive / Denoising ---
            # 1. 原始样本
            recon_1, mu_1, logvar_1, z_1 = model(batch_S)
            
            # 2. 加噪样本 (模拟测量噪声)
            noise = torch.randn_like(batch_S) * AUG_NOISE_STD_FOR_REC
            recon_2, mu_2, logvar_2, z_2 = model(batch_S + noise)
            
            # --- Loss Calculation ---
            # 1. Reconstruction Loss (MSE) - 只需要尽量还原原始干净信号
            loss_rec = F.mse_loss(recon_1, batch_S) + F.mse_loss(recon_2, batch_S)
            
            # 2. KL Divergence
            if MODEL_TYPE == "VAE":
                # 计算两个流的 KL 平均
                kld_1 = -0.5 * torch.sum(1 + logvar_1 - mu_1.pow(2) - logvar_1.exp()) / batch_S.size(0)
                kld_2 = -0.5 * torch.sum(1 + logvar_2 - mu_2.pow(2) - logvar_2.exp()) / batch_S.size(0)
                loss_kld = (kld_1 + kld_2) / 2
                
                # Beta Annealing
                beta = cyclcal_beta(global_step, total_steps, BETA_CYCLES, BETA_RATIO, BETA_START, BETA_STOP)
            else:
                loss_kld = torch.tensor(0.0).to(device)
                beta = 0.0

            # 3. Contrastive Loss (Projected Latent)
            # 拉近同一个样本的 aug1 和 aug2，推开不同样本
            p1 = model.proj_head(z_1)
            p2 = model.proj_head(z_2)
            loss_contrast = nt_xent_loss(p1, p2, temperature=TEMP)

            # Total Loss
            loss = loss_rec + beta * loss_kld + CONTRAST_W * loss_contrast
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_rec += loss_rec.item()
            train_kld += loss_kld.item()
            train_cont += loss_contrast.item()
            global_step += 1

        # 平均 Log
        avg_rec = train_rec / len(train_loader)
        avg_kld = train_kld / len(train_loader)
        avg_cnt = train_cont / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_S, _, _ in val_loader:
                batch_S = batch_S.to(device)
                recon, mu, logvar, _ = model(batch_S)
                l_rec = F.mse_loss(recon, batch_S)
                if MODEL_TYPE == "VAE":
                    l_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_S.size(0)
                else:
                    l_kld = torch.tensor(0.0)
                val_loss += (l_rec + beta * l_kld).item() # 验证集通常不加对比loss

        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)

        print(f"Ep {epoch+1}/{MAX_EPOCHS} | Rec: {avg_rec:.5f} | KL: {avg_kld:.5f} (b={beta:.3f}) | Cont: {avg_cnt:.4f} | Val: {avg_val_loss:.5f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    print("Training Finished.")
    print(f"Best Model Saved to: {MODEL_PATH}")
    
    # 导出全量数据的隐向量 (Embeddings)
    # 用于后续 World Model 训练或可视化
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("Generating Embeddings for all data...")
    full_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    embeddings = []
    
    with torch.no_grad():
        for batch_S, _, _ in full_loader:
            batch_S = batch_S.to(device)
            mu, _ = model.encode(batch_S)
            embeddings.append(mu.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    np.save(OUTPUT_DIR / "S_embedded.npy", embeddings)
    print(f"Embeddings saved to {OUTPUT_DIR / 'S_embedded.npy'} shape: {embeddings.shape}")

if __name__ == "__main__":
    train()