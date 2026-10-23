# -*- coding: utf-8 -*-
"""
改进要点：
1) 设备管理：自动选择 GPU/CPU，并把模型与张量搬到对应 device。
2) 损失记录修正：按 “每个 epoch 的平均损失” 记录，并同时支持验证集评估。
3) 可选保存：--save-model 保存模型与 scaler；--save-plots 保存损失曲线与重建对比图到 --outdir。
4) 规范化流程：仅用训练集 fit MinMaxScaler，再对 train/val 分别 transform，避免数据泄露。
5) 其他：随机种子、命令行参数、异常与边界处理。
"""

import os
import json
import math
import pickle
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# ---------------------------
# 实用函数
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------------
# 数据集
# ---------------------------
class JammingDataset(Dataset):
    """VAE 自编码任务：输入=输出"""
    def __init__(self, tensor):
        self.data = tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x


# ---------------------------
# 模型定义：VAE
# ---------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.enc_fc = nn.Linear(input_dim, hidden_dim)
        self.enc_act = nn.ReLU(inplace=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_act = nn.ReLU(inplace=True)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = self.enc_act(self.enc_fc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 重参数化：z = μ + σ * ε
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_act(self.dec_fc1(z))
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """重建损失 + β * KL 散度"""
    recon_loss = nn.MSELoss(reduction="mean")(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


# ---------------------------
# 训练与验证
# ---------------------------
def run_epoch(model, dataloader, optimizer=None, device="cpu", beta=1.0):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_recon, total_kld, n = 0.0, 0.0, 0.0, 0

    for x, _ in dataloader:
        x = x.to(device, non_blocking=True)
        recon, mu, logvar = model(x)
        loss, recon_loss, kld = vae_loss(recon, x, mu, logvar, beta=beta)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kld += kld.item() * bs
        n += bs

    # 返回 epoch 平均指标
    return (total_loss / n, total_recon / n, total_kld / n)


# ---------------------------
# 主流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="VAE 改进版（设备管理、损失记录修正、可选保存与绘图）")
    # 数据路径（默认与原脚本一致的三份 JSON；若你的数据更多，可自行扩展）
    parser.add_argument("--base", type=str, default="../dataset", help="数据目录")
    parser.add_argument("--type-file", type=str, default="Encoded_Jamming_Type.json")
    parser.add_argument("--jammer-cf-file", type=str, default="Jammer_carrierFrequency.json")
    parser.add_argument("--jamming-params-file", type=str, default="Jamming_Params.json")

    # 训练超参
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0, help="β-VAE 权重（KL 系数）")
    parser.add_argument("--val-split", type=float, default=0.2, help="验证集比例 [0,1)")

    # 运行与保存
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./outputs")
    parser.add_argument("--save-model", action="store_true", help="保存模型与 scaler")
    parser.add_argument("--save-plots", action="store_true", help="保存曲线图与重建图")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")


    # -----------------------
    # 1) 读取原始 JSON
    # -----------------------
    base = args.base
    type_path = os.path.join(base, args.type_file)
    jammer_cf_path = os.path.join(base, args.jammer_cf_file)
    jamming_params_path = os.path.join(base, args.jamming_params_file)

    jamming_type = load_json(type_path)               # [N]，整数编码
    jammer_cf = load_json(jammer_cf_path)             # [N, 2]，载频与增益
    jamming_params = load_json(jamming_params_path)   # [N, K]，干扰参数向量

    # -----------------------
    # 2) 组装样本向量
    # -----------------------
    num_samples = min(len(jamming_type), len(jammer_cf), len(jamming_params))
    if num_samples == 0:
        raise ValueError("数据为空，请检查 JSON 文件内容与路径。")

    samples = []
    for i in range(num_samples):
        row = []
        # 干扰类型（单值）
        row.append(float(jamming_type[i]))
        # 干扰机载频/增益（2 值）
        # 确保是长度为 2 的列表/元组
        cf_pair = jammer_cf[i]
        if not isinstance(cf_pair, (list, tuple)) or len(cf_pair) < 2:
            raise ValueError(f"第 {i} 条 Jammer_carrierFrequency 非法：{cf_pair}")
        row.extend([float(cf_pair[0]), float(cf_pair[1])])
        # 干扰参数（若干值）
        params = jamming_params[i]
        if not isinstance(params, (list, tuple)):
            raise ValueError(f"第 {i} 条 Jamming_Params 非法：{params}")
        row.extend([float(x) for x in params])
        samples.append(row)

    X = np.asarray(samples, dtype=np.float32)
    n_feat = X.shape[1]
    print(f"[Info] 样本数: {X.shape[0]}, 特征维度: {n_feat}")

    # -----------------------
    # 3) 训练/验证划分
    # -----------------------
    val_ratio = max(0.0, min(0.9, args.val_split))
    n_total = X.shape[0]
    idx = np.arange(n_total)
    np.random.shuffle(idx)
    n_val = int(math.floor(n_total * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X[train_idx]
    X_val = X[val_idx] if n_val > 0 else None

    # -----------------------
    # 4) 规范化：仅用训练集 fit
    # -----------------------
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val) if X_val is not None else None

    # 转成张量并搬到设备
    train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_norm, dtype=torch.float32) if X_val_norm is not None else None

    # -----------------------
    # 5) DataLoader
    # -----------------------
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        JammingDataset(train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=False
    )
    val_loader = None
    if val_tensor is not None and len(val_tensor) > 0:
        val_loader = DataLoader(
            JammingDataset(val_tensor),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            drop_last=False
        )

    # -----------------------
    # 6) 模型与优化器
    # -----------------------
    vae = VAE(input_dim=n_feat, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # -----------------------
    # 7) 训练循环
    # -----------------------
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kld": [],
        "val_loss": [],
        "val_recon": [],
        "val_kld": []
    }

    for epoch in range(1, args.epochs + 1):
        # 训练
        tr_loss, tr_recon, tr_kld = run_epoch(vae, train_loader, optimizer=optimizer, device=device, beta=args.beta)
        history["train_loss"].append(tr_loss)
        history["train_recon"].append(tr_recon)
        history["train_kld"].append(tr_kld)

        # 验证
        if val_loader is not None:
            with torch.no_grad():
                vae.eval()
                va_loss, va_recon, va_kld = run_epoch(vae, val_loader, optimizer=None, device=device, beta=args.beta)
            history["val_loss"].append(va_loss)
            history["val_recon"].append(va_recon)
            history["val_kld"].append(va_kld)
            print(f"[Epoch {epoch:03d}] Train: {tr_loss:.6f} (recon {tr_recon:.6f}, kld {tr_kld:.6f}) | "
                  f"Val: {va_loss:.6f} (recon {va_recon:.6f}, kld {va_kld:.6f})")
        else:
            print(f"[Epoch {epoch:03d}] Train: {tr_loss:.6f} (recon {tr_recon:.6f}, kld {tr_kld:.6f})")

    # -----------------------
    # 8) 可选保存：模型 & scaler
    # -----------------------
    if args.save_model:
        model_path = os.path.join(args.outdir, "vae.pth")
        torch.save(vae.state_dict(), model_path)
        print(f"[Save] 模型已保存到: {model_path}")

        scaler_path = os.path.join(args.outdir, "minmax_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[Save] Scaler 已保存到: {scaler_path}")

        cfg_path = os.path.join(args.outdir, "train_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        print(f"[Save] 训练配置已保存到: {cfg_path}")

    # -----------------------
    # 9) 可选保存：曲线图 & 重建图
    # -----------------------
    if args.save_plots:
        # (a) 损失曲线
        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        if len(history["val_loss"]) > 0:
            plt.plot(history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VAE Loss Curve")
        plt.legend()
        loss_fig = os.path.join(args.outdir, "loss_curve.png")
        plt.savefig(loss_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Save] 损失曲线已保存到: {loss_fig}")

        # (b) 重建对比（从验证集取一条；若无验证集则取训练集第一条）
        vae.eval()
        if val_tensor is not None and len(val_tensor) > 0:
            x0 = val_tensor[0:1].to(device)
            raw0 = X_val[0:1]  # 未归一化（原始尺度用来 inverse 时对齐）
        else:
            x0 = train_tensor[0:1].to(device)
            raw0 = X_train[0:1]

        with torch.no_grad():
            recon0, _, _ = vae(x0)
        # 当前 recon0、x0 是归一化后的数值，需要 inverse_transform 回到原尺度
        x0_np = x0.detach().cpu().numpy()
        recon0_np = recon0.detach().cpu().numpy()
        x0_inv = scaler.inverse_transform(x0_np)[0]
        recon0_inv = scaler.inverse_transform(recon0_np)[0]

        # 绘制原始尺度下的对比
        plt.figure()
        plt.plot(x0_inv, label="original")
        plt.plot(recon0_inv, label="reconstructed")
        plt.title("Input vs Reconstructed (original scale)")
        plt.legend()
        recon_fig = os.path.join(args.outdir, "reconstruction.png")
        plt.savefig(recon_fig, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Save] 重建对比图已保存到: {recon_fig}")


if __name__ == "__main__":
    main()
