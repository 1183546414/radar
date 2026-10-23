# visualize_from_checkpoint.py
# 从检查点加载模型并进行可视化操作

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================
# 1) 检查点加载函数
# =========================
def load_checkpoint(model, checkpoint_path, device):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已从检查点加载: {checkpoint_path}")
    
    # 如果有训练信息，也打印出来
    if 'epoch' in checkpoint:
        print(f"训练轮次: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"最佳损失: {checkpoint['loss']:.6f}")
    
    return model

# =========================
# 2) 数据加载和预处理（与训练代码相同）
# =========================
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# 加载数据
jammer_cf = load_json("dataset/data_2/json/Jammer_carrierFrequency.json")
jamming_params = load_json("dataset/data_2/json/Jamming_Params.json")
jamming_type = load_json("dataset/data_2/json/Encoded_Jamming_Type.json")

N = min(len(jammer_cf), len(jamming_params), len(jamming_type))

# 数据预处理
cf_array = np.asarray(jammer_cf[:N], dtype=np.float32)
pr_array = np.asarray(jamming_params[:N], dtype=np.float32)

scaler_cf = MinMaxScaler()
scaler_pr = MinMaxScaler()
cf_scaled = scaler_cf.fit_transform(cf_array).astype(np.float32)
pr_scaled = scaler_pr.fit_transform(pr_array).astype(np.float32)

numeric_scaled = np.concatenate([cf_scaled, pr_scaled], axis=1).astype(np.float32)
D_num = numeric_scaled.shape[1]

def is_scalar_label_list(lst):
    for x in lst:
        if isinstance(x, (list, tuple, np.ndarray)):
            return False
    return True

if is_scalar_label_list(jamming_type[:N]):
    labels = np.asarray(jamming_type[:N], dtype=np.int64).reshape(-1, 1)
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    cond_onehot = ohe.fit_transform(labels).astype(np.float32)
else:
    cond_onehot = np.asarray(jamming_type[:N], dtype=np.float32)
C = cond_onehot.shape[1]

x_enc = np.concatenate([cond_onehot, numeric_scaled], axis=1).astype(np.float32)
cond = cond_onehot.astype(np.float32)

# 转换为张量
x_enc_t = torch.tensor(x_enc, dtype=torch.float32)
y_num_t = torch.tensor(numeric_scaled, dtype=torch.float32)
cond_t = torch.tensor(cond, dtype=torch.float32)

# =========================
# 3) 模型定义（与训练代码相同）
# =========================
class ResBlock(nn.Module):
    def __init__(self, dim, pdrop=0.03):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.do = nn.Dropout(pdrop)
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        h = self.lin1(x)
        h = self.act(h)
        h = self.do(h)
        h = self.lin2(h)
        out = self.ln(x + h)
        out = self.act(out)
        return out

class StrongCurveAE(nn.Module):
    def __init__(self, input_dim, cond_dim, num_numeric,
                 widths=(768, 768, 768), latent=384, depth=3, pdrop=0.03):
        super().__init__()
        self.num_numeric = num_numeric
        D_in = input_dim

        w1, w2, w3 = widths
        self.enc_in = nn.Sequential(
            nn.Linear(D_in, w1), nn.GELU(),
            ResBlock(w1, pdrop),
        )
        self.enc_mid = nn.Sequential(
            nn.Linear(w1, w2), nn.GELU(),
            *[ResBlock(w2, pdrop) for _ in range(depth)]
        )
        self.enc_out = nn.Sequential(
            nn.Linear(w2, w3), nn.GELU(),
            ResBlock(w3, pdrop),
            nn.Linear(w3, latent), nn.GELU()
        )

        d_in = latent + cond_dim
        self.dec_in = nn.Sequential(
            nn.Linear(d_in, w3), nn.GELU(),
            ResBlock(w3, pdrop)
        )
        self.dec_mid = nn.Sequential(
            nn.Linear(w3, w2), nn.GELU(),
            *[ResBlock(w2, pdrop) for _ in range(depth)]
        )
        self.dec_out = nn.Sequential(
            nn.Linear(w2, w1), nn.GELU(),
            ResBlock(w1, pdrop),
            nn.Linear(w1, num_numeric),
            nn.Sigmoid()
        )

    def forward(self, x_enc, cond):
        h1 = self.enc_in(x_enc)
        h2 = self.enc_mid(h1)
        z = self.enc_out(h2)

        zc = torch.cat([z, cond], dim=1)
        d1 = self.dec_in(zc)
        d2 = self.dec_mid(d1)
        y = self.dec_out(d2)
        return y, z

# =========================
# 4) 可视化函数
# =========================

def _save_bar_top_variance(Z, k=20, out_png="latent_topvar.png"):
    """保存隐变量方差最大的前k个维度的条形图"""
    vars_ = Z.var(axis=0)
    idx = np.argsort(vars_)[::-1][:k]
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(np.arange(k), vars_[idx])
    plt.xticks(np.arange(k), [f"z{j}" for j in idx], rotation=60)
    plt.ylabel("Variance")
    plt.title(f"Top-{k} Latent Variances")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[Saved] {out_png}")
    return idx, vars_[idx]

def visualize_reconstruction(model, device, x_enc_t, cond_t, y_num_t, num_samples=6, checkpoint_name=""):
    """可视化重建结果"""
    model.eval()
    
    # 随机选择样本
    sample_indices = np.random.choice(len(x_enc_t), size=num_samples, replace=False)
    
    # 创建画布
    plt.figure(figsize=(12, 8))
    
    # 遍历每个样本
    for i, sample_idx in enumerate(sample_indices):
        with torch.no_grad():
            # 获取样本数据
            x1 = x_enc_t[sample_idx].unsqueeze(0).to(device)
            c1 = cond_t[sample_idx].unsqueeze(0).to(device)
            y_true = y_num_t[sample_idx].cpu().numpy()
            y_pred = model(x1, c1)[0].squeeze(0).cpu().numpy()

        # 绘制每个样本的图
        plt.subplot(2, 3, i + 1)
        plt.plot(y_true, label="Original")
        plt.plot(y_pred, label="Reconstructed")
        plt.title(f"Sample #{sample_idx}")
        plt.xlabel("Feature Index")
        plt.ylabel("Scaled Value [0,1]")
        plt.legend()
    
    plt.suptitle(f"Reconstruction Results ({checkpoint_name})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"reconstruction_{checkpoint_name.replace('.pth', '')}.png", dpi=160)
    plt.show()

def extract_latents(model, x_enc_t, cond_t, batch_size=2048, device=None):
    """提取隐变量"""
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    N = x_enc_t.size(0)
    latents, labels = [], []
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = x_enc_t[i:i+batch_size].to(device)
            cb = cond_t[i:i+batch_size].to(device)
            _, z = model(xb, cb)
            latents.append(z.cpu().numpy())
            labels.append(cb.argmax(dim=1).cpu().numpy())
            
    Z = np.concatenate(latents, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    return Z, y

def _save_scatter_2d(X2d, y, title, out_png):
    plt.figure(figsize=(8, 6))
    
    # 增加点的大小和透明度，使重叠点更明显
    sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=y, s=15, alpha=0.6, cmap="tab20")
    
    # 计算数据的百分位数范围，避免异常值影响可视化
    x01, x99 = np.percentile(X2d[:, 0], [1, 99])
    y01, y99 = np.percentile(X2d[:, 1], [1, 99])
    
    # 设置坐标轴范围，排除极端异常值
    plt.xlim(x01 - 0.1*(x99-x01), x99 + 0.1*(x99-x01))
    plt.ylim(y01 - 0.1*(y99-y01), y99 + 0.1*(y99-y01))
    
    plt.title(title)
    plt.xlabel("Dim-1")
    plt.ylabel("Dim-2")
    
    # 类别不多时展示图例
    if len(np.unique(y)) <= 20:
        plt.legend(*sc.legend_elements(num=len(np.unique(y))), loc="best", title="Class")
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[Saved] {out_png}")

def visualize_latent_space(model, device, x_enc_t, cond_t, checkpoint_name=""):
    """可视化隐空间"""
    print("提取隐变量...")
    Z, cls = extract_latents(model, x_enc_t, cond_t, batch_size=2048, device=device)
    
    # 添加隐变量方差条形图可视化
    print("生成隐变量方差条形图...")
    top_dim_indices, top_variances = _save_bar_top_variance(
        Z, k=20, out_png=f"latent_topvar_{checkpoint_name.replace('.pth', '')}.png"
    )
    
    print(f"[PCA] 使用 {Z.shape[0]} 个样本进行降维")
    # PCA可视化
    print("进行PCA降维...")
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(Z)
    _save_scatter_2d(Z_pca, cls, "Latent PCA (colored by class)", "latent_pca.png")

    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=cls, s=10, cmap="tab20", alpha=0.7)
    plt.colorbar(scatter, label="Class")
    plt.title(f"Latent Space PCA ({checkpoint_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"latent_pca_{checkpoint_name.replace('.pth', '')}.png", dpi=160)
    plt.show()
    

    # t-SNE可视化（抽样以减少计算量）
    print("进行t-SNE降维...")
    sample_size = min(2000, len(Z))
    indices = np.random.choice(len(Z), size=sample_size, replace=False)
    Z_sample = Z[indices]
    cls_sample = cls[indices]
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    Z_tsne = tsne.fit_transform(Z_sample)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=cls_sample, s=10, cmap="tab20", alpha=0.7)
    plt.colorbar(scatter, label="Class")
    plt.title(f"Latent Space t-SNE ({checkpoint_name})")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.tight_layout()
    plt.savefig(f"latent_tsne_{checkpoint_name.replace('.pth', '')}.png", dpi=160)
    plt.show()

def latent_traversal(model, device, x_enc_t, cond_t, y_num_t, sample_idx=0, dims=None, 
                     sigmas=(2., 1., 0., -1., -2.), checkpoint_name=""):
    """隐变量遍历可视化"""
    model.eval()
    
    # 先统计各维标准差
    Z_all, _ = extract_latents(model, x_enc_t, cond_t, batch_size=4096, device=device)
    z_std = Z_all.std(axis=0) + 1e-8

    # 基准样本
    x = x_enc_t[sample_idx:sample_idx+1].to(device)
    c = cond_t[sample_idx:sample_idx+1].to(device)
    
    with torch.no_grad():
        y_base, z0 = model(x, c)
    
    z0 = z0.clone()
    
    # 默认取方差最大的前4维
    if dims is None:
        dims = np.argsort(z_std)[::-1][:4].tolist()

    cols, rows = len(dims), len(sigmas)
    plt.figure(figsize=(3.5*cols, 2.8*rows))
    xi = np.arange(y_base.shape[1])
    y_base_np = y_base.squeeze(0).detach().cpu().numpy()

    for col, d in enumerate(dims):
        for row, k in enumerate(sigmas):
            z = z0.clone()
            z[:, d] = z0[:, d] + k * z_std[d]
            
            # 解码
            zc = torch.cat([z, c], dim=1)
            d1 = model.dec_in(zc)
            d2 = model.dec_mid(d1)
            y_hat = model.dec_out(d2).squeeze(0).detach().cpu().numpy()

            ax = plt.subplot(rows, cols, row*cols + col + 1)
            ax.plot(xi, y_base_np, linewidth=1.0, label="base")
            ax.plot(xi, y_hat, linewidth=1.0, label=f"z{d} {k:+.0f}σ")
            if row == 0: ax.set_title(f"Dim z{d} traversal")
            if col == 0: ax.set_ylabel("scaled value")
            if row == 0 and col == 0: ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Latent Traversal (Sample #{sample_idx}, {checkpoint_name})", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"latent_traversal_{checkpoint_name.replace('.pth', '')}.png", dpi=180)
    plt.show()

# =========================
# 5) 主函数
# =========================
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型实例
    model = StrongCurveAE(
        input_dim=x_enc_t.shape[1],
        cond_dim=C,
        num_numeric=D_num,
        widths=(768, 768, 768),
        latent=384, depth=3, pdrop=0.03
    ).to(device)
    
    # 检查点路径
    checkpoint_dir = "checkpoints"
    checkpoint_files = [
        "best_model.pth",
        "final_model.pth",
        "latest_checkpoint.pth"
    ]
    
    # 查找可用的检查点
    available_checkpoints = []
    for file in checkpoint_files:
        path = os.path.join(checkpoint_dir, file)
        if os.path.exists(path):
            available_checkpoints.append((file, path))
    
    if not available_checkpoints:
        print("没有找到可用的检查点文件")
        return
    
    print("可用的检查点:")
    for i, (name, path) in enumerate(available_checkpoints):
        print(f"{i+1}. {name}")
    
    # 选择检查点
    choice = input("请选择要加载的检查点编号 (默认为1): ").strip()
    if not choice:
        choice = 1
    else:
        choice = int(choice)
    
    if choice < 1 or choice > len(available_checkpoints):
        print("无效的选择")
        return
    
    checkpoint_name, checkpoint_path = available_checkpoints[choice-1]
    
    # 加载检查点
    model = load_checkpoint(model, checkpoint_path, device)
    
    # 选择可视化选项
    print("\n可视化选项:")
    print("1. 重建结果可视化")
    print("2. 隐空间可视化")
    print("3. 隐变量遍历")
    print("4. 全部可视化")
    
    option = input("请选择可视化选项 (默认为4): ").strip()
    if not option:
        option = 4
    else:
        option = int(option)
    
    # 执行可视化
    if option == 1 or option == 4:
        print("\n进行重建结果可视化...")
        visualize_reconstruction(model, device, x_enc_t, cond_t, y_num_t, 
                                num_samples=6, checkpoint_name=checkpoint_name)
    
    if option == 2 or option == 4:
        print("\n进行隐空间可视化...")
        visualize_latent_space(model, device, x_enc_t, cond_t, checkpoint_name)
    
    if option == 3 or option == 4:
        print("\n进行隐变量遍历...")
        sample_idx = int(input("请输入样本索引 (默认为0): ") or 0)
        latent_traversal(model, device, x_enc_t, cond_t, y_num_t, 
                        sample_idx=sample_idx, checkpoint_name=checkpoint_name)
    
    print("\n可视化完成!")

if __name__ == "__main__":
    main()