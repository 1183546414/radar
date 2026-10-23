# ae_curve_fit_extreme.py
# 极致贴合版：大容量条件AE + 强形状损失 + 长微调
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# =========================
# 1) 数据加载
# =========================
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

jammer_cf = load_json("dataset/data_2/json/Jammer_carrierFrequency.json")    # [freq, gain]
jamming_params = load_json("dataset/data_2/json/Jamming_Params.json")        # 其余数值参数
jamming_type = load_json("dataset/data_2/json/Encoded_Jamming_Type.json")    # 类别（标量或one-hot）

N = min(len(jammer_cf), len(jamming_params), len(jamming_type))

# =========================
# 2) 组装数值特征 + 条件类别（分组归一化）
# =========================
cf_array = np.asarray(jammer_cf[:N], dtype=np.float32)       # [N, 2]
pr_array = np.asarray(jamming_params[:N], dtype=np.float32)  # [N, P]

scaler_cf = MinMaxScaler()
scaler_pr = MinMaxScaler()
cf_scaled = scaler_cf.fit_transform(cf_array).astype(np.float32)  # [0,1]
pr_scaled = scaler_pr.fit_transform(pr_array).astype(np.float32)  # [0,1]

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
    cond_onehot = ohe.fit_transform(labels).astype(np.float32)   # [N, C]
else:
    cond_onehot = np.asarray(jamming_type[:N], dtype=np.float32) # [N, C]
C = cond_onehot.shape[1]

# 编码器输入： [cond_onehot | numeric_scaled]
x_enc = np.concatenate([cond_onehot, numeric_scaled], axis=1).astype(np.float32)  # [N, C + D_num]
cond  = cond_onehot.astype(np.float32)

# =========================
# 3) 张量化
# =========================
x_enc_t = torch.tensor(x_enc, dtype=torch.float32)
y_num_t = torch.tensor(numeric_scaled, dtype=torch.float32)
cond_t  = torch.tensor(cond, dtype=torch.float32)

# =========================
# 4) 特征权重（方差倒数，归一）
# =========================
col_var = np.var(numeric_scaled, axis=0) + 1e-8
w_feat = 1.0 / col_var
w_feat = w_feat / w_feat.mean()
w_feat_t = torch.tensor(w_feat, dtype=torch.float32)  # [D_num]

# =========================
# 5) 数据集 / 加载器
# =========================
class JammingDataset(Dataset):
    def __init__(self, x_enc, y_num, cond):
        self.x_enc = x_enc
        self.y_num = y_num
        self.cond  = cond
    def __len__(self):
        return self.x_enc.size(0)
    def __getitem__(self, idx):
        return self.x_enc[idx], self.y_num[idx], self.cond[idx]

dataset = JammingDataset(x_enc_t, y_num_t, cond_t)
# 为了拟合更“细”，可以增大 batch（根据显存调整）
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# =========================
# 6) 模型：更宽更深的条件AE（残差 + GELU + LayerNorm）
# =========================
class ResBlock(nn.Module):
    def __init__(self, dim, pdrop=0.03):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act  = nn.GELU()
        self.do   = nn.Dropout(pdrop)
        self.ln   = nn.LayerNorm(dim)
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
            nn.Sigmoid()  # 对齐[0,1]归一化
        )

    def forward(self, x_enc, cond):
        h1 = self.enc_in(x_enc)
        h2 = self.enc_mid(h1)
        z  = self.enc_out(h2)

        zc = torch.cat([z, cond], dim=1)
        d1 = self.dec_in(zc)
        d2 = self.dec_mid(d1)
        y  = self.dec_out(d2)
        return y, z

# =========================
# 7) 强形状一致性损失：L1 + L2 + 一阶导数 + 余弦相似
# =========================
def diff1(x):
    return x[:, 1:] - x[:, :-1]

def weighted_loss(pred, target, w_feat,
                  w_l1=1.0, w_l2=0.25, w_diff=1.4, w_cos=0.4):
    w = w_feat.view(1, -1)

    l1 = (pred - target).abs() * w
    l1 = l1.mean()

    l2 = ((pred - target) ** 2) * w
    l2 = l2.mean()

    dp, dt = diff1(pred), diff1(target)
    w_d = w[:, 1:]
    ldiff = (dp - dt).abs() * w_d
    ldiff = ldiff.mean()

    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    targ_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (pred_norm * targ_norm).sum(dim=1).mean()
    lcos = 1.0 - cos_sim

    loss = w_l1 * l1 + w_l2 * l2 + w_diff * ldiff + w_cos * lcos
    return loss, {'l1': l1.item(), 'l2': l2.item(), 'ldiff': ldiff.item(), 'lcos': lcos.item()}

# =========================
# 8) 训练与微调配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StrongCurveAE(
    input_dim = x_enc_t.shape[1],
    cond_dim  = C,
    num_numeric = D_num,
    widths=(768, 768, 768),
    latent=384, depth=3, pdrop=0.03
).to(device)

# 阶段1：主训练（使用更稳定的学习率策略）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # 降低初始学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6
)

# 早停（主训练更“耐心”）
class EarlyStopping:
    def __init__(self, patience=80, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.stop = False
    def step(self, metric):
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

early = EarlyStopping(patience=80, min_delta=1e-6)

# =========================
# 9) 阶段1：主训练
# =========================
BATCHES_PER_EPOCH = len(dataloader)
model.train()
for epoch in range(600):  # 更长的主训练
    running, l1m, l2m, ldm, lcm = 0.0, 0.0, 0.0, 0.0, 0.0
    for step, (x_b, y_b, c_b) in enumerate(dataloader):
        x_b = x_b.to(device); y_b = y_b.to(device); c_b = c_b.to(device)
        pred, _ = model(x_b, c_b)

        loss, parts = weighted_loss(
            pred, y_b, w_feat_t.to(device),
            w_l1=1.0, w_l2=0.25, w_diff=1.4, w_cos=0.4
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        #scheduler.step(epoch + step / max(1, BATCHES_PER_EPOCH))

        running += loss.item()
        l1m += parts['l1']; l2m += parts['l2']; ldm += parts['ldiff']; lcm += parts['lcos']

    n = len(dataloader)
    avg = running / n
    print(f"[Stage1] Epoch {epoch+1:03d} | loss {avg:.6f} | L1 {l1m/n:.6f} | L2 {l2m/n:.6f} | d1 {ldm/n:.6f} | cos {lcm/n:.6f}")

    # 在每个epoch结束后更新学习率
    scheduler.step(avg)  # 添加这行

    early.step(avg)
    if early.stop:
        print("[Stage1] Early stopping.")
        break

# =========================
# 10) 阶段2：低学习率微调（极致贴合）
# =========================
# 微调时降低学习率、去掉重启，让模型在当前 basin 内细致贴合
for g in optimizer.param_groups:
    g['lr'] = 1e-5
scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

fine_tune_epochs = 200  # 更长一点微调
model.train()
for epoch in range(fine_tune_epochs):
    running = 0.0
    for x_b, y_b, c_b in dataloader:
        x_b = x_b.to(device); y_b = y_b.to(device); c_b = c_b.to(device)
        pred, _ = model(x_b, c_b)

        # 微调阶段：进一步提高形状损失比重
        loss, _ = weighted_loss(
            pred, y_b, w_feat_t.to(device),
            w_l1=1.0, w_l2=0.2, w_diff=1.6, w_cos=0.5
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running += loss.item()

    scheduler_ft.step()
    print(f"[Stage2-FT] Epoch {epoch+1:03d} | loss {running/len(dataloader):.6f}")

# =========================
# 11) 评估
# =========================
model.eval()
with torch.no_grad():
    for x_b, y_b, c_b in dataloader:
        x_b = x_b.to(device); y_b = y_b.to(device); c_b = c_b.to(device)
        pred, _ = model(x_b, c_b)
        mae = (pred - y_b).abs().mean().item()
        mse = nn.functional.mse_loss(pred, y_b).item()
        print(f"[Eval] batch MAE={mae:.6f} | MSE={mse:.6f}")
        break



# # =========================
# # 12) 可视化（第50个样本；改 sample_idx 可换）
# # =========================
# sample_idx = 70
# with torch.no_grad():
#     x1 = x_enc_t[sample_idx].unsqueeze(0).to(device)
#     c1 = cond_t[sample_idx].unsqueeze(0).to(device)
#     y_true = y_num_t[sample_idx].cpu().numpy()
#     y_pred = model(x1, c1)[0].squeeze(0).cpu().numpy()

# plt.figure()
# plt.plot(y_true, label="Original (numeric)")
# plt.plot(y_pred, label="Reconstructed (numeric)")
# plt.title("Numeric Features: Original vs Reconstructed — Extreme Fit AE")
# plt.xlabel("Feature Index")
# plt.ylabel("Scaled Value [0,1]")
# plt.legend()
# plt.tight_layout()
# plt.show()





import numpy as np

# 随机抽取20个样本的索引
sample_indices = np.random.choice(len(x_enc_t), size=20, replace=False)

# 创建一个画布来显示所有图
plt.figure(figsize=(10, 12))

# 遍历每个样本
for i, sample_idx in enumerate(sample_indices):
    with torch.no_grad():
        # 获取样本数据
        x1 = x_enc_t[sample_idx].unsqueeze(0).to(device)
        c1 = cond_t[sample_idx].unsqueeze(0).to(device)
        y_true = y_num_t[sample_idx].cpu().numpy()
        y_pred = model(x1, c1)[0].squeeze(0).cpu().numpy()

    # 绘制每个样本的图
    plt.subplot(5, 4, i + 1)  # 5行4列的子图
    plt.plot(y_true, label="Original (numeric)")
    plt.plot(y_pred, label="Reconstructed (numeric)")
    plt.title(f"Sample #{sample_idx}")
    plt.xlabel("Feature Index")
    plt.ylabel("Scaled Value [0,1]")
    plt.legend()

# 调整布局并保存图像
plt.tight_layout()
plt.savefig("reconstructed_samples.png", dpi=160)
plt.show()





import numpy as np
import matplotlib.pyplot as plt
import torch

# 随机抽取6个样本的索引
sample_indices = np.random.choice(len(x_enc_t), size=6, replace=False)

# 创建一个画布来显示所有图
plt.figure(figsize=(12, 8))  # 调整画布大小适配2×3布局

# 遍历每个样本
for i, sample_idx in enumerate(sample_indices):
    with torch.no_grad():
        # 获取样本数据
        x1 = x_enc_t[sample_idx].unsqueeze(0).to(device)
        c1 = cond_t[sample_idx].unsqueeze(0).to(device)
        y_true = y_num_t[sample_idx].cpu().numpy()
        y_pred = model(x1, c1)[0].squeeze(0).cpu().numpy()

    # 绘制每个样本的图
    plt.subplot(2, 3, i + 1)  # 2行3列的子图
    plt.plot(y_true, label="Original (numeric)")
    plt.plot(y_pred, label="Reconstructed (numeric)")
    plt.title(f"Sample #{sample_idx}")
    plt.xlabel("Feature Index")
    plt.ylabel("Scaled Value [0,1]")
    plt.legend()

# 调整布局并保存图像
plt.tight_layout()
plt.savefig("reconstructed_samples_6.png", dpi=160)
plt.show()






# =========================
# 13) 隐变量可视化（无需checkpoint，直接使用当前内存中的模型与张量）
# =========================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# 可能尚未导入，单独引入降维工具
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception as _e:
    print("[Warn] 需要 scikit-learn 以进行 PCA / t-SNE：pip install scikit-learn")
    raise

# ---------- 基础健壮性检查 ----------
_need = []
for name in ["model", "x_enc_t", "cond_t", "y_num_t", "device"]:
    if name not in globals():
        _need.append(name)
if _need:
    raise RuntimeError(f"下列变量在当前脚本中未定义：{_need}。请确保把本段代码放在训练/构图之后。")

model.eval()

@torch.no_grad()
def extract_latents(model, x_enc_t, cond_t, batch_size=2048, device=None):
    """批量跑编码器，收集全部 z 以及类别标签（从 one-hot 取 argmax）。"""
    if device is None:
        device = next(model.parameters()).device
    N = x_enc_t.size(0)
    latents, labels = [], []
    for i in range(0, N, batch_size):
        xb = x_enc_t[i:i+batch_size].to(device)
        cb = cond_t[i:i+batch_size].to(device)
        _, z = model(xb, cb)          # forward -> (y, z)
        latents.append(z.cpu().numpy())
        labels.append(cb.argmax(dim=1).cpu().numpy())
    Z = np.concatenate(latents, axis=0).astype(np.float32)  # [N, D_lat]
    y = np.concatenate(labels,  axis=0).astype(np.int64)    # [N]
    return Z, y

def _save_scatter_2d(X2d, y, title, out_png):
    plt.figure(figsize=(6.2, 5.6))
    sc = plt.scatter(X2d[:, 0], X2d[:, 1], c=y, s=6, cmap="tab20")
    plt.title(title); plt.xlabel("Dim-1"); plt.ylabel("Dim-2")
    # 类别不多时展示图例
    if len(np.unique(y)) <= 20:
        plt.legend(*sc.legend_elements(num=len(np.unique(y))), loc="best", title="Class")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print(f"[Saved] {out_png}")




def _save_bar_top_variance(Z, k=20, out_png="latent_topvar.png"):
    vars_ = Z.var(axis=0)
    idx = np.argsort(vars_)[::-1][:k]
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(np.arange(k), vars_[idx])
    plt.xticks(np.arange(k), [f"z{j}" for j in idx], rotation=60)
    plt.ylabel("Variance"); plt.title(f"Top-{k} Latent Variances")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print(f"[Saved] {out_png}")
    return idx, vars_[idx]





def _save_corr_latent_to_features(Z, Y, out_png="latent_feat_corr.png", max_latents=64):
    """画 |Corr(z_i, feature_j)| 热图，为避免过大，默认只取前 max_latents 个潜变量维度。"""
    L = min(Z.shape[1], max_latents)
    Zc = (Z[:, :L] - Z[:, :L].mean(axis=0, keepdims=True)) / (Z[:, :L].std(axis=0, keepdims=True) + 1e-8)
    Yc = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-8)
    corr = (Zc.T @ Yc) / (Z.shape[0] - 1)  # [L, D_num]
    plt.figure(figsize=(min(12, 0.35 * Y.shape[1] + 2), min(10, 0.25 * L + 2)))
    plt.imshow(np.abs(corr), aspect="auto", interpolation="nearest")
    plt.colorbar(label="|Correlation|")
    plt.xlabel("Original numeric features"); plt.ylabel("Latent dims")
    plt.title(f"|Corr(z, features)| (first {L} latents)")
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    print(f"[Saved] {out_png}")
    return corr

@torch.no_grad()
def _decode_from_z(model, z, cond):
    """按照模型解码路径：cat([z, cond]) -> dec_in -> dec_mid -> dec_out。"""
    zc = torch.cat([z, cond], dim=1)
    d1 = model.dec_in(zc)
    d2 = model.dec_mid(d1)
    y  = model.dec_out(d2)
    return y

def latent_traversal(model, x_enc_t, cond_t,
                     sample_idx=0, dims=None, sigmas=(2., 1., 0., -1., -2.),
                     out_png="latent_traversal.png", device=None):
    """固定样本、改变单一潜变量维度（±kσ）观测重构变化。"""
    if device is None:
        device = next(model.parameters()).device

    # 先统计各维标准差（用于 kσ）
    Z_all, _ = extract_latents(model, x_enc_t, cond_t, batch_size=4096, device=device)
    z_std = Z_all.std(axis=0) + 1e-8

    # 基准样本
    x = x_enc_t[sample_idx:sample_idx+1].to(device)
    c = cond_t[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        y_base, z0 = model(x, c)  # (1, D_num), (1, D_lat)
    z0 = z0.clone()

    # 默认取方差最大的前4维
    if dims is None:
        dims = np.argsort(z_std)[::-1][:4].tolist()

    cols, rows = len(dims), len(sigmas)
    plt.figure(figsize=(3.5*cols, 2.8*rows), constrained_layout=True)
    xi = np.arange(y_base.shape[1])
    y_base_np = y_base.squeeze(0).detach().cpu().numpy()

    for col, d in enumerate(dims):
        for row, k in enumerate(sigmas):
            z = z0.clone()
            z[:, d] = z0[:, d] + k * z_std[d]
            y_hat = _decode_from_z(model, z.to(device), c).squeeze(0).detach().cpu().numpy()

            ax = plt.subplot(rows, cols, row*cols + col + 1)
            ax.plot(xi, y_base_np, linewidth=1.0, label="base")
            ax.plot(xi, y_hat,     linewidth=1.0, label=f"z{d} {k:+.0f}σ")
            if row == 0: ax.set_title(f"Dim z{d} traversal")
            if col == 0: ax.set_ylabel("scaled value")
            if row == 0 and col == 0: ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Latent traversal (sample #{sample_idx})", y=1.02, fontsize=12)
    plt.savefig(out_png, dpi=180); plt.close()
    print(f"[Saved] {out_png}")

# ---------- 主流程 ----------
print("\n[LatViz] 开始提取隐变量 z ...")
Z, cls = extract_latents(model, x_enc_t, cond_t, batch_size=2048, device=device)
print("cls:", cls)
N, D_lat = Z.shape

print(f"[LatViz] 收到潜变量：N={N}, D_lat={D_lat}")

# 2D 降维：PCA
pca = PCA(n_components=2, random_state=0)
Z_pca = pca.fit_transform(Z)
_save_scatter_2d(Z_pca, cls, "Latent PCA (colored by class)", "latent_pca.png")

# 2D 降维：t-SNE（如 N 很大，自动抽样至 ≤8000）
tsne_sample = min(N, 8000)
sel = np.random.RandomState(0).choice(N, size=tsne_sample, replace=False) if N > tsne_sample else np.arange(N)
perp = max(5, min(50, tsne_sample // 50))  # 简单自适应
Z_tsne = TSNE(n_components=2, perplexity=perp, learning_rate="auto", init="pca", random_state=0).fit_transform(Z[sel])
_save_scatter_2d(Z_tsne, cls[sel], f"Latent t-SNE (N={len(sel)}, perp={perp})", "latent_tsne.png")

# 方差 Top-K
top_idx, _ = _save_bar_top_variance(Z, k=20, out_png="latent_topvar.png")

# 与原始数值特征相关性（热图）
Y = y_num_t.cpu().numpy()
corr = _save_corr_latent_to_features(Z, Y, out_png="latent_feat_corr.png", max_latents=64)

# 打印前若干潜变量与原始特征的 |corr| Top-3
abs_corr = np.abs(corr)
for i in range(min(abs_corr.shape[0], 10)):
    top3 = abs_corr[i].argsort()[-3:][::-1]
    print(f"[LatViz] z{i} top-3 |corr| features -> idx={top3.tolist()}, val={abs_corr[i, top3].round(3).tolist()}")

# 潜变量遍历（默认挑选方差最大的前4维）
latent_traversal(model, x_enc_t, cond_t,
                 sample_idx=0,
                 dims=top_idx[:4],
                 sigmas=(2., 1., 0., -1., -2.),
                 out_png="latent_traversal.png",
                 device=device)

print("[LatViz] 完成：latent_pca.png, latent_tsne.png, latent_topvar.png, latent_feat_corr.png, latent_traversal.png\n")

