# ae_curve_fit_strong.py
# 目标：将重构曲线尽可能贴合原曲线（高容量 AE + 多项形状一致性损失）
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

jammer_cf = load_json("dataset/data_1/json/Jammer_carrierFrequency.json")    # [freq, gain]
jamming_params = load_json("dataset/data_1/json/Jamming_Params.json")        # 其余数值参数
jamming_type = load_json("dataset/data_1/json/Encoded_Jamming_Type.json")    # 类别（标量或one-hot）

N = min(len(jammer_cf), len(jamming_params), len(jamming_type))

# =========================
# 2) 组装数值特征 + 条件类别
# =========================
# 数值：分组归一化（carrierFrequency 两列 + 其他参数）
cf_array = np.asarray(jammer_cf[:N], dtype=np.float32)          # [N, 2]
pr_array = np.asarray(jamming_params[:N], dtype=np.float32)     # [N, P]

scaler_cf = MinMaxScaler()
scaler_pr = MinMaxScaler()
cf_scaled = scaler_cf.fit_transform(cf_array).astype(np.float32)    # [N, 2] -> [0,1]
pr_scaled = scaler_pr.fit_transform(pr_array).astype(np.float32)    # [N, P] -> [0,1]

numeric_scaled = np.concatenate([cf_scaled, pr_scaled], axis=1)     # [N, D_num]
D_num = numeric_scaled.shape[1]

# 类别：兼容标量/one-hot 两种情况
def is_scalar_label_list(lst):
    for x in lst:
        if isinstance(x, (list, tuple, np.ndarray)):
            return False
    return True

if is_scalar_label_list(jamming_type[:N]):
    labels = np.asarray(jamming_type[:N], dtype=np.int64).reshape(-1, 1)
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    cond_onehot = ohe.fit_transform(labels).astype(np.float32)       # [N, C]
else:
    cond_onehot = np.asarray(jamming_type[:N], dtype=np.float32)     # [N, C]
C = cond_onehot.shape[1]

# 编码器输入： [cond_onehot | numeric_scaled]
x_enc = np.concatenate([cond_onehot, numeric_scaled], axis=1).astype(np.float32)  # [N, C + D_num]
cond = cond_onehot.astype(np.float32)                                             # 解码用条件输入

# =========================
# 3) PyTorch 张量
# =========================
x_enc_t = torch.tensor(x_enc, dtype=torch.float32)
y_num_t = torch.tensor(numeric_scaled, dtype=torch.float32)  # 仅对数值特征做重构
cond_t  = torch.tensor(cond, dtype=torch.float32)

# =========================
# 4) 特征权重（方差倒数）
# =========================
col_var = np.var(numeric_scaled, axis=0) + 1e-8
w_feat = (1.0 / col_var)
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
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =========================
# 6) 模型：高容量 AE（残差 + 跳连），条件解码
# =========================
class ResBlock(nn.Module):
    def __init__(self, dim, pdrop=0.05):
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
                 widths=(512, 512, 512), latent=256, depth=2, pdrop=0.05):
        super().__init__()
        self.num_numeric = num_numeric
        D_in = input_dim

        # Encoder: D_in -> w1 -> w2 -> w3 -> latent
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

        # Decoder: concat(latent, cond) -> mirror
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
            nn.Sigmoid()  # 与 [0,1] 归一化匹配
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
# 7) 损失函数：L1 + L2 + 一阶导数 + 余弦相似（带特征权重）
# =========================
def diff1(x, dim=1):
    # 一阶差分：x[...,1:] - x[...,:-1]
    return x[:, 1:] - x[:, :-1]

def weighted_loss(pred, target, w_feat,
                  w_l1=1.0, w_l2=0.5, w_diff=0.8, w_cos=0.2):
    # pred/target: [B, D]
    w = w_feat.view(1, -1)

    l1 = (pred - target).abs() * w
    l1 = l1.mean()

    l2 = ((pred - target) ** 2) * w
    l2 = l2.mean()

    # 形状一致性：一阶导数（差分）对齐
    dp = diff1(pred); dt = diff1(target)
    # 缩放权重到长度 D-1
    w_d = w[:, 1:]
    ldiff = (dp - dt).abs() * w_d
    ldiff = ldiff.mean()

    # 余弦相似（1 - cos），对整体轮廓进行约束
    # 加 1e-8 防止除零
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    targ_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (pred_norm * targ_norm).sum(dim=1).mean()
    lcos = 1.0 - cos_sim

    loss = w_l1 * l1 + w_l2 * l2 + w_diff * ldiff + w_cos * lcos
    return loss, {'l1': l1.item(), 'l2': l2.item(), 'ldiff': ldiff.item(), 'lcos': lcos.item()}

# =========================
# 8) 训练配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StrongCurveAE(
    input_dim = x_enc_t.shape[1],
    cond_dim  = C,
    num_numeric = D_num,
    widths=(512, 512, 512),
    latent=256, depth=2, pdrop=0.05
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

# 余弦退火重启
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-5
)

# 早停
class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5):
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

early = EarlyStopping(patience=40, min_delta=1e-6)

# =========================
# 9) 训练循环
# =========================
BATCHES_PER_EPOCH = len(dataloader)
model.train()
for epoch in range(400):  # 如需更贴合，可适当增加上限
    running, l1m, l2m, ldm, lcm = 0.0, 0.0, 0.0, 0.0, 0.0
    for step, (x_b, y_b, c_b) in enumerate(dataloader):
        x_b = x_b.to(device); y_b = y_b.to(device); c_b = c_b.to(device)
        pred, _ = model(x_b, c_b)

        loss, parts = weighted_loss(
            pred, y_b, w_feat_t.to(device),
            w_l1=1.0, w_l2=0.5, w_diff=0.8, w_cos=0.2
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + step / max(1, BATCHES_PER_EPOCH))

        running += loss.item()
        l1m += parts['l1']; l2m += parts['l2']; ldm += parts['ldiff']; lcm += parts['lcos']

    n = len(dataloader)
    avg = running / n
    print(f"Epoch {epoch+1:03d} | loss {avg:.6f} | L1 {l1m/n:.6f} | L2 {l2m/n:.6f} | d1 {ldm/n:.6f} | cos {lcm/n:.6f}")

    early.step(avg)
    if early.stop:
        print("Early stopping triggered.")
        break

# =========================
# 10) 评估（整批）
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

# =========================
# 11) 可视化（第50个样本；改 sample_idx 可换样本）
# =========================
sample_idx = 49
with torch.no_grad():
    x1 = x_enc_t[sample_idx].unsqueeze(0).to(device)
    c1 = cond_t[sample_idx].unsqueeze(0).to(device)
    y_true = y_num_t[sample_idx].cpu().numpy()
    y_pred = model(x1, c1)[0].squeeze(0).cpu().numpy()

plt.figure()
plt.plot(y_true, label="Original (numeric)")
plt.plot(y_pred, label="Reconstructed (numeric)")
plt.title("Numeric Features: Original vs Reconstructed (Shape-Constrained AE)")
plt.xlabel("Feature Index")
plt.ylabel("Scaled Value [0,1]")
plt.legend()
plt.tight_layout()
plt.show()
