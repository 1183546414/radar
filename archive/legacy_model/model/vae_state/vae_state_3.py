# vae_state_improved.py
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# =============================
# 1) 数据加载
# =============================
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# 数据文件路径（与你原文件一致）
jammer_cf = load_json("dataset/data_1/json/Jammer_carrierFrequency.json")   # 每条: [频率, 增益]
jamming_params = load_json("dataset/data_1/json/Jamming_Params.json")       # 其他数值参数，长度不定
jamming_type = load_json("dataset/data_1/json/Encoded_Jamming_Type.json")   # 类别（可能是整数，也可能已编码）

# =============================
# 2) 组装样本：数值特征 + 类别标签
# =============================
num_samples = min(len(jammer_cf), len(jamming_params), len(jamming_type))

# 将数值特征拼在一起（不含类别）
numeric_list = []
for i in range(num_samples):
    # jammer_cf[i]: [freq, gain]
    # jamming_params[i]: 列表（其长度可变但在同一数据集里应一致）
    numeric_list.append(list(jammer_cf[i]) + list(jamming_params[i]))
numeric_array = np.asarray(numeric_list, dtype=np.float32)  # [N, D_num]

# 处理类别：兼容两种情况 ——（1）纯整数标签；（2）已经是 one-hot 或多维向量
jt = jamming_type
# 判断是否是“标量标签”列表（如 [0,1,2,...]）
def _is_scalar_list(lst):
    # list/np.ndarray，且每个元素是int/float，且不是list/ndarray
    for x in lst:
        if isinstance(x, (list, tuple, np.ndarray)):
            return False
    return True

if _is_scalar_list(jt):
    # 情况A：标量类别 -> One-Hot
    labels_np = np.asarray(jt, dtype=np.int64).reshape(-1, 1)
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    onehot_array = ohe.fit_transform(labels_np).astype(np.float32)  # [N, C]
    class_indices = labels_np.reshape(-1)  # [N]
else:
    # 情况B：已是向量 -> 认为是 one-hot（或分布），取 argmax 作为类别索引
    oh = np.asarray(jt, dtype=np.float32)              # [N, C]
    class_indices = np.argmax(oh, axis=1).astype(np.int64)
    onehot_array = oh

num_numeric = numeric_array.shape[1]
num_classes = onehot_array.shape[1]

# 数值特征做 MinMax 到 [0,1]，与 Sigmoid 输出匹配
num_scaler = MinMaxScaler()
numeric_scaled = num_scaler.fit_transform(numeric_array).astype(np.float32)  # [N, D_num]

# 输入给编码器的是： [ one-hot 类别 | 数值特征 ] 的拼接
x_full = np.concatenate([onehot_array, numeric_scaled], axis=1).astype(np.float32)  # [N, C + D_num]

# 目标：
# - 数值重构目标：numeric_scaled
# - 类别目标：class_indices（用于交叉熵）
x_full_t = torch.tensor(x_full, dtype=torch.float32)
y_numeric_t = torch.tensor(numeric_scaled, dtype=torch.float32)
y_class_t = torch.tensor(class_indices, dtype=torch.long)

# =============================
# 3) 数据集与加载器
# =============================
class JammingDataset(Dataset):
    def __init__(self, x_full, y_numeric, y_class):
        self.x_full = x_full
        self.y_numeric = y_numeric
        self.y_class = y_class

    def __len__(self):
        return self.x_full.size(0)

    def __getitem__(self, idx):
        return self.x_full[idx], self.y_numeric[idx], self.y_class[idx]

dataset = JammingDataset(x_full_t, y_numeric_t, y_class_t)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =============================
# 4) 模型：VAE（双头解码）
# =============================
class VAE(nn.Module):
    def __init__(self, input_dim, num_numeric, num_classes,
                 hidden_dim=128, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.num_numeric = num_numeric
        self.num_classes = num_classes

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 公共解码基干
        self.decoder_base = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 数值重构头：输出到 [0,1]
        self.decoder_numeric = nn.Sequential(
            nn.Linear(hidden_dim, num_numeric),
            nn.Sigmoid()
        )
        # 类别头：输出 logits（不加激活，交叉熵内部做 softmax）
        self.decoder_class = nn.Linear(hidden_dim, num_classes)

    def reparameterize(self, mu, logvar, sample=True):
        if not self.training or not sample:
            # 评估或显式关闭采样 -> 直接用均值
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=True):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar, sample=sample)
        d = self.decoder_base(z)
        recon_numeric = self.decoder_numeric(d)     # [B, D_num]
        class_logits = self.decoder_class(d)        # [B, C]
        return recon_numeric, class_logits, mu, logvar

# =============================
# 5) 损失函数（β-VAE + 双头）
# =============================
def vae_loss(recon_numeric, target_numeric, class_logits, target_class,
             mu, logvar, beta=0.1, mse_weight=1.0, ce_weight=1.0):
    # 数值重构：MSE
    mse = nn.MSELoss()(recon_numeric, target_numeric)
    # 类别：交叉熵
    ce = nn.CrossEntropyLoss()(class_logits, target_class)
    # KL
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse_weight * mse + ce_weight * ce + beta * kld
    return loss, mse, ce, kld

# =============================
# 6) 训练
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(
    input_dim=x_full_t.shape[1],
    num_numeric=num_numeric,
    num_classes=num_classes,
    hidden_dim=128,
    latent_dim=16
).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=10, verbose=True)

num_epochs = 300
beta = 0.1
mse_w, ce_w = 1.0, 1.0

loss_list = []
mse_list = []
ce_list = []
kld_list = []

vae.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total_kld = 0.0

    for x_batch, y_num_batch, y_cls_batch in dataloader:
        x_batch = x_batch.to(device)
        y_num_batch = y_num_batch.to(device)
        y_cls_batch = y_cls_batch.to(device)

        recon_num, cls_logits, mu, logvar = vae(x_batch, sample=True)
        loss, mse, ce, kld = vae_loss(recon_num, y_num_batch, cls_logits, y_cls_batch,
                                      mu, logvar, beta=beta, mse_weight=mse_w, ce_weight=ce_w)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_ce += ce.item()
        total_kld += kld.item()
        loss_list.append(loss.item())
        mse_list.append(mse.item())
        ce_list.append(ce.item())
        kld_list.append(kld.item())

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_ce = total_ce / len(dataloader)
    avg_kld = total_kld / len(dataloader)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | MSE {avg_mse:.4f} | CE {avg_ce:.4f} | KLD {avg_kld:.4f}")

# =============================
# 7) 评估（用 mu，不采样）
# =============================
vae.eval()
with torch.no_grad():
    # 取一批数据做评估
    for x_batch, y_num_batch, y_cls_batch in dataloader:
        x_b = x_batch.to(device)
        y_num_b = y_num_batch.to(device)
        y_cls_b = y_cls_batch.to(device)

        # 前向时禁止采样（sample=False），或手写 mu->decoder
        recon_num, cls_logits, mu, logvar = vae(x_b, sample=False)

        # 指标
        batch_mse = nn.MSELoss()(recon_num, y_num_b).item()
        pred_cls = cls_logits.argmax(dim=1)
        acc = (pred_cls == y_cls_b).float().mean().item()

        print(f"[Eval] Numeric MSE: {batch_mse:.6f} | Class Acc: {acc*100:.2f}%")
        # 仅看这一批
        break

# =============================
# 8) 可视化（数值特征：原始 vs 重构）
# =============================

# 选择第50个样本（索引49）
sample_idx = 49
input_full = x_full_t[sample_idx].unsqueeze(0).to(device)
target_numeric = y_numeric_t[sample_idx].cpu().numpy()


with torch.no_grad():
    # 用 mu 进行解码，避免采样噪声
    h = vae.encoder(input_full)
    mu = vae.fc_mu(h)
    d = vae.decoder_base(mu)
    recon_numeric = vae.decoder_numeric(d).squeeze(0).cpu().numpy()  # [D_num]

# 画数值特征曲线（都是归一化到[0,1]）
plt.figure()
plt.plot(target_numeric, label="Original (numeric features)")
plt.plot(recon_numeric, label="Reconstructed (numeric features)")
plt.legend()
plt.title("Numeric Features: Original vs Reconstructed (No Sampling, Using μ)")
plt.xlabel("Feature Index")
plt.ylabel("Scaled Value [0,1]")
plt.tight_layout()
plt.show()
