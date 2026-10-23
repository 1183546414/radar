# ae_curve_fit_extreme.py
# 极致贴合版：大容量条件AE + 强形状损失 + 长微调 + 检查点功能
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os
import time

# =========================
# 1) 检查点配置
# =========================
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, scheduler, early_stopping, loss_history, filename):
    """保存训练状态到检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'early_stopping_state': {
            'best': early_stopping.best,
            'count': early_stopping.count,
            'stop': early_stopping.stop
        },
        'loss_history': loss_history,
        'timestamp': time.time()
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, filename))
    print(f"检查点已保存: {filename}")

def load_checkpoint(model, optimizer, scheduler, early_stopping, filename):
    """从检查点加载训练状态"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(checkpoint_path):
        print(f"检查点不存在: {filename}")
        return 0, []  # 返回初始epoch和空损失历史
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if early_stopping and 'early_stopping_state' in checkpoint:
        early_stopping.best = checkpoint['early_stopping_state']['best']
        early_stopping.count = checkpoint['early_stopping_state']['count']
        early_stopping.stop = checkpoint['early_stopping_state']['stop']
    
    print(f"已从检查点加载: {filename} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch'], checkpoint.get('loss_history', [])

# =========================
# 2) 数据加载
# =========================
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

jammer_cf = load_json("dataset/data_2/json/Jammer_carrierFrequency.json")    # [freq, gain]
jamming_params = load_json("dataset/data_2/json/Jamming_Params.json")        # 其余数值参数
jamming_type = load_json("dataset/data_2/json/Encoded_Jamming_Type.json")    # 类别

N = min(len(jammer_cf), len(jamming_params), len(jamming_type))

# =========================
# 3) 组装数值特征 + 条件类别（分组归一化）
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
# 4) 张量化
# =========================
x_enc_t = torch.tensor(x_enc, dtype=torch.float32)
y_num_t = torch.tensor(numeric_scaled, dtype=torch.float32)
cond_t  = torch.tensor(cond, dtype=torch.float32)

# =========================
# 5) 特征权重（方差倒数，归一）
# =========================
col_var = np.var(numeric_scaled, axis=0) + 1e-8
w_feat = 1.0 / col_var
w_feat = w_feat / w_feat.mean()
w_feat_t = torch.tensor(w_feat, dtype=torch.float32)  # [D_num]

# =========================
# 6) 数据集 / 加载器
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
# 为了拟合更"细"，可以增大 batch（根据显存调整）
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# =========================
# 7) 模型：更宽更深的条件AE（残差 + GELU + LayerNorm）
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
# 8) 强形状一致性损失：L1 + L2 + 一阶导数 + 余弦相似
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
# 9) 训练与微调配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StrongCurveAE(
    input_dim = x_enc_t.shape[1],
    cond_dim  = C,
    num_numeric = D_num,
    widths=(768, 768, 768),
    latent=384, depth=3, pdrop=0.03
).to(device)

# 阶段1：主训练（较大学习率，支持余弦重启）
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6
)

# 早停（主训练更"耐心"）
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
# 10) 检查点加载（如果存在）
# =========================
start_epoch = 0
loss_history = []
# 尝试加载最新的检查点
start_epoch, loss_history = load_checkpoint(model, optimizer, scheduler, early, "latest_checkpoint.pth")

# =========================
# 11) 阶段1：主训练
# =========================
BATCHES_PER_EPOCH = len(dataloader)
model.train()
for epoch in range(start_epoch, 600):  # 更长的主训练
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
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += loss.item()
        l1m += parts['l1']; l2m += parts['l2']; ldm += parts['ldiff']; lcm += parts['lcos']

    n = len(dataloader)
    avg = running / n
    loss_history.append(avg)
    
    # 在每个epoch结束后更新学习率
    scheduler.step(avg)
    
    print(f"[Stage1] Epoch {epoch+1:03d} | loss {avg:.6f} | L1 {l1m/n:.6f} | L2 {l2m/n:.6f} | d1 {ldm/n:.6f} | cos {lcm/n:.6f}")
    
    # 每10个epoch保存一次检查点
    if (epoch + 1) % 10 == 0:
        save_checkpoint(epoch + 1, model, optimizer, scheduler, early, loss_history, "latest_checkpoint.pth")
    
    # 保存最佳模型
    if early.best is None or avg < early.best:
        save_checkpoint(epoch + 1, model, optimizer, scheduler, early, loss_history, "best_model.pth")
    
    early.step(avg)
    if early.stop:
        print("[Stage1] Early stopping.")
        break

# =========================
# 12) 阶段2：低学习率微调（极致贴合）
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
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        running += loss.item()

    scheduler_ft.step()
    avg_loss = running/len(dataloader)
    loss_history.append(avg_loss)
    print(f"[Stage2-FT] Epoch {epoch+1:03d} | loss {avg_loss:.6f}")
    
    # 每10个epoch保存一次检查点
    if (epoch + 1) % 10 == 0:
        save_checkpoint(600 + epoch + 1, model, optimizer, scheduler_ft, early, loss_history, "latest_checkpoint_ft.pth")

# 保存最终模型
save_checkpoint(600 + fine_tune_epochs, model, optimizer, scheduler_ft, early, loss_history, "final_model.pth")

# =========================
# 13) 评估函数
# =========================
def evaluate_model(model, dataloader, device, w_feat_t):
    """评估模型性能"""
    model.eval()
    total_loss, total_mae, total_mse = 0.0, 0.0, 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x_b, y_b, c_b in dataloader:
            x_b = x_b.to(device); y_b = y_b.to(device); c_b = c_b.to(device)
            pred, _ = model(x_b, c_b)
            
            # 计算损失
            loss, _ = weighted_loss(
                pred, y_b, w_feat_t.to(device),
                w_l1=1.0, w_l2=0.25, w_diff=1.4, w_cos=0.4
            )
            total_loss += loss.item()
            
            # 计算MAE和MSE
            mae = (pred - y_b).abs().mean().item()
            mse = nn.functional.mse_loss(pred, y_b).item()
            total_mae += mae
            total_mse += mse
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_mse = total_mse / num_batches
    
    print(f"[Eval] Average Loss: {avg_loss:.6f} | MAE: {avg_mae:.6f} | MSE: {avg_mse:.6f}")
    return avg_loss, avg_mae, avg_mse

# 评估模型
print("\n评估训练后的模型:")
evaluate_model(model, dataloader, device, w_feat_t)

# =========================
# 14) 从检查点加载并验证的函数
# =========================
def validate_from_checkpoint(checkpoint_name):
    """从检查点加载模型并进行验证"""
    # 创建新模型实例
    eval_model = StrongCurveAE(
        input_dim = x_enc_t.shape[1],
        cond_dim  = C,
        num_numeric = D_num,
        widths=(768, 768, 768),
        latent=384, depth=3, pdrop=0.03
    ).to(device)
    
    # 创建虚拟优化器和调度器
    eval_optimizer = torch.optim.AdamW(eval_model.parameters(), lr=1e-4)
    eval_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        eval_optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6
    )
    eval_early = EarlyStopping(patience=80, min_delta=1e-6)
    
    # 加载检查点
    epoch, _ = load_checkpoint(eval_model, eval_optimizer, eval_scheduler, eval_early, checkpoint_name)
    
    # 评估模型
    print(f"\n评估检查点 '{checkpoint_name}' (epoch {epoch}):")
    evaluate_model(eval_model, dataloader, device, w_feat_t)
    
    return eval_model

# 示例：验证最佳模型
best_model = validate_from_checkpoint("best_model.pth")

# 示例：验证最终模型
final_model = validate_from_checkpoint("final_model.pth")

# =========================
# 15) 可视化代码（保持不变）
# =========================
# [此处保留原有的可视化代码]