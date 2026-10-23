"""
train_q_value_regression.py
==============================
这是一个“纯监督版 Q 值回归”脚本。
不需要 d3rlpy 或 next_state，只拟合即时奖励 R。
适用于样本独立、无时序信息的任务（如雷达信号或静态动作评分问题）。
--------------------------------------------
依赖安装：
    pip install torch numpy scikit-learn matplotlib
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================
# 基础配置
# ==========================
from radar.paths import DATA_V2_PROCESSED, OUT_Q_VALUE

DATA_PATH = str(DATA_V2_PROCESSED)
RESULT_PATH = str(OUT_Q_VALUE)
os.makedirs(RESULT_PATH, exist_ok=True)

SEED = 42
EPOCHS = 300
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
PATIENCE = 20  # 早停
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================
# 加载数据
# ==========================
print("Loading data...")
S = np.load(os.path.join(DATA_PATH, "S_std.npy"))
A = np.load(os.path.join(DATA_PATH, "A_std.npy"))
R = np.load(os.path.join(DATA_PATH, "R_std.npy")).reshape(-1)

X = np.concatenate([S, A], axis=1)
print(f"Data shape: X={X.shape}, R={R.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, R, test_size=0.2, random_state=SEED)

# ==========================
# 定义 Q 网络
# ==========================
class QNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = QNet(X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

# ==========================
# 提前停止机制
# ==========================
best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": []}

# ==========================
# 训练循环
# ==========================
print("Training...")
for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    idx = np.random.permutation(len(X_train))
    batch_losses = []
    for i in range(0, len(X_train), BATCH_SIZE):
        batch_idx = idx[i:i+BATCH_SIZE]
        xb = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(device)
        yb = torch.tensor(y_train[batch_idx], dtype=torch.float32).unsqueeze(1).to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    # ---- Validation ----
    model.eval()
    with torch.no_grad():
        xb_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        yb_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        pred_val = model(xb_val)
        val_loss = loss_fn(pred_val, yb_val).item()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # ---- Early Stopping ----
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(RESULT_PATH, "best_q_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

# ==========================
# 加载最优模型评估
# ==========================
model.load_state_dict(torch.load(os.path.join(RESULT_PATH, "best_q_model.pt")))
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy().ravel()

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"\nFinal Evaluation: RMSE={rmse:.4f}, R2={r2:.4f}")

# ==========================
# 绘制结果
# ==========================
plt.figure(figsize=(7, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, "loss_curve.png"))
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred, s=10, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("True R")
plt.ylabel("Predicted Q(s,a)")
plt.title(f"Q Value Regression | R2={r2:.3f}, RMSE={rmse:.3f}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, "val_scatter.png"))
plt.show()

print("\n✅ Training complete. Model saved to ./result/best_q_model.pt")
