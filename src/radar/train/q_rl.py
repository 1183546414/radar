"""
train_q_cql_offline.py
===========================
离线强化学习：使用 Conservative Q-Learning (CQL)
基于 d3rlpy 实现，可直接训练 Q(s,a) 网络。
--------------------------------------------
依赖安装：
    pip install d3rlpy
    pip install numpy matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===========================
# 基础配置
# ===========================
from radar.paths import DATA_V2_PROCESSED, OUT_Q_VALUE

DATA_PATH = str(DATA_V2_PROCESSED)
RESULT_PATH = str(OUT_Q_VALUE)
os.makedirs(RESULT_PATH, exist_ok=True)

# ===========================
# 加载数据
# ===========================
print("Loading data...")
S = np.load(os.path.join(DATA_PATH, "S_std.npy"))
A = np.load(os.path.join(DATA_PATH, "A_std.npy"))
R = np.load(os.path.join(DATA_PATH, "R_std.npy")).reshape(-1)

# 构造 MDP 数据集（离线RL数据结构）
# 这里没有 done 标志，因此假设全为 False
terminals = np.zeros_like(R)
dataset = MDPDataset(S, A, R, terminals)

# 拆分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(
    np.concatenate([S, A], axis=1), R, test_size=0.2, random_state=42
)
print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")

# ===========================
# 创建并配置 CQL 模型
# ===========================
cql = CQL(
    use_gpu=False,                # 若有GPU可设为True
    learning_rate=3e-4,
    batch_size=256,
    alpha=1.0,                    # 保守项权重，越大越保守
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    n_critics=2,
    n_epochs=50,                  # 训练轮数
    encoder_factory="default"
)

# ===========================
# 训练
# ===========================
print("Training CQL model...")
cql.fit(
    dataset=dataset,
    n_steps=50000,
    n_steps_per_epoch=1000,
    eval_episodes=None,
    save_interval=10,
    logdir=os.path.join(RESULT_PATH, "cql_logs")
)

# ===========================
# 验证阶段：预测与评估
# ===========================
print("Evaluating...")
# 将验证集拆开为 S, A
S_val = X_val[:, :S.shape[1]]
A_val = X_val[:, S.shape[1]:]
pred_Q = cql.predict_value(S_val, A_val)

# 计算指标
r2 = r2_score(y_val, pred_Q)
rmse = np.sqrt(mean_squared_error(y_val, pred_Q))

print(f"\nFinal Evaluation:")
print(f"R2 = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")

# ===========================
# 保存模型与结果
# ===========================
MODEL_PATH = os.path.join(RESULT_PATH, "cql_model.pt")
cql.save_model(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# 绘制预测结果散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_val, pred_Q, s=8, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("True R")
plt.ylabel("Predicted Q(s,a)")
plt.title(f"CQL Offline RL | R2={r2:.3f}, RMSE={rmse:.3f}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, "cql_val_scatter.png"))
plt.show()

print("\n✅ Training complete. Check result folder for saved model and plots.")
