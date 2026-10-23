# train_supervised_sinr_regressor.py
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1) 你的 SINR 计算（沿用原逻辑）
# ----------------------------
def calculate_sinr_reward(jammer_params, radar_params):
    # 频率单位转换 (GHz -> Hz)
    f_j = jammer_params['freq'] * 1e9
    bw_j = jammer_params['bw'] * 1e9
    gain_j = jammer_params['gain']

    f_r = radar_params['freq'] * 1e9
    bw_r = radar_params['bw'] * 1e9
    gain_r = radar_params['gain']

    # 重叠带宽
    overlap_min = max(f_j - bw_j/2, f_r - bw_r/2)
    overlap_max = min(f_j + bw_j/2, f_r + bw_r/2)
    overlap_bw = max(0.0, overlap_max - overlap_min)

    signal_p = 10 ** (gain_r / 10.0)

    effective_bw_j = bw_j if bw_j > 0 else 1e6
    jammer_p = (10 ** (gain_j / 10.0)) * (overlap_bw / effective_bw_j)

    sinr = 10 * np.log10(signal_p / (jammer_p + 1e-9))
    return float(sinr)

# ----------------------------
# 2) 固定的候选 radar 动作表（建议固定seed，可复现）
# ----------------------------
def make_radar_candidates(n=200, seed=42):
    rng = random.Random(seed)
    table = []
    for _ in range(n):
        table.append({
            "freq": rng.uniform(3.0, 5.0),          # GHz
            "gain": rng.randint(20, 40),            # dB
            "bw": rng.uniform(0.005, 0.025),        # GHz
            "pw": rng.randint(20000, 40000),        # ns (按你原定义)
            "type_idx": rng.randint(0, 3),          # 0~3
        })
    return table

def radar_to_features(r):
    # 连续特征 + type_idx归一化（也可改one-hot）
    return np.array([
        r["freq"],          # GHz
        r["gain"],          # dB
        r["bw"],            # GHz
        r["pw"],            # ns
        r["type_idx"] / 3.0 # 归一化到[0,1]
    ], dtype=np.float32)

# ----------------------------
# 3) 数据构造： (jammer + radar) -> sinr
# ----------------------------
class SinrDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------
# 4) MLP 回归网络
# ----------------------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# 5) 训练与验证
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    crit = nn.MSELoss()
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = crit(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    crit = nn.MSELoss()
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = crit(pred, yb)
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

# ----------------------------
# 6) 推理：给定 jammer，选最大预测 SINR 的 radar
# ----------------------------
@torch.no_grad()
def pick_best_radar(model, scaler, jammer_feat_row, radar_candidates, device):
    # jammer_feat_row: shape (J,)
    jammer_feat_row = np.asarray(jammer_feat_row, dtype=np.float32)

    X_list = []
    for r in radar_candidates:
        rf = radar_to_features(r)
        X_list.append(np.concatenate([jammer_feat_row, rf], axis=0))
    X = np.stack(X_list, axis=0)

    Xs = scaler.transform(X)
    Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
    preds = model(Xt).squeeze(1).cpu().numpy()  # (N,)
    best_idx = int(np.argmax(preds))
    return radar_candidates[best_idx], float(preds[best_idx])

# ----------------------------
# 7) 主程序
# ----------------------------
def main():
    # === 你要改的地方：CSV路径 ===
    csv_path = "Top5_Vertical_Comparison.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {csv_path}，请把CSV放到当前目录或改路径。")

    df = pd.read_csv(csv_path)

    # jammer 状态列（沿用你原来那组）
    jammer_cols = [
        'Query_Jammer_CarrierFreq_GHz', 'Query_Jammer_Gain_dB', 'Query_Jammer_Bandwidth_GHz',
        'Query_Jammer_Sweep_Step_GHz', 'Query_Jammer_Sweep_Power', 'Query_Jammer_IntSample_Dur_ns',
        'Query_Jammer_IntSample_Delay_ns', 'Query_Jammer_IntSample_Intv_ns', 'Query_Jammer_FalseTarget_Num',
        'Query_Jammer_FalseTarget_Intv_m', 'Query_Jammer_FalseTarget_Delay_m'
    ]

    # 取主 query：每 5 行一个 query（你原训练就是这么取的）
    jammer_rows = df.iloc[::5][jammer_cols].values.astype(np.float32)

    # 候选 radar 配置
    radar_candidates = make_radar_candidates(n=200, seed=42)

    # 构造监督数据
    # 每个 jammer 采样 M 个 radar，形成 (N*M) 条样本
    M = 30  # 每个 jammer采样多少个 radar（越大数据越多，但训练更慢）
    rng = np.random.default_rng(123)

    X_list, y_list = [], []
    for j_feat in jammer_rows:
        # 从候选集中采样 M 个 radar
        idxs = rng.choice(len(radar_candidates), size=M, replace=False)
        # jam config 只需要 freq/bw/gain 参与你 sinr 计算
        jammer_config = {
            "freq": float(j_feat[jammer_cols.index("Query_Jammer_CarrierFreq_GHz")]),
            "bw": float(j_feat[jammer_cols.index("Query_Jammer_Bandwidth_GHz")]),
            "gain": float(j_feat[jammer_cols.index("Query_Jammer_Gain_dB")]),
        }
        for k in idxs:
            r = radar_candidates[k]
            rf = radar_to_features(r)
            x = np.concatenate([j_feat, rf], axis=0)
            sinr = calculate_sinr_reward(jammer_config, r)
            X_list.append(x)
            y_list.append(sinr)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)

    # 划分训练/验证
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # 标准化（强烈推荐）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = SinrDataset(X_train_s, y_train)
    val_ds = SinrDataset(X_val_s, y_val)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(in_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # 训练
    best_val = float("inf")
    best_path = "sinr_mlp_best.pth"
    epochs = 30

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {ep:02d}/{epochs} | train_mse={tr:.6f} | val_mse={va:.6f}")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "jammer_cols": jammer_cols
            }, best_path)

    print(f"训练完成，最佳验证 MSE={best_val:.6f}，模型已保存到 {best_path}")

    # 演示：对第一个 jammer，选最优 radar
    # 加载 best 模型
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 重建 scaler
    scaler2 = StandardScaler()
    scaler2.mean_ = ckpt["scaler_mean"]
    scaler2.scale_ = ckpt["scaler_scale"]
    scaler2.var_ = (scaler2.scale_ ** 2)
    scaler2.n_features_in_ = X.shape[1]

    jammer_feat0 = jammer_rows[0]
    best_radar, pred_sinr = pick_best_radar(model, scaler2, jammer_feat0, radar_candidates, device)

    # 计算真实 sinr（用于对比）
    jammer_config0 = {
        "freq": float(jammer_feat0[jammer_cols.index("Query_Jammer_CarrierFreq_GHz")]),
        "bw": float(jammer_feat0[jammer_cols.index("Query_Jammer_Bandwidth_GHz")]),
        "gain": float(jammer_feat0[jammer_cols.index("Query_Jammer_Gain_dB")]),
    }
    true_sinr = calculate_sinr_reward(jammer_config0, best_radar)

    print("\n=== Demo: 为一个 jammer 选最优 radar ===")
    print("best_radar =", best_radar)
    print(f"pred_sinr  = {pred_sinr:.4f} dB")
    print(f"true_sinr  = {true_sinr:.4f} dB (用公式算的真实值)")

if __name__ == "__main__":
    main()