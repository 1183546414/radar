
"""
Train a deep Q regressor Q(s, a) -> SINR from JSON files.

Inputs (all expected under DATA_DIR):
- Encoded_Jamming_Type.json       -> list[int] or list[float]
- Jammer_carrierFrequency.json    -> list[list[number]] or list[number]
- Jamming_Params.json             -> list[list[number]] or list[number]
- radar_trans.json                -> list[list[Any]] rows: [Radar_carrierFrequency, ..., maybe strings]
- SINR.json                       -> list[float] as reward

State S := concat(Encoded_Jamming_Type, Jammer_carrierFrequency..., Jamming_Params...)
Action A := from radar_trans.json:
   - Radar_carrierFrequency := first numeric in each row
   - Radar_Params           := remaining numerics in the row (strings are treated as categorical tokens)
   - Categorical tokens     := one-hot, multi-label per row (presence vector)

Output:
- Trained PyTorch model (q_regressor.pt)
- Standard scalers and token vocabulary (joblib + json)
- A quick eval printout and an example inference
"""

import os
import json
import random
from typing import Any, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# -----------------------------
# Config
# -----------------------------
DATA_DIR=  "./dataset/data_2/json"

#DATA_DIR = "/mnt/data"
PATH_ENC_JAM_TYPE = os.path.join(DATA_DIR, "Encoded_Jamming_Type.json")
PATH_JAM_CF        = os.path.join(DATA_DIR, "Jammer_carrierFrequency.json")
PATH_JAM_PARAMS    = os.path.join(DATA_DIR, "Jamming_Params.json")
PATH_RADAR_TRANS   = os.path.join(DATA_DIR, "radar_trans.json")
PATH_SINR          = os.path.join(DATA_DIR, "SINR.json")

ARTIFACT_DIR = os.path.join(DATA_DIR, "q_regressor_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None  # use None for non-numeric

def pad_or_truncate(seq: List[float], target_len: int, pad_value: float = 0.0) -> List[float]:
    seq = list(seq)
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [pad_value] * (target_len - len(seq))

def infer_max_len(list_of_rows: List[Any], cap: int) -> int:
    max_len = 0
    for row in list_of_rows:
        if isinstance(row, list):
            max_len = max(max_len, len(row))
        else:
            max_len = max(max_len, 1)
    return min(max_len, cap)

def split_numeric_and_tokens(row: List[Any]) -> Tuple[List[float], List[str]]:
    nums, toks = [], []
    for v in row:
        f = to_float(v)
        if f is None:
            toks.append(str(v))
        else:
            nums.append(f)
    return nums, toks


# -----------------------------
# Load data数据加载和对齐
# -----------------------------
print("Loading JSON ...")
enc_jam_type = load_json(PATH_ENC_JAM_TYPE)
jam_cf       = load_json(PATH_JAM_CF)
jam_params   = load_json(PATH_JAM_PARAMS)
radar_trans  = load_json(PATH_RADAR_TRANS)
sinr_list    = load_json(PATH_SINR)

# Align lengths
N = min(len(enc_jam_type), len(jam_cf), len(jam_params), len(radar_trans), len(sinr_list))
enc_jam_type = enc_jam_type[:N]
jam_cf       = jam_cf[:N]
jam_params   = jam_params[:N]
radar_trans  = radar_trans[:N]
sinr_list    = sinr_list[:N]
print(f"Aligned dataset length: {N}")


# -----------------------------
# Build State S  构建状态S
# -----------------------------
# Encoded_Jamming_Type -> (N, 1)
enc_jam_type_arr = np.asarray(enc_jam_type, dtype=np.float32).reshape(-1, 1)

# Jammer_carrierFrequency may be list[list] or list[number]
jam_cf_maxlen = infer_max_len(jam_cf, cap=8)
jam_cf_arr = []
for row in jam_cf:
    row = row if isinstance(row, list) else [row]
    row = [to_float(x) or 0.0 for x in row]
    jam_cf_arr.append(pad_or_truncate(row, jam_cf_maxlen, 0.0))
jam_cf_arr = np.asarray(jam_cf_arr, dtype=np.float32)

# Jamming_Params (variable length)
jam_params_maxlen = infer_max_len(jam_params, cap=64)
jam_params_arr = []
for row in jam_params:
    row = row if isinstance(row, list) else [row]
    row = [to_float(x) or 0.0 for x in row]
    jam_params_arr.append(pad_or_truncate(row, jam_params_maxlen, 0.0))
jam_params_arr = np.asarray(jam_params_arr, dtype=np.float32)

S = np.concatenate([enc_jam_type_arr, jam_cf_arr, jam_params_arr], axis=1)
print("State S shape:", S.shape)


# -----------------------------
# Build Action A from radar_trans 构建动作A
# -----------------------------
radar_numeric_rows: List[List[float]] = []
radar_token_rows:   List[List[str]]   = []

for row in radar_trans:
    row = row if isinstance(row, list) else [row]
    nums, toks = split_numeric_and_tokens(row)
    radar_numeric_rows.append(nums)
    radar_token_rows.append(toks)

# First numeric in each row -> Radar_carrierFrequency
radar_cf = [(nums[0] if len(nums)>0 else 0.0) for nums in radar_numeric_rows]
radar_cf = np.asarray(radar_cf, dtype=np.float32).reshape(-1, 1)

# Remaining numerics -> Radar_Params (pad)
radar_params_only = [nums[1:] if len(nums)>1 else [] for nums in radar_numeric_rows]
radar_params_maxlen = infer_max_len(radar_params_only, cap=64)
radar_params_arr = []
for row in radar_params_only:
    row = [x if x is not None else 0.0 for x in row]
    radar_params_arr.append(pad_or_truncate(row, radar_params_maxlen, 0.0))
radar_params_arr = np.asarray(radar_params_arr, dtype=np.float32)

# Categorical tokens -> multi-hot
all_tokens = sorted({tok for toks in radar_token_rows for tok in toks})
token_index = {tok:i for i, tok in enumerate(all_tokens)}
A_tokens = np.zeros((N, len(all_tokens)), dtype=np.float32)
for i, toks in enumerate(radar_token_rows):
    for t in toks:
        if t in token_index:
            A_tokens[i, token_index[t]] = 1.0

A = np.concatenate([radar_cf, radar_params_arr, A_tokens], axis=1)
print("Action A shape:", A.shape, "| token dims:", len(all_tokens))

# -----------------------------
# Reward R
# -----------------------------
R = np.asarray(sinr_list[:N], dtype=np.float32).reshape(-1, 1)
print("Reward R shape:", R.shape)

# -----------------------------
# Train/Val split + scaling  数据预处理
# -----------------------------
X_sa = np.concatenate([S, A], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X_sa, R, test_size=0.2, random_state=SEED)

scaler_x = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)  # scale target helps regression stability

X_train_n = scaler_x.transform(X_train).astype(np.float32)
X_val_n   = scaler_x.transform(X_val).astype(np.float32)
y_train_n = scaler_y.transform(y_train).astype(np.float32)
y_val_n   = scaler_y.transform(y_val).astype(np.float32)

joblib.dump(scaler_x, os.path.join(ARTIFACT_DIR, "scaler_x.joblib"))
joblib.dump(scaler_y, os.path.join(ARTIFACT_DIR, "scaler_y.joblib"))
with open(os.path.join(ARTIFACT_DIR, "radar_token_vocab.json"), "w", encoding="utf-8") as f:
    json.dump(token_index, f, ensure_ascii=False, indent=2)


# -----------------------------
# Torch dataset
# -----------------------------
class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ArrayDataset(X_train_n, y_train_n)
val_ds   = ArrayDataset(X_val_n, y_val_n)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False)


# -----------------------------
# Q regressor model
# -----------------------------
## 减小网络规模
#hidden = [128, 64, 32]  # 替代原来的[512, 512, 256]

class QRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [256,128], dropout: float = 0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

in_dim = X_train_n.shape[1]
model = QRegressor(in_dim).train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss()  # Huber

# -----------------------------
# Training
# -----------------------------
def evaluate():
    model.eval()
    mae, mse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            mse += torch.mean((pred - yb)**2).item() * xb.size(0)
            mae += torch.mean(torch.abs(pred - yb)).item() * xb.size(0)
            n += xb.size(0)
    model.train()
    return mae / n, mse / n

best_mse = float("inf")
patience, bad = 10, 0

for epoch in range(200):
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val_mae, val_mse = evaluate()
    if val_mse < best_mse - 1e-6:
        best_mse = val_mse
        bad = 0
        torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "q_regressor.pt"))
    else:
        bad += 1
    if (epoch+1) % 20 == 0 or bad >= patience:
        print(f"Epoch {epoch+1:03d} | val_MAE(z)={val_mae:.4f} val_MSE(z)={val_mse:.4f} {'*' if bad==0 else ''}")
    if bad >= patience:
        print("Early stopping.")
        break

# Reload best and report real-scale MAE
model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "q_regressor.pt"), map_location=device))
model.eval()

with torch.no_grad():
    pred_val_n = []
    for xb, yb in val_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        pred_val_n.append(pred)
    pred_val_n = np.vstack(pred_val_n)
    # invert scaling
    pred_val = scaler_y.inverse_transform(pred_val_n)
    y_val_real = scaler_y.inverse_transform(y_val_n)
    real_mae = np.mean(np.abs(pred_val - y_val_real))
    real_rmse = np.sqrt(np.mean((pred_val - y_val_real)**2))
print(f"Validation MAE (SINR units): {real_mae:.4f} | RMSE: {real_rmse:.4f}")

# Example inference on the first sample
full_X = np.concatenate([S, A], axis=1)
x0 = full_X[0:1]
x0_n = scaler_x.transform(x0).astype(np.float32)
with torch.no_grad():
    y0_pred_n = model(torch.from_numpy(x0_n).to(device)).cpu().numpy()
y0_pred = scaler_y.inverse_transform(y0_pred_n)[0,0]
print(f"Example prediction on sample 0 -> predicted SINR ≈ {y0_pred:.4f}")

print("Artifacts saved to:", ARTIFACT_DIR)
