"""
Train a deep Q regressor Q(s, a) -> SINR from JSON files with improvements.

Key improvements:
1. Data exploration & outlier handling
2. Feature engineering with domain knowledge
3. Optimized model architecture for small datasets
4. Advanced training strategies (LR scheduling, early stopping)
5. Data augmentation
6. Physical constraints on output
7. Baseline comparison with XGBoost
8. Comprehensive error analysis

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
- Comprehensive evaluation metrics and error analysis
- Example inference with confidence estimation
"""

import os
import json
import random
from typing import Any, List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "./dataset/data_2/json"
# DATA_DIR = "/mnt/data"
PATH_ENC_JAM_TYPE = os.path.join(DATA_DIR, "Encoded_Jamming_Type.json")
PATH_JAM_CF = os.path.join(DATA_DIR, "Jammer_carrierFrequency.json")
PATH_JAM_PARAMS = os.path.join(DATA_DIR, "Jamming_Params.json")
PATH_RADAR_TRANS = os.path.join(DATA_DIR, "radar_trans.json")
PATH_SINR = os.path.join(DATA_DIR, "SINR.json")

ARTIFACT_DIR = os.path.join(DATA_DIR, "q_regressor_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Training hyperparameters
BATCH_SIZE = 128
VAL_SPLIT = 0.2
DROPOUT_RATE = 0.3
HIDDEN_LAYERS = [256, 128]  # Reduced size for small dataset
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
LR_SCHEDULER_PATIENCE = 8
MIN_DELTA = 1e-5
SINR_LOWER_BOUND = -20.0  # Physical constraint for SINR (dB)
SINR_UPPER_BOUND = 30.0   # Physical constraint for SINR (dB)
DATA_AUGMENTATION_FACTOR = 0.3  # 30% additional synthetic samples


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: str) -> Any:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        raise


def to_float(x):
    """Convert to float or return None if conversion fails."""
    try:
        return float(x)
    except Exception:
        return None


def pad_or_truncate(seq: List[float], target_len: int, pad_value: float = 0.0) -> List[float]:
    """Pad or truncate sequence to target length."""
    seq = list(seq)
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [pad_value] * (target_len - len(seq))


def infer_max_len(list_of_rows: List[Any], cap: int) -> int:
    """Infer maximum length of rows with cap."""
    max_len = 0
    for row in list_of_rows:
        if isinstance(row, list):
            max_len = max(max_len, len(row))
        else:
            max_len = max(max_len, 1)
    return min(max_len, cap)


def split_numeric_and_tokens(row: List[Any]) -> Tuple[List[float], List[str]]:
    """Split row into numeric values and string tokens."""
    nums, toks = [], []
    for v in row:
        f = to_float(v)
        if f is None:
            toks.append(str(v))
        else:
            nums.append(f)
    return nums, toks


# -----------------------------
# Data exploration and cleaning
# -----------------------------
def analyze_and_clean_data(sinr_list: List[float], 
                          jam_cf: List, 
                          radar_cf: List[float]) -> Tuple[List[bool], np.ndarray]:
    """
    Analyze data distribution and identify outliers.
    Returns mask for valid samples and frequency difference feature.
    """
    # Convert SINR to numpy array for analysis
    sinr_arr = np.array(sinr_list)
    
    # 1. SINR outlier detection using IQR
    q1, q3 = np.percentile(sinr_arr, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    sinr_mask = (sinr_arr >= lower_bound) & (sinr_arr <= upper_bound)
    
    # 2. Analyze frequency differences (domain knowledge)
    freq_diffs = []
    for i in range(len(jam_cf)):
        jam_freq = jam_cf[i] if isinstance(jam_cf[i], list) else [jam_cf[i]]
        jam_freq = [to_float(f) for f in jam_freq if to_float(f) is not None]
        
        if len(jam_freq) > 0 and isinstance(radar_cf[i], (int, float)):
            # Calculate min absolute frequency difference
            diff = min([abs(float(radar_cf[i]) - f) for f in jam_freq if f is not None])
            freq_diffs.append(diff)
        else:
            freq_diffs.append(0.0)
    
    freq_diffs = np.array(freq_diffs)
    freq_q1, freq_q3 = np.percentile(freq_diffs, [25, 75])
    freq_iqr = freq_q3 - freq_q1
    freq_mask = (freq_diffs >= (freq_q1 - 3 * freq_iqr)) & (freq_diffs <= (freq_q3 + 3 * freq_iqr))
    
    # Combine masks - keep samples that pass both filters
    valid_mask = sinr_mask & freq_mask
    
    print(f"Data analysis: {np.sum(valid_mask)}/{len(valid_mask)} samples retained after outlier removal")
    print(f"SINR range after cleaning: [{sinr_arr[valid_mask].min():.2f}, {sinr_arr[valid_mask].max():.2f}] dB")
    
    return valid_mask, freq_diffs


def augment_data(X: np.ndarray, y: np.ndarray, factor: float = 0.3, noise_std: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment data by adding Gaussian noise to existing samples.
    factor: proportion of new samples to add (e.g., 0.3 = 30% more samples)
    noise_std: standard deviation of noise relative to feature scale
    """
    n_samples, n_features = X.shape
    n_aug = int(n_samples * factor)
    
    if n_aug == 0:
        return X, y
    
    # Select random indices to augment
    idx = np.random.choice(n_samples, n_aug, replace=True)
    X_aug = X[idx].copy()
    y_aug = y[idx].copy()
    
    # Add Gaussian noise (scale-aware)
    for i in range(n_features):
        feature_std = np.std(X[:, i])
        X_aug[:, i] += np.random.normal(0, noise_std * feature_std, n_aug)
    
    # Add small noise to targets
    y_std = np.std(y)
    y_aug += np.random.normal(0, noise_std * y_std, n_aug).reshape(-1, 1)
    
    # Combine original and augmented data
    X_combined = np.vstack([X, X_aug])
    y_combined = np.vstack([y, y_aug])
    
    print(f"Data augmentation: added {n_aug} samples ({factor*100:.0f}% increase)")
    return X_combined, y_combined


def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Error Analysis"):
    """Plot comprehensive error analysis."""
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    plt.figure(figsize=(15, 10))
    
    # 1. Error distribution
    plt.subplot(2, 2, 1)
    plt.hist(errors, bins=30)
    plt.title(f'Error Distribution (MAE={mae:.4f})')
    plt.xlabel('Error (True - Predicted)')
    plt.ylabel('Frequency')
    
    # 2. True vs Predicted
    plt.subplot(2, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('True vs Predicted')
    plt.xlabel('True SINR')
    plt.ylabel('Predicted SINR')
    
    # 3. Error vs True Value
    plt.subplot(2, 2, 3)
    plt.scatter(y_true, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Error vs True Value')
    plt.xlabel('True SINR')
    plt.ylabel('Error (True - Predicted)')
    
    # 4. Absolute Error vs True Value
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, np.abs(errors), alpha=0.5)
    plt.title('Absolute Error vs True Value')
    plt.xlabel('True SINR')
    plt.ylabel('Absolute Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, f"{title.lower().replace(' ', '_')}_analysis.png"))
    plt.close()
    
    # Save error statistics
    error_stats = {
        "mae": float(mae),
        "rmse": float(rmse),
        "median_error": float(np.median(errors)),
        "95_percentile_error": float(np.percentile(np.abs(errors), 95))
    }
    with open(os.path.join(ARTIFACT_DIR, f"{title.lower().replace(' ', '_')}_stats.json"), "w") as f:
        json.dump(error_stats, f, indent=2)
    
    return error_stats


# -----------------------------
# Load data
# -----------------------------
print("Loading JSON ...")
enc_jam_type = load_json(PATH_ENC_JAM_TYPE)
jam_cf = load_json(PATH_JAM_CF)
jam_params = load_json(PATH_JAM_PARAMS)
radar_trans = load_json(PATH_RADAR_TRANS)
sinr_list = load_json(PATH_SINR)

# Align lengths
N = min(len(enc_jam_type), len(jam_cf), len(jam_params), len(radar_trans), len(sinr_list))
enc_jam_type = enc_jam_type[:N]
jam_cf = jam_cf[:N]
jam_params = jam_params[:N]
radar_trans = radar_trans[:N]
sinr_list = sinr_list[:N]
print(f"Initial dataset length: {N}")

# -----------------------------
# Build State S
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
# Build Action A from radar_trans
# -----------------------------
radar_numeric_rows: List[List[float]] = []
radar_token_rows: List[List[str]] = []

for row in radar_trans:
    row = row if isinstance(row, list) else [row]
    nums, toks = split_numeric_and_tokens(row)
    radar_numeric_rows.append(nums)
    radar_token_rows.append(toks)

# First numeric in each row -> Radar_carrierFrequency
radar_cf = [(nums[0] if len(nums) > 0 else 0.0) for nums in radar_numeric_rows]
radar_cf = np.asarray(radar_cf, dtype=np.float32).reshape(-1, 1)

# Remaining numerics -> Radar_Params (pad)
radar_params_only = [nums[1:] if len(nums) > 1 else [] for nums in radar_numeric_rows]
radar_params_maxlen = infer_max_len(radar_params_only, cap=64)
radar_params_arr = []
for row in radar_params_only:
    row = [x if x is not None else 0.0 for x in row]
    radar_params_arr.append(pad_or_truncate(row, radar_params_maxlen, 0.0))
radar_params_arr = np.asarray(radar_params_arr, dtype=np.float32)

# Categorical tokens -> multi-hot
all_tokens = sorted({tok for toks in radar_token_rows for tok in toks})
token_index = {tok: i for i, tok in enumerate(all_tokens)}
A_tokens = np.zeros((N, len(all_tokens)), dtype=np.float32)
for i, toks in enumerate(radar_token_rows):
    for t in toks:
        if t in token_index:
            A_tokens[i, token_index[t]] = 1.0

A = np.concatenate([radar_cf, radar_params_arr, A_tokens], axis=1)
print("Action A shape:", A.shape, "| token dims:", len(all_tokens))

# -----------------------------
# Data exploration and cleaning
# -----------------------------
valid_mask, freq_diffs = analyze_and_clean_data(sinr_list, jam_cf, radar_cf.flatten())

# Apply mask to all data
if np.sum(valid_mask) < N:
    enc_jam_type = [enc_jam_type[i] for i in range(N) if valid_mask[i]]
    jam_cf = [jam_cf[i] for i in range(N) if valid_mask[i]]
    jam_params = [jam_params[i] for i in range(N) if valid_mask[i]]
    radar_trans = [radar_trans[i] for i in range(N) if valid_mask[i]]
    sinr_list = [sinr_list[i] for i in range(N) if valid_mask[i]]
    
    S = S[valid_mask]
    A = A[valid_mask]
    freq_diffs = freq_diffs[valid_mask]
    
    N = len(sinr_list)
    print(f"Dataset length after cleaning: {N}")

# Add domain-specific feature: frequency difference
freq_diffs = freq_diffs.reshape(-1, 1)
S = np.concatenate([S, freq_diffs], axis=1)
print("State S shape after adding domain feature:", S.shape)

# -----------------------------
# Reward R
# -----------------------------
R = np.asarray(sinr_list, dtype=np.float32).reshape(-1, 1)
print("Reward R shape:", R.shape)

# -----------------------------
# Train/Val split + scaling
# -----------------------------
# Create combined input (S, A) for training
X_sa = np.concatenate([S, A], axis=1)

# Split data (stratify by SINR to ensure balanced train/val)
sinr_bins = pd.cut(R.flatten(), bins=10, labels=False) if N > 100 else None
stratify_param = sinr_bins if sinr_bins is not None else None

X_train, X_val, y_train, y_val = train_test_split(
    X_sa, R, test_size=VAL_SPLIT, random_state=SEED, stratify=stratify_param
)

# Data augmentation
if DATA_AUGMENTATION_FACTOR > 0:
    X_train, y_train = augment_data(X_train, y_train, factor=DATA_AUGMENTATION_FACTOR)

# Scale data
scaler_x = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train_n = scaler_x.transform(X_train).astype(np.float32)
X_val_n = scaler_x.transform(X_val).astype(np.float32)
y_train_n = scaler_y.transform(y_train).astype(np.float32)
y_val_n = scaler_y.transform(y_val).astype(np.float32)

# Save preprocessing artifacts
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
val_ds = ArrayDataset(X_val_n, y_val_n)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, drop_last=False)

# -----------------------------
# Q regressor model with physical constraints
# -----------------------------
class QRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [256, 128], dropout: float = 0.3):
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
        pred = self.net(x)
        # Apply physical constraints on SINR output
        return torch.clamp(pred, SINR_LOWER_BOUND, SINR_UPPER_BOUND)

in_dim = X_train_n.shape[1]
model = QRegressor(in_dim, hidden=HIDDEN_LAYERS, dropout=DROPOUT_RATE).train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, 'min', patience=LR_SCHEDULER_PATIENCE, factor=0.5, verbose=True
)
loss_fn = nn.SmoothL1Loss()  # Huber loss

# -----------------------------
# Training with improved evaluation
# -----------------------------
def evaluate(model, loader, scaler_y, device, return_predictions=False):
    """Evaluate model with option to return predictions."""
    model.eval()
    total_loss, mae, mse, n = 0.0, 0.0, 0.0, 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            # Convert to numpy for MAE/RMSE in original scale
            pred_np = pred.cpu().numpy()
            yb_np = yb.cpu().numpy()
            
            # Inverse transform to original scale
            pred_real = scaler_y.inverse_transform(pred_np)
            yb_real = scaler_y.inverse_transform(yb_np)
            
            # Calculate metrics in original scale
            batch_mae = np.mean(np.abs(pred_real - yb_real))
            batch_mse = np.mean((pred_real - yb_real) ** 2)
            
            total_loss += loss.item() * xb.size(0)
            mae += batch_mae * xb.size(0)
            mse += batch_mse * xb.size(0)
            n += xb.size(0)
            
            if return_predictions:
                all_preds.append(pred_real)
                all_targets.append(yb_real)
    
    model.train()
    results = {
        'loss': total_loss / n,
        'mae': mae / n,
        'rmse': np.sqrt(mse / n)
    }
    
    if return_predictions:
        results['preds'] = np.vstack(all_preds)
        results['targets'] = np.vstack(all_targets)
    
    return results

# -----------------------------
# XGBoost baseline for comparison
# -----------------------------
def train_xgboost_baseline(X_train, y_train, X_val, y_val, scaler_y):
    """Train XGBoost model as baseline for comparison."""
    print("\nTraining XGBoost baseline for comparison...")
    
    # Inverse transform y to original scale for XGBoost
    y_train_orig = scaler_y.inverse_transform(y_train).ravel()
    y_val_orig = scaler_y.inverse_transform(y_val).ravel()
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED
    )
    xgb_model.fit(
        X_train, y_train_orig,
        eval_set=[(X_val, y_val_orig)],
        early_stopping_rounds=30,
        verbose=False
    )
    
    # Evaluate
    xgb_pred = xgb_model.predict(X_val)
    xgb_mae = np.mean(np.abs(xgb_pred - y_val_orig))
    xgb_rmse = np.sqrt(np.mean((xgb_pred - y_val_orig) ** 2))
    
    print(f"XGBoost baseline - MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}")
    
    # Save XGBoost model
    xgb_model.save_model(os.path.join(ARTIFACT_DIR, "xgboost_baseline.json"))
    
    # Feature importance
    feat_importance = xgb_model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feat_importance)), feat_importance)
    plt.title('XGBoost Feature Importance')
    plt.savefig(os.path.join(ARTIFACT_DIR, "xgboost_feature_importance.png"))
    plt.close()
    
    return xgb_mae, xgb_rmse, xgb_model

# -----------------------------
# Main training loop
# -----------------------------
print("\nStarting training...")
best_val_rmse = float("inf")
patience_counter = 0
train_history = {'train_mae': [], 'val_mae': [], 'train_rmse': [], 'val_rmse': []}

for epoch in range(MAX_EPOCHS):
    # Training
    model.train()
    train_mae, train_rmse, n = 0.0, 0.0, 0
    
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        # Track metrics in original scale
        pred_np = pred.detach().cpu().numpy()
        yb_np = yb.cpu().numpy()
        
        # Inverse transform
        pred_real = scaler_y.inverse_transform(pred_np)
        yb_real = scaler_y.inverse_transform(yb_np)
        
        train_mae += np.mean(np.abs(pred_real - yb_real)) * xb.size(0)
        train_rmse += np.mean((pred_real - yb_real) ** 2) * xb.size(0)
        n += xb.size(0)
    
    train_mae /= n
    train_rmse = np.sqrt(train_rmse / n)
    
    # Validation
    val_metrics = evaluate(model, val_loader, scaler_y, device)
    val_mae = val_metrics['mae']
    val_rmse = val_metrics['rmse']
    
    # Learning rate scheduling
    scheduler.step(val_rmse)
    
    # Record history
    train_history['train_mae'].append(train_mae)
    train_history['val_mae'].append(val_mae)
    train_history['train_rmse'].append(train_rmse)
    train_history['val_rmse'].append(val_rmse)
    
    # Early stopping check
    if val_rmse < best_val_rmse - MIN_DELTA:
        best_val_rmse = val_rmse
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "q_regressor.pt"))
        print(f"Epoch {epoch+1:03d} | New best model - val_MAE={val_mae:.4f} val_RMSE={val_rmse:.4f} *")
    else:
        patience_counter += 1
        if (epoch + 1) % 10 == 0 or patience_counter >= EARLY_STOPPING_PATIENCE - 5:
            print(f"Epoch {epoch+1:03d} | val_MAE={val_mae:.4f} val_RMSE={val_rmse:.4f} (patience: {patience_counter})")
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# -----------------------------
# Final evaluation and analysis
# -----------------------------
# Load best model
model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "q_regressor.pt"), map_location=device))
model.eval()

# Comprehensive evaluation
train_metrics = evaluate(model, train_loader, scaler_y, device, return_predictions=True)
val_metrics = evaluate(model, val_loader, scaler_y, device, return_predictions=True)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_history['train_mae'], label='Train MAE')
plt.plot(train_history['val_mae'], label='Val MAE')
plt.title('MAE during Training')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_history['train_rmse'], label='Train RMSE')
plt.plot(train_history['val_rmse'], label='Val RMSE')
plt.title('RMSE during Training')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "training_history.png"))
plt.close()

# Error analysis
print("\nPerforming error analysis...")
plot_error_analysis(val_metrics['targets'], val_metrics['preds'], "Validation")

# XGBoost baseline comparison
xgb_mae, xgb_rmse, xgb_model = train_xgboost_baseline(
    X_train, y_train, X_val, y_val, scaler_y
)

# Compare model performances
performance_comparison = {
    "deep_q_regressor": {
        "val_mae": float(val_metrics['mae']),
        "val_rmse": float(val_metrics['rmse'])
    },
    "xgboost_baseline": {
        "val_mae": float(xgb_mae),
        "val_rmse": float(xgb_rmse)
    }
}

with open(os.path.join(ARTIFACT_DIR, "performance_comparison.json"), "w") as f:
    json.dump(performance_comparison, f, indent=2)

# Example inference with confidence estimation
print("\nExample inference with confidence estimation:")
full_X = np.concatenate([S, A], axis=1)
x0 = full_X[0:1]
x0_n = scaler_x.transform(x0).astype(np.float32)

# Monte Carlo dropout for uncertainty estimation
model.train()  # Enable dropout for inference
n_samples = 50
pred_samples = []

with torch.no_grad():
    for _ in range(n_samples):
        y0_pred_n = model(torch.from_numpy(x0_n).to(device)).cpu().numpy()
        pred_samples.append(scaler_y.inverse_transform(y0_pred_n)[0, 0])

pred_mean = np.mean(pred_samples)
pred_std = np.std(pred_samples)
true_value = R[0][0]

print(f"Sample 0 - True SINR: {true_value:.4f} dB")
print(f"Predicted SINR: {pred_mean:.4f} dB (±{1.96*pred_std:.4f} dB, 95% CI)")
print(f"Confidence interval: [{pred_mean-1.96*pred_std:.4f}, {pred_mean+1.96*pred_std:.4f}] dB")

# Save uncertainty analysis
uncertainty_analysis = {
    "true_value": float(true_value),
    "predicted_mean": float(pred_mean),
    "predicted_std": float(pred_std),
    "confidence_interval_95": [float(pred_mean-1.96*pred_std), float(pred_mean+1.96*pred_std)]
}
with open(os.path.join(ARTIFACT_DIR, "uncertainty_analysis.json"), "w") as f:
    json.dump(uncertainty_analysis, f, indent=2)

print(f"\nAll artifacts saved to: {ARTIFACT_DIR}")
print("Training completed successfully!")
