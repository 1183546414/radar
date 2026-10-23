# -*- coding: utf-8 -*-
"""
LightGBM 基线：用 [S_std, A_std] → 回归 R_std
- 随机切分 / 指标: RMSE/MAE/R2/Pearson（标准化空间）
- 保存: 模型、验证预测 CSV、特征重要度
用法:
  python train_q_lightgbm.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import json, math

DATA_DIR = Path("./dataset/data_2/json/processed_dataset")
OUT_DIR  = Path("./result/q_lgbm_runs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2

S = np.load(DATA_DIR/"S_std.npy")
A = np.load(DATA_DIR/"A_std.npy")
R = np.load(DATA_DIR/"R_std.npy").reshape(-1)

X = np.concatenate([S, A], axis=1)
X_tr, X_val, y_tr, y_val = train_test_split(X, R, test_size=TEST_SIZE, random_state=SEED)

train_set = lgb.Dataset(X_tr, label=y_tr)
val_set   = lgb.Dataset(X_val, label=y_val)

params = dict(
    objective="regression",
    metric=["l2"],     # l2 -> MSE
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_data_in_leaf=50,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=1.0,
    verbose=-1,
    seed=SEED
)

model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, val_set],
    valid_names=["train","valid"],
    num_boost_round=3000,
    early_stopping_rounds=200,
    verbose_eval=100
)

y_pred = model.predict(X_val, num_iteration=model.best_iteration)
mse = float(np.mean((y_val - y_pred)**2))
rmse = float(math.sqrt(mse))
mae = float(mean_absolute_error(y_val, y_pred))
r2  = float(r2_score(y_val, y_pred))
pearson = float(np.corrcoef(y_val, y_pred)[0,1]) if y_pred.std()>1e-12 and y_val.std()>1e-12 else float("nan")

metrics = dict(rmse_std=rmse, mae_std=mae, r2_std=r2, pearson_std=pearson)
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# 保存模型与预测
model.save_model(str(OUT_DIR/"q_lgbm.txt"))
pd.DataFrame({"y_true_std":y_val, "y_pred_std":y_pred}).to_csv(OUT_DIR/"val_pred.csv", index=False)

# 保存特征重要度
imp = pd.DataFrame({"feature_id": np.arange(X.shape[1]),
                    "gain": model.feature_importance(importance_type="gain"),
                    "split": model.feature_importance(importance_type="split")})
imp.to_csv(OUT_DIR/"feature_importance.csv", index=False)
