# -*- coding: utf-8 -*-
"""
从模拟 CSV 构建 (S, A, R) 并做中位数填补 + 标准化。

主数据源:
  data/simulated/simulated_radar_jamming_high_contrast_10k.csv

输出 (data/v2/processed/):
  S_raw.npy, A_raw.npy, R_raw.npy
  S_std.npy, A_std.npy, R_std.npy
  artifacts.pkl
  jammer_type_labels.json
  scenario_id.npy          # 每行所属干扰场景（固定 S 下 K 个 A）
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV, DATA_V2_PROCESSED

OUT_DIR = DATA_V2_PROCESSED

# 状态 S：干扰机侧（与 scripts/simulation/train_vae.py 列划分一致，适配 high_contrast CSV）
S_CAT_COLS = ["Jammer_Type"]
S_NUM_COLS = [
    "Jammer_CarrierFreq_GHz",
    "Jammer_Gain_dB",
    "Jammer_Bandwidth_GHz",
    "Jammer_Sweep_Step_GHz",
    "Jammer_IntSample_Dur_ns",
    "Jammer_IntSample_Delay_ns",
    "Jammer_IntSample_Intv_ns",
    "Jammer_FalseTarget_Num",
]

# 动作 A：雷达侧
A_CAT_COLS = ["Radar_Type", "Radar_ChirpType"]
A_NUM_COLS = [
    "Radar_CarrierFreq_GHz",
    "Radar_Gain_dB",
    "Radar_Bandwidth_GHz",
    "Radar_PulseWidth_ns",
]

R_COL = "SINR_dB"
SCENARIO_COL = "scenario_id"
ACTION_IDX_COL = "action_idx"


def _one_hot(
    df: pd.DataFrame, cat_cols: Sequence[str], encoder: Optional[OneHotEncoder] = None
) -> Tuple[np.ndarray, OneHotEncoder]:
    data = df[list(cat_cols)].astype(str)
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(data)
    else:
        encoded = encoder.transform(data)
    return encoded.astype(np.float64), encoder


def _numeric_matrix(df: pd.DataFrame, num_cols: Sequence[str]) -> np.ndarray:
    return df[list(num_cols)].astype(np.float64).to_numpy()


def build_feature_matrix(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    encoder: Optional[OneHotEncoder] = None,
) -> Tuple[np.ndarray, OneHotEncoder]:
    cat_part, encoder = _one_hot(df, cat_cols, encoder)
    num_part = _numeric_matrix(df, num_cols)
    return np.hstack([cat_part, num_part]), encoder


def row_to_state_vector(row: pd.Series, artifacts: Dict[str, Any]) -> np.ndarray:
    """将单行 CSV（Series）转为与 S_raw 同维度的特征向量。"""
    df = row.to_frame().T
    vec, _ = build_feature_matrix(
        df,
        artifacts["s_cat_cols"],
        artifacts["s_num_cols"],
        encoder=artifacts["enc_S"],
    )
    return vec[0]


def row_to_action_vector(row: pd.Series, artifacts: Dict[str, Any]) -> np.ndarray:
    """将单行 CSV 中的雷达侧字段转为与 A_raw 同维度的特征向量。"""
    df = row.to_frame().T
    vec, _ = build_feature_matrix(
        df,
        artifacts["a_cat_cols"],
        artifacts["a_num_cols"],
        encoder=artifacts["enc_A"],
    )
    return vec[0]


def action_vectors_to_std(raw_vecs: np.ndarray, artifacts: Dict[str, Any]) -> np.ndarray:
    """A_raw 批量 -> A_std（与 build_dataset 一致）。"""
    imp = artifacts["imp_A"].transform(raw_vecs)
    return artifacts["sc_A"].transform(imp).astype(np.float32)


def row_to_action_std(row: pd.Series, artifacts: Dict[str, Any]) -> np.ndarray:
    """单行雷达参数 -> 标准化动作向量。"""
    return action_vectors_to_std(
        row_to_action_vector(row, artifacts).reshape(1, -1), artifacts
    )[0]


def load_simulated_csv(csv_path: Path = DATA_SIMULATED_HIGH_CONTRAST_CSV) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"模拟数据 CSV 不存在: {csv_path}")
    df = None
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        df = pd.read_csv(csv_path, encoding="latin-1")
    required = set(S_CAT_COLS + S_NUM_COLS + A_CAT_COLS + A_NUM_COLS + [R_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {sorted(missing)}")
    if SCENARIO_COL not in df.columns:
        raise ValueError(
            f"CSV 缺少 {SCENARIO_COL}，请用 scripts/simulation/simulate_data.py "
            f"(NUM_STATES×ACTIONS_PER_STATE) 重新生成数据"
        )
    return df


def dataset_main(csv_path: Path = DATA_SIMULATED_HIGH_CONTRAST_CSV) -> None:
    print("数据源 CSV:", csv_path)
    print("输出目录:", OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_simulated_csv(csv_path)
    n = len(df)
    print(f"读取样本数: {n}")

    S_raw, enc_S = build_feature_matrix(df, S_CAT_COLS, S_NUM_COLS)
    A_raw, enc_A = build_feature_matrix(df, A_CAT_COLS, A_NUM_COLS)
    R_raw = df[[R_COL]].astype(np.float64).to_numpy()

    imp_S = SimpleImputer(strategy="median")
    imp_A = SimpleImputer(strategy="median")
    imp_R = SimpleImputer(strategy="median")

    S_imp = imp_S.fit_transform(S_raw)
    A_imp = imp_A.fit_transform(A_raw)
    R_imp = imp_R.fit_transform(R_raw)

    sc_S = StandardScaler()
    sc_A = StandardScaler()
    sc_R = StandardScaler()

    S_std = sc_S.fit_transform(S_imp)
    A_std = sc_A.fit_transform(A_imp)
    R_std = sc_R.fit_transform(R_imp)

    jammer_labels = df["Jammer_Type"].astype(str).tolist()
    with open(OUT_DIR / "jammer_type_labels.json", "w", encoding="utf-8") as f:
        json.dump(jammer_labels, f, ensure_ascii=False)

    scenario_id = df[SCENARIO_COL].astype(np.int64).to_numpy()
    np.save(OUT_DIR / "scenario_id.npy", scenario_id)
    n_scenarios = int(df[SCENARIO_COL].nunique())
    actions_per = int(df.groupby(SCENARIO_COL).size().mode().iloc[0])

    artifacts: Dict[str, Any] = {
        "imp_S": imp_S,
        "imp_A": imp_A,
        "imp_R": imp_R,
        "sc_S": sc_S,
        "sc_A": sc_A,
        "sc_R": sc_R,
        "enc_S": enc_S,
        "enc_A": enc_A,
        "s_cat_cols": list(S_CAT_COLS),
        "s_num_cols": list(S_NUM_COLS),
        "a_cat_cols": list(A_CAT_COLS),
        "a_num_cols": list(A_NUM_COLS),
        "source_csv": str(csv_path.resolve()),
        "n_scenarios": n_scenarios,
        "actions_per_scenario": actions_per,
        "scenario_col": SCENARIO_COL,
    }

    np.save(OUT_DIR / "S_raw.npy", S_raw)
    np.save(OUT_DIR / "A_raw.npy", A_raw)
    np.save(OUT_DIR / "R_raw.npy", R_raw)
    np.save(OUT_DIR / "S_std.npy", S_std)
    np.save(OUT_DIR / "A_std.npy", A_std)
    np.save(OUT_DIR / "R_std.npy", R_std)

    with open(OUT_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Done. Shapes -> S_raw:{S_raw.shape}, A_raw:{A_raw.shape}, R_raw:{R_raw.shape}")
    print(f"Standardized -> S_std:{S_std.shape}, A_std:{A_std.shape}, R_std:{R_std.shape}")
    print(f"干扰类型数: {df['Jammer_Type'].nunique()}, 已保存 jammer_type_labels.json")
    print(f"干扰场景数: {n_scenarios}, 每场景动作数: {actions_per}, 已保存 scenario_id.npy")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    dataset_main()
