# -*- coding: utf-8 -*-
"""
Load S, A, R from provided JSON files and apply median imputation + standardization.

Inputs (expected at the same directory as this script or adjust paths):
- Encoded_Jamming_Type.json
- Jammer_carrierFrequency.json
- Jamming_Params.json
- radar_trans.json
- SINR.json

Outputs (in ./processed_dataset):
- S_raw.npy, A_raw.npy, R_raw.npy
- S_std.npy, A_std.npy, R_std.npy
- artifacts.pkl  (sklearn imputers & scalers, plus column encoders for strings)
"""


import json, pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "data_2" / "json"

print(DATA_DIR)
OUT_DIR = DATA_DIR / "processed_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def one_hot_from_indices(indices: np.ndarray) -> np.ndarray:
    idx = indices.astype(int).ravel()
    K = int(idx.max()) + 1 if idx.size else 0
    out = np.zeros((idx.shape[0], K), dtype=float)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out

def encode_strings_columnwise(matrix: List[List[Any]]) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    maxw = max(len(r) if isinstance(r, (list, tuple)) else 1 for r in matrix)
    norm = []
    for r in matrix:
        if not isinstance(r, (list, tuple)):
            r = [r]
        rr = list(r) + [None] * (maxw - len(r))
        norm.append(rr[:maxw])
    encoders: List[Dict[str, int]] = [dict() for _ in range(maxw)]
    next_ids = [0] * maxw
    out = np.full((len(norm), maxw), np.nan, dtype=float)
    for i, row in enumerate(norm):
        for j, v in enumerate(row):
            if isinstance(v, (int, float, np.number)) or v is None:
                out[i, j] = np.nan if v is None else float(v)
            elif isinstance(v, str):
                if v not in encoders[j]:
                    encoders[j][v] = next_ids[j]
                    next_ids[j] += 1
                out[i, j] = float(encoders[j][v])
            else:
                try:
                    out[i, j] = float(v)
                except Exception:
                    s = str(v)
                    if s not in encoders[j]:
                        encoders[j][s] = next_ids[j]
                        next_ids[j] += 1
                    out[i, j] = float(encoders[j][s])
    return out, encoders

def dataset_main():
    encoded_jamming_type = read_json(DATA_DIR / "Encoded_Jamming_Type.json")
    jammer_carrier_freq = read_json(DATA_DIR / "Jammer_carrierFrequency.json")
    jamming_params      = read_json(DATA_DIR / "Jamming_Params.json")
    radar_trans         = read_json(DATA_DIR / "radar_trans.json")
    sinr_list           = read_json(DATA_DIR / "SINR.json")


    # Encoded_Jamming_Type: one-hot if scalar indices; keep if already vectors
    # --- Encoded_Jamming_Type ---
    ejt_arr = np.array(encoded_jamming_type, dtype=object).reshape(-1,)

    if ejt_arr.size > 0 and isinstance(ejt_arr[0], (list, tuple)):
        # 已经是向量（例如 embedding） -> 展开拼接
        max_len = max(len(v) for v in ejt_arr)
        ejt_vec = np.zeros((ejt_arr.shape[0], max_len), dtype=float)
        for i, v in enumerate(ejt_arr):
            vv = np.array(v, dtype=float).ravel()
            ejt_vec[i, :len(vv)] = vv
    else:
        # 是单个标量（int/float），直接 reshape 成列向量，不再 one-hot
        ejt_vec = np.array(ejt_arr, dtype=float).reshape(-1, 1)




    # Jammer_carrierFrequency
    if isinstance(jammer_carrier_freq, list) and len(jammer_carrier_freq) > 0 and isinstance(jammer_carrier_freq[0], list):
        jcf_arr = np.array(jammer_carrier_freq, dtype=float)
    else:
        jcf_arr = np.array(jammer_carrier_freq, dtype=float).reshape(-1, 1)

    # Jamming_Params
    jp_arr, jp_encoders = encode_strings_columnwise(jamming_params)

    # Align
    n_state = min(ejt_vec.shape[0], jcf_arr.shape[0], jp_arr.shape[0])
    ejt_vec = ejt_vec[:n_state]
    jcf_arr = jcf_arr[:n_state]
    jp_arr  = jp_arr[:n_state]
    S_raw = np.concatenate([ejt_vec, jcf_arr, jp_arr], axis=1)

    # Actions from radar_trans
    rt_matrix = [r if isinstance(r, list) else [r] for r in radar_trans]
    maxw_rt = max(len(r) for r in rt_matrix)
    rt_norm = [row + [None]*(maxw_rt - len(row)) for row in rt_matrix]
    carrier_col = [row[0] for row in rt_norm]
    params_cols = [row[1:] for row in rt_norm]
    carrier_numeric, _ = encode_strings_columnwise([[v] for v in carrier_col])
    params_numeric, radar_param_encoders = encode_strings_columnwise(params_cols)
    A_raw = np.concatenate([carrier_numeric, params_numeric], axis=1)

    n_total = min(S_raw.shape[0], A_raw.shape[0], len(sinr_list))
    S_raw = S_raw[:n_total]
    A_raw = A_raw[:n_total]
    R_raw = np.array(sinr_list, dtype=float).reshape(-1,1)[:n_total]

    # Impute + Standardize
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

    artifacts = {
        "imp_S": imp_S, "imp_A": imp_A, "imp_R": imp_R,
        "sc_S": sc_S, "sc_A": sc_A, "sc_R": sc_R,
        "jp_encoders": jp_encoders,
        "radar_param_encoders": radar_param_encoders,
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
    print(f"Saved to: {OUT_DIR}")
    print("ejt_vec.shape =", ejt_vec.shape)
    print("jcf_arr.shape =", jcf_arr.shape)
    print("jp_arr.shape  =", jp_arr.shape)
    print("S_raw.shape   =", S_raw.shape)

if __name__ == "__main__":
    dataset_main()
