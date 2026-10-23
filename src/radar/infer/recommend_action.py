# -*- coding: utf-8 -*-
"""给定干扰状态 S，在该场景 K 个候选雷达动作上选预测 SINR 最高的 A。"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from radar.data.build_dataset import load_simulated_csv, row_to_state_vector
from radar.frozen_vae import encode_states, load_frozen_vae_encoder
from radar.paths import DATA_SIMULATED_HIGH_CONTRAST_CSV, DATA_V2_PROCESSED, OUT_Q_VALUE
from radar.train.train_q_latent import MODEL_PATH, QLatentNet, predict_q

ART_PATH = DATA_V2_PROCESSED / "artifacts.pkl"
A_STD_PATH = DATA_V2_PROCESSED / "A_std.npy"
R_STD_PATH = DATA_V2_PROCESSED / "R_std.npy"
A_RAW_PATH = DATA_V2_PROCESSED / "A_raw.npy"
R_RAW_PATH = DATA_V2_PROCESSED / "R_raw.npy"
SCENARIO_ID_PATH = DATA_V2_PROCESSED / "scenario_id.npy"


def _load_q_model(device: torch.device) -> tuple[QLatentNet, int, int]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"未找到 Q 网络权重，请先运行 main.py 第 3 步: {MODEL_PATH}"
        )
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = QLatentNet(int(ckpt["z_dim"]), int(ckpt["a_dim"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, int(ckpt["z_dim"]), int(ckpt["a_dim"])


def recommend_action(
    z: np.ndarray,
    a_candidates: np.ndarray,
    model: QLatentNet,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """返回 [(候选索引, 预测 Q), ...] 按 Q 降序。"""
    q = predict_q(model, z, a_candidates, device)
    k = min(top_k, len(q))
    order = np.argsort(-q)[:k]
    return [(int(i), float(q[i])) for i in order]


def recommend_action_main(query_row_idx: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = _load_q_model(device)
    vae, _, _ = load_frozen_vae_encoder(device)

    with open(ART_PATH, "rb") as f:
        artifacts = pickle.load(f)

    A_std = np.load(A_STD_PATH)
    A_raw = np.load(A_RAW_PATH)
    R_std = np.load(R_STD_PATH).reshape(-1)
    R_raw = np.load(R_RAW_PATH).reshape(-1)
    scenario_ids = np.load(SCENARIO_ID_PATH)

    csv_path = Path(
        artifacts.get("source_csv", DATA_SIMULATED_HIGH_CONTRAST_CSV)
    )
    df = load_simulated_csv(csv_path)
    query_row = df.iloc[query_row_idx]
    query_sid = int(scenario_ids[query_row_idx])

    s_vec = row_to_state_vector(query_row, artifacts).astype(np.float32)
    s_imp = artifacts["imp_S"].transform(s_vec.reshape(1, -1))
    s_std = artifacts["sc_S"].transform(s_imp).astype(np.float32)
    z = encode_states(vae, s_std, device)

    pool_idx = np.where(scenario_ids == query_sid)[0]
    a_cand_std = A_std[pool_idx]
    a_cand_raw = A_raw[pool_idx]
    r_cand_raw = R_raw[pool_idx]

    top = recommend_action(z, a_cand_std, model, device, top_k=5)
    q_all = predict_q(model, z, a_cand_std, device)

    oracle_local = int(np.argmax(r_cand_raw))
    pred_local = int(np.argmax(q_all))
    oracle_sinr = float(r_cand_raw[oracle_local])
    pred_sinr = float(r_cand_raw[pred_local])
    regret_db = oracle_sinr - pred_sinr

    print(f"\n=== 动作推荐演示 (CSV row={query_row_idx}, scenario_id={query_sid}) ===")
    print("干扰状态 (CSV):", query_row.filter(like="Jammer_").to_dict())
    print(f"候选动作数 K={len(pool_idx)}（同一场景下的雷达参数组合）")
    print(
        f"Oracle 最优 SINR={oracle_sinr:.4f} dB | "
        f"Q 推荐 SINR={pred_sinr:.4f} dB | regret={regret_db:.4f} dB"
    )
    print(f"Oracle Hit@1: {'是' if pred_local == oracle_local else '否'}")

    print("\n--- Top-5 推荐雷达动作（本场景内 Q(z,A) 最高）---")
    for rank, (ci, q_pred) in enumerate(top, start=1):
        raw = a_cand_raw[ci]
        sinr_db = float(r_cand_raw[ci])
        tag = ""
        if ci == pred_local:
            tag = " [Q推荐]"
        if ci == oracle_local:
            tag += " [Oracle最优]"
        print(
            f"[{rank}] action_idx={pool_idx[ci]}, pred_Q_std={q_pred:.4f}, "
            f"SINR_dB={sinr_db:.4f}{tag}, A_raw={raw.tolist()}"
        )


if __name__ == "__main__":
    recommend_action_main()
