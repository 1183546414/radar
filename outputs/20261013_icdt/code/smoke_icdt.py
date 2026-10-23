# -*- coding: utf-8 -*-
"""ICDT（In-Context Decision Transformer）数据可行性冒烟测试。

验证三件事（不碰仿真器，纯数据管线）：
  1. Round2 数据可加载：z[128] 潜向量 / A_std[6] / R_std / scenario_id / CSV 干扰类型
  2. 序列构造正确：观测 token(m 个) + 邻居经验 token(5) + 查询 z token + 候选动作 token
  3. 小 Transformer 能过拟合单批（信息能从 token 流到预测头 = 架构/维度/掩码无误）

序列布局（固定长度 15，类型嵌入区分 token 种类，无位置嵌入——集合无序）：
  [ obs×8槽(不足补零+mask) | nb×5 | query_z×1 | cand_a×1 ]
  统一 token 特征 136 维: [a(6); r(1); sim(1); z(128)] 缺失补零 + 类型 one-hot(4) = 140 维

成功标准：train loss 降到初始的 <5%，train 相关系数 > 0.95。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20261013_icdt/code/smoke_icdt.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROUND = HERE.parents[1]
for _p in HERE.parents:
    if (_p / "src" / "radar" / "paths.py").is_file():
        sys.path.insert(0, str(_p / "src"))
        break

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---- Round2 数据路径（与 20261007 轮相同套路：只读不改）----
ROUND2 = HERE.parents[3] / "outputs" / "20260909_signal_level"
ROUND2_PROCESSED = ROUND2 / "processed"
ROUND2_MODELS = ROUND2 / "models"
ROUND2_CSV = ROUND2 / "data" / "simulated_radar_jamming_high_contrast_10k_signal.csv"

# ---- 序列 / 模型常量 ----
A_DIM = 9          # 雷达动作维度（标准化；A_raw 9 列，含离散与连续分量）
Z_DIM = 128        # 干扰状态潜向量维度
N_OBS_SLOTS = 8    # 场景内观测槽位（m ∈ 0..8，不足补零）
K_NEIGHBORS = 5    # 邻居经验 token 数
TYPE_OBS, TYPE_NB, TYPE_QUERY, TYPE_CAND = 0, 1, 2, 3
N_TYPES = 4
FEAT_DIM = A_DIM + 1 + 1 + Z_DIM + N_TYPES  # 140
SEQ_LEN = N_OBS_SLOTS + K_NEIGHBORS + 2     # 15
D_MODEL, N_HEADS, N_LAYERS, FF_DIM = 128, 4, 4, 256

BATCH = 256
STEPS = 600
LR = 3e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    z = np.load(ROUND2_MODELS / "latent_q" / "state_latent_mu.npy").astype(np.float32)
    A_std = np.load(ROUND2_PROCESSED / "A_std.npy").astype(np.float32)
    R_std = np.load(ROUND2_PROCESSED / "R_std.npy").reshape(-1).astype(np.float32)
    scenario_ids = np.load(ROUND2_PROCESSED / "scenario_id.npy")
    assert z.shape[1] == Z_DIM, f"z 维度不符: {z.shape}"
    assert len(z) == len(A_std) == len(R_std) == len(scenario_ids)

    df = pd.read_csv(ROUND2_CSV, usecols=["scenario_id", "Jammer_Type"])
    df = df.drop_duplicates(subset=["scenario_id"])
    jammer_by_sid = dict(zip(df["scenario_id"].astype(int), df["Jammer_Type"].astype(str)))

    # 场景 → 行索引（5000 场景 × 每场景 20 动作；同场景共享同一干扰状态，z 取首行）
    sid_to_rows: dict[int, np.ndarray] = {}
    for sid in np.unique(scenario_ids):
        rows = np.where(scenario_ids == sid)[0]
        assert len(rows) == 20, f"场景 {sid} 行数 {len(rows)} != 20"
        sid_to_rows[int(sid)] = rows
    all_sids = np.array(sorted(sid_to_rows.keys()))
    return z, A_std, R_std, sid_to_rows, all_sids, jammer_by_sid


def build_neighbors(z, A_std, R_std, sid_to_rows, all_sids, jammer_by_sid, top_k=K_NEIGHBORS):
    """按干扰类型分组，组内 z 余弦 top-k 邻居（排除自身）。

    返回 nb_a/nb_r/nb_sim: [n_sid, top_k, ...]，邻居经验 = 邻居场景内 R_std 最高的动作。
    """
    z_sid = np.stack([z[sid_to_rows[int(s)][0]] for s in all_sids])  # [n_sid, 128]
    z_norm = z_sid / (np.linalg.norm(z_sid, axis=1, keepdims=True) + 1e-8)
    type_of = np.array([jammer_by_sid[int(s)] for s in all_sids])

    # 每场景的最优动作与 R（邻居经验的"这么打最有效"信息）
    best_a = np.stack([A_std[sid_to_rows[int(s)][int(np.argmax(R_std[sid_to_rows[int(s)]]))]]
                       for s in all_sids])  # [n_sid, 6]
    best_r = np.array([R_std[sid_to_rows[int(s)][int(np.argmax(R_std[sid_to_rows[int(s)]]))]]
                       for s in all_sids], dtype=np.float32)  # [n_sid]

    n = len(all_sids)
    nb_a = np.zeros((n, top_k, A_DIM), dtype=np.float32)
    nb_r = np.zeros((n, top_k), dtype=np.float32)
    nb_sim = np.zeros((n, top_k), dtype=np.float32)
    for t in np.unique(type_of):
        idx = np.where(type_of == t)[0]
        if len(idx) <= top_k + 1:
            continue
        sims = z_norm[idx] @ z_norm[idx].T  # 组内余弦 [g, g]
        for i, row_i in enumerate(idx):
            sims_i = sims[i].copy()
            sims_i[i] = -2.0  # 排除自身
            nb = idx[np.argsort(-sims_i)[:top_k]]
            nb_a[row_i] = best_a[nb]
            nb_r[row_i] = best_r[nb]
            nb_sim[row_i] = sims_i[np.argsort(-sims_i)[:top_k]]
    assert (nb_sim > -1).all(), "存在未填充的邻居（类型组过小）"
    return nb_a, nb_r, nb_sim


def make_token(a, r, sim, z_vec, ttype):
    """统一 token 特征 [a;r;sim;z] 缺失补零 + 类型 one-hot。"""
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    feat[:A_DIM] = a
    if r is not None:
        feat[A_DIM] = r
    if sim is not None:
        feat[A_DIM + 1] = sim
    if z_vec is not None:
        feat[A_DIM + 2: A_DIM + 2 + Z_DIM] = z_vec
    feat[A_DIM + 2 + Z_DIM + ttype] = 1.0
    return feat


def sample_batch(rng, z, A_std, R_std, sid_to_rows, all_sids,
                 nb_a, nb_r, nb_sim, batch=BATCH):
    """构造 ICDT 序列批（固定形状，obs 不足补零）。

    每样本：随机场景 → 随机 1 行做查询（候选+目标），其余 19 行采 m∈0..8 做观测。
    """
    B = batch
    toks = np.zeros((B, SEQ_LEN, FEAT_DIM), dtype=np.float32)
    pad = np.ones((B, SEQ_LEN), dtype=bool)  # True = padding
    target = np.zeros(B, dtype=np.float32)

    z_sid = {int(s): z[sid_to_rows[int(s)][0]] for s in all_sids}
    for b in range(B):
        sid = int(rng.choice(all_sids))
        rows = sid_to_rows[sid]
        q = int(rng.integers(0, 20))
        target[b] = R_std[rows[q]]
        obs_pool = np.array([r for r in rows if r != rows[q]])
        m = int(rng.integers(0, 9))
        obs_rows = rng.choice(obs_pool, size=m, replace=False) if m > 0 else []

        pos = 0
        for r_i in obs_rows:  # obs tokens（最多 8 槽）
            toks[b, pos] = make_token(A_std[r_i], R_std[r_i], None, None, TYPE_OBS)
            pad[b, pos] = False
            pos += 1
        pos = N_OBS_SLOTS
        for k in range(K_NEIGHBORS):  # nb tokens
            toks[b, pos + k] = make_token(nb_a[sid, k], nb_r[sid, k], nb_sim[sid, k],
                                          None, TYPE_NB)
            pad[b, pos + k] = False
        pos = N_OBS_SLOTS + K_NEIGHBORS
        toks[b, pos] = make_token(np.zeros(A_DIM), None, None, z_sid[sid], TYPE_QUERY)
        toks[b, pos + 1] = make_token(A_std[rows[q]], None, None, None, TYPE_CAND)
        pad[b, pos] = False
        pad[b, pos + 1] = False
        assert (~pad[b]).sum() >= 7  # 每样本至少 nb×5 + query + cand 共 7 个有效 token
    return (torch.from_numpy(toks).to(DEVICE),
            torch.from_numpy(pad).to(DEVICE),
            torch.from_numpy(target).to(DEVICE))


class ICDTSmoke(nn.Module):
    """冒烟用小 Transformer：bidirectional self-attention（集合推理，非自回归）。

    手动堆叠 encoder 层（不用 nn.TransformerEncoder 包装）：规避其 eval 模式
    融合内核（fast path）与 src_key_padding_mask 的 NaN 兼容问题。
    """

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(FEAT_DIM, D_MODEL)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                D_MODEL, N_HEADS, FF_DIM, dropout=0.1,
                batch_first=True, norm_first=True)
            for _ in range(N_LAYERS)
        ])
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, toks, pad_mask):
        h = self.in_proj(toks)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=pad_mask)
        return self.head(h[:, -1]).squeeze(-1)  # 候选 token 在序列末位


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    t0 = time.perf_counter()
    z, A_std, R_std, sid_to_rows, all_sids, jammer_by_sid = load_data()
    nb_a, nb_r, nb_sim = build_neighbors(z, A_std, R_std, sid_to_rows, all_sids, jammer_by_sid)
    t_load = time.perf_counter() - t0
    print(f"[数据] N={len(z)} 行, {len(all_sids)} 场景, "
          f"类型数={len(set(jammer_by_sid.values()))}, 加载+邻居 {t_load:.1f}s")
    print(f"[邻居] nb_a{nb_a.shape} nb_r{nb_r.shape} "
          f"sim均值={nb_sim.mean():.3f} 最小={nb_sim.min():.3f}")

    # 固定 train / val 批（过拟合单批验证：两批同分布不同采样）
    tr_toks, tr_pad, tr_y = sample_batch(rng, z, A_std, R_std, sid_to_rows,
                                         all_sids, nb_a, nb_r, nb_sim)
    va_toks, va_pad, va_y = sample_batch(rng, z, A_std, R_std, sid_to_rows,
                                         all_sids, nb_a, nb_r, nb_sim)
    print(f"[序列] toks{tuple(tr_toks.shape)} (SEQ_LEN={SEQ_LEN}, FEAT_DIM={FEAT_DIM}), "
          f"obs平均槽占用={float((~tr_pad[:, :N_OBS_SLOTS]).float().mean()):.2f}/8")

    model = ICDTSmoke().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"[模型] 参数量={n_params/1e3:.1f}K, device={DEVICE}")

    loss_fn = nn.MSELoss()
    first_loss = None
    history = []
    torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        model.train()
        pred = model(tr_toks, tr_pad)
        loss = loss_fn(pred, tr_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = float(loss.item())
        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vp = model(va_toks, va_pad)
                vloss = loss_fn(vp, va_y)
            pt, pv = float(loss.item()), float(vloss.item())
            history.append({"step": step, "train_mse": pt, "val_mse": pv})
            print(f"step {step:4d} | train {pt:.4f} | val {pv:.4f}")
    dt = time.perf_counter() - t0

    with torch.no_grad():
        model.eval()
        ptr = model(tr_toks, tr_pad).cpu().numpy()
        ytr = tr_y.cpu().numpy()
    corr = float(np.corrcoef(ptr, ytr)[0, 1])
    ratio = float(history[-1]["train_mse"]) / first_loss
    r2 = 1 - history[-1]["train_mse"] / float(ytr.var())
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6
               if DEVICE.type == "cuda" else float("nan"))

    ok = (ratio < 0.05) and (corr > 0.95)
    print("\n========== 冒烟结论 ==========")
    print(f"loss 收敛比   : {ratio:.4f}  (要求 < 0.05)")
    print(f"train 相关系数: {corr:.4f}  (要求 > 0.95)")
    print(f"train R^2     : {r2:.4f} | val MSE {history[-1]['val_mse']:.4f}")
    print(f"耗时 {dt:.1f}s ({dt/STEPS*1000:.1f} ms/步) | 显存峰值 {peak_mb:.0f} MB")
    print(f"结果: {'PASS ✅ 可以进入全量训练' if ok else 'FAIL ❌ 需排查'}")

    report = {
        "n_rows": int(len(z)), "n_scenarios": int(len(all_sids)),
        "seq_len": SEQ_LEN, "feat_dim": FEAT_DIM, "params_k": n_params / 1e3,
        "first_train_mse": first_loss, "final_train_mse": history[-1]["train_mse"],
        "loss_ratio": ratio, "train_corr": corr, "train_r2": r2,
        "val_mse": history[-1]["val_mse"], "steps": STEPS, "batch": BATCH,
        "elapsed_s": dt, "ms_per_step": dt / STEPS * 1000, "peak_mem_mb": peak_mb,
        "pass": bool(ok), "history": history,
    }
    out = ROUND / "smoke_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入: {out}")


if __name__ == "__main__":
    main()
