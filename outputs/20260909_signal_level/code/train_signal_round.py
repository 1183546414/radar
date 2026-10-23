# -*- coding: utf-8 -*-
"""Round 20260909 路线 A 第 2 步：用信号级重标注数据重建 processed 并重训 Q 网络。

三阶段（全部产物写入本轮文件夹，原训练代码与原产物零改动）：
  1. build_dataset.dataset_main(relabeled_csv) -> processed/（S/A 特征与原版一致，R 换信号级）
  2. train_q_latent_main()   -> models/latent_q/          （q_only / retrieve_q* 用）
  3. train_q_retrieval_main()-> models/latent_q_retrieval/（retrieve_q_ctx* 用）

关键 patch（运行期，原模块常量不动）：
  * retrieval_ranking.DATA_V2_PROCESSED -> 本轮 processed
    （原始 artifacts.pkl 的 source_csv 是失效的 Mac 路径，不 patch 会 FileNotFoundError）
  * retrieval_ranking.sinr_from_csv_row -> SignalSinrBackend.sinr（JSR=10dB）
    使 ranking loss 与早停指标（val_rank_nll / 检索池 Hit@1）的 oracle 与重标注标签同源。

VAE 与状态嵌入不重训（S 未变，z 嵌入仍然有效，直接复用原 outputs/state_vae）。

用法:
    D:\\conda\\envs\\radar\\python.exe outputs/20260909_signal_level/code/train_signal_round.py
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
sys.path.insert(0, str(HERE.parent))  # 便于 import signal_sinr

ROUND_PROCESSED = ROUND / "processed"
ROUND_MODELS = ROUND / "models"
RELABEL_CSV = ROUND / "data" / "simulated_radar_jamming_high_contrast_10k_signal.csv"
RUN_REPORT = ROUND / "train_report.json"
JSR_DB = 10.0

_BACKEND = None


def _signal_truth_fn():
    """信号级真值函数（单例后端，跨阶段共享缓存）。"""
    global _BACKEND
    if _BACKEND is None:
        from signal_sinr import SignalSinrBackend

        _BACKEND = SignalSinrBackend(jsr_db=JSR_DB)
    return _BACKEND.sinr


report: dict = {"stages": {}}


def _stage(name: str, fn) -> dict:
    t0 = time.time()
    print(f"\n{'=' * 60}\n=== 阶段: {name}\n{'=' * 60}", flush=True)
    fn()
    dt = round((time.time() - t0) / 60, 2)
    report["stages"][name] = {"elapsed_min": dt}
    print(f"\n=== {name} 完成，耗时 {dt} 分钟 ===", flush=True)
    return report["stages"][name]


def stage_build_processed() -> None:
    from radar.data import build_dataset as bd

    assert RELABEL_CSV.is_file(), f"重标注 CSV 不存在: {RELABEL_CSV}"
    bd.OUT_DIR = ROUND_PROCESSED  # 原版 DATA_V2_PROCESSED 不动
    bd.dataset_main(csv_path=RELABEL_CSV)


def stage_train_q_latent() -> None:
    from radar.train import retrieval_ranking as rr
    from radar.train import train_q_latent as tql

    rr.DATA_V2_PROCESSED = ROUND_PROCESSED  # 否则 build 读原始 artifacts（失效 source_csv）
    rr.sinr_from_csv_row = _signal_truth_fn()  # ranking/早停 oracle 与重标注同源
    tql.DATA_V2_PROCESSED = ROUND_PROCESSED
    tql.OUT_DIR = ROUND_MODELS / "latent_q"
    tql.LATENT_CACHE = tql.OUT_DIR / "state_latent_mu.npy"
    tql.MODEL_PATH = tql.OUT_DIR / "q_latent_best.pt"
    tql.METRICS_PATH = tql.OUT_DIR / "metrics.json"
    tql.SCENARIO_ID_PATH = ROUND_PROCESSED / "scenario_id.npy"
    tql.R_RAW_PATH = ROUND_PROCESSED / "R_raw.npy"
    tql.train_q_latent_main()


def stage_train_q_retrieval() -> None:
    from radar.train import retrieval_ranking as rr
    from radar.train import train_q_retrieval as tqr

    rr.DATA_V2_PROCESSED = ROUND_PROCESSED
    rr.sinr_from_csv_row = _signal_truth_fn()
    tqr.DATA_V2_PROCESSED = ROUND_PROCESSED
    tqr.RETRIEVAL_OUT_DIR = ROUND_MODELS / "latent_q_retrieval"
    tqr.RETRIEVAL_MODEL_PATH = tqr.RETRIEVAL_OUT_DIR / "q_retrieval_best.pt"
    tqr.RETRIEVAL_METRICS_PATH = tqr.RETRIEVAL_OUT_DIR / "metrics.json"
    tqr.Z_CTX_CACHE = tqr.RETRIEVAL_OUT_DIR / "z_ctx_per_row_same_type_weighted.npy"
    tqr.Q_LATENT_MODEL_PATH = ROUND_MODELS / "latent_q" / "q_latent_best.pt"
    tqr.SCENARIO_ID_PATH = ROUND_PROCESSED / "scenario_id.npy"
    tqr.train_q_retrieval_main()


def main() -> None:
    t0 = time.time()
    assert ROUND_PROCESSED.is_dir() or True
    _stage("build_processed(信号级 R)", stage_build_processed)
    _stage("train_q_latent", stage_train_q_latent)
    _stage("train_q_retrieval", stage_train_q_retrieval)
    report["total_min"] = round((time.time() - t0) / 60, 2)
    report["round_processed"] = str(ROUND_PROCESSED)
    report["round_models"] = str(ROUND_MODELS)
    report["jsr_db"] = JSR_DB
    report["ranking_oracle"] = "radarsimpy_signal_level"
    if _BACKEND is not None:
        report["backend_stats"] = _BACKEND.stats()
    with open(RUN_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，报告: {RUN_REPORT}")


if __name__ == "__main__":
    main()
