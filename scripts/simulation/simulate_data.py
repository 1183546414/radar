"""
高对比度仿真数据：固定干扰状态 S，对每个状态采样 K 个雷达动作 A，计算 SINR。

配置: NUM_STATES=5000, ACTIONS_PER_STATE=20 → 100000 行。
"""
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from radar.data.high_contrast_sim import generate_scenario_dataset

NUM_STATES = 5000
ACTIONS_PER_STATE = 20
OUTPUT_FILE = ROOT / "data" / "simulated" / "simulated_radar_jamming_high_contrast_10k.csv"


if __name__ == "__main__":
    random.seed(42)
    import numpy as np

    np.random.seed(42)

    df = generate_scenario_dataset(
        NUM_STATES, ACTIONS_PER_STATE, scenario_id_start=0
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"数据集已生成: {OUTPUT_FILE}")
    print(f"干扰状态数 NUM_STATES={NUM_STATES}, 每状态动作数 K={ACTIONS_PER_STATE}")
    print(f"总行数: {len(df)} (期望 {NUM_STATES * ACTIONS_PER_STATE})")
    print(f"唯一 scenario_id: {df['scenario_id'].nunique()}")

    print("\n对比度分析 (SINR_dB):")
    print(f"  最高: {df['SINR_dB'].max()} dB")
    print(f"  最低: {df['SINR_dB'].min()} dB")
    print(f"  平均: {df['SINR_dB'].mean():.2f} dB")

    per_scenario = df.groupby("scenario_id")["SINR_dB"].agg(["min", "max", "std"])
    print(f"\n每状态 SINR 跨度 (max-min) 均值: {(per_scenario['max'] - per_scenario['min']).mean():.2f} dB")
