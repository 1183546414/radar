import pandas as pd
import numpy as np
import random
import time
from pathlib import Path

# ==========================================
# 1. 全局配置
# ==========================================
NUM_SAMPLES = 200
OUTPUT_DIR = Path("radar-10.11/radar/dataset/data_4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "simulated_jammer_only_10k.csv"

JAMMER_CONFIG = {
    'types': [
        'jammer_swept',                 # 扫频
        'jammer_multi_false_targets',   # 多假目标
        'jammer_InterruptedSampling',   # 间歇采样
        'jammer_narrow_band',           # 窄带噪声
        'jammer_wide_band',             # 宽带阻塞
        'jammer_close',                 # 近距离压制
        'jammer_RANDOM'                 # 随机
    ],
    'freq_min': 3.0e9, 
    'freq_max': 5.0e9,
    'gain_min': 20, 
    'gain_max': 40
}

# ==========================================
# 2. 辅助函数
# ==========================================

def get_random_100_aligned(min_val, max_val):
    """生成能被100整除的随机整数 (用于纳秒参数对齐)"""
    start = int(min_val // 100)
    end = int(max_val // 100)
    return random.randint(start, end) * 100

def hz_to_ghz(hz_value):
    """Hz 转 GHz 并保留3位小数"""
    return round(hz_value / 1e9, 3) if hz_value > 0 else 0.0

def random_freq_hz(min_f, max_f):
    """生成内部计算用的 Hz 频率 (10MHz步进)"""
    steps = int((max_f - min_f) / 10e6)
    return min_f + random.randint(0, steps) * 10e6

# ==========================================
# 3. 核心生成逻辑
# ==========================================

def generate_jammer_sample():
    # 基础参数
    j_type = random.choice(JAMMER_CONFIG['types'])
    j_freq_hz = random_freq_hz(JAMMER_CONFIG['freq_min'], JAMMER_CONFIG['freq_max'])
    j_gain = random.randint(JAMMER_CONFIG['gain_min'], JAMMER_CONFIG['gain_max'])
    
    # 初始化特定参数（默认值为 0）
    j_params = {
        'Jammer_Bandwidth_Hz': 0,
        'Jammer_Sweep_Step_Hz': 0,
        'Jammer_Sweep_Power': 0,
        'Jammer_IntSample_Dur_ns': 0,
        'Jammer_IntSample_Delay_ns': 0,
        'Jammer_IntSample_Intv_ns': 0,
        'Jammer_FalseTarget_Num': 0,
        'Jammer_FalseTarget_Intv_m': 0,
        'Jammer_FalseTarget_Delay_m': 0
    }

    # 根据干扰类型填充具体参数
    if j_type == 'jammer_swept':
        j_params['Jammer_Bandwidth_Hz'] = int(random.choice([100e6, 200e6, 400e6]))
        j_params['Jammer_Sweep_Step_Hz'] = int(random.choice([5e6, 10e6, 20e6]))
        j_params['Jammer_Sweep_Power'] = int(random.uniform(20, 50))

    elif j_type == 'jammer_wide_band':
        j_params['Jammer_Bandwidth_Hz'] = int(random.uniform(100e6, 500e6))

    elif j_type == 'jammer_narrow_band':
        j_params['Jammer_Bandwidth_Hz'] = int(random.uniform(5e6, 20e6))

    elif j_type == 'jammer_InterruptedSampling':
        dur = get_random_100_aligned(1000, 50000)
        delay = get_random_100_aligned(1000, 50000)
        min_intv = dur + delay + 100
        j_params['Jammer_IntSample_Dur_ns'] = dur
        j_params['Jammer_IntSample_Delay_ns'] = delay
        j_params['Jammer_IntSample_Intv_ns'] = get_random_100_aligned(min_intv, min_intv + 50000)
        j_params['Jammer_Bandwidth_Hz'] = 20e6 

    elif j_type == 'jammer_multi_false_targets':
        j_params['Jammer_FalseTarget_Num'] = random.randint(3, 12)
        j_params['Jammer_FalseTarget_Intv_m'] = int(random.uniform(500, 3000))
        j_params['Jammer_FalseTarget_Delay_m'] = 33000
        j_params['Jammer_Bandwidth_Hz'] = 20e6

    elif j_type == 'jammer_close':
        j_gain = 40 # 强压制
        j_params['Jammer_Bandwidth_Hz'] = 50e6

    # 封装输出行
    return {
        'Jammer_Type': j_type,
        'Jammer_CarrierFreq_GHz': hz_to_ghz(j_freq_hz),
        'Jammer_Gain_dB': j_gain,
        'Jammer_Bandwidth_GHz': hz_to_ghz(j_params['Jammer_Bandwidth_Hz']),
        'Jammer_Sweep_Step_GHz': hz_to_ghz(j_params['Jammer_Sweep_Step_Hz']),
        'Jammer_Sweep_Power': j_params['Jammer_Sweep_Power'],
        'Jammer_IntSample_Dur_ns': j_params['Jammer_IntSample_Dur_ns'],
        'Jammer_IntSample_Delay_ns': j_params['Jammer_IntSample_Delay_ns'],
        'Jammer_IntSample_Intv_ns': j_params['Jammer_IntSample_Intv_ns'],
        'Jammer_FalseTarget_Num': j_params['Jammer_FalseTarget_Num'],
        'Jammer_FalseTarget_Intv_m': j_params['Jammer_FalseTarget_Intv_m'],
        'Jammer_FalseTarget_Delay_m': j_params['Jammer_FalseTarget_Delay_m']
    }

# ==========================================
# 4. 执行生成
# ==========================================

print(f"开始生成干扰机参数数据...")
start_t = time.time()

data = [generate_jammer_sample() for _ in range(NUM_SAMPLES)]
df = pd.DataFrame(data)

# 强制列顺序（干扰机专用布局）
cols = [
    'Jammer_Type', 'Jammer_CarrierFreq_GHz', 'Jammer_Gain_dB', 'Jammer_Bandwidth_GHz',
    'Jammer_Sweep_Step_GHz', 'Jammer_Sweep_Power',
    'Jammer_IntSample_Dur_ns', 'Jammer_IntSample_Delay_ns', 'Jammer_IntSample_Intv_ns',
    'Jammer_FalseTarget_Num', 'Jammer_FalseTarget_Intv_m', 'Jammer_FalseTarget_Delay_m'
]
df = df[cols]

df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"生成完毕！耗时: {time.time()-start_t:.2f}s")
print(f"保存路径: {OUTPUT_FILE}")
print("\n数据预览:")
print(df.head())