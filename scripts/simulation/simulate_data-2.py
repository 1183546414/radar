

import pandas as pd
import numpy as np
import random
import time

# ==========================================
# 1. 全局配置
# ==========================================
NUM_SAMPLES = 10000
OUTPUT_FILE = "data/simulated/simulated_radar_jamming_flat_GHz_10k.csv"

# --- 雷达参数范围 ---
RADAR_CONFIG = {
    # 频率: 3.0GHz - 5.0GHz
    'freq_min': 3.0e9, 
    'freq_max': 5.0e9,
    # 增益: 20-40 dB (任意整数)
    'gain_min': 20,
    'gain_max': 40,
    # 带宽: 5MHz - 25MHz (内部计算用Hz，输出转GHz)
    'bw_min': 5e6, 
    'bw_max': 25e6,
    # 脉宽: 20us - 40us (20000ns - 40000ns)
    'pw_min': 20000, 
    'pw_max': 40000,
    
    'radar_types': ['线性调频0', '非线性调频1', '相位编码3', '频率编码2'],
    'chirp_type': 'NegativeTriangle'
}

# --- 干扰机参数范围 ---
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
    # 增益: 20-40 dB
    'gain_min': 20, 
    'gain_max': 40
}

# ==========================================
# 2. 辅助函数
# ==========================================

def get_random_100_aligned(min_val, max_val):
    """生成能被100整除的随机整数 [min_val, max_val]"""
    # 比如 20000 到 40000
    start = int(min_val // 100)
    end = int(max_val // 100)
    return random.randint(start, end) * 100

def hz_to_ghz(hz_value):
    """Hz 转 GHz 并保留3位小数"""
    if hz_value == 0: return 0.0
    val_ghz = hz_value / 1e9
    return round(val_ghz, 3)

def random_freq_hz(min_f, max_f):
    """生成内部计算用的 Hz 频率 (10MHz步进)"""
    steps = int((max_f - min_f) / 10e6)
    return min_f + random.randint(0, steps) * 10e6

# ==========================================
# 3. 核心生成逻辑
# ==========================================

def calculate_sinr_internal(r_freq_hz, r_bw_hz, r_gain, j_freq_hz, j_bw_hz, j_gain, j_type):
    """
    内部使用 Hz 进行物理计算，确保准确性。
    """
    # 干扰带宽默认为 10MHz 防止除零
    real_j_bw = j_bw_hz if j_bw_hz > 0 else 10e6
    
    # 1. 频率碰撞检测
    freq_dist = abs(r_freq_hz - j_freq_hz)
    collision_radius = (r_bw_hz / 2) + (real_j_bw / 2)
    
    base_sinr = 25.0
    
    if freq_dist < collision_radius:
        # --- 发生碰撞 ---
        overlap_factor = 1.0 - (freq_dist / collision_radius)
        
        # 惩罚项
        power_penalty = (j_gain * 1.2) * overlap_factor
        
        # 增益项 (脉压增益 ~ log带宽)
        processing_gain = np.log10(r_bw_hz) * 3.0
        
        final_sinr = base_sinr - power_penalty + processing_gain + (r_gain * 0.5)
        
        # 类型特定修正
        if j_type == 'jammer_swept': final_sinr -= 5.0
        if j_type == 'jammer_close': final_sinr -= 10.0
        
    else:
        # --- 未碰撞 ---
        final_sinr = base_sinr - (j_gain * 0.1) + np.random.normal(0, 1.0)
        
    return round(final_sinr, 4)


def generate_sample():
    # -------------------------------------------------
    # A. 生成雷达数据 (先用 Hz/ns 内部生成，最后转换)
    # -------------------------------------------------
    r_freq_hz = random_freq_hz(RADAR_CONFIG['freq_min'], RADAR_CONFIG['freq_max'])
    r_bw_hz = int(random.uniform(RADAR_CONFIG['bw_min'], RADAR_CONFIG['bw_max']))
    r_pw_ns = get_random_100_aligned(RADAR_CONFIG['pw_min'], RADAR_CONFIG['pw_max']) # 被100整除
    r_gain = random.randint(RADAR_CONFIG['gain_min'], RADAR_CONFIG['gain_max'])      # 20-40 整数
    
    radar_raw = {
        'Radar_Type': random.choice(RADAR_CONFIG['radar_types']),
        'Radar_ChirpType': RADAR_CONFIG['chirp_type'],
        'Radar_CarrierFreq_Hz': r_freq_hz,
        'Radar_Gain_dB': r_gain,
        'Radar_Bandwidth_Hz': r_bw_hz,
        'Radar_PulseWidth_ns': r_pw_ns,
    }

    # -------------------------------------------------
    # B. 生成干扰数据 (先用 Hz/ns 内部生成)
    # -------------------------------------------------
    j_type = random.choice(JAMMER_CONFIG['types'])
    j_freq_hz = random_freq_hz(JAMMER_CONFIG['freq_min'], JAMMER_CONFIG['freq_max'])
    j_gain = random.randint(JAMMER_CONFIG['gain_min'], JAMMER_CONFIG['gain_max']) # 20-40 整数
    
    # 初始化特定参数为 0
    j_params_raw = {
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

    # 根据类型填充
    if j_type == 'jammer_swept':
        j_params_raw['Jammer_Bandwidth_Hz'] = int(random.choice([100e6, 200e6, 300e6, 400e6]))
        j_params_raw['Jammer_Sweep_Step_Hz'] = int(random.choice([5e6, 10e6, 20e6]))
        j_params_raw['Jammer_Sweep_Power'] = int(random.uniform(20, 50))

    elif j_type == 'jammer_wide_band':
        j_params_raw['Jammer_Bandwidth_Hz'] = int(random.uniform(100e6, 500e6))

    elif j_type == 'jammer_narrow_band':
        j_params_raw['Jammer_Bandwidth_Hz'] = int(random.uniform(5e6, 20e6))

    elif j_type == 'jammer_InterruptedSampling':
        # 这里给一个物理带宽用于计算SINR，尽管间歇采样主要靠时域参数
        j_params_raw['Jammer_Bandwidth_Hz'] = 20e6 
        # 时间参数：必须被100整除
        dur = get_random_100_aligned(1000, 50000)
        delay = get_random_100_aligned(1000, 50000)
        # 间隔 > 持续+延迟，且也要整除
        min_intv = dur + delay + 100
        intv = get_random_100_aligned(min_intv, min_intv + 50000)
        
        j_params_raw['Jammer_IntSample_Dur_ns'] = dur
        j_params_raw['Jammer_IntSample_Delay_ns'] = delay
        j_params_raw['Jammer_IntSample_Intv_ns'] = intv

    elif j_type == 'jammer_multi_false_targets':
        j_params_raw['Jammer_Bandwidth_Hz'] = 20e6
        j_params_raw['Jammer_FalseTarget_Num'] = random.randint(3, 12)
        j_params_raw['Jammer_FalseTarget_Intv_m'] = int(random.uniform(500, 3000))
        j_params_raw['Jammer_FalseTarget_Delay_m'] = 33000

    elif j_type == 'jammer_close':
        j_gain = 35 # 强制高增益（也在20-40范围内）
        j_params_raw['Jammer_Bandwidth_Hz'] = 50e6

    # -------------------------------------------------
    # C. 计算 SINR (用 Hz)
    # -------------------------------------------------
    sinr = calculate_sinr_internal(
        r_freq_hz, r_bw_hz, r_gain,
        j_freq_hz, j_params_raw['Jammer_Bandwidth_Hz'], j_gain,
        j_type
    )

    # -------------------------------------------------
    # D. 格式转换与输出 (Hz -> GHz, 保留小数)
    # -------------------------------------------------
    output_row = {
        # --- Jammer State ---
        'Jammer_Type': j_type,
        'Jammer_CarrierFreq_GHz': hz_to_ghz(j_freq_hz),
        'Jammer_Gain_dB': j_gain,
        'Jammer_Bandwidth_GHz': hz_to_ghz(j_params_raw['Jammer_Bandwidth_Hz']),
        
        # 扫频
        'Jammer_Sweep_Step_GHz': hz_to_ghz(j_params_raw['Jammer_Sweep_Step_Hz']),
        'Jammer_Sweep_Power': j_params_raw['Jammer_Sweep_Power'],
        
        # 间歇采样 (ns, 整数)
        'Jammer_IntSample_Dur_ns': j_params_raw['Jammer_IntSample_Dur_ns'],
        'Jammer_IntSample_Delay_ns': j_params_raw['Jammer_IntSample_Delay_ns'],
        'Jammer_IntSample_Intv_ns': j_params_raw['Jammer_IntSample_Intv_ns'],
        
        # 假目标
        'Jammer_FalseTarget_Num': j_params_raw['Jammer_FalseTarget_Num'],
        'Jammer_FalseTarget_Intv_m': j_params_raw['Jammer_FalseTarget_Intv_m'],
        'Jammer_FalseTarget_Delay_m': j_params_raw['Jammer_FalseTarget_Delay_m'],
        
        # --- Radar Action ---
        'Radar_CarrierFreq_GHz': hz_to_ghz(r_freq_hz),
        'Radar_Gain_dB': r_gain,
        'Radar_Bandwidth_GHz': hz_to_ghz(r_bw_hz),
        'Radar_PulseWidth_ns': r_pw_ns,
        'Radar_Type': radar_raw['Radar_Type'],
        'Radar_ChirpType': radar_raw['Radar_ChirpType'],
        
        # --- Reward ---
        'SINR_dB': sinr
    }
    
    return output_row

# ==========================================
# 4. 主程序
# ==========================================

print(f"正在生成 {NUM_SAMPLES} 条数据 (单位已更新为 GHz, 精度3位)...")
start_time = time.time()

data_list = []
for i in range(NUM_SAMPLES):
    data_list.append(generate_sample())

df = pd.DataFrame(data_list)

# 调整列顺序
cols_order = [
    'Jammer_Type', 
    'Jammer_CarrierFreq_GHz', 'Jammer_Gain_dB', 'Jammer_Bandwidth_GHz',
    'Radar_CarrierFreq_GHz', 'Radar_Gain_dB', 'Radar_Bandwidth_GHz', 'Radar_PulseWidth_ns', 'Radar_Type',
    'SINR_dB',
    'Jammer_Sweep_Step_GHz', 'Jammer_Sweep_Power',
    'Jammer_IntSample_Dur_ns', 'Jammer_IntSample_Delay_ns', 'Jammer_IntSample_Intv_ns',
    'Jammer_FalseTarget_Num', 'Jammer_FalseTarget_Intv_m', 'Jammer_FalseTarget_Delay_m',
    'Radar_ChirpType'
]

# 保证列存在
final_cols = [c for c in cols_order if c in df.columns]
df = df[final_cols]

# 保存
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"完成! 耗时 {time.time()-start_time:.2f}s")
print(f"文件已保存至: {OUTPUT_FILE}")
print("\n数据预览 (前5行):")
print(df.head()) # 这里去掉了 to_markdown 避免报错

print("\n整除检查 (示例):")
print(f"PulseWidth % 100 == 0? {all(df['Radar_PulseWidth_ns'] % 100 == 0)}")
print(f"IntSample_Dur % 100 == 0? {all(df[df['Jammer_IntSample_Dur_ns']>0]['Jammer_IntSample_Dur_ns'] % 100 == 0)}")