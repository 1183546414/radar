import pandas as pd
import numpy as np
import random
import time

# ==========================================
# 1. 配置
# ==========================================
NUM_SAMPLES = 10000
OUTPUT_FILE = "./radar-10.11/radar/dataset/data_4/simulated_radar_jamming_high_contrast_10k.csv"

RADAR_CONFIG = {
    'freq_min': 3.0e9, 'freq_max': 5.0e9,
    'gain_min': 20, 'gain_max': 40,
    'bw_min': 5e6, 'bw_max': 25e6,
    'pw_min': 20000, 'pw_max': 40000,
    'radar_types': ['线性调频0', '非线性调频1', '相位编码3', '频率编码2'],
    'chirp_type': 'NegativeTriangle'
}

JAMMER_CONFIG = {
    'types': ['jammer_swept', 'jammer_multi_false_targets', 'jammer_InterruptedSampling', 
              'jammer_narrow_band', 'jammer_wide_band', 'jammer_close', 'jammer_RANDOM'],
    'freq_min': 3.0e9, 'freq_max': 5.0e9,
    'gain_min': 20, 'gain_max': 40
}

# ==========================================
# 2. 辅助工具
# ==========================================
def get_random_100_aligned(min_val, max_val):
    return random.randint(int(min_val // 100), int(max_val // 100)) * 100

def hz_to_ghz(hz_value):
    return round(hz_value / 1e9, 3) if hz_value > 0 else 0.0

def random_freq_hz(min_f, max_f):
    steps = int((max_f - min_f) / 10e6)
    return min_f + random.randint(0, steps) * 10e6

# ==========================================
# 3. 高对比度 SINR 计算公式 (物理启发)
# ==========================================
def calculate_sinr_high_contrast(r_freq, r_bw, r_gain, j_freq, j_bw, j_gain, j_type):
    real_j_bw = j_bw if j_bw > 0 else 10e6
    freq_dist = abs(r_freq - j_freq)
    collision_radius = (r_bw / 2) + (real_j_bw / 2)
    
    if freq_dist < collision_radius:
        # 【干扰区 - 表现极差】
        overlap_factor = 1.0 - (freq_dist / collision_radius)
        
        # 1. 基础惩罚：进入干扰区先扣 20dB (断崖式下跌)
        base_jammed_sinr = 10.0 
        
        # 2. 功率压制：加大干扰增益的影响 (系数 2.0)
        power_penalty = (j_gain * 1.5) * overlap_factor
        
        # 3. 救命稻草：雷达带宽带来的脉压增益 (稍微挽回一点)
        processing_gain = np.log10(r_bw) * 2.0
        
        final_sinr = base_jammed_sinr - power_penalty + processing_gain + (r_gain * 0.1)
        
        # 针对特定致命干扰
        if j_type == 'jammer_close': final_sinr -= 15.0
        if j_type == 'jammer_swept': final_sinr -= 8.0
    else:
        # 【安全区 - 表现极佳】
        # 基础高分 38dB
        final_sinr = 38.0 + (r_gain * 0.1) - (j_gain * 0.05)
        # 模拟极小的环境波动
        final_sinr += np.random.normal(0, 0.5)
        
    return round(final_sinr, 4)

# ==========================================
# 4. 生成逻辑
# ==========================================
def generate_sample():
    # 雷达
    r_freq = random_freq_hz(RADAR_CONFIG['freq_min'], RADAR_CONFIG['freq_max'])
    r_bw = int(random.uniform(RADAR_CONFIG['bw_min'], RADAR_CONFIG['bw_max']))
    r_pw = get_random_100_aligned(RADAR_CONFIG['pw_min'], RADAR_CONFIG['pw_max'])
    r_gain = random.randint(RADAR_CONFIG['gain_min'], RADAR_CONFIG['gain_max'])
    
    # 干扰机
    j_type = random.choice(JAMMER_CONFIG['types'])
    j_freq = random_freq_hz(JAMMER_CONFIG['freq_min'], JAMMER_CONFIG['freq_max'])
    j_gain = random.randint(JAMMER_CONFIG['gain_min'], JAMMER_CONFIG['gain_max'])
    
    j_bw = 0
    j_params = {k: 0 for k in ['Sweep_Step', 'Sweep_Power', 'IntDur', 'IntDelay', 'IntIntv', 'FT_Num', 'FT_Intv', 'FT_Delay']}

    if j_type == 'jammer_swept':
        j_bw = int(random.choice([200e6, 300e6, 400e6]))
        j_params['Sweep_Step'] = int(random.choice([5e6, 10e6]))
        j_params['Sweep_Power'] = random.randint(20, 50)
    elif j_type == 'jammer_wide_band':
        j_bw = int(random.uniform(200e6, 500e6))
    elif j_type == 'jammer_narrow_band':
        j_bw = int(random.uniform(10e6, 30e6))
    elif j_type == 'jammer_InterruptedSampling':
        j_bw = 20e6
        j_params['IntDur'] = get_random_100_aligned(2000, 20000)
        j_params['IntDelay'] = get_random_100_aligned(1000, 10000)
        j_params['IntIntv'] = j_params['IntDur'] + j_params['IntDelay'] + 200
    elif j_type == 'jammer_multi_false_targets':
        j_bw = 20e6
        j_params['FT_Num'] = random.randint(5, 15)
        j_params['FT_Intv'] = random.randint(1000, 3000)
        j_params['FT_Delay'] = 33000
    elif j_type == 'jammer_close':
        j_bw = 60e6
        j_gain = 40 # 极高增益

    sinr = calculate_sinr_high_contrast(r_freq, r_bw, r_gain, j_freq, j_bw, j_gain, j_type)

    return {
        'Jammer_Type': j_type,
        'Jammer_CarrierFreq_GHz': hz_to_ghz(j_freq),
        'Jammer_Gain_dB': j_gain,
        'Jammer_Bandwidth_GHz': hz_to_ghz(j_bw),
        'Radar_CarrierFreq_GHz': hz_to_ghz(r_freq),
        'Radar_Gain_dB': r_gain,
        'Radar_Bandwidth_GHz': hz_to_ghz(r_bw),
        'Radar_PulseWidth_ns': r_pw,
        'Radar_Type': random.choice(RADAR_CONFIG['radar_types']),
        'SINR_dB': sinr,
        'Jammer_Sweep_Step_GHz': hz_to_ghz(j_params['Sweep_Step']),
        'Jammer_IntSample_Dur_ns': j_params['IntDur'],
        'Jammer_IntSample_Delay_ns': j_params['IntDelay'],
        'Jammer_IntSample_Intv_ns': j_params['IntIntv'],
        'Jammer_FalseTarget_Num': j_params['FT_Num'],
        'Radar_ChirpType': RADAR_CONFIG['chirp_type']
    }

# ==========================================
# 5. 执行
# ==========================================
data_list = [generate_sample() for _ in range(NUM_SAMPLES)]
df = pd.DataFrame(data_list)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"高对比度数据集已生成: {OUTPUT_FILE}")
print("\n对比度分析:")
print(f"最高 SINR: {df['SINR_dB'].max()} dB")
print(f"最低 SINR: {df['SINR_dB'].min()} dB")
print(f"平均 SINR: {df['SINR_dB'].mean():.2f} dB")