import json
import numpy as np

# ---------- Step 1. 读取 6 个 JSON 文件 ----------
with open("Encoded_Jamming_Type.json", "r", encoding="utf-8") as f:
    encoded = json.load(f)   # (N,) 或 (N, d1)

with open("Jammer_carrierFrequency.json", "r", encoding="utf-8") as f:
    jam_cf = json.load(f)    # (N,) 或 (N, d2)

with open("Jamming_Params.json", "r", encoding="utf-8") as f:
    jam_params = json.load(f)  # (N, d3)

with open("Radar_carrierFrequency.json", "r", encoding="utf-8") as f:
    rad_cf = json.load(f)    # (N,) 或 (N, d4)

with open("Radar_Params.json", "r", encoding="utf-8") as f:
    rad_params = json.load(f)  # (N, d5)

with open("SINR.json", "r", encoding="utf-8") as f:
    sinr = json.load(f)      # (N,)

# ---------- Step 2. 保证每个数组是二维 ----------
def ensure_2d(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

enc = ensure_2d(encoded)
jam_cf = ensure_2d(jam_cf)
jam_params = ensure_2d(jam_params)
rad_cf = ensure_2d(rad_cf)
rad_params = ensure_2d(rad_params)
sinr = np.array(sinr, dtype=np.float32).reshape(-1, 1)

# ---------- Step 3. 拼接 ----------
# state = [Encoded_Jamming_Type, Jammer_carrierFrequency, Jamming_Params]
state = np.concatenate([enc, jam_cf, jam_params], axis=1)

# action = [Radar_carrierFrequency, Radar_Params]
action = np.concatenate([rad_cf, rad_params], axis=1)

# reward = SINR
reward = sinr

# ---------- Step 4. 保存 ----------
with open("state.json", "w", encoding="utf-8") as f:
    json.dump(state.tolist(), f, ensure_ascii=False)

with open("action.json", "w", encoding="utf-8") as f:
    json.dump(action.tolist(), f, ensure_ascii=False)

with open("reward.json", "w", encoding="utf-8") as f:
    json.dump(reward.flatten().tolist(), f, ensure_ascii=False)
