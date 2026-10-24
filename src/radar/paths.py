"""项目路径配置（所有脚本应从此模块读取路径）。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据
DATA_V1_JSON = PROJECT_ROOT / "data" / "v1" / "json"
DATA_V1_RAW = PROJECT_ROOT / "data" / "v1" / "raw"
DATA_V2_JSON = PROJECT_ROOT / "data" / "v2" / "json"
DATA_V2_RAW = PROJECT_ROOT / "data" / "v2" / "raw"
DATA_V2_PROCESSED = PROJECT_ROOT / "data" / "v2" / "processed"
DATA_SIMULATED = PROJECT_ROOT / "data" / "simulated"
DATA_SIMULATED_HIGH_CONTRAST_CSV = (
    DATA_SIMULATED / "simulated_radar_jamming_high_contrast_10k.csv"
)
DATA_RADAR_RAW = PROJECT_ROOT / "data" / "radar_raw"

# 训练产出
OUT_STATE_VAE = PROJECT_ROOT / "outputs" / "state_vae"
OUT_Q_VALUE = PROJECT_ROOT / "outputs" / "q_value"
OUT_Q_FILM = OUT_Q_VALUE / "film_advanced"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_CHECKPOINTS = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_LOGS = PROJECT_ROOT / "outputs" / "logs"
LOG_MAIN = OUT_LOGS / "main.log"

# 兼容旧名
DATA_JSON_DIR = DATA_V2_JSON
PROCESSED_DIR = DATA_V2_PROCESSED
