from train_vae import ResidualBlock,Encoder,StateVAE
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.spatial.distance import cdist

# ====== 1. 环境与路径配置 ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_OUTPUT_DIR = Path("./radar-10.11/radar/dataset/data_4/vae_output")
MODEL_PATH = VAE_OUTPUT_DIR / "best_vae_model.pth"
EMBEDDED_DATA_PATH = VAE_OUTPUT_DIR / "S_embedded.npy"
SCALER_PATH = VAE_OUTPUT_DIR / "scaler_s.joblib"
ENCODER_PATH = VAE_OUTPUT_DIR / "encoder_s.joblib"

# 待测试的小样本数据路径
NEW_DATA_PATH = "radar-10.11/radar/dataset/data_4/simulated_jammer_only_10k.csv" 

# ====== 2. 重新定义模型结构 (必须与训练时一致) ======
# 注意：这里需要包含 ResidualBlock, Encoder, StateVAE 类
# (为了篇幅，这里假设你已经引入了前文中定义的这些类)

# ====== 3. 相似度匹配类 ======
class LatentMatcher:
    def __init__(self, model_path, embedded_path, scaler_path, encoder_path, latent_dim=64):
        # 1. 加载预处理器
        self.scaler = joblib.load(scaler_path)
        self.encoder_s = joblib.load(encoder_path)
        
        # 2. 加载已知隐向量库 (10000, 64)
        self.database_embeddings = np.load(embedded_path)
        
        # 3. 初始化并加载模型
        # 获取输入维度 (one-hot 后的长度 + 数值特征长度)
        # 简单通过一次 dummy 转换来获取维度
        dummy_df = pd.DataFrame([{
            'Jammer_Type': 'jammer_swept', 'Jammer_CarrierFreq_GHz': 0, 'Jammer_Gain_dB': 0,
            'Jammer_Bandwidth_GHz': 0, 'Jammer_Sweep_Step_GHz': 0, 'Jammer_Sweep_Power': 0,
            'Jammer_IntSample_Dur_ns': 0, 'Jammer_IntSample_Delay_ns': 0, 'Jammer_IntSample_Intv_ns': 0,
            'Jammer_FalseTarget_Num': 0, 'Jammer_FalseTarget_Intv_m': 0, 'Jammer_FalseTarget_Delay_m': 0
        }])
        s_dim = self._preprocess(dummy_df).shape[1]
        
        self.model = StateVAE(s_dim, latent_dim).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print(f"模型与数据库加载完成。数据库规模: {self.database_embeddings.shape}")

    def _preprocess(self, df):
        """对输入数据进行与训练时完全一致的预处理"""
        s_cat_cols = ['Jammer_Type']
        s_num_cols = [
            'Jammer_CarrierFreq_GHz', 'Jammer_Gain_dB', 'Jammer_Bandwidth_GHz',
            'Jammer_Sweep_Step_GHz', 'Jammer_Sweep_Power',
            'Jammer_IntSample_Dur_ns', 'Jammer_IntSample_Delay_ns', 'Jammer_IntSample_Intv_ns',
            'Jammer_FalseTarget_Num', 'Jammer_FalseTarget_Intv_m', 'Jammer_FalseTarget_Delay_m'
        ]
        
        # 类别特征编码
        cat_encoded = self.encoder_s.transform(df[s_cat_cols])
        # 数值特征标准化
        num_scaled = self.scaler.transform(df[s_num_cols])
        
        return np.hstack([cat_encoded, num_scaled]).astype(np.float32)

    def find_most_similar(self, input_csv_path, top_k=1):
        # 1. 读取新数据
        df_new = pd.read_csv(input_csv_path).head(10) # 示例：取前10条进行匹配
        S_processed = self._preprocess(df_new)
        
        # 2. 提取隐向量
        with torch.no_grad():
            tensor_S = torch.FloatTensor(S_processed).to(DEVICE)
            mu, _ = self.model.encode(tensor_S)
            query_embeddings = mu.cpu().numpy()
        
        # 3. 计算距离 (欧氏距离)
        # cdist 会返回 (query_count, database_count) 的距离矩阵
        distances = cdist(query_embeddings, self.database_embeddings, metric='euclidean')
        
        results = []
        for i in range(len(query_embeddings)):
            # 找到距离最小的索引
            best_idx = np.argmin(distances[i])
            min_dist = distances[i][best_idx]
            
            results.append({
                'query_index': i,
                'matched_db_index': best_idx,
                'distance': min_dist
            })
            
        return results, df_new

# ====== 4. 执行匹配 ======
if __name__ == "__main__":
    matcher = LatentMatcher(MODEL_PATH, EMBEDDED_DATA_PATH, SCALER_PATH, ENCODER_PATH)
    
    match_results, original_queries = matcher.find_most_similar(NEW_DATA_PATH)
    
    print("\n--- 匹配结果预览 ---")
    for res in match_results:
        print(f"新数据索引 {res['query_index']} -> 最相似数据库索引 {res['matched_db_index']} (距离: {res['distance']:.4f})")