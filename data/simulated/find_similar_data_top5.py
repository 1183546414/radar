import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.spatial.distance import cdist
# 假设 ResidualBlock, Encoder, StateVAE 已经在 train_vae.py 中定义
from train_vae import ResidualBlock, Encoder, StateVAE 

# ====== 1. 环境与路径配置 ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_OUTPUT_DIR = Path("./dataset/data_4/vae_output")
MODEL_PATH = VAE_OUTPUT_DIR / "best_vae_model.pth"
EMBEDDED_DATA_PATH = VAE_OUTPUT_DIR / "S_embedded.npy"
SCALER_PATH = VAE_OUTPUT_DIR / "scaler_s.joblib"
ENCODER_PATH = VAE_OUTPUT_DIR / "encoder_s.joblib"
NEW_DATA_PATH = "./dataset/data_4/simulated_jammer_only_10k.csv"

# ====== 2. 匹配类声明 ======
class LatentMatcher:
    def __init__(self, model_path, embedded_path, scaler_path, encoder_path, latent_dim=64):
        self.scaler = joblib.load(scaler_path)
        self.encoder_s = joblib.load(encoder_path)
        self.database_embeddings = np.load(embedded_path)
        
        # 自动推断输入维度
        cat_dim = sum(len(cat) for cat in self.encoder_s.categories_)
        num_dim = len(self.scaler.mean_)
        s_dim = cat_dim + num_dim
        
        self.model = StateVAE(s_dim, latent_dim).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print(f"成功加载模型。数据库规模: {self.database_embeddings.shape}")

    def _preprocess(self, df):
        s_cat_cols = ['Jammer_Type']
        s_num_cols = [
            'Jammer_CarrierFreq_GHz', 'Jammer_Gain_dB', 'Jammer_Bandwidth_GHz',
            'Jammer_Sweep_Step_GHz', 'Jammer_Sweep_Power',
            'Jammer_IntSample_Dur_ns', 'Jammer_IntSample_Delay_ns', 'Jammer_IntSample_Intv_ns',
            'Jammer_FalseTarget_Num', 'Jammer_FalseTarget_Intv_m', 'Jammer_FalseTarget_Delay_m'
        ]
        cat_encoded = self.encoder_s.transform(df[s_cat_cols])
        num_scaled = self.scaler.transform(df[s_num_cols])
        return np.hstack([cat_encoded, num_scaled]).astype(np.float32)

    def find_top_k(self, input_csv_path, top_k=5):
        # 1. 读取数据 (这里示例读取前20条进行匹配)
        df_new = pd.read_csv(input_csv_path).head(20) 
        S_processed = self._preprocess(df_new)
        
        # 2. 提取隐向量
        with torch.no_grad():
            tensor_S = torch.FloatTensor(S_processed).to(DEVICE)
            mu, _ = self.model.encode(tensor_S)
            query_embeddings = mu.cpu().numpy()
        
        # 3. 计算欧氏距离矩阵 (Queries x Database)
        distances = cdist(query_embeddings, self.database_embeddings, metric='euclidean')
        
        # 4. 提取 Top-K
        results_list = []
        for i in range(len(query_embeddings)):
            # 对距离进行排序，取前 K 个索引
            top_k_indices = np.argsort(distances[i])[:top_k]
            
            row_result = {'Query_Sample_Idx': i}
            for rank in range(top_k):
                idx = top_k_indices[rank]
                dist = distances[i][idx]
                # 在结果表中添加：第 N 名的索引和距离
                row_result[f'Top_{rank+1}_DB_Idx'] = idx
                row_result[f'Top_{rank+1}_Dist'] = round(float(dist), 4)
            
            results_list.append(row_result)
            
        return pd.DataFrame(results_list)

# ====== 3. 执行并输出表格 ======
if __name__ == "__main__":
    matcher = LatentMatcher(MODEL_PATH, EMBEDDED_DATA_PATH, SCALER_PATH, ENCODER_PATH)
    
    # 查找 Top 5
    top5_df = matcher.find_top_k(NEW_DATA_PATH, top_k=5)
    
    # 设置显示选项，方便在控制台查看完整表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + "="*50)
    print("           VAE 隐空间 Top-5 相似度匹配结果")
    print("="*50)
    print(top5_df)
    
    # 保存结果到 CSV
    SAVE_PATH = VAE_OUTPUT_DIR / "top5_similarity_results.csv"
    top5_df.to_csv(SAVE_PATH, index=False)
    print(f"\n[完成] 详细表格已保存至: {SAVE_PATH}")