import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.spatial.distance import cdist

# ... (ResidualBlock, Encoder, Decoder, StateVAE 类保持不变) ...
# ====== 1. 模型基础组件 (与训练代码完全一致) ======

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.fc(x)



class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[1024, 1024, 768, 512]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(0.05)
            ))
            layers.append(ResidualBlock(h_d)) 
            in_d = h_d
        self.extract = nn.Sequential(*layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.extract(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[512, 768, 1024, 1024]):
        super().__init__()
        layers = []
        in_d = latent_dim
        for h_d in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(0.05)
            ))
            layers.append(ResidualBlock(h_d))
            in_d = h_d
        self.decode_net = nn.Sequential(*layers)
        self.residual_head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.decode_net(z)
        return self.residual_head(h)

class StateVAE(nn.Module):
    def __init__(self, s_dim, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(s_dim, latent_dim)
        self.decoder = Decoder(latent_dim, s_dim)
        
        # 投影头 (用于训练时的对比学习，推理时可不使用，但必须声明以匹配权重)
        self.proj_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 32)
        )

    def reparameterize(self, mu, logvar):
        # 推理模式下通常直接返回 mu，或者你可以根据 MODEL_TYPE 选择
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        delta = self.decoder(z)
        recon_x = x + delta # 假设 skip_coef=1.0
        return recon_x, mu, logvar, z

    def encode(self, x):
        """推理专用：直接返回均值作为隐向量"""
        mu, logvar = self.encoder(x)
        return mu, logvar



class LatentMatcher:
    def __init__(self, model_path, embedded_path, scaler_path, encoder_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(scaler_path)
        self.encoder_s = joblib.load(encoder_path)
        self.db_embeddings = np.load(embedded_path)
        
        cat_dim = sum(len(cat) for cat in self.encoder_s.categories_)
        num_dim = len(self.scaler.mean_)
        self.s_dim = cat_dim + num_dim
        
        self.model = StateVAE(self.s_dim, latent_dim=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"初始化成功。输入维度: {self.s_dim}, 数据库记录数: {len(self.db_embeddings)}")

    def _preprocess(self, df):
        s_cat_cols = ['Jammer_Type']
        s_num_cols = [
            'Jammer_CarrierFreq_GHz', 'Jammer_Gain_dB', 'Jammer_Bandwidth_GHz',
            'Jammer_Sweep_Step_GHz', 'Jammer_Sweep_Power',
            'Jammer_IntSample_Dur_ns', 'Jammer_IntSample_Delay_ns', 'Jammer_IntSample_Intv_ns',
            'Jammer_FalseTarget_Num', 'Jammer_FalseTarget_Intv_m', 'Jammer_FalseTarget_Delay_m'
        ]
        cat_data = self.encoder_s.transform(df[s_cat_cols])
        num_data = self.scaler.transform(df[s_num_cols])
        return np.hstack([cat_data, num_data]).astype(np.float32)

    # ====== 核心改动：支持 Top-K 匹配 ======
    def match(self, new_data_path, top_k=5):
        df_new = pd.read_csv(new_data_path)
        s_input = self._preprocess(df_new)
        tensor_s = torch.FloatTensor(s_input).to(self.device)
        
        with torch.no_grad():
            mu, _ = self.model.encode(tensor_s)
            query_vecs = mu.cpu().numpy()
            
        # 计算距离矩阵 (Query数量, DB数量)
        dists = cdist(query_vecs, self.db_embeddings, metric='euclidean')
        
        # 使用 argsort 获取前 K 个最小距离的索引
        # sorted_indices 形状为 (Query数量, DB数量)
        top_k_indices = np.argsort(dists, axis=1)[:, :top_k]
        
        results = []
        for i in range(len(df_new)):
            res_row = {'new_data_row': i}
            for j in range(top_k):
                db_idx = top_k_indices[i, j]
                distance = dists[i, db_idx]
                # 动态填充 Top_N 的索引和距离
                res_row[f'Top_{j+1}_Index'] = db_idx
                res_row[f'Top_{j+1}_Dist'] = round(float(distance), 4)
            results.append(res_row)
            
        return results, df_new

# ====== 3. 使用示例 ======
if __name__ == "__main__":
    config = {
        'model': "data/simulated/vae_output/best_vae_model.pth",
        'db_vecs': "data/simulated/vae_output/S_embedded.npy",
        'scaler': "data/simulated/vae_output/scaler_s.joblib",
        'encoder': "data/simulated/vae_output/encoder_s.joblib",
        'test_csv': "data/simulated/simulated_jammer_only_10k.csv"
    }
    
    matcher = LatentMatcher(config['model'], config['db_vecs'], config['scaler'], config['encoder'])
    
    # 执行 Top-5 匹配
    matches_list, df_new = matcher.match(config['test_csv'], top_k=5)

    # 1. 转换为结果 DataFrame
    df_results = pd.DataFrame(matches_list)

    # 2. 合并数据：将匹配到的 Top-5 信息拼接到原始数据右侧
    # 我们只需要 df_results 中排除 'new_data_row' 后的列
    final_df = pd.concat([df_new, df_results.drop(columns=['new_data_row'])], axis=1)

    # 3. 保存到本地
    SAVE_PATH = "data/simulated/top5_match_results.csv"
    final_df.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

    print(f"\n--- Top-5 匹配完成 ---")
    print(f"结果已保存至: {SAVE_PATH}")
    print("\n结果预览 (前3行):")
    print(final_df.filter(like='Top_').head(3)) # 仅预览匹配相关的列