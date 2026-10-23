import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.spatial.distance import cdist

# ==========================================
# 1. 模型核心组件 (必须保留，以匹配 .pth 加载)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
    def forward(self, x): return x + self.fc(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[1024, 1024, 768, 512]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Sequential(nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(0.05)), ResidualBlock(h_d)])
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
            layers.extend([nn.Sequential(nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(0.05)), ResidualBlock(h_d)])
            in_d = h_d
        self.decode_net = nn.Sequential(*layers)
        self.residual_head = nn.Linear(hidden_dims[-1], output_dim)
    def forward(self, z): return self.residual_head(self.decode_net(z))

class StateVAE(nn.Module):
    def __init__(self, s_dim, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(s_dim, latent_dim)
        self.decoder = Decoder(latent_dim, s_dim)
        self.proj_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 32))
    def encode(self, x): return self.encoder(x)

# ==========================================
# 2. 纵向拼接匹配器
# ==========================================
class VerticalTopNMatcher:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(config['scaler'])
        self.encoder_s = joblib.load(config['encoder'])
        self.db_embeddings = np.load(config['db_vecs'])
        self.df_database = pd.read_csv(config['database_csv'])
        
        s_dim = sum(len(c) for c in self.encoder_s.categories_) + len(self.scaler.mean_)
        self.model = StateVAE(s_dim).to(self.device)
        self.model.load_state_dict(torch.load(config['model'], map_location=self.device))
        self.model.eval()

    def _preprocess(self, df):
        s_cat_cols = ['Jammer_Type']
        s_num_cols = [
            'Jammer_CarrierFreq_GHz', 'Jammer_Gain_dB', 'Jammer_Bandwidth_GHz',
            'Jammer_Sweep_Step_GHz', 'Jammer_Sweep_Power',
            'Jammer_IntSample_Dur_ns', 'Jammer_IntSample_Delay_ns', 'Jammer_IntSample_Intv_ns',
            'Jammer_FalseTarget_Num', 'Jammer_FalseTarget_Intv_m', 'Jammer_FalseTarget_Delay_m'
        ]
        cat = self.encoder_s.transform(df[s_cat_cols])
        num = self.scaler.transform(df[s_num_cols])
        return np.hstack([cat, num]).astype(np.float32)

    def match_vertical(self, test_csv, top_k=5):
        df_query = pd.read_csv(test_csv)
        s_input = self._preprocess(df_query)
        
        with torch.no_grad():
            mu, _ = self.model.encode(torch.FloatTensor(s_input).to(self.device))
            query_vecs = mu.cpu().numpy()

        dists = cdist(query_vecs, self.db_embeddings, metric='euclidean')
        top_k_indices = np.argsort(dists, axis=1)[:, :top_k]

        all_vertical_rows = []
        print(f"执行纵向拼接处理...")

        for i in range(len(df_query)):
            # 基础 Query 数据行块 (加前缀)
            q_row_base = df_query.iloc[[i]].reset_index(drop=True)
            q_row_base.columns = [f"Query_{c}" for c in q_row_base.columns]
            
            # 对每一个查询，生成 top_k 条记录
            for rank in range(top_k):
                idx = top_k_indices[i, rank]
                dist = dists[i, idx]
                
                # 提取匹配到的库数据块 (加前缀)
                db_row = self.df_database.iloc[[idx]].reset_index(drop=True)
                db_row.columns = [f"Matched_{c}" for c in db_row.columns]
                
                # 记录这是第几名以及距离
                info_cols = pd.DataFrame({
                    'Match_Rank': [rank + 1],
                    'Latent_Distance': [round(dist, 4)],
                    'Original_DB_Index': [idx]
                })
                
                # 横向拼接：[Query数据] + [排名信息] + [匹配到的库数据]
                combined_single_rank = pd.concat([q_row_base, info_cols, db_row], axis=1)
                all_vertical_rows.append(combined_single_rank)

        return pd.concat(all_vertical_rows, axis=0).reset_index(drop=True)

# ==========================================
# 3. 运行
# ==========================================
if __name__ == "__main__":
    cfg = {
        'model': "data/simulated/vae_output/best_vae_model.pth",
        'db_vecs': "data/simulated/vae_output/S_embedded.npy",
        'scaler': "data/simulated/vae_output/scaler_s.joblib",
        'encoder': "data/simulated/vae_output/encoder_s.joblib",
        'database_csv': "data/simulated/simulated_radar_jamming_flat_GHz_10k.csv",
        'test_csv': "data/simulated/simulated_jammer_only_10k.csv"
    }

    matcher = VerticalTopNMatcher(cfg)
    
    # 执行匹配
    final_df = matcher.match_vertical(cfg['test_csv'], top_k=5)

    # 保存结果
    OUTPUT_PATH = "data/simulated/Top5_Vertical_Comparison.csv"
    final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"完成！生成的表格行数：{len(final_df)} (原始样本数 * 5)")
    print(f"文件已保存至: {OUTPUT_PATH}")