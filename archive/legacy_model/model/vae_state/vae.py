import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

# 读取所有 JSON 数据
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# 加载 JSON 文件路径
base_path = "../dataset/"
jamming_type = load_json(os.path.join(base_path, "Encoded_Jamming_Type.json"))
jammer_cf = load_json(os.path.join(base_path, "Jammer_carrierFrequency.json"))
jamming_params = load_json(os.path.join(base_path, "Jamming_Params.json"))
radar_cf = load_json(os.path.join(base_path, "Radar_carrierFrequency.json"))
radar_params = load_json(os.path.join(base_path, "Radar_Params.json"))

# 组装样本
samples = []

num_samples = min(len(jamming_type), len(jammer_cf), len(jamming_params), len(radar_cf), len(radar_params))
for i in range(num_samples):
    sample = []

    # 编码干扰类型
    sample.append(jamming_type[i])

    # 添加干扰器参数
    sample.extend([jammer_cf[i][0], jammer_cf[i][1]])

    # 添加干扰参数（数值型字段）
    for data in jamming_params[i]:
        sample.append(data)

    # 添加雷达参数
    sample.extend([radar_cf[i][0], radar_cf[i][1]])


    for data in radar_params[i]:
        sample.append(data)

    samples.append(sample)
    print(sample)



# 转为张量
#进行数据归一化
data_tensor = torch.tensor(samples, dtype=torch.float32)
mean = data_tensor.mean(0)
std = data_tensor.std(0)
data_tensor = (data_tensor - mean) / (std + 1e-8)

data_list = data_tensor.tolist()
with open("../dataset/normalized_data.json", "w") as f:
    json.dump(data_list, f)


#定义 Dataset 和 DataLoader
class JammingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # VAE 自编码器：输入 = 输出

dataset = JammingDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)



class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_dec(z))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

vae = VAE(input_dim=data_tensor.shape[1])
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# 简单训练循环
for epoch in range(1000):
    for x, _ in dataloader:
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
