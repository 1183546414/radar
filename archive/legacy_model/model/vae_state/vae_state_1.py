import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 加载数据
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

#jammer_cf = load_json("dataset\data_1\json\Jammer_carrierFrequency.json")
jammer_cf = load_json("dataset/data_2/json/Jammer_carrierFrequency.json")
jamming_params = load_json("dataset/data_2/json/Jamming_Params.json")
jamming_type = load_json("dataset/data_2/json/Encoded_Jamming_Type.json")

# 合成样本
samples = []
num_samples = min(len(jammer_cf), len(jamming_params), len(jamming_type))
for i in range(num_samples):
    sample = []
    sample.append(jamming_type[i])  # 添加干扰类型编码
    sample.extend(jammer_cf[i])     # 添加 [频率, 增益]
    sample.extend(jamming_params[i])  # 添加干扰参数
    samples.append(sample)

# 转为张量
from sklearn.preprocessing import MinMaxScaler

#进行归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(samples)  # samples 是二维列表
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)


#声明Dataset类
class JammingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # VAE 是自编码器：输入 = 目标输出

dataset = JammingDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


#VAE类
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    mse = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

vae = VAE(input_dim=data_tensor.shape[1])
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

loss_list = []


# 训练 VAE
vae.train()
for epoch in range(300):
    total_loss = 0
    for x, _ in dataloader:
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_list.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


vae.eval()
with torch.no_grad():
    for x, _ in dataloader:
        recon_x, mu, logvar = vae(x)
        mse = nn.MSELoss()(recon_x, x)
        print(f"Validation MSE: {mse.item():.6f}")
        break  # 查看一批



# 可视化一个样本的重构效果
input_sample = data_tensor[50].unsqueeze(0)
with torch.no_grad():
    recon_sample, _, _ = vae(input_sample)

plt.plot(input_sample.squeeze().numpy(), label="Original")
plt.plot(recon_sample.squeeze().numpy(), label="Reconstructed")
plt.legend()
plt.title("Input vs Reconstructed Sample")
plt.show()


