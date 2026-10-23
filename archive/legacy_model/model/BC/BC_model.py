# Now plug the extracted numeric arrays into the Offline RL + IL training pipeline from above.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path

# Standardize
state_scaler = StandardScaler().fit(S)
action_scaler = StandardScaler().fit(A)
reward_scaler = StandardScaler().fit(R)

Xs = state_scaler.transform(S).astype(np.float32)
As = action_scaler.transform(A).astype(np.float32)
Rs = reward_scaler.transform(R).astype(np.float32)

class RadarDataset2(Dataset):
    def __init__(self, Xs, As, Rs):
        self.Xs = torch.from_numpy(Xs)
        self.As = torch.from_numpy(As)
        self.Rs = torch.from_numpy(Rs)
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, idx): return self.Xs[idx], self.As[idx], self.Rs[idx]

dataset = RadarDataset2(Xs, As, Rs)
val_len = max(1, int(0.2 * len(dataset)))
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(0))

batch_size = min(64, max(8, len(train_ds)//2))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(64,64), act=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(64,64)):
        super().__init__()
        self.backbone = MLP(state_dim, action_dim, hidden=hidden)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = -5.0
        self.log_std_max = 1.5
    def forward(self, s):
        mu = self.backbone(s)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std
    def log_prob(self, s, a):
        mu, std = self(s)
        var = std * std
        logp = -0.5 * (((a - mu) ** 2) / (var + 1e-8) + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))
        return logp.sum(dim=-1)
    def sample(self, s):
        mu, std = self(s)
        return mu + std * torch.randn_like(mu)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(128,128)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden=hidden)
    def forward(self, s, a):
        return self.net(torch.cat([s,a], dim=-1))

state_dim, action_dim = 3, 2
actor = GaussianActor(state_dim, action_dim).to(device)
critic = Critic(state_dim, action_dim).to(device)

actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-3, weight_decay=5e-4)

def eval_losses():
    actor.eval(); critic.eval()
    with torch.no_grad():
        s_all, a_all, r_all = next(iter(val_loader))
        s_all, a_all, r_all = s_all.to(device), a_all.to(device), r_all.to(device)
        bc = (-actor.log_prob(s_all, a_all)).mean().item()
        q_mse = F.mse_loss(critic(s_all, a_all), r_all).item()
    return bc, q_mse

# Phase I: BC pretrain (few epochs given small data)
epochs_bc = 50
for ep in range(1, epochs_bc+1):
    actor.train()
    for s, a, r in train_loader:
        s = s.to(device); a = a.to(device)
        loss_bc = (-actor.log_prob(s, a)).mean()
        actor_opt.zero_grad(); loss_bc.backward(); actor_opt.step()
    if ep % 10 == 0 or ep == 1:
        bc, q = eval_losses()
        print(f"[BC] epoch {ep:03d} | val NLL {bc:.4f} | val Q-MSE {q:.4f}")

# Phase II: Joint training
lambda_rl = 0.7
epochs_joint = 100
for ep in range(1, epochs_joint+1):
    actor.train(); critic.train()
    for s, a, r in train_loader:
        s = s.to(device); a = a.to(device); r = r.to(device)

        # Critic supervised regression
        q = critic(s, a)
        loss_q = F.mse_loss(q, r)
        critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()

        # Actor: BC + maximize Q(s, a~π)
        logp = actor.log_prob(s, a)
        a_samp = actor.sample(s)  # standardized action space
        q_act = critic(s, a_samp)
        loss_actor = (-logp).mean() + lambda_rl * (-q_act.mean())
        actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()

    if ep % 20 == 0 or ep == 1:
        bc, q = eval_losses()
        print(f"[Joint] epoch {ep:03d} | val NLL {bc:.4f} | val Q-MSE {q:.4f}")

# Save artifacts
outdir = Path("/mnt/data/offline_rl_il_radar_parsed")
outdir.mkdir(parents=True, exist_ok=True)
torch.save(actor.state_dict(), outdir/"actor.pt")
torch.save(critic.state_dict(), outdir/"critic.pt")

import joblib
joblib.dump(state_scaler, outdir/"state_scaler.pkl")
joblib.dump(action_scaler, outdir/"action_scaler.pkl")
joblib.dump(reward_scaler, outdir/"reward_scaler.pkl")

print("Saved to:", outdir)

# Quick inference helper on the parsed/imputed arrays
def infer_action_from_state_raw(raw_state3):
    s_std = state_scaler.transform(np.asarray(raw_state3, dtype=np.float32).reshape(1,-1)).astype(np.float32)
    s_t = torch.from_numpy(s_std).to(device)
    actor.eval()
    with torch.no_grad():
        mu, std = actor(s_t)
        a_std = mu.cpu().numpy()[0]
    a_raw = action_scaler.inverse_transform(a_std.reshape(1,-1))[0]
    return a_raw

print("Example action for first state's raw values:", infer_action_from_state_raw(S[0]))
