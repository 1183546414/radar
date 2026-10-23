
# -*- coding: utf-8 -*-
"""
Offline RL + IL (Behavior Cloning + Offline RL) for Radar Anti-Jamming (JSON inputs)

Inputs (all JSON files are lists/arrays of equal length N):
- State  = concat( Encoded_Jamming_Type (1d), Jammer_carrierFrequency (2d), Jamming_Params (2d or more) )
- Action = concat( Radar_carrierFrequency (2d), Radar_Params (2d or more) )
- Reward = SINR (1d float list)

Example run:
  python offline_rl_il_from_json.py \
      --encoded_jamming_type /mnt/data/Encoded_Jamming_Type.json \
      --jammer_carrier_frequency /mnt/data/Jammer_carrierFrequency.json \
      --jamming_params /mnt/data/Jamming_Params.json \
      --radar_carrier_frequency /mnt/data/Radar_carrierFrequency.json \
      --radar_params /mnt/data/Radar_Params.json \
      --sinr /mnt/data/SINR.json \
      --epochs_bc 100 --epochs_joint 200 --lambda_rl 0.7

Note: This is a compact, stable baseline. Tune hidden sizes, lr, and lambda_rl for better results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import math


def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def ensure_2d(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def impute_median(arr: np.ndarray) -> np.ndarray:
    med = np.nanmedian(arr, axis=0)
    inds = np.where(np.isnan(arr))
    if inds[0].size > 0:
        arr[inds] = np.take(med, inds[1])
    return arr


class RadarDataset(Dataset):
    def __init__(self, S, A, R):
        self.S = torch.from_numpy(S.astype(np.float32))
        self.A = torch.from_numpy(A.astype(np.float32))
        self.R = torch.from_numpy(R.astype(np.float32))
    def __len__(self):
        return self.S.shape[0]
    def __getitem__(self, idx):
        return self.S[idx], self.A[idx], self.R[idx]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128,128), act=nn.ReLU):
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
    """π(a|s) = N(μ(s), diag(σ^2)); actions are in standardized space."""
    def __init__(self, state_dim, action_dim, hidden=(128,128)):
        super().__init__()
        self.backbone = MLP(state_dim, action_dim, hidden=hidden)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = -5.0
        self.log_std_max = 2.0
    def forward(self, s):
        mu = self.backbone(s)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std
    def log_prob(self, s, a):
        mu, std = self(s)
        var = std * std
        logp = -0.5 * (((a - mu) ** 2) / (var + 1e-8) + 2 * torch.log(std + 1e-8) + math.log(2*math.pi))
        return logp.sum(dim=-1)
    def sample(self, s):
        mu, std = self(s)
        return mu + std * torch.randn_like(mu)


class Critic(nn.Module):
    """Q(s,a) -> standardized reward"""
    def __init__(self, state_dim, action_dim, hidden=(256,256)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden=hidden)
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


def build_SAR(args):
    # Load components
    enc = ensure_2d(load_json_array(args.encoded_jamming_type))          # (N,1) after ensure_2d
    jam_cf = ensure_2d(load_json_array(args.jammer_carrier_frequency))   # (N,2) expected
    jam_params = ensure_2d(load_json_array(args.jamming_params))         # (N,Kj)
    rad_cf = ensure_2d(load_json_array(args.radar_carrier_frequency))    # (N,2) expected
    rad_params = ensure_2d(load_json_array(args.radar_params))           # (N,Kr)
    sinr = ensure_2d(load_json_array(args.sinr))                         # (N,1)

    # Validate lengths
    N = enc.shape[0]
    for arr, name in [(jam_cf,"Jammer_carrierFrequency"), (jam_params,"Jamming_Params"),
                      (rad_cf,"Radar_carrierFrequency"), (rad_params,"Radar_Params"), (sinr,"SINR")]:
        if arr.shape[0] != N:
            raise ValueError(f"Length mismatch: Encoded_Jamming_Type has {N}, but {name} has {arr.shape[0]}")

    # Build state/action/reward
    S = np.concatenate([enc, jam_cf, jam_params], axis=1).astype(np.float32)
    A = np.concatenate([rad_cf, rad_params], axis=1).astype(np.float32)
    R = sinr.astype(np.float32)

    # Impute NaNs
    S = impute_median(S)
    A = impute_median(A)
    R = impute_median(R)

    return S, A, R


def train(args):
    # Prepare data
    S_raw, A_raw, R_raw = build_SAR(args)

    # Standardize
    ss = StandardScaler().fit(S_raw)
    as_ = StandardScaler().fit(A_raw)
    rs = StandardScaler().fit(R_raw)
    S = ss.transform(S_raw).astype(np.float32)
    A = as_.transform(A_raw).astype(np.float32)
    R = rs.transform(R_raw).astype(np.float32)

    # Torch dataset
    dataset = RadarDataset(S, A, R)
    val_len = max(1, int(args.val_ratio * len(dataset)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(0))

    batch_size = min(args.batch_size, max(8, len(train_ds)//2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = S.shape[1]
    action_dim = A.shape[1]
    actor = GaussianActor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, weight_decay=5e-4)

    def eval_on_val():
        actor.eval(); critic.eval()
        bc_losses, q_mses = [], []
        with torch.no_grad():
            for s, a, r in val_loader:
                s, a, r = s.to(device), a.to(device), r.to(device)
                bc_losses.append((-actor.log_prob(s, a)).mean().item())
                q_mses.append(F.mse_loss(critic(s, a), r).item())
        return (float(np.mean(bc_losses)) if bc_losses else float("nan"),
                float(np.mean(q_mses)) if q_mses else float("nan"))

    # Phase I: Behavior Cloning pretrain
    for ep in range(1, args.epochs_bc+1):
        actor.train()
        for s, a, _ in train_loader:
            s, a = s.to(device), a.to(device)
            loss_bc = (-actor.log_prob(s, a)).mean()
            actor_opt.zero_grad(); loss_bc.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_opt.step()
        if ep % max(1, args.log_interval) == 0:
            vbc, vq = eval_on_val()
            print(f"[BC] epoch {ep:03d} | val NLL {vbc:.4f} | val Q-MSE {vq:.4f}")

    # Phase II: Joint Offline RL + IL
    for ep in range(1, args.epochs_joint+1):
        actor.train(); critic.train()
        for s, a, r in train_loader:
            s, a, r = s.to(device), a.to(device), r.to(device)

            # Critic supervised regression to standardized reward
            q = critic(s, a)
            loss_q = F.mse_loss(q, r)
            critic_opt.zero_grad(); loss_q.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            critic_opt.step()

            # Actor: BC + maximize Q(s, a~π) + optional conservative penalty
            logp = actor.log_prob(s, a)
            a_samp = actor.sample(s)
            q_act = critic(s, a_samp)
            loss_actor = (-logp).mean() + args.lambda_rl * (-q_act.mean())

            if args.conservative > 0.0:
                loss_actor += args.conservative * F.mse_loss(a_samp, a)

            actor_opt.zero_grad(); loss_actor.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_opt.step()

        if ep % max(1, args.log_interval) == 0:
            vbc, vq = eval_on_val()
            print(f"[Joint] epoch {ep:03d} | val NLL {vbc:.4f} | val Q-MSE {vq:.4f}")

    # Save artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), out_dir/"actor.pt")
    torch.save(critic.state_dict(), out_dir/"critic.pt")
    joblib.dump(ss, out_dir/"state_scaler.pkl")
    joblib.dump(as_, out_dir/"action_scaler.pkl")
    joblib.dump(rs, out_dir/"reward_scaler.pkl")
    print("Saved models & scalers to:", out_dir)

    # Inference helper example
    actor.eval()
    with torch.no_grad():
        s0_std = torch.from_numpy(S[:1]).to(device)
        mu, std = actor(s0_std)
        a_pred_std = mu.cpu().numpy()[0]
        a_pred_raw = as_.inverse_transform(a_pred_std.reshape(1,-1))[0]
        print("Example inferred action (original units):", a_pred_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoded_jamming_type", type=str, required=True)
    ap.add_argument("--jammer_carrier_frequency", type=str, required=True)
    ap.add_argument("--jamming_params", type=str, required=True)
    ap.add_argument("--radar_carrier_frequency", type=str, required=True)
    ap.add_argument("--radar_params", type=str, required=True)
    ap.add_argument("--sinr", type=str, required=True)

    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_actor", type=float, default=3e-3)
    ap.add_argument("--lr_critic", type=float, default=3e-3)
    ap.add_argument("--epochs_bc", type=int, default=100)
    ap.add_argument("--epochs_joint", type=int, default=200)
    ap.add_argument("--lambda_rl", type=float, default=0.7)
    ap.add_argument("--conservative", type=float, default=0.0)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="./offline_rl_il_from_json_out")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
