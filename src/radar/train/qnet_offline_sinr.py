# qnet_offline_sar.py
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


# ================== 数据集封装 ==================
class SAROfflineDataset(Dataset):
    """
    用离线数据 (S, A, R) 训练“动作评分器”
      - S: [N, s_dim] 已标准化的状态
      - A: [N, a_dim] 已标准化的动作
      - R: [N, 1]     目标（这里用 SINR 或与之单调相关的标量）
    """
    def __init__(self, S, A, R):
        self.S = torch.as_tensor(S, dtype=torch.float32)
        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)
        assert len(self.S) == len(self.A) == len(self.R), "S/A/R 数量不一致"

    def __len__(self):
        return len(self.S)

    def __getitem__(self, i):
        return self.S[i], self.A[i], self.R[i]


# ================== 标准化器（可选） ==================
class Standardizer:
    """
    对输入/输出做标准化：x' = (x - mean) / std
    若 std 非法（0或极小），退化为不缩放。
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, unbiased=False, keepdim=True)
        self.std = torch.where(self.std < 1e-6, torch.ones_like(self.std), self.std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return x
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return x
        return x * self.std + self.mean


# ================== Q 网络 ==================
class QNet(nn.Module):
    """
    一个简单的 MLP：输入 concat([s, a])，输出 1 维评分（SINR 估计）
    """
    def __init__(self, s_dim: int, a_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        in_dim = s_dim + a_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


# ================== 训练配置 ==================
@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6
    val_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    early_stop_patience: int = 20
    standardize_inputs: bool = True
    standardize_target: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ================== 训练主函数 ==================
class TrainedQNet:
    """
    打包好的结果：包含模型、（可选）标准化器以及推理接口。
    """
    def __init__(
        self,
        model: QNet,
        s_scaler: Optional[Standardizer],
        a_scaler: Optional[Standardizer],
        r_scaler: Optional[Standardizer],
        device: str,
    ):
        self.model = model.to(device)
        self.s_scaler = s_scaler
        self.a_scaler = a_scaler
        self.r_scaler = r_scaler
        self.device = device

    @torch.no_grad()
    def predict(self, S: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        传入 numpy 的 S, A，返回 numpy 的预测 SINR（与训练时 R 的标度一致）
        """
        self.model.eval()
        s = torch.as_tensor(S, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        if self.s_scaler: s = self.s_scaler.transform(s)
        if self.a_scaler: a = self.a_scaler.transform(a)
        pred = self.model(s, a)
        if self.r_scaler: pred = self.r_scaler.inverse(pred)
        return pred.squeeze(-1).cpu().numpy()


def train_qnet(dataset: SAROfflineDataset, cfg: TrainConfig) -> Tuple[TrainedQNet, dict]:
    torch.manual_seed(cfg.seed)
    device = cfg.device

    # ---- 拆分训练/验证 ----
    n_total = len(dataset)
    n_val = max(1, int(n_total * cfg.val_ratio))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    # ---- 标准化（可选）：fit 只在训练集上 ----
    s_scaler = a_scaler = r_scaler = None
    if cfg.standardize_inputs or cfg.standardize_target:
        with torch.no_grad():
            S_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
            A_train = torch.stack([train_set[i][1] for i in range(len(train_set))])
            R_train = torch.stack([train_set[i][2] for i in range(len(train_set))])

        if cfg.standardize_inputs:
            s_scaler, a_scaler = Standardizer(), Standardizer()
            s_scaler.fit(S_train)
            a_scaler.fit(A_train)

            # 将标准化后的副本喂给 DataLoader（避免每步都 transform 开销）
            def _apply_inplace(subset):
                for i in range(len(subset)):
                    s, a, r = subset[i]
                    s = s_scaler.transform(s.unsqueeze(0)).squeeze(0)
                    a = a_scaler.transform(a.unsqueeze(0)).squeeze(0)
                    subset.dataset.S[subset.indices[i]] = s
                    subset.dataset.A[subset.indices[i]] = a

            _apply_inplace(train_set); _apply_inplace(val_set)

        if cfg.standardize_target:
            r_scaler = Standardizer()
            r_scaler.fit(R_train)
            def _apply_r_inplace(subset):
                for i in range(len(subset)):
                    s, a, r = subset[i]
                    r = r_scaler.transform(r.unsqueeze(0)).squeeze(0)
                    subset.dataset.R[subset.indices[i]] = r
            _apply_r_inplace(train_set); _apply_r_inplace(val_set)

    # ---- DataLoader ----
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device.startswith("cuda")))
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device.startswith("cuda")))

    # ---- 模型 / 优化器 ----
    s_dim = dataset.S.shape[1]
    a_dim = dataset.A.shape[1]
    model = QNet(s_dim, a_dim, hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_state = None
    patience = cfg.early_stop_patience
    history = {"train_mse": [], "val_mse": [], "val_mae": []}

    # ---- 训练循环 ----
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for s, a, r in train_loader:
            s, a, r = s.to(device), a.to(device), r.to(device)
            pred = model(s, a)
            loss = loss_fn(pred, r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(s)
        train_mse = train_loss / n_train

        # 验证
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for s, a, r in val_loader:
                s, a, r = s.to(device), a.to(device), r.to(device)
                pred = model(s, a)
                val_loss += loss_fn(pred, r).item() * len(s)
                val_mae  += (pred - r).abs().mean().item() * len(s)
        val_mse = val_loss / n_val
        val_mae = val_mae / n_val

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        # 早停
        if val_mse + 1e-12 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience == 0:
                break

    # 恢复最佳
    if best_state is not None:
        model.load_state_dict(best_state)

    wrapped = TrainedQNet(model, s_scaler, a_scaler, r_scaler, device)
    return wrapped, history



# ================== 用法示例（把这段当作脚本运行时生效） ==================
if __name__ == "__main__":


    # S.shape == [N, s_dim], A.shape == [N, a_dim], R.shape == [N, 1]
    from radar.paths import DATA_V2_PROCESSED
    S = np.load(DATA_V2_PROCESSED / "S_std.npy")
    A = np.load(DATA_V2_PROCESSED / "A_std.npy")
    R = np.load(DATA_V2_PROCESSED / "R_std.npy")

    print(S.shape, A.shape, R.shape, S.dtype)
    ds = SAROfflineDataset(S, A, R)
    cfg = TrainConfig(epochs=300, batch_size=4096, lr=1e-3)
    qnet, hist = train_qnet(ds, cfg)


    import numpy as np
    B = 64
    idx = np.random.choice(len(S), size=B, replace=False)
    s_batch_numpy = S[idx].astype("float32")
    a_batch_numpy = A[idx].astype("float32")
    # 评分任意 (s, a)
    sinr_pred = qnet.predict(s_batch_numpy, a_batch_numpy) 
    # # 这里用随机数据演示：实际使用时把 S, A, R 换成你的离线数据
    # N, s_dim, a_dim = 50000, 10, 4
    # rng = np.random.default_rng(0)
    # S = rng.normal(size=(N, s_dim)).astype(np.float32)
    # A = rng.normal(size=(N, a_dim)).astype(np.float32)

    # # 人为构造一个“真值”关系：R = f(S,A) + 噪声
    # W_s = rng.normal(size=(s_dim, 1)).astype(np.float32)
    # W_a = rng.normal(size=(a_dim, 1)).astype(np.float32)
    # R = S @ W_s + A @ W_a + 0.1 * rng.normal(size=(N, 1)).astype(np.float32)

    # ds = SAROfflineDataset(S, A, R)
    # cfg = TrainConfig(epochs=200, batch_size=2048, lr=2e-3, early_stop_patience=20)

    # qnet, hist = train_qnet(ds, cfg)
    # print(f"最佳验证 MSE: {min(hist['val_mse']):.6f}, MAE: {min(hist['val_mae']):.6f}")

    # # 推理：单条/批量
    # S_test = rng.normal(size=(3, s_dim)).astype(np.float32)
    # A_test = rng.normal(size=(3, a_dim)).astype(np.float32)
    # pred = qnet.predict(S_test, A_test)
    # print("预测 SINR: ", pred)
