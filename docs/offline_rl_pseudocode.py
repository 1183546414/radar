# ============================================================
# Radar Anti-Jamming Offline RL — Pseudocode (PyTorch-style)
# ============================================================
# 依赖建议：
# - numpy, pandas, scikit-learn
# - torch, torch.utils.data
# - faiss 或 sklearn.neighbors.NearestNeighbors (二选一做相似检索)
# - tqdm, matplotlib(可选做可视化)
# ------------------------------------------------------------

import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors   # 或者用 faiss
from sklearn.model_selection import train_test_split

# =============== 0) 配置 ===============

CFG = dict(
    data_json_dir = "./dataset/data_2/json",  # 放 JSON 的目录
    batch_size    = 256,
    lr_vae        = 3e-4,
    lr_q          = 1e-3,
    lr_pi         = 1e-3,
    latent_dim    = 16,      # VAE 潜变量维度
    hidden_dim    = 128,     # 通用隐层
    epochs_vae    = 50,
    epochs_q      = 50,
    epochs_pi     = 20,
    topK_retrieve = 50,      # 相似检索候选数量
    topM_actions  = 10,      # 从候选集中挑前 M 个动作做 Q 打分
    reliable_q_z  = 0.5,     # Q 方差或不确定性阈（伪代码占位）
    device        = "cuda" if torch.cuda.is_available() else "cpu",
)



# =============== 1) 数据加载与处理 ===============
# 从多个 JSON 文件读取字段并构造 S, A, R，完成缺失值填补 & 标准化

def load_json_rows(json_dir: str) -> list:
    rows = []
    for p in Path(json_dir).glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            item = json.load(f)
            rows.append(item)
    return rows

def extract_arrays(rows: list):
    """将 JSON 行提取为 S, A, R 的数值向量（list/ndarray）
       你需要根据真实 JSON 结构展开字段（示例仅示意）
    """
    S_list, A_list, R_list = [], [], []
    for x in rows:
        # ---- 构造状态 S ----
        # Encoded_Jamming_Type: 已编码好的类别向量或索引 -> onehot/embedding 后拼接
        s_jam_type = x["Encoded_Jamming_Type"]              # e.g. [t1, t2, ...] 或 int
        s_jcf      = x["Jammer_carrierFrequency"]           # float
        s_jparams  = x["Jamming_Params"]                    # list/vec
        S = concat_as_vector(s_jam_type, s_jcf, s_jparams)  # 需自己实现 concat_as_vector
        # ---- 构造动作 A ----
        a_rcf     = x["Radar_carrierFrequency"]
        a_rparams = x["Radar_Params"]
        A = concat_as_vector(a_rcf, a_rparams)
        # ---- 奖励 R ----
        R = x["SINR"]  # float

        S_list.append(S)
        A_list.append(A)
        R_list.append(R)
    S = np.asarray(S_list, dtype=float)
    A = np.asarray(A_list, dtype=float)
    R = np.asarray(R_list, dtype=float).reshape(-1, 1)
    return S, A, R

def concat_as_vector(*parts):
    """把若干标量/向量拼成一个一维 numpy 数组。占位实现。"""
    flat = []
    for p in parts:
        if isinstance(p, (list, tuple, np.ndarray)):
            flat.extend(np.array(p, dtype=float).ravel().tolist())
        else:
            flat.append(float(p))
    return np.array(flat, dtype=float)

def preprocess(S, A, R):
    """缺失值填补（中位数）+ 标准化。返回处理后的数据和拟合器。"""
    imp_S = SimpleImputer(strategy="median")
    imp_A = SimpleImputer(strategy="median")
    imp_R = SimpleImputer(strategy="median")

    S_ = imp_S.fit_transform(S)
    A_ = imp_A.fit_transform(A)
    R_ = imp_R.fit_transform(R)

    sc_S = StandardScaler()
    sc_A = StandardScaler()
    sc_R = StandardScaler()  # 也可不标 R，看你的 Q 训练习惯

    S_std = sc_S.fit_transform(S_)
    A_std = sc_A.fit_transform(A_)
    R_std = sc_R.fit_transform(R_)

    artifacts = dict(imp_S=imp_S, imp_A=imp_A, imp_R=imp_R,
                     sc_S=sc_S, sc_A=sc_A, sc_R=sc_R)
    return S_std, A_std, R_std, artifacts




# =============== 2) 数据集封装 ===============

class SAROfflineDataset(Dataset):
    def __init__(self, S, A, R):
        self.S = torch.as_tensor(S, dtype=torch.float32)
        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.R = torch.as_tensor(R, dtype=torch.float32)

    def __len__(self): return len(self.R)
    def __getitem__(self, i): return self.S[i], self.A[i], self.R[i]



# =============== 3) 模型结构 ===============
# 3.1 State-VAE（仅对 S 编码）

class StateVAE(nn.Module):
    def __init__(self, s_dim, latent_dim=16, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, s_dim),
        )
    def encode(self, s):
        h = self.enc(s)
        return self.mu(h), self.logvar(h)
    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps
    def forward(self, s):
        mu, logvar = self.encode(s)
        z = self.reparam(mu, logvar)
        s_rec = self.dec(z)
        return s_rec, mu, logvar, z

def vae_loss(s, s_rec, mu, logvar, beta=1.0):
    rec = F.mse_loss(s_rec, s, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld, dict(rec=rec.item(), kld=kld.item())



# 3.2 价值评估器 Q(s, a) → 预测 SINR（可当 FQI/DQN 的函数近似器）
class QNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)



# 3.3 行为先验 π_b(a|s)（离散动作或连续动作的密度估计都可；此处示例为离散化/分类头）
# 若动作为连续，可把 A 做聚类得到 codebook，再做分类；或直接做条件高斯回归（MDN）。
class BehaviorPrior(nn.Module):
    def __init__(self, s_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
    def forward(self, s):
        logits = self.net(s)
        return F.log_softmax(logits, dim=-1)  # 便于用 NLLLoss



# 注：如果你的动作本质是连续向量，推荐两种路线：
#  (A) 动作库离散化：用 KMeans 对 A 聚 K 类，训练 π_b 预测 action_id，再用质心作为动作。
#  (B) 条件高斯/MDN：输出 μ(s), Σ(s) 拟合 A，推断时采样或取 μ(s)。

# =============== 4) 训练流程（离线） ===============

def train_state_vae(model, loader, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        for S, A, R in loader:
            S = S.to(CFG["device"])
            s_rec, mu, logvar, _ = model(S)
            loss, _ = vae_loss(S, s_rec, mu, logvar, beta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        # 可加入早停/验证集
    return model

def train_q_net(model, loader, epochs, lr):
    # 离线拟合：最小化 (Q(s,a)-SINR)^2
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for S, A, R in loader:
            S, A, R = S.to(CFG["device"]), A.to(CFG["device"]), R.to(CFG["device"])
            pred = model(S, A)
            loss = mse(pred, R)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_behavior_prior(model, loader, epochs, lr, to_action_id):
    # 若已把 A → 离散 action_id（通过聚类/映射）
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    nll = nn.NLLLoss()
    model.train()
    for ep in range(epochs):
        for S, A, R in loader:
            S = S.to(CFG["device"])
            y = torch.as_tensor(to_action_id(A.numpy()), dtype=torch.long, device=CFG["device"])
            logp = model(S)                # [B, K]
            loss = nll(logp, y)           # 行为克隆：最大化历史动作似然
            opt.zero_grad(); loss.backward(); opt.step()
    return model

# =============== 5) 离线库索引（相似检索） ===============

class StateIndex:
    """用 VAE 的均值向量 mu(s) 做索引；也可用 z 或最后一层嵌入。"""
    def __init__(self, knn_k=50):
        self.knn = NearestNeighbors(n_neighbors=knn_k, metric="euclidean")
        self.bank_mu = None   # [N, latent_dim]
        self.bank_A  = None   # [N, a_dim]
        self.bank_S  = None   # 可选保存原始 S
        self.bank_R  = None   # 可选保存 R 供分析

    def fit(self, vae, S_np, A_np, R_np=None):
        vae.eval()
        with torch.no_grad():
            S_t = torch.as_tensor(S_np, dtype=torch.float32, device=CFG["device"])
            mu, logvar = vae.encode(S_t)
            mu_np = mu.cpu().numpy()
        self.bank_mu = mu_np
        self.bank_A  = A_np
        self.bank_S  = S_np
        self.bank_R  = R_np
        self.knn.fit(mu_np)

    def query(self, vae, s_new_np, topK):
        vae.eval()
        with torch.no_grad():
            s_t = torch.as_tensor(s_new_np[None, :], dtype=torch.float32, device=CFG["device"])
            mu, _ = vae.encode(s_t)
            q = mu.cpu().numpy()
        dists, idxs = self.knn.kneighbors(q, n_neighbors=topK, return_distance=True)
        idxs = idxs[0]  # shape: [K]
        return idxs, dists[0]

# =============== 6) 快通道决策（在线推断伪代码） ===============

def select_action_fast_path(vae, qnet, pi_b, index: StateIndex, s_new_np,
                            candidate_M=10, fallback_threshold=CFG["reliable_q_z"]):
    """
    1) 用 VAE 编码 s_new；在索引里找 topK 相似样本；
    2) 从这些样本的动作里做去重 & 频次排序，取前 M 个候选；
    3) 用 Q(s_new, a) 打分，选最高 Q 的动作；
    4) 若 Q 可信度不足（例如集成方差高/校准残差大），回退 π_b。
    """
    # 1) 相似检索
    idxs, _ = index.query(vae, s_new_np, CFG["topK_retrieve"])
    candidate_actions = index.bank_A[idxs]  # [K, a_dim]

    # 2) 简单候选生成：按唯一动作出现频次排序后取前 M
    uniq, counts = np.unique(candidate_actions, axis=0, return_counts=True)
    order = np.argsort(-counts)
    A_cand = uniq[order[:candidate_M]]  # shape [M, a_dim]

    # 3) Q 打分
    s_t = torch.as_tensor(s_new_np[None, :].repeat(len(A_cand), axis=0),
                          dtype=torch.float32, device=CFG["device"])
    a_t = torch.as_tensor(A_cand, dtype=torch.float32, device=CFG["device"])
    with torch.no_grad():
        q_pred = qnet(s_t, a_t).squeeze(-1)                 # [M]
        # 这里可加入不确定性估计，如多模型集成方差；伪代码占位：
        q_uncertainty = torch.zeros_like(q_pred)            # [M], 0 表示完全信任

    best_idx = int(torch.argmax(q_pred).item())
    best_q  = float(q_pred[best_idx].item())
    best_unc = float(q_uncertainty[best_idx].item())
    a_best = A_cand[best_idx]

    # 4) 可信度检查（占位逻辑）
    if best_unc > fallback_threshold:
        # 回退到行为先验：若 π_b 是分类器，输出 argmax；若是条件高斯，取 μ(s)
        with torch.no_grad():
            s_single = torch.as_tensor(s_new_np[None, :], dtype=torch.float32, device=CFG["device"])
            logp = pi_b(s_single)               # [1, K], 离散示例
            a_id = int(torch.argmax(logp, dim=-1).item())
        a_best = id_to_action(a_id)             # 需要你提供从 id → 连续动作向量的映射

    return a_best

# =============== 7) 慢通道轻量更新（分布变化时） ===============

def slow_path_update(qnet, pi_b, new_S, new_A, new_R, steps=500, freeze_q=True):
    """
    触发条件（外部监控决定）：PSI/KS 漂移大、近期 SINR 下滑、累计新数据量达阈等
    - Q：冻结大部分参数，只训练最后一层
    - π：在新数据上做行为克隆  loss = -log π(a|s)
    """
    qnet.train(); pi_b.train()

    # 1) 冻结 Q 的大部分参数
    if freeze_q:
        for n, p in qnet.named_parameters():
            p.requires_grad = (n.endswith(".weight") or n.endswith(".bias")) and ("-1" in n or "2" in n)  # 占位规则
        # 实际中：把 qnet.net[-1] 的参数 requires_grad=True，其他 False

    # 2) 优化器
    opt_q  = torch.optim.AdamW(filter(lambda p: p.requires_grad, qnet.parameters()), lr=CFG["lr_q"])
    opt_pi = torch.optim.AdamW(pi_b.parameters(), lr=CFG["lr_pi"])

    # 3) 张量
    S_t = torch.as_tensor(new_S, dtype=torch.float32, device=CFG["device"])
    A_t = torch.as_tensor(new_A, dtype=torch.float32, device=CFG["device"])
    R_t = torch.as_tensor(new_R, dtype=torch.float32, device=CFG["device"])

    # 4) 轻量更新迭代（小批量可选，这里为简化示例）
    mse, nll = nn.MSELoss(), nn.NLLLoss()

    for step in range(steps):
        # --- 更新 Q：回归 SINR ---
        pred = qnet(S_t, A_t)
        loss_q = mse(pred, R_t)
        opt_q.zero_grad(); loss_q.backward(); opt_q.step()

        # --- 更新 π：行为克隆 ---
        y_id = torch.as_tensor(to_action_id(A_t.detach().cpu().numpy()),
                               dtype=torch.long, device=CFG["device"])
        logp = pi_b(S_t)
        loss_pi = nll(logp, y_id)
        opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()

    # 注：实际中应有验证集早停 & 学习率调度，并记录 OPE 指标变化

# =============== 8) 训练/索引构建主流程 ===============

def main_train_and_build():
    # -- 加载与预处理 --
    rows = load_json_rows(CFG["data_json_dir"])
    S, A, R = extract_arrays(rows)
    S_std, A_std, R_std, artifacts = preprocess(S, A, R)

    s_dim, a_dim = S_std.shape[1], A_std.shape[1]

    S_tr, S_va, A_tr, A_va, R_tr, R_va = train_test_split(S_std, A_std, R_std, test_size=0.2, random_state=42)

    ds_tr = SAROfflineDataset(S_tr, A_tr, R_tr)
    dl_tr = DataLoader(ds_tr, batch_size=CFG["batch_size"], shuffle=True, drop_last=False)

    # -- 建模 --
    vae  = StateVAE(s_dim, CFG["latent_dim"], CFG["hidden_dim"]).to(CFG["device"])
    qnet = QNet(s_dim, a_dim, CFG["hidden_dim"]).to(CFG["device"])

    # 若动作离散化：先离线把 A_std → action_id, 并提供 to_action_id / id_to_action
    # 这里默认你已经准备好这两个函数（聚类+映射），否则可以先仅训练 VAE/Q，π_b 之后补上
    num_actions = get_num_actions()  # 占位
    pi_b = BehaviorPrior(s_dim, num_actions, CFG["hidden_dim"]).to(CFG["device"])

    # -- 训练 --
    vae  = train_state_vae(vae,  dl_tr, CFG["epochs_vae"], CFG["lr_vae"])
    qnet = train_q_net(qnet, dl_tr, CFG["epochs_q"],  CFG["lr_q"])
    pi_b = train_behavior_prior(pi_b, dl_tr, CFG["epochs_pi"], CFG["lr_pi"], to_action_id)

    # -- 构建相似检索索引 --
    index = StateIndex(knn_k=CFG["topK_retrieve"])
    index.fit(vae, S_std, A_std, R_std)  # 用全体/库数据构建索引

    # -- 返回训练产物 --
    return dict(artifacts=artifacts, vae=vae, qnet=qnet, pi_b=pi_b, index=index)

# =============== 9) 快通道推断示意 ===============

def fast_path_infer(pipeline_objs, s_new_raw_json):
    """
    输入：单条新状态的原始 JSON（受干扰雷达数据）
    输出：推荐动作 a_best（已是标准化空间；落地时需反标准化到物理量域）
    """
    arts  = pipeline_objs["artifacts"]
    vae   = pipeline_objs["vae"]
    qnet  = pipeline_objs["qnet"]
    pi_b  = pipeline_objs["pi_b"]
    index = pipeline_objs["index"]

    # ---- 原始 JSON → S ----
    s_new = concat_as_vector(
        s_new_raw_json["Encoded_Jamming_Type"],
        s_new_raw_json["Jammer_carrierFrequency"],
        s_new_raw_json["Jamming_Params"]
    ).reshape(1, -1)

    # 缺失填补+标准化
    s_new = arts["imp_S"].transform(s_new)
    s_new = arts["sc_S"].transform(s_new)[0]   # 1D

    # ---- 快通道选择动作 ----
    a_std = select_action_fast_path(vae, qnet, pi_b, index, s_new)

    # 反标准化回物理动作
    a_std_2d = a_std.reshape(1, -1)
    a_unscaled = arts["sc_A"].inverse_transform(a_std_2d)[0]
    # 如果用了离散质心，需要把质心再映射回原始动作域（含载频/参数），此处略
    return a_unscaled

# =============== 10) 占位函数（你需要按实际实现） ===============

def to_action_id(A_np: np.ndarray) -> np.ndarray:
    """把连续动作向量映射为离散 action_id（例如 KMeans.labels_）。
       需要你事先完成：A_np -> 聚类 -> 保存聚类器 -> 这里调用聚类器.predict 返回 id。
    """
    raise NotImplementedError

def id_to_action(a_id: int) -> np.ndarray:
    """把离散 id 映射回代表性动作（如聚类质心）。"""
    raise NotImplementedError

def get_num_actions() -> int:
    """返回离散动作簇的数量 K。"""
    raise NotImplementedError

# =============== 11) 入口 ===============

if __name__ == "__main__":
    pipe = main_train_and_build()

    # 示例：新样本的 JSON（占位）
    new_json = {
        "Encoded_Jamming_Type": [0,1,0,0],  # 例如 onehot
        "Jammer_carrierFrequency": 1.72e9,
        "Jamming_Params": [0.3, 0.7, 5.0],
    }
    a_recommend = fast_path_infer(pipe, new_json)
    print("Recommended Action (physical space):", a_recommend)

    # 当触发慢通道：
    # new_S, new_A, new_R = ... # 收集的新数据（需先用相同 artifacts 标准化）
    # slow_path_update(pipe["qnet"], pipe["pi_b"], new_S, new_A, new_R, steps=500, freeze_q=True)
