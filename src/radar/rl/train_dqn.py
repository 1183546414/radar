import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque

# ==========================================
# 1. 奖励计算逻辑 (严格对齐 simulate_data-2.py)
# ==========================================
def calculate_sinr_reward(jammer_params, radar_params):
    """
    根据上传脚本中的逻辑计算 SINR。
    目标是干扰效果越好（SINR越低），Reward 越高。
    """
    # 频率单位转换 (GHz -> Hz)
    f_j = jammer_params['freq'] * 1e9
    bw_j = jammer_params['bw'] * 1e9
    gain_j = jammer_params['gain']
    
    f_r = radar_params['freq'] * 1e9
    bw_r = radar_params['bw'] * 1e9
    gain_r = radar_params['gain']
    
    # 计算重叠带宽 (Overlap Bandwidth)
    overlap_min = max(f_j - bw_j/2, f_r - bw_r/2)
    overlap_max = min(f_j + bw_j/2, f_r + bw_r/2)
    overlap_bw = max(0, overlap_max - overlap_min)
    
    # 计算信号功率和干扰功率 (简化版 dB 转换)
    # P_signal ∝ Radar_Gain
    # P_jammer ∝ Jammer_Gain * (Overlap / Jammer_BW)
    signal_p = 10**(gain_r / 10.0)
    
    # 避免除以 0
    effective_bw_j = bw_j if bw_j > 0 else 1e6
    jammer_p = (10**(gain_j / 10.0)) * (overlap_bw / effective_bw_j)
    
    # 计算 SINR (dB)
    sinr = 10 * np.log10(signal_p / (jammer_p + 1e-9))
    
    # 强化学习 Reward 设计：
    # 我们的目标是优化雷达参数来对抗干扰，因此希望 SINR 越高越好
    return sinr  

# ==========================================
# 2. DQN 神经网络定义
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. DQN 智能体 (Agent)
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1.0 # 探索率
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = QNetwork(state_dim, action_dim).to(self.device)
        self.target_model = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            
            target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            
            current_q = self.model(state)
            target_q = current_q.clone().detach()
            target_q[action] = target
            
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 4. 离散化动作空间映射
# ==========================================
# 雷达有 5 个参数，我们通过定义“预设配置表”将其映射为离散 Action
def get_action_map():
    # 这里从您的 RADAR_CONFIG 范围中生成一组离散动作组合
    action_table = []
    # 简单生成 100 组随机动作作为动作空间索引
    # 实际应用中可以采用网格采样 (Grid Search)
    for _ in range(100):
        action_table.append({
            'freq': random.uniform(3.0, 5.0),
            'gain': random.randint(20, 40),
            'bw': random.uniform(0.005, 0.025), # GHz
            'pw': random.randint(20000, 40000),
            'type_idx': random.randint(0, 3) # ['线性调频0', '非线性调频1', '相位编码3', '频率编码2']
        })
    return action_table

# ==========================================
# 5. 主训练程序
# ==========================================
if __name__ == "__main__":
    # A. 加载您上传的匹配数据
    df = pd.read_csv("Top5_Vertical_Comparison.csv")
    
    # 状态 S 的列名
    state_cols = [
        'Query_Jammer_CarrierFreq_GHz', 'Query_Jammer_Gain_dB', 'Query_Jammer_Bandwidth_GHz',
        'Query_Jammer_Sweep_Step_GHz', 'Query_Jammer_Sweep_Power', 'Query_Jammer_IntSample_Dur_ns',
        'Query_Jammer_IntSample_Delay_ns', 'Query_Jammer_IntSample_Intv_ns', 'Query_Jammer_FalseTarget_Num',
        'Query_Jammer_FalseTarget_Intv_m', 'Query_Jammer_FalseTarget_Delay_m'
    ]
    
    # B. 初始化
    action_list = get_action_map()
    agent = DQNAgent(state_dim=len(state_cols), action_dim=len(action_list))
    batch_size = 32
    episodes = 50
    
    print("开始训练 DQN...")
    
    for ep in range(episodes):
        total_reward = 0
        # 遍历数据集中的样本作为状态
        for i in range(0, len(df), 5): # 步长为5是因为每个Query有5个Top匹配，我们取主Query即可
            state = df.loc[i, state_cols].values.astype(np.float32)
            
            # 1. Agent 选择一个动作 (雷达配置)
            action_idx = agent.act(state)
            radar_config = action_list[action_idx]
            
            # 2. 计算 Reward
            jammer_config = {
                'freq': df.loc[i, 'Query_Jammer_CarrierFreq_GHz'],
                'bw': df.loc[i, 'Query_Jammer_Bandwidth_GHz'],
                'gain': df.loc[i, 'Query_Jammer_Gain_dB']
            }
            reward = calculate_sinr_reward(jammer_config, radar_config)
            
            # 3. 观测下一个状态 (这里简化为循环取下一个样本)
            next_idx = (i + 5) % len(df)
            next_state = df.loc[next_idx, state_cols].values.astype(np.float32)
            
            # 4. 存储经验并学习
            agent.remember(state, action_idx, reward, next_state)
            total_reward += reward
            
            if i % 100 == 0:
                agent.replay(batch_size)
        
        agent.update_target_network()
        print(f"Episode: {ep+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # 保存模型
    torch.save(agent.model.state_dict(), "dqn_radar_agent.pth")
    print("训练完成，模型已保存。")