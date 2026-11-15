import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import priLLM.game.equilibrium

# ==============================================================================
# 1. 可微均衡层 (FollowerLayer)
#    - 它根据输入的“有效价格”和其他参数，计算出理论上的均衡流量。
# ==============================================================================
class FollowerLayer(nn.Module):
    def __init__(self, n_max: int, m_max: int):
        super().__init__()
        self.n_max, self.m_max = n_max, m_max

        # --- CVXPY 模型定义 ---
        x = cp.Variable((n_max, m_max), nonneg=True)
        q = cp.sum(x, axis=0)

        gamma_price = cp.Parameter((m_max,))
        dist_w = cp.Parameter((n_max, m_max))
        scale_vec = cp.Parameter((m_max,), nonneg=True)
        demand = cp.Parameter((n_max,))

        # 势函数 Φ(x)
        obj = cp.Minimize(
            gamma_price @ q +
            cp.sum(cp.multiply(dist_w, x)) +
            0.5 * (cp.sum_squares(x @ cp.diag(scale_vec)) +
                   cp.sum_squares(cp.multiply(scale_vec, q)))
        )
        cons = [cp.sum(x, axis=1) == demand]

        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(
            prob,
            parameters=[gamma_price, dist_w, scale_vec, demand],
            variables=[x]
        )

    def forward(self, p_eff_bat, cap_bat, dist_cat, demand_bat, wp, wd, wq):
        B, K, m_max = p_eff_bat.shape
        n_max = self.n_max
        BK = B * K

        wp = wp.view(B, 1, 1).expand(B, K, 1)
        wd = wd.view(B, 1, 1, 1).expand(B, K, 1, 1)
        wq_sqrt = torch.sqrt(wq.clamp(min=1e-8)).view(B, 1, 1).expand(B, K, 1)

        gamma_price = (wp * p_eff_bat).reshape(BK, m_max)
        dist_w = (wd * dist_cat).reshape(BK, n_max, m_max)
        
        inv_cap = 1.0 / cap_bat.clamp(min=1e-8)
        scale_vec = (wq_sqrt * torch.sqrt(inv_cap)).reshape(BK, m_max)
        
        demand_f = demand_bat.reshape(BK, n_max)

        x_flat, = self.layer(gamma_price, dist_w, scale_vec, demand_f)
        
        x_star = x_flat.view(B, K, n_max, m_max)
        q_star = x_star.sum(dim=2)
        return q_star

# ==============================================================================
# 2. 学习器 (QualityLearner)
#    - 这个类是整个方案的核心。它包含要学习的 `quality` 参数，
#      并定义了如何从名义价格计算有效价格。
# ==============================================================================
class QualityLearner(nn.Module):
    def __init__(self, n_max, m_max, wp, wd, wq, initial_quality=None):
        super().__init__()
        self.follower_layer = FollowerLayer(n_max, m_max)

        # --- 学习目标：每个模型的质量参数 ---
        if initial_quality is None:
            # 随机初始化质量，让模型自己去学
            initial_quality = torch.randn(m_max) * 0.1 
        self.quality = nn.Parameter(initial_quality)
        
        # --- 将已知的权重注册为 buffer (不参与训练但随模型移动) ---
        self.register_buffer('wp', torch.tensor(wp, dtype=torch.float32))
        self.register_buffer('wd', torch.tensor(wd, dtype=torch.float32))
        self.register_buffer('wq', torch.tensor(wq, dtype=torch.float32))

    def forward(self, p_nominal_bat, cap_bat, dist_cat, demand_bat):
        """
        执行一次完整的预测。
        输入的是名义价格和物理参数。
        """
        B, K, m_max = p_nominal_bat.shape

        # --- 核心逻辑：计算有效价格 ---
        # p_eff = p_nominal - quality
        # quality 从 (m_max,) 扩展到 (B, K, m_max)
        # quality_expanded = self.quality.view(1, 1, -1).expand(B, K, -1)
        p_eff_bat = p_nominal_bat - self.quality
        
        # 调用下层均衡求解器
        q_pred = self.follower_layer(
            p_eff_bat,
            cap_bat,
            dist_cat,
            demand_bat,
            self.wp.expand(B),
            self.wd.expand(B),
            self.wq.expand(B),
        )
        return q_pred
    
# ==============================================================================
# 3. 模拟数据生成与主训练循环（如果有真实数据，这部分可以省略）
# ==============================================================================

def train_quality_learner_with_one_day_data(n_users, m_providers, wp, wd, wq, p_nominal_data, cap_data, dist_data, demand_data, q_real_data, ):
    initial_quality_result = priLLM.game.equilibrium.find_w0_for_flow(q_real_data, cap_data, dist_data, p_nominal_data, wp, wd, wq)
    initial_quality_guess = torch.tensor(initial_quality_result[0], dtype=torch.float32)
    
    print(f"初始猜测质量: {initial_quality_guess}")
    model = QualityLearner(n_users, m_providers, wp, wd, wq, initial_quality=-initial_quality_guess/wp)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()


    p_nominal_data = torch.tensor(p_nominal_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    cap_data = torch.tensor(cap_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, m)
    dist_data = torch.tensor(dist_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, m)
    demand_data = torch.tensor(demand_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, n)
    q_real_data = torch.tensor(q_real_data, dtype=torch.float32).unsqueeze(0)  # (1, 1, 1, m)

    for epoch in range(800):
        # 前向传播：用当前的 quality 猜测来预测 q
        q_pred = model(p_nominal_data, cap_data, dist_data, demand_data)

        # 计算预测流量和真实流量之间的损失
        loss = loss_fn(q_pred, q_real_data)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")
            print(f"  > 当前 quality: {model.quality.data.numpy()}")
        # if epoch == 99:
        #     print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")
        #     print(f"  > 当前 quality: {model.quality.data.numpy()}")

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # --- 5. 结果展示 ---
    print("\n--- 训练完成 ---")
    score = model.quality.data.numpy()
    min_quality = score.min()
    score = score - min_quality  # 确保质量非负
    print(f"最终学习到的 quality: {score}")





# --- 模型列表（顺序很重要） ---
llm_list = [
    'anthropic/claude-sonnet-4',
    'qwen/qwen3-coder',
    'google/gemini-2.5-flash',
    'google/gemini-2.5-pro',
    'anthropic/claude-3.7-sonnet',
    'moonshotai/kimi-k2',
]
 
# --- 来自截图的 token 使用量（单位：Billion tokens） ---
token_usage_B = {
    'anthropic/claude-sonnet-4': 463.0,  # 463 B
    'qwen/qwen3-coder':          108.0,  # 108 B
    'google/gemini-2.5-flash':    80.8,  # 80.8 B
    'google/gemini-2.5-pro':      68.7,  # 68.7 B
    'anthropic/claude-3.7-sonnet':62.2,  # 62.2 B
    'moonshotai/kimi-k2':         30.7,  # 30.7 B
}
 
# --- 你抓到的 (price, latency_text, throughput_text)，与 llm_list 一一对应 ---
raw_results = [
    ('3',     '2.20', '66.19'),
    ('0.30',  '1.70', '54.66'),
    ('0.30',  '0.57', '62.27'),
    ('1.25',  '2.57', '82.43'),
    ('3',     '2.00', '59.46'),
    ('0.088', '0.63', '50.62'),
]
 
# 结构化为 dict 方便索引；同时把字符串转成 float
model_metrics = {
    name: {
        'price': float(p),                 # 价格（保持你抓取的单位）
        'latency_s': float(lat),           # 文本延迟（秒）
        'throughput_tok_s': float(tput),   # 吞吐（tokens/s）
        'tokens_B': token_usage_B[name],   # 累计 token 使用量（Billion）
    }
    for name, (p, lat, tput) in zip(llm_list, raw_results)
}
 
# 如果你需要按顺序取出向量来喂给后续训练/优化代码：
prices =              [model_metrics[m]['price']             for m in llm_list]
latencies_s =         [model_metrics[m]['latency_s']         for m in llm_list]
throughputs_tok_s =   [model_metrics[m]['throughput_tok_s']  for m in llm_list]
tokens_B =            [model_metrics[m]['tokens_B']          for m in llm_list]
 
print("prices          :", prices)
print("latencies_s     :", latencies_s)
print("throughputs_tok_s:", throughputs_tok_s)
print("tokens_B        :", tokens_B)
 
# 下面保留你原来的入口；由于函数签名需要很多业务数据，这里不做虚构，只示例如何调用。
