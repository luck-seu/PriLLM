import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 导入真实数据接口
from priLLM.calibration.real_data import load_real_market_data_for_training
# 导入反向求解价格的函数
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from priLLM.game.equilibrium import find_price_for_flow
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# 设置IEEE论文格式的图形参数（与exp_plot.py保持一致）
plt.rcParams.update({
    'font.size': 8,          
    'axes.titlesize': 8,     
    'axes.labelsize': 8,     
    'xtick.labelsize': 6,    
    'ytick.labelsize': 6,    
    'legend.fontsize': 8,    
    'lines.linewidth': 1.5,   
    'lines.markersize': 4,    
    'figure.dpi': 300,        
    'savefig.dpi': 300,       
    'savefig.bbox': 'tight',  
    'savefig.pad_inches': 0.1, 
    'font.family': 'serif',   
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,       
    'ps.fonttype': 42,        
})

# IEEE双栏格式的标准尺寸（英寸）
SINGLE_COLUMN_WIDTH = 3.2    
DOUBLE_COLUMN_WIDTH = 7.14   
GOLDEN_RATIO = (1 + 5**0.5) / 2  



# ==============================================================================
# 1. 可微均衡层 (FollowerLayer) - 无需修改
#    - 你的 FollowerLayer 代码放在这里，保持原样。
# ==============================================================================
class FollowerLayer(nn.Module):
    # ... 你的 FollowerLayer 代码 ...
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
# 2. 新的学习器 (InterpretableCostLearner)
#    - 学习目标是 wq, wd, 和 biases。
# ==============================================================================
class InterpretableCostLearner(nn.Module):
    def __init__(self, n_max, m_max, initial_wq=0.1, initial_wd=0.3, initial_biases=None):
        super().__init__()
        # 内部包含一个 FollowerLayer 实例
        self.follower_layer = FollowerLayer(n_max, m_max)
 
        # --- 新的学习目标 ---
        # 1. 全局权重 wq 和 wd (通过学习log来保证为正)
        self.log_wq = nn.Parameter(torch.log(torch.tensor(initial_wq, dtype=torch.float32)))
        self.log_wd = nn.Parameter(torch.log(torch.tensor(initial_wd, dtype=torch.float32)))
 
        # 2. 每个模型特定的偏差 (beta_j)
        if initial_biases is None:
            initial_biases = torch.zeros(m_max)
        # 确保 initial_biases 是一个 tensor
        if not isinstance(initial_biases, torch.Tensor):
            initial_biases = torch.tensor(initial_biases, dtype=torch.float32)
        self.biases = nn.Parameter(initial_biases)
 
        # --- 价格权重 wp 固定为 1.0 ---
        # 将其注册为 buffer，这样它会随模型移动 (e.g., to GPU) 但不被视为可训练参数
        self.register_buffer('wp', torch.tensor(1.0, dtype=torch.float32))
 
    def forward(self, p_nominal_bat, cap_bat, dist_cat, demand_bat):
        """
        执行一次完整的预测。
        输入的是名义价格和物理参数。
        """
        B, K, m_max = p_nominal_bat.shape
        
        # --- 1. 从学习到的参数中获取实际权重 ---
        wq = torch.exp(self.log_wq)
        wd = torch.exp(self.log_wd)
        
        # --- 2. 核心逻辑：计算有效价格 ---
        # p_eff = p_nominal + biases
        # 注意：这里的符号是加号。因为 bias 代表的是除了 p, Q, d 之外的“额外成本”。
        # 一个负的 bias (β_j < 0) 意味着这个模型有品牌溢价或内在质量高，从而降低了用户的有效成本。
        # biases 从 (m_max,) 扩展到 (B, K, n_max)
        # biases_expanded = self.biases.view(1, 1, -1).expand(B, K, -1)
        p_eff_bat = p_nominal_bat + self.biases
        
        # --- 3. 调用下层均衡求解器 ---
        # 将学习到的 wq, wd 和固定的 wp 传递给 FollowerLayer
        q_pred = self.follower_layer(
            p_eff_bat=p_eff_bat,
            cap_bat=cap_bat,
            dist_cat=dist_cat,
            demand_bat=demand_bat,
            wp=self.wp.expand(B),  # 固定的 wp
            wd=wd.expand(B),       # 学习到的 wd
            wq=wq.expand(B),       # 学习到的 wq
        )
        return q_pred

def plot_learned_stackelberg_flow_comparison(
    model: InterpretableCostLearner,
    q_real: torch.Tensor,
    p_nominal: torch.Tensor,
    cap_data: torch.Tensor,
    dist_data: torch.Tensor,
    demand_data: torch.Tensor,
    model_names: List[str],
    start_date: str,
    save_dir: str = "stackelberg_results",
    show_plot: bool = True,
    top_n_models: int = 3
) -> None:
    """
    绘制学习完成后的真实流量与Stackelberg博弈计算流量的对比图
    
    Args:
        model: 训练好的可解释成本学习器
        q_real: 真实流量数据 [batch, days, providers]
        p_nominal: 名义价格数据 [batch, days, providers]
        cap_data: TPS容量数据 [batch, days, providers]
        dist_data: TTFT延迟数据 [batch, days, users, providers]
        demand_data: 用户需求数据 [batch, days, users]
        model_names: 模型名称列表
        start_date: 开始日期，格式 YYYY-MM-DD
        save_dir: 保存目录
        show_plot: 是否显示图形
        top_n_models: 显示流量前N的模型
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用学习到的参数进行预测
    model.eval()
    with torch.no_grad():
        q_pred = model(p_nominal, cap_data, dist_data, demand_data)
        
        # 应用约束：如果真实流量为0，则预测流量也为0
        zero_mask = (q_real == 0)
        q_pred = q_pred * (~zero_mask).float()  # 将真实流量为0的位置的预测流量也设为0
    
    # 获取数据形状
    batch_size, num_days, num_providers = q_real.shape
    
    # 创建日期标签
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_dt + timedelta(days=i) for i in range(num_days)]
    
    # 获取流量最高的前N个LLM
    avg_real_flows = q_real.mean(dim=(0, 1))  # [providers]
    top_indices = torch.argsort(avg_real_flows, descending=True)[:top_n_models]
    
    # 定义每个LLM的颜色和标记
    colors = ['#000000', '#555555', '#888888', '#BB0000', '#0000BB', '#00BB00']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    # 1. 绘制流量对比时间序列图
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH/GOLDEN_RATIO))
    
    legend_handles = []
    legend_labels = []
    
    for i, model_idx in enumerate(top_indices):
        model_name = model_names[model_idx]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        model_letter = chr(ord('A') + i)  # A, B, C for models
        
        # 绘制真实值（实线）
        line1 = ax.plot(dates, q_real[0, :, model_idx], 
               color=color, 
               linestyle='-',
               linewidth=1.2,
               marker=marker,
               markersize=4,
               markerfacecolor=color,
               markeredgecolor='white',
               markeredgewidth=0.5,
               label=f'Model {model_letter} (True)')
        
        # 绘制Stackelberg预测（虚线）
        line2 = ax.plot(dates, q_pred[0, :, model_idx], 
               color=color, 
               linestyle='--',
               linewidth=1.2,
               marker=marker,
               markersize=4,
               markerfacecolor='white',
               markeredgecolor=color,
               markeredgewidth=1,
               label=f'Model {model_letter} (Stackelberg)')
        
        legend_handles.extend([line1[0], line2[0]])
        legend_labels.extend([f'Model {model_letter} (True)', f'Model {model_letter} (Stackelberg)'])
    
    # 设置图形属性
    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (M tokens)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # 设置x轴日期格式
    if num_days <= 7:
        interval = 1
    elif num_days <= 14:
        interval = 2
    else:
        interval = max(1, num_days // 7)
        
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存主图
    flow_plot_path = os.path.join(save_dir, "stackelberg_flow_comparison.pdf")
    plt.savefig(flow_plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(flow_plot_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
    
    # 保存legend
    legend_fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 0.6))
    legend = legend_fig.legend(legend_handles, legend_labels, loc='center', ncol=3, fontsize=6,
                              frameon=True, fancybox=False, shadow=False,
                              columnspacing=1.0, handletextpad=0.5)
    legend_path = os.path.join(save_dir, "stackelberg_flow_comparison_legend.pdf")
    plt.savefig(legend_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(legend_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(legend_fig)
    
    if show_plot:
        plt.show()
    plt.close()
    
    # 2. 绘制散点图：真实流量 vs Stackelberg流量
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH/2, SINGLE_COLUMN_WIDTH/2))
    
    # 展平所有数据用于散点图
    q_real_flat = q_real.flatten().detach().numpy()
    q_pred_flat = q_pred.flatten().detach().numpy()
    
    # 计算统计指标
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(q_real_flat, q_pred_flat)
    mae = mean_absolute_error(q_real_flat, q_pred_flat)
    
    # 绘制散点图
    ax.scatter(q_real_flat, q_pred_flat, 
              alpha=0.6, 
              s=12, 
              color='black',
              marker='o',
              edgecolors='gray',
              linewidth=0.3)
    
    # 绘制45度参考线
    max_val = max(q_real_flat.max(), q_pred_flat.max())
    min_val = min(q_real_flat.min(), q_pred_flat.min())
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin], 
           [min_val - margin, max_val + margin], 
           'k--', alpha=0.8, linewidth=1)
    
    # 设置轴范围
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # 添加拟合线
    z = np.polyfit(q_real_flat, q_pred_flat, 1)
    p = np.poly1d(z)
    x_fit = np.array([min_val, max_val])
    y_fit = p(x_fit)
    ax.plot(x_fit, y_fit, 
           color='red', 
           linestyle='-', alpha=0.8, linewidth=1)
    
    ax.set_xlabel('True Flow (M tokens)', fontsize=8)
    ax.set_ylabel('Stackelberg Flow (M tokens)', fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # 添加统计信息作为文本
    textstr = f'$R^2$={r2:.3f}\nMAE={mae:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    scatter_plot_path = os.path.join(save_dir, "stackelberg_scatter_comparison.pdf")
    plt.savefig(scatter_plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(scatter_plot_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
    
    if show_plot:
        plt.show()
    plt.close()
    
    # 3. 绘制学习到的参数
    with torch.no_grad():
        learned_wq = torch.exp(model.log_wq).item()
        learned_wd = torch.exp(model.log_wd).item()
        learned_biases = model.biases.detach().numpy()
    
    # 绘制偏差参数条形图
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH/2))
    
    # 只显示前8个模型的偏差
    display_indices = top_indices[:min(8, len(model_names))]
    display_biases = learned_biases[display_indices]
    display_names = [model_names[i] for i in display_indices]
    
    # 创建简短的模型标签
    short_labels = [f'Model {chr(ord("A") + i)}' for i in range(len(display_names))]
    
    bars = ax.bar(range(len(display_biases)), display_biases,
                 color=['gray' if b >= 0 else 'black' for b in display_biases],
                 edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Learned Bias (β)')
    ax.set_xticks(range(len(display_biases)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 在条形图上添加数值
    for bar, bias in zip(bars, display_biases):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
               f'{bias:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=6)
    
    plt.tight_layout()
    
    bias_plot_path = os.path.join(save_dir, "learned_biases.pdf")
    plt.savefig(bias_plot_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(bias_plot_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
    
    if show_plot:
        plt.show()
    plt.close()
    
    # 4. 输出详细的数值对比
    print(f"\n=== Stackelberg博弈流量预测结果分析 ===")
    print(f"学习到的参数:")
    print(f"  拥堵权重 (w_q): {learned_wq:.4f}")
    print(f"  延迟权重 (w_d): {learned_wd:.4f}")
    print(f"  价格权重 (w_p): 1.0000 (固定)")
    
    print(f"\n整体预测性能:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  平均绝对误差 (MAE): {mae:.2f} M tokens")
    
    # 计算每个模型的预测性能
    print(f"\n前{top_n_models}名模型的详细对比:")
    print(f"{'模型':<15} {'平均真实流量':<12} {'平均预测流量':<12} {'MAE':<8} {'偏差':<8}")
    print("-" * 65)
    
    for i, model_idx in enumerate(top_indices):
        model_name = model_names[model_idx]
        model_letter = chr(ord('A') + i)
        
        real_avg = q_real[0, :, model_idx].mean().item()
        pred_avg = q_pred[0, :, model_idx].mean().item()
        model_mae = torch.abs(q_real[0, :, model_idx] - q_pred[0, :, model_idx]).mean().item()
        bias = learned_biases[model_idx]
        
        print(f"Model {model_letter:<6} {real_avg:>8.1f}M   {pred_avg:>8.1f}M   {model_mae:>6.1f}  {bias:>6.2f}")
    
    print(f"\n图表已保存到目录: {save_dir}/")
    print(f"  - stackelberg_flow_comparison.pdf: 流量时间序列对比")
    print(f"  - stackelberg_scatter_comparison.pdf: 真实vs预测散点图")
    print(f"  - learned_biases.pdf: 学习到的模型偏差参数")



def train_interpretable_cost_learner( # 1. 重命名函数
    start_date: str,
    num_days: int,
    # 2. 移除固定的 wp, wd, wq，它们现在是学习目标
    initial_wq: float = 0.1,      # 提供初始猜测
    initial_wd: float = 0.3,      # 提供初始猜测
    learning_rate: float = 0.01,  # 学习可解释模型时，学习率通常需要调小
    num_epochs: int = 5000,
    data_file: str = "data/release/market_inputs_top20_month.csv",
    min_price_threshold: float = 0.0,
    verbose: bool = True
) -> Tuple[InterpretableCostLearner, List[str], dict]: # 3. 修改返回类型
    """
    使用真实市场数据训练可解释的成本学习器
    """
    
    if verbose:
        print("=== 使用真实数据训练可解释成本参数 ===")
        print(f"训练参数:")
        print(f"  起始日期: {start_date}")
        print(f"  训练天数: {num_days}")
        # 4. 更新打印信息
        print(f"  初始权重猜测: wq={initial_wq}, wd={initial_wd}")
        print(f"  学习率: {learning_rate}")
        print(f"  训练轮数: {num_epochs}")
    
    # 1. 加载真实数据
    try:
        p_nom_data, cap_data, dist_data, demand_data, q_real_data, model_names = \
            load_real_market_data_for_training(start_date, num_days, data_file)
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise
    
    # 2. 筛选价格大于阈值的模型
    avg_prices = p_nom_data.mean(dim=(0, 1))  # 计算每个模型的平均价格
    valid_indices = torch.where(avg_prices > min_price_threshold)[0]
    
    if len(valid_indices) == 0:
        raise ValueError(f"没有模型的价格大于阈值 ${min_price_threshold}/M tokens")
    
    # 筛选数据和模型名称
    p_nom_data_filtered = p_nom_data[:, :, valid_indices]
    cap_data_filtered = cap_data[:, :, valid_indices]
    dist_data_filtered = dist_data[:, :, :, valid_indices]
    q_real_data_filtered = q_real_data[:, :, valid_indices]
    model_names_filtered = [model_names[i] for i in valid_indices]

    # 修正：demand_data也需要被过滤，其数值应为被筛选后模型的流量总和
    demand_data_filtered = q_real_data_filtered.sum(dim=2, keepdim=True)
    
    if verbose:
        print(f"\n筛选结果:")
        print(f"  原始模型数量: {len(model_names)}")
        print(f"  筛选后模型数量: {len(model_names_filtered)}")
        print(f"  被排除的模型 (价格 <= ${min_price_threshold}):")
        excluded_indices = [i for i in range(len(model_names)) if i not in valid_indices]
        for i in excluded_indices:
            avg_price = avg_prices[i].item()
            print(f"    - {model_names[i]} (平均价格: ${avg_price:.3f}/M tokens)")
        
        print(f"\n参与训练的模型:")
        for i, model in enumerate(model_names_filtered):
            idx = valid_indices[i]
            avg_price = avg_prices[idx].item()
            avg_speed = cap_data[0, :, idx].mean().item()
            avg_ttft = dist_data[0, :, 0, idx].mean().item()
            print(f"  {i+1}. {model}")
            print(f"     平均价格: ${avg_price:.3f}/M tokens")
            print(f"     平均速度: {avg_speed:.1f} tokens/s")
            print(f"     平均TTFT: {avg_ttft:.1f} ms")
        
        # 输出详细的q_real_data_filtered信息
        print_q_real_filtered_info_and_format(
            q_real_data_filtered, 
            model_names_filtered, 
            start_date,
            "训练数据 q_real_filtered 详细分析"
        )
 
    # 3. 初始化模型 (这是核心修改)
    n_users = 1
    m_providers = len(model_names_filtered)

    # 调用你的 find_optimal_prices_for_target_flow 函数
    # 这个函数会在内部循环调用 find_price_for_flow
    # 我们传入固定的初始 wq 和 wd
    solved_prices_tensor, solve_results = find_optimal_prices_for_target_flow(
        target_flow=q_real_data_filtered,
        cap_data=cap_data_filtered,
        dist_data=dist_data_filtered,
        demand_data=demand_data_filtered, # demand_data也需要被过滤
        model_names=model_names_filtered,
        wp=1.0,  # 价格权重固定为1
        wd=initial_wd, # 使用初始猜测值
        wq=initial_wq, # 使用初始猜测值
        verbose=verbose # 可以设为False来减少这部分的输出
    )
    
    # 计算平均的名义价格和求解出的有效价格
    p_nominal = p_nom_data_filtered
    p_solved = solved_prices_tensor
    
    # 计算初始偏差，这里可以选择使用平均值或为每一天计算一个偏差
    initial_biases_guess = (p_solved - p_nominal).mean(dim=1).squeeze(0)
    # initial_biases_guess = (p_solved - p_nominal)

    if verbose:
        print(f"\n  步骤2: 根据求解价格和真实价格计算初始偏差 (biases)...")
        biases_list = initial_biases_guess.numpy()
        # print(f"  计算出的初始偏差 (biases): [{', '.join([f'{b:.3f}' for b in biases_list])}]")
        # 解释一下：负偏差意味着该模型价格比理论上应有的价格要低，说明它有内在价值
        # neg_bias_models = [model_names_filtered[i] for i, b in enumerate(biases_list) if b < 0]
        # if neg_bias_models:
            # print(f"  模型 {', '.join(neg_bias_models)} 初始被认为具有内在价值 (负偏差)。")
    
    if verbose:
        print(f"\n初始化模型:")
        print(f"  用户数: {n_users}")
        print(f"  筛选后服务商数: {m_providers}")
    
    # 使用新的模型类进行初始化
    # TODO 一个简单想法：初始化wq，让权重花费，距离花费和价格花费大致相同
    model = InterpretableCostLearner(
        n_max=n_users,          
        m_max=m_providers,      
        initial_wq=initial_wq,
        initial_wd=initial_wd,
        initial_biases=initial_biases_guess 
    )

    # 优化器会自动包含所有 nn.Parameter (log_wq, log_wd, biases)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # 正则化依然重要
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=100, min_lr=1e-6 # 可能需要更耐心的调度器
    )
    loss_fn = nn.MSELoss()
    
    # 3. 训练循环 (大部分保持不变)
    training_history = {
        'losses': [],
        'wq_history': [],
        'wd_history': [],
        'biases_history': [],
        'epochs': []
    }
    
    if verbose:
        print(f"\n=== 开始训练 ===")
    
    best_loss = float('inf')
    best_params = None
    
    for epoch in range(num_epochs):
        # 前向传播 (调用新模型的forward)
        # print(f"p_nom_data_shape:{ p_nom_data_filtered.shape }")
        q_pred = model(p_nom_data_filtered, cap_data_filtered, dist_data_filtered, demand_data_filtered)
        
        # 计算损失 (这部分不变)
        loss = loss_fn(q_pred, q_real_data_filtered)
        
        # 反向传播和优化 (这部分不变)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪对这种模型很重要
        optimizer.step()
        
        # 7. 记录历史 (记录新的参数)
        current_loss = loss.item()
        with torch.no_grad():
            current_wq = torch.exp(model.log_wq).item()
            current_wd = torch.exp(model.log_wd).item()
            current_biases = model.biases.data.clone()
 
        scheduler.step(current_loss)
        
        training_history['losses'].append(current_loss)
        training_history['wq_history'].append(current_wq)
        training_history['wd_history'].append(current_wd)
        training_history['biases_history'].append(current_biases.numpy().copy())
        training_history['epochs'].append(epoch + 1)
        
        # 保存最佳模型
        if current_loss < best_loss:
            best_loss = current_loss
            # 保存最佳参数
            with torch.no_grad():
                best_params = {
                    'wq': torch.exp(model.log_wq).item(),
                    'wd': torch.exp(model.log_wd).item(),
                    'biases': model.biases.data.clone()
                }
            # ... (你的早停逻辑)
        
        # 8. 打印进度 (显示新的参数)
        if verbose and (epoch + 1) % max(1, num_epochs // 100) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:4d}/{num_epochs}] Loss: {current_loss:.6f}, LR: {current_lr:.6f}")
            # 打印学习到的权重
            print(f"  Learned Weights -> w_q (congestion): {current_wq:.4f}, w_d (delay): {current_wd:.4f}")
            # 打印部分偏差值
            biases_list = current_biases.numpy()
            print(f"  Learned Biases (top 3): [{', '.join([f'{b:.2f}' for b in biases_list[:3]])}, ...]")
 
    # ... (训练完成后的分析部分也需要相应修改)
 
    # 4. 训练完成，分析结果
    if verbose:
        print(f"\n=== 训练完成 ===")
        print(f"最终损失: {training_history['losses'][-1]:.6f}")
        print(f"最佳损失: {best_loss:.6f}")
        
        # 9. 显示最终学习到的参数
        final_wq = best_params['wq']
        final_wd = best_params['wd']
        final_biases = best_params['biases'].numpy()
        
        print(f"\n最终学习到的成本参数:")
        print(f"  全局拥堵权重 (w_q): {final_wq:.4f}")
        print(f"  全局延迟权重 (w_d): {final_wd:.4f}")
        print(f"\n学习到的模型特定偏差 (β_j):{final_biases}")
        
        # 偏差的负值越大，代表该模型的内在质量/品牌价值越高
        ranking_indices = np.argsort(-final_biases) # 按 -bias 排序，即按质量排序
        
        # print(f"{'排名':<4} {'模型名称':<20} {'偏差 (β_j)':<12} {'解读'}")
        # print("-" * 60)
        # for rank, idx in enumerate(ranking_indices):
        #     model_name = model_names_filtered[idx]
        #     bias_val = final_biases[idx]
        #     interpretation = "高质量/品牌溢价" if bias_val < 0 else "低质量/品牌折价"
        #     print(f"{rank+1:<4} {model_name:<20} {bias_val:<12.3f} {interpretation}")
 
    # 10. 返回训练好的模型
    # 确保将最佳参数加载回模型
    with torch.no_grad():
        model.log_wq.data = torch.log(torch.tensor(best_params['wq']))
        model.log_wd.data = torch.log(torch.tensor(best_params['wd']))
        model.biases.data = best_params['biases']
    
    # 11. 绘制学习结果对比图
    if verbose:
        print(f"\n=== 生成可视化结果 ===")
        try:
            plot_learned_stackelberg_flow_comparison(
                model=model,
                q_real=q_real_data_filtered,
                p_nominal=p_nom_data_filtered,
                cap_data=cap_data_filtered,
                dist_data=dist_data_filtered,
                demand_data=demand_data_filtered,
                model_names=model_names_filtered,
                start_date=start_date,
                save_dir=f"stackelberg_results_{start_date}_{num_days}days",
                show_plot=False,  # 在训练过程中不显示图形
                top_n_models=min(5, len(model_names_filtered))
            )
        except Exception as e:
            print(f"绘图过程中出现错误: {e}")
            print("继续执行，跳过可视化步骤")
        
    return model, model_names_filtered, training_history

def predict_market_share(
    model: train_interpretable_cost_learner,
    p_nominal: torch.Tensor,
    cap: torch.Tensor, 
    dist: torch.Tensor,
    demand: torch.Tensor,
    model_names: List[str],
    q_real: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> torch.Tensor:
    """
    使用训练好的模型预测市场份额，并可选择与真实流量对比
    
    Args:
        model: 训练好的QualityLearner
        p_nominal: 名义价格
        cap: TPS容量
        dist: TTFT延迟
        demand: 用户需求
        model_names: 模型名称
        q_real: 可选的真实流量数据，用于对比
        verbose: 是否打印结果
    
    Returns:
        torch.Tensor: 预测的流量分配
    """
    
    with torch.no_grad():
        q_pred = model(p_nominal, cap, dist, demand)
    
    if verbose:
        print(f"\n=== 市场份额预测 ===")
        batch_size, num_days, num_providers = q_pred.shape
        
        if q_real is not None:
            # 如果有真实数据，显示对比
            print(f"显示预测流量与真实流量对比:")
            
            num_days_to_show = min(3, num_days)  # 只显示前3天
            for day in range(num_days_to_show):
                print_flow_comparison(
                    q_real, q_pred, model_names,
                    day_idx=day, title=f"市场份额预测对比",
                    show_top_n=8
                )
        else:
            # 只显示预测结果
            print(f"显示预测流量:")
            for day in range(min(3, num_days)):  # 只显示前3天
                print(f"\n第 {day+1} 天预测:")
                day_total = q_pred[0, day, :].sum().item()
                
                # 按预测流量排序
                day_flows = q_pred[0, day, :].detach().numpy()
                sorted_indices = np.argsort(day_flows)[::-1]  # 从高到低
                
                print(f"总预测流量: {day_total:.1f}M tokens")
                print(f"{'排名':<4} {'模型名称':<20} {'预测流量':<12} {'市场份额':<8}")
                print("-" * 50)
                
                for rank, idx in enumerate(sorted_indices[:8]):  # 显示前8名
                    flow = day_flows[idx]
                    share = flow / day_total * 100 if day_total > 0 else 0
                    print(f"{rank+1:<4} {model_names[idx]:<20} {flow:>8.1f}M   {share:>6.1f}%")
    
    return q_pred

def find_optimal_prices_for_target_flow(
    target_flow: torch.Tensor,
    cap_data: torch.Tensor,
    dist_data: torch.Tensor, 
    demand_data: torch.Tensor,
    model_names: List[str],
    wp: float = 0.6,
    wd: float = 0.3,
    wq: float = 0.1,
    w0: float = 0.0,
    solver: str = "OSQP",
    verbose: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    给定目标流量分配，使用数学优化方法反向求解最优价格
    
    Args:
        target_flow: 目标流量分配 [batch, days, providers]
        cap_data: TPS容量数据 [batch, days, providers]  
        dist_data: TTFT延迟数据 [batch, days, users, providers]
        demand_data: 用户需求数据 [batch, days, users]
        model_names: 模型名称列表
        wp, wd, wq, w0: 用户偏好权重参数
        solver: CVXPY求解器名称
        verbose: 是否打印详细信息
        
    Returns:
        Tuple[torch.Tensor, dict]:
        - 反向求解得到的价格 [batch, days, providers]
        - 求解状态和统计信息
    """
    
    if verbose:
        print(f"\n=== 反向求解最优价格 ===")
        print(f"目标流量形状: {target_flow.shape}")
        print(f"用户偏好权重: wp={wp}, wd={wd}, wq={wq}, w0={w0}")
        print(f"求解器: {solver}")
    
    batch_size, num_days, num_providers = target_flow.shape
    n_users = dist_data.shape[2]
    
    # 存储所有求解结果
    solved_prices = torch.zeros_like(target_flow)
    solve_results = {
        'successful_days': 0,
        'failed_days': 0,
        'statuses': [],
        'price_statistics': {},
        'solver_info': []
    }
    
    for batch_idx in range(batch_size):
        for day_idx in range(num_days):
            try:
                # 提取当天数据 
                x_target = target_flow[batch_idx, day_idx, :].detach().numpy()  # (m,)
                c_day = cap_data[batch_idx, day_idx, :].detach().numpy()        # (m,) 
                d_day = dist_data[batch_idx, day_idx, :, :].detach().numpy()    # (n, m)
                demand_day = demand_data[batch_idx, day_idx, :].detach().numpy() # (n,)
                
                # 将单个用户的流量扩展为多用户格式以符合反向求解函数的要求
                # 假设每个用户的流量分配与总流量成比例
                if n_users == 1:
                    # 单用户情况：直接使用目标流量
                    x_star_matrix = x_target.reshape(1, -1)  # (1, m)
                else:
                    # 多用户情况：按需求比例分配流量
                    total_demand = demand_day.sum()
                    if total_demand > 0:
                        x_star_matrix = np.zeros((n_users, num_providers))
                        for i in range(n_users):
                            user_ratio = demand_day[i] / total_demand
                            x_star_matrix[i, :] = x_target * user_ratio
                    else:
                        x_star_matrix = np.zeros((n_users, num_providers))
                
                # 调用反向求解函数
                p_found, lambda_found, status = find_price_for_flow(
                    x_star=x_star_matrix,
                    c=c_day,
                    d=d_day,
                    w_p=wp,
                    w_d=wd, 
                    w_q=wq,
                    w0=w0,
                    solver=solver
                )
                
                solve_results['statuses'].append(status)
                solve_results['solver_info'].append({
                    'batch': batch_idx,
                    'day': day_idx,
                    'status': status,
                    'x_target_sum': x_target.sum(),
                    'demand_sum': demand_day.sum()
                })
                
                if p_found is not None:
                    # 求解成功
                    solved_prices[batch_idx, day_idx, :] = torch.from_numpy(p_found).float()
                    solve_results['successful_days'] += 1
                    
                    if verbose and day_idx < 3:  # 只打印前3天的详细信息
                        print(f"\n第 {day_idx+1} 天求解成功 (状态: {status}):")
                        print(f"  目标总流量: {x_target.sum():.1f}M tokens")
                        print(f"  求解得到的价格:")
                        for i, (name, price) in enumerate(zip(model_names, p_found)):
                            target_flow_i = x_target[i]
                            print(f"    {name}: ${price:.4f}/M tokens (目标流量: {target_flow_i:.1f}M)")
                else:
                    # 求解失败
                    solve_results['failed_days'] += 1
                    # 使用默认价格或上一天的价格
                    if day_idx > 0:
                        solved_prices[batch_idx, day_idx, :] = solved_prices[batch_idx, day_idx-1, :]
                    else:
                        solved_prices[batch_idx, day_idx, :] = torch.ones(num_providers) * 2.0  # 默认价格
                    
                    if verbose:
                        print(f"\n第 {day_idx+1} 天求解失败 (状态: {status})")
                        print(f"  使用默认价格或前一天价格")
                        
            except Exception as e:
                solve_results['failed_days'] += 1
                if verbose:
                    print(f"\n第 {day_idx+1} 天求解出错: {e}")
                    
                # 使用默认价格
                if day_idx > 0:
                    solved_prices[batch_idx, day_idx, :] = solved_prices[batch_idx, day_idx-1, :]
                else:
                    solved_prices[batch_idx, day_idx, :] = torch.ones(num_providers) * 2.0
    
    # 计算价格统计信息
    if solve_results['successful_days'] > 0:
        successful_prices = solved_prices[solved_prices > 0]  # 过滤掉失败的默认值
        solve_results['price_statistics'] = {
            'mean': successful_prices.mean().item(),
            'std': successful_prices.std().item(),
            'min': successful_prices.min().item(),
            'max': successful_prices.max().item(),
            'median': successful_prices.median().item()
        }

    
    return solved_prices, solve_results

def print_flow_comparison(
    q_real: torch.Tensor,
    q_pred: torch.Tensor,
    model_names: List[str],
    day_idx: int = 0,
    title: str = "流量对比",
    show_top_n: int = 5
) -> None:
    """
    打印真实流量和预测流量的对比
    
    Args:
        q_real: 真实流量数据 [batch, days, providers]
        q_pred: 预测流量数据 [batch, days, providers]
        model_names: 模型名称列表
        day_idx: 要显示的天数索引
        title: 标题
        show_top_n: 显示前N个模型
    """
    print(f"\n=== {title} (第{day_idx+1}天) ===")
    
    # 获取当天的真实和预测流量
    real_flow = q_real[0, day_idx, :].detach().numpy()
    pred_flow = q_pred[0, day_idx, :].detach().numpy()
    
    # 计算总流量和市场份额
    total_real = real_flow.sum()
    total_pred = pred_flow.sum()
    
    # 创建对比数据
    flow_data = []
    for i, model_name in enumerate(model_names):
        real_share = (real_flow[i] / total_real * 100) if total_real > 0 else 0
        pred_share = (pred_flow[i] / total_pred * 100) if total_pred > 0 else 0
        error = abs(pred_flow[i] - real_flow[i])
        relative_error = (error / real_flow[i] * 100) if real_flow[i] > 0 else 0
        
        flow_data.append({
            'name': model_name,
            'real_flow': real_flow[i],
            'pred_flow': pred_flow[i],
            'real_share': real_share,
            'pred_share': pred_share,
            'error': error,
            'relative_error': relative_error
        })
    
    # 按真实流量排序
    flow_data.sort(key=lambda x: x['real_flow'], reverse=True)
    
    print(f"总流量: 真实={total_real:.1f}M tokens, 预测={total_pred:.1f}M tokens")
    print(f"{'模型名称':<20} {'真实流量':<12} {'预测流量':<12} {'真实份额':<8} {'预测份额':<8} {'绝对误差':<10} {'相对误差':<8}")
    print("-" * 90)
    
    # 显示前N个模型
    for i, data in enumerate(flow_data[:show_top_n]):
        print(f"{data['name']:<20} "
              f"{data['real_flow']:>8.1f}M   "
              f"{data['pred_flow']:>8.1f}M   "
              f"{data['real_share']:>6.1f}%  "
              f"{data['pred_share']:>6.1f}%  "
              f"{data['error']:>8.1f}M   "
              f"{data['relative_error']:>6.1f}%")
    
    if len(flow_data) > show_top_n:
        print(f"... (还有 {len(flow_data) - show_top_n} 个模型)")
    
    # 计算整体误差统计
    mae = np.mean([d['error'] for d in flow_data])
    mse = np.mean([d['error']**2 for d in flow_data])
    rmse = np.sqrt(mse)
    mape = np.mean([d['relative_error'] for d in flow_data if d['real_flow'] > 0])
    
    print(f"\n误差统计:")
    print(f"  平均绝对误差 (MAE): {mae:.2f}M tokens")
    print(f"  均方根误差 (RMSE): {rmse:.2f}M tokens")
    print(f"  平均相对误差 (MAPE): {mape:.1f}%")

def print_q_real_filtered_info_and_format(
    q_real_filtered: torch.Tensor,
    model_names_filtered: List[str],
    start_date: str,
    title: str = "q_real_filtered 数据分析"
) -> None:
    """
    打印 q_real_filtered 的详细信息和以行列格式输出数据
    
    Args:
        q_real_filtered: 过滤后的真实流量数据，形状为 [batch_size, num_days, num_providers]
        model_names_filtered: 过滤后的模型名称列表
        start_date: 起始日期
        title: 输出标题
    """
    print(f"\n=== {title} ===")
    
    # 1. 打印数据的基本信息
    batch_size, num_days, num_providers = q_real_filtered.shape
    print(f"数据形状: {q_real_filtered.shape}")
    print(f"  - batch_size (批次大小): {batch_size}")
    print(f"  - num_days (天数): {num_days}")
    print(f"  - num_providers (LLM提供商数): {num_providers}")
    print(f"模型名称: {model_names_filtered}")
    
    # 3. 以行（天数）列（LLM提供商）格式输出数据
    print(f"\n=== 行（天数）列（LLM提供商）格式输出 ===")
    
    # 创建日期列表
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [(start_dt + timedelta(days=i)).strftime('%m-%d') for i in range(num_days)]
    
    # 输出表头
    header = f"{'日期':<8}"
    for model_name in model_names_filtered:
        # 截断长模型名称
        short_name = model_name[:12] + '...' if len(model_name) > 15 else model_name
        header += f"{short_name:>15}"
    header += f"{'日总计':>15}"
    print(header)
    print("-" * len(header))
    
    # 输出每日数据
    for day in range(num_days):
        row = f"{dates[day]:<8}"
        daily_total = 0
        for provider in range(num_providers):
            flow = q_real_filtered[0, day, provider].item()
            daily_total += flow
            row += f"{flow:>15.1f}"
        row += f"{daily_total:>15.1f}"
        print(row)
    
    # 输出列总计（每个模型的总流量）
    footer = f"{'总计':<8}"
    grand_total = 0
    for provider in range(num_providers):
        model_total = q_real_filtered[0, :, provider].sum().item()
        grand_total += model_total
        footer += f"{model_total:>15.1f}"
    footer += f"{grand_total:>15.1f}"
    print("-" * len(header))
    print(footer)
    
    # 4. 以行（用户/批次）列（LLM）格式输出数据（如果有多个批次）
    if batch_size > 1:
        print(f"\n=== 行（批次）列（LLM提供商）格式输出 ===")
        
        # 计算每个批次的总流量
        batch_totals = q_real_filtered.sum(dim=(1,2))  # 每个批次的总流量
        
        # 输出表头
        header = f"{'批次':<8}"
        for model_name in model_names_filtered:
            short_name = model_name[:12] + '...' if len(model_name) > 15 else model_name
            header += f"{short_name:>15}"
        header += f"{'批次总计':>15}"
        print(header)
        print("-" * len(header))
        
        # 输出每个批次的平均数据
        for batch in range(batch_size):
            row = f"{batch:<8}"
            batch_total = 0
            for provider in range(num_providers):
                # 计算该批次该模型的平均日流量
                avg_flow = q_real_filtered[batch, :, provider].mean().item()
                batch_total += avg_flow * num_days
                row += f"{avg_flow:>15.1f}"
            row += f"{batch_total:>15.1f}"
            print(row)

# 添加调用示例，在输出 q_real_filtered 的地方调用这个函数
def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='使用真实数据训练Stackelberg博弈质量学习器')
    parser.add_argument('--start_date', type=str, required=True, 
                       help='起始日期，格式：YYYY-MM-DD')
    parser.add_argument('--num_days', type=int, required=True,
                       help='训练使用的天数')
    parser.add_argument('--wp', type=float, default=0.6,
                       help='价格敏感度权重 (默认: 0.6)')
    parser.add_argument('--wd', type=float, default=0.3,
                       help='TTFT延迟敏感度权重 (默认: 0.3)')
    parser.add_argument('--wq', type=float, default=0.1,
                       help='拥堵/TPS敏感度权重 (默认: 0.1)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='学习率 (默认: 0.01)')
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='训练轮数 (默认: 500)')
    parser.add_argument('--data_file', type=str, default='data/release/market_inputs_top20_month.csv',
                       help='数据文件路径 (默认: data/release/market_inputs_top20_month.csv)')
    parser.add_argument('--min_price_threshold', type=float, default=0.0,
                       help='最小价格阈值，只训练价格大于此值的模型 (默认: 0.0; 0 = 只排除免费模型)')
    parser.add_argument('--save_model', type=str, default='',
                       help='保存训练好的模型到指定路径')
    parser.add_argument('--demo_reverse_pricing', action='store_true',
                       help='演示反向价格求解功能')
    parser.add_argument('--show_plots', action='store_true',
                       help='训练完成后显示可视化图表')
    
    args = parser.parse_args()
    
    # 验证日期格式
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        print("错误：日期格式应为 YYYY-MM-DD")
        return
    
    # 训练模型
    try:
        trained_model, model_names, history = train_interpretable_cost_learner(
            start_date=args.start_date,
            num_days=args.num_days,
            initial_wd=args.wd,
            initial_wq=args.wq,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            data_file=args.data_file,
            min_price_threshold=args.min_price_threshold,
            verbose=True
        )
        
        # 如果用户指定了显示图表，重新加载数据并绘制图表
        if args.show_plots:
            print(f"\n=== 显示训练结果可视化 ===")
            try:
                # 重新加载数据用于可视化
                from priLLM.calibration.real_data import load_real_market_data_for_training
                p_nom_data, cap_data, dist_data, demand_data, q_real_data, all_model_names = \
                    load_real_market_data_for_training(args.start_date, args.num_days, args.data_file)
                
                # 筛选数据（与训练时的筛选保持一致）
                avg_prices = p_nom_data.mean(dim=(0, 1))
                valid_indices = torch.where(avg_prices > args.min_price_threshold)[0]
                
                p_nom_filtered = p_nom_data[:, :, valid_indices]
                cap_filtered = cap_data[:, :, valid_indices]
                dist_filtered = dist_data[:, :, :, valid_indices]
                q_real_filtered = q_real_data[:, :, valid_indices]
                demand_filtered = q_real_filtered.sum(dim=2, keepdim=True)
                
                # 调用可视化函数
                plot_learned_stackelberg_flow_comparison(
                    model=trained_model,
                    q_real=q_real_filtered,
                    p_nominal=p_nom_filtered,
                    cap_data=cap_filtered,
                    dist_data=dist_filtered,
                    demand_data=demand_filtered,
                    model_names=model_names,
                    start_date=args.start_date,
                    save_dir=f"stackelberg_results_{args.start_date}_{args.num_days}days",
                    show_plot=True,  # 显示图表
                    top_n_models=min(3, len(model_names))
                )
            except Exception as e:
                print(f"可视化过程中出现错误: {e}")
                print("继续执行，跳过可视化步骤")
        
        # 保存模型（如果指定）
        if args.save_model:
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'model_names': model_names,
                'training_history': history,
                'hyperparameters': {
                    'wp': args.wp, 'wd': args.wd, 'wq': args.wq,
                    'start_date': args.start_date, 'num_days': args.num_days,
                    'min_price_threshold': args.min_price_threshold
                }
            }, args.save_model)
            print(f"\n模型已保存到: {args.save_model}")
        
        print(f"\n训练成功完成！")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 额外的可视化函数：从保存的模型生成图表
# ==============================================================================
def visualize_saved_model(
    model_path: str,
    start_date: str,
    num_days: int,
    data_file: str = "data/release/market_inputs_top20_month.csv",
    min_price_threshold: float = 0.0,
    save_dir: str = None,
    show_plot: bool = True
) -> None:
    """
    从保存的模型文件生成可视化图表
    
    Args:
        model_path: 保存的模型文件路径
        start_date: 起始日期
        num_days: 天数
        data_file: 数据文件路径
        min_price_threshold: 最小价格阈值
        save_dir: 保存目录，如果为None则自动生成
        show_plot: 是否显示图表
    """
    try:
        # 加载保存的模型
        checkpoint = torch.load(model_path, map_location='cpu')
        hyperparams = checkpoint['hyperparameters']
        model_names = checkpoint['model_names']
        
        print(f"=== 从保存的模型生成可视化 ===")
        print(f"模型文件: {model_path}")
        print(f"训练参数: {hyperparams}")
        
        # 重新加载数据
        from priLLM.calibration.real_data import load_real_market_data_for_training
        p_nom_data, cap_data, dist_data, demand_data, q_real_data, all_model_names = \
            load_real_market_data_for_training(start_date, num_days, data_file)
        
        # 筛选数据
        avg_prices = p_nom_data.mean(dim=(0, 1))
        valid_indices = torch.where(avg_prices > min_price_threshold)[0]
        
        p_nom_filtered = p_nom_data[:, :, valid_indices]
        cap_filtered = cap_data[:, :, valid_indices]
        dist_filtered = dist_data[:, :, :, valid_indices]
        q_real_filtered = q_real_data[:, :, valid_indices]
        demand_filtered = q_real_filtered.sum(dim=2, keepdim=True)
        
        # 重建模型
        n_users = 1
        m_providers = len(model_names)
        model = InterpretableCostLearner(
            n_max=n_users,
            m_max=m_providers,
            initial_wq=0.1,
            initial_wd=0.3
        )
        
        # 加载训练好的参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 设置保存目录
        if save_dir is None:
            save_dir = f"stackelberg_visualization_{start_date}_{num_days}days"
        
        # 生成可视化
        plot_learned_stackelberg_flow_comparison(
            model=model,
            q_real=q_real_filtered,
            p_nominal=p_nom_filtered,
            cap_data=cap_filtered,
            dist_data=dist_filtered,
            demand_data=demand_filtered,
            model_names=model_names,
            start_date=start_date,
            save_dir=save_dir,
            show_plot=show_plot,
            top_n_models=min(3, len(model_names))
        )
        
        print(f"可视化已完成，结果保存到: {save_dir}")
        
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认参数进行演示
    import sys
    if len(sys.argv) == 1:
        print("=== 演示模式 ===")
        print("选择演示类型:")
        print("1. 质量学习器训练演示")
        print("2. 反向价格求解演示")
        print("3. 完整功能演示")
        
        choice = input("请输入选择 (1-3, 默认1): ").strip()
        if not choice:
            choice = "1"
        
        # 演示参数
        demo_start_date = "2024-12-01"
        demo_num_days = 7
        
    else:
        main()


