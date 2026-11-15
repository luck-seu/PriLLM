import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import argparse
from datetime import datetime, timedelta

# 导入原有的模型定义
from priLLM.calibration.quality_learner import QualityLearner, FollowerLayer
# 导入真实数据接口
from priLLM.calibration.real_data import load_real_market_data_for_training
# 导入反向求解价格的函数
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from priLLM.game.equilibrium import find_price_for_flow

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

def train_quality_learner_with_real_data(
    start_date: str,
    num_days: int,
    wp: float = 0.6,
    wd: float = 0.3, 
    wq: float = 0.1,
    learning_rate: float = 0.5,
    num_epochs: int = 5000,
    data_file: str = "data/release/market_inputs_top20_month.csv",
    min_price_threshold: float = 0.0,
    verbose: bool = True
) -> Tuple[QualityLearner, List[str], dict]:
    """
    使用真实市场数据训练质量学习器
    
    Args:
        start_date (str): 起始日期，格式 'YYYY-MM-DD'
        num_days (int): 训练使用的天数
        wp (float): 价格敏感度权重
        wd (float): TTFT延迟敏感度权重  
        wq (float): 拥堵/TPS敏感度权重
        learning_rate (float): 学习率
        num_epochs (int): 训练轮数
        data_file (str): 数据文件路径
        min_price_threshold (float): 最小价格阈值，只有价格大于此值的模型才参与训练
        verbose (bool): 是否打印详细信息
    
    Returns:
        Tuple[QualityLearner, List[str], dict]:
        - 训练好的模型
        - 模型名称列表
        - 训练历史 (损失值等)
    """
    
    if verbose:
        print("=== 使用真实数据训练Stackelberg博弈质量学习器 ===")
        print(f"训练参数:")
        print(f"  起始日期: {start_date}")
        print(f"  训练天数: {num_days}")
        print(f"  用户偏好权重: wp={wp}, wd={wd}, wq={wq}")
        print(f"  学习率: {learning_rate}")
        print(f"  训练轮数: {num_epochs}")
        print(f"  最小价格阈值: ${min_price_threshold}/M tokens")
    
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

        # # 打印需求调整信息
        # original_demand = demand_data.squeeze().mean().item()
        # filtered_demand = demand_data_filtered.squeeze().mean().item()
        # print(f"\n需求数据调整:")
        # print(f"  原始平均每日总需求: {original_demand:.1f}M tokens")
        # print(f"  筛选后平均每日总需求: {filtered_demand:.1f}M tokens")
        
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
    
    # 3. 初始化模型
    n_users = 1  # 简化为单用户
    m_providers = len(model_names_filtered)  # 使用筛选后的模型数量
    
    # 初始猜测：利用模型快速求解近似解
    solved_prices, solve_results = find_optimal_prices_for_target_flow(
        target_flow=q_real_data_filtered,
        cap_data=cap_data_filtered,
        dist_data=dist_data_filtered,
        demand_data=demand_data_filtered,
        model_names=model_names_filtered,
        wp=wp, wd=wd, wq=wq,
        verbose=verbose
    )
    initial_quality_guess = p_nom_data_filtered - solved_prices
    # initial_quality_guess = torch.zeros(m_providers)
    # initial_quality_guess[0] = -0.7
    # initial_quality_guess[1] = 4.8
    # initial_quality_guess[2] = 8.2
    # initial_quality_guess[3] = 2.2
    
    if verbose:
        print(f"\n初始化模型:")
        print(f"  用户数: {n_users}")
        print(f"  筛选后服务商数: {m_providers}")
        initial_quality_list = initial_quality_guess.numpy()
        print(f"  初始质量猜测: [{initial_quality_list}]")
    
    model = QualityLearner(n_users, m_providers, wp, wd, wq, initial_quality=initial_quality_guess)
    # 添加权重衰减(L2正则化)来防止参数过大
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 使用ReduceLROnPlateau调度器：只在loss停止下降时才降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # 监控loss最小值
        factor=0.9,           # 每次将学习率乘以0.8
        patience=50,          # 等待50个epoch没有改善再降低学习率
        min_lr=1e-6           # 最小学习率
    )
    loss_fn = nn.MSELoss()
    loss_fn_kl = nn.KLDivLoss(reduction='batchmean')
    
    # 3. 训练循环
    training_history = {
        'losses': [],
        'quality_history': [],
        'epochs': []
    }
    
    if verbose:
        print(f"\n=== 开始训练 ===")
    
    best_loss = float('inf')
    best_quality = None
    patience_counter = 0  # 早停计数器
    early_stopping_patience = 5000  # 早停耐心值
    
    for epoch in range(num_epochs):
        # 前向传播（使用筛选后的数据）
        q_pred = model(p_nom_data_filtered, cap_data_filtered, dist_data_filtered, demand_data_filtered)
        
        # 计算损失
        loss = loss_fn(q_pred, q_real_data_filtered)

        smooth = 1e-8
        # 归一化为概率分布
        q_pred_dist = (q_pred + smooth) / (q_pred.sum(dim=-1, keepdim=True) + smooth * m_providers)
        q_real_dist = (q_real_data_filtered + smooth) / (q_real_data_filtered.sum(dim=-1, keepdim=True) + smooth * m_providers)

        # KLDivLoss期望的输入是log-probabilities
        log_q_pred_dist = torch.log(q_pred_dist)

        loss_kl = loss_fn_kl(log_q_pred_dist, q_real_dist)
        
        # 反向传播
        optimizer.zero_grad()
        total_loss = loss + 0.1 * loss_kl # 0.1是权重，需要调整
        total_loss.backward()
        
        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # 记录历史
        current_loss = loss.item()
        current_quality = model.quality.data.clone()
        
        # 更新学习率 - 基于当前loss
        scheduler.step(current_loss)
        
        training_history['losses'].append(current_loss)
        training_history['quality_history'].append(current_quality.numpy().copy())
        training_history['epochs'].append(epoch + 1)
        
        # 保存最佳模型
        if current_loss < best_loss:
            best_loss = current_loss
            best_quality = current_quality.clone()
            patience_counter = 0  # 重置早停计数器
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\n早停触发：连续 {early_stopping_patience} 个epoch没有改善")
                print(f"最佳损失: {best_loss:.6f}")
            break
        
        # 打印进度（显示更多信息，包括当前学习率）
        if verbose and (epoch + 1) % max(1, num_epochs // 100) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:4d}/{num_epochs}] Loss: {current_loss:.6f}, LR: {current_lr:.6f}")
            quality_list = current_quality.numpy()
            print(f"  当前质量参数: [{quality_list}]")
            print(f"  最佳损失: {best_loss:.6f}, 早停计数: {patience_counter}/{early_stopping_patience}")
            
            # 检查质量参数是否合理
            max_quality = torch.max(torch.abs(current_quality)).item()
            if max_quality > 10.0:
                print(f"  警告: 质量参数过大 (最大绝对值: {max_quality:.3f}), 可能需要调整学习率或正则化")
            
            # 每隔较大间隔显示详细的流量对比 (减少输出频率)
            if (epoch + 1) % max(1, num_epochs // 20) == 0:
                print_flow_comparison(
                    q_real_data_filtered, q_pred, model_names_filtered,
                    day_idx=0, title=f"训练进度流量对比 (Epoch {epoch+1})",
                    show_top_n=5
                )
    
    # 4. 训练完成，分析结果
    if verbose:
        print(f"\n=== 训练完成 ===")
        print(f"训练epoch数: {len(training_history['losses'])}")
        print(f"最终损失: {training_history['losses'][-1]:.6f}")
        print(f"最佳损失: {best_loss:.6f}")
        print(f"最终学习率: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 显示最终的流量对比 - 多天详细分析
        print(f"\n=== 最终流量对比分析 ===")
        
        # 计算最终预测
        with torch.no_grad():
            final_q_pred = model(p_nom_data_filtered, cap_data_filtered, dist_data_filtered, demand_data_filtered)
        
        # 显示前几天的详细对比
        num_days_to_show = min(3, p_nom_data_filtered.shape[1])
        for day in range(num_days_to_show):
            print_flow_comparison(
                q_real_data_filtered, final_q_pred, model_names_filtered,
                day_idx=day, title=f"最终流量对比", show_top_n=10
            )
        
        # 计算整体预测性能
        total_mae = 0
        total_mse = 0
        total_samples = 0
        
        for day in range(p_nom_data_filtered.shape[1]):
            real_flow = q_real_data_filtered[0, day, :].detach().numpy()
            pred_flow = final_q_pred[0, day, :].detach().numpy()
            
            mae_day = np.mean(np.abs(pred_flow - real_flow))
            mse_day = np.mean((pred_flow - real_flow)**2)
            
            total_mae += mae_day
            total_mse += mse_day
            total_samples += 1
        
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        avg_rmse = np.sqrt(total_mse / total_samples) if total_samples > 0 else 0
        
        print(f"\n=== 整体预测性能 (所有{total_samples}天) ===")
        print(f"平均MAE: {avg_mae:.2f}M tokens")
        print(f"平均RMSE: {avg_rmse:.2f}M tokens")
        
        print(f"\n学习到的质量参数:")
        final_quality = model.quality.detach().numpy()[0][0]
        for i, (model_name, quality) in enumerate(zip(model_names_filtered, final_quality)):
            print(f"  {model_name}: {quality:.2f}")
        
        # # 质量参数统计
        # quality_stats = {
        #     'mean': np.mean(final_quality),
        #     'std': np.std(final_quality),
        #     'min': np.min(final_quality),
        #     'max': np.max(final_quality),
        #     'range': np.max(final_quality) - np.min(final_quality)
        # }
        # print(f"\n质量参数统计:")
        # print(f"  均值: {quality_stats['mean']:.4f}")
        # print(f"  标准差: {quality_stats['std']:.4f}")
        # print(f"  范围: [{quality_stats['min']:.4f}, {quality_stats['max']:.4f}]")
        # print(f"  极差: {quality_stats['range']:.4f}")
        
        # 质量排名
        quality_ranking = np.argsort(final_quality)[::-1]  # 从高到低
        print(f"\n质量排名 (从高到低):")
        for rank, idx in enumerate(quality_ranking):
            print(f"  {rank+1}. {model_names_filtered[idx]} (质量: {final_quality[idx]:.2f})")

    return model, model_names_filtered, training_history

def predict_market_share(
    model: QualityLearner,
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
    
    if verbose:
        # print(f"\n=== 反向求解结果统计 ===")
        # print(f"成功求解天数: {solve_results['successful_days']}/{num_days}")
        # print(f"失败天数: {solve_results['failed_days']}/{num_days}")
        # print(f"成功率: {solve_results['successful_days']/num_days*100:.1f}%")
        
        if solve_results['successful_days'] > 0:
            stats = solve_results['price_statistics']
            # print(f"\n成功求解的价格统计:")
            # print(f"  平均价格: ${stats['mean']:.4f}/M tokens")
            # print(f"  价格标准差: ${stats['std']:.4f}")
            # print(f"  价格范围: [${stats['min']:.4f}, ${stats['max']:.4f}]")
            # print(f"  中位数价格: ${stats['median']:.4f}")
            
            # 显示每个模型的平均价格
            print(f"\n各模型平均求解价格:")
            avg_prices_by_model = solved_prices.mean(dim=(0, 1))
            for i, (name, avg_price) in enumerate(zip(model_names, avg_prices_by_model)):
                if avg_price > 0:  # 只显示成功求解的
                    print(f"  {name}: ${avg_price:.4f}/M tokens")
        
        # 显示失败情况分析
        if solve_results['failed_days'] > 0:
            failed_statuses = [info['status'] for info in solve_results['solver_info'] 
                             if info['status'] not in ('optimal', 'optimal_inaccurate')]
            if failed_statuses:
                from collections import Counter
                status_counts = Counter(failed_statuses)
                print(f"\n失败状态分析:")
                for status, count in status_counts.items():
                    print(f"  {status}: {count} 次")
    
    return solved_prices, solve_results

def validate_price_solution(
    solved_prices: torch.Tensor,
    target_flow: torch.Tensor,
    cap_data: torch.Tensor,
    dist_data: torch.Tensor,
    demand_data: torch.Tensor,
    model_names: List[str],
    wp: float = 0.6,
    wd: float = 0.3,
    wq: float = 0.1,
    w0: float = 0.0,
    verbose: bool = True
) -> dict:
    """
    验证反向求解得到的价格是否能够产生目标流量分配
    
    Args:
        solved_prices: 反向求解得到的价格
        target_flow: 目标流量分配
        cap_data, dist_data, demand_data: 市场数据
        model_names: 模型名称
        wp, wd, wq, w0: 用户偏好权重
        verbose: 是否打印详细信息
        
    Returns:
        dict: 验证结果统计
    """
    
    if verbose:
        print(f"\n=== 验证价格求解结果 ===")
    
    # 使用反向求解的价格重新计算流量分配
    from priLLM.game.equilibrium import solve_weighted_linear_congestion_eq
    
    batch_size, num_days, num_providers = solved_prices.shape
    validation_results = {
        'total_days': 0,
        'accurate_days': 0,
        'flow_errors': [],
        'relative_errors': [],
        'day_results': []
    }
    
    for batch_idx in range(batch_size):
        for day_idx in range(num_days):
            try:
                # 提取数据
                p_solved = solved_prices[batch_idx, day_idx, :].detach().numpy()
                c_day = cap_data[batch_idx, day_idx, :].detach().numpy() 
                d_day = dist_data[batch_idx, day_idx, :, :].detach().numpy()
                demand_day = demand_data[batch_idx, day_idx, :].detach().numpy()
                target_day = target_flow[batch_idx, day_idx, :].detach().numpy()
                
                # 使用求解的价格计算流量分配
                x_computed, q_computed, _, status = solve_weighted_linear_congestion_eq(
                    p=p_solved,
                    c=c_day,
                    d=d_day,
                    D=demand_day,
                    w_p=wp,
                    w_d=wd,
                    w_q=wq,
                    w0=w0
                )
                
                if status in ('optimal', 'optimal_inaccurate'):
                    # 计算总流量（如果是多用户，需要求和）
                    if x_computed.ndim == 2:
                        computed_flow = x_computed.sum(axis=0)  # 多用户情况
                    else:
                        computed_flow = x_computed  # 单用户情况（一般情况下，不会发生这种情况）

                    if verbose:
                        print(f"\n第 {day_idx+1} 天计算得到的流量分布:")
                        for i, (name, flow) in enumerate(zip(model_names, computed_flow)):
                            print(f"  {name}: {flow:.2f}M tokens")
                    
                    # 计算误差
                    flow_error = np.abs(computed_flow - target_day)
                    total_error = flow_error.sum()
                    relative_error = total_error / target_day.sum() * 100 if target_day.sum() > 0 else 0
                    
                    validation_results['flow_errors'].append(total_error)
                    validation_results['relative_errors'].append(relative_error)
                    validation_results['total_days'] += 1
                    
                    # 认为相对误差小于5%为准确
                    if relative_error < 5.0:
                        validation_results['accurate_days'] += 1
                    
                    day_result = {
                        'day': day_idx,
                        'total_error': total_error,
                        'relative_error': relative_error,
                        'target_sum': target_day.sum(),
                        'computed_sum': computed_flow.sum(),
                        'status': status
                    }
                    validation_results['day_results'].append(day_result)
                    
                    # if verbose and day_idx < 3:
                    #     print(f"\n第 {day_idx+1} 天验证结果:")
                    #     print(f"  目标总流量: {target_day.sum():.1f}M tokens")
                    #     print(f"  计算总流量: {computed_flow.sum():.1f}M tokens") 
                    #     print(f"  总误差: {total_error:.2f}M tokens")
                    #     print(f"  相对误差: {relative_error:.2f}%")
                    #     print(f"  求解状态: {status}")
                        
            except Exception as e:
                if verbose:
                    print(f"第 {day_idx+1} 天验证失败: {e}")
    
    # 计算验证统计
    # if validation_results['total_days'] > 0:
    #     avg_error = np.mean(validation_results['flow_errors'])
    #     avg_relative_error = np.mean(validation_results['relative_errors'])
    #     accuracy_rate = validation_results['accurate_days'] / validation_results['total_days'] * 100
        
    #     validation_results.update({
    #         'average_error': avg_error,
    #         'average_relative_error': avg_relative_error,
    #         'accuracy_rate': accuracy_rate
    #     })
        
    #     if verbose:
    #         print(f"\n=== 验证结果统计 ===")
    #         print(f"验证天数: {validation_results['total_days']}")
    #         print(f"准确天数 (误差<5%): {validation_results['accurate_days']}")
    #         print(f"准确率: {accuracy_rate:.1f}%")
    #         print(f"平均绝对误差: {avg_error:.2f}M tokens")
    #         print(f"平均相对误差: {avg_relative_error:.2f}%")
    
    return validation_results

def demo_reverse_price_solving(
    start_date: str = "2024-12-01",
    num_days: int = 7,
    data_file: str = "data/release/market_inputs_top20_month.csv",
    min_price_threshold: float = 0.0,
    verbose: bool = True
) -> None:
    """
    演示反向价格求解功能
    
    Args:
        start_date: 起始日期
        num_days: 演示天数
        data_file: 数据文件路径
        min_price_threshold: 最小价格阈值
        verbose: 是否打印详细信息
    """
    
    if verbose:
        print("=== 反向价格求解演示 ===")
        print(f"演示参数: 日期={start_date}, 天数={num_days}")
    
    try:
        # 1. 加载真实市场数据
        p_nom_data, cap_data, dist_data, demand_data, q_real_data, model_names = \
            load_real_market_data_for_training(start_date, num_days, data_file)
        
        # 2. 筛选价格大于阈值的模型
        avg_prices = p_nom_data.mean(dim=(0, 1))
        valid_indices = torch.where(avg_prices > min_price_threshold)[0]
        
        if len(valid_indices) == 0:
            print("错误：没有符合价格阈值的模型")
            return
        
        # 筛选数据
        p_nom_filtered = p_nom_data[:, :, valid_indices]
        cap_filtered = cap_data[:, :, valid_indices]
        dist_filtered = dist_data[:, :, :, valid_indices]
        q_real_filtered = q_real_data[:, :, valid_indices]
        demand_filtered = q_real_filtered.sum(dim=2, keepdim=True)
        model_names_filtered = [model_names[i] for i in valid_indices]
        
        if verbose:
            print(f"\n使用 {len(model_names_filtered)} 个模型进行演示:")
            for i, name in enumerate(model_names_filtered):
                avg_price = avg_prices[valid_indices[i]].item()
                print(f"  {i+1}. {name} (平均价格: ${avg_price:.3f}/M tokens)")
            
            # 调用新函数输出详细的q_real_filtered信息
            print_q_real_filtered_info_and_format(
                q_real_filtered, 
                model_names_filtered, 
                start_date,
                "q_real_filtered 真实流量数据详细分析"
            )

        # 3. 使用真实流量作为目标，反向求解价格
        print(f"\n=== 第一步：反向求解价格 ===")
        print("使用真实市场流量作为目标，反向求解理论最优价格...")
        
        solved_prices, solve_results = find_optimal_prices_for_target_flow(
            target_flow=q_real_filtered,
            cap_data=cap_filtered,
            dist_data=dist_filtered,
            demand_data=demand_filtered,
            model_names=model_names_filtered,
            wp=0.6, wd=0.3, wq=0.1,
            verbose=verbose
        )
        
        # 4. 验证求解结果
        print(f"\n=== 第二步：验证求解结果 ===")
        print("验证求解的价格是否能重现目标流量...")
        
        validation_results = validate_price_solution(
            solved_prices=solved_prices,
            target_flow=q_real_filtered,
            cap_data=cap_filtered,
            dist_data=dist_filtered,
            demand_data=demand_filtered,
            model_names=model_names_filtered,
            wp=0.6, wd=0.3, wq=0.1,
            verbose=verbose
        )
        
        # 5. 对比原始价格与求解价格
        if verbose and solve_results['successful_days'] > 0:
            print(f"\n=== 第三步：价格对比分析 ===")
            print("对比原始市场价格与反向求解价格:")
            
            # 计算平均价格
            original_avg_prices = p_nom_filtered.mean(dim=(0, 1))
            solved_avg_prices = solved_prices.mean(dim=(0, 1))
            
            print(f"{'模型名称':<20} {'原始价格':<12} {'求解价格':<12} {'价格差异':<12} {'相对差异':<10}")
            print("-" * 70)
            
            for i, name in enumerate(model_names_filtered):
                orig_price = original_avg_prices[i].item()
                solved_price = solved_avg_prices[i].item()
                price_diff = solved_price - orig_price
                relative_diff = (price_diff / orig_price * 100) if orig_price > 0 else 0
                
                print(f"{name:<20} "
                      f"${orig_price:>7.4f}   "
                      f"${solved_price:>7.4f}   "
                      f"${price_diff:>+7.4f}   "
                      f"{relative_diff:>+6.1f}%")
            
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()

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
    
    # # 5. 市场份额分析
    # print(f"\n=== 市场份额分析 ===")
    # print(f"{'模型名称':<20} {'平均日流量':>12} {'市场份额':>10} {'最高日流量':>12} {'最低日流量':>12}")
    # print("-" * 70)
    
    # total_market = q_real_filtered.sum(dim=2).mean().item()  # 平均日总流量
    
    # for i, model_name in enumerate(model_names_filtered):
    #     avg_daily = q_real_filtered[0, :, i].mean().item()
    #     max_daily = q_real_filtered[0, :, i].max().item()
    #     min_daily = q_real_filtered[0, :, i].min().item()
    #     market_share = (avg_daily / total_market * 100) if total_market > 0 else 0
        
    #     print(f"{model_name:<20} "
    #           f"{avg_daily:>9.1f}M   "
    #           f"{market_share:>7.1f}%   "
    #           f"{max_daily:>9.1f}M   "
    #           f"{min_daily:>9.1f}M")
    
    # # 6. 时间序列趋势分析
    # print(f"\n=== 时间序列趋势 ===")
    # print("每日总流量变化:")
    # daily_totals = q_real_filtered.sum(dim=2)[0]  # 每日总流量
    # for day in range(num_days):
    #     trend = ""
    #     if day > 0:
    #         change = daily_totals[day] - daily_totals[day-1]
    #         if change > 0:
    #             trend = f" (↑{change.item():.1f})"
    #         elif change < 0:
    #             trend = f" (↓{abs(change.item()):.1f})"
    #         else:
    #             trend = " (=)"
    #     print(f"  第{day+1}天 ({dates[day]}): {daily_totals[day].item():.1f}M tokens{trend}")

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
    
    args = parser.parse_args()
    
    # 验证日期格式
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        print("错误：日期格式应为 YYYY-MM-DD")
        return
    
    # 检查是否运行反向价格求解演示
    if args.demo_reverse_pricing:
        print("=== 运行反向价格求解演示 ===")
        demo_reverse_price_solving(
            start_date=args.start_date,
            num_days=args.num_days,
            data_file=args.data_file,
            min_price_threshold=args.min_price_threshold,
            verbose=True
        )
        return
    
    # 训练模型
    try:
        trained_model, model_names, history = train_quality_learner_with_real_data(
            start_date=args.start_date,
            num_days=args.num_days,
            wp=args.wp,
            wd=args.wd,
            wq=args.wq,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            data_file=args.data_file,
            min_price_threshold=args.min_price_threshold,
            verbose=True
        )
        
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
        
        try:
            if choice == "2":
                print("\n=== 反向价格求解演示 ===")
                demo_reverse_price_solving(
                    start_date=demo_start_date,
                    num_days=demo_num_days,
                    verbose=True
                )
            elif choice == "3":
                print("\n=== 完整功能演示 ===")
                
                # 先运行质量学习器训练
                print("\n第一部分：训练质量学习器")
                trained_model, model_names, history = train_quality_learner_with_real_data(
                    start_date=demo_start_date,
                    num_days=demo_num_days,
                    num_epochs=100,  # 演示用较少epoch
                    min_price_threshold=0.0,
                    verbose=True
                )
                
                # 然后运行反向价格求解
                print("\n第二部分：反向价格求解分析")
                demo_reverse_price_solving(
                    start_date=demo_start_date,
                    num_days=demo_num_days,
                    verbose=True
                )
                
                print("\n=== 完整演示完成 ===")
                
            else:  # choice == "1" 或其他
                print("\n=== 质量学习器训练演示 ===")
                trained_model, model_names, history = train_quality_learner_with_real_data(
                    start_date=demo_start_date,
                    num_days=demo_num_days,
                    num_epochs=200,
                    min_price_threshold=0.0,
                    verbose=True
                )
                print("\n演示训练完成！")
                
                # 演示额外的预测功能
                print(f"\n=== 演示预测功能 ===")
                try:
                    # 重新加载数据进行预测演示
                    from priLLM.calibration.real_data import load_real_market_data_for_training
                    p_nom_data, cap_data, dist_data, demand_data, q_real_data, all_model_names = \
                        load_real_market_data_for_training(demo_start_date, demo_num_days)
                    
                    # 筛选与训练模型对应的数据
                    avg_prices = p_nom_data.mean(dim=(0, 1))
                    valid_indices = torch.where(avg_prices > args.min_price_threshold)[0]
                    
                    p_nom_demo = p_nom_data[:, :, valid_indices]
                    cap_demo = cap_data[:, :, valid_indices]
                    dist_demo = dist_data[:, :, :, valid_indices]
                    q_real_demo = q_real_data[:, :, valid_indices]
                    demand_demo = q_real_demo.sum(dim=2, keepdim=True)
                    
                    # 使用训练好的模型进行预测，并对比真实数据
                    predict_market_share(
                        trained_model, p_nom_demo, cap_demo, dist_demo, demand_demo,
                        model_names, q_real=q_real_demo, verbose=True
                    )
                    
                except Exception as e:
                    print(f"预测演示失败: {e}")
                
                # 可以添加额外的分析和可视化
                print(f"\n最终训练损失: {history['losses'][-1]:.6f}")
            
        except Exception as e:
            print(f"演示失败: {e}")
            print("\n请检查数据文件是否存在，或尝试其他日期范围。")
            print("示例用法：")
            print("python train_with_real_data.py --start_date 2024-12-01 --num_days 7 --min_price_threshold 0.0")
            print("python train_with_real_data.py --start_date 2024-12-01 --num_days 7 --demo_reverse_pricing")
    else:
        main()
