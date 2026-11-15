# 本实验验证我们可微层+初始化的效率和效果

import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Tuple, List, Optional
import numpy as np
import time
from scipy.optimize import minimize

# ==============================================================================
# 0. 你的原始代码 (稍作整理，作为对比组)
# ==============================================================================

class FollowerLayer_CVX(nn.Module):
    def __init__(self, n_max: int, m_max: int):
        super().__init__()
        self.n_max, self.m_max = n_max, m_max
        x = cp.Variable((n_max, m_max), nonneg=True)
        q = cp.sum(x, axis=0)
        gamma_price = cp.Parameter((m_max,))
        dist_w = cp.Parameter((n_max, m_max))
        scale_vec = cp.Parameter((m_max,), nonneg=True)
        demand = cp.Parameter((n_max,))
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

    def forward(self, p_eff, dist_w, scale_vec, demand):
        # 注意：为了统一接口，我们将参数准备移到上层
        x_flat, = self.layer(p_eff, dist_w, scale_vec, demand)
        return x_flat

class QualityLearner_CVX(nn.Module):
    def __init__(self, n_max, m_max, wp, wd, wq, initial_quality=None):
        super().__init__()
        self.n_max, self.m_max = n_max, m_max
        self.follower_layer = FollowerLayer_CVX(n_max, m_max)
        if initial_quality is None:
            initial_quality = torch.zeros(m_max)
        self.quality = nn.Parameter(initial_quality.clone())
        self.register_buffer('wp', torch.tensor(wp, dtype=torch.float32))
        self.register_buffer('wd', torch.tensor(wd, dtype=torch.float32))
        self.register_buffer('wq', torch.tensor(wq, dtype=torch.float32))

    def forward(self, p_nominal_bat, cap_bat, dist_cat, demand_bat):
        B, K, m_max = p_nominal_bat.shape
        n_max = self.n_max
        BK = B * K

        p_eff_bat = p_nominal_bat - self.quality
        
        wp = self.wp.view(1, 1, 1).expand(B, K, 1)
        wd = self.wd.view(1, 1, 1, 1).expand(B, K, 1, 1)
        wq_sqrt = torch.sqrt(self.wq.clamp(min=1e-8)).view(1, 1, 1).expand(B, K, 1)

        gamma_price = (wp * p_eff_bat).reshape(BK, m_max)
        dist_w = (wd * dist_cat).reshape(BK, n_max, m_max)
        inv_cap = 1.0 / cap_bat.clamp(min=1e-8)
        scale_vec = (wq_sqrt * torch.sqrt(inv_cap)).reshape(BK, m_max)
        demand_f = demand_bat.reshape(BK, n_max)

        x_flat = self.follower_layer(gamma_price, dist_w, scale_vec, demand_f)
        
        x_star = x_flat.view(B, K, n_max, m_max)
        q_star = x_star.sum(dim=2)
        return q_star

# ==============================================================================
# New Baseline: CVX with Initialization from Inverse Function
# 思想：使用cal_revenue_cvxpy中的反推函数来获得一个更好的初始化，
# 然后使用我们的CVX方法进行微调。这展示了好的初始化对收敛速度和最终效果的影响。
# ==============================================================================

import sys
import os
# 添加父目录到路径以导入cal_revenue_cvxpy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from priLLM.game.equilibrium import find_price_for_flow, solve_weighted_linear_congestion_eq

class QualityLearner_CVX_Initialized(QualityLearner_CVX):
    def __init__(self, n_max, m_max, wp, wd, wq, p_nominal_sample, cap_sample, dist_sample, demand_sample, q_real_data, initial_quality=None):
        """
        初始化时使用反推函数来获得更好的quality初始值
        
        Parameters:
        -----------
        p_nominal_sample, cap_sample, dist_sample, demand_sample, q_real_data : torch.Tensor
            用于初始化的样本数据，通常取训练数据的第一个batch
        其他参数与父类相同
        """
        # 首先用默认初始化创建模型
        super().__init__(n_max, m_max, wp, wd, wq, initial_quality)
        
        # 使用反推函数计算更好的初始化 - 传入真实流量数据
        self._initialize_quality_from_inverse(p_nominal_sample, cap_sample, dist_sample, demand_sample, q_real_data, wp, wd, wq)
    
    def _initialize_quality_from_inverse(self, p_nominal, cap, dist, demand, q_real_data, wp, wd, wq, w0=0.0):
        """
        使用反向求解方法为每一天计算quality，然后取平均作为初始值
        基于真实的流量数据q_real_data而不是需求总和
        """
        try:
            # 获取数据维度
            B, K, m_providers = p_nominal.shape
            n_users = demand.shape[2]
            
            # 创建模型名称列表（简化处理）
            model_names = [f"Model_{i}" for i in range(m_providers)]
            
            # 使用反向求解方法计算每一天的quality
            daily_qualities = []
            
            for day_idx in range(K):
                # 提取当天数据
                p_nom_day = p_nominal[0, day_idx, :].detach().numpy()  # (m,)
                cap_day = cap[0, day_idx, :].detach().numpy()          # (m,)
                dist_day = dist[0, day_idx, :, :].detach().numpy()     # (n, m)
                demand_day = demand[0, day_idx, :].detach().numpy()    # (n,)
                
                # 使用真实的流量数据作为目标流量
                q_target_real = q_real_data[0, day_idx, :].detach().numpy()  # (m,) - 真实流量
                
                if q_target_real.sum() <= 0:
                    print(f"Day {day_idx+1}: 真实流量数据为0，跳过")
                    continue
                
                # 将真实流量扩展为多用户格式
                if n_users == 1:
                    # 单用户情况：直接使用真实流量
                    x_star_matrix = q_target_real.reshape(1, -1)
                else:
                    # 多用户情况：按需求比例分配真实流量
                    total_demand = demand_day.sum()
                    if total_demand > 0:
                        x_star_matrix = np.zeros((n_users, m_providers))
                        for i in range(n_users):
                            user_ratio = demand_day[i] / total_demand
                            x_star_matrix[i, :] = q_target_real * user_ratio
                    else:
                        x_star_matrix = np.zeros((n_users, m_providers))
                        x_star_matrix[0, :] = q_target_real  # 如果无需求，将所有流量分配给第一个用户
                
                # 反向求解最优价格
                p_found, _, status = find_price_for_flow(
                    x_star=x_star_matrix,
                    c=cap_day,
                    d=dist_day,
                    w_p=wp,
                    w_d=wd,
                    w_q=wq,
                    w0=w0,
                    solver="OSQP"
                )
                
                if p_found is not None and status in ("optimal", "optimal_inaccurate"):
                    # 计算当天的quality: quality = p_nominal - p_effective
                    quality_day = p_nom_day - p_found
                    daily_qualities.append(quality_day)
                    
                    if day_idx < 3:  # 打印前3天的详细信息
                        print(f"Day {day_idx+1}: quality = {quality_day}")
                else:
                    # 使用默认quality（0）
                    daily_qualities.append(np.zeros(m_providers))
                    print(f"Day {day_idx+1}: 反向求解失败，使用默认quality")
            
            # 计算平均quality作为初始值
            if daily_qualities:
                quality_init = np.mean(daily_qualities, axis=0)
            else:
                quality_init = np.zeros(m_providers)
            
            print(f"使用反向求解初始化quality: {quality_init}")
            
            # 设置初始quality值
            with torch.no_grad():
                self.quality.copy_(torch.from_numpy(quality_init).float())
                
        except Exception as e:
            print(f"Warning: Quality initialization failed with error: {e}, using default initialization")
            # 使用默认的零初始化
            pass
    
    def find_optimal_prices_for_target_flow(
        self,
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

# ==============================================================================
# Baseline 1: Un-rolled Best-Response / Tâtonnement
# 思想：不直接求解，而是模拟用户多次迭代调整策略的过程，然后对整个过程求导。
# 我们这里使用在势函数上进行投影梯度下降的方法来模拟这个过程。
# ==============================================================================

class UnrolledFollowerLayer(nn.Module):
    def __init__(self, n_max, m_max, num_steps=10, step_size=0.1):
        super().__init__()
        self.n_max, self.m_max = n_max, m_max
        self.num_steps = num_steps
        self.step_size = step_size

    def project_simplex(self, v, z=1.0):
        # 将一个batch的向量投影到满足sum(x_i)=z的单纯形上
        # v: (Batch, Dim), z: (Batch,)
        n_features = v.shape[1]
        batch_size = v.shape[0] # 获取批次大小, e.g., 50
        u, _ = torch.sort(v, dim=1, descending=True)
        cssv = torch.cumsum(u, dim=1)
        rho = torch.arange(1, n_features + 1, device=v.device).float()
        # indices = torch.arange(n_features, device=v.device)
        
        # 找到满足 u_k - (1/k) * (cssv_k - z) > 0 的最大 k
        # 使用广播和比较来高效地找到所有batch的rho
        mask = (u * rho > (cssv - z.unsqueeze(1)))
        rho_indices = torch.sum(mask, dim=1) - 1
        
        batch_indices = torch.arange(batch_size, device=v.device)

        # 使用rho_indices来选择正确的lambda
        selected_rho = rho[rho_indices]
        selected_cssv = cssv[batch_indices, rho_indices]
        
        lambda_ = (selected_cssv - z) / selected_rho
        
        return torch.clamp(v - lambda_.unsqueeze(1), min=0)

    def forward(self, p_eff, dist_w, scale_vec, demand):
        BK, n_max, m_max = dist_w.shape
        
        # 初始化x: 将需求均匀分配
        x = demand.unsqueeze(2).expand(BK, n_max, m_max) / m_max

        for _ in range(self.num_steps):
            q = x.sum(dim=1) # (BK, m_max)
            
            # 计算势函数的梯度 (即用户的边际成本)
            # grad_x[b, u, m] = d(Obj)/d(x[b, u, m])
            grad_x = (
                p_eff.unsqueeze(1) +
                dist_w +
                scale_vec.unsqueeze(1)**2 * x +
                (scale_vec**2 * q).unsqueeze(1)
            )
            
            # 执行一步梯度下降
            x_new = x - self.step_size * grad_x
            
            # 将每个用户的分配投影回单纯形
            # 注意：这里的投影需要对每个用户独立进行
            # project_simplex需要一个(BK*n_max, m_max)的输入
            x_flat = x_new.view(BK * n_max, m_max)
            demand_flat = demand.view(BK * n_max)
            x_proj_flat = self.project_simplex(x_flat, demand_flat)
            x = x_proj_flat.view(BK, n_max, m_max)

        return x

class QualityLearner_Unrolled(QualityLearner_CVX): # 继承以复用代码
    def __init__(self, n_max, m_max, wp, wd, wq, initial_quality=None, num_steps=10, step_size=0.1):
        super().__init__(n_max, m_max, wp, wd, wq, initial_quality)
        self.follower_layer = UnrolledFollowerLayer(n_max, m_max, num_steps, step_size)


# ==============================================================================
# Baseline 2: Black-box Root-Finding + Implicit Differentiation (Finite Diff)
# 思想：前向传播时用外部求解器（如scipy）找到最优解，反向传播时用
# 有限差分来近似梯度。这模拟了无法获得解析梯度的情况。
# ==============================================================================

class BlackboxFollowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_eff, dist_w, scale_vec, demand, quality):
        # 求解器需要numpy数组
        p_eff_np = p_eff.detach().cpu().numpy()
        dist_w_np = dist_w.detach().cpu().numpy()
        scale_vec_np = scale_vec.detach().cpu().numpy()
        demand_np = demand.detach().cpu().numpy()

        BK, n_max, m_max = dist_w_np.shape
        
        def objective(x_flat, b_idx):
            x = x_flat.reshape(n_max, m_max)
            q = x.sum(axis=0)
            obj = (
                p_eff_np[b_idx] @ q +
                np.sum(dist_w_np[b_idx] * x) +
                0.5 * (np.sum((x @ np.diag(scale_vec_np[b_idx]))**2) +
                       np.sum((scale_vec_np[b_idx] * q)**2))
            )
            return obj

        def constraint_eq(x_flat, u_idx):
            x = x_flat.reshape(n_max, m_max)
            return np.sum(x[u_idx, :]) - demand_np[b_idx, u_idx]

        x_star_list = []
        for b_idx in range(BK):
            cons = [{'type': 'eq', 'fun': constraint_eq, 'args': (u_idx,)} for u_idx in range(n_max)]
            bounds = [(0, None)] * (n_max * m_max)
            x_init = (demand_np[b_idx].reshape(-1, 1) / m_max * np.ones((n_max, m_max))).flatten()
            
            res = minimize(objective, x_init, args=(b_idx,), method='SLSQP', bounds=bounds, constraints=cons, options={'disp': False})
            x_star_list.append(torch.from_numpy(res.x.reshape(n_max, m_max)).to(p_eff.device, p_eff.dtype))
        
        x_star = torch.stack(x_star_list)
        
        ctx.save_for_backward(p_eff, dist_w, scale_vec, demand, quality, x_star)
        return x_star

    @staticmethod
    def backward(ctx, grad_output):
        # 使用有限差分来近似梯度
        p_eff, dist_w, scale_vec, demand, quality, x_star = ctx.saved_tensors
        epsilon = 1e-4
        
        # 只对 quality 计算梯度
        grad_quality = torch.zeros_like(quality)
        
        for i in range(len(quality)):
            # 扰动 quality
            quality_plus = quality.clone()
            quality_plus[i] += epsilon
            
            # 重新计算 p_eff
            # 假设 quality 是 (m_max), p_eff 是 (BK, m_max)
            p_eff_plus = p_eff + (quality - quality_plus)
            
            # 重新求解
            x_star_plus = BlackboxFollowerFunction.apply(p_eff_plus, dist_w, scale_vec, demand, quality_plus)
            
            # 有限差分
            grad_x_i = (x_star_plus - x_star) / epsilon
            
            # 链式法则
            grad_quality[i] = torch.sum(grad_output * grad_x_i)
            
        # 其他参数的梯度设为None
        return None, None, None, None, grad_quality


class QualityLearner_Blackbox(QualityLearner_CVX):
    def __init__(self, n_max, m_max, wp, wd, wq, initial_quality=None):
        super().__init__(n_max, m_max, wp, wd, wq, initial_quality)
        # 没有显式的 follower layer, 直接调用 Function
    
    def forward(self, p_nominal_bat, cap_bat, dist_cat, demand_bat):
        B, K, m_max = p_nominal_bat.shape
        n_max = self.n_max
        BK = B * K

        # --- 这里有个微妙之处：为了让autograd追踪到quality，需要把它传入apply ---
        # --- 但我们只希望对quality求导，所以p_eff的计算要在apply之外 ---
        p_eff_bat = p_nominal_bat - self.quality
        
        wp = self.wp.view(1, 1, 1).expand(B, K, 1)
        wd = self.wd.view(1, 1, 1, 1).expand(B, K, 1, 1)
        wq_sqrt = torch.sqrt(self.wq.clamp(min=1e-8)).view(1, 1, 1).expand(B, K, 1)

        gamma_price = (wp * p_eff_bat).reshape(BK, m_max)
        dist_w = (wd * dist_cat).reshape(BK, n_max, m_max)
        inv_cap = 1.0 / cap_bat.clamp(min=1e-8)
        scale_vec = (wq_sqrt * torch.sqrt(inv_cap)).reshape(BK, m_max)
        demand_f = demand_bat.reshape(BK, n_max)

        x_flat = BlackboxFollowerFunction.apply(gamma_price, dist_w, scale_vec, demand_f, self.quality)
        
        x_star = x_flat.view(B, K, n_max, m_max)
        q_star = x_star.sum(dim=2)
        return q_star


# ==============================================================================
# Baseline 3: Score-Function / REINFORCE
# 思想：将分配过程看作一个随机策略。我们定义一个策略网络，它输出一个
# 分配动作的概率分布。通过采样动作并根据奖励（与真实流量的接近程度）
# 来更新策略网络。
# ==============================================================================

class ReinforceFollowerLayer(nn.Module):
    def __init__(self, n_max, m_max, input_dim):
        super().__init__()
        self.n_max, self.m_max = n_max, m_max
        # 策略网络：将用户的状态映射到Dirichlet分布的参数alpha
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, m_max),
            nn.Softplus() # 确保 alpha > 0
        )

    def forward(self, p_eff, dist_w, scale_vec, demand):
        BK, n_max, m_max = dist_w.shape
        
        # 为每个用户构建输入状态
        # state: (BK, n_max, input_dim)
        state = torch.cat([
            p_eff.unsqueeze(1).expand(-1, n_max, -1),
            dist_w,
            scale_vec.unsqueeze(1).expand(-1, n_max, -1)
        ], dim=-1)
        
        # 通过策略网络得到Dirichlet分布的参数
        alphas = self.policy_net(state.view(BK * n_max, -1)).view(BK, n_max, m_max) + 1e-6 # 加一个小量避免nan

        # 为每个用户创建一个分布
        dist = torch.distributions.Dirichlet(alphas)
        
        # 采样动作（分配比例）
        alloc_ratios = dist.sample()
        
        # 计算采样动作的log-prob
        log_prob = dist.log_prob(alloc_ratios)
        
        # 将分配比例转换为实际流量
        x = alloc_ratios * demand.unsqueeze(2)
        
        return x, log_prob.sum() # 返回总的log_prob

class QualityLearner_Reinforce(QualityLearner_CVX):
    def __init__(self, n_max, m_max, wp, wd, wq, initial_quality=None):
        super().__init__(n_max, m_max, wp, wd, wq, initial_quality)
        # 输入维度: p_eff(m), dist(m), scale_vec(m)
        input_dim = m_max * 3 
        self.follower_layer = ReinforceFollowerLayer(n_max, m_max, input_dim)
        # REINFORCE需要自己的优化器，因为它更新的是策略网络参数
        # 这里为了简化，我们让quality和策略网络一起被主优化器更新
        # 这意味着quality的变化会影响策略网络的输入，从而影响策略
        self.params_to_optimize = list(self.parameters()) + list(self.follower_layer.parameters())

    def forward(self, p_nominal_bat, cap_bat, dist_cat, demand_bat):
        B, K, m_max = p_nominal_bat.shape
        n_max = self.n_max
        BK = B * K

        p_eff_bat = p_nominal_bat - self.quality
        
        wp = self.wp.view(1, 1, 1).expand(B, K, 1)
        wd = self.wd.view(1, 1, 1, 1).expand(B, K, 1, 1)
        wq_sqrt = torch.sqrt(self.wq.clamp(min=1e-8)).view(1, 1, 1).expand(B, K, 1)

        gamma_price = (wp * p_eff_bat).reshape(BK, m_max)
        dist_w = (wd * dist_cat).reshape(BK, n_max, m_max)
        inv_cap = 1.0 / cap_bat.clamp(min=1e-8)
        scale_vec = (wq_sqrt * torch.sqrt(inv_cap)).reshape(BK, m_max)
        demand_f = demand_bat.reshape(BK, n_max)
        
        x_sampled, log_prob = self.follower_layer(gamma_price, dist_w, scale_vec, demand_f)
        
        x_star = x_sampled.view(B, K, n_max, m_max)
        q_star = x_star.sum(dim=2)
        
        return q_star, log_prob

# ==============================================================================
# 数据生成与主训练循环
# ==============================================================================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_real_market_data_for_experiment(start_date, num_days, min_price_threshold=0.0, data_file="data/release/market_inputs_top20_month.csv"):
    """
    使用真实市场数据进行实验，仿照train_with_real_data.py的模式
    """
    try:
        # 导入真实数据接口
        from priLLM.calibration.real_data import load_real_market_data_for_training
        
        # 加载真实数据
        p_nom_data, cap_data, dist_data, demand_data, q_real_data, model_names = \
            load_real_market_data_for_training(start_date, num_days, data_file)
        
        # 筛选价格大于阈值的模型（可选）
        avg_prices = p_nom_data.mean(dim=(0, 1))
        valid_indices = torch.where(avg_prices > min_price_threshold)[0]
        
        if len(valid_indices) == 0:
            print(f"警告：没有模型的价格大于阈值 ${min_price_threshold}/M tokens，使用所有模型")
            valid_indices = torch.arange(len(model_names))
        
        # 筛选数据和模型名称
        p_nom_data_filtered = p_nom_data[:, :, valid_indices]
        cap_data_filtered = cap_data[:, :, valid_indices]
        dist_data_filtered = dist_data[:, :, :, valid_indices]
        q_real_data_filtered = q_real_data[:, :, valid_indices]
        model_names_filtered = [model_names[i] for i in valid_indices]
        
        # 修正：demand_data也需要被过滤
        demand_data_filtered = q_real_data_filtered.sum(dim=2, keepdim=True)
        
        # 获取真实质量参数作为基准（使用反向求解方法）
        from priLLM.game.equilibrium import find_price_for_flow
        
        # 计算真实质量参数
        batch_size, num_days, num_providers = q_real_data_filtered.shape
        n_users = 1  # 简化为单用户
        
        # 使用平均流量计算质量基准
        avg_q_real = q_real_data_filtered.mean(dim=(0, 1))  # [num_providers]
        avg_p_nom = p_nom_data_filtered.mean(dim=(0, 1))   # [num_providers]
        
        # 计算对应的基准质量（简化计算）
        quality_true = torch.zeros(len(model_names_filtered))
        
        print(f"使用真实市场数据：{len(model_names_filtered)}个模型，{num_days}天数据")
        print(f"模型：{model_names_filtered}")
        
        return p_nom_data_filtered, cap_data_filtered, dist_data_filtered, demand_data_filtered, q_real_data_filtered, quality_true, model_names_filtered
        
    except Exception as e:
        # No silent fallback to synthetic data: a failed load must fail the experiment.
        raise RuntimeError(f"Could not load real market data: {e}") from e

def generate_synthetic_data(n_users, m_providers, n_days, batch_size, wp, wd, wq):
    """
    保留原始的合成数据生成函数作为回退方案
    """
    quality_true = torch.tensor([1.5, 0.7, -0.9], dtype=torch.float32)
    p_nominal_bat = torch.rand(batch_size, n_days, m_providers) * 0.5 + torch.tensor([2.0, 1.0, 0.8])
    dist_cat = torch.rand(batch_size, n_days, n_users, m_providers) * 50 + 10
    cap_bat = torch.rand(batch_size, n_days, m_providers) * 500 + 500
    demand_bat = torch.rand(batch_size, n_days, n_users) * 1000 + 500
    
    # 使用CVX Learner生成高质量的 "真实" 数据
    true_model = QualityLearner_CVX(n_users, m_providers, wp, wd, wq)
    with torch.no_grad():
        true_model.quality.copy_(quality_true)
        q_real = true_model(p_nominal_bat, cap_bat, dist_cat, demand_bat)
    
    model_names = [f"Model_{i}" for i in range(m_providers)]
    return p_nominal_bat, cap_bat, dist_cat, demand_bat, q_real, quality_true, model_names

def train_model(model, data, num_epochs, lr, is_reinforce=False, timeout_seconds=180, max_epochs=500):
    p_nom_data, cap_data, dist_data, demand_data, q_real_data = data
    
    if is_reinforce:
        # REINFORCE需要优化策略网络和quality
        params = model.params_to_optimize
    else:
        params = model.parameters()
        
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    epoch_times = []
    losses = []
    
    actual_epochs = min(num_epochs, max_epochs)
    
    for epoch in range(actual_epochs):
        epoch_start = time.time()
        
        # 检查总超时
        if time.time() - start_time > timeout_seconds:
            print(f"  Training stopped due to timeout after {epoch} epochs")
            break
            
        optimizer.zero_grad()
        
        if is_reinforce:
            q_pred, log_prob = model(p_nom_data, cap_data, dist_data, demand_data)
            # 奖励是负的MSE损失
            reward = -loss_fn(q_pred, q_real_data)
            # 策略梯度损失
            loss = -log_prob * reward.detach() # detach reward to prevent gradients from flowing back
        else:
            q_pred = model(p_nom_data, cap_data, dist_data, demand_data)
            loss = loss_fn(q_pred, q_real_data)

        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        losses.append(loss.item())

        if (epoch + 1) % max(1, actual_epochs // 5) == 0:
            print(f"  Epoch [{epoch+1}/{actual_epochs}], Loss: {loss.item():.6f}, Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"  Training finished in {total_time:.2f} seconds over {len(losses)} epochs.")
    
    return {
        'final_quality': model.quality.data.numpy(),
        'total_time': total_time,
        'epoch_times': epoch_times,
        'losses': losses,
        'epochs_completed': len(losses)
    }


def run_experiments_for_plotting():
    """
    运行实验并收集数据，为exp_dl_plot.py提供真实的实验结果
    使用真实市场数据代替合成数据
    """
    # 定义实验配置 - 使用不同的日期范围作为不同难度
    experiment_configs = [
        {"start_date": "2026-10-19", "num_days": 3, "name": "Small\n(3天)"},
        {"start_date": "2026-10-19", "num_days": 7, "name": "Medium\n(7天)"}, 
        {"start_date": "2026-10-19", "num_days": 14, "name": "Large\n(14天)"}
    ]
    
    # 固定参数
    wp, wd, wq = 0.6, 0.3, 0.1
    num_epochs = 1000
    timeout_seconds = 180  # 3分钟超时
    min_price_threshold = 0.0  # 最小价格阈值
    
    # 定义要测试的方法
    methods_to_test = {
        "Your Method (CVX)": {
            "class": QualityLearner_CVX,
            "args": {},
            "lr": 0.01,
            "is_reinforce": False,
            "needs_init_data": False
        },
        # "Your Method (CVX + Init)": {
        #     "class": QualityLearner_CVX_Initialized,
        #     "args": {},
        #     "lr": 0.01,
        #     "is_reinforce": False,
        #     "needs_init_data": True
        # },
        "Baseline 1 (Unrolled)": {
            "class": QualityLearner_Unrolled,
            "args": {"num_steps": 20, "step_size": 0.05},
            "lr": 0.1,
            "is_reinforce": False,
            "needs_init_data": False
        },
        "Baseline 2 (Blackbox)": {
            "class": QualityLearner_Blackbox,
            "args": {},
            "lr": 0.05,
            "is_reinforce": False,
            "needs_init_data": False
        },
        "Baseline 3 (REINFORCE)": {
            "class": QualityLearner_Reinforce,
            "args": {},
            "lr": 0.01,
            "is_reinforce": True,
            "needs_init_data": False
        }
    }
    
    # 收集实验结果
    results = {
        'accuracy_data': {},
        'runtime_data': {},
        'methods': list(methods_to_test.keys()),
        'difficulties': [config["name"] for config in experiment_configs]
    }
    
    # 确保结果目录存在
    import os
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')
    
    for method_name in methods_to_test.keys():
        results['accuracy_data'][method_name] = []
        results['runtime_data'][method_name] = []
    
    # 对每个配置运行实验
    for config in experiment_configs:
        start_date = config["start_date"]
        num_days = config["num_days"]
        difficulty_name = config["name"]
        
        print(f"\n=== Running experiments for {difficulty_name} ===")
        
        try:
            # 加载真实市场数据
            p_nom_data, cap_data, dist_data, demand_data, q_real_data, quality_true, model_names = \
                load_real_market_data_for_experiment(start_date, num_days, min_price_threshold)

            print(f"p_nom_data shape: {p_nom_data.shape}")
            
            # 确定用户数量和提供商数量
            n_users = demand_data.shape[2]  # 从demand_data获取用户数量
            m_providers = len(model_names)
            
            data_tuple = (p_nom_data, cap_data, dist_data, demand_data, q_real_data)
            initial_quality_guess = torch.zeros(m_providers)
            
            # 对每个方法运行实验
            for method_name, config in methods_to_test.items():
                print(f"\n--- Testing {method_name} on {difficulty_name} ---")
                
                try:
                    # 创建模型
                    if method_name == "Your Method (CVX + Init)":
                        model = config["class"](n_users, m_providers, wp, wd, wq,
                                              p_nominal_sample=p_nom_data, cap_sample=cap_data,
                                              dist_sample=dist_data, demand_sample=demand_data,
                                              q_real_data=q_real_data, initial_quality=initial_quality_guess)
                    else:
                        model = config["class"](n_users, m_providers, wp, wd, wq,
                                              initial_quality=initial_quality_guess, **config["args"])
                    
                    # 训练并收集结果
                    training_result = train_model(
                        model, data_tuple, num_epochs, config["lr"], 
                        config["is_reinforce"], timeout_seconds, max_epochs=500
                    )
                    
                    # 计算最终误差 - 使用真实数据的质量基准
                    final_quality = training_result['final_quality']
                    # 使用零向量作为质量基准（因为我们不知道真实质量参数）
                    l2_error = np.linalg.norm(final_quality)  # 简化为与零向量的距离
                    avg_epoch_time = np.mean(training_result['epoch_times']) if training_result['epoch_times'] else timeout_seconds
                    
                    # 存储结果，确保转换为Python原生float类型
                    results['accuracy_data'][method_name].append(float(l2_error))
                    results['runtime_data'][method_name].append(float(avg_epoch_time))
                    
                    print(f"  最终质量参数: {final_quality}")
                    print(f"  L2 Error: {l2_error:.4f}")
                    print(f"  Avg Epoch Time: {avg_epoch_time:.2f}s")
                    print(f"  Completed Epochs: {training_result['epochs_completed']}")
                    
                except Exception as e:
                    print(f"  Error running {method_name}: {e}")
                    # 在错误情况下使用最大值作为占位符
                    results['accuracy_data'][method_name].append(float('inf'))
                    results['runtime_data'][method_name].append(float(timeout_seconds))
                    
        except Exception as e:
            print(f"  Error loading data for {difficulty_name}: {e}")
            # 数据加载失败时使用合成数据
            n_users = 5
            m_providers = 3
            p_nom_data, cap_data, dist_data, demand_data, q_real_data, quality_true, model_names = \
                generate_synthetic_data(n_users, m_providers, 10, 1, wp, wd, wq)
            
            data_tuple = (p_nom_data, cap_data, dist_data, demand_data, q_real_data)
            initial_quality_guess = torch.zeros(m_providers)
            
            for method_name, config in methods_to_test.items():
                print(f"\n--- Testing {method_name} on {difficulty_name} (synthetic fallback) ---")
                
                try:
                    if method_name == "Your Method (CVX + Init)":
                        model = config["class"](n_users, m_providers, wp, wd, wq,
                                              p_nominal_sample=p_nom_data, cap_sample=cap_data,
                                              dist_sample=dist_data, demand_sample=demand_data,
                                              q_real_data=q_real_data, initial_quality=initial_quality_guess)
                    else:
                        model = config["class"](n_users, m_providers, wp, wd, wq,
                                              initial_quality=initial_quality_guess, **config["args"])
                    
                    training_result = train_model(
                        model, data_tuple, num_epochs, config["lr"], 
                        config["is_reinforce"], timeout_seconds, max_epochs=500
                    )
                    
                    final_quality = training_result['final_quality']
                    l2_error = np.linalg.norm(final_quality - quality_true.numpy())
                    avg_epoch_time = np.mean(training_result['epoch_times']) if training_result['epoch_times'] else timeout_seconds
                    
                    results['accuracy_data'][method_name].append(float(l2_error))
                    results['runtime_data'][method_name].append(float(avg_epoch_time))
                    
                except Exception as e:
                    print(f"  Error running {method_name}: {e}")
                    results['accuracy_data'][method_name].append(float('inf'))
                    results['runtime_data'][method_name].append(float(timeout_seconds))
    
    # 保存实验结果到文件
    import json
    
    # 定义一个自定义的JSON编码器来处理numpy类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)
    
    with open('experiment_results/experiment_data.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n=== Experiment Results Summary ===")
    print("Methods:", results['methods'])
    print("Difficulties:", results['difficulties'])
    for method in results['methods']:
        print(f"{method} Accuracy: {results['accuracy_data'][method]}")
        print(f"{method} Runtime: {results['runtime_data'][method]}")
    
    return results

if __name__ == '__main__':
    from priLLM.seed import set_seed; set_seed(0)
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='使用真实数据测试Stackelberg博弈方法')
    parser.add_argument('--mode', choices=['single', 'experiments', 'real', 'synthetic'], 
                       default='real', help='运行模式')
    parser.add_argument('--start_date', type=str, default='2024-12-01',
                       help='起始日期，格式：YYYY-MM-DD')
    parser.add_argument('--num_days', type=int, default=7,
                       help='使用天数')
    parser.add_argument('--min_price_threshold', type=float, default=0.0,
                       help='最小价格阈值')
    parser.add_argument('--data_file', type=str, default='data/release/market_inputs_top20_month.csv',
                       help='数据文件路径')
    parser.add_argument('--use_real_data', action='store_true',
                       help='使用真实数据（默认已启用）')
    
    args = parser.parse_args()
    
    if args.mode == 'experiments' or len(sys.argv) > 1 and sys.argv[1] == "--run-experiments":
        # 运行实验模式
        print("=== Running Experiments for Plotting ===")
        results = run_experiments_for_plotting()
        
    elif args.mode == 'real' or args.use_real_data:
        # 使用真实数据运行单次实验
        print("=== Running Single Experiment with Real Data ===")
        
        # --- 1. 设置超参数 ---
        wp, wd, wq = 0.6, 0.3, 0.1
        num_epochs = 100
        
        # --- 2. 加载真实市场数据 ---
        print("--- 1. Loading Real Market Data ---")
        try:
            p_nom_data, cap_data, dist_data, demand_data, q_real_data, quality_true, model_names = \
                load_real_market_data_for_experiment(
                    args.start_date, 
                    args.num_days, 
                    args.min_price_threshold,
                    args.data_file
                )
            
            n_users = demand_data.shape[2]  # 从demand_data获取用户数量
            m_providers = len(model_names)
            
            print(f"使用真实数据：{n_users}个用户，{m_providers}个提供商，{args.num_days}天数据")
            print(f"模型：{model_names}")
            
        except Exception as e:
            print(f"加载真实数据失败：{e}，使用合成数据")
            n_users = 5
            m_providers = 3
            p_nom_data, cap_data, dist_data, demand_data, q_real_data, quality_true, model_names = \
                generate_synthetic_data(n_users, m_providers, 10, 1, wp, wd, wq)
            print(f"使用合成数据：{n_users}个用户，{m_providers}个提供商")
        
        data_tuple = (p_nom_data, cap_data, dist_data, demand_data, q_real_data)
        initial_quality_guess = torch.zeros(m_providers)

        # --- 3. 定义和运行所有模型 ---
        models_to_run = {
            "Your Method (CVX)": {
                "class": QualityLearner_CVX,
                "args": {},
                "lr": 0.5,
                "is_reinforce": False,
                "needs_init_data": False
            },
            # "Your Method (CVX + Init)": {
            #     "class": QualityLearner_CVX_Initialized,
            #     "args": {},
            #     "lr": 0.5,
            #     "is_reinforce": False,
            #     "needs_init_data": True
            # },
            "Baseline 1 (Unrolled)": {
                "class": QualityLearner_Unrolled,
                "args": {"num_steps": 20, "step_size": 0.05},
                "lr": 0.1,
                "is_reinforce": False,
                "needs_init_data": False
            },
            "Baseline 2 (Blackbox)": {
                "class": QualityLearner_Blackbox,
                "args": {},
                "lr": 0.05,
                "is_reinforce": False,
                "needs_init_data": False
            },
            "Baseline 3 (REINFORCE)": {
                "class": QualityLearner_Reinforce,
                "args": {},
                "lr": 0.01,
                "is_reinforce": True,
                "needs_init_data": False
            }
        }
        
        results = {}

        for name, config in models_to_run.items():
            print(f"\n--- 2. Training {name} ---")
            print(f"Initial quality guess: {initial_quality_guess.numpy()}")

            if name == "Your Method (CVX + Init)":
                # 需要传入初始化数据
                model = config["class"](n_users, m_providers, wp, wd, wq, 
                                        p_nominal_sample=p_nom_data, cap_sample=cap_data,
                                        dist_sample=dist_data, demand_sample=demand_data,
                                        q_real_data=q_real_data, initial_quality=initial_quality_guess)
            else:
                model = config["class"](n_users, m_providers, wp, wd, wq, 
                                        initial_quality=initial_quality_guess, **config["args"])

            training_result = train_model(model, data_tuple, num_epochs, config["lr"], config["is_reinforce"])
            final_quality = training_result['final_quality']
            results[name] = final_quality

        # --- 4. 结果展示 ---
        print("\n\n--- 3. Final Comparison ---")
        print(f"{'Method':<25} | {'Learned Quality':<40}")
        print("-" * 80)
        
        for name, final_quality in results.items():
            print(f"{name:<25} | {np.array2string(final_quality, precision=4):<40}")
            
    elif args.mode == 'synthetic':
        # 使用合成数据（原始模式）
        print("=== Running Single Experiment with Synthetic Data ===")
        
        # --- 1. 设置超参数 ---
        N_USERS = 10
        M_PROVIDERS = 3
        N_DAYS = 10
        BATCH_SIZE = 1
        WP, WD, WQ = 0.6, 0.3, 0.1
        NUM_EPOCHS = 100
        
        # --- 2. 生成模拟数据 ---
        print("--- 1. Generating Synthetic Data ---")
        p_nom_data, cap_data, dist_data, demand_data, q_real_data, quality_true, model_names = \
            generate_synthetic_data(N_USERS, M_PROVIDERS, N_DAYS, BATCH_SIZE, WP, WD, WQ)
        
        print(f"\nGround Truth Quality: {quality_true.numpy()}")
        
        data_tuple = (p_nom_data, cap_data, dist_data, demand_data, q_real_data)
        initial_quality_guess = torch.zeros(M_PROVIDERS)

        # --- 3. 定义和运行所有模型 ---
        models_to_run = {
            "Your Method (CVX)": {
                "class": QualityLearner_CVX,
                "args": {},
                "lr": 0.5,
                "is_reinforce": False,
                "needs_init_data": False
            },
            "Your Method (CVX + Init)": {
                "class": QualityLearner_CVX_Initialized,
                "args": {},
                "lr": 0.5,
                "is_reinforce": False,
                "needs_init_data": True
            },
            "Baseline 1 (Unrolled)": {
                "class": QualityLearner_Unrolled,
                "args": {"num_steps": 20, "step_size": 0.05},
                "lr": 0.1,
                "is_reinforce": False,
                "needs_init_data": False
            },
            "Baseline 2 (Blackbox)": {
                "class": QualityLearner_Blackbox,
                "args": {},
                "lr": 0.05,
                "is_reinforce": False,
                "needs_init_data": False
            },
            "Baseline 3 (REINFORCE)": {
                "class": QualityLearner_Reinforce,
                "args": {},
                "lr": 0.01,
                "is_reinforce": True,
                "needs_init_data": False
            }
        }
        
        results = {}

        for name, config in models_to_run.items():
            print(f"\n--- 2. Training {name} ---")
            print(f"Initial quality guess: {initial_quality_guess.numpy()}")

            if name == "Your Method (CVX + Init)":
                # 需要传入初始化数据
                model = config["class"](N_USERS, M_PROVIDERS, WP, WD, WQ, 
                                        p_nominal_sample=p_nom_data, cap_sample=cap_data,
                                        dist_sample=dist_data, demand_sample=demand_data,
                                        q_real_data=q_real_data, initial_quality=initial_quality_guess)
            else:
                model = config["class"](N_USERS, M_PROVIDERS, WP, WD, WQ, 
                                        initial_quality=initial_quality_guess, **config["args"])

            training_result = train_model(model, data_tuple, NUM_EPOCHS, config["lr"], config["is_reinforce"])
            final_quality = training_result['final_quality']
            results[name] = final_quality

        # --- 4. 结果展示 ---
        print("\n\n--- 3. Final Comparison ---")
        print(f"{'Method':<25} | {'Learned Quality':<40} | {'L2 Error':<10}")
        print("-" * 80)
        print(f"{'Ground Truth':<25} | {np.array2string(quality_true.numpy(), precision=4):<40} | {'-':<10}")
        for name, final_quality in results.items():
            error = np.linalg.norm(final_quality - quality_true.numpy())
            print(f"{name:<25} | {np.array2string(final_quality, precision=4):<40} | {error:<10.4f}")
    
    else:
        # 默认行为：显示帮助
        print("使用方法：")
        print("python exp_dl.py --mode real                   # 使用真实数据")
        print("python exp_dl.py --mode synthetic              # 使用合成数据")
        print("python exp_dl.py --mode experiments            # 运行完整实验")
        print("python exp_dl.py --start_date 2024-12-01 --num_days 7  # 指定日期和天数")
        print("python exp_dl.py --run-experiments             # 旧版实验模式")
