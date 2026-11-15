import pandas as pd
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import os


def read_market_table(data_file: str) -> pd.DataFrame:
    """Read a market table in either format and return the calibration format
    (Date, model_name, total_token_usage_M, output_speed, time_to_first_token [s], price [$/M tokens], elo_score).

    * calibration format: returned as is;
    * release format (data/release/market_inputs_top20_month.csv: date, app_id, model, tokens, tps, ttft_ms,
      price_prompt_per_mtok, price_completion_per_mtok, elo_score): apps are summed per (date, model); tps and ttft are
      the token-weighted means; price is the 3:1 input:output blend of the two listing prices.
    """
    if data_file.endswith('.pkl'):
        return pd.read_pickle(data_file)
    df = pd.read_csv(data_file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    df['Date'] = pd.to_datetime(df['date'])
    df['price'] = (3 * df['price_prompt_per_mtok'] + df['price_completion_per_mtok']) / 4
    df['w'] = df['tokens']
    df['tps_w'] = df['tps'] * df['w']
    df['ttft_w'] = df['ttft_ms'] / 1000.0 * df['w']
    g = df.groupby(['Date', 'model'], as_index=False).agg(
        tokens=('tokens', 'sum'), w=('w', 'sum'), tps_w=('tps_w', 'sum'), ttft_w=('ttft_w', 'sum'),
        price=('price', 'first'), elo_score=('elo_score', 'first'))
    out = pd.DataFrame({
        'Date': g['Date'],
        'total_token_usage_M': g['tokens'] / 1e6,
        'output_speed': g['tps_w'] / g['w'],
        'time_to_first_token': g['ttft_w'] / g['w'],
        'model_name': g['model'],
        'elo_score': g['elo_score'],
        'price': g['price'],
    })
    return out

def load_real_data(
    start_date: str, 
    num_days: int, 
    data_file: str = "data/release/market_inputs_top20_month.csv"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    从真实数据文件中加载指定日期范围的数据
    
    Args:
        start_date (str): 起始日期，格式为 'YYYY-MM-DD'
        num_days (int): 要加载的天数
        data_file (str): 数据文件路径
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        - p_nominal_bat: 名义价格 (batch_size=1, num_days, num_providers)
        - cap_bat: TPS容量 (batch_size=1, num_days, num_providers) 
        - dist_cat: TTFT延迟 (batch_size=1, num_days, num_users=1, num_providers)
        - demand_bat: 用户需求 (batch_size=1, num_days, num_users=1)
        - model_names: 模型名称列表
    """
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件 {data_file} 不存在！请先运行 load_process_data.py 生成数据。")
    
    # 加载数据
    print(f"正在加载数据文件: {data_file}")
    df = read_market_table(data_file)
    
    # 转换日期
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + timedelta(days=num_days - 1)
    
    print(f"筛选日期范围: {start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')}")
    
    # 筛选日期范围
    df_filtered = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy()
    
    if df_filtered.empty:
        raise ValueError(f"在指定日期范围 {start_date} 到 {end_dt.strftime('%Y-%m-%d')} 内没有找到数据！")
    
    # 获取所有可用的模型
    available_models = df_filtered['model_name'].unique()
    print(f"找到 {len(available_models)} 个模型: {list(available_models)}")
    
    # 筛选有完整数据的模型（价格、速度、TTFT都不为空）
    valid_models = []
    for model in available_models:
        model_data = df_filtered[df_filtered['model_name'] == model]
        # 检查是否有价格、速度和TTFT数据
        if (not model_data['price'].isna().all() and 
            not model_data['output_speed'].isna().all() and 
            not model_data['time_to_first_token'].isna().all()):
            valid_models.append(model)
    
    if len(valid_models) < 2:
        print(f"警告：只找到 {len(valid_models)} 个有完整数据的模型: {valid_models}")
        print("将使用所有可用模型，缺失数据将被填充...")
        valid_models = list(available_models)
    
    print(f"使用 {len(valid_models)} 个模型进行分析: {valid_models}")
    
    # 筛选模型数据
    df_models = df_filtered[df_filtered['model_name'].isin(valid_models)].copy()
    
    # 创建日期范围
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    # 初始化张量
    num_providers = len(valid_models)
    num_users = 1  # 简化为单用户场景
    batch_size = 1
    
    p_nominal_bat = torch.zeros(batch_size, num_days, num_providers)
    cap_bat = torch.zeros(batch_size, num_days, num_providers)
    dist_cat = torch.zeros(batch_size, num_days, num_users, num_providers)
    demand_bat = torch.zeros(batch_size, num_days, num_users)
    
    # 填充数据
    for day_idx, date in enumerate(date_range):
        day_data = df_models[df_models['Date'] == date]
        
        for provider_idx, model_name in enumerate(valid_models):
            model_day_data = day_data[day_data['model_name'] == model_name]
            
            if not model_day_data.empty:
                row = model_day_data.iloc[0]
                
                # 价格 (美元/百万token)
                price = row['price'] if not pd.isna(row['price']) else 1.0
                p_nominal_bat[0, day_idx, provider_idx] = price
                
                # TPS容量 (tokens/second)
                speed = row['output_speed'] if not pd.isna(row['output_speed']) else 100.0
                cap_bat[0, day_idx, provider_idx] = speed
                
                # TTFT延迟 (毫秒)
                ttft = row['time_to_first_token'] if not pd.isna(row['time_to_first_token']) else 500.0
                dist_cat[0, day_idx, 0, provider_idx] = ttft
                
            else:
                # 如果某天某个模型没有数据，使用默认值或前向填充
                if day_idx > 0:
                    # 前向填充
                    p_nominal_bat[0, day_idx, provider_idx] = p_nominal_bat[0, day_idx-1, provider_idx]
                    cap_bat[0, day_idx, provider_idx] = cap_bat[0, day_idx-1, provider_idx]
                    dist_cat[0, day_idx, 0, provider_idx] = dist_cat[0, day_idx-1, 0, provider_idx]
                else:
                    # 使用默认值
                    p_nominal_bat[0, day_idx, provider_idx] = 1.0
                    cap_bat[0, day_idx, provider_idx] = 100.0
                    dist_cat[0, day_idx, 0, provider_idx] = 500.0
        
        # 模拟用户需求 (可以根据实际token usage来调整)
        # 这里简化为固定需求，实际可以根据历史用量数据来设定
        daily_usage = df_models[df_models['Date'] == date]['total_token_usage_M'].sum()
        if pd.isna(daily_usage) or daily_usage == 0:
            daily_usage = 0  # 默认需求
        demand_bat[0, day_idx, 0] = daily_usage
    
    # 数据验证和清理
    print("\n数据统计:")
    print(f"价格范围: {p_nominal_bat.min().item():.3f} - {p_nominal_bat.max().item():.3f} $/M tokens")
    print(f"TPS范围: {cap_bat.min().item():.1f} - {cap_bat.max().item():.1f} tokens/s")
    print(f"TTFT范围: {dist_cat.min().item():.1f} - {dist_cat.max().item():.1f} ms")
    print(f"需求范围: {demand_bat.min().item():.1f} - {demand_bat.max().item():.1f} M tokens")
    
    return p_nominal_bat, cap_bat, dist_cat, demand_bat, valid_models

def load_real_market_data_for_training(
    start_date: str,
    num_days: int,
    data_file: str = "data/release/market_inputs_top20_month.csv"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    加载真实市场数据并计算实际的流量分配用于训练
    
    Args:
        start_date (str): 起始日期
        num_days (int): 天数
        data_file (str): 数据文件路径
    
    Returns:
        Tuple包含:
        - p_nominal_bat: 名义价格
        - cap_bat: TPS容量  
        - dist_cat: TTFT延迟
        - demand_bat: 用户需求
        - q_real_bat: 实际流量分配 (从token usage计算得出)
        - model_names: 模型名称列表
    """
    
    # 首先加载基础数据
    p_nominal_bat, cap_bat, dist_cat, demand_bat, model_names = load_real_data(
        start_date, num_days, data_file
    )
    
    # 加载DataFrame来计算实际流量分配
    df = read_market_table(data_file)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + timedelta(days=num_days - 1)
    
    df_filtered = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
    df_models = df_filtered[df_filtered['model_name'].isin(model_names)].copy()
    
    # 计算实际流量分配
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    num_providers = len(model_names)
    batch_size = 1
    
    q_real_bat = torch.zeros(batch_size, num_days, num_providers)
    
    for day_idx, date in enumerate(date_range):
        day_data = df_models[df_models['Date'] == date]
        
        # 计算当天总的token使用量
        total_usage = day_data['total_token_usage_M'].sum()
        
        if total_usage > 0:
            # 按实际使用量计算流量分配比例
            for provider_idx, model_name in enumerate(model_names):
                model_usage = day_data[day_data['model_name'] == model_name]['total_token_usage_M'].sum()
                q_real_bat[0, day_idx, provider_idx] = model_usage
        else:
            # 如果没有使用量数据，使用简单的启发式方法
            # 基于价格和性能的简单分配
            for provider_idx, model_name in enumerate(model_names):
                # # 质量更高、价格更低的模型获得更多流量
                # price = p_nominal_bat[0, day_idx, provider_idx].item()
                # speed = cap_bat[0, day_idx, provider_idx].item()
                # ttft = dist_cat[0, day_idx, 0, provider_idx].item()
                
                # # 简单的效用函数：速度高、延迟低、价格低的模型更受欢迎
                # utility = speed / (price * ttft + 1e-6)
                q_real_bat[0, day_idx, provider_idx] = 0
    
    # 归一化流量分配（确保每天的总流量等于需求）
    for day_idx in range(num_days):
        day_total = q_real_bat[0, day_idx, :].sum()
        if day_total > 0:
            q_real_bat[0, day_idx, :] = q_real_bat[0, day_idx, :] * demand_bat[0, day_idx, 0] / day_total
    
    print(f"\n实际流量分配统计:")
    print(f"流量范围: {q_real_bat.min().item():.1f} - {q_real_bat.max().item():.1f} M tokens")
    print(f"平均每日总流量: {q_real_bat.sum(dim=2).mean().item():.1f} M tokens")
    
    return p_nominal_bat, cap_bat, dist_cat, demand_bat, q_real_bat, model_names

if __name__ == "__main__":
    # 测试接口
    try:
        # 测试加载2025年7月1日开始的7天数据
        start_date = "2026-10-26"
        num_days = 7
        
        print("=== 测试真实数据加载接口 ===")
        p_nom, cap, dist, demand, models = load_real_data(start_date, num_days)
        
        print(f"\n加载的数据形状:")
        print(f"价格: {p_nom.shape}")  
        print(f"容量: {cap.shape}")
        print(f"延迟: {dist.shape}")
        print(f"需求: {demand.shape}")
        print(f"模型数量: {len(models)}")
        
        print("\n=== 测试训练数据加载接口 ===")
        p_nom, cap, dist, demand, q_real, models = load_real_market_data_for_training(start_date, num_days)
        
        print(f"\n训练数据形状:")
        print(f"实际流量: {q_real.shape}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保数据文件存在，并检查日期范围是否有效。")
