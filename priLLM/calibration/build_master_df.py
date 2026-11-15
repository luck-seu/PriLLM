import pandas as pd
import glob
import os

def _get_model_group_info():
    """
    定义模型组信息，用于合并相关模型的数据
    返回: (model_groups, main_models)
    - model_groups: 字典，键为统一模型名，值为该组包含的所有模型名列表
    - main_models: 字典，键为统一模型名，值为该组的主模型名（用于获取speed和TTFT）
    """
    model_groups = {
        'gemini-2.5-flash-unified': [
            'gemini-2.5-flash',
            'gemini-2.5-flash-preview', 
            'gemini-2.5-flash-preview-0520'
        ],
        'gemini-2.5-pro-unified': [
            'gemini-2.5-pro',
            'gemini-2.5-pro-preview',
            'gemini-2.5-pro-preview-0506'
        ],
        'deepseek-r1-unified': [
            'deepseek-r1',
            'deepseek-r1_free',
            'deepseek-r1-0528-free',
        ],
        'deepseek-chat-v3-0324-unified': [
            'deepseek-chat-v3-0324',
            'deepseek-chat-v3-0324-free'
        ],
    }
    
    main_models = {
        'gemini-2.5-flash-unified': 'gemini-2.5-flash',
        'gemini-2.5-pro-unified': 'gemini-2.5-pro',
        'deepseek-r1-unified': 'deepseek-r1',
        'deepseek-chat-v3-0324-unified': 'deepseek-chat-v3-0324',
    }
    
    return model_groups, main_models

def _parse_model_name(filename):
    """
    一个辅助函数，用于从复杂的文件名中解析出标准的模型名称。
    例如: 'anthropic_claude-sonnet-4_daily_token_usage.csv' -> 'claude-sonnet-4'
          'claude-3-7-sonnet_output_speed.csv' -> 'claude-3.7-sonnet'
    """
    # 移除文件扩展名和指标后缀
    name = filename.replace('.csv', '')
    suffixes_to_remove = ['_output_speed', '_time_to_first_token', '_daily_token_usage']
    for suffix in suffixes_to_remove:
        name = name.replace(suffix, '')
    
    # 移除已知的提供商前缀
    prefixes_to_remove = ['anthropic_', 'google_', 'openai_', 'mistralai_', 'meta-llama_']
    for prefix in prefixes_to_remove:
        name = name.replace(prefix, '')
        
    # 处理一些特殊命名，例如 claude-4-sonnet-thinking -> claude-sonnet-4
    # 你可以根据需要扩展这里的规则
    if 'claude-4-sonnet-thinking' in name:
        return 'claude-sonnet-4' # 统一命名
    if 'claude-3-7-sonnet' in name:
        return 'claude-3.7-sonnet'
    
    # 处理 deepseek 模型的特殊命名
    if 'deepseek_deepseek-r1_free' in name:
        return 'deepseek-r1_free'
    if 'deepseek_deepseek-r1-0528_free' in name:
        return 'deepseek-r1-0528-free'
    if 'deepseek_deepseek-chat-v3-0324_free' in name:
        return 'deepseek-chat-v3-0324-free'
    if 'deepseek_deepseek-chat-v3-0324' in name:
        return 'deepseek-chat-v3-0324'
    if 'deepseek-v3-0324' in name:
        return 'deepseek-chat-v3-0324'
    if name == 'deepseek-r1':
        return 'deepseek-r1'
    
    if 'v3-0324' in name:
        return 'deepseek-chat-v3-0324'
    if name == 'gemini-2-0-flash':
        return 'gemini-2.0-flash-001'
    if name == 'gemini-2-0-flash-lite-001':
        return 'gemini-2.0-flash-lite-001'
    if name == 'gemini-2-5-flash':
        return 'gemini-2.5-flash'
    if name == 'gemini-2-5-flash-lite':
        return 'gemini-2.5-flash-lite'
    if name == 'gpt-4-1-mini':
        return 'gpt-4.1-mini'
    if name == 'gpt-4-1':
        return 'gpt-4.1'
    if name == 'llama-3-3-instruct-70b':
        return 'llama-3.3-70b-instruct'
    
    # 处理Gemini 2.5模型的特殊命名
    if 'gemini-2-5-flash-preview-0520' in name or 'gemini-2.5-flash-preview-0520' in name:
        return 'gemini-2.5-flash-preview-0520'
    if 'gemini-2-5-flash-preview' in name or 'gemini-2.5-flash-preview' in name:
        return 'gemini-2.5-flash-preview'
    if 'gemini-2-5-pro-preview-0506' in name or 'gemini-2.5-pro-preview-0506' in name:
        return 'gemini-2.5-pro-preview-0506'
    if 'gemini-2-5-pro-preview' in name or 'gemini-2.5-pro-preview' in name:
        return 'gemini-2.5-pro-preview'
    if 'gemini-2-5-pro' in name or 'gemini-2.5-pro' in name:
        return 'gemini-2.5-pro'
    if 'gemini-2-5-flash-lite' in name or 'gemini-2.5-flash-lite' in name:
        return 'gemini-2.5-flash-lite'
        
    return name

def _get_unified_model_name(model_name):
    """
    将模型名称映射到统一的模型组名称
    """
    model_groups, _ = _get_model_group_info()
    
    for unified_name, group_models in model_groups.items():
        if model_name in group_models:
            return unified_name
    
    return model_name  # 如果不在任何组中，返回原名称

def load_and_preprocess_data(data_directory='.'):
    """
    加载并预处理目录中的所有模型数据文件。

    Args:
        data_directory (str): 存放CSV文件的目录路径。

    Returns:
        pd.DataFrame: 一个包含所有模型、所有指标的、整洁的Master DataFrame。
    """
    print(f"Searching for CSV files in '{data_directory}'...")
    all_files = glob.glob(os.path.join(data_directory, '*.csv'))
    
    # 排除输出文件，避免重复读取
    all_files = [f for f in all_files if not os.path.basename(f).startswith('master_df')]
    
    if not all_files:
        print("Error: No CSV files found in the directory.")
        return pd.DataFrame()

    model_groups, main_models = _get_model_group_info()

    # 1. 读取所有文件并将其分组到每个模型
    model_data_parts = {}
    for f in all_files:
        filename = os.path.basename(f)
        original_model_name = _parse_model_name(filename)
        unified_model_name = _get_unified_model_name(original_model_name)
        
        df_part = pd.read_csv(f)
        
        # 将'Date'列转换为datetime对象，这是时间序列分析的关键
        df_part['Date'] = pd.to_datetime(df_part['Date'])
        
        # 2. 标准化列名并处理数据
        data_type = None
        if 'daily_token_usage' in filename:
            # 计算总token用量，单位是百万(M)，以匹配价格单位
            df_part['total_token_usage_M'] = (df_part['prompt_tokens'] + 
                                              df_part.get('reasoning_tokens', 0) + # .get以防没有该列
                                              df_part['completion_tokens'])
            df_part = df_part[['Date', 'total_token_usage_M']] # 这行代码的意思是只保留日期和总token用量列
            data_type = 'usage'
        elif 'output_speed' in filename:
            df_part = df_part[['Date', 'output_speed']] # 确保只有需要的列
            data_type = 'speed'
        elif 'time_to_first_token' in filename:
            df_part = df_part[['Date', 'time_to_first_token']]
            data_type = 'ttft'
        
        # 对于统一模型组，需要特殊处理
        if unified_model_name in model_groups:
            # 对于speed和ttft数据，只使用主模型的数据
            if data_type in ['speed', 'ttft'] and original_model_name != main_models[unified_model_name]:
                print(f"Skipping {data_type} data from {original_model_name} (not main model for {unified_model_name})")
                continue
            
        # 按统一模型名称收集数据片段
        if unified_model_name not in model_data_parts:
            model_data_parts[unified_model_name] = []
        
        # 为数据片段添加来源信息
        df_part['_source_model'] = original_model_name
        df_part['_data_type'] = data_type
        model_data_parts[unified_model_name].append(df_part)

    # 3. 合并每个模型的数据片段，特别处理统一模型组
    combined_model_dfs = []
    for unified_model_name, parts in model_data_parts.items():
        if unified_model_name in model_groups:
            # 对于统一模型组，需要特殊的合并逻辑
            print(f"Processing unified model group: {unified_model_name}")
            
            # 分离不同类型的数据
            usage_parts = [p for p in parts if p['_data_type'].iloc[0] == 'usage']
            speed_parts = [p for p in parts if p['_data_type'].iloc[0] == 'speed']
            ttft_parts = [p for p in parts if p['_data_type'].iloc[0] == 'ttft']
            
            # 合并token usage数据（累加）
            merged_usage = None
            if usage_parts:
                merged_usage = usage_parts[0][['Date', 'total_token_usage_M']].copy()
                for i in range(1, len(usage_parts)):
                    usage_df = usage_parts[i][['Date', 'total_token_usage_M']]
                    merged_usage = pd.merge(merged_usage, usage_df, on='Date', how='outer', suffixes=('', '_temp'))
                    merged_usage['total_token_usage_M'] = merged_usage['total_token_usage_M'].fillna(0) + merged_usage['total_token_usage_M_temp'].fillna(0)
                    merged_usage = merged_usage.drop('total_token_usage_M_temp', axis=1)
                    
            # 使用主模型的speed和ttft数据
            merged_speed = speed_parts[0][['Date', 'output_speed']] if speed_parts else None
            merged_ttft = ttft_parts[0][['Date', 'time_to_first_token']] if ttft_parts else None
            
            # 合并所有数据类型
            merged_df = merged_usage if merged_usage is not None else pd.DataFrame()
            if merged_speed is not None:
                if merged_df.empty:
                    merged_df = merged_speed
                else:
                    merged_df = pd.merge(merged_df, merged_speed, on='Date', how='outer')
            if merged_ttft is not None:
                if merged_df.empty:
                    merged_df = merged_ttft
                else:
                    merged_df = pd.merge(merged_df, merged_ttft, on='Date', how='outer')
            
        else:
            # 对于非统一模型组，使用原有逻辑
            merged_df = parts[0].drop(['_source_model', '_data_type'], axis=1, errors='ignore')
            for i in range(1, len(parts)):
                part_df = parts[i].drop(['_source_model', '_data_type'], axis=1, errors='ignore')
                merged_df = pd.merge(merged_df, part_df, on='Date', how='outer')
        
        merged_df['model_name'] = unified_model_name
        combined_model_dfs.append(merged_df)
        
    # 4. 将所有模型的数据合并成一个Master DataFrame
    if not combined_model_dfs:
        print("Error: Could not process any model data.")
        return pd.DataFrame()
        
    master_df = pd.concat(combined_model_dfs, ignore_index=True)
    
    # 5. 排序并处理缺失值
    master_df = master_df.sort_values(by=['model_name', 'Date']).reset_index(drop=True)
    
    # 对于性能指标，缺失值通常用前向填充(ffill)或后向填充(bfill)比较合理
    # 这意味着如果某天没有速度数据，我们假设它和前一天/后一天相同
    # 我们按模型分组填充，避免一个模型的数据泄露到另一个模型
    master_df['output_speed'] = master_df.groupby('model_name')['output_speed'].transform(lambda x: x.ffill().bfill())
    master_df['time_to_first_token'] = master_df.groupby('model_name')['time_to_first_token'].transform(lambda x: x.ffill().bfill())
    
    # 对于用量，缺失值通常意味着0
    master_df['total_token_usage_M'].fillna(0, inplace=True)

    # 6. 添加静态特征（ELO, Price） - 使用占位符
    # 你应该用你的真实数据替换这些字典
    elo_scores = {
        'claude-sonnet-4': 1250,
        'claude-3.7-sonnet': 1280,
        # ... 为其他模型添加 ...
    }
    prices_per_million_tokens = { # 假设是输入和输出的综合价格
        # Claude 模型
        'claude-sonnet-4': 3.0,
        'claude-3.7-sonnet': 3.0,
        
        # Gemini 模型
        'gemini-2.0-flash-001': 0.1,
        'gemini-2.0-flash-lite-001': 0.075,
        'gemini-2.5-flash-unified': 0.15,
        'gemini-2.5-pro-unified': 1.25,
        'gemini-2.5-flash-lite': 0.1,
        
        # DeepSeek 模型
        'deepseek-v3': 0.12,
        'deepseek-r1-unified': 0.0,  # 免费模型
        'deepseek-chat-v3-0324-unified': 0.12,
        
        # OpenAI 模型
        'gpt-4o-mini': 0.15,
        'gpt-4.1': 2.0,
        'gpt-4.1-mini': 0.4,
        
        # 其他模型
        'mistral-nemo': 0.008,
        'llama-3.3-70b-instruct': 0.038,
        # ... 为其他模型添加 ...
    }
    
    master_df['elo_score'] = master_df['model_name'].map(elo_scores)
    master_df['price'] = master_df['model_name'].map(prices_per_million_tokens)

    # 填充可能因新模型加入而产生的静态特征缺失值
    master_df['elo_score'].fillna(master_df.groupby('model_name')['elo_score'].transform('first'), inplace=True)
    master_df['price'].fillna(master_df.groupby('model_name')['price'].transform('first'), inplace=True)

    print("\nData preprocessing complete. Master DataFrame created.")
    return master_df

def save_master_df(master_df, filename='master_df.csv'):
    """
    保存master_df到文件
    
    Args:
        master_df (pd.DataFrame): 要保存的Master DataFrame
        filename (str): 保存的文件名，默认为'master_df.csv'
    """
    try:
        # 保存为CSV格式
        master_df.to_csv(filename, index=False)
        print(f"Master DataFrame已成功保存到: {filename}")
        
        # 同时保存为pickle格式以保持数据类型
        pickle_filename = filename.replace('.csv', '.pkl')
        master_df.to_pickle(pickle_filename)
        print(f"Master DataFrame已成功保存到: {pickle_filename}")
        
    except Exception as e:
        print(f"保存Master DataFrame时出错: {e}")

if __name__ == "__main__":
    # 加载并预处理数据
    master_df = load_and_preprocess_data()
    print(master_df)
    
    # 保存master_df
    if not master_df.empty:
        save_master_df(master_df)
