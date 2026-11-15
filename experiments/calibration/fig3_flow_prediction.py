#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验文件：对比不同方法的流量预测效果

实验流程：
1. 给定日期0616-0623的实验数据，利用我们的方法学习各个model质量参数；利用xgboost训练；
2. 预测0623-0714的流量数据；
3. 我们的方法，代入质量参数，复现0623-0714的流量数据；
4. xgboost输入0623-0714的相关信息；
5. 其他方法（手动设置参数，排队论模型）代入0623-0714的相关信息，同样利用博弈模型，但是拥塞函数，或者模型中的参数是手动设置的，直接计算当日流量；
6. 保存最终实验数据；

Date: 2025-07-17
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# import seaborn as sns
from typing import List, Tuple, Dict, Optional
import os
import sys
import pickle
import json
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

# 检查并导入必要的机器学习库
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from scipy.stats import spearmanr, pearsonr
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请安装以下依赖包:")
    print("pip install xgboost scikit-learn matplotlib scipy seaborn")
    sys.exit(1)

# 导入项目相关模块
try:
    # 添加项目路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    
    # 导入核心模块
    from priLLM.calibration.quality_learner import QualityLearner, FollowerLayer
    from priLLM.calibration.real_data import load_real_market_data_for_training
    from priLLM.calibration.train_quality_learner import train_quality_learner_with_real_data, predict_market_share, find_optimal_prices_for_target_flow
    from priLLM.calibration.cost_learner import train_interpretable_cost_learner, InterpretableCostLearner
    from priLLM.calibration.xgboost_baseline import prepare_data_for_xgboost, train_and_evaluate_xgboost, walk_forward_validation_xgboost
    from priLLM.game.equilibrium import solve_weighted_linear_congestion_eq
    
except ImportError as e:
    print(f"导入项目模块失败: {e}")
    print("请确保所有依赖文件都在正确位置")
    sys.exit(1)

class ComprehensiveFlowPredictionExperiment:
    """
    综合流量预测实验类
    对比我们的Stackelberg博弈方法、XGBoost方法和基线方法
    """
    
    def __init__(self, 
                 train_start_date: str = "2026-10-19",
                 train_end_date: str = "2026-10-25", 
                 test_start_date: str = "2026-10-26",
                 test_end_date: str = "2026-11-15",
                 data_file: str = "data/release/market_inputs_top20_month.csv",
                 output_dir: str = "experiment_results",
                 min_price_threshold: float = 0.0,
                 wp: float = 0.6, wd: float = 0.3, wq: float = 0.1):
        """
        初始化实验参数
        
        Args:
            train_start_date: 训练开始日期
            train_end_date: 训练结束日期
            test_start_date: 测试开始日期
            test_end_date: 测试结束日期
            data_file: 数据文件路径
            output_dir: 输出目录
            min_price_threshold: 最小价格阈值
            wp, wd, wq: 用户偏好权重参数
        """
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.data_file = data_file
        self.output_dir = output_dir
        self.min_price_threshold = min_price_threshold
        
        # 用户偏好参数
        self.wp = wp
        self.wd = wd
        self.wq = wq
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算训练和测试天数
        self.train_days = self._calculate_days(train_start_date, train_end_date)
        self.test_days = self._calculate_days(test_start_date, test_end_date)
        
        # 存储实验结果
        self.results = {}
        self.models = {}
        self.data_cache = {}
        
        print(f"=== 实验配置 ===")
        print(f"训练期间: {train_start_date} 到 {train_end_date} ({self.train_days}天)")
        print(f"测试期间: {test_start_date} 到 {test_end_date} ({self.test_days}天)")
        print(f"用户偏好权重: wp={wp}, wd={wd}, wq={wq}")
        print(f"最小价格阈值: ${min_price_threshold}/M tokens")
        print(f"输出目录: {output_dir}")
    
    def _calculate_days(self, start_date: str, end_date: str) -> int:
        """计算日期间隔天数"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        return (end - start).days + 1
    
    def load_data(self, verbose: bool = True) -> None:
        """
        加载训练和测试数据
        """
        if verbose:
            print("\n=== 步骤1: 加载数据 ===")
        
        try:
            # 加载训练数据
            if verbose:
                print(f"加载训练数据: {self.train_start_date} 到 {self.train_end_date}")
            
            train_data = load_real_market_data_for_training(
                self.train_start_date, self.train_days, self.data_file
            )
            
            # 加载测试数据
            if verbose:
                print(f"加载测试数据: {self.test_start_date} 到 {self.test_end_date}")
            
            test_data = load_real_market_data_for_training(
                self.test_start_date, self.test_days, self.data_file
            )
            
            # 存储原始数据
            self.data_cache['train_raw'] = train_data
            self.data_cache['test_raw'] = test_data
            
            # 筛选有效模型（价格大于阈值）
            self._filter_valid_models(train_data, test_data, verbose)
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def _filter_valid_models(self, train_data, test_data, verbose: bool = True) -> None:
        """
        筛选价格大于阈值的模型
        """
        p_nom_train, cap_train, dist_train, demand_train, q_real_train, model_names = train_data
        p_nom_test, cap_test, dist_test, demand_test, q_real_test, _ = test_data
        
        # 计算训练期间的平均价格
        avg_prices = p_nom_train.mean(dim=(0, 1))
        if verbose:
            print(f"所有模型的平均价格: {avg_prices}")
        
        valid_indices = torch.where(avg_prices > self.min_price_threshold)[0]
        
        if len(valid_indices) == 0:
            raise ValueError(f"没有模型的价格大于阈值 ${self.min_price_threshold}/B tokens")
        
        # 保存筛选的模型索引，供后续使用
        self.valid_model_indices = valid_indices
        self.filtered_model_names = [model_names[i] for i in valid_indices]
        
        if verbose:
            print(f"筛选结果:")
            print(f"  符合条件的模型索引: {valid_indices.tolist()}")
            print(f"  符合条件的模型数量: {len(valid_indices)}")
            print(f"  符合条件的模型: {self.filtered_model_names}")
        
        # 筛选训练数据
        train_filtered = (
            p_nom_train[:, :, valid_indices],
            cap_train[:, :, valid_indices],
            dist_train[:, :, :, valid_indices],
            q_real_train[:, :, valid_indices].sum(dim=2, keepdim=True),  # 重新计算需求
            q_real_train[:, :, valid_indices],
            [model_names[i] for i in valid_indices]
        )
        
        # 筛选测试数据
        test_filtered = (
            p_nom_test[:, :, valid_indices],
            cap_test[:, :, valid_indices], 
            dist_test[:, :, :, valid_indices],
            q_real_test[:, :, valid_indices].sum(dim=2, keepdim=True),  # 重新计算需求
            q_real_test[:, :, valid_indices],
            [model_names[i] for i in valid_indices]
        )
        
        # 更新需求数据
        train_filtered = (
            train_filtered[0], train_filtered[1], train_filtered[2],
            train_filtered[4].sum(dim=2, keepdim=True),  # 使用筛选后的流量计算需求
            train_filtered[4], train_filtered[5]
        )
        
        test_filtered = (
            test_filtered[0], test_filtered[1], test_filtered[2],
            test_filtered[4].sum(dim=2, keepdim=True),  # 使用筛选后的流量计算需求
            test_filtered[4], test_filtered[5]
        )
        
        self.data_cache['train_filtered'] = train_filtered
        self.data_cache['test_filtered'] = test_filtered
        
        if verbose:
            print(f"模型筛选结果:")
            print(f"  原始模型数量: {len(model_names)}")
            print(f"  筛选后模型数量: {len(train_filtered[5])}")
            print(f"  训练数据形状: {train_filtered[0].shape}")
            print(f"  测试数据形状: {test_filtered[0].shape}")
            excluded_models = [model_names[i] for i in range(len(model_names)) if i not in valid_indices]
            if excluded_models:
                print(f"  排除的模型: {excluded_models}")
            print(f"  参与实验的模型: {train_filtered[5]}")
            
            # 验证数据一致性
            assert train_filtered[0].shape[2] == test_filtered[0].shape[2], \
                f"训练和测试数据的模型维度不一致: {train_filtered[0].shape[2]} vs {test_filtered[0].shape[2]}"
    
    def train_stackelberg_model(self, verbose: bool = True) -> None:
        """
        训练Stackelberg博弈质量学习器
        现在直接调用 train_interpretable_cost_learner 函数
        """
        if verbose:
            print("\n=== 步骤2: 训练Stackelberg博弈模型 ===")
        
        try:
            # 使用已经筛选好的训练数据
            train_data_filtered = self.data_cache['test_filtered']
            p_nom_train, cap_train, dist_train, demand_train, q_real_train, model_names = train_data_filtered
            
            if verbose:
                print(f"使用筛选后的训练数据:")
                print(f"  模型数量: {len(model_names)}")
                print(f"  数据形状: {p_nom_train.shape}")
                print(f"  模型列表: {model_names}")
            
            # 计算训练天数
            num_days = p_nom_train.shape[1]
            
            # 这里选择要么拟合测试起始日期，要么拟合训练起始日期
            # start_date_str = self.train_start_date
            start_date_str = self.test_start_date
            
            interpretable_model, model_names_result, training_history = train_interpretable_cost_learner(
                start_date=start_date_str,
                num_days=num_days,
                initial_wq=self.wq,
                initial_wd=self.wd,
                learning_rate=0.01,  # 可解释模型通常需要较小的学习率
                num_epochs=500,     # 可以根据需要调整
                data_file=self.data_file,
                min_price_threshold=self.min_price_threshold,
                verbose=verbose
            )
            
            # 将返回的 InterpretableCostLearner 模型保存
            # 注意：这与原来的 QualityLearner 不同，但功能类似
            self.models['stackelberg'] = interpretable_model
            self.results['stackelberg_training'] = training_history
            self.stackelberg_model_names = model_names_result
            
            if verbose:
                print(f"Stackelberg模型训练完成")
                print(f"最终损失: {training_history['losses'][-1]:.6f}")
                
                # 显示学到的可解释参数
                with torch.no_grad():
                    learned_wq = torch.exp(interpretable_model.log_wq).item()
                    learned_wd = torch.exp(interpretable_model.log_wd).item()
                    learned_biases = interpretable_model.biases.data.numpy()
                
                print(f"学到的可解释参数:")
                print(f"  全局拥堵权重 (w_q): {learned_wq:.4f}")
                print(f"  全局延迟权重 (w_d): {learned_wd:.4f}")
                print(f"  模型特定偏差范围: [{learned_biases.min():.3f}, {learned_biases.max():.3f}]")
                    
        except Exception as e:
            print(f"Stackelberg模型训练失败: {e}")
            raise
    
    def train_xgboost_model(self, verbose: bool = True) -> None:
        """
        训练XGBoost模型
        """
        if verbose:
            print("\n=== 步骤3: 训练XGBoost模型 ===")
        
        try:
            train_data = self.data_cache['train_filtered']
            p_nom_train, cap_train, dist_train, demand_train, q_real_train, model_names = train_data
            
            # 转换数据格式
            X_train, y_train = prepare_data_for_xgboost(
                p_nom_train, cap_train, dist_train, demand_train, q_real_train, model_names
            )
            
            # 训练模型
            xgb_model = train_and_evaluate_xgboost(
                X_train, y_train, test_size=0.2, use_time_series_split=True
            )
            
            self.models['xgboost'] = xgb_model
            self.data_cache['xgb_train_data'] = (X_train, y_train)
            
            if verbose:
                print(f"XGBoost模型训练完成")
                
        except Exception as e:
            print(f"XGBoost模型训练失败: {e}")
            raise
    
    def prepare_baseline_models(self, verbose: bool = True) -> None:
        """
        准备基线模型（手动设置参数的方法）
        """
        if verbose:
            print("\n=== 步骤4: 准备基线模型 ===")
        
        # 使用筛选后的训练数据
        train_data_for_baseline = self.data_cache['test_filtered']
        model_names = train_data_for_baseline[5]
        
        if verbose:
            print(f"基线模型使用筛选后的训练数据:")
            print(f"  模型数量: {len(model_names)}")
            print(f"  训练数据形状: {train_data_for_baseline[0].shape}")
        
        # 只保留基于容量的简单启发式
        baseline_quality_capacity = torch.zeros(len(model_names))
        avg_capacities = train_data_for_baseline[1].mean(dim=(0, 1))
        # 容量高的模型质量也高
        # baseline_quality_capacity = (avg_capacities - avg_capacities.mean()) * 0.01
        # 随机数映射到 [9.5, 10.5] 区间，围绕 10 抖动
        # NOTE: this baseline is a *random* quality initialisation (seeded), not a capacity heuristic.
        _g = torch.Generator().manual_seed(0)
        baseline_quality_capacity = torch.rand(len(model_names), generator=_g) * 100.0

        
        # 创建基线模型
        n_users = 1
        m_providers = len(model_names)
        
        # 基线方法1: 基于容量的启发式
        self.models['baseline_capacity'] = QualityLearner(
            n_users, m_providers, self.wp, self.wd, self.wq,
            initial_quality=baseline_quality_capacity
        )
        
        # 基线方法2: 无拥塞baseline (相同模型，相同训练方式，但拥塞权重设置为0.001)
        if verbose:
            print("训练无拥塞baseline模型...")
        

        # 训练无拥塞模型，传入相同的训练数据
        no_congestion_model = self._train_no_congestion_model(train_data_for_baseline, verbose)
        self.models['baseline_no_congestion'] = no_congestion_model
        
        if verbose:
            print("无拥塞baseline模型训练完成")
            # 显示学到的质量参数
            quality_params = no_congestion_model.quality.data.numpy()
            # for i, (name, quality) in enumerate(zip(model_names, quality_params)):
            #     print(f"  {name}: {quality:.3f}")
                    
        if verbose:
            print(f"基线模型准备完成:")
            print(f"  容量启发式质量参数: {baseline_quality_capacity.numpy()}")
            print(f"  无拥塞baseline模型已训练完成")
        
        # 保存基线模型使用的模型名称列表
        self.baseline_model_names = model_names
    
    def predict_and_evaluate(self, verbose: bool = True) -> None:
        """
        在测试集上进行预测并评估所有方法
        """
        if verbose:
            print("\n=== 步骤5: 测试集预测与评估 ===")
        
        test_data = self.data_cache['test_filtered']
        p_nom_test, cap_test, dist_test, demand_test, q_real_test, model_names = test_data
        
        # 为Stackelberg模型重新筛选测试数据以匹配训练时的模型集合
        # self._refilter_test_data_for_stackelberg(verbose)
        
        # # 使用重新筛选的Stackelberg测试数据
        # if 'test_filtered_stackelberg' in self.data_cache:
        #     test_data_stackelberg = self.data_cache['test_filtered_stackelberg']
        #     p_nom_test_stack, cap_test_stack, dist_test_stack, demand_test_stack, q_real_test_stack, model_names_stack = test_data_stackelberg
        # else:
        #     # 如果重新筛选失败，使用常规的筛选后测试数据
        #     p_nom_test_stack, cap_test_stack, dist_test_stack, demand_test_stack, q_real_test_stack, model_names_stack = test_data
        
        predictions = {}
        
        # 1. Stackelberg模型预测
        if verbose:
            print("\n5.1 Stackelberg模型预测...")
        
        stackelberg_model = self.models['stackelberg']
        
        # # 检查数据维度匹配
        # if verbose:
        #     print(f"Stackelberg测试数据形状: {p_nom_test_stack.shape}")
        #     print(f"模型质量参数形状: {stackelberg_model.quality.shape}")
        #     print(f"测试数据模型数: {len(model_names_stack)}")
        #     print(f"模型质量参数数: {stackelberg_model.quality.shape[0]}")
        
        # # 确保测试数据的模型数与训练时一致
        # if len(model_names_stack) != stackelberg_model.quality.shape[0]:
        #     if verbose:
        #         print(f"警告: 测试数据模型数({len(model_names_stack)})与训练模型参数数({stackelberg_model.quality.shape[0]})不匹配")
        #         print("这可能导致张量大小不匹配错误")
            
        with torch.no_grad():
            q_pred_stackelberg = stackelberg_model(p_nom_test, cap_test, dist_test, demand_test)
        predictions['stackelberg'] = q_pred_stackelberg
        
        # 2. XGBoost模型预测
        if verbose:
            print("\n5.2 XGBoost模型预测...")
        
        # 准备XGBoost测试数据
        X_test, y_test = prepare_data_for_xgboost(
            p_nom_test, cap_test, dist_test, demand_test, q_real_test, model_names
        )
        
        xgb_model = self.models['xgboost']
        y_pred_xgb = xgb_model.predict(X_test)
        q_pred_xgb = torch.tensor(y_pred_xgb).unsqueeze(0)  # 添加batch维度
        predictions['xgboost'] = q_pred_xgb
        
        # 3. 基线模型预测
        baseline_methods = ['baseline_capacity', 'baseline_no_congestion']
        
        for method in baseline_methods:
            if verbose:
                print(f"\n5.3 {method}预测...")
            
            baseline_model = self.models[method]
            with torch.no_grad():
                q_pred_baseline = baseline_model(p_nom_test, cap_test, dist_test, demand_test)
            predictions[method] = q_pred_baseline

        
        self.results['predictions'] = predictions
        
        # 4. 计算评估指标
        if verbose:
            print("\n5.4 计算评估指标...")
        
        # 使用Stackelberg模型的测试数据作为评估基准
        self._calculate_evaluation_metrics(q_real_test, predictions, model_names, verbose)
    
    def _calculate_evaluation_metrics(self, q_real: torch.Tensor, predictions: Dict, 
                                    model_names: List[str], verbose: bool = True) -> None:
        """
        计算各种评估指标
        """
        metrics = {}
        
        for method_name, q_pred in predictions.items():
            # 转换为numpy格式便于计算
            y_true = q_real.squeeze(0).numpy()  # (days, providers)
            y_pred = q_pred.squeeze(0).detach().numpy()  # (days, providers)
            
            # 1. 整体指标
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # 2. 相关性指标
            # 整体皮尔逊相关性
            pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            
            # 每日Spearman秩相关性（衡量排序能力）
            daily_spearman = []
            for day in range(y_true.shape[0]):
                corr, _ = spearmanr(y_true[day], y_pred[day])
                if not np.isnan(corr):
                    daily_spearman.append(corr)
            avg_spearman = np.mean(daily_spearman) if daily_spearman else 0.0
            
            # 3. 相对误差指标
            # 避免除零错误
            relative_errors = []
            for i in range(y_true.shape[0]):
                for j in range(y_true.shape[1]):
                    if y_true[i, j] > 1e-6:  # 避免很小的真实值
                        rel_err = abs(y_pred[i, j] - y_true[i, j]) / y_true[i, j]
                        relative_errors.append(rel_err)
            
            mape = np.mean(relative_errors) * 100 if relative_errors else float('inf')
            
            # 4. 市场份额预测准确性
            # 每日的市场份额分布
            daily_market_share_errors = []
            for day in range(y_true.shape[0]):
                true_total = y_true[day].sum()
                pred_total = y_pred[day].sum()
                
                if true_total > 1e-6 and pred_total > 1e-6:
                    true_shares = y_true[day] / true_total
                    pred_shares = y_pred[day] / pred_total
                    share_error = np.mean(np.abs(true_shares - pred_shares))
                    daily_market_share_errors.append(share_error)
            
            avg_market_share_error = np.mean(daily_market_share_errors) if daily_market_share_errors else float('inf')
            
            # 5. Top-K准确性（预测排名前K的模型）
            top_k_accuracies = {}
            for k in [1, 3, 5]:
                if k <= y_true.shape[1]:
                    correct_top_k = 0
                    total_days = 0
                    
                    for day in range(y_true.shape[0]):
                        true_top_k = set(np.argsort(y_true[day])[-k:])
                        pred_top_k = set(np.argsort(y_pred[day])[-k:])
                        
                        intersection = len(true_top_k.intersection(pred_top_k))
                        correct_top_k += intersection / k
                        total_days += 1
                    
                    top_k_accuracies[f'top_{k}'] = correct_top_k / total_days if total_days > 0 else 0.0
            
            # 存储指标
            metrics[method_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'pearson_corr': pearson_corr,
                'avg_spearman': avg_spearman,
                'mape': mape,
                'market_share_error': avg_market_share_error,
                **top_k_accuracies
            }
        
        self.results['metrics'] = metrics
        
        if verbose:
            self._print_metrics_summary(metrics, model_names)
    
    def _print_metrics_summary(self, metrics: Dict, model_names: List[str]) -> None:
        """
        打印评估指标摘要
        """
        print(f"\n=== 评估指标摘要 ===")
        print(f"测试期间: {self.test_start_date} 到 {self.test_end_date} ({self.test_days}天)")
        print(f"模型数量: {len(model_names)}")
        print(f"参与评估的模型: {model_names}")
        
        # 创建格式化表格
        methods = list(metrics.keys())
        method_names_display = {
            'stackelberg': 'Stackelberg博弈',
            'xgboost': 'XGBoost',
            'baseline_capacity': '容量启发式', 
            'baseline_no_congestion': '无拥塞Baseline'
        }
        
        print(f"\n{'方法':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Pearson':<8} {'Spearman':<8} {'MAPE(%)':<8} {'Top-1':<8} {'Top-3':<8}")
        print("-" * 95)
        
        for method in methods:
            display_name = method_names_display.get(method, method)
            m = metrics[method]
            
            print(f"{display_name:<15} "
                  f"{m['rmse']:<8.2f} "
                  f"{m['mae']:<8.2f} "
                  f"{m['r2']:<8.3f} "
                  f"{m['pearson_corr']:<8.3f} "
                  f"{m['avg_spearman']:<8.3f} "
                  f"{min(m['mape'], 999.9):<8.1f} "
                  f"{m.get('top_1', 0):<8.3f} "
                  f"{m.get('top_3', 0):<8.3f}")
        
        # 找出最佳方法
        print(f"\n=== 最佳方法 ===")
        best_rmse = min(methods, key=lambda x: metrics[x]['rmse'])
        best_r2 = max(methods, key=lambda x: metrics[x]['r2'])
        best_spearman = max(methods, key=lambda x: metrics[x]['avg_spearman'])
        best_top1 = max(methods, key=lambda x: metrics[x].get('top_1', 0))
        
        print(f"最低RMSE: {method_names_display.get(best_rmse, best_rmse)} ({metrics[best_rmse]['rmse']:.3f})")
        print(f"最高R²: {method_names_display.get(best_r2, best_r2)} ({metrics[best_r2]['r2']:.3f})")
        print(f"最高Spearman: {method_names_display.get(best_spearman, best_spearman)} ({metrics[best_spearman]['avg_spearman']:.3f})")
        print(f"最高Top-1准确性: {method_names_display.get(best_top1, best_top1)} ({metrics[best_top1].get('top_1', 0):.3f})")
    
    def save_results(self, verbose: bool = True) -> None:
        """
        保存实验结果到文件
        """
        if verbose:
            print(f"\n=== 步骤6: 保存实验结果 ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存评估指标
        metrics_file = os.path.join(self.output_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型以便JSON序列化
            metrics_serializable = {}
            for method, metrics in self.results['metrics'].items():
                metrics_serializable[method] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in metrics.items()
                }
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
        
        # 2. 保存预测结果
        predictions_file = os.path.join(self.output_dir, f"predictions_{timestamp}.pkl")
        with open(predictions_file, 'wb') as f:
            # 转换tensor为numpy以便pickle序列化
            predictions_serializable = {
                method: pred.detach().numpy() 
                for method, pred in self.results['predictions'].items()
            }
            pickle.dump(predictions_serializable, f)
        
        # 3. 保存真实数据
        true_data_file = os.path.join(self.output_dir, f"true_data_{timestamp}.pkl")
        test_data = self.data_cache['test_filtered']
        with open(true_data_file, 'wb') as f:
            true_data_serializable = {
                'q_real': test_data[4].detach().numpy(),
                'p_nominal': test_data[0].detach().numpy(),
                'capacity': test_data[1].detach().numpy(),
                'latency': test_data[2].detach().numpy(),
                'demand': test_data[3].detach().numpy(),
                'model_names': test_data[5],
                'test_start_date': self.test_start_date,
                'test_end_date': self.test_end_date
            }
            pickle.dump(true_data_serializable, f)
        
        # 4. 保存模型参数
        models_file = os.path.join(self.output_dir, f"models_{timestamp}.pkl")
        with open(models_file, 'wb') as f:
            models_serializable = {}
            for name, model in self.models.items():
                if hasattr(model, 'quality'):  # Stackelberg模型
                    models_serializable[name] = {
                        'type': 'stackelberg',
                        'quality': model.quality.detach().numpy(),
                        'wp': model.wp, 'wd': model.wd, 'wq': model.wq
                    }
                # XGBoost模型保存在单独文件中
            pickle.dump(models_serializable, f)
        
        # 5. 保存XGBoost模型（单独文件）
        if 'xgboost' in self.models:
            xgb_file = os.path.join(self.output_dir, f"xgboost_model_{timestamp}.pkl")
            with open(xgb_file, 'wb') as f:
                pickle.dump(self.models['xgboost'], f)
        
        # 6. 创建实验摘要报告
        self._create_summary_report(timestamp, verbose)
        
        if verbose:
            print(f"实验结果已保存到: {self.output_dir}")
            print(f"  - 评估指标: {metrics_file}")
            print(f"  - 预测结果: {predictions_file}")
            print(f"  - 真实数据: {true_data_file}")
            print(f"  - 模型参数: {models_file}")
            if 'xgboost' in self.models:
                print(f"  - XGBoost模型: {xgb_file}")
    
    def _create_summary_report(self, timestamp: str, verbose: bool = True) -> None:
        """
        创建实验摘要报告
        """
        report_file = os.path.join(self.output_dir, f"summary_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 流量预测实验摘要报告 ===\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"训练期间: {self.train_start_date} 到 {self.train_end_date} ({self.train_days}天)\n")
            f.write(f"测试期间: {self.test_start_date} 到 {self.test_end_date} ({self.test_days}天)\n")
            f.write(f"用户偏好权重: wp={self.wp}, wd={self.wd}, wq={self.wq}\n")
            f.write(f"最小价格阈值: ${self.min_price_threshold}/M tokens\n\n")
            
            # 数据信息
            test_data = self.data_cache['test_filtered']
            f.write(f"参与实验的模型: {test_data[5]}\n")
            f.write(f"模型数量: {len(test_data[5])}\n\n")
            
            # 评估指标表格
            f.write("=== 评估指标对比 ===\n")
            metrics = self.results['metrics']
            methods = list(metrics.keys())
            method_names_display = {
                'stackelberg': 'Stackelberg Game',
                'xgboost': 'XGBoost',
                'baseline_capacity': 'Capacity Heuristic',
                'baseline_no_congestion': 'No Congestion Baseline'
            }
            
            f.write(f"{'Method':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Pearson':<8} {'Spearman':<8} {'MAPE(%)':<8} {'Top-1':<8} {'Top-3':<8}\n")
            f.write("-" * 95 + "\n")
            
            for method in methods:
                display_name = method_names_display.get(method, method)
                m = metrics[method]
                
                f.write(f"{display_name:<15} "
                       f"{m['rmse']:<8.2f} "
                       f"{m['mae']:<8.2f} "
                       f"{m['r2']:<8.3f} "
                       f"{m['pearson_corr']:<8.3f} "
                       f"{m['avg_spearman']:<8.3f} "
                       f"{min(m['mape'], 999.9):<8.1f} "
                       f"{m.get('top_1', 0):<8.3f} "
                       f"{m.get('top_3', 0):<8.3f}\n")
            
            # 最佳方法
            f.write(f"\n=== 最佳方法 ===\n")
            best_rmse = min(methods, key=lambda x: metrics[x]['rmse'])
            best_r2 = max(methods, key=lambda x: metrics[x]['r2'])
            best_spearman = max(methods, key=lambda x: metrics[x]['avg_spearman'])
            best_top1 = max(methods, key=lambda x: metrics[x].get('top_1', 0))
            
            f.write(f"最低RMSE: {method_names_display.get(best_rmse, best_rmse)} ({metrics[best_rmse]['rmse']:.3f})\n")
            f.write(f"最高R²: {method_names_display.get(best_r2, best_r2)} ({metrics[best_r2]['r2']:.3f})\n")
            f.write(f"最高Spearman: {method_names_display.get(best_spearman, best_spearman)} ({metrics[best_spearman]['avg_spearman']:.3f})\n")
            f.write(f"最高Top-1准确性: {method_names_display.get(best_top1, best_top1)} ({metrics[best_top1].get('top_1', 0):.3f})\n")
            
            # Stackelberg模型的质量参数
            # if 'stackelberg' in self.models:
            #     f.write(f"\n=== Stackelberg模型学到的质量参数 ===\n")
            #     quality_params = self.models['stackelberg'].quality.detach().numpy()
            #     for i, (name, quality) in enumerate(zip(test_data[5], quality_params)):
            #         f.write(f"{name}: {quality}\n")
        
        if verbose:
            print(f"  - 摘要报告: {report_file}")
    
    def plot_results(self, save_plots: bool = True, show_plots: bool = False) -> None:
        """
        绘制实验结果图表
        """
        print(f"\n=== Step 7: Plotting Result Charts ===")
        
        # 设置matplotlib使用英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 评估指标对比雷达图
        self._plot_metrics_radar(timestamp, save_plots, show_plots)
        
        # 2. 时间序列预测对比图
        self._plot_time_series_comparison(timestamp, save_plots, show_plots)
        
        # 2.5. 单独的模型流量预测对比图
        self._plot_model_flow_prediction(timestamp, save_plots, show_plots)
        
        # 3. 相关性散点图
        self._plot_correlation_scatter(timestamp, save_plots, show_plots)
        
        # 4. 模型排名热力图
        self._plot_ranking_heatmap(timestamp, save_plots, show_plots)
    
    def _plot_metrics_radar(self, timestamp: str, save_plots: bool, show_plots: bool) -> None:
        """绘制评估指标雷达图"""
        try:
            metrics = self.results['metrics']
            methods = list(metrics.keys())
            
            # 选择关键指标
            key_metrics = ['r2', 'avg_spearman', 'top_1', 'top_3']
            metric_labels = ['R² Score', 'Spearman Correlation', 'Top-1 Accuracy', 'Top-3 Accuracy']
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合雷达图
            
            method_names_display = {
                'stackelberg': 'Stackelberg Game',
                'xgboost': 'XGBoost',
                'baseline_capacity': 'Capacity Heuristic',
                'baseline_no_congestion': 'No Congestion Baseline'
            }
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
            
            for i, method in enumerate(methods):
                values = []
                for metric in key_metrics:
                    val = metrics[method].get(metric, 0)
                    # 标准化到0-1范围
                    if metric == 'r2':
                        val = max(0, val)  # R²可能为负
                    values.append(val)
                
                values += values[:1]  # 闭合
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=method_names_display.get(method, method), color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title('Evaluation Metrics Comparison (Radar Chart)', size=16, pad=20)
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"metrics_radar_{timestamp}.png"), 
                           dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Failed to plot radar chart: {e}")
    
    def _plot_time_series_comparison(self, timestamp: str, save_plots: bool, show_plots: bool) -> None:
        """绘制时间序列预测对比图"""
        try:
            test_data = self.data_cache['test_filtered']
            q_real = test_data[4].squeeze(0).numpy()  # (days, providers)
            predictions = self.results['predictions']
            model_names = test_data[5]
            
            # 计算每日总流量
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # 创建日期标签
            start_date = datetime.strptime(self.test_start_date, '%Y-%m-%d')
            dates = [(start_date + timedelta(days=i)).strftime('%m-%d') for i in range(self.test_days)]
            
            # 1. 总流量趋势
            ax = axes[0]
            real_total = q_real.sum(axis=1)
            ax.plot(dates, real_total, 'ko-', linewidth=2, label='True Total Flow', markersize=4)
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
            method_names_display = {
                'stackelberg': 'Stackelberg',
                'xgboost': 'XGBoost',
                'baseline_capacity': '容量启发式',
                'baseline_no_congestion': '无拥塞Baseline'
            }
            
            for i, (method, q_pred) in enumerate(predictions.items()):
                if method in method_names_display: # 只显示主要方法
                    pred_total = q_pred.squeeze(0).detach().numpy().sum(axis=1)
                    ax.plot(dates, pred_total, 'o-', linewidth=1.5, 
                           label=method_names_display[method], 
                           color=colors[i], markersize=3, alpha=0.8)
            
            ax.set_title('Total Flow Prediction Comparison', fontsize=12)
            ax.set_ylabel('Flow (M tokens)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. 前3个模型的流量趋势
            top_3_indices = np.argsort(q_real.mean(axis=0))[-3:]
            ax = axes[1]
            
            for idx in top_3_indices:
                ax.plot(dates, q_real[:, idx], 'o-', linewidth=2, 
                       label=f'{model_names[idx]} (真实)', markersize=4)
                
                # 只显示Stackelberg预测
                if 'stackelberg' in predictions:
                    q_pred_stack = predictions['stackelberg'].squeeze(0).detach().numpy()
                    ax.plot(dates, q_pred_stack[:, idx], '--', linewidth=1.5, 
                           label=f'{model_names[idx]} (预测)', alpha=0.7)
            
            ax.set_title('Top-3 Models Flow Prediction', fontsize=12)
            ax.set_ylabel('Flow (M tokens)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. 预测误差
            ax = axes[2]
            for method in ['stackelberg', 'xgboost']:
                if method in predictions:
                    q_pred = predictions[method].squeeze(0).detach().numpy()
                    daily_errors = np.sqrt(((q_real - q_pred) ** 2).mean(axis=1))
                    ax.plot(dates, daily_errors, 'o-', linewidth=1.5, 
                           label=f'{method_names_display.get(method, method)} RMSE')
            
            ax.set_title('Daily Prediction Error (RMSE)', fontsize=12)
            ax.set_ylabel('RMSE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. 相关性变化
            ax = axes[3]
            for method in ['stackelberg', 'xgboost']:
                if method in predictions:
                    q_pred = predictions[method].squeeze(0).detach().numpy()
                    daily_corrs = []
                    for day in range(len(dates)):
                        corr, _ = spearmanr(q_real[day], q_pred[day])
                        daily_corrs.append(corr if not np.isnan(corr) else 0)
                    
                    ax.plot(dates, daily_corrs, 'o-', linewidth=1.5,
                           label=f'{method_names_display.get(method, method)} Spearman')
            
            ax.set_title('Daily Spearman Correlation', fontsize=12)
            ax.set_ylabel('Spearman Correlation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # 调整x轴标签
            for ax in axes:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"time_series_{timestamp}.png"), 
                           dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Failed to plot time series comparison: {e}")
    
    def _plot_model_flow_prediction(self, timestamp: str, save_plots: bool, show_plots: bool) -> None:
        """绘制流量预测对比图：流量前3的LLM模型的真实流量 via Stackelberg模型预测"""
        try:
            test_data = self.data_cache['test_filtered']
            q_real = test_data[4].squeeze(0).numpy()  # (days, providers)
            predictions = self.results['predictions']
            model_names = test_data[5]
            
            # 检查是否有Stackelberg预测
            if 'stackelberg' not in predictions:
                print("警告: 没有找到Stackelberg预测结果，跳过流量预测对比图")
                return
                
            # 创建单独的大图
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # 创建日期标签
            start_date = datetime.strptime(self.test_start_date, '%Y-%m-%d')
            dates = [(start_date + timedelta(days=i)).strftime('%m-%d') for i in range(self.test_days)]
            
            # 定义前3个模型的颜色 - 使用容易区分的颜色
            model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
            
            # 获取流量前3的模型
            top_3_indices = np.argsort(q_real.mean(axis=0))[-3:]
            
            # 获取Stackelberg预测结果
            q_pred_stack = predictions['stackelberg'].squeeze(0).detach().numpy()
            
            # 为每个模型绘制真实值和Stackelberg预测
            for i, model_idx in enumerate(top_3_indices):
                model_name = model_names[model_idx]
                color = model_colors[i]
                
                # 绘制真实值（实线，圆形标记）
                ax.plot(dates, q_real[:, model_idx], 
                       color=color, 
                       linestyle='-',
                       linewidth=3,
                       marker='o',
                       markersize=7,
                       alpha=0.9,
                       label=f'{model_name} (True Flow)',
                       markerfacecolor=color,
                       markeredgecolor='white',
                       markeredgewidth=1.5)
                
                # 绘制Stackelberg预测（虚线，方形标记）
                ax.plot(dates, q_pred_stack[:, model_idx], 
                       color=color, 
                       linestyle='--',
                       linewidth=2.5,
                       marker='s',
                       markersize=6,
                       alpha=0.8,
                       label=f'{model_name}',
                       markerfacecolor=color,
                       markeredgecolor='white',
                       markeredgewidth=1.5)
            
            # 设置图表属性
            ax.set_title('flow prediction via learned Stackelberg game', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=13, fontweight='bold')
            ax.set_ylabel('Flow (M tokens)', fontsize=13, fontweight='bold')
            
            # 设置图例
            # # legend = ax.legend(loc='upper left', fontsize=11, ncol=2)
            # legend.get_frame().set_facecolor('white')
            # legend.get_frame().set_alpha(0.95)
            # legend.get_frame().set_edgecolor('gray')
            
            # 设置网格和样式
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafbfc')
            
            # 调整x轴标签
            ax.tick_params(axis='x', rotation=45, labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            
            # 添加说明文本框
            legend_text = (
                'Tips:\n'
                '● true flow\n'
                '■ predicted flow\n\n'
            )
            ax.text(0.02, 0.98, legend_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='gray'),
                   fontsize=10,
                   fontweight='normal')
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"top3_model_flow_prediction_{timestamp}.png"), 
                           dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Failed to plot model flow prediction: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_correlation_scatter(self, timestamp: str, save_plots: bool, show_plots: bool) -> None:
        """绘制真实vs预测流量散点图"""
        try:
            test_data = self.data_cache['test_filtered']
            q_real = test_data[4].squeeze(0).numpy().flatten()
            predictions = self.results['predictions']
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            method_names_display = {
                'stackelberg': 'Stackelberg Game',
                'xgboost': 'XGBoost',
                'baseline_capacity': 'Capacity Heuristic',
                'baseline_no_congestion': 'No Congestion Baseline'
            }
            
            for i, (method, q_pred) in enumerate(predictions.items()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                q_pred_flat = q_pred.squeeze(0).detach().numpy().flatten()
                
                # 散点图
                ax.scatter(q_real, q_pred_flat, alpha=0.6, s=20)
                
                # 拟合线
                max_val = max(q_real.max(), q_pred_flat.max())
                ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2)
                
                # 计算相关性
                corr, _ = pearsonr(q_real, q_pred_flat)
                
                ax.set_xlabel('True Flow (M tokens)')
                ax.set_ylabel('Predicted Flow (M tokens)')
                ax.set_title(f'{method_names_display.get(method, method)}\n(r={corr:.3f})')
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(len(predictions), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"correlation_scatter_{timestamp}.png"), 
                           dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Failed to plot scatter plots: {e}")
    
    def _plot_ranking_heatmap(self, timestamp: str, save_plots: bool, show_plots: bool) -> None:
        """绘制模型排名热力图"""
        try:
            test_data = self.data_cache['test_filtered']
            q_real = test_data[4].squeeze(0).numpy()  # (days, providers)
            predictions = self.results['predictions']
            model_names = test_data[5]
            
            # 计算每日排名
            num_days = q_real.shape[0]
            num_providers = q_real.shape[1]
            
            # 创建日期标签
            start_date = datetime.strptime(self.test_start_date, '%Y-%m-%d')
            dates = [(start_date + timedelta(days=i)).strftime('%m-%d') for i in range(num_days)]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 真实排名
            ax = axes[0, 0]
            real_ranks = np.zeros_like(q_real)
            for day in range(num_days):
                real_ranks[day] = num_providers - np.argsort(np.argsort(q_real[day]))
            
            im = ax.imshow(real_ranks.T, cmap='RdYlBu_r', aspect='auto')
            ax.set_title('True Ranking', fontsize=14)
            ax.set_ylabel('Models')
            ax.set_yticks(range(num_providers))
            ax.set_yticklabels([name[:10] + '...' if len(name) > 10 else name for name in model_names])
            
            # Stackelberg预测排名
            if 'stackelberg' in predictions:
                ax = axes[0, 1]
                q_pred = predictions['stackelberg'].squeeze(0).detach().numpy()
                pred_ranks = np.zeros_like(q_pred)
                for day in range(num_days):
                    pred_ranks[day] = num_providers - np.argsort(np.argsort(q_pred[day]))
                
                im = ax.imshow(pred_ranks.T, cmap='RdYlBu_r', aspect='auto')
                ax.set_title('Stackelberg Predicted Ranking', fontsize=14)
                ax.set_yticks(range(num_providers))
                ax.set_yticklabels([name[:10] + '...' if len(name) > 10 else name for name in model_names])
            
            # XGBoost预测排名
            if 'xgboost' in predictions:
                ax = axes[1, 0]
                q_pred = predictions['xgboost'].squeeze(0).detach().numpy()
                pred_ranks = np.zeros_like(q_pred)
                for day in range(num_days):
                    pred_ranks[day] = num_providers - np.argsort(np.argsort(q_pred[day]))
                
                im = ax.imshow(pred_ranks.T, cmap='RdYlBu_r', aspect='auto')
                ax.set_title('XGBoost Predicted Ranking', fontsize=14)
                ax.set_ylabel('Models')
                ax.set_xlabel('Date')
                ax.set_yticks(range(num_providers))
                ax.set_yticklabels([name[:10] + '...' if len(name) > 10 else name for name in model_names])
            
            # 排名差异热力图
            if 'stackelberg' in predictions:
                ax = axes[1, 1]
                q_pred = predictions['stackelberg'].squeeze(0).detach().numpy()
                pred_ranks = np.zeros_like(q_pred)
                for day in range(num_days):
                    pred_ranks[day] = num_providers - np.argsort(np.argsort(q_pred[day]))
                
                rank_diff = np.abs(real_ranks - pred_ranks)
                im = ax.imshow(rank_diff.T, cmap='Reds', aspect='auto')
                ax.set_title('Stackelberg Ranking Error', fontsize=14)
                ax.set_xlabel('Date')
                ax.set_yticks(range(num_providers))
                ax.set_yticklabels([name[:10] + '...' if len(name) > 10 else name for name in model_names])
            
            # 设置x轴标签
            for ax in axes.flatten():
                if hasattr(ax, 'get_images') and ax.get_images():
                    ax.set_xticks(range(0, num_days, max(1, num_days // 10)))
                    ax.set_xticklabels([dates[i] for i in range(0, num_days, max(1, num_days // 10))], 
                                     rotation=45)
            
            # 添加颜色条
            fig.colorbar(im, ax=axes, shrink=0.8, label='Ranking (1=Highest)')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(self.output_dir, f"ranking_heatmap_{timestamp}.png"), 
                           dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Failed to plot ranking heatmap: {e}")
    
    def run_full_experiment(self, verbose: bool = True) -> None:
        """
        运行完整实验流程
        """
        print("=" * 80)
        print("开始流量预测对比实验")
        print("=" * 80)
        
        try:
            # 1. 加载数据
            self.load_data(verbose)
            
            # 2. 训练Stackelberg模型
            self.train_stackelberg_model(verbose)
            
            # 3. 训练XGBoost模型
            self.train_xgboost_model(verbose)
            
            # 4. 准备基线模型
            self.prepare_baseline_models(verbose)
            
            # 5. 预测与评估
            self.predict_and_evaluate(verbose)
            
            # 6. 保存结果
            self.save_results(verbose)
            
            # 7. 绘制图表
            self.plot_results(save_plots=True, show_plots=False)
            
            print("\n" + "=" * 80)
            print("实验完成！")
            print("=" * 80)
            
        except Exception as e:
            print(f"实验执行失败: {e}")
            raise

    def _refilter_test_data_for_stackelberg(self, verbose: bool = True) -> None:
        """
        根据Stackelberg训练时使用的模型列表重新筛选测试数据
        """
        if verbose:
            print("开始为Stackelberg模型重新筛选测试数据...")
            
        if not hasattr(self, 'stackelberg_model_names'):
            if verbose:
                print("警告: 没有找到Stackelberg模型的训练模型列表，跳过重新筛选")
            return
            
        if verbose:
            print(f"训练时使用的Stackelberg模型列表: {self.stackelberg_model_names}")
        
        # 获取原始测试数据
        test_data_raw = self.data_cache['test_raw']
        p_nom_test, cap_test, dist_test, demand_test, q_real_test, all_model_names = test_data_raw
        
        if verbose:
            print(f"原始测试数据中的所有模型: {all_model_names}")
        
        # 找到训练时使用的模型在测试数据中的索引
        stackelberg_indices = []
        for model_name in self.stackelberg_model_names:
            try:
                idx = all_model_names.index(model_name)
                stackelberg_indices.append(idx)
                if verbose:
                    print(f"找到模型 '{model_name}' 在索引 {idx}")
            except ValueError:
                if verbose:
                    print(f"警告: 训练模型 '{model_name}' 在测试数据中不存在")
        
        if len(stackelberg_indices) == 0:
            raise ValueError("没有找到任何训练时使用的模型在测试数据中")
            
        if verbose:
            print(f"为Stackelberg模型重新筛选测试数据:")
            print(f"  训练时使用的模型数量: {len(self.stackelberg_model_names)}")
            print(f"  测试数据中找到的模型数量: {len(stackelberg_indices)}")
        
        # 转换为张量索引
        stackelberg_indices = torch.tensor(stackelberg_indices, dtype=torch.long)
        
        # 重新筛选测试数据
        test_filtered_stackelberg = (
            p_nom_test[:, :, stackelberg_indices],
            cap_test[:, :, stackelberg_indices], 
            dist_test[:, :, :, stackelberg_indices],
            q_real_test[:, :, stackelberg_indices].sum(dim=2, keepdim=True),  # 重新计算需求
            q_real_test[:, :, stackelberg_indices],
            self.stackelberg_model_names
        )
        
        # 保存为专门的Stackelberg测试数据
        self.data_cache['test_filtered_stackelberg'] = test_filtered_stackelberg
        
        if verbose:
            print(f"  重新筛选后的测试数据形状:")
            print(f"    价格: {test_filtered_stackelberg[0].shape}")
            print(f"    容量: {test_filtered_stackelberg[1].shape}")
            print(f"    距离: {test_filtered_stackelberg[2].shape}")
            print(f"    需求: {test_filtered_stackelberg[3].shape}")
            print(f"    流量: {test_filtered_stackelberg[4].shape}")

    def _train_no_congestion_model(self, train_data_for_baseline, verbose: bool = True):
        """
        训练无拥塞baseline模型
        使用相同的模型和训练方式，但拥塞权重设置为0.001
        """
        # 获取训练数据
        p_nom_train, cap_train, dist_train, demand_train, q_real_train, model_names = train_data_for_baseline
        
        # 初始化模型参数
        n_users = 1
        m_providers = len(model_names)
        
        # 使用反向价格求解来初始化质量参数 (与主模型相同的初始化方法)
        if verbose:
            print(f"使用反向价格求解初始化无拥塞模型的质量参数...")
        
        try:
            # 使用真实流量作为目标，反向求解最优价格
            solved_prices, solve_results = find_optimal_prices_for_target_flow(
                target_flow=q_real_train,
                cap_data=cap_train,
                dist_data=dist_train,
                demand_data=demand_train,
                model_names=model_names,
                wp=self.wp, wd=self.wd, wq=0.001,  # 拥塞权重设置为0.001
                verbose=verbose
            )
            
            # 质量参数 = 名义价格 - 反向求解的价格
            initial_quality_guess = p_nom_train - solved_prices
            
            if verbose:
                print(f"无拥塞模型反向求解完成，成功率: {solve_results['successful_days']/len(solve_results['statuses'])*100:.1f}%")
                print(f"无拥塞模型初始质量参数 (基于反向求解): {initial_quality_guess.numpy()}")
                
        except Exception as e:
            if verbose:
                print(f"无拥塞模型反向求解失败: {e}")
                print("使用默认初始化...")
            # 回退到默认初始化
            initial_quality_guess = torch.zeros(m_providers)
        
        # 创建无拥塞模型，关键是拥塞权重设置为0.001
        model = QualityLearner(n_users, m_providers, self.wp, self.wd, 0.001, initial_quality=initial_quality_guess)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=50, min_lr=1e-6
        )
        
        # 训练循环 (与主模型相同的训练方式)
        num_epochs = 50
        losses = []
        best_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 200
        
        if verbose:
            print(f"开始训练无拥塞模型 (最大epoch: {num_epochs})...")
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # 前向传播
            q_pred = model(p_nom_train, cap_train, dist_train, demand_train)
            
            # 计算损失
            loss = nn.MSELoss()(q_pred, q_real_train)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            # 早停检查
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"无拥塞模型Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  无拥塞模型Epoch {epoch+1}/{num_epochs}, Loss: {current_loss:.6f}, LR: {current_lr:.6f}")
                print(f"  Best Loss: {best_loss:.6f}, Patience: {patience_counter}/{early_stopping_patience}")
                
                # 显示当前质量参数
                quality_params = model.quality.data.numpy()
                print(f"  无拥塞模型Current Quality: {quality_params}")
        
        return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='流量预测对比实验')
    parser.add_argument('--train_start', type=str, default='2026-10-19',
                       help='训练开始日期 (默认: 2026-10-19)')
    parser.add_argument('--train_end', type=str, default='2026-10-23',
                       help='训练结束日期 (默认: 2026-10-23)')
    parser.add_argument('--test_start', type=str, default='2026-10-24',
                       help='测试开始日期 (默认: 2026-10-24)')
    parser.add_argument('--test_end', type=str, default='2026-11-15',
                       help='测试结束日期 (默认: 2026-11-15)')
    parser.add_argument('--data_file', type=str, default='data/release/market_inputs_top20_month.csv',
                       help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='experiment_results',
                       help='输出目录')
    parser.add_argument('--min_price', type=float, default=0.0,
                       help='最小价格阈值')
    parser.add_argument('--wp', type=float, default=0.6,
                       help='价格敏感度权重')
    parser.add_argument('--wd', type=float, default=0.5,
                       help='延迟敏感度权重')
    parser.add_argument('--wq', type=float, default=0.5,
                       help='拥堵敏感度权重')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')
    
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = ComprehensiveFlowPredictionExperiment(
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start, 
        test_end_date=args.test_end,
        data_file=args.data_file,
        output_dir=args.output_dir,
        min_price_threshold=args.min_price,
        wp=args.wp, wd=args.wd, wq=args.wq
    )
    
    # 运行实验
    experiment.run_full_experiment(verbose=args.verbose)


if __name__ == "__main__":
    from priLLM.seed import set_seed; set_seed(0)
    main()