import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import spearmanr, pearsonr
import os
from typing import Dict, List, Optional, Tuple, Any
import json
import matplotlib.dates as mdates

# 设置IEEE论文格式的图形参数
plt.rcParams.update({
    'font.size': 8,          # 基础字体大小，与正文相同
    'axes.titlesize': 8,     # 标题字体大小
    'axes.labelsize': 8,     # 轴标签字体大小
    'xtick.labelsize': 6,    # x轴刻度标签字体大小
    'ytick.labelsize': 6,    # y轴刻度标签字体大小
    'legend.fontsize': 8,    # 图例字体大小
    'lines.linewidth': 1.5,   # 线条宽度
    'lines.markersize': 4,    # 标记大小
    'figure.dpi': 300,        # 图形分辨率
    'savefig.dpi': 300,       # 保存图形分辨率
    'savefig.bbox': 'tight',  # 紧凑保存
    'savefig.pad_inches': 0.1, # 保存时的边距
    'font.family': 'serif',   # 使用衬线字体
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,       # 确保字体嵌入PDF
    'ps.fonttype': 42,        # 确保字体嵌入PS
})

# IEEE双栏格式的标准尺寸（英寸）
SINGLE_COLUMN_WIDTH = 3.2    # 单栏宽度
DOUBLE_COLUMN_WIDTH = 7.14   # 双栏宽度
GOLDEN_RATIO = (1 + 5**0.5) / 2  # 黄金比例用于高度计算

class ExperimentPlotter:
    """实验结果可视化类 - IEEE论文格式"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        """
        初始化绘图器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义方法名称映射（适合论文的简洁名称）
        self.method_names_display = {
            'stackelberg': 'Stackelberg',
            'xgboost': 'XGBoost',
            'baseline_capacity': 'Capacity',
            'baseline_no_congestion': 'No Congestion',
            'baseline_price': 'Price',
            'baseline_latency': 'Latency',
            'baseline_market_share': 'Market Share'
        }
        
        # 定义IEEE论文友好的颜色方案（适合黑白打印）
        self.colors = ['#000000', '#555555', '#888888', '#BBBBBB', '#FF0000', '#0000FF', '#00AA00']
        self.linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
        self.markers = ['o', 's', '^', 'v', 'D', 'p', 'h']
        
        # 模型特定颜色（支持更多LLM）
        self.model_colors = ['#000000', '#555555', '#888888', '#BB0000', '#0000BB', '#00BB00', '#BB00BB', '#BBBB00', '#00BBBB', '#BB5500']
    
    def plot_correlation_scatter(self, 
                               q_real: np.ndarray, 
                               predictions: Dict[str, np.ndarray],
                               model_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
        """
        绘制真实vs预测流量散点图 - IEEE单栏格式（每个子图都是单栏宽度）
        
        Args:
            q_real: 真实流量数据，形状为 (days, providers) 或展平后的一维数组
            predictions: 预测结果字典，键为方法名，值为预测数据
            model_names: 模型名称列表
            save_path: 保存路径，如果为None则使用默认路径
            show_plot: 是否显示图形
        """
        # 处理数据形状
        if len(q_real.shape) == 2:
            q_real_flat = q_real.flatten()
        else:
            q_real_flat = q_real
            
        # 为每个方法创建单独的单栏宽度子图
        num_methods = len(predictions)
        
        # 设置1/2单栏宽度
        fig_width = SINGLE_COLUMN_WIDTH/2
        fig_height = fig_width  # 保持正方形比例，适合散点图
        
        # 为每个方法创建单独的图
        for i, (method, q_pred) in enumerate(predictions.items()):
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            
            # 处理预测数据形状
            if len(q_pred.shape) == 3:
                q_pred_flat = q_pred.squeeze(0).flatten()
            elif len(q_pred.shape) == 2:
                q_pred_flat = q_pred.flatten()
            else:
                q_pred_flat = q_pred
            
            # 计算统计指标
            try:
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(q_real_flat, q_pred_flat)
                mae = mean_absolute_error(q_real_flat, q_pred_flat)
                corr, _ = pearsonr(q_real_flat, q_pred_flat)
            except ImportError:
                r2 = 0.0
                mae = np.mean(np.abs(q_real_flat - q_pred_flat))
                corr = np.corrcoef(q_real_flat, q_pred_flat)[0, 1]
            
            # 绘制散点图
            ax.scatter(q_real_flat, q_pred_flat, 
                      alpha=0.6, 
                      s=12, 
                      color=self.colors[i % len(self.colors)],
                      marker=self.markers[i % len(self.markers)],
                      edgecolors='black',
                      linewidth=0.3)
            
            # 绘制45度参考线
            max_val = max(q_real_flat.max(), q_pred_flat.max())
            min_val = min(q_real_flat.min(), q_pred_flat.min())
            margin = (max_val - min_val) * 0.1
            ax.plot([min_val - margin, max_val + margin], 
                   [min_val - margin, max_val + margin], 
                   'k--', alpha=0.8, linewidth=1)
            
            # 先设置轴的范围（基于数据范围，不受拟合线影响）
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            
            # 添加拟合线（保持原始形状，允许延伸到图外）
            z = np.polyfit(q_real_flat, q_pred_flat, 1)
            p = np.poly1d(z)
            # 使用数据范围创建拟合线，不裁剪形状
            x_fit = np.array([min_val, max_val])
            y_fit = p(x_fit)
            ax.plot(x_fit, y_fit, 
                   color=self.colors[i % len(self.colors)], 
                   linestyle='-', alpha=0.8, linewidth=1, label='Fitted line')
            
            
            # 设置标题和标签
            method_name = self.method_names_display.get(method, method)
            # ax.set_title(f'{method_name}\n$R^2$={r2:.3f}, MAE={mae:.2f}', 
            #             fontsize=11, pad=8, fontweight='bold')
            ax.set_xlabel('True Flow (B tokens)', fontsize=8)
            ax.set_ylabel('Fitted Flow (B tokens)', fontsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # 设置相等的纵横比
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            
            # 为每个方法创建单独的保存路径
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                individual_save_path = os.path.join(self.output_dir, f"correlation_scatter_{method}_{timestamp}.pdf")
            else:
                # 从原始路径中提取文件名和扩展名
                base_path = save_path.replace('.pdf', '')
                individual_save_path = f"{base_path}_{method}.pdf"
            
            plt.savefig(individual_save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
            plt.savefig(individual_save_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
            
            if show_plot:
                plt.show()
            plt.close()
    
    def plot_model_flow_prediction(self, 
                                 q_real: np.ndarray,
                                 predictions: Dict[str, np.ndarray],
                                 test_start_date: str,
                                 test_days: int,
                                 model_names: List[str],
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True) -> None:
        """
        绘制模型流量预测对比图 - IEEE格式，legend分离
        
        Args:
            q_real: 真实流量数据，形状为 (days, providers)
            predictions: 预测结果字典
            test_start_date: 测试开始日期 (YYYY-MM-DD)
            test_days: 测试天数
            model_names: 模型名称列表
            save_path: 保存路径
            show_plot: 是否显示图形
        """
        # 检查是否有Stackelberg预测
        if 'stackelberg' not in predictions:
            print("Warning: No Stackelberg predictions found, skipping flow prediction plot")
            return
            
        # 创建日期标签
        start_date = datetime.strptime(test_start_date, '%Y-%m-%d')
        dates = [start_date + timedelta(days=i) for i in range(test_days)]
        date_strings = [date.strftime('%m-%d') for date in dates]
        
        # 使用单栏宽度创建图形
        fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH/GOLDEN_RATIO))
        
        # 获取流量前3的模型
        top_3_indices = np.argsort(q_real.mean(axis=0))[-3:]
        
        # 获取Stackelberg预测结果
        q_pred_stack = predictions['stackelberg']
        if len(q_pred_stack.shape) == 3:
            q_pred_stack = q_pred_stack.squeeze(0)
        
        # 定义每个LLM的marker
        llm_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # 存储图例信息
        legend_handles = []
        legend_labels = []
        
        # 为每个模型绘制真实值和Stackelberg预测
        for i, model_idx in enumerate(top_3_indices):
            model_name = model_names[model_idx]
            color = self.model_colors[i % len(self.model_colors)]
            marker = llm_markers[i % len(llm_markers)]
            model_letter = chr(ord('A') + i)  # A, B, C for models
            
            # 绘制真实值（实线）
            line1 = ax.plot(dates, q_real[:, model_idx], 
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
            line2 = ax.plot(dates, q_pred_stack[:, model_idx], 
                   color=color, 
                   linestyle='--',
                   linewidth=1.2,
                   marker=marker,
                   markersize=4,
                   markerfacecolor='white',
                   markeredgecolor=color,
                   markeredgewidth=1,
                   label=f'Model {model_letter} (Pred)')
            
            # 收集图例信息
            legend_handles.extend([line1[0], line2[0]])
            model_letter = chr(ord('A') + i)  # A, B, C for models
            legend_labels.extend([f'Model {model_letter} (True)', f'Model {model_letter} (Pred)'])
        
        # 设置图形属性（不包含legend）
        # ax.set_title('Flow Fitting of Top-3 Models', fontweight='bold', pad=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow (B tokens)')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # 设置x轴日期格式，动态调整间隔
        date_range = len(dates)
        if date_range <= 7:
            interval = 1  # 7天内每天显示
        elif date_range <= 14:
            interval = 2  # 14天内每2天显示
        elif date_range <= 21:
            interval = 3  # 21天内每3天显示
        else:
            interval = max(1, date_range // 7)  # 最多显示7个刻度
            
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 旋转x轴标签，避免重叠
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"model_flow_prediction_{timestamp}.pdf")
        
        # 保存主图（不包含legend）
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.savefig(save_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
        
        # 创建并保存单独的legend
        legend_save_path = save_path.replace('.pdf', '_legend.pdf')
        legend_png_path = save_path.replace('.pdf', '_legend.png')
        
        # 创建legend图片 - 使用IEEE单栏宽度
        fig_legend = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 0.6))
        legend = fig_legend.legend(legend_handles, legend_labels, loc='center', ncol=3, fontsize=4, 
                                  frameon=True, fancybox=False, shadow=False,
                                  columnspacing=1.0, handletextpad=0.5)
        fig_legend.canvas.draw()
        
        # 保存legend
        plt.savefig(legend_save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.savefig(legend_png_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig_legend)
        
        if show_plot:
            plt.show()
        plt.close()
        
        if show_plot:
            plt.show()
        plt.close()
    
    def load_results_from_json(self, json_path: str) -> Dict[str, Any]:
        """从JSON文件加载实验结果"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_ieee_performance_comparison(self, 
                                         metrics_data: Dict[str, Any],
                                         save_path: Optional[str] = None) -> None:
        """
        创建IEEE格式的性能比较图表
        
        Args:
            metrics_data: 包含各方法性能指标的数据
            save_path: 保存路径
        """
        # 提取指标数据
        methods = list(metrics_data.keys())
        r2_scores = [metrics_data[method].get('r2_score', 0) for method in methods]
        mae_scores = [metrics_data[method].get('mae', 0) for method in methods]
        
        # 创建双子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH/2.5))
        
        # 方法名称映射
        method_labels = [self.method_names_display.get(method, method) for method in methods]
        
        # R² 分数条形图
        bars1 = ax1.bar(range(len(methods)), r2_scores, 
                       color=['black', 'gray', 'darkgray', 'lightgray'][:len(methods)],
                       edgecolor='black', linewidth=0.5)
        ax1.set_title('$R^2$ Score', fontweight='bold')
        ax1.set_ylabel('$R^2$')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(method_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.0)
        
        # 在条形图上添加数值
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=7)
        
        # MAE 分数条形图
        bars2 = ax2.bar(range(len(methods)), mae_scores,
                       color=['black', 'gray', 'darkgray', 'lightgray'][:len(methods)],
                       edgecolor='black', linewidth=0.5)
        ax2.set_title('MAE', fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(method_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 在条形图上添加数值
        for bar, score in zip(bars2, mae_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(mae_scores)*0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"performance_comparison_{timestamp}.pdf")
        
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.savefig(save_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def create_ieee_method_comparison_table(self, 
                                          metrics_data: Dict[str, Any],
                                          save_path: Optional[str] = None) -> None:
        """
        创建IEEE格式的方法比较表格（以图片形式保存）
        
        Args:
            metrics_data: 包含各方法性能指标的数据
            save_path: 保存路径
        """
        # 准备表格数据
        methods = list(metrics_data.keys())
        method_labels = [self.method_names_display.get(method, method) for method in methods]
        
        table_data = []
        for method in methods:
            data = metrics_data[method]
            row = [
                self.method_names_display.get(method, method),
                f"{data.get('r2_score', 0):.3f}",
                f"{data.get('mae', 0):.2f}",
                f"{data.get('rmse', 0):.2f}",
                f"{data.get('correlation', 0):.3f}"
            ]
            table_data.append(row)
        
        # 创建表格图
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH*0.6))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=['Method', '$R^2$', 'MAE', 'RMSE', 'Correlation'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # 设置表头样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6E6')
            else:
                cell.set_facecolor('#F5F5F5')
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
        
        plt.title('Performance Comparison of Different Methods', 
                 fontweight='bold', pad=20)
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"method_comparison_table_{timestamp}.pdf")
        
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.savefig(save_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def plot_from_saved_results(self, 
                              results_json_path: str,
                              save_dir: Optional[str] = None,
                              show_plot: bool = True) -> None:
        """
        从保存的实验结果文件中绘制图表
        
        Args:
            results_json_path: 结果JSON文件路径（metrics文件）或目录路径
            save_dir: 保存目录，如果为None则使用默认目录
            show_plot: 是否显示图形
        """
        import pickle
        import os
        
        if save_dir:
            self.output_dir = save_dir
            
        # 检查路径是文件还是目录
        if os.path.isfile(results_json_path):
            # 如果是单个JSON文件，尝试从同目录加载对应的pkl文件
            base_dir = os.path.dirname(results_json_path)
            file_base = os.path.basename(results_json_path).replace('metrics_', '').replace('.json', '')
        else:
            # 如果是目录，查找最新的文件
            base_dir = results_json_path
            files = [f for f in os.listdir(base_dir) if f.startswith('metrics_') and f.endswith('.json')]
            if not files:
                raise ValueError("未找到metrics文件")
            latest_file = sorted(files)[-1]
            results_json_path = os.path.join(base_dir, latest_file)
            file_base = latest_file.replace('metrics_', '').replace('.json', '')
        
        try:
            # 加载JSON指标文件
            with open(results_json_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            # 查找对应的pkl文件
            predictions_file = os.path.join(base_dir, f"predictions_{file_base}.pkl")
            true_data_file = os.path.join(base_dir, f"true_data_{file_base}.pkl")
            
            if not os.path.exists(predictions_file) or not os.path.exists(true_data_file):
                print(f"警告: 找不到对应的pkl文件。期望的文件:")
                print(f"  - {predictions_file}")
                print(f"  - {true_data_file}")
                return
            
            # 加载预测和真实数据
            with open(predictions_file, 'rb') as f:
                predictions_data = pickle.load(f)
            
            with open(true_data_file, 'rb') as f:
                true_data = pickle.load(f)
            
            # 提取数据
            q_real = true_data['q_real']  # 形状应该为 (days, providers)
            predictions = {}
            
            # 处理预测数据格式
            for method, pred_data in predictions_data.items():
                if isinstance(pred_data, np.ndarray):
                    predictions[method] = pred_data
                else:
                    # 如果是torch tensor或其他格式，转换为numpy
                    predictions[method] = np.array(pred_data)
            
            # 确保q_real是2D数组
            if len(q_real.shape) == 3:
                q_real = q_real.squeeze(0)  # 移除batch维度
            
            model_names = true_data.get('model_names', [f'Model_{i}' for i in range(q_real.shape[1])])
            test_start_date = true_data.get('test_start_date', '2026-10-26')
            test_days = q_real.shape[0]
            
            # 绘制各种图表
            self.plot_correlation_scatter(q_real, predictions, model_names)
            self.plot_model_flow_prediction(q_real, predictions, test_start_date, test_days, model_names)
            self.create_ieee_performance_comparison(metrics_data)
            self.create_ieee_method_comparison_table(metrics_data)
            
            print(f"已从实验结果生成所有图表，保存到 {self.output_dir}")
            
        except Exception as e:
            print(f"加载或处理数据时出错: {e}")
            import traceback
            traceback.print_exc()

# 保留原有的示例绘图函数 - IEEE格式

if __name__ == "__main__":
    # 示例用法
    plotter = ExperimentPlotter()
    
    results_json_path = "./experiment_results/"
    save_dir = "./experiment_plots/"
    show_plots = False

    plotter.plot_from_saved_results(results_json_path, save_dir, show_plots)