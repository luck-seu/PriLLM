import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# 1. 设置 IEEE 论文格式的全局绘图参数
#    (严格按照您提供的规范)
# ==============================================================================
def set_ieee_plot_style():
    """Applies strict IEEE publication-quality plotting styles."""
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6, # 稍微调小图例字体以适应紧凑图形
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05, # 稍微减小边距
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# IEEE 双栏格式的标准尺寸（英寸）
SINGLE_COLUMN_WIDTH = 3.4    # 单栏宽度 (通常在3.3到3.5之间)
DOUBLE_COLUMN_WIDTH = 7.16   # 双栏宽度
GOLDEN_RATIO = (1 + 5**0.5) / 2

# ==============================================================================
# 2. 生成模拟实验数据
#    (请将这部分替换为您的真实测量结果)
# ==============================================================================
def load_experiment_data():
    """
    从实验结果文件中加载真实的实验数据
    """
    import json
    import os
    
    # 尝试加载真实的实验数据
    data_file = 'experiment_results/experiment_data.json'
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            print("Loaded real experiment data from experiment_results/experiment_data.json")
            
            # 映射方法名称以匹配原始格式
            method_mapping = {
                'Your Method (CVX)': 'Differentiable Layer',
                'Your Method (CVX_Initialized)': 'Differentiable Layer with Init.',
                'Baseline 1 (Unrolled)': 'Unrolled BPTT',
                'Baseline 2 (Blackbox)': 'Blackbox',
                'Baseline 3 (REINFORCE)': 'REINFORCE'
            }
            
            methods = [method_mapping.get(m, m) for m in data['methods']]
            difficulties = data['difficulties']
            
            # 重命名方法以匹配绘图格式
            accuracy_data = {}
            runtime_data = {}
            
            for old_name, new_name in method_mapping.items():
                if old_name in data['accuracy_data']:
                    accuracy_data[new_name] = np.array(data['accuracy_data'][old_name])
                    runtime_data[new_name] = np.array(data['runtime_data'][old_name])
            
            return methods, difficulties, accuracy_data, runtime_data
            
        except Exception as e:
            raise RuntimeError(f"Could not read {data_file}: {e}") from e

    raise FileNotFoundError(
        f"{data_file} not found. Run experiments/calibration/table2_learning_methods.py first; "
        "this script never substitutes placeholder data.")


# ==============================================================================
# 3. 绘图函数
# ==============================================================================

def plot_accuracy_vs_difficulty(methods, difficulties, data):
    """图1: 绘制不同方法在不同难度下的准确性对比 (分组柱状图)"""
    fig_height = SINGLE_COLUMN_WIDTH/2 / GOLDEN_RATIO
    plt.figure(figsize=(SINGLE_COLUMN_WIDTH/2, fig_height))
    
    x = np.arange(len(difficulties))
    width = 0.2
    n_methods = len(methods)
    
    for i, method in enumerate(methods):
        offset = width * (i - (n_methods - 1) / 2)
        plt.bar(x + offset, data[method], width, label=method)
    
    # TODO
    # 配色太丑，也需要换
    plt.yscale('log') # 使用对数坐标轴，因为运行时间差异可能很大
    plt.ylabel('Quality MSE (Lower is Better)')
    plt.title('Accuracy across Problem Scales')
    plt.xticks(x, difficulties)
    plt.legend()
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    
    # 保存图形
    plt.savefig('accuracy_vs_difficulty.pdf')
    print("已生成: accuracy_vs_difficulty.pdf")


def plot_runtime_vs_difficulty(methods, difficulties, data):
    """图2: 绘制不同方法在不同难度下的效率对比 (分组柱状图, Y轴对数)"""
    fig_height = SINGLE_COLUMN_WIDTH / GOLDEN_RATIO
    plt.figure(figsize=(SINGLE_COLUMN_WIDTH, fig_height))

    x = np.arange(len(difficulties))
    width = 0.2
    n_methods = len(methods)
    
    for i, method in enumerate(methods):
        offset = width * (i - (n_methods - 1) / 2)
        plt.bar(x + offset, data[method], width, label=method)

    plt.yscale('log') # 使用对数坐标轴，因为运行时间差异可能很大
    plt.ylabel('Time per Epoch (s)')
    # plt.title('Efficiency across Problem Scales')
    plt.xticks(x, difficulties)
    plt.legend()
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    
    # 保存图形
    plt.savefig('runtime_vs_difficulty.pdf')
    print("已生成: runtime_vs_difficulty.pdf")


def plot_tradeoff_scatter(methods, accuracy_data, runtime_data):
    """图3: 绘制准确性-效率权衡散点图 (Pareto 分析)"""
    fig_height = SINGLE_COLUMN_WIDTH / GOLDEN_RATIO
    plt.figure(figsize=(SINGLE_COLUMN_WIDTH, fig_height*0.9))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', '^', 'd', 'v', 'p', '*']  # 扩展标记列表以支持更多方法
    
    for i, method in enumerate(methods):
        runtimes = runtime_data[method]
        accuracies = accuracy_data[method]
        
        # 绘制散点和连接线
        plt.plot(runtimes, accuracies, marker=markers[i], linestyle='--', 
                 color=colors[i], label=method, alpha=0.8)

    plt.xscale('log') # 运行时间通常用对数坐标更清晰
    plt.xlabel('Time to Accuracy (s)')
    plt.yscale('log') # 准确性通常用对数坐标更清晰
    plt.ylabel('MSE')
    # plt.title('Accuracy-Efficiency Trade-off')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 在左下角标注 "理想用户群"
    plt.text(plt.xlim()[0] * 1.5, plt.ylim()[0] * 1.5, '← Ideal Region', 
             ha='left', va='bottom', fontsize=6, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.2))

    # 保存图形
    plt.savefig('tradeoff_scatter.pdf')
    print("已生成: tradeoff_scatter.pdf")

# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 应用绘图样式
    set_ieee_plot_style()
    
    # 获取实验数据（优先使用真实数据，否则使用模拟数据）
    methods, difficulties, accuracy_data, runtime_data = load_experiment_data()
    
    # 创建保存图形的目录
    if not os.path.exists('figures'):
        os.makedirs('figures')
    os.chdir('figures') # 将图形保存在子目录中
    
    # 生成并保存所有图形
    plot_accuracy_vs_difficulty(methods, difficulties, accuracy_data)
    plot_runtime_vs_difficulty(methods, difficulties, runtime_data)
    plot_tradeoff_scatter(methods, accuracy_data, runtime_data)
    
    # 如果希望在脚本运行时显示图形，取消下面一行的注释
    # plt.show()
    
    os.chdir('..') # 返回上级目录
    print("\n所有图形已成功生成在 'figures' 文件夹中。")

