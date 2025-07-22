import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Statis import StatisticalComparison
import scienceplots

def read_and_split_data(file_path):
    """读取数据并按状态和组别拆分"""
    df = pd.read_csv(file_path, sep='\t')

    # 按状态拆分
    open_eyes_data = df[df['state'] == 'open-eyes']
    close_eyes_data = df[df['state'] == 'close-eyes']

    # 进一步按谵妄组拆分
    open_pod = open_eyes_data[open_eyes_data['delirium'] == 1]
    open_nopod = open_eyes_data[open_eyes_data['delirium'] == 0]

    close_pod = close_eyes_data[close_eyes_data['delirium'] == 1]
    close_nopod = close_eyes_data[close_eyes_data['delirium'] == 0]

    return {
        'open-eyes': {'POD': open_pod, 'noPOD': open_nopod},
        'close-eyes': {'POD': close_pod, 'noPOD': close_nopod}
    }

def create_box_plots(all_data, output_path):
    """为网络特征创建箱线图"""
    plt.style.use(['science', 'nature'])

    # 定义要分析的网络特征
    feature_names = ['small_worldness_alpha', 'global_efficiency_alpha', 
                     'modularity_alpha', 'characteristic_path_length_alpha']

    # 为睁眼和闭眼分别创建图形
    for state in ['open-eyes', 'close-eyes']:
        fig, axes = plt.subplots(1, len(feature_names), figsize=(20, 5), sharey=False)
        fig.suptitle(f'Brain Network Features Comparison ({state})', fontsize=16)

        state_data = all_data[state]

        for i, feature in enumerate(feature_names):
            ax = axes[i]

            # 获取两组的数据
            pod_values = state_data['POD'][feature].dropna().tolist()
            nopod_values = state_data['noPOD'][feature].dropna().tolist()

            if not pod_values or not nopod_values:
                ax.set_title(f"{feature}\n(No data)")
                continue

            box_data = [nopod_values, pod_values]

            # 绘制箱线图
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)

            # 设置颜色
            colors = ['#2ecc71', '#e74c3c'] # noPOD: 绿色, POD: 红色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # 计算并标注P值
            comparison = StatisticalComparison(nopod_values, pod_values)
            p_value = comparison.perform_statistical_test()

            y_max = max(max(pod_values), max(nopod_values))
            y_min = min(min(pod_values), min(nopod_values))
            p_value_height = y_max + (y_max - y_min) * 0.1

            ax.plot([1, 2], [p_value_height, p_value_height], 'k-', linewidth=1)
            text_color = 'red' if p_value < 0.05 else 'black'
            ax.text(1.5, p_value_height, f'p={p_value:.3f}',
                    ha='center', va='bottom', color=text_color, fontsize=12)

            ax.set_title(feature, fontsize=12)
            ax.set_xticklabels(['noPOD', 'POD'])
            ax.tick_params(axis='x', labelsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存图片
        plt.savefig(os.path.join(output_path, f'Network_Features_{state}_boxplot.jpg'),
                    bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # 设置输入输出路径
    data_path = "bo_save_csv/data/process_network_median.tsv"
    output_path = "./bo_save_jpg/Box"

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 读取并处理数据
    all_data = read_and_split_data(data_path)

    # 创建图表
    create_box_plots(all_data, output_path)
    print(f"网络特征箱线图已生成并保存到: {output_path}")

if __name__ == "__main__":
    main()