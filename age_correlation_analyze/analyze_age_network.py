import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from Correlation import CorrelationAnalyzer # 确保 Correlation.py 在同一个目录下
import os

def plot_network_age_correlation(data_path, save_path):
    """
    计算并绘制全局网络特征与年龄的相关性散点图
    """
    # --- 1. 加载数据 ---
    try:
        df = pd.read_csv(data_path, sep='\t')
    except FileNotFoundError:
        print(f"错误：找不到文件 {data_path}")
        return

    # 识别网络特征列（以 'alpha' 结尾的特征）
    network_features = [col for col in df.columns if 'alpha' in col and 'delirium' not in col]
    states = ['open-eyes', 'close-eyes']

    # --- 2. 绘制散点图 ---
    plt.style.use(['science', 'nature'])

    for state in states:
        state_data = df[df['state'] == state]

        # 为每个网络特征画一张图
        for feature in network_features:
            plt.figure(figsize=(10, 8))

            # 提取特征和年龄数据，并处理缺失值
            feature_data = state_data[feature].values
            age_data = state_data['age'].values

            valid_indices = ~pd.isna(age_data) & ~pd.isna(feature_data)
            filtered_age_data = age_data[valid_indices]
            filtered_feature_data = feature_data[valid_indices]

            if len(filtered_age_data) < 2:
                print(f"跳过 {feature} ({state})，有效数据点不足。")
                continue

            # --- 3. 计算相关性并标注 ---
            analyzer = CorrelationAnalyzer(filtered_feature_data, filtered_age_data)
            corr_result = analyzer.calculate_correlation()
            corr_coef = corr_result['correlation']
            p_value = corr_result['p_value']
            
            # 绘制散点图和回归线
            sns.regplot(x=filtered_age_data, y=filtered_feature_data, 
                        scatter_kws={'alpha':0.6}, line_kws={'color':'red'})

            # 在图的右上角添加文字标注
            stats_text = (f"Correlation Method: {corr_result['method']}\n"
                          f"Correlation: {corr_coef:.3f}\n"
                          f"P-value: {p_value:.3f}")
                          
            # 根据p值决定文字颜色
            text_color = 'red' if p_value < 0.05 else 'black'
            
            plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                     fontsize=14, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     color=text_color)

            plt.title(f'Age Correlation for {feature} ({state})', fontsize=16)
            plt.xlabel('Age', fontsize=12)
            plt.ylabel(feature.replace('_', ' ').title(), fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()

            # 保存图片
            save_dir = os.path.join(save_path, 'Network')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'age_corr_{state}_{feature}.jpg'), dpi=300)
            plt.close()

if __name__ == "__main__":
    # 设置文件路径
    os.makedirs('bo_save_jpg/AgeCorrelation', exist_ok=True)
    data_file_path = "bo_save_csv/data/process_network_median.tsv"
    save_image_path = "./bo_save_jpg/AgeCorrelation"
    
    plot_network_age_correlation(data_file_path, save_image_path)
    print("Network age correlation analysis complete.")