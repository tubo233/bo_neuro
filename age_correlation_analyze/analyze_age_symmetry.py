import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from Correlation import CorrelationAnalyzer # 确保 Correlation.py 在同一个目录下
import os

def plot_symmetry_age_correlation(data_path, save_path):
    """
    计算并绘制Symmetry特征与年龄的相关性条形图
    """
    # --- 1. 加载数据 ---
    df = pd.read_csv(data_path, sep='\t')
    
    # 获取所有对称性特征和通道对
    features = ['total_ratio','total_dB','delta_ratio','delta_dB','theta_ratio','theta_dB',
                'alpha_ratio','alpha_dB','beta_ratio','beta_dB','gamma_ratio','gamma_dB']
    channel_pairs = sorted(df['target_channels'].unique())
    states = ['open-eyes', 'close-eyes']
    
    # --- 2. 计算相关性 ---
    results = []
    for state in states:
        state_data = df[df['state'] == state]
        for pair in channel_pairs:
            for feature in features:
                # 提取特定通道对和特征的数据
                subset_data = state_data[state_data['target_channels'] == pair]
                
                # 获取特征和年龄数据，并处理缺失值
                feature_data = subset_data[feature].values
                age_data = subset_data['age'].values
                
                valid_indices = ~pd.isna(age_data)
                filtered_age_data = age_data[valid_indices]
                filtered_feature_data = feature_data[valid_indices]
                
                # 计算相关性
                if len(filtered_age_data) > 1:
                    analyzer = CorrelationAnalyzer(filtered_feature_data, filtered_age_data)
                    corr_result = analyzer.calculate_correlation()
                    
                    results.append({
                        'state': state,
                        'channel_pair': pair,
                        'feature': feature,
                        'correlation': corr_result['correlation'],
                        'p_value': corr_result['p_value']
                    })

    results_df = pd.DataFrame(results)
    
    # --- 3. 绘制条形图 ---
    plt.style.use(['science', 'nature'])
    
    for state in states:
        state_results = results_df[results_df['state'] == state]
        
        # 每个特征画一张图
        for feature in features:
            feature_results = state_results[state_results['feature'] == feature].sort_values(by='correlation', ascending=False)
            
            # 【修改一】让图形高度根据通道对的数量动态变化
            num_pairs = len(feature_results)
            # 为每个条形分配约0.5英寸的高度，并设置一个最小高度8
            figure_height = max(8, 0.5 * num_pairs)
            plt.figure(figsize=(12, figure_height))
            
            # 【修改二】消除seaborn警告
            # 将y轴变量'channel_pair'同时赋给hue，并关闭图例
            barplot = sns.barplot(x='correlation', y='channel_pair', data=feature_results, palette="vlag", hue='channel_pair', legend=False)
            
            # 标记显著的p值
            # 【修改三】使用enumerate确保星号位置准确
            for i, row in enumerate(feature_results.itertuples()):
                if row.p_value < 0.05:
                    # 在条形末端标记星号
                    # 根据相关性的正负决定星号在条的左边还是右边
                    ha_align = 'left' if row.correlation >= 0 else 'right'
                    offset = 0.01 if row.correlation >= 0 else -0.01
                    plt.text(row.correlation + offset, i, '*', color='black', ha=ha_align, va='center', fontsize=16)

            plt.title(f'Age Correlation for {feature} ({state})', fontsize=16)
            plt.xlabel('Correlation Coefficient with Age', fontsize=12)
            plt.ylabel('Channel Pair', fontsize=12)
            plt.axvline(x=0, color='grey', linestyle='--')
            
            # 使用bbox_inches='tight'可以更好地自动调整边距
            # plt.tight_layout() # 这句可以省略，因为savefig中的bbox_inches='tight'效果更好

            # 保存图片
            save_dir = os.path.join(save_path, 'Symmetry')
            os.makedirs(save_dir, exist_ok=True)
            # 在保存时使用 bbox_inches='tight' 来裁剪空白边缘
            plt.savefig(os.path.join(save_dir, f'age_corr_{state}_{feature}.jpg'), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    # 设置文件路径
    os.makedirs('bo_save_jpg/AgeCorrelation', exist_ok=True)
    data_file_path = "bo_save_csv/data/process_symmetry_median.tsv"
    save_image_path = "./bo_save_jpg/AgeCorrelation"
    
    plot_symmetry_age_correlation(data_file_path, save_image_path)
    print("Symmetry age correlation analysis complete.")