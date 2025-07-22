import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from Correlation import CorrelationAnalyzer # 确保 Correlation.py 在同一个目录下
import os

def plot_tf_age_correlation(data_path, save_path):
    """
    计算并绘制Time-Frequency每个频率点与年龄的相关性
    """
    # --- 1. 加载数据 ---
    df = pd.read_csv(data_path, sep='\t')

    # 获取所有通道和状态
    channels = sorted(df['ch_name'].unique())
    states = ['open-eyes', 'close-eyes']
    
    # 提取频率列
    freq_cols = [col for col in df.columns if 'Hz' in col]
    freqs = np.array([float(col.split()[0]) for col in freq_cols])

    # --- 2. 绘制每个通道的相关性曲线 ---
    plt.style.use(['science', 'nature'])
    
    for state in states:
        state_data = df[df['state'] == state]
        
        # 为每个通道画一张图
        for ch in channels:
            ch_data = state_data[state_data['ch_name'] == ch]
            age_data = ch_data['age'].values
            
            correlations = []
            p_values = []
            
            # 计算每个频率点的相关性
            for freq_col in freq_cols:
                feature_data = ch_data[freq_col].values
                
                valid_indices = ~pd.isna(age_data)
                filtered_age_data = age_data[valid_indices]
                filtered_feature_data = feature_data[valid_indices]
                
                if len(filtered_age_data) > 1:
                    analyzer = CorrelationAnalyzer(filtered_feature_data, filtered_age_data)
                    corr_result = analyzer.calculate_correlation()
                    correlations.append(corr_result['correlation'])
                    p_values.append(corr_result['p_value'])
                else:
                    correlations.append(np.nan)
                    p_values.append(np.nan)
            
            # --- 3. 绘图 ---
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, correlations, marker='o', linestyle='-', markersize=4)
            
            # 标记显著的点
            for i, p in enumerate(p_values):
                if p < 0.05:
                    plt.plot(freqs[i], correlations[i], 'r*', markersize=10)
            
            plt.axhline(y=0, color='grey', linestyle='--')
            plt.title(f'Age Correlation on Channel {ch} ({state})', fontsize=16)
            plt.xlabel('Frequency (Hz)', fontsize=12)
            plt.ylabel('Correlation Coefficient with Age', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # 创建图例
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='w', marker='*', markerfacecolor='r', markersize=10, label='p < 0.05')]
            plt.legend(handles=legend_elements)
            
            # 保存图片
            save_dir = os.path.join(save_path, 'Time-Frequency', state)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'age_corr_{ch}.jpg'), dpi=300)
            plt.close()


if __name__ == "__main__":
    # 设置文件路径
    os.makedirs('bo_save_jpg/AgeCorrelation', exist_ok=True)
    data_file_path = "bo_save_csv/data/process_time-frequency_median.tsv"
    save_image_path = "./bo_save_jpg/AgeCorrelation"
    
    plot_tf_age_correlation(data_file_path, save_image_path)
    print("Time-frequency age correlation analysis complete.")