import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from Correlation import CorrelationAnalyzer # 确保 Correlation.py 在同一个目录下
import os

def plot_symmetry_lor_correlation_centered_p(data_path, save_path):
    """
    计算并绘制Symmetry特征与LOR浓度的相关性条形图（排序后），并将P值居中标注。
    """
    # --- 1. 加载数据 ---
    df = pd.read_csv(data_path, sep='\t')

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
                subset_data = state_data[state_data['target_channels'] == pair]

                feature_data = subset_data[feature].values
                lor_data = subset_data['LORCeP'].values # 使用LOR浓度

                valid_indices = ~pd.isna(lor_data)
                filtered_lor_data = lor_data[valid_indices]
                filtered_feature_data = feature_data[valid_indices]

                if len(filtered_lor_data) > 1:
                    analyzer = CorrelationAnalyzer(filtered_feature_data, filtered_lor_data)
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

        for feature in features:
            feature_results = state_results[state_results['feature'] == feature].sort_values(by='correlation', ascending=False)

            if feature_results.empty:
                continue

            num_pairs = len(feature_results)
            figure_height = max(8, 0.5 * num_pairs)
            plt.figure(figsize=(12, figure_height))

            barplot = sns.barplot(x='correlation', y='channel_pair', data=feature_results, palette="vlag", hue='channel_pair', legend=False)

            # 【核心修改】将P值标注在条形图的中间
            for i, row in enumerate(feature_results.itertuples()):
                p_text = f"p={row.p_value:.3f}"
                
                # 新逻辑: 将文本放置在条形长度的一半处
                x_position = row.correlation / 2
                
                # 新逻辑: 使用白色粗体文本，以确保在深色条形上清晰可见
                plt.text(x=x_position, 
                         y=i, 
                         s=p_text, 
                         color='black', 
                         ha='center', # 水平居中对齐
                         va='center', # 垂直居中对齐
                         fontsize=9,
                         fontweight='bold') # 加粗

            plt.title(f'LOR(CeProp) Correlation for {feature} ({state}) (Sorted)', fontsize=16)
            plt.xlabel('Correlation Coefficient with LOR(CeProp)', fontsize=12)
            plt.ylabel('Channel Pair', fontsize=12)
            plt.axvline(x=0, color='grey', linestyle='--')

            save_dir = os.path.join(save_path, 'Symmetry_Sorted_P_Centered')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'lor_corr_p_centered_{state}_{feature}.jpg'), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    data_file_path = "bo_save_csv/data/process_symmetry_median.tsv"
    save_image_path = "./bo_save_jpg/LORCorrelation"

    plot_symmetry_lor_correlation_centered_p(data_file_path, save_image_path)
    print("Symmetry LOR correlation analysis with centered p-values (sorted) complete.")