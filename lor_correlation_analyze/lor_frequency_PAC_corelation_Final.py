import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
from scipy import stats
from Statis import StatisticalComparison
from Correlation import CorrelationAnalyzer

class LORCePCorrelationAnalyzer:
    def __init__(self):
        self.frequency_features = [
            'delta_proportion', 'theta_proportion', 'alpha_proportion', 'beta_proportion', 'gamma_proportion',
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
            'state_entropy', 'spectral_entropy', 'sef_95', 'total_power'
        ]
        
        self.pac_features = [
            'delta_alpha_mi','delta_alpha_mvl',
            'delta_beta_mi','delta_beta_mvl','delta_gamma_mi',
            'delta_gamma_mvl','theta_alpha_mi','theta_alpha_mvl',
            'theta_beta_mi','theta_beta_mvl','theta_gamma_mi',
            'theta_gamma_mvl','alpha_beta_mi','alpha_beta_mvl',
            'alpha_gamma_mi','alpha_gamma_mvl'
        ]
        
    def load_and_process_data(self, frequency_filepath, pac_filepath):
        """加载并预处理频率和PAC数据"""
        freq_df = pd.read_csv(frequency_filepath, sep='\t')
        pac_df = pd.read_csv(pac_filepath, sep='\t')
        
        # 获取所有通道的交集
        self.channels = sorted(set(freq_df['ch_name'].unique()) & set(pac_df['ch_name'].unique()))
        
        return freq_df, pac_df
        
    def calculate_lorcep_correlation(self, df, features, state, feature_type):
        """计算LORCeP与特征的相关性"""
        # 筛选特定状态的数据
        state_data = df[df['state'] == state]
        
        results = []
        for channel in self.channels:
            for feature in features:
                # 获取当前通道的数据
                channel_data = state_data[state_data['ch_name'] == channel].copy()
                
                # 使用IQR方法检测异常值
                Q1 = channel_data[feature].quantile(0.25)
                Q3 = channel_data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 30 * IQR
                upper_bound = Q3 + 30 * IQR
                is_normal = (channel_data[feature] >= lower_bound) & (channel_data[feature] <= upper_bound)
                
                # 移除异常值
                channel_data = channel_data[is_normal]
                
                # 获取特征数据和LORCeP数据
                feature_data = channel_data[feature].values
                lorcep_data = channel_data['LORCeP'].values
                
                # 处理LORCeP数据中的缺失值
                valid_indices = ~pd.isna(lorcep_data) & ~pd.isna(feature_data)
                filtered_lorcep_data = lorcep_data[valid_indices]
                filtered_feature_data = feature_data[valid_indices]
                
                if len(filtered_lorcep_data) > 2:  # 至少需要3个数据点来计算相关性
                    # 计算与LORCeP的相关性
                    analyzer = CorrelationAnalyzer(filtered_feature_data, filtered_lorcep_data)
                    correlation_result = analyzer.calculate_correlation()
                    correlation = correlation_result['correlation']
                    p_value_corr = correlation_result['p_value']
                    
                    # 计算效应大小分类
                    effect_size = 'small' if abs(correlation) < 0.3 else ('medium' if abs(correlation) < 0.5 else 'large')
                    
                    # 计算置信区间（可选）
                    z_score = 0.5 * np.log((1 + correlation) / (1 - correlation))
                    se = 1 / np.sqrt(len(filtered_lorcep_data) - 3)
                    ci_lower = np.tanh(z_score - 1.96 * se)
                    ci_upper = np.tanh(z_score + 1.96 * se)
                    
                    results.append({
                        'feature_type': feature_type,
                        'state': state,
                        'channel': channel,
                        'feature': feature,
                        'correlation': correlation,
                        'correlation_p_value': p_value_corr,
                        'n_samples': len(filtered_lorcep_data),
                        'effect_size': effect_size,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'is_significant': p_value_corr < 0.05
                    })
        
        return pd.DataFrame(results)
    
    def plot_topomap(self, data, features, title, filename, feature_type):
        """绘制地形图"""
        plt.style.use(['science', 'nature'])
        
        try:
            # 创建montage和info
            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(ch_names=self.channels, sfreq=300, ch_types='eeg')
            info.set_montage(montage)

            # 根据特征类型设置图表布局
            if feature_type == 'frequency':
                fig, axes = plt.subplots(3, 5, figsize=(15, 9))
                n_features = 14
            else:  # PAC
                fig, axes = plt.subplots(4, 4, figsize=(20, 20))
                n_features = 16
                
            plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05,
                              top=0.92, wspace=0.2, hspace=0.2)
            fig.suptitle(title, fontsize=16)
            
            axes_flat = axes.flatten()
            
            for idx, feature in enumerate(features):
                if idx < n_features:
                    ax = axes_flat[idx]
                    feature_data = data[data['feature'] == feature]
                    feature_values = []
                    p_values = []
                    
                    # 获取通道位置信息
                    pos = info.get_montage().get_positions()['ch_pos']
                    pos_list = [pos[ch_name] for ch_name in self.channels]
                    
                    for ch in self.channels:
                        ch_value = feature_data[feature_data['channel'] == ch]['correlation'].values
                        ch_p_value = feature_data[feature_data['channel'] == ch]['correlation_p_value'].values
                        
                        feature_values.append(ch_value[0] if len(ch_value) > 0 else 0)
                        p_values.append(ch_p_value[0] if len(ch_p_value) > 0 else 1.0)
                    
                    feature_values = np.array(feature_values)
                    p_values = np.array(p_values)
                    
                    # 方案2：通过修改电极标签样式来显示显著性
                    # 创建自定义的电极标签
                    electrode_labels = []
                    electrode_colors = []
                    for ch_name, p_val in zip(self.channels, p_values):
                        if p_val < 0.05:
                            electrode_labels.append(f"{ch_name}*")  # 添加星号
                            electrode_colors.append('red')
                        else:
                            electrode_labels.append(ch_name)
                            electrode_colors.append('black')
                    
                    # 绘制地形图（不显示默认的电极标签）
                    im = mne.viz.plot_topomap(
                        feature_values, info, axes=ax, show=False,
                        names=None,  # 不显示默认标签
                        cmap='RdBu_r',
                        outlines='head',
                        sensors=False,  # 不显示默认的传感器点
                        contours=6,
                        size=2,
                        sphere=0.11,
                        vlim=(-1, 1)
                    )
                    
                    # 手动添加电极标签和传感器点
                    for idx_ch, (ch_name, pos_ch, color, label) in enumerate(zip(self.channels, pos_list, electrode_colors, electrode_labels)):
                        # 添加传感器点
                        point_color = 'red' if color == 'red' else 'black'
                        point_size = 8 if color == 'red' else 6
                        ax.plot(pos_ch[0], pos_ch[1], 'o', color=point_color, 
                               markersize=point_size, markeredgecolor='white', 
                               markeredgewidth=1, transform=ax.transData, zorder=4)
                        
                        # 添加电极标签
                        ax.text(pos_ch[0], pos_ch[1] - 0.04, label, 
                               color=color, fontsize=9, fontweight='bold' if color == 'red' else 'normal',
                               ha='center', va='top', transform=ax.transData, zorder=5)
                    
                    ax.set_title(feature, pad=2, fontdict={'size': 15})
            
            # 移除多余的子图
            if feature_type == 'frequency':
                for idx in range(14, 15):
                    fig.delaxes(axes_flat[idx])
            
            # 添加颜色条
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(im[0], cax=cax)
            cbar.set_label('Correlation with LORCeP', fontsize=12)
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘图错误: {str(e)}")

    def create_summary_plots(self, all_results):
        """创建汇总图表"""
        plt.style.use(['science', 'nature'])
        
        # 1. 按特征类型和状态分组的相关性分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LORCeP Correlation Distribution by Feature Type and State', fontsize=16)
        
        states = ['open-eyes', 'close-eyes']
        feature_types = ['frequency', 'pac']
        
        for i, state in enumerate(states):
            for j, feature_type in enumerate(feature_types):
                ax = axes[i, j]
                
                # 筛选数据
                subset = all_results[
                    (all_results['state'] == state) & 
                    (all_results['feature_type'] == feature_type)
                ]
                
                if not subset.empty:
                    # 绘制直方图
                    correlations = subset['correlation'].values
                    correlations = correlations[~np.isnan(correlations)]
                    
                    ax.hist(correlations, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{feature_type.capitalize()} - {state}', fontsize=14)
                    ax.set_xlabel('Correlation with LORCeP')
                    ax.set_ylabel('Frequency')
                    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                    
                    # 添加统计信息
                    mean_corr = np.mean(correlations)
                    std_corr = np.std(correlations)
                    ax.text(0.05, 0.95, f'Mean: {mean_corr:.3f}\nStd: {std_corr:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('./bo_save_jpg/LORCorrelation/correlation_distribution_summary_final.jpg', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 显著性相关性的数量统计
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算显著性相关性的数量
        significant_counts = []
        labels = []
        
        for state in states:
            for feature_type in feature_types:
                subset = all_results[
                    (all_results['state'] == state) & 
                    (all_results['feature_type'] == feature_type) &
                    (all_results['correlation_p_value'] < 0.05)
                ]
                significant_counts.append(len(subset))
                labels.append(f'{feature_type}\n{state}')
        
        bars = ax.bar(labels, significant_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        ax.set_title('Number of Significant Correlations with LORCeP (p < 0.05)', fontsize=16)
        ax.set_ylabel('Number of Significant Correlations')
        
        # 添加数值标签
        for bar, count in zip(bars, significant_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=12)
        
        # 3. 效应大小分析
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Effect Size Distribution by Feature Type and State', fontsize=16)
        
        for i, state in enumerate(states):
            for j, feature_type in enumerate(feature_types):
                ax = axes[i, j]
                
                subset = all_results[
                    (all_results['state'] == state) & 
                    (all_results['feature_type'] == feature_type)
                ]
                
                if not subset.empty:
                    # 按效应大小分类计数
                    effect_counts = subset['effect_size'].value_counts()
                    colors = {'small': 'lightblue', 'medium': 'orange', 'large': 'red'}
                    
                    bars = ax.bar(effect_counts.index, effect_counts.values, 
                                 color=[colors.get(x, 'gray') for x in effect_counts.index])
                    
                    ax.set_title(f'{feature_type.capitalize()} - {state}', fontsize=14)
                    ax.set_ylabel('Count')
                    ax.set_xlabel('Effect Size')
                    
                    # 添加数值标签
                    for bar, count in zip(bars, effect_counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('./bo_save_jpg/LORCorrelation/effect_size_distribution_final.jpg', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def run_analysis(self):
        """运行完整分析流程"""
        # 创建保存目录
        Path("./bo_save_jpg/LORCorrelation/").mkdir(parents=True, exist_ok=True)
        Path("./bo_save_csv/LORCorrelation/").mkdir(parents=True, exist_ok=True)
        
        # 读取数据
        freq_df, pac_df = self.load_and_process_data(
            "bo_save_csv/data/process_frequency_distribution_median.tsv",
            "bo_save_csv/data/process_pac_median.tsv"
        )
        
        all_results = []
        
        # 对每个状态进行分析
        for state in ['open-eyes', 'close-eyes']:
            print(f"Processing state: {state}")
            
            # 分析频率特征与LORCeP的相关性
            freq_results = self.calculate_lorcep_correlation(
                freq_df, self.frequency_features, state, 'frequency'
            )
            all_results.append(freq_results)
            
            # 分析PAC特征与LORCeP的相关性
            pac_results = self.calculate_lorcep_correlation(
                pac_df, self.pac_features, state, 'pac'
            )
            all_results.append(pac_results)
            
            # 绘制频率特征地形图
            self.plot_topomap(
                freq_results,
                self.frequency_features,
                f'Frequency Features - LORCeP Correlation ({state})',
                f"./bo_save_jpg/LORCorrelation/frequency_lorcep_correlation_{state}_final.jpg",
                'frequency'
            )
            
            # 绘制PAC特征地形图
            self.plot_topomap(
                pac_results,
                self.pac_features,
                f'PAC Features - LORCeP Correlation ({state})',
                f"./bo_save_jpg/LORCorrelation/pac_lorcep_correlation_{state}_final.jpg",
                'pac'
            )
        
        # 合并所有结果
        all_results_df = pd.concat(all_results, ignore_index=True)
        
        # 保存详细结果
        all_results_df.to_csv("./bo_save_csv/LORCorrelation/lorcep_correlation_results.csv", index=False)
        
        # 创建汇总统计
        summary_stats = all_results_df.groupby(['feature_type', 'state']).agg({
            'correlation': ['mean', 'std', 'min', 'max'],
            'correlation_p_value': lambda x: (x < 0.05).sum(),  # 显著性相关性数量
            'n_samples': 'mean'
        }).round(4)
        
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats.to_csv("./bo_save_csv/LORCorrelation/lorcep_correlation_summary.csv")
        
        # 创建汇总图表
        self.create_summary_plots(all_results_df)
        
        print("Analysis completed successfully!")
        print(f"Total correlations calculated: {len(all_results_df)}")
        print(f"Significant correlations (p < 0.05): {(all_results_df['correlation_p_value'] < 0.05).sum()}")

if __name__ == "__main__":
    analyzer = LORCePCorrelationAnalyzer()
    analyzer.run_analysis()