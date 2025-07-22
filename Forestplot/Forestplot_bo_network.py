import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scienceplots
from Statis import StatisticalComparison
import os

class ForestPlotter:
    def __init__(self):
        self.features = ['small_worldness_alpha', 'global_efficiency_alpha', 
                         'modularity_alpha', 'characteristic_path_length_alpha'
                        ]
        # 为新网络特征创建一个新的分组
        self.feature_groups = {
            'network_alpha': ['small_worldness_alpha', 'global_efficiency_alpha', 
                              'modularity_alpha', 'characteristic_path_length_alpha']
        }
        
        # 新组定义颜色
        self.group_colors = {
            'network_alpha': '#B3D1E6',  # 使用浅蓝色
        }
        
        # 以及对应的深色
        self.group_colors_dark = {
            'network_alpha': '#336699',  # 使用深蓝色
        }
    def remove_outliers(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 30 * IQR
        upper_bound = Q3 + 30 * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def calculate_odds_ratio(self, feature_data_pod, feature_data_nopod):
        # 计算中位数作为分界点
        median = np.median(pd.concat([feature_data_pod, feature_data_nopod]))
        
        # 计算高于和低于中位数的数量
        a = sum(feature_data_pod > median)    # POD组高于中位数
        b = sum(feature_data_pod <= median)   # POD组低于中位数
        c = sum(feature_data_nopod > median)  # noPOD组高于中位数
        d = sum(feature_data_nopod <= median) # noPOD组低于中位数
        
        # 添加平滑因子0.5防止除零
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        
        # 计算95%置信区间
        log_or = np.log(odds_ratio)
        se = np.sqrt(1/(a+0.5) + 1/(b+0.5) + 1/(c+0.5) + 1/(d+0.5))
        ci_lower = np.exp(log_or - 1.96*se)
        ci_upper = np.exp(log_or + 1.96*se)
        
        return odds_ratio, ci_lower, ci_upper

    def plot_forest(self):
    # 读取数据
        data = pd.read_csv('./bo_save_csv/data/process_network_median.tsv', sep='\t')

    # 创建一个空的DataFrame来存储结果
        results_df = pd.DataFrame()

    # 创建图形
        plt.style.use(['science', 'nature','no-latex'])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle('Forest Plot - Brain Network Features', fontsize=20)

        for state, ax in zip(['open-eyes', 'close-eyes'], [ax1, ax2]):
            state_data = data[data['state'] == state]

            odds_ratios = []
            ci_lowers = []
            ci_uppers = []
            p_values = []
            feature_colors = []

            for feature in self.features:
            # 使用 self.feature_groups 和 self.group_colors_dark
                group = next(g for g, features in self.feature_groups.items() 
                       if feature in features)
                feature_colors.append(self.group_colors_dark[group])

                pod_data = state_data[state_data['delirium'] == 1][feature]
                nopod_data = state_data[state_data['delirium'] == 0][feature]

                pod_data = self.remove_outliers(pod_data)
                nopod_data = self.remove_outliers(nopod_data)

                or_value, ci_lower, ci_upper = self.calculate_odds_ratio(pod_data, nopod_data)

                comparison = StatisticalComparison(pod_data.values, nopod_data.values)
                p_value = comparison.perform_statistical_test()

                odds_ratios.append(or_value)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                p_values.append(p_value)

                new_row = {
                    'feature': feature,
                    'state': state,
                    'OR': or_value,
                    'p_value': p_value
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

            current_group = None
            group_start_idx = 0

            for idx, (or_value, ci_lower, ci_upper, p_value) in enumerate(
                zip(odds_ratios, ci_lowers, ci_uppers, p_values)):

                feature_name = self.features[idx]
                # 使用 self.feature_groups
                group = next(g for g, features in self.feature_groups.items() 
                           if feature_name in features)

                if group != current_group:
                    if current_group:
                    # 使用 self.group_colors
                        ax.axhspan(group_start_idx-0.5, idx-0.5, 
                              color=self.group_colors[current_group], alpha=0.3)
                    current_group = group
                    group_start_idx = idx

                line_color = 'red' if p_value < 0.05 else '#4169E1'

                ax.plot([ci_lower, ci_upper], [idx, idx], color=line_color, linewidth=2)
                ax.scatter(or_value, idx, color=line_color, s=200)

        # 使用 self.group_colors
            ax.axhspan(group_start_idx-0.5, len(self.features)-0.5, 
                    color=self.group_colors[current_group], alpha=0.3)

            ax.set_yticks(range(len(self.features)))
            ax.set_yticklabels(self.features, fontsize=20)

            for label, color in zip(ax.get_yticklabels(), feature_colors):
                label.set_color(color)

            ax.set_xscale('log')
            ax.set_xticks([0.1, 0.5, 1, 2, 10])
            ax.set_xticklabels(['0.1', '0.5', '1', '2', '10'], fontsize=10)
            ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Odds Ratio (95% CI)', fontsize=15)

            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(range(len(self.features)))
            p_value_labels = [f'p={p:.3f}' for p in p_values]
            ax2.set_yticklabels(p_value_labels, fontsize=16)

            for i, p in enumerate(p_values):
                color = 'red' if p < 0.05 else '#4169E1'
                ax2.get_yticklabels()[i].set_color(color)

            ax.set_title(f'{state}', fontsize=20)

        plt.tight_layout()

        os.makedirs('./bo_save_jpg/ForestPlot/', exist_ok=True)
        os.makedirs('./bo_save_csv/ForestPlot/', exist_ok=True)

        plt.savefig(f'./bo_save_jpg/ForestPlot/Network_Features_Forestplot.jpg', dpi=300, bbox_inches='tight')
        results_df.to_csv('./bo_save_csv/ForestPlot/network_features_statistics_ForestPlot_results.csv', index=False)
        plt.close()

if __name__ == "__main__":
    forest_plotter = ForestPlotter()
    forest_plotter.plot_forest()