import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import statsmodels.api as sm
from Statis import StatisticalComparison  # 确保Statis.py在同一个目录下
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def reproduce_finding(data_path, save_path):
    """
    复现"谵妄组LOR浓度显著更低"的结论，并计算OR值。
    """
    # --- 1. 加载数据并提取每位患者的唯一信息 ---
    try:
        df = pd.read_csv(data_path, sep='\t')
    except FileNotFoundError:
        print(f"错误：找不到文件 {data_path}")
        return None

    # 按'filename'分组，获取每个患者唯一的LORCeP和delirium状态
    patient_df = df[['filename', 'LORCeP', 'delirium']].drop_duplicates()

    # 分组
    pod_group_lor = patient_df[patient_df['delirium'] == 1]['LORCeP'].dropna()
    nopod_group_lor = patient_df[patient_df['delirium'] == 0]['LORCeP'].dropna()

    if pod_group_lor.empty or nopod_group_lor.empty:
        print("错误：一个或两个组中没有有效的LORCeP数据。")
        return None

    # --- 2. 绘制箱型图并进行统计检验 ---
    plt.style.use(['science'])  # 移除nature样式避免LaTeX冲突
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置支持Unicode的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 10))

    # 修复seaborn boxplot的警告
    ax = sns.boxplot(data=patient_df, x='delirium', y='LORCeP', 
                     palette=['#2ecc71', '#e74c3c'])
    
    # 执行统计检验 (T-test or Mann-Whitney U)
    comparison = StatisticalComparison(pod_group_lor.values, nopod_group_lor.values)
    p_value_group_diff = comparison.perform_statistical_test()
    
    # --- 3. 计算优势比 (Odds Ratio) ---
    logit_data = patient_df.dropna(subset=['LORCeP', 'delirium'])
    Y = logit_data['delirium']
    X = logit_data['LORCeP']
    X = sm.add_constant(X)  # 添加截距项

    # 拟合模型
    logit_model = sm.Logit(Y, X).fit(disp=0)  # disp=0 抑制输出
    
    # 获取OR值和置信区间
    params = logit_model.params
    conf = logit_model.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    
    # 指数化得到真正的OR和其置信区间
    or_results = np.exp(conf)
    
    # 提取LORCeP对应的OR值
    lor_or_result = or_results.loc['LORCeP']
    
    # 提取p值
    p_value_or = logit_model.pvalues['LORCeP']

    # 计算组间均值差异
    mean_pod = pod_group_lor.mean()
    mean_nopod = nopod_group_lor.mean()
    
    # 获取文件名用于标题
    file_name = os.path.basename(data_path).replace('.tsv', '').replace('process_', '')
    
    plt.title(f'LOR(CeProp) Comparison: {file_name}', fontsize=16, fontweight='bold')
    plt.ylabel('LOR(CeProp)', fontsize=12)
    plt.xticks([0, 1], ['noPOD', 'POD'], fontsize=12)
    
    # 在图中标注统计结果
    y_max = patient_df['LORCeP'].max()
    y_min = patient_df['LORCeP'].min()
    y_range = y_max - y_min
    
    # 主要统计信息
    stats_text = f'P-value = {p_value_group_diff:.3f}\n'
    stats_text += f'OR = {lor_or_result["OR"]:.3f} ({lor_or_result["2.5%"]:.2f}-{lor_or_result["97.5%"]:.2f})\n'
    stats_text += f'OR P-value = {p_value_or:.3f}\n'
    stats_text += f'Mean noPOD = {mean_nopod:.3f}\n'
    stats_text += f'Mean POD = {mean_pod:.3f}'
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 判断结论符合性
    p_significant = p_value_group_diff < 0.05
    or_significant = p_value_or < 0.05
    or_protective = lor_or_result['OR'] < 1.0  # OR < 1 表示保护性因子
    pod_lower = mean_pod < mean_nopod  # POD组LOR更低
    
    # 结论文本 - 使用英文避免LaTeX Unicode问题
    if p_significant and or_significant and or_protective and pod_lower:
        conclusion = "CONCLUSION: POD group has significantly lower LOR"
        conclusion_color = 'lightgreen'
    elif p_significant and pod_lower:
        conclusion = "PARTIAL: POD group has lower LOR but OR not significant"
        conclusion_color = 'lightyellow'
    else:
        conclusion = "NO DIFFERENCE: No significant difference found"
        conclusion_color = 'lightcoral'
    
    plt.text(0.5, 0.02, conclusion, transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor=conclusion_color, alpha=0.8))
    
    # 保存图片
    safe_filename = file_name.replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(save_path, f'lor_comparison_{safe_filename}.jpg'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回结果用于汇总
    return {
        'file': file_name,
        'p_value': p_value_group_diff,
        'or': lor_or_result['OR'],
        'or_ci_lower': lor_or_result['2.5%'],
        'or_ci_upper': lor_or_result['97.5%'],
        'or_p_value': p_value_or,
        'mean_nopod': mean_nopod,
        'mean_pod': mean_pod,
        'conclusion': conclusion.split(': ')[1] if ': ' in conclusion else conclusion
    }

def analyze_all_files():
    """
    分析所有数据文件并生成汇总报告
    """
    # 设置路径
    data_dir = "bo_save_csv/data/"
    save_path = "bo_save_jpg/Conclusion_Verify"
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 获取所有tsv文件
    file_pattern = os.path.join(data_dir, "process_*.tsv")
    data_files = glob.glob(file_pattern)
    
    if not data_files:
        print(f"No matching data files found in {data_dir}")
        return
    
    print(f"Found {len(data_files)} data files, starting analysis...")
    
    # 存储所有结果
    results = []
    
    # 逐个分析文件
    for data_file in data_files:
        print(f"\nAnalyzing: {os.path.basename(data_file)}")
        result = reproduce_finding(data_file, save_path)
        if result:
            results.append(result)
    
    # 生成汇总报告
    if results:
        summary_df = pd.DataFrame(results)
        
        # 保存汇总表格
        summary_path = os.path.join(save_path, 'analysis_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # 打印汇总结果
        print("\n" + "="*80)
        print("Summary Analysis Results:")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            print(f"\nFile: {row['file']}")
            print(f"  Group difference P-value: {row['p_value']:.3f}")
            print(f"  OR value: {row['or']:.3f} ({row['or_ci_lower']:.2f}-{row['or_ci_upper']:.2f})")
            print(f"  OR P-value: {row['or_p_value']:.3f}")
            print(f"  noPOD mean: {row['mean_nopod']:.3f}")
            print(f"  POD mean: {row['mean_pod']:.3f}")
            print(f"  Conclusion: {row['conclusion']}")
        
        # 统计符合结论的文件数量
        conclusion_match = summary_df['conclusion'].str.contains('POD group has significantly lower LOR', na=False).sum()
        partial_match = summary_df['conclusion'].str.contains('POD group has lower LOR but OR not significant', na=False).sum()
        no_match = len(summary_df) - conclusion_match - partial_match
        
        print(f"\nOverall Statistics:")
        print(f"  Full conclusion match: {conclusion_match}/{len(summary_df)} files")
        print(f"  Partial conclusion match: {partial_match}/{len(summary_df)} files")
        print(f"  No conclusion match: {no_match}/{len(summary_df)} files")
        
        print(f"\nAll charts saved to: {save_path}")
        print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    analyze_all_files()