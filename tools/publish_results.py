import pandas as pd
import os
import re
import json

def update_markdown_table(file_path, marker_start, marker_end, table_md):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = re.compile(f"{re.escape(marker_start)}.*?{re.escape(marker_end)}", re.DOTALL)
    if not pattern.search(content):
        print(f"Markers not found in {file_path}")
        return
    
    new_content = pattern.sub(f"""{marker_start}
{table_md}
{marker_end}""", content)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print(f"Updated {file_path}")

def df_to_markdown(df):
    """Simple replacement for df.to_markdown() that doesn't need tabulate."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(map(str, row.tolist())) + " |")
    return "\n".join([header, sep] + rows)

def publish_phase4():
    csv_path = 'phase4/results/experiment_results.csv'
    if not os.path.exists(csv_path):
        print("No Phase 4 results found.")
        return
    
    df = pd.read_csv(csv_path)
    # Mapping columns to readable names
    cols = ['pair', 'qnn_acc_mean', 'fair_acc_mean', 'p_value', 'significant']
    df_sub = df[cols].copy()
    df_sub.columns = ['Pair', 'QNN Accuracy', 'Fair Classical', 'P-Value', 'Significant?']
    
    # Format decimals
    df_sub['QNN Accuracy'] = df_sub['QNN Accuracy'].map(lambda x: f"{x*100:.1f}%")
    df_sub['Fair Classical'] = df_sub['Fair Classical'].map(lambda x: f"{x*100:.1f}%")
    df_sub['P-Value'] = df_sub['P-Value'].map(lambda x: f"{x:.4f}")
    
    table_md = df_to_markdown(df_sub)
    
    update_markdown_table('README.md', '<!-- P4_RESULTS_START -->', '<!-- P4_RESULTS_END -->', table_md)
    update_markdown_table('phase4/README.md', '<!-- P4_RESULTS_START -->', '<!-- P4_RESULTS_END -->', table_md)

    # Publish aggregate test results if available
    agg_path = 'phase4/results/aggregate_test.json'
    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            agg = json.load(f)
        agg_md = format_aggregate_results(agg)
        update_markdown_table('README.md', '<!-- P4_AGGREGATE_START -->', '<!-- P4_AGGREGATE_END -->', agg_md)
        update_markdown_table('phase4/README.md', '<!-- P4_AGGREGATE_START -->', '<!-- P4_AGGREGATE_END -->', agg_md)
    else:
        print("No aggregate test results found (phase4/results/aggregate_test.json).")


def format_aggregate_results(agg):
    """Format aggregate_test.json into a markdown summary."""
    lines = []
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Mean Delta (QNN − Fair) | {agg.get('mean_delta', 0)*100:+.2f}% |")
    lines.append(f"| Std Delta | {agg.get('std_delta', 0)*100:.2f}% |")
    lines.append(f"| Effect Size (Cohen's d) | {agg.get('effect_size_d', 0):.3f} |")
    lines.append(f"| One-sample t-test p | {agg.get('ttest_p', 1):.4f} |")
    lines.append(f"| Wilcoxon signed-rank p | {agg.get('wilcoxon_p', 1):.4f} |")
    lines.append(f"| QNN Wins | {agg.get('qnn_wins', 0)} / {agg.get('n_pairs', 0)} |")
    lines.append(f"| Fair Classical Wins | {agg.get('fair_wins', 0)} / {agg.get('n_pairs', 0)} |")
    return "\n".join(lines)

def publish_phase5():
    csv_path = 'phase5/results/comprehensive_k_summary.csv'
    if not os.path.exists(csv_path):
        print("No Phase 5 results found.")
        return
    
    df = pd.read_csv(csv_path)
    cols = ['k', 'qnn_acc_mean', 'fair_acc_mean']
    df_sub = df[cols].copy()
    df_sub.columns = ['K (Categories)', 'QNN (4x4 PCA)', 'Fair Classical (4x4)']
    
    # Format decimals
    df_sub['QNN (4x4 PCA)'] = df_sub['QNN (4x4 PCA)'].map(lambda x: f"{x*100:.1f}%")
    df_sub['Fair Classical (4x4)'] = df_sub['Fair Classical (4x4)'].map(lambda x: f"{x*100:.1f}%")
    
    table_md = df_to_markdown(df_sub)
    
    update_markdown_table('README.md', '<!-- P5_RESULTS_START -->', '<!-- P5_RESULTS_END -->', table_md)
    update_markdown_table('phase5/README.md', '<!-- P5_RESULTS_START -->', '<!-- P5_RESULTS_END -->', table_md)

if __name__ == "__main__":
    os.makedirs('tools', exist_ok=True)
    publish_phase4()
    publish_phase5()
