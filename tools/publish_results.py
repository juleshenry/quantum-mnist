import pandas as pd
import os
import re

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
    
    new_content = pattern.sub(f"{marker_start}
{table_md}
{marker_end}", content)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print(f"Updated {file_path}")

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
    
    table_md = df_sub.to_markdown(index=False)
    
    update_markdown_table('README.md', '<!-- P4_RESULTS_START -->', '<!-- P4_RESULTS_END -->', table_md)
    update_markdown_table('phase4/README.md', '<!-- P4_RESULTS_START -->', '<!-- P4_RESULTS_END -->', table_md)

def publish_phase5():
    csv_path = 'phase5/results/scientific_k_summary.csv'
    if not os.path.exists(csv_path):
        print("No Phase 5 results found.")
        return
    
    df = pd.read_csv(csv_path)
    cols = ['k', 'q_acc_mean', 'c_acc_mean']
    df_sub = df[cols].copy()
    df_sub.columns = ['K (Categories)', 'QNN (4x4 PCA)', 'Fair Classical (4x4)']
    
    # Format decimals
    df_sub['QNN (4x4 PCA)'] = df_sub['QNN (4x4 PCA)'].map(lambda x: f"{x*100:.1f}%")
    df_sub['Fair Classical (4x4)'] = df_sub['Fair Classical (4x4)'].map(lambda x: f"{x*100:.1f}%")
    
    table_md = df_sub.to_markdown(index=False)
    
    update_markdown_table('README.md', '<!-- P5_RESULTS_START -->', '<!-- P5_RESULTS_END -->', table_md)
    update_markdown_table('phase5/README.md', '<!-- P5_RESULTS_START -->', '<!-- P5_RESULTS_END -->', table_md)

if __name__ == "__main__":
    os.makedirs('tools', exist_ok=True)
    publish_phase4()
    publish_phase5()
