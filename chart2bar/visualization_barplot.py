import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from pathlib import Path

# Set Times New Roman as default font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


def load_performance_data(csv_path: Path) -> pd.DataFrame:
    """Load performance data from the last row of the CSV file.
    
    For Ophthalmology data, if there are two accuracy rows at the end,
    this function will calculate the mean of both rows.
    """
    df = pd.read_csv(csv_path)
    
    # Check if this is Ophthalmology data with two accuracy rows
    # Look at the last two rows to see if both contain accuracy percentages
    last_row = df.iloc[-1]
    second_last_row = df.iloc[-2]
    
    # Determine the number of metadata columns to skip
    first_cols = df.columns[:10]
    metadata_cols = 0
    for col in first_cols:
        val = last_row[col]
        if pd.isna(val) or (isinstance(val, str) and '%' not in val):
            metadata_cols += 1
        else:
            break
    
    # Get model columns (after metadata columns)
    model_columns = df.columns[metadata_cols:]
    
    # Check if second-to-last row also has percentages (Ophthalmology case)
    has_two_accuracy_rows = False
    for col in model_columns[:3]:  # Check first 3 model columns
        val1 = last_row[col]
        val2 = second_last_row[col]
        if (pd.notna(val1) and isinstance(val1, str) and '%' in val1 and
            pd.notna(val2) and isinstance(val2, str) and '%' in val2):
            has_two_accuracy_rows = True
            break
    
    performance_data = []
    
    for col in model_columns:
        # Skip POv1 models (Prompt Optimized variants)
        if 'pov' in col.lower():
            continue
        
        if has_two_accuracy_rows:
            # Calculate mean of two rows for Ophthalmology
            val1 = last_row[col]
            val2 = second_last_row[col]
            
            if (pd.notna(val1) and isinstance(val1, str) and '%' in val1 and
                pd.notna(val2) and isinstance(val2, str) and '%' in val2):
                perf_float_1 = float(val1.replace('%', ''))
                perf_float_2 = float(val2.replace('%', ''))
                perf_mean = (perf_float_1 + perf_float_2) / 2
                performance_data.append({'model': col, 'performance': perf_mean})
        else:
            # Use single row (Eurorad case)
            perf_value = last_row[col]
            if pd.notna(perf_value) and isinstance(perf_value, str) and '%' in perf_value:
                perf_float = float(perf_value.replace('%', ''))
                performance_data.append({'model': col, 'performance': perf_float})
    
    return pd.DataFrame(performance_data)

def categorize_model(model_name: str) -> tuple:
    """Group models by base type and extract version info."""
    model_lower = model_name.lower()
    if 'gpt5' in model_lower or 'gpt-5' in model_lower:
        return 'GPT-5', model_name.split('-')[-1] if '-' in model_name else 'v1'
    elif 'o4-mini' in model_lower:
        return 'GPT-o4-mini', model_name.split('-')[-1] if '-' in model_name else 'v1'
    elif 'deepseek' in model_lower or 'ds-r1' in model_lower:
        return 'DeepSeek-R1', model_name.split()[-1] if 'v' in model_name else 'v1'
    elif 'qwen3' in model_lower or 'qwen-3' in model_lower:
        return 'Qwen3-235b', model_name.split()[-1] if 'v' in model_name else 'v1'
    elif 'oss-20b' in model_lower or 'oss20b' in model_lower:
        config = 'L' if '(L)' in model_name else 'M' if '(M)' in model_name else 'H'
        version = model_name.split()[-1] if 'v' in model_name else 'v1'
        return f'gpt-oss-20b ({config})', version
    elif 'oss-120b' in model_lower or 'oss120b' in model_lower:
        config = 'L' if '(L)' in model_name else 'M' if '(M)' in model_name else 'H'
        version = model_name.split()[-1] if 'v' in model_name else 'v1'
        return f'gpt-oss-120b ({config})', version
    else:
        return 'Other', 'v1'

def create_barplot(perf_df: pd.DataFrame, dataset_name: str, output_dir: Path):
    """Create and save a barplot for the given performance data."""
    
    # Add model grouping
    perf_df[['model_group', 'version']] = perf_df['model'].apply(lambda x: pd.Series(categorize_model(x)))
    
    # Calculate mean and std for each model group
    group_stats = perf_df.groupby('model_group')['performance'].agg(['mean', 'std']).fillna(0)
    
    # Sort groups for consistent ordering - group OSS models together
    group_order = ['GPT-5', 'GPT-o4-mini', 'DeepSeek-R1', 'Qwen3-235b', 'gpt-oss-20b (L)', 'gpt-oss-20b (M)', 'gpt-oss-20b (H)', 
                   'gpt-oss-120b (L)', 'gpt-oss-120b (M)', 'gpt-oss-120b (H)']
    group_stats = group_stats.reindex([g for g in group_order if g in group_stats.index])
    
    # Nature-style consistent color scheme (matching radar plots)
    NATURE_COLORS = {
        'GPT-5': '#F28E8C',          # Coral
        'GPT-o4-mini': '#56B3C4',     # Teal
        'DeepSeek-R1': '#E6C24F',     # Gold
        'Qwen3-235b': '#B88FD6',      # Lilac
        'gpt-oss-20b (L)': '#F9B8B2', # Light coral
        'gpt-oss-20b (M)': '#F49389', # Medium coral
        'gpt-oss-20b (H)': '#ED7C72', # Dark coral
        'gpt-oss-120b (L)': '#D6B8E9',# Light purple
        'gpt-oss-120b (M)': '#C59BDD',# Medium purple
        'gpt-oss-120b (H)': '#B280D1',# Dark purple
    }
    
    # Apply colors to groups
    color_map = {}
    for group in group_stats.index:
        color_map[group] = NATURE_COLORS.get(group, '#999999')  # Gray fallback

    # Create positions with tight spacing within OSS groups
    group_names = list(group_stats.index)
    x_positions = []
    current_pos = 0
    
    for i, group_name in enumerate(group_names):
        if i > 0:
            prev_group = group_names[i-1]
            
            # Tight spacing within OSS model variants (0.6 instead of 1.0)
            if ('oss-20b' in prev_group and 'oss-20b' in group_name) or \
               ('oss-120b' in prev_group and 'oss-120b' in group_name):
                current_pos += 0.6
            # Normal spacing for all other transitions
            else:
                current_pos += 1.0
        
        x_positions.append(current_pos)
    
    x_positions = np.array(x_positions)
    
    # Define different bar widths for visual grouping
    bar_widths = []
    for group in group_names:
        if 'oss-20b' in group or 'oss-120b' in group:
            bar_widths.append(0.6)  # Narrower bars for OSS models
        else:
            bar_widths.append(0.8)  # Standard width for other models
    
    # Create bar plot (width: 42cm, height: 12cm)
    # Convert cm to inches: 42cm / 2.54 = 16.535433in, 12cm / 2.54 = 4.724409in
    fig, ax = plt.subplots(figsize=(42/2.54, 12/2.54))
    ax.set_facecolor('white')
    bars = []
    for i, (pos, width) in enumerate(zip(x_positions, bar_widths)):
        bar = ax.bar(pos, group_stats['mean'].iloc[i], yerr=group_stats['std'].iloc[i], 
                     capsize=5, color=color_map.get(group_names[i], 'gray'),
                     edgecolor='white', linewidth=1.2, alpha=0.85, width=width)
        bars.extend(bar)
    
    # Customize x-axis labels - make them cleaner
    clean_labels = []
    for group in group_stats.index:
        if 'oss-20b' in group:
            clean_labels.append(group.replace('gpt-oss-20b (', '').replace(')', ''))
        elif 'oss-120b' in group:
            clean_labels.append(group.replace('gpt-oss-120b (', '').replace(')', ''))
        else:
            clean_labels.append(group)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(clean_labels, rotation=0, ha='center')
    ax.set_xlabel("LLMs", fontsize=12, labelpad=15)
    ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    
    # Adjust y-axis range based on dataset
    if dataset_name == "ophthalmology":
        ax.set_ylim(50, 90)
    else:  # eurorad
        ax.set_ylim(60, 100)
    
    # Add value labels on bars (above error bars)
    for i, (pos, mean_val, std_val) in enumerate(zip(x_positions, group_stats['mean'], group_stats['std'])):
        ax.text(pos, mean_val + std_val + 0.3, f'{mean_val:.1f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Add subtle group indicators using brackets for OSS models
    oss20b_positions = [pos for pos, group in zip(x_positions, group_names) if 'oss-20b' in group]
    oss120b_positions = [pos for pos, group in zip(x_positions, group_names) if 'oss-120b' in group]
    
    # Adjust bracket position based on dataset
    if dataset_name == "ophthalmology":
        y_bracket = 47
    else:  # eurorad
        y_bracket = 57
    
    if oss20b_positions:
        # Draw a bracket pointing upward for OSS-20B bars
        ax.plot([min(oss20b_positions)-0.3, max(oss20b_positions)+0.3], [y_bracket, y_bracket], 
                'k-', linewidth=1, alpha=0.7, clip_on=False)
        ax.plot([min(oss20b_positions)-0.3, min(oss20b_positions)-0.3], [y_bracket, y_bracket+0.3], 
                'k-', linewidth=1, alpha=0.7, clip_on=False)
        ax.plot([max(oss20b_positions)+0.3, max(oss20b_positions)+0.3], [y_bracket, y_bracket+0.3], 
                'k-', linewidth=1, alpha=0.7, clip_on=False)
        ax.text(np.mean(oss20b_positions), y_bracket-0.8, 'gpt-oss-20b', ha='center', va='center',
                fontsize=10, color='black', clip_on=False)
    
    if oss120b_positions:
        # Draw a bracket pointing upward for OSS-120B bars
        ax.plot([min(oss120b_positions)-0.3, max(oss120b_positions)+0.3], [y_bracket, y_bracket], 
                'k-', linewidth=1, alpha=0.7, clip_on=False)
        ax.plot([min(oss120b_positions)-0.3, min(oss120b_positions)-0.3], [y_bracket, y_bracket+0.3], 
                'k-', linewidth=1, alpha=0.7, clip_on=False)
        ax.plot([max(oss120b_positions)+0.3, max(oss120b_positions)+0.3], [y_bracket, y_bracket+0.3], 
                'k-', linewidth=1, alpha=0.7, clip_on=False)
        ax.text(np.mean(oss120b_positions), y_bracket-0.8, 'gpt-oss-120b', ha='center', va='center',
                fontsize=10, color='black', clip_on=False)
    
    fig.tight_layout()
    # Adjust subplot to make room for brackets below (after tight_layout)
    fig.subplots_adjust(bottom=0.18)
    
    # Save figure
    output_path = output_dir / f'{dataset_name}_llm_results.png'
    fig.savefig(output_path, dpi=900, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Generated {dataset_name} plot: {output_path}")


# Main execution: Process both datasets
if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of datasets to process
    datasets = [
        ("OSS Benchmarking Results - Eurorad.csv", "eurorad"),
        ("OSS Benchmarking Results - Ophthalmology.csv", "ophthalmology")
    ]
    
    print("Generating bar plots for all datasets...")
    print("=" * 60)
    
    for csv_filename, dataset_name in datasets:
        csv_path = data_dir / csv_filename
        
        if not csv_path.exists():
            print(f"⚠ Skipping {dataset_name}: {csv_path} not found")
            continue
        
        print(f"\nProcessing {dataset_name}...")
        
        # Check if dataset has two accuracy rows
        df_check = pd.read_csv(csv_path)
        last_row = df_check.iloc[-1]
        second_last_row = df_check.iloc[-2]
        
        # Quick check for two accuracy rows
        sample_col = df_check.columns[5]  # Check a model column
        if (pd.notna(last_row[sample_col]) and isinstance(last_row[sample_col], str) and '%' in last_row[sample_col] and
            pd.notna(second_last_row[sample_col]) and isinstance(second_last_row[sample_col], str) and '%' in second_last_row[sample_col]):
            print(f"  → Detected two accuracy rows, using mean of both")
        
        perf_df = load_performance_data(csv_path)
        create_barplot(perf_df, dataset_name, output_dir)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
