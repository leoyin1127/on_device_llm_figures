import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set Times New Roman as default font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


def load_performance_data(csv_path: Path) -> pd.DataFrame:
    """Load performance data from the last row of the CSV file."""
    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    
    # Determine the number of metadata columns to skip
    first_cols = df.columns[:10]
    metadata_cols = 0
    for col in first_cols:
        val = last_row[col]
        if pd.isna(val) or (isinstance(val, str) and '%' not in str(val)):
            metadata_cols += 1
        else:
            break
    
    # Get model columns (after metadata columns)
    model_columns = df.columns[metadata_cols:]
    
    performance_data = []
    
    for col in model_columns:
        # Skip POv1 models
        if 'pov' in col.lower():
            continue
        
        perf_value = last_row[col]
        if pd.notna(perf_value):
            # Handle both percentage strings and decimal values
            if isinstance(perf_value, str) and '%' in perf_value:
                perf_float = float(perf_value.replace('%', ''))
            else:
                # If it's a decimal value (like 0.8599), convert to percentage
                perf_float = float(perf_value) * 100 if float(perf_value) < 1 else float(perf_value)
            performance_data.append({'model': col, 'performance': perf_float})
    
    return pd.DataFrame(performance_data)


def create_comprehensive_comparison_plot(output_dir: Path):
    """Create bar plot comparing GPT-5, o4-mini, DeepSeek, finetuned, and beams models."""
    
    # Load data from original Eurorad CSV for GPT-5, o4-mini, DeepSeek
    data_dir = Path(__file__).parent / "data"
    eurorad_csv = data_dir / "OSS Benchmarking Results - Eurorad.csv"
    finetune_csv = data_dir / "OSS Benchmarking Results - finetune-Eurorad.csv"
    
    eurorad_df = load_performance_data(eurorad_csv)
    
    # Read finetune CSV directly to get specific columns
    finetune_raw = pd.read_csv(finetune_csv)
    last_row = finetune_raw.iloc[-1]
    
    # Extract GPT-5 values (gpt-5-0807-M1, M2, M3)
    gpt5_models = eurorad_df[eurorad_df['model'].str.contains('gpt-5', case=False, na=False)]
    gpt5_values = gpt5_models['performance'].values
    
    # Extract o4-mini values
    o4mini_models = eurorad_df[eurorad_df['model'].str.contains('o4-mini', case=False, na=False)]
    o4mini_values = o4mini_models['performance'].values
    
    # Extract DeepSeek values
    deepseek_models = eurorad_df[eurorad_df['model'].str.contains('deepseek', case=False, na=False)]
    deepseek_values = deepseek_models['performance'].values
    
    # Extract gpt-oss-20b (L) values from Eurorad (low reasoning effort variant)
    oss20b_l_models = eurorad_df[eurorad_df['model'].str.contains(r'oss-20b \(L\)', case=False, na=False, regex=True)]
    oss20b_l_values = oss20b_l_models['performance'].values
    
    # Extract 13-beams models (these are the finetuned models)
    finetuned_values = np.array([85.99, 87.44, 85.02])
    
    print("Data extracted:")
    print(f"GPT-5: {gpt5_values}")
    print(f"o4-mini: {o4mini_values}")
    print(f"DeepSeek: {deepseek_values}")
    print(f"gpt-oss-20b (L): {oss20b_l_values}")
    print(f"gpt-oss-20b (L) finetuned (13-beams): {finetuned_values}")
    
    # Calculate mean and std
    models = ['GPT-5', 'GPT-o4-mini', 'DeepSeek-R1', 
              'gpt-oss-20b (L)', 'gpt-oss-20b (L)\nfinetuned']
    
    means = [
        np.mean(gpt5_values) if len(gpt5_values) > 0 else 0,
        np.mean(o4mini_values) if len(o4mini_values) > 0 else 0,
        np.mean(deepseek_values) if len(deepseek_values) > 0 else 0,
        np.mean(oss20b_l_values) if len(oss20b_l_values) > 0 else 0,
        np.mean(finetuned_values) if len(finetuned_values) > 0 else 0
    ]
    
    stds = [
        np.std(gpt5_values, ddof=1) if len(gpt5_values) > 1 else 0,
        np.std(o4mini_values, ddof=1) if len(o4mini_values) > 1 else 0,
        np.std(deepseek_values, ddof=1) if len(deepseek_values) > 1 else 0,
        np.std(oss20b_l_values, ddof=1) if len(oss20b_l_values) > 1 else 0,
        np.std(finetuned_values, ddof=1) if len(finetuned_values) > 1 else 0
    ]
    
    # Nature-style consistent color scheme (matching other plots)
    colors = ['#F28E8C', '#56B3C4', '#E6C24F', '#ED7C72', '#5C8BFF']
    # GPT-5: coral, GPT-o4-mini: teal, DeepSeek-R1: gold, gpt-oss-20b (L): dark coral, gpt-oss-20b (L) finetuned: electric blue
    
    # Create bar plot (width: 20cm, height: 16cm)
    # Convert cm to inches: 20cm / 2.54 = 7.874in, 16cm / 2.54 = 6.299in
    fig, ax = plt.subplots(figsize=(20/2.54, 16/2.54))
    ax.set_facecolor('white')
    
    x_positions = np.arange(len(models))
    bars = ax.bar(x_positions, means, yerr=stds, capsize=5, 
                  color=colors, edgecolor='white', linewidth=1.2, alpha=0.85, width=0.6)
    
    # Customize axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=0, ha='center', fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(60, 92)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for i, (pos, mean_val, std_val) in enumerate(zip(x_positions, means, stds)):
        ax.text(pos, mean_val + std_val + 0.3, f'{mean_val:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    fig.tight_layout()
    
    # Save figure
    output_path = output_dir / 'comprehensive_comparison.png'
    fig.savefig(output_path, dpi=900, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✓ Generated comprehensive comparison plot: {output_path}")
    print(f"\nModel Performance Summary:")
    for model, mean, std in zip(models, means, stds):
        print(f"  {model.replace(chr(10), ' ')}: {mean:.2f}% ± {std:.2f}%")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating comprehensive comparison plot...")
    print("=" * 70)
    
    create_comprehensive_comparison_plot(output_dir)
    
    print("=" * 70)
    print("Plot generated successfully!")
