import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
from pathlib import Path

def parse_nmed_csv(filename):
    """Parse NMED CSV file and extract scores properly"""
    model_errors = {}
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Get header to understand column positions
    header = lines[0].strip().split(',')
    model_columns = header[6:]  # Skip first 6 metadata columns
    
    # Create mapping of original columns to filtered columns
    all_model_columns = header[6:]  # All original model columns
    filtered_model_indices = []
    filtered_model_names = []
    
    for idx, model in enumerate(all_model_columns):
        model_clean = model.strip()
        # Skip deepseek v2 and v3 (keep only v1)
        if 'deepseek' in model_clean.lower() and ('-v2' in model_clean.lower() or '-v3' in model_clean.lower()):
            continue
        # Skip beam variants (we have regular oss20b (L) already)
        if 'beam' in model_clean.lower():
            continue
        
        # Standardize model names to match barplot naming conventions
        # 1. GPT-5: gpt5-0807-m1 -> GPT-5
        if 'gpt5' in model_clean.lower():
            model_clean = 'GPT-5'
        
        # 2. GPT-o4-mini: o4-mini-m1 -> GPT-o4-mini
        elif 'o4-mini' in model_clean.lower():
            model_clean = 'GPT-o4-mini'
        
        # 3. DeepSeek: deepseek-0528-v1 or deepseek-r1-0528-v1 -> DeepSeek-R1
        elif 'deepseek' in model_clean.lower():
            model_clean = 'DeepSeek-R1'
        
        # 4. Qwen3: qwen3-235b -> Qwen3-235b
        elif 'qwen3' in model_clean.lower() or 'qwen-3' in model_clean.lower():
            model_clean = 'Qwen3-235b'
        
        # 5. OSS models: oss20b (L) -> gpt-oss-20b (L), etc.
        elif 'oss20b' in model_clean.lower():
            if '(l)' in model_clean.lower():
                model_clean = 'gpt-oss-20b (L)'
            elif '(m)' in model_clean.lower():
                model_clean = 'gpt-oss-20b (M)'
            elif '(h)' in model_clean.lower():
                model_clean = 'gpt-oss-20b (H)'
        
        # 5. OSS 120b models: oss120b (L) -> gpt-oss-120b (L), etc.
        elif 'oss120b' in model_clean.lower():
            if '(l)' in model_clean.lower():
                model_clean = 'gpt-oss-120b (L)'
            elif '(m)' in model_clean.lower():
                model_clean = 'gpt-oss-120b (M)'
            elif '(h)' in model_clean.lower():
                model_clean = 'gpt-oss-120b (H)'
        
        filtered_model_indices.append(idx)
        filtered_model_names.append(model_clean)
    
    # Initialize error lists for each filtered model
    for model in filtered_model_names:
        model_errors[model] = []
    
    for i, line in enumerate(lines[1:], 1):  # Skip header
        line = line.strip()
        if not line:
            continue
            
        # Split line by comma
        parts = line.split(',')
        
        # Check if this line contains scores (ends with numeric values)
        if len(parts) >= len(header):
            # Check if the last few parts contain numeric scores
            score_parts = parts[6:]  # Skip metadata columns
            numeric_count = 0
            scores = []
            
            for part in score_parts:
                part = part.strip().strip('"')
                if part and re.match(r'^[0-9]*\.?[0-9]+$', part):
                    numeric_count += 1
                    scores.append(float(part))
                elif part:
                    scores.append(None)
                else:
                    scores.append(None)
            
            # If we have enough numeric scores, this is likely a score line
            if numeric_count >= 5:  # At least 5 numeric scores
                # Get human eval score (should be in column 5, index 5)
                human_score_str = parts[5].strip().strip('"')
                
                if human_score_str and re.match(r'^[0-9]*\.?[0-9]+$', human_score_str):
                    human_score = float(human_score_str)
                    
                    # Calculate errors for each filtered model using original indices
                    for idx, model_name in zip(filtered_model_indices, filtered_model_names):
                        if idx < len(scores) and scores[idx] is not None:
                            error = human_score - scores[idx]
                            model_errors[model_name].append(error)
    
    return model_errors

def create_violin_plot(model_errors, title, filename, output_dir):
    """Create violin plot for a single dataset"""
    # Nature-style consistent color scheme
    NATURE_COLORS = {
        'GPT-5': '#F28E8C',
        'GPT-o4-mini': '#56B3C4',
        'DeepSeek-R1': '#E6C24F',
        'Qwen3-235b': '#B88FD6',
        'gpt-oss-20b (L)': '#F9B8B2',
        'gpt-oss-20b (M)': '#F49389',
        'gpt-oss-20b (H)': '#ED7C72',
        'gpt-oss-120b (L)': '#D6B8E9',
        'gpt-oss-120b (M)': '#C59BDD',
        'gpt-oss-120b (H)': '#B280D1',
    }
    
    # Create series and values lists following the reference format
    series = []
    values = []

    # Add errors for each model (following the reference pattern)
    for model, errors in model_errors.items():
        if errors:  # Only add models that have data
            values.extend(errors)
            series.extend([model] * len(errors))

    # Create DataFrame following the reference format
    dic = {"series": series, "values": values}
    df = pd.DataFrame(dic)

    # Create figure following the reference style
    fig = go.Figure()

    # Add violin plots for each category with consistent colors
    for category in df['series'].unique():
        color = NATURE_COLORS.get(category, '#999999')  # Gray fallback
        fig.add_trace(go.Violin(
            x=df['series'][df['series'] == category],
            y=df['values'][df['series'] == category],
            fillcolor=color,
            line=dict(color=color),
            opacity=0.7
        ))

    # Add box plot overlay (following reference format)
    fig.add_trace(go.Box(x=df['series'],
                         y=df['values'],
                         width=0.1,
                         fillcolor="lightgray",
                         line=dict(width=0.8, color='black')))

    # Update layout following the reference format
    # Figure size: 42cm width × 12cm height
    # Convert to pixels at 96 DPI: 42cm/2.54*96 = 1587.4px, 12cm/2.54*96 = 453.5px
    fig_width = int(42 / 2.54 * 96)  # 1587 pixels
    fig_height = int(12 / 2.54 * 96)  # 454 pixels
    
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[-4.1, 5.5])
    fig.update_layout(
        font=dict(family='Times New Roman', size=17, color="#000000"),
        width=fig_width,
        height=fig_height,
        xaxis=dict(title=""),
        yaxis=dict(title="Error",
                   tickvals=[-4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                   ticktext=["-4.00", "-3.00", "-2.00", "-1.00", "0.00", "1.00", "2.00", "3.00", "4.00", "5.00"],
                   showticklabels=True),
        title=dict(text='')
    )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=25))

    # Save the plot following reference format
    output_path = output_dir / filename
    fig.write_image(str(output_path), width=fig_width, height=fig_height, scale=4)
    
    return len(values), model_errors, output_path

# Set up paths
data_dir = Path(__file__).parent / "data"
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Process diagnosis data
print("Loading and parsing diagnosis data...")
diagnosis_csv = data_dir / "OSS Benchmarking Results - NMED Diagnosis.csv"
diagnosis_model_errors = parse_nmed_csv(str(diagnosis_csv))

print("Creating diagnosis violin plot...")
diagnosis_count, diagnosis_errors, diagnosis_output = create_violin_plot(diagnosis_model_errors, "Diagnosis", "nmed_diagnosis_violin.png", output_dir)

# Process treatment data  
print("Loading and parsing treatment data...")
treatment_csv = data_dir / "OSS Benchmarking Results - NMED Treatment.csv"
treatment_model_errors = parse_nmed_csv(str(treatment_csv))

print("Creating treatment violin plot...")
treatment_count, treatment_errors, treatment_output = create_violin_plot(treatment_model_errors, "Treatment", "nmed_treatment_violin.png", output_dir)

# Display summary statistics
print(f"\n=== DIAGNOSIS DATA ===")
print(f"Total error calculations: {diagnosis_count}")
for model, errors in diagnosis_errors.items():
    if errors:
        print(f"{model}: Count: {len(errors)}, Mean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}")

print(f"\n=== TREATMENT DATA ===")
print(f"Total error calculations: {treatment_count}")
for model, errors in treatment_errors.items():
    if errors:
        print(f"{model}: Count: {len(errors)}, Mean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}")

print(f"\nGenerated files:")
print(f"- {diagnosis_output}")
print(f"- {treatment_output}")

# Uncomment to show plot interactively
# fig.show()
