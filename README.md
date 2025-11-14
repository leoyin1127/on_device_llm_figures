# On-device LLMs Benchmarking - Data Visualization

Data visualization scripts for generating publication figures from on-device LLMs benchmark results.

## Usage

```bash
# Setup
uv sync

# Generate bar charts
uv run python chart2bar/visualization_barplot.py
uv run python chart2bar/finetuned_comparison_plot.py

# Generate radar plots
uv run python chart2radar/eurorad_radar_plots.py

# Generate violin plots
uv run python chart2violin/nmed_error_violin_plot.py
```

## Structure

- `chart2bar/` - Bar charts comparing base model accuracy and finetuned models
- `chart2radar/` - Radar plots for multi-dimensional comparison  
- `chart2violin/` - Violin plots for error distribution analysis

Place CSV data files in respective `data/` directories. Output figures are saved to `output/` directories.