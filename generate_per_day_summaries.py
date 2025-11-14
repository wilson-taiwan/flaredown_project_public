#!/usr/bin/env python3
"""
Generate per-day training summary files for Random Forest models.
This aligns with the per-horizon approach used in publication tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def generate_per_day_summary(condition: str):
    """
    Generate a per-day training summary file for the specified condition.
    
    Args:
        condition: 'depression' or 'anxiety'
    """
    # Paths
    base_path = Path(__file__).parent
    results_path = base_path / 'random_forest_output' / 'test_results' / condition
    metrics_file = results_path / 'evaluation_metrics.csv'
    original_summary = results_path / 'training_summary.txt'
    output_file = results_path / 'training_summary_per_day.txt'
    
    # Read the CSV with per-day metrics
    df = pd.read_csv(metrics_file)
    
    # Filter out the 'Overall' row and keep only day-specific rows
    df_days = df[df['day'] != 'Overall'].copy()
    
    # Convert day column to int for proper sorting
    df_days['day'] = df_days['day'].astype(int)
    df_days = df_days.sort_values('day')
    
    # Calculate mean and std across all days
    mae_per_day = df_days['test_mae_mean'].values
    rmse_per_day = df_days['test_rmse_mean'].values
    r2_per_day = df_days['test_r2_mean'].values
    
    mean_mae = np.mean(mae_per_day)
    std_mae = np.std(mae_per_day, ddof=1)  # Sample std
    
    mean_rmse = np.mean(rmse_per_day)
    std_rmse = np.std(rmse_per_day, ddof=1)
    
    mean_r2 = np.mean(r2_per_day)
    std_r2 = np.std(r2_per_day, ddof=1)
    
    # Read original summary for configuration details
    with open(original_summary, 'r') as f:
        original_lines = f.readlines()
    
    # Extract configuration section
    config_start = None
    config_end = None
    timestamp_line = None
    
    for i, line in enumerate(original_lines):
        if 'Timestamp:' in line:
            timestamp_line = line.strip()
        if 'Configuration:' in line:
            config_start = i
        if 'Total Training Time:' in line:
            config_end = i
            break
    
    config_section = original_lines[config_start:config_end] if config_start and config_end else []
    
    # Generate new summary file
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"RANDOM FOREST {condition.upper()} FORECASTING - PER-DAY TRAINING SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if timestamp_line:
        output_lines.append(f"Original Training: {timestamp_line.split(':', 1)[1].strip()}")
    output_lines.append("")
    
    # Add configuration section
    if config_section:
        output_lines.extend([line.rstrip() for line in config_section])
        output_lines.append("")
    
    output_lines.append("=" * 80)
    output_lines.append("PER-DAY PERFORMANCE METRICS")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("This summary reports metrics using the PER-DAY approach:")
    output_lines.append("  1. Calculate MAE/RMSE/R² for each forecasting horizon (days 11-20)")
    output_lines.append("  2. Compute mean and standard deviation across all horizons")
    output_lines.append("")
    output_lines.append("This aligns with the methodology used in publication-ready tables and")
    output_lines.append("ensures fair comparison with LLM models that report per-horizon metrics.")
    output_lines.append("")
    output_lines.append("-" * 80)
    output_lines.append("")
    
    # Summary statistics
    output_lines.append(f"Mean Per-Day Performance (averaged across {len(df_days)} forecasting horizons):")
    output_lines.append(f"  • Mean MAE:  {mean_mae:.4f} ± {std_mae:.4f}")
    output_lines.append(f"  • Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    output_lines.append(f"  • Mean R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    output_lines.append("")
    output_lines.append("-" * 80)
    output_lines.append("")
    
    # Detailed per-day breakdown
    output_lines.append("Detailed Performance by Forecasting Horizon:")
    output_lines.append("")
    output_lines.append(f"{'Day':<6} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    output_lines.append("-" * 48)
    
    for _, row in df_days.iterrows():
        day = int(row['day'])
        mae = row['test_mae_mean']
        rmse = row['test_rmse_mean']
        r2 = row['test_r2_mean']
        output_lines.append(f"{day:<6} {mae:<12.4f} {rmse:<12.4f} {r2:<12.4f}")
    
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Add note about difference from original summary
    output_lines.append("NOTE: Comparison with Original Training Summary")
    output_lines.append("-" * 80)
    output_lines.append("")
    output_lines.append("The original training_summary.txt reports GLOBAL metrics:")
    output_lines.append("  • Averages all predictions across all horizons into single values")
    output_lines.append("  • Weights each horizon by the number of test samples")
    output_lines.append("")
    output_lines.append("This per-day summary reports PER-HORIZON metrics:")
    output_lines.append("  • Calculates metrics for each forecasting day separately")
    output_lines.append("  • Treats each horizon equally in the final average")
    output_lines.append("")
    output_lines.append("Use THIS summary for:")
    output_lines.append("  ✓ Publication tables comparing RF with LLM models")
    output_lines.append("  ✓ Statistical tests (t-tests, Wilcoxon) on per-horizon performance")
    output_lines.append("  ✓ Analyzing how error scales with forecasting distance")
    output_lines.append("")
    output_lines.append("Use the ORIGINAL summary for:")
    output_lines.append("  ✓ Quick overall model performance summaries")
    output_lines.append("  ✓ Abstract or high-level performance descriptions")
    output_lines.append("")
    output_lines.append("=" * 80)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✓ Generated: {output_file}")
    print(f"  Mean MAE:  {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  Mean R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    print()


def main():
    """Generate per-day summaries for both conditions."""
    print("Generating per-day training summaries...")
    print("=" * 80)
    print()
    
    for condition in ['depression', 'anxiety']:
        print(f"Processing {condition}...")
        generate_per_day_summary(condition)
    
    print("=" * 80)
    print("✓ All per-day summaries generated successfully!")
    print()
    print("Files created:")
    print("  • random_forest_output/test_results/depression/training_summary_per_day.txt")
    print("  • random_forest_output/test_results/anxiety/training_summary_per_day.txt")


if __name__ == '__main__':
    main()
