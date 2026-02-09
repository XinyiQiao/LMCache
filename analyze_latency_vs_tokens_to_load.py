#!/usr/bin/env python3
"""
Analyze how latency_sec changes with tokens_to_load_from_lmcache
for requests where tokens_to_load_from_lmcache > 0.

Creates visualizations and statistical analysis.
"""

import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Analyze latency vs tokens_to_load_from_lmcache relationship"
    )
    parser.add_argument(
        "--input", default="joined_request_analysis.csv",
        help="Path to joined CSV file",
    )
    parser.add_argument(
        "--output", default="latency_vs_tokens_analysis.png",
        help="Output plot file",
    )
    args = parser.parse_args()

    # Read CSV file
    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    # Filter: only rows where tokens_to_load_from_lmcache > 0
    filtered_rows = [
        r for r in rows
        if r["tokens_to_load_from_lmcache"] and 
           int(r["tokens_to_load_from_lmcache"]) > 0 and
           r["latency_sec"] and
           float(r["latency_sec"]) > 0
    ]

    if not filtered_rows:
        print("No rows found with tokens_to_load_from_lmcache > 0")
        return

    tokens_to_load = np.array([int(r["tokens_to_load_from_lmcache"]) for r in filtered_rows])
    latency = np.array([float(r["latency_sec"]) for r in filtered_rows])

    print(f"Total requests analyzed: {len(filtered_rows)}")
    print(f"\nTokens to load statistics:")
    print(f"  Min: {tokens_to_load.min()}")
    print(f"  Max: {tokens_to_load.max()}")
    print(f"  Mean: {tokens_to_load.mean():.2f}")
    print(f"  Median: {np.median(tokens_to_load):.2f}")
    print(f"  Std: {tokens_to_load.std():.2f}")
    
    print(f"\nLatency statistics (seconds):")
    print(f"  Min: {latency.min():.3f}")
    print(f"  Max: {latency.max():.3f}")
    print(f"  Mean: {latency.mean():.3f}")
    print(f"  Median: {np.median(latency):.3f}")
    print(f"  Std: {latency.std():.3f}")

    # Calculate correlation using numpy
    correlation_matrix = np.corrcoef(tokens_to_load, latency)
    correlation = correlation_matrix[0, 1]
    print(f"\nCorrelation analysis:")
    print(f"  Pearson correlation: {correlation:.4f}")
    
    # Linear regression using numpy polyfit
    coeffs = np.polyfit(tokens_to_load, latency, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Calculate R-squared
    y_pred = slope * tokens_to_load + intercept
    ss_res = np.sum((latency - y_pred) ** 2)
    ss_tot = np.sum((latency - np.mean(latency)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nLinear regression:")
    print(f"  Slope: {slope:.6f} (seconds per token)")
    print(f"  Intercept: {intercept:.4f} seconds")
    print(f"  R-squared: {r_squared:.4f}")

    # Bin analysis with detailed statistics
    print(f"\n{'='*80}")
    print(f"Detailed Analysis by Token Bucket:")
    print(f"{'='*80}")
    bins = [0, 100, 500, 1000, 2000, 5000, 10000, float('inf')]
    bin_labels = ['0-100', '100-500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '10K+']
    
    print(f"\n{'Bucket':<12} {'Count':<8} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'P25':<10} {'P75':<10} {'Std':<10}")
    print(f"{'-'*80}")
    
    for i in range(len(bins) - 1):
        mask = (tokens_to_load >= bins[i]) & (tokens_to_load < bins[i + 1])
        if mask.sum() > 0:
            bin_latency = latency[mask]
            bin_tokens = tokens_to_load[mask]
            print(f"{bin_labels[i]:<12} {mask.sum():<8} "
                  f"{bin_latency.mean():<10.3f} {np.median(bin_latency):<10.3f} "
                  f"{bin_latency.min():<10.3f} {bin_latency.max():<10.3f} "
                  f"{np.percentile(bin_latency, 25):<10.3f} {np.percentile(bin_latency, 75):<10.3f} "
                  f"{bin_latency.std():<10.3f}")
    
    print(f"\n{'='*80}")
    print(f"Summary by Bucket (Latency in seconds):")
    print(f"{'='*80}")
    
    for i in range(len(bins) - 1):
        mask = (tokens_to_load >= bins[i]) & (tokens_to_load < bins[i + 1])
        if mask.sum() > 0:
            bin_latency = latency[mask]
            bin_tokens = tokens_to_load[mask]
            print(f"\n{bin_labels[i]} tokens (n={mask.sum()}):")
            print(f"  Token range: {bin_tokens.min():.0f} - {bin_tokens.max():.0f} (mean: {bin_tokens.mean():.1f})")
            print(f"  Latency - Mean: {bin_latency.mean():.3f}s, Median: {np.median(bin_latency):.3f}s")
            print(f"  Latency - Min: {bin_latency.min():.3f}s, Max: {bin_latency.max():.3f}s")
            print(f"  Latency - P25: {np.percentile(bin_latency, 25):.3f}s, P75: {np.percentile(bin_latency, 75):.3f}s")
            print(f"  Latency - Std: {bin_latency.std():.3f}s")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Scatter plot with regression line
    ax = axes[0, 0]
    ax.scatter(tokens_to_load, latency, alpha=0.3, s=10, color='steelblue')
    
    # Add regression line
    x_line = np.linspace(tokens_to_load.min(), tokens_to_load.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, 
            label=f'y = {slope:.6f}x + {intercept:.3f}\nR² = {r_squared:.4f}')
    
    ax.set_xlabel('Tokens to Load from LMCache', fontsize=11)
    ax.set_ylabel('Latency (seconds)', fontsize=11)
    ax.set_title('Latency vs Tokens to Load (with regression line)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Binned box plot
    ax = axes[0, 1]
    bin_data = []
    bin_positions = []
    bin_labels_plot = []
    
    for i in range(len(bins) - 1):
        mask = (tokens_to_load >= bins[i]) & (tokens_to_load < bins[i + 1])
        if mask.sum() > 0:
            bin_data.append(latency[mask])
            bin_positions.append(i + 1)
            bin_labels_plot.append(bin_labels[i])
    
    bp = ax.boxplot(bin_data, positions=bin_positions, 
                    tick_labels=bin_labels_plot, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Tokens to Load (binned)', fontsize=11)
    ax.set_ylabel('Latency (seconds)', fontsize=11)
    ax.set_title('Latency Distribution by Token Range', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Histogram of tokens to load
    ax = axes[1, 0]
    ax.hist(tokens_to_load, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Tokens to Load from LMCache', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Tokens to Load', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Histogram of latency
    ax = axes[1, 1]
    ax.hist(latency, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Latency (seconds)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Latency', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
