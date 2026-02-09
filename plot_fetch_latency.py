#!/usr/bin/env python3
"""
Zoom in on the LMCache Fetch group (tokens_to_load > 0, lmcache_hit_tokens > 0).
Visualize the relationship between request latency and # of tokens loaded from LMCache.

Two panels:
  - Scatter plot with trend line
  - Box plot bucketed by tokens-to-load range
"""

import csv
import argparse
import statistics

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="joined_request_analysis.csv",
        help="Path to joined CSV file",
    )
    parser.add_argument(
        "--output", default="fetch_latency_vs_tokens.png",
        help="Output plot file",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    # LMCache Fetch group only
    fetch = [
        r for r in rows
        if int(r["lmcache_hit_tokens"]) > 0
        and int(r["tokens_to_load_from_lmcache"]) > 0
        and r["latency_sec"]
    ]
    print(f"LMCache Fetch requests: {len(fetch)}")

    tokens = np.array([int(r["tokens_to_load_from_lmcache"]) for r in fetch])
    latency = np.array([float(r["latency_sec"]) for r in fetch])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # --- Left panel: scatter + trend ---
    ax1.scatter(tokens, latency, alpha=0.25, s=10, color="#F44336")

    # Bin-averaged trend line
    bin_edges = np.arange(0, tokens.max() + 2000, 2000)
    bin_meds = []
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        mask = (tokens >= bin_edges[i]) & (tokens < bin_edges[i + 1])
        if mask.sum() >= 5:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_meds.append(np.median(latency[mask]))

    ax1.plot(bin_centers, bin_meds, color="black", linewidth=2,
             marker="o", markersize=5, label="Bin median (2k-token bins)")

    # Linear fit
    coef = np.polyfit(tokens, latency, 1)
    x_fit = np.linspace(tokens.min(), tokens.max(), 100)
    ax1.plot(x_fit, np.polyval(coef, x_fit), color="#1976D2",
             linestyle="--", linewidth=1.5,
             label=f"Linear fit: {coef[0]*1000:.2f}ms per 1k tokens")

    ax1.set_xlabel("Tokens to Load from LMCache", fontsize=12)
    ax1.set_ylabel("Request Latency (s)", fontsize=12)
    ax1.set_title("Request Latency vs Tokens Loaded from LMCache", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Right panel: box plot by bucket ---
    buckets = [
        (0, 1000, "0-1k"),
        (1000, 3000, "1k-3k"),
        (3000, 5000, "3k-5k"),
        (5000, 10000, "5k-10k"),
        (10000, 20000, "10k-20k"),
        (20000, 50000, "20k+"),
    ]

    box_data = []
    box_labels = []
    for lo, hi, label in buckets:
        vals = latency[(tokens >= lo) & (tokens < hi)]
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(f"{label}\n(n={len(vals)})")

    bp = ax2.boxplot(
        box_data,
        tick_labels=box_labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for box in bp["boxes"]:
        box.set_facecolor("#F44336")
        box.set_alpha(0.6)

    # Add median labels
    for i, vals in enumerate(box_data):
        med = np.median(vals)
        ax2.annotate(
            f"{med:.1f}s",
            xy=(i + 1, med),
            xytext=(i + 1, med + 1.5),
            fontsize=8, ha="center",
        )

    ax2.set_xlabel("Tokens to Load from LMCache", fontsize=12)
    ax2.set_ylabel("Request Latency (s)", fontsize=12)
    ax2.set_title("Latency Distribution by Load Bucket", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
