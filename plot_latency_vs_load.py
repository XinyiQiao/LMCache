#!/usr/bin/env python3
"""
Plot request latency vs tokens to load from LMCache for requests
with a prefix cache hit (excluding cold starts).

Left panel:  Box plot comparing GPU Hit vs LMCache Fetch latency distributions
Right panel: Scatter plot of latency vs tokens to load
"""

import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="joined_request_analysis.csv",
        help="Path to joined CSV file",
    )
    parser.add_argument(
        "--output", default="latency_vs_tokens_to_load.png",
        help="Output plot file",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    # Filter: only prefix cache hits (exclude cold starts)
    rows = [
        r for r in rows
        if int(r["lmcache_hit_tokens"]) > 0 and r["latency_sec"]
    ]

    tokens_to_load = [int(r["tokens_to_load_from_lmcache"]) for r in rows]
    latency = [float(r["latency_sec"]) for r in rows]

    # Split into GPU Hit (load=0) vs LMCache Fetch (load>0)
    gpu_y = [l for l, t in zip(latency, tokens_to_load) if t == 0]
    lmc_y = [l for l, t in zip(latency, tokens_to_load) if t > 0]

    fig, ax = plt.subplots(figsize=(6, 6))

    bp = ax.boxplot(
        [gpu_y, lmc_y],
        tick_labels=[f"GPU Hit\n(n={len(gpu_y)})", f"LMCache Fetch\n(n={len(lmc_y)})"],
        patch_artist=True,
        showfliers=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
    )
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#F44336")
    bp["boxes"][1].set_alpha(0.7)

    # Add median labels
    for i, data in enumerate([gpu_y, lmc_y]):
        med = np.median(data)
        ax.annotate(
            f"med={med:.1f}s",
            xy=(i + 1, med),
            xytext=(i + 1.35, med),
            fontsize=9,
            va="center",
        )

    ax.set_ylabel("Request Latency (s)", fontsize=12)
    ax.set_title(
        "Request Latency: GPU Hit vs LMCache Fetch\n(prefix cache hit, excl. cold starts)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    print(f"  GPU Hit points:       {len(gpu_y)}")
    print(f"  LMCache Fetch points: {len(lmc_y)}")


if __name__ == "__main__":
    main()
