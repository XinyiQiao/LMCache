#!/usr/bin/env python3
"""
Plot vLLM engine stats over time from joined_request_analysis.csv.

Visualizes the 10-second engine metrics:
  - Concurrent requests (running + waiting)
  - External cache hit rate (%)
  - Prompt throughput (tok/s)
  - Generation throughput (tok/s)
  - GPU KV cache usage (%)
  - Prefix cache hit rate (%)
"""

import csv
import argparse
from datetime import datetime
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_engine_stats(path):
    """Extract unique engine stat snapshots from joined CSV.

    Each row has the nearest engine stats snapshot attached, so many rows
    share the same snapshot. We deduplicate by (timestamp, all metric values).
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))

    seen = OrderedDict()
    for r in rows:
        if not r["concurrent_requests"]:
            continue
        # Use all metric values as key to deduplicate
        key = (
            r["prefix_cache_hit_rate"],
            r["external_cache_hit_rate"],
            r["gpu_kv_usage_pct"],
            r["concurrent_requests"],
            r["prompt_throughput"],
            r["generation_throughput"],
        )
        if key not in seen:
            # Use server_timestamp if available, else client_timestamp
            ts_str = r["server_timestamp"] or r["client_timestamp"]
            seen[key] = {
                "timestamp": datetime.fromisoformat(ts_str),
                "concurrent": int(r["concurrent_requests"]),
                "ext_hit": float(r["external_cache_hit_rate"]),
                "pfx_hit": float(r["prefix_cache_hit_rate"]),
                "gpu_kv": float(r["gpu_kv_usage_pct"]),
                "prompt_tp": float(r["prompt_throughput"]),
                "gen_tp": float(r["generation_throughput"]),
            }

    stats = sorted(seen.values(), key=lambda s: s["timestamp"])
    print(f"Loaded {len(stats)} unique engine stat snapshots")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="joined_request_analysis.csv",
        help="Path to joined CSV file",
    )
    parser.add_argument(
        "--output", default="engine_stats_over_time.png",
        help="Output plot file",
    )
    args = parser.parse_args()

    stats = load_engine_stats(args.input)

    timestamps = [s["timestamp"] for s in stats]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # --- Panel 1: Concurrent requests ---
    ax = axes[0]
    ax.plot(timestamps, [s["concurrent"] for s in stats],
            color="#1976D2", linewidth=1.2)
    ax.fill_between(timestamps, [s["concurrent"] for s in stats],
                     alpha=0.15, color="#1976D2")
    ax.set_ylabel("Concurrent Requests")
    ax.set_title("Concurrent Requests (Running + Waiting)")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Cache hit rates ---
    ax = axes[1]
    ax.plot(timestamps, [s["ext_hit"] for s in stats],
            color="#F44336", linewidth=1.2, label="External Cache Hit %")
    ax.plot(timestamps, [s["pfx_hit"] for s in stats],
            color="#2196F3", linewidth=1.2, label="Prefix Cache Hit %")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Cache Hit Rates")
    ax.legend(loc="right", fontsize=9)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Throughput ---
    ax = axes[2]
    ax.plot(timestamps, [s["prompt_tp"] for s in stats],
            color="#FF9800", linewidth=1.2, label="Prompt Throughput")
    ax.plot(timestamps, [s["gen_tp"] for s in stats],
            color="#4CAF50", linewidth=1.2, label="Generation Throughput")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Throughput")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: GPU KV cache usage ---
    ax = axes[3]
    ax.plot(timestamps, [s["gpu_kv"] for s in stats],
            color="#9C27B0", linewidth=1.2)
    ax.fill_between(timestamps, [s["gpu_kv"] for s in stats],
                     alpha=0.15, color="#9C27B0")
    ax.set_ylabel("GPU KV Usage (%)")
    ax.set_title("GPU KV Cache Usage")
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time")

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.suptitle("vLLM Engine Stats Over Time", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
