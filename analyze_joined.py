#!/usr/bin/env python3
"""
Analyze joined_request_analysis.csv to compare performance when KV cache
resides in GPU memory vs when it must be fetched from LMCache.

Groups requests into three categories based on per-request LMCache lookup:
  - GPU Hit:      lmcache_hit_tokens > 0, tokens_to_load == 0
                  (prefix found, KV still in GPU — no LMCache fetch needed)
  - LMCache Fetch: lmcache_hit_tokens > 0, tokens_to_load > 0
                  (prefix found, KV evicted from GPU — must fetch from LMCache)
  - Cold Start:   lmcache_hit_tokens == 0
                  (no prefix match at all)
"""

import csv
import argparse
import statistics
from collections import defaultdict


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def percentile(data, pct):
    """Return the pct-th percentile of sorted data."""
    s = sorted(data)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]


def fmt(val, width=12, decimals=3):
    if val is None or val == "N/A":
        return f"{'N/A':<{width}}"
    return f"{val:<{width}.{decimals}f}"


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def classify(row):
    hit = int(row["lmcache_hit_tokens"])
    load = int(row["tokens_to_load_from_lmcache"])
    if hit == 0:
        return "Cold Start"
    elif load == 0:
        return "GPU Hit"
    else:
        return "LMCache Fetch"


def summarize_latency(groups):
    print_section("1. LATENCY COMPARISON: GPU Hit vs LMCache Fetch")
    print(
        "\n  GPU Hit:       prefix matched in LMCache, KV still resident in GPU"
        "\n  LMCache Fetch: prefix matched in LMCache, KV evicted — fetched from LMCache"
        "\n  Cold Start:    no prefix match (first occurrence)\n"
    )

    header = f"{'Group':<18} {'Count':<8} {'Mean(s)':<10} {'Med(s)':<10} {'P25(s)':<10} {'P75(s)':<10} {'P95(s)':<10} {'P99(s)':<10}"
    print(header)
    print("-" * len(header))

    results = {}
    for label in ["GPU Hit", "LMCache Fetch", "Cold Start"]:
        lats = sorted([float(r["latency_sec"]) for r in groups[label]])
        if not lats:
            continue
        results[label] = {
            "count": len(lats),
            "mean": statistics.mean(lats),
            "median": statistics.median(lats),
            "p25": percentile(lats, 25),
            "p75": percentile(lats, 75),
            "p95": percentile(lats, 95),
            "p99": percentile(lats, 99),
        }
        r = results[label]
        print(
            f"{label:<18} {r['count']:<8} {r['mean']:<10.3f} {r['median']:<10.3f} "
            f"{r['p25']:<10.3f} {r['p75']:<10.3f} {r['p95']:<10.3f} {r['p99']:<10.3f}"
        )

    if "GPU Hit" in results and "LMCache Fetch" in results:
        gpu = results["GPU Hit"]
        lmc = results["LMCache Fetch"]
        print(f"\n  >>> Mean latency increase (LMCache Fetch vs GPU Hit): "
              f"{((lmc['mean'] / gpu['mean']) - 1) * 100:+.1f}%")
        print(f"  >>> Median latency increase: "
              f"{((lmc['median'] / gpu['median']) - 1) * 100:+.1f}%")

    return results


def summarize_by_token_bucket(groups):
    print_section("2. LATENCY BY PROMPT TOKEN BUCKET (apples-to-apples)")
    print("\n  Controls for prompt size to isolate LMCache fetch overhead.\n")

    buckets = [
        (0, 2000),
        (2000, 5000),
        (5000, 10000),
        (10000, 20000),
        (20000, 50000),
    ]

    header = (
        f"{'Tokens':<15} "
        f"{'GPU N':<7} {'GPU Med':<10} {'GPU Mean':<10} "
        f"{'LMC N':<7} {'LMC Med':<10} {'LMC Mean':<10} "
        f"{'Med Ratio':<10}"
    )
    print(header)
    print("-" * len(header))

    for lo, hi in buckets:
        gpu_lats = sorted(
            [float(r["latency_sec"]) for r in groups["GPU Hit"]
             if lo <= int(r["prompt_tokens"]) < hi]
        )
        lmc_lats = sorted(
            [float(r["latency_sec"]) for r in groups["LMCache Fetch"]
             if lo <= int(r["prompt_tokens"]) < hi]
        )

        gpu_med = statistics.median(gpu_lats) if gpu_lats else None
        gpu_mean = statistics.mean(gpu_lats) if gpu_lats else None
        lmc_med = statistics.median(lmc_lats) if lmc_lats else None
        lmc_mean = statistics.mean(lmc_lats) if lmc_lats else None

        ratio = f"{lmc_med / gpu_med:.2f}x" if gpu_med and lmc_med else "N/A"
        hi_str = str(hi) if hi < 50000 else "50k+"

        print(
            f"{lo}-{hi_str:<10} "
            f"{len(gpu_lats):<7} {fmt(gpu_med, 10)} {fmt(gpu_mean, 10)} "
            f"{len(lmc_lats):<7} {fmt(lmc_med, 10)} {fmt(lmc_mean, 10)} "
            f"{ratio:<10}"
        )


def summarize_tokens(groups):
    print_section("3. PROMPT SIZE & CACHE HIT PROFILE")

    header = f"{'Group':<18} {'Count':<8} {'Avg Prompt':<12} {'Avg Hit Tok':<12} {'Avg Load Tok':<12} {'Hit Ratio':<10}"
    print()
    print(header)
    print("-" * len(header))

    for label in ["GPU Hit", "LMCache Fetch", "Cold Start"]:
        rows = groups[label]
        if not rows:
            continue
        prompt = statistics.mean([int(r["prompt_tokens"]) for r in rows])
        hit = statistics.mean([int(r["lmcache_hit_tokens"]) for r in rows])
        load = statistics.mean([int(r["tokens_to_load_from_lmcache"]) for r in rows])
        ratio = hit / prompt if prompt > 0 else 0
        print(
            f"{label:<18} {len(rows):<8} {prompt:<12.0f} {hit:<12.0f} "
            f"{load:<12.0f} {ratio:<10.1%}"
        )


def summarize_engine_state(groups):
    print_section("4. ENGINE STATE AT REQUEST TIME")
    print("\n  Server-level metrics (sampled every ~10s) nearest to each request.\n")

    header = (
        f"{'Group':<18} {'GPU KV%':<10} {'Concurrent':<12} "
        f"{'Prompt TP':<12} {'Gen TP':<12} {'Pfx Hit%':<10} {'Ext Hit%':<10}"
    )
    print(header)
    print("-" * len(header))

    for label in ["GPU Hit", "LMCache Fetch", "Cold Start"]:
        rows = groups[label]
        if not rows:
            continue
        valid = [r for r in rows if r["gpu_kv_usage_pct"]]
        if not valid:
            continue
        gpu_pct = statistics.mean([float(r["gpu_kv_usage_pct"]) for r in valid])
        conc = statistics.mean([int(r["concurrent_requests"]) for r in valid])
        pthr = statistics.mean([float(r["prompt_throughput"]) for r in valid])
        gthr = statistics.mean([float(r["generation_throughput"]) for r in valid])
        pfx = statistics.mean([float(r["prefix_cache_hit_rate"]) for r in valid])
        ext = statistics.mean([float(r["external_cache_hit_rate"]) for r in valid])
        print(
            f"{label:<18} {gpu_pct:<10.1f} {conc:<12.1f} "
            f"{pthr:<12.1f} {gthr:<12.1f} {pfx:<10.1f} {ext:<10.1f}"
        )


def summarize_load_vs_latency(groups):
    print_section("5. LATENCY vs TOKENS TO LOAD (LMCache Fetch group only)")
    print("\n  How does the number of tokens fetched from LMCache affect latency?\n")

    rows = groups["LMCache Fetch"]
    buckets = [
        (0, 256),
        (256, 1000),
        (1000, 5000),
        (5000, 10000),
        (10000, 50000),
    ]

    header = f"{'Tokens to Load':<18} {'Count':<8} {'Med Lat(s)':<12} {'Mean Lat(s)':<12} {'Avg Prompt':<12}"
    print(header)
    print("-" * len(header))

    for lo, hi in buckets:
        in_bucket = [
            r for r in rows
            if lo <= int(r["tokens_to_load_from_lmcache"]) < hi
        ]
        if not in_bucket:
            continue
        lats = [float(r["latency_sec"]) for r in in_bucket]
        toks = [int(r["prompt_tokens"]) for r in in_bucket]
        hi_str = str(hi) if hi < 50000 else "50k+"
        print(
            f"{lo}-{hi_str:<13} {len(in_bucket):<8} "
            f"{statistics.median(lats):<12.3f} {statistics.mean(lats):<12.3f} "
            f"{statistics.mean(toks):<12.0f}"
        )


def summarize_findings(groups):
    print_section("6. KEY FINDINGS")

    gpu_rows = groups["GPU Hit"]
    lmc_rows = groups["LMCache Fetch"]

    gpu_lats = [float(r["latency_sec"]) for r in gpu_rows]
    lmc_lats = [float(r["latency_sec"]) for r in lmc_rows]

    gpu_valid = [r for r in gpu_rows if r["gpu_kv_usage_pct"]]
    lmc_valid = [r for r in lmc_rows if r["gpu_kv_usage_pct"]]

    print(f"""
  Requests analyzed: {len(gpu_rows) + len(lmc_rows) + len(groups['Cold Start'])} total
    - GPU Hit (KV in GPU):        {len(gpu_rows)} ({len(gpu_rows)*100/(len(gpu_rows)+len(lmc_rows)+len(groups['Cold Start'])):.1f}%)
    - LMCache Fetch (KV evicted): {len(lmc_rows)} ({len(lmc_rows)*100/(len(gpu_rows)+len(lmc_rows)+len(groups['Cold Start'])):.1f}%)
    - Cold Start:                 {len(groups['Cold Start'])} ({len(groups['Cold Start'])*100/(len(gpu_rows)+len(lmc_rows)+len(groups['Cold Start'])):.1f}%)

  Latency (GPU Hit vs LMCache Fetch):
    - GPU Hit:       mean={statistics.mean(gpu_lats):.3f}s, median={statistics.median(gpu_lats):.3f}s
    - LMCache Fetch: mean={statistics.mean(lmc_lats):.3f}s, median={statistics.median(lmc_lats):.3f}s
    - Median ratio:  {statistics.median(lmc_lats)/statistics.median(gpu_lats):.2f}x

  System conditions when each group occurs:
    - GPU Hit:       avg GPU KV usage={statistics.mean([float(r['gpu_kv_usage_pct']) for r in gpu_valid]):.1f}%, avg concurrent={statistics.mean([int(r['concurrent_requests']) for r in gpu_valid]):.0f}
    - LMCache Fetch: avg GPU KV usage={statistics.mean([float(r['gpu_kv_usage_pct']) for r in lmc_valid]):.1f}%, avg concurrent={statistics.mean([int(r['concurrent_requests']) for r in lmc_valid]):.0f}

  Confounds:
    - LMCache Fetch requests occur under higher GPU pressure and concurrency.
    - The latency difference reflects BOTH LMCache fetch overhead AND system
      contention. See Section 2 (token-bucketed comparison) to partially
      control for prompt size.""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze joined request logs: GPU Hit vs LMCache Fetch"
    )
    parser.add_argument(
        "--input", default="joined_request_analysis.csv",
        help="Path to joined CSV file"
    )
    args = parser.parse_args()

    rows = load_csv(args.input)
    print(f"Loaded {len(rows)} requests from {args.input}")

    # Classify each request
    groups = defaultdict(list)
    for r in rows:
        groups[classify(r)].append(r)

    print(f"\nClassification:")
    for label in ["GPU Hit", "LMCache Fetch", "Cold Start"]:
        print(f"  {label}: {len(groups[label])}")

    # Run all analyses
    summarize_latency(groups)
    summarize_by_token_bucket(groups)
    summarize_tokens(groups)
    summarize_engine_state(groups)
    summarize_load_vs_latency(groups)
    summarize_findings(groups)


if __name__ == "__main__":
    main()
