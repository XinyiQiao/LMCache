#!/usr/bin/env python3
"""
Parse vLLM + LMCache logs to study KV cache behavior.

Extracts:
1. Engine-level metrics (every 10 seconds):
   - Prefix cache hit rate (GPU)
   - External prefix cache hit rate (LMCache)
   - GPU KV cache usage
   - Concurrency (Running + Waiting requests)

2. Request-level LMCache metrics:
   - Total tokens per request
   - LMCache hit tokens (cached in LMCache)
   - Tokens that need to be loaded from LMCache to GPU

3. KV cache store/offload metrics:
   - Tokens stored
   - Store latency and throughput

Goal: Identify when prefix cache hits but KV cache was evicted from GPU to LMCache,
requiring fetching from LMCache (which may increase latency).
"""

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from collections import defaultdict
import argparse


@dataclass
class EngineStats:
    """Engine-level statistics logged every 10 seconds"""
    timestamp: datetime
    prompt_throughput: float  # tokens/s
    generation_throughput: float  # tokens/s
    running_reqs: int
    waiting_reqs: int
    gpu_kv_usage_pct: float
    prefix_cache_hit_rate: float  # GPU prefix cache
    external_cache_hit_rate: float  # LMCache hit rate


@dataclass
class LMCacheRequestLookup:
    """LMCache lookup info for a request"""
    timestamp: datetime
    request_id: str
    total_tokens: int
    lmcache_hit_tokens: int
    need_to_load: int  # tokens to load from LMCache to GPU


@dataclass
class LMCacheStoreEvent:
    """KV cache store event"""
    timestamp: datetime
    request_id: str
    stored_tokens: int
    total_tokens: int
    skip_leading_tokens: int
    size_gb: float
    cost_ms: float
    throughput_gbs: float
    offload_time_ms: float
    put_time_ms: float


def parse_timestamp_from_log(log_line: str) -> Optional[datetime]:
    """Extract timestamp from log line"""
    # Format: INFO MM-DD HH:MM:SS
    match = re.search(r'INFO (\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})', log_line)
    if match:
        date_str, time_str = match.groups()
        # Assume current year (2026 from the logs)
        return datetime.strptime(f"2026-{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

    # LMCache format: [2026-01-29 22:23:54,704]
    match = re.search(r'\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]', log_line)
    if match:
        date_str, time_str, ms_str = match.groups()
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return dt.replace(microsecond=int(ms_str) * 1000)

    return None


def parse_engine_stats(line: str) -> Optional[EngineStats]:
    """Parse engine statistics line"""
    if 'loggers.py' not in line or 'Engine 000:' not in line:
        return None

    timestamp = parse_timestamp_from_log(line)
    if not timestamp:
        return None

    # Extract metrics
    prompt_tp = re.search(r'Avg prompt throughput: ([\d.]+)', line)
    gen_tp = re.search(r'Avg generation throughput: ([\d.]+)', line)
    running = re.search(r'Running: (\d+) reqs', line)
    waiting = re.search(r'Waiting: (\d+) reqs', line)
    gpu_usage = re.search(r'GPU KV cache usage: ([\d.]+)%', line)
    prefix_hit = re.search(r'Prefix cache hit rate: ([\d.]+)%', line)
    external_hit = re.search(r'External prefix cache hit rate: ([\d.]+)%', line)

    if all([prompt_tp, gen_tp, running, waiting, gpu_usage, prefix_hit, external_hit]):
        return EngineStats(
            timestamp=timestamp,
            prompt_throughput=float(prompt_tp.group(1)),
            generation_throughput=float(gen_tp.group(1)),
            running_reqs=int(running.group(1)),
            waiting_reqs=int(waiting.group(1)),
            gpu_kv_usage_pct=float(gpu_usage.group(1)),
            prefix_cache_hit_rate=float(prefix_hit.group(1)),
            external_cache_hit_rate=float(external_hit.group(1))
        )
    return None


def parse_lmcache_lookup(line: str) -> Optional[LMCacheRequestLookup]:
    """Parse LMCache lookup line"""
    if 'Reqid:' not in line or 'LMCache hit tokens' not in line:
        return None

    timestamp = parse_timestamp_from_log(line)
    if not timestamp:
        return None

    match = re.search(
        r'Reqid: ([^,]+), Total tokens (\d+), LMCache hit tokens: (-?\d+), need to load: (-?\d+)',
        line
    )
    if match:
        return LMCacheRequestLookup(
            timestamp=timestamp,
            request_id=match.group(1),
            total_tokens=int(match.group(2)),
            lmcache_hit_tokens=int(match.group(3)),
            need_to_load=int(match.group(4))
        )
    return None


def parse_lmcache_store(line: str, request_line: str = None) -> Optional[LMCacheStoreEvent]:
    """Parse LMCache store completion line"""
    if 'Stored' not in line or 'throughput' not in line:
        return None

    timestamp = parse_timestamp_from_log(line)
    if not timestamp:
        return None

    # Pattern: Stored X out of total Y tokens. size: X gb, cost Y ms, throughput: Z GB/s; offload_time: X ms, put_time: Y ms
    match = re.search(
        r'Stored (\d+) out of total (\d+) tokens\. size: ([\d.]+) gb, cost ([\d.]+) ms, throughput: ([\d.]+) GB/s; offload_time: ([\d.]+) ms, put_time: ([\d.]+) ms',
        line
    )
    if match:
        return LMCacheStoreEvent(
            timestamp=timestamp,
            request_id="",  # Will be filled from context
            stored_tokens=int(match.group(1)),
            total_tokens=int(match.group(2)),
            skip_leading_tokens=0,
            size_gb=float(match.group(3)),
            cost_ms=float(match.group(4)),
            throughput_gbs=float(match.group(5)),
            offload_time_ms=float(match.group(6)),
            put_time_ms=float(match.group(7))
        )
    return None


def parse_log_file(log_path: str):
    """Parse the vLLM log file and extract all metrics"""
    engine_stats = []
    request_lookups = []
    store_events = []

    print(f"Parsing log file: {log_path}")

    with open(log_path, 'r', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} lines...")

            # Try parsing engine stats
            stats = parse_engine_stats(line)
            if stats:
                engine_stats.append(stats)
                continue

            # Try parsing LMCache lookup
            lookup = parse_lmcache_lookup(line)
            if lookup:
                request_lookups.append(lookup)
                continue

            # Try parsing LMCache store
            store = parse_lmcache_store(line)
            if store:
                store_events.append(store)
                continue

    print(f"\nParsed:")
    print(f"  - {len(engine_stats):,} engine stats records")
    print(f"  - {len(request_lookups):,} LMCache lookup events")
    print(f"  - {len(store_events):,} LMCache store events")

    return engine_stats, request_lookups, store_events


def analyze_cache_behavior(engine_stats, request_lookups, store_events):
    """Analyze the cache behavior to identify eviction patterns"""
    print("\n" + "="*80)
    print("ANALYSIS: Cache Behavior Over Time")
    print("="*80)

    if not engine_stats:
        print("No engine stats found!")
        return

    # Phase detection: identify when external cache usage increases
    # This indicates KV cache was evicted from GPU and is being loaded from LMCache

    print("\n--- Timeline of Cache Usage ---")
    print(f"{'Timestamp':<20} {'Running':<8} {'Waiting':<8} {'GPU KV%':<10} {'Prefix%':<10} {'External%':<10} {'Status'}")
    print("-" * 100)

    phases = []
    current_phase = None

    for stats in engine_stats:
        # Determine phase
        if stats.external_cache_hit_rate > 5:
            phase = "LOADING_FROM_LMCACHE"
        elif stats.gpu_kv_usage_pct > 95:
            phase = "GPU_NEAR_FULL"
        elif stats.gpu_kv_usage_pct < 50 and stats.prefix_cache_hit_rate > 90:
            phase = "WARM_GPU_CACHE"
        else:
            phase = "NORMAL"

        if phase != current_phase:
            phases.append((stats.timestamp, phase))
            current_phase = phase

        status = ""
        if stats.external_cache_hit_rate > 10:
            status = "⚠️  KV fetch from LMCache!"
        elif stats.gpu_kv_usage_pct > 97:
            status = "📦 GPU cache pressure"

        print(f"{stats.timestamp.strftime('%H:%M:%S'):<20} "
              f"{stats.running_reqs:<8} {stats.waiting_reqs:<8} "
              f"{stats.gpu_kv_usage_pct:<10.1f} {stats.prefix_cache_hit_rate:<10.1f} "
              f"{stats.external_cache_hit_rate:<10.1f} {status}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Concurrency stats
    concurrencies = [s.running_reqs + s.waiting_reqs for s in engine_stats]
    running_only = [s.running_reqs for s in engine_stats]
    print(f"\nConcurrency (Running + Waiting):")
    print(f"  Min: {min(concurrencies)}, Max: {max(concurrencies)}, Avg: {sum(concurrencies)/len(concurrencies):.1f}")
    print(f"  Running only - Min: {min(running_only)}, Max: {max(running_only)}, Avg: {sum(running_only)/len(running_only):.1f}")

    # GPU KV cache usage
    gpu_usages = [s.gpu_kv_usage_pct for s in engine_stats]
    print(f"\nGPU KV Cache Usage (%):")
    print(f"  Min: {min(gpu_usages):.1f}, Max: {max(gpu_usages):.1f}, Avg: {sum(gpu_usages)/len(gpu_usages):.1f}")

    # Prefix cache hit rates
    prefix_hits = [s.prefix_cache_hit_rate for s in engine_stats]
    external_hits = [s.external_cache_hit_rate for s in engine_stats]
    print(f"\nPrefix Cache Hit Rate (GPU) (%):")
    print(f"  Min: {min(prefix_hits):.1f}, Max: {max(prefix_hits):.1f}, Avg: {sum(prefix_hits)/len(prefix_hits):.1f}")
    print(f"\nExternal Cache Hit Rate (LMCache) (%):")
    print(f"  Min: {min(external_hits):.1f}, Max: {max(external_hits):.1f}, Avg: {sum(external_hits)/len(external_hits):.1f}")

    # Identify eviction periods
    print("\n" + "="*80)
    print("EVICTION ANALYSIS")
    print("="*80)

    eviction_periods = []
    in_eviction = False
    eviction_start = None

    for stats in engine_stats:
        # Eviction indicator: GPU near full AND external hit rate increasing
        is_evicting = stats.gpu_kv_usage_pct > 95 and stats.external_cache_hit_rate > 0.5

        if is_evicting and not in_eviction:
            eviction_start = stats.timestamp
            in_eviction = True
        elif not is_evicting and in_eviction:
            eviction_periods.append((eviction_start, stats.timestamp))
            in_eviction = False

    if eviction_periods:
        print(f"\nDetected {len(eviction_periods)} eviction period(s):")
        for start, end in eviction_periods:
            duration = (end - start).total_seconds()
            print(f"  {start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')} (duration: {duration:.0f}s)")
    else:
        print("\nNo clear eviction periods detected.")

    # Analyze performance during high external hit rate (fetching from LMCache)
    print("\n" + "="*80)
    print("PERFORMANCE DURING LMCACHE FETCH")
    print("="*80)

    high_external = [s for s in engine_stats if s.external_cache_hit_rate > 10]
    low_external = [s for s in engine_stats if s.external_cache_hit_rate <= 1]

    if high_external and low_external:
        avg_prompt_tp_high = sum(s.prompt_throughput for s in high_external) / len(high_external)
        avg_prompt_tp_low = sum(s.prompt_throughput for s in low_external) / len(low_external)
        avg_gen_tp_high = sum(s.generation_throughput for s in high_external) / len(high_external)
        avg_gen_tp_low = sum(s.generation_throughput for s in low_external) / len(low_external)

        print(f"\nWhen External Cache Hit Rate > 10% ({len(high_external)} samples):")
        print(f"  Avg Prompt Throughput: {avg_prompt_tp_high:,.1f} tokens/s")
        print(f"  Avg Generation Throughput: {avg_gen_tp_high:,.1f} tokens/s")

        print(f"\nWhen External Cache Hit Rate <= 1% ({len(low_external)} samples):")
        print(f"  Avg Prompt Throughput: {avg_prompt_tp_low:,.1f} tokens/s")
        print(f"  Avg Generation Throughput: {avg_gen_tp_low:,.1f} tokens/s")

        print(f"\nThroughput difference when fetching from LMCache:")
        print(f"  Prompt Throughput: {(avg_prompt_tp_high - avg_prompt_tp_low) / avg_prompt_tp_low * 100:+.1f}%")
        print(f"  Generation Throughput: {(avg_gen_tp_high - avg_gen_tp_low) / avg_gen_tp_low * 100:+.1f}%")

    # LMCache lookup analysis
    if request_lookups:
        print("\n" + "="*80)
        print("LMCACHE REQUEST LOOKUP ANALYSIS")
        print("="*80)

        total_requests = len(request_lookups)
        requests_with_lmcache_hit = [r for r in request_lookups if r.lmcache_hit_tokens > 0]
        requests_needing_load = [r for r in request_lookups if r.need_to_load > 0]

        print(f"\nTotal LMCache lookups: {total_requests:,}")
        print(f"Requests with LMCache hits: {len(requests_with_lmcache_hit):,} ({len(requests_with_lmcache_hit)/total_requests*100:.1f}%)")
        print(f"Requests needing to load from LMCache: {len(requests_needing_load):,} ({len(requests_needing_load)/total_requests*100:.1f}%)")

        if requests_with_lmcache_hit:
            avg_hit_tokens = sum(r.lmcache_hit_tokens for r in requests_with_lmcache_hit) / len(requests_with_lmcache_hit)
            print(f"Average LMCache hit tokens per hit request: {avg_hit_tokens:.1f}")

        if requests_needing_load:
            avg_load_tokens = sum(r.need_to_load for r in requests_needing_load) / len(requests_needing_load)
            print(f"Average tokens to load from LMCache: {avg_load_tokens:.1f}")

    # Store events analysis
    if store_events:
        print("\n" + "="*80)
        print("LMCACHE STORE (OFFLOAD) ANALYSIS")
        print("="*80)

        total_stores = len(store_events)
        total_tokens_stored = sum(s.stored_tokens for s in store_events)
        avg_store_latency = sum(s.cost_ms for s in store_events) / total_stores
        avg_throughput = sum(s.throughput_gbs for s in store_events) / total_stores
        avg_offload_time = sum(s.offload_time_ms for s in store_events) / total_stores

        print(f"\nTotal store events: {total_stores:,}")
        print(f"Total tokens stored to LMCache: {total_tokens_stored:,}")
        print(f"Average store latency: {avg_store_latency:.2f} ms")
        print(f"Average offload time (GPU->CPU): {avg_offload_time:.2f} ms")
        print(f"Average store throughput: {avg_throughput:.2f} GB/s")


def export_to_csv(engine_stats, request_lookups, store_events, output_prefix: str):
    """Export parsed data to CSV files for further analysis"""
    import csv

    # Export engine stats
    if engine_stats:
        with open(f"{output_prefix}_engine_stats.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'prompt_throughput', 'generation_throughput',
                'running_reqs', 'waiting_reqs', 'gpu_kv_usage_pct',
                'prefix_cache_hit_rate', 'external_cache_hit_rate'
            ])
            for s in engine_stats:
                writer.writerow([
                    s.timestamp.isoformat(), s.prompt_throughput, s.generation_throughput,
                    s.running_reqs, s.waiting_reqs, s.gpu_kv_usage_pct,
                    s.prefix_cache_hit_rate, s.external_cache_hit_rate
                ])
        print(f"Exported engine stats to {output_prefix}_engine_stats.csv")

    # Export request lookups
    if request_lookups:
        with open(f"{output_prefix}_request_lookups.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'request_id', 'total_tokens',
                'lmcache_hit_tokens', 'need_to_load'
            ])
            for r in request_lookups:
                writer.writerow([
                    r.timestamp.isoformat(), r.request_id, r.total_tokens,
                    r.lmcache_hit_tokens, r.need_to_load
                ])
        print(f"Exported request lookups to {output_prefix}_request_lookups.csv")

    # Export store events
    if store_events:
        with open(f"{output_prefix}_store_events.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'stored_tokens', 'total_tokens',
                'size_gb', 'cost_ms', 'throughput_gbs', 'offload_time_ms', 'put_time_ms'
            ])
            for s in store_events:
                writer.writerow([
                    s.timestamp.isoformat(), s.stored_tokens, s.total_tokens,
                    s.size_gb, s.cost_ms, s.throughput_gbs, s.offload_time_ms, s.put_time_ms
                ])
        print(f"Exported store events to {output_prefix}_store_events.csv")


def main():
    parser = argparse.ArgumentParser(description='Parse vLLM + LMCache logs')
    parser.add_argument('log_file', help='Path to the vLLM log file')
    parser.add_argument('--export-csv', metavar='PREFIX', help='Export data to CSV files with given prefix')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary, skip timeline')
    args = parser.parse_args()

    # Parse log file
    engine_stats, request_lookups, store_events = parse_log_file(args.log_file)

    # Analyze cache behavior
    analyze_cache_behavior(engine_stats, request_lookups, store_events)

    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(engine_stats, request_lookups, store_events, args.export_csv)


if __name__ == '__main__':
    main()
