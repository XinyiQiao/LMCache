#!/usr/bin/env python3
"""
Join client-side request logs (minisweagent.log) with server-side vLLM/LMCache logs (vllm2.log)
on request ID to analyze the impact of KV cache eviction on request latency.

Metrics collected per request:
- Request latency (client-side)
- Prompt tokens
- LMCache hit tokens
- Tokens that need to be loaded from LMCache
- Whether KV fetch from LMCache was needed
- Prefix cache hit rate (server state at request time)
- External cache hit rate (server state at request time)
- GPU KV cache usage %
- Concurrent requests (running + waiting)
- Prompt/generation throughput

Goal: Understand when prefix cache hits but KV cache was evicted from GPU,
      does fetching from LMCache increase request latency?
"""

import re
import csv
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics


@dataclass
class ClientRequest:
    """Request info from client-side log"""
    timestamp: datetime
    request_id: str
    latency_sec: float
    model: str = ""


@dataclass
class LMCacheLookup:
    """LMCache lookup info from server-side log"""
    timestamp: datetime
    request_id: str
    total_tokens: int
    lmcache_hit_tokens: int
    need_to_load: int


@dataclass
class LMCacheStore:
    """LMCache store event from server-side log"""
    timestamp: datetime
    request_id: str
    stored_tokens: int
    total_tokens: int
    size_gb: float
    cost_ms: float
    throughput_gbs: float


@dataclass
class EngineStats:
    """Engine statistics from server-side log (every ~10 seconds)"""
    timestamp: datetime
    prompt_throughput: float
    generation_throughput: float
    running_reqs: int
    waiting_reqs: int
    gpu_kv_usage_pct: float
    prefix_cache_hit_rate: float
    external_cache_hit_rate: float


@dataclass
class JoinedRequest:
    """Joined request data from client and server"""
    request_id: str
    # Client metrics
    client_timestamp: Optional[datetime] = None
    latency_sec: Optional[float] = None
    # Server LMCache metrics
    server_timestamp: Optional[datetime] = None
    prompt_tokens: Optional[int] = None
    lmcache_hit_tokens: Optional[int] = None
    tokens_to_load_from_lmcache: Optional[int] = None
    kv_fetch_needed: Optional[bool] = None
    # Server engine state at request time
    prefix_cache_hit_rate: Optional[float] = None
    external_cache_hit_rate: Optional[float] = None
    gpu_kv_usage_pct: Optional[float] = None
    concurrent_requests: Optional[int] = None
    prompt_throughput: Optional[float] = None
    generation_throughput: Optional[float] = None
    # Store metrics (if available)
    store_cost_ms: Optional[float] = None
    store_throughput_gbs: Optional[float] = None


def parse_timestamp(timestamp_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """Parse timestamp string to datetime"""
    try:
        # Handle milliseconds
        if "," in timestamp_str:
            main_part, ms = timestamp_str.rsplit(",", 1)
            dt = datetime.strptime(main_part, fmt)
            return dt.replace(microsecond=int(ms) * 1000)
        return datetime.strptime(timestamp_str, fmt)
    except ValueError:
        return None


def parse_client_log(log_path: str) -> Dict[str, ClientRequest]:
    """
    Parse client-side log for request IDs and latencies.

    Expected format:
    2026-02-02 20:16:56,254 - minisweagent.litellm_model - INFO - Request time for hosted_vllm/model: 0.411s, id: chatcmpl-xxx

    Adjust the regex pattern based on your actual log format.
    """
    requests = {}

    # Pattern for request with ID - adjust based on your actual log format
    # Actual minisweagent format: Request time for chatcmpl-xxx: Xs
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Request time for (chatcmpl-[^\s:]+): ([\d.]+)s'

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                timestamp_str, request_id, latency_str = match.groups()
                timestamp = parse_timestamp(timestamp_str)
                latency = float(latency_str)
                model = ""

                if timestamp and request_id:
                    # Clean up request_id (remove quotes, trailing punctuation)
                    request_id = request_id.strip('"\'')

                    requests[request_id] = ClientRequest(
                        timestamp=timestamp,
                        request_id=request_id,
                        latency_sec=latency,
                        model=model
                    )

    return requests


def parse_vllm_log(log_path: str) -> Tuple[Dict[str, LMCacheLookup], Dict[str, LMCacheStore], List[EngineStats]]:
    """
    Parse vLLM/LMCache server log for:
    - LMCache lookup events (per request)
    - LMCache store events (per request)
    - Engine stats (periodic)
    """
    lookups = {}
    stores = {}
    engine_stats = []

    # Patterns
    lookup_pattern = r'\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\].*Reqid: ([^,]+), Total tokens (\d+), LMCache hit tokens: (-?\d+), need to load: (-?\d+)'

    store_pattern = r'\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\].*Storing KV cache for (\d+) out of (\d+) tokens.*for request ([^\s]+)'

    store_complete_pattern = r'\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\].*Stored (\d+) out of total (\d+) tokens\. size: ([\d.]+) GB, cost ([\d.]+) ms, throughput: ([\d.]+) GB/s'

    engine_pattern = r'INFO (\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2}).*Engine 000:.*Avg prompt throughput: ([\d.]+).*Avg generation throughput: ([\d.]+).*Running: (\d+) reqs.*Waiting: (\d+) reqs.*GPU KV cache usage: ([\d.]+)%.*Prefix cache hit rate: ([\d.]+)%.*External prefix cache hit rate: ([\d.]+)%'

    # Track last store request for matching with completion
    last_store_request = None

    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            # Try LMCache lookup
            match = re.search(lookup_pattern, line)
            if match:
                date_str, time_str, ms, req_id, total, hit, need_load = match.groups()
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                timestamp = timestamp.replace(microsecond=int(ms) * 1000)

                lookups[req_id] = LMCacheLookup(
                    timestamp=timestamp,
                    request_id=req_id,
                    total_tokens=int(total),
                    lmcache_hit_tokens=int(hit),
                    need_to_load=int(need_load)
                )
                continue

            # Try store start (to get request ID)
            match = re.search(store_pattern, line)
            if match:
                date_str, time_str, ms, stored, total, req_id = match.groups()
                last_store_request = req_id.strip()
                continue

            # Try store complete
            match = re.search(store_complete_pattern, line)
            if match and last_store_request:
                date_str, time_str, ms, stored, total, size, cost, throughput = match.groups()
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                timestamp = timestamp.replace(microsecond=int(ms) * 1000)

                stores[last_store_request] = LMCacheStore(
                    timestamp=timestamp,
                    request_id=last_store_request,
                    stored_tokens=int(stored),
                    total_tokens=int(total),
                    size_gb=float(size),
                    cost_ms=float(cost),
                    throughput_gbs=float(throughput)
                )
                last_store_request = None
                continue

            # Try engine stats
            match = re.search(engine_pattern, line)
            if match:
                month, day, time_str, prompt_tp, gen_tp, running, waiting, gpu_usage, prefix_hit, external_hit = match.groups()
                # Assume year 2026 based on logs
                timestamp = datetime.strptime(f"2026-{month}-{day} {time_str}", "%Y-%m-%d %H:%M:%S")

                engine_stats.append(EngineStats(
                    timestamp=timestamp,
                    prompt_throughput=float(prompt_tp),
                    generation_throughput=float(gen_tp),
                    running_reqs=int(running),
                    waiting_reqs=int(waiting),
                    gpu_kv_usage_pct=float(gpu_usage),
                    prefix_cache_hit_rate=float(prefix_hit),
                    external_cache_hit_rate=float(external_hit)
                ))

    # Sort engine stats by timestamp
    engine_stats.sort(key=lambda x: x.timestamp)

    return lookups, stores, engine_stats


def find_nearest_engine_stats(timestamp: datetime, engine_stats: List[EngineStats],
                               max_delta_sec: float = 30.0) -> Optional[EngineStats]:
    """Find the engine stats entry nearest to the given timestamp"""
    if not engine_stats:
        return None

    best = None
    best_delta = float('inf')

    for stats in engine_stats:
        delta = abs((stats.timestamp - timestamp).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = stats

    if best_delta <= max_delta_sec:
        return best
    return None


def join_requests(client_requests: Dict[str, ClientRequest],
                  lookups: Dict[str, LMCacheLookup],
                  stores: Dict[str, LMCacheStore],
                  engine_stats: List[EngineStats]) -> List[JoinedRequest]:
    """Join client requests with server-side data using prefix matching.

    Server request IDs have a suffix appended (e.g., chatcmpl-xxx becomes chatcmpl-xxx-suffix).
    We match by checking if the server ID starts with the client ID.
    """
    joined = []

    # Build a mapping from client ID prefix to server lookup
    # Server IDs are like: chatcmpl-9d56f6f05443d58a-ae4680b6
    # Client IDs are like: chatcmpl-9d56f6f05443d58a
    client_to_server_lookup = {}
    client_to_server_store = {}

    for server_id, lookup in lookups.items():
        # Try to find matching client ID (server ID starts with client ID)
        for client_id in client_requests.keys():
            if server_id.startswith(client_id):
                client_to_server_lookup[client_id] = lookup
                break

    for server_id, store in stores.items():
        for client_id in client_requests.keys():
            if server_id.startswith(client_id):
                client_to_server_store[client_id] = store
                break

    # Join based on client requests
    for req_id, cr in client_requests.items():
        jr = JoinedRequest(request_id=req_id)

        # Client data
        jr.client_timestamp = cr.timestamp
        jr.latency_sec = cr.latency_sec

        # Server lookup data (matched by prefix)
        if req_id in client_to_server_lookup:
            lookup = client_to_server_lookup[req_id]
            jr.server_timestamp = lookup.timestamp
            jr.prompt_tokens = lookup.total_tokens
            jr.lmcache_hit_tokens = lookup.lmcache_hit_tokens
            jr.tokens_to_load_from_lmcache = lookup.need_to_load if lookup.need_to_load > 0 else 0
            # KV fetch needed if there are hit tokens AND need_to_load > 0
            jr.kv_fetch_needed = lookup.lmcache_hit_tokens > 0 and lookup.need_to_load > 0

        # Server store data (matched by prefix)
        if req_id in client_to_server_store:
            store = client_to_server_store[req_id]
            jr.store_cost_ms = store.cost_ms
            jr.store_throughput_gbs = store.throughput_gbs

        # Find nearest engine stats
        timestamp = jr.server_timestamp or jr.client_timestamp
        if timestamp:
            stats = find_nearest_engine_stats(timestamp, engine_stats)
            if stats:
                jr.prefix_cache_hit_rate = stats.prefix_cache_hit_rate
                jr.external_cache_hit_rate = stats.external_cache_hit_rate
                jr.gpu_kv_usage_pct = stats.gpu_kv_usage_pct
                jr.concurrent_requests = stats.running_reqs + stats.waiting_reqs
                jr.prompt_throughput = stats.prompt_throughput
                jr.generation_throughput = stats.generation_throughput

        joined.append(jr)

    return joined


def export_to_csv(joined: List[JoinedRequest], output_path: str):
    """Export joined data to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'request_id',
            'client_timestamp',
            'server_timestamp',
            'latency_sec',
            'prompt_tokens',
            'lmcache_hit_tokens',
            'tokens_to_load_from_lmcache',
            'kv_fetch_needed',
            'prefix_cache_hit_rate',
            'external_cache_hit_rate',
            'gpu_kv_usage_pct',
            'concurrent_requests',
            'prompt_throughput',
            'generation_throughput',
            'store_cost_ms',
            'store_throughput_gbs'
        ])

        for jr in joined:
            writer.writerow([
                jr.request_id,
                jr.client_timestamp.isoformat() if jr.client_timestamp else '',
                jr.server_timestamp.isoformat() if jr.server_timestamp else '',
                jr.latency_sec if jr.latency_sec is not None else '',
                jr.prompt_tokens if jr.prompt_tokens is not None else '',
                jr.lmcache_hit_tokens if jr.lmcache_hit_tokens is not None else '',
                jr.tokens_to_load_from_lmcache if jr.tokens_to_load_from_lmcache is not None else '',
                jr.kv_fetch_needed if jr.kv_fetch_needed is not None else '',
                jr.prefix_cache_hit_rate if jr.prefix_cache_hit_rate is not None else '',
                jr.external_cache_hit_rate if jr.external_cache_hit_rate is not None else '',
                jr.gpu_kv_usage_pct if jr.gpu_kv_usage_pct is not None else '',
                jr.concurrent_requests if jr.concurrent_requests is not None else '',
                jr.prompt_throughput if jr.prompt_throughput is not None else '',
                jr.generation_throughput if jr.generation_throughput is not None else '',
                jr.store_cost_ms if jr.store_cost_ms is not None else '',
                jr.store_throughput_gbs if jr.store_throughput_gbs is not None else ''
            ])

    print(f"Exported {len(joined)} joined requests to {output_path}")


def analyze_latency_by_cache_state(joined: List[JoinedRequest]):
    """Analyze request latency grouped by cache state"""
    print("\n" + "="*80)
    print("ANALYSIS: Request Latency vs KV Cache State")
    print("="*80)

    # Filter to requests with both latency and cache state info
    valid = [jr for jr in joined if jr.latency_sec is not None and jr.external_cache_hit_rate is not None]

    if not valid:
        print("No valid joined requests found with both latency and cache state data.")
        return

    print(f"\nTotal joined requests with latency data: {len(valid)}")

    # Group by external cache hit rate
    def get_cache_state(ext_rate):
        if ext_rate > 50:
            return "HIGH_LMCACHE_FETCH (>50%)"
        elif ext_rate > 10:
            return "MODERATE_LMCACHE_FETCH (10-50%)"
        elif ext_rate > 1:
            return "LOW_LMCACHE_FETCH (1-10%)"
        else:
            return "NORMAL_GPU_CACHE (<1%)"

    by_state = defaultdict(list)
    for jr in valid:
        state = get_cache_state(jr.external_cache_hit_rate)
        by_state[state].append(jr.latency_sec)

    print("\n--- Request Latency by Cache State ---\n")
    print(f"{'Cache State':<35} {'Count':<10} {'Avg(s)':<12} {'Median(s)':<12} {'P95(s)':<12}")
    print("-" * 80)

    state_order = [
        "NORMAL_GPU_CACHE (<1%)",
        "LOW_LMCACHE_FETCH (1-10%)",
        "MODERATE_LMCACHE_FETCH (10-50%)",
        "HIGH_LMCACHE_FETCH (>50%)"
    ]

    results = {}
    for state in state_order:
        if state in by_state:
            latencies = sorted(by_state[state])
            p95_idx = int(len(latencies) * 0.95)
            p95 = latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1]

            results[state] = {
                'count': len(latencies),
                'avg': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': p95
            }
            print(f"{state:<35} {len(latencies):<10} {statistics.mean(latencies):<12.3f} "
                  f"{statistics.median(latencies):<12.3f} {p95:<12.3f}")

    # Impact analysis
    if "NORMAL_GPU_CACHE (<1%)" in results and "HIGH_LMCACHE_FETCH (>50%)" in results:
        normal = results["NORMAL_GPU_CACHE (<1%)"]
        high = results["HIGH_LMCACHE_FETCH (>50%)"]

        print("\n" + "="*80)
        print("KEY FINDING: Impact of LMCache Fetch on Request Latency")
        print("="*80)
        print(f"\nWhen GPU serving (external hit <1%):")
        print(f"  Avg latency: {normal['avg']:.3f}s, Median: {normal['median']:.3f}s")
        print(f"\nWhen fetching from LMCache (external hit >50%):")
        print(f"  Avg latency: {high['avg']:.3f}s, Median: {high['median']:.3f}s")
        print(f"\n>>> Latency increase: {((high['avg']/normal['avg'])-1)*100:+.1f}%")
        print(f">>> Requests are {high['avg']/normal['avg']:.2f}x slower when fetching from LMCache")


def analyze_by_kv_fetch_needed(joined: List[JoinedRequest]):
    """Analyze latency based on whether KV fetch from LMCache was needed"""
    print("\n" + "="*80)
    print("ANALYSIS: Request Latency vs KV Fetch Needed")
    print("="*80)

    fetch_needed = [jr for jr in joined if jr.latency_sec and jr.kv_fetch_needed == True]
    no_fetch = [jr for jr in joined if jr.latency_sec and jr.kv_fetch_needed == False]

    if fetch_needed and no_fetch:
        fetch_latencies = [jr.latency_sec for jr in fetch_needed]
        no_fetch_latencies = [jr.latency_sec for jr in no_fetch]

        print(f"\nRequests WITH KV fetch from LMCache: {len(fetch_needed)}")
        print(f"  Avg latency: {statistics.mean(fetch_latencies):.3f}s")
        print(f"  Median: {statistics.median(fetch_latencies):.3f}s")

        print(f"\nRequests WITHOUT KV fetch (GPU serving): {len(no_fetch)}")
        print(f"  Avg latency: {statistics.mean(no_fetch_latencies):.3f}s")
        print(f"  Median: {statistics.median(no_fetch_latencies):.3f}s")

        if statistics.mean(no_fetch_latencies) > 0:
            impact = (statistics.mean(fetch_latencies) / statistics.mean(no_fetch_latencies) - 1) * 100
            print(f"\n>>> KV fetch impact: {impact:+.1f}% latency increase")


def analyze_latency_vs_tokens_to_load(joined: List[JoinedRequest]):
    """Analyze correlation between tokens to load and latency"""
    print("\n" + "="*80)
    print("ANALYSIS: Request Latency vs Tokens to Load from LMCache")
    print("="*80)

    valid = [jr for jr in joined
             if jr.latency_sec is not None
             and jr.tokens_to_load_from_lmcache is not None
             and jr.tokens_to_load_from_lmcache > 0]

    if not valid:
        print("No requests with tokens to load from LMCache found.")
        return

    # Bucket by tokens to load
    buckets = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 50000), (50000, float('inf'))]

    print(f"\n{'Tokens to Load Range':<25} {'Count':<10} {'Avg Latency(s)':<15}")
    print("-" * 50)

    for low, high in buckets:
        in_bucket = [jr.latency_sec for jr in valid if low <= jr.tokens_to_load_from_lmcache < high]
        if in_bucket:
            high_str = str(high) if high != float('inf') else "+"
            print(f"{low}-{high_str:<20} {len(in_bucket):<10} {statistics.mean(in_bucket):<15.3f}")


def main():
    parser = argparse.ArgumentParser(description='Join client and server request logs')
    parser.add_argument('--client-log', default='minisweagent.log',
                        help='Path to client-side log file')
    parser.add_argument('--server-log', default='vllm.log',
                        help='Path to vLLM server log file')
    parser.add_argument('--output', default='joined_request_analysis.csv',
                        help='Output CSV file path')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis after joining')
    args = parser.parse_args()

    print("="*80)
    print("Request Log Join: Client (minisweagent) <-> Server (vLLM/LMCache)")
    print("="*80)

    # Parse client log
    print(f"\nParsing client log: {args.client_log}")
    client_requests = parse_client_log(args.client_log)
    print(f"  Found {len(client_requests)} requests with IDs and latencies")

    # Parse server log
    print(f"\nParsing server log: {args.server_log}")
    lookups, stores, engine_stats = parse_vllm_log(args.server_log)
    print(f"  Found {len(lookups)} LMCache lookup events")
    print(f"  Found {len(stores)} LMCache store events")
    print(f"  Found {len(engine_stats)} engine stats entries")

    # Join
    print("\nJoining on request ID...")
    joined = join_requests(client_requests, lookups, stores, engine_stats)

    # Stats on join
    matched = [jr for jr in joined if jr.latency_sec is not None and jr.prompt_tokens is not None]
    client_only = [jr for jr in joined if jr.latency_sec is not None and jr.prompt_tokens is None]
    server_only = [jr for jr in joined if jr.latency_sec is None and jr.prompt_tokens is not None]

    print(f"\nJoin Results:")
    print(f"  Matched (both sides): {len(matched)}")
    print(f"  Client only: {len(client_only)}")
    print(f"  Server only: {len(server_only)}")

    # Export
    export_to_csv(joined, args.output)

    # Analyze if requested
    if args.analyze and matched:
        analyze_latency_by_cache_state(joined)
        analyze_by_kv_fetch_needed(joined)
        analyze_latency_vs_tokens_to_load(joined)


if __name__ == '__main__':
    main()
