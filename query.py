#!/usr/bin/env python3
"""
Query vLLM Prometheus metrics endpoint to collect workload and cache metrics.

Usage:
    python query.py --host localhost --port 8000
    python query.py --host localhost --port 8000 --continuous --interval 5
"""

import argparse
import asyncio
import logging
import sys
from typing import Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def get_workload(host: str, port: int) -> dict[str, Any]:
    """Get current workload metrics from vLLM server via Prometheus /metrics endpoint.

    Args:
        host: Server hostname or IP address
        port: Server port number

    Returns:
        dict[str, Any]: Dictionary containing:
            - num_requests_running: Number of requests currently being processed
            - num_requests_waiting: Number of requests waiting in queue
            - kv_cache_usage: KV cache utilization percentage (0.0-1.0)
            - inter_token_latency_sum: Sum of inter-token latencies
            - inter_token_latency_count: Count of inter-token latency samples
            - prefix_cache_hit_rate: Prefix cache hit rate
            - external_cache_hit_rate: External (LMCache) hit rate
    """
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    try:
        # Handle IPv6 addresses by wrapping in brackets if needed
        formatted_host = host
        if ':' in host and not host.startswith('['):
            formatted_host = f"[{host}]"
        metrics_url = f"http://{formatted_host}:{port}/metrics"

        logger.info(f"Fetching metrics from {metrics_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    return {"error": f"Failed to fetch metrics: HTTP {response.status}"}
                metrics_text = await response.text()

        if not metrics_text.strip():
            logger.warning("Prometheus /metrics endpoint returned empty response")
            return {"error": "Empty metrics response"}

        # Parse Prometheus format metrics
        result: dict[str, Any] = {}

        # Metric name variants across vLLM versions
        running_names = {"vllm:num_requests_running", "vllm_num_requests_running"}
        waiting_names = {"vllm:num_requests_waiting", "vllm_num_requests_waiting"}
        kv_cache_names = {
            "vllm:kv_cache_usage_perc",
            "vllm_kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
        }
        # Inter-token latency histogram (sum/count for computing average)
        itl_sum_names = {
            "vllm:inter_token_latency_seconds_sum",
            "vllm_inter_token_latency_seconds_sum",
        }
        itl_count_names = {
            "vllm:inter_token_latency_seconds_count",
            "vllm_inter_token_latency_seconds_count",
        }
        # Prefix cache and external cache hit rates
        prefix_cache_names = {
            "vllm:prefix_cache_hit_rate",
            "vllm_prefix_cache_hit_rate",
        }
        external_cache_names = {
            "vllm:external_cache_hit_rate",
            "vllm_external_cache_hit_rate",
        }

        # Collect all metric names for debugging
        found_metric_names = set()

        for line in metrics_text.split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Parse metric lines: metric_name{labels} value or metric_name value
            if " " in line:
                metric_part, value_str = line.rsplit(" ", 1)
                try:
                    value = float(value_str)
                except ValueError:
                    continue

                # Extract metric name (without labels)
                metric_name = metric_part.split("{")[0]
                found_metric_names.add(metric_name)

                # Extract the metrics we care about
                if metric_name in running_names:
                    result["num_requests_running"] = int(value)
                elif metric_name in waiting_names:
                    result["num_requests_waiting"] = int(value)
                elif metric_name in kv_cache_names:
                    result["kv_cache_usage"] = value
                elif metric_name in itl_sum_names:
                    result["inter_token_latency_sum"] = value
                elif metric_name in itl_count_names:
                    result["inter_token_latency_count"] = int(value)
                elif metric_name in prefix_cache_names:
                    result["prefix_cache_hit_rate"] = value
                elif metric_name in external_cache_names:
                    result["external_cache_hit_rate"] = value

        # Compute average inter-token latency if both sum and count are available
        if "inter_token_latency_sum" in result and "inter_token_latency_count" in result:
            if result["inter_token_latency_count"] > 0:
                result["avg_inter_token_latency_ms"] = (
                    result["inter_token_latency_sum"] / result["inter_token_latency_count"]
                ) * 1000  # Convert to milliseconds

        # Warn if expected metrics not found
        if not result:
            vllm_metrics = [m for m in found_metric_names if "vllm" in m.lower()]
            logger.warning(
                f"No workload metrics found in Prometheus response. "
                f"Found {len(found_metric_names)} metrics total, "
                f"{len(vllm_metrics)} vllm-related: {vllm_metrics[:10]}"
            )
            result["warning"] = "No workload metrics found"
            result["available_vllm_metrics"] = vllm_metrics[:20]

        return result

    except asyncio.TimeoutError:
        logger.error("Timeout connecting to metrics endpoint")
        return {"error": "Connection timeout"}
    except Exception as e:
        logger.error(f"Error getting workload metrics: {e}")
        return {"error": str(e)}


def print_metrics(metrics: dict[str, Any], iteration: int = None):
    """Pretty print metrics."""
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    header = f"--- Metrics (iteration {iteration}) ---" if iteration else "--- vLLM Metrics ---"
    print(f"\n{header}")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"  {key}: {value[:5]}...")  # Truncate long lists
        else:
            print(f"  {key}: {value}")


async def collect_metrics_loop(host: str, port: int, interval: float = 10.0, count: int = None):
    """Continuously collect metrics at specified interval.

    Args:
        host: Server hostname
        port: Server port
        interval: Seconds between collections
        count: Number of collections (None for infinite)
    """
    iteration = 0
    while count is None or iteration < count:
        metrics = await get_workload(host, port)
        print_metrics(metrics, iteration + 1)

        iteration += 1
        if count is None or iteration < count:
            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Query vLLM Prometheus metrics endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single query
    python query.py --host localhost --port 8000

    # Continuous monitoring every 5 seconds
    python query.py --host localhost --port 8000 --interval 5 --continuous

    # Collect 10 samples
    python query.py --host localhost --port 8000 --interval 5 --count 10
        """
    )
    parser.add_argument("--host", default="localhost", help="vLLM server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000)")
    parser.add_argument("--interval", type=float, default=10.0, help="Collection interval in seconds (default: 10)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--count", type=int, help="Number of samples to collect")

    args = parser.parse_args()

    if args.continuous:
        print(f"Collecting metrics from {args.host}:{args.port} every {args.interval}s (Ctrl+C to stop)")
        try:
            asyncio.run(collect_metrics_loop(args.host, args.port, args.interval))
        except KeyboardInterrupt:
            print("\nStopped.")
    elif args.count:
        print(f"Collecting {args.count} samples from {args.host}:{args.port}")
        asyncio.run(collect_metrics_loop(args.host, args.port, args.interval, args.count))
    else:
        # Single query
        metrics = asyncio.run(get_workload(args.host, args.port))
        print_metrics(metrics)
        if "error" in metrics:
            sys.exit(1)


if __name__ == "__main__":
    main()
