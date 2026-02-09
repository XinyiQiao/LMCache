#!/usr/bin/env python3
"""
Benchmark CPU -> GPU data transfer cost for KV cache.

This isolates the data transfer overhead to understand how much of the
~4.5s latency overhead in LMCache fetch is due to CPU->GPU transfer
vs other factors (scheduling, allocation, etc.).

Tests:
1. Transfer time vs data size (varying token counts)
2. Pinned vs non-pinned memory
3. Multiple transfer patterns (contiguous vs chunked)
"""

import torch
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import statistics


@dataclass
class KVCacheConfig:
    """Configuration matching typical LLM KV cache shapes."""
    num_layers: int = 32          # e.g., Llama-7B has 32 layers
    num_kv_heads: int = 8         # GQA heads for KV
    head_dim: int = 128           # head dimension
    dtype: torch.dtype = torch.float16  # typical KV cache dtype


def get_kv_cache_size_gb(num_tokens: int, config: KVCacheConfig) -> float:
    """Calculate KV cache size in GB for given token count."""
    # KV cache shape: [num_layers, 2, num_tokens, num_kv_heads, head_dim]
    # 2 for K and V
    bytes_per_element = 2 if config.dtype == torch.float16 else 4
    total_elements = config.num_layers * 2 * num_tokens * config.num_kv_heads * config.head_dim
    return (total_elements * bytes_per_element) / (1024 ** 3)


def create_kv_cache_cpu(num_tokens: int, config: KVCacheConfig,
                         pinned: bool = False) -> torch.Tensor:
    """Create a KV cache tensor on CPU."""
    shape = (config.num_layers, 2, num_tokens, config.num_kv_heads, config.head_dim)
    if pinned:
        tensor = torch.empty(shape, dtype=config.dtype, pin_memory=True)
    else:
        tensor = torch.empty(shape, dtype=config.dtype)
    # Fill with random data to simulate real KV cache
    tensor.uniform_(-1, 1)
    return tensor


def benchmark_transfer(cpu_tensor: torch.Tensor, device: torch.device,
                       num_iterations: int = 10, warmup: int = 3) -> Tuple[float, float, float]:
    """
    Benchmark CPU -> GPU transfer time.

    Returns: (mean_ms, std_ms, throughput_gb_s)
    """
    size_gb = cpu_tensor.numel() * cpu_tensor.element_size() / (1024 ** 3)
    times = []

    # Warmup
    for _ in range(warmup):
        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize()
        del gpu_tensor
        torch.cuda.empty_cache()

    # Benchmark
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # convert to ms

        del gpu_tensor
        torch.cuda.empty_cache()

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0
    throughput = size_gb / (mean_ms / 1000)  # GB/s

    return mean_ms, std_ms, throughput


def benchmark_async_transfer(cpu_tensor: torch.Tensor, device: torch.device,
                              num_iterations: int = 10, warmup: int = 3) -> Tuple[float, float, float]:
    """
    Benchmark async CPU -> GPU transfer (non_blocking=True).
    """
    size_gb = cpu_tensor.numel() * cpu_tensor.element_size() / (1024 ** 3)
    times = []

    # Warmup
    for _ in range(warmup):
        gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        torch.cuda.synchronize()
        del gpu_tensor
        torch.cuda.empty_cache()

    # Benchmark
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        torch.cuda.synchronize()  # Wait for transfer to complete

        end = time.perf_counter()
        times.append((end - start) * 1000)

        del gpu_tensor
        torch.cuda.empty_cache()

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0
    throughput = size_gb / (mean_ms / 1000)

    return mean_ms, std_ms, throughput


def benchmark_chunked_transfer(cpu_tensor: torch.Tensor, device: torch.device,
                                chunk_size_tokens: int, config: KVCacheConfig,
                                num_iterations: int = 10, warmup: int = 3) -> Tuple[float, float]:
    """
    Benchmark chunked transfer (simulating LMCache's chunk-based approach).
    """
    num_tokens = cpu_tensor.shape[2]
    num_chunks = (num_tokens + chunk_size_tokens - 1) // chunk_size_tokens
    times = []

    # Warmup
    for _ in range(warmup):
        gpu_chunks = []
        for i in range(num_chunks):
            start_tok = i * chunk_size_tokens
            end_tok = min((i + 1) * chunk_size_tokens, num_tokens)
            chunk = cpu_tensor[:, :, start_tok:end_tok, :, :]
            gpu_chunks.append(chunk.to(device, non_blocking=False))
        torch.cuda.synchronize()
        del gpu_chunks
        torch.cuda.empty_cache()

    # Benchmark
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        gpu_chunks = []
        for i in range(num_chunks):
            start_tok = i * chunk_size_tokens
            end_tok = min((i + 1) * chunk_size_tokens, num_tokens)
            chunk = cpu_tensor[:, :, start_tok:end_tok, :, :]
            gpu_chunks.append(chunk.to(device, non_blocking=False))
        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

        del gpu_chunks
        torch.cuda.empty_cache()

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0

    return mean_ms, std_ms


def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU->GPU transfer for KV cache')
    parser.add_argument('--num-layers', type=int, default=32, help='Number of transformer layers')
    parser.add_argument('--num-kv-heads', type=int, default=8, help='Number of KV heads')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--iterations', type=int, default=10, help='Number of benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device(args.device)
    config = KVCacheConfig(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )

    print("=" * 80)
    print("CPU -> GPU Transfer Benchmark for KV Cache")
    print("=" * 80)
    print(f"\nConfig: {config.num_layers} layers, {config.num_kv_heads} KV heads, "
          f"{config.head_dim} head_dim, {config.dtype}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Iterations per test: {args.iterations}")

    # Token counts to test (matching the analysis buckets)
    token_counts = [256, 1000, 2000, 5000, 10000, 20000, 30000]

    # =========================================================================
    # Test 1: Transfer time vs token count (pinned memory)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Transfer Time vs Token Count (Pinned Memory)")
    print("=" * 80)
    print(f"\n{'Tokens':<10} {'Size (GB)':<12} {'Time (ms)':<15} {'Std (ms)':<12} {'Throughput':<15}")
    print("-" * 65)

    pinned_results = []
    for num_tokens in token_counts:
        size_gb = get_kv_cache_size_gb(num_tokens, config)
        cpu_tensor = create_kv_cache_cpu(num_tokens, config, pinned=True)
        mean_ms, std_ms, throughput = benchmark_transfer(cpu_tensor, device, args.iterations)
        pinned_results.append((num_tokens, size_gb, mean_ms, std_ms, throughput))
        print(f"{num_tokens:<10} {size_gb:<12.4f} {mean_ms:<15.3f} {std_ms:<12.3f} {throughput:<15.2f} GB/s")
        del cpu_tensor

    # =========================================================================
    # Test 2: Transfer time vs token count (non-pinned memory)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Transfer Time vs Token Count (Non-Pinned Memory)")
    print("=" * 80)
    print(f"\n{'Tokens':<10} {'Size (GB)':<12} {'Time (ms)':<15} {'Std (ms)':<12} {'Throughput':<15}")
    print("-" * 65)

    nonpinned_results = []
    for num_tokens in token_counts:
        size_gb = get_kv_cache_size_gb(num_tokens, config)
        cpu_tensor = create_kv_cache_cpu(num_tokens, config, pinned=False)
        mean_ms, std_ms, throughput = benchmark_transfer(cpu_tensor, device, args.iterations)
        nonpinned_results.append((num_tokens, size_gb, mean_ms, std_ms, throughput))
        print(f"{num_tokens:<10} {size_gb:<12.4f} {mean_ms:<15.3f} {std_ms:<12.3f} {throughput:<15.2f} GB/s")
        del cpu_tensor

    # =========================================================================
    # Test 3: Async transfer (pinned memory)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Async Transfer (Pinned Memory, non_blocking=True)")
    print("=" * 80)
    print(f"\n{'Tokens':<10} {'Size (GB)':<12} {'Time (ms)':<15} {'Std (ms)':<12} {'Throughput':<15}")
    print("-" * 65)

    for num_tokens in token_counts:
        size_gb = get_kv_cache_size_gb(num_tokens, config)
        cpu_tensor = create_kv_cache_cpu(num_tokens, config, pinned=True)
        mean_ms, std_ms, throughput = benchmark_async_transfer(cpu_tensor, device, args.iterations)
        print(f"{num_tokens:<10} {size_gb:<12.4f} {mean_ms:<15.3f} {std_ms:<12.3f} {throughput:<15.2f} GB/s")
        del cpu_tensor

    # =========================================================================
    # Test 4: Chunked transfer (simulating LMCache chunks, typically 256 tokens)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Chunked Transfer (256-token chunks, Pinned Memory)")
    print("=" * 80)
    print(f"\n{'Tokens':<10} {'Chunks':<10} {'Time (ms)':<15} {'Std (ms)':<12} {'Overhead vs Contiguous':<20}")
    print("-" * 70)

    chunk_size = 256
    for i, num_tokens in enumerate(token_counts):
        if num_tokens < chunk_size:
            continue
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        cpu_tensor = create_kv_cache_cpu(num_tokens, config, pinned=True)
        mean_ms, std_ms = benchmark_chunked_transfer(cpu_tensor, device, chunk_size, config, args.iterations)

        # Compare to contiguous transfer
        contiguous_ms = pinned_results[i][2]
        overhead = ((mean_ms / contiguous_ms) - 1) * 100

        print(f"{num_tokens:<10} {num_chunks:<10} {mean_ms:<15.3f} {std_ms:<12.3f} {overhead:+.1f}%")
        del cpu_tensor

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate transfer time for typical LMCache fetch scenario
    # From the analysis: median tokens_to_load ranges from 256 to 20k
    print("\nTypical LMCache fetch transfer times (pinned memory):")
    for num_tokens, size_gb, mean_ms, std_ms, throughput in pinned_results:
        print(f"  {num_tokens:>6} tokens ({size_gb:.3f} GB): {mean_ms:.1f} ms")

    print(f"\nObserved latency overhead in benchmark: ~4500 ms (4.5s)")
    print(f"Max transfer time for 30k tokens:       {pinned_results[-1][2]:.1f} ms")
    print(f"\n>>> Transfer cost accounts for {pinned_results[-1][2] / 4500 * 100:.1f}% of the observed overhead")
    print(">>> The remaining overhead is likely from scheduling, allocation, or other factors")


if __name__ == "__main__":
    main()
