"""Micro-benchmark for GEMM_Mat_Trf CPU materialization overhead."""

from __future__ import absolute_import, print_function

import time
import numpy as np

try:
    from vta.runtime_gemm_mat_trf import vta_gemm_mat_trf_cpu
    HAS_VTA = True
except ImportError:
    HAS_VTA = False


def benchmark_cpu_transform(batch_size, in_dim, out_dim, num_trials=10, dtype=np.int8):
    """Benchmark CPU-side matrix transform."""
    
    # Allocate tensors
    input_blocks = np.random.randint(-128, 127, (batch_size, in_dim), dtype=dtype)
    transform_matrix = np.random.randint(-128, 127, (out_dim, in_dim), dtype=dtype)
    
    # Warm-up
    for _ in range(2):
        _ = vta_gemm_mat_trf_cpu(input_blocks, transform_matrix)
    
    # Benchmark
    times = []
    for _ in range(num_trials):
        t0 = time.time()
        output = vta_gemm_mat_trf_cpu(input_blocks, transform_matrix)
        times.append((time.time() - t0) * 1000.0)  # ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Compute ops and throughput
    ops = batch_size * in_dim * out_dim * 2  # multiply + accumulate
    throughput = ops / (avg_time / 1000.0) / 1e9  # GOPS
    
    return {
        'avg_ms': avg_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'gops': throughput,
        'output_shape': output.shape,
    }


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    if not HAS_VTA:
        print("VTA not available; benchmark skipped")
        return
    
    print("=" * 72)
    print("GEMM_Mat_Trf CPU Materialization Benchmark")
    print("=" * 72)
    print()
    
    # Test cases: (batch, in_dim, out_dim)
    test_cases = [
        # Small transforms (typical for dense subgraphs)
        (1, 16, 16, "Small (1x16x16)"),
        (1, 64, 64, "Medium (1x64x64)"),
        (1, 128, 128, "Large (1x128x128)"),
        (4, 16, 16, "Batch-4 Small (4x16x16)"),
        (4, 64, 64, "Batch-4 Medium (4x64x64)"),
        (8, 32, 32, "Batch-8 Medium (8x32x32)"),
    ]
    
    results = {}
    for batch, in_dim, out_dim, label in test_cases:
        print(f"Benchmarking {label}...")
        try:
            result = benchmark_cpu_transform(batch, in_dim, out_dim, num_trials=15)
            results[label] = result
            print(f"  Time: {result['avg_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print(f"  Throughput: {result['gops']:.2f} GOPS")
            print(f"  Output: {result['output_shape']}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()
    
    print("=" * 72)
    print("Benchmark Summary")
    print("=" * 72)
    print()
    print(f"{'Config':<30} {'Time (ms)':<15} {'GOPS':<15}")
    print("-" * 60)
    for label in [k for k, _, _, _ in test_cases]:
        if label in results:
            r = results[label]
            print(f"{label:<30} {r['avg_ms']:>8.3f}      {r['gops']:>8.2f}")
    print()


if __name__ == "__main__":
    run_benchmark_suite()

