#!/usr/bin/env python3
"""Benchmark script for comparing different computational backends.

This script benchmarks the performance of scipy and taichi backends
for black hole calculations.

Usage:
    python benchmark.py --engine=scipy
    python benchmark.py --engine=taichi
    python benchmark.py --compare  # Compare all backends
"""

import argparse
import time
import sys
from typing import Dict, Tuple
import numpy as np

# Import backend factory
try:
    from luminet.backends import get_backend, list_available_backends
except ImportError:
    print("Error: Cannot import luminet backends")
    print("Make sure you're running from the luminet directory")
    sys.exit(1)


class BenchmarkResult:
    """Store benchmark results."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.results = {}

    def add_result(self, test_name: str, time_ms: float, metadata: dict = None):
        """Add a benchmark result."""
        self.results[test_name] = {
            'time_ms': time_ms,
            'metadata': metadata or {}
        }

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*70}")
        print(f"BENCHMARK RESULTS: {self.backend_name.upper()}")
        print(f"{'='*70}")

        for test_name, result in self.results.items():
            print(f"\n{test_name}:")
            print(f"  Time: {result['time_ms']:.2f} ms")
            if result['metadata']:
                for key, value in result['metadata'].items():
                    print(f"  {key}: {value}")


def benchmark_backend(
    backend_name: str,
    resolutions: list = [50, 100, 200],
    repeat: int = 3
) -> BenchmarkResult:
    """Run benchmarks for a specific backend.

    Args:
        backend_name: Name of backend to benchmark
        resolutions: List of resolutions to test
        repeat: Number of times to repeat each test

    Returns:
        BenchmarkResult object
    """
    print(f"\nInitializing {backend_name} backend...")

    try:
        backend = get_backend(backend_name)
        print(f"✓ Backend initialized: {backend.get_backend_name()}")
        print(f"  Supports GPU: {backend.supports_gpu()}")
        print(f"  Supports vectorization: {backend.supports_vectorization()}")
    except Exception as e:
        print(f"✗ Failed to initialize backend: {e}")
        return None

    result = BenchmarkResult(backend_name)

    # Benchmark 1: calc_q function
    print(f"\n[1/5] Benchmarking calc_q...")
    p_values = np.linspace(2.1, 100, 10000)

    times = []
    for _ in range(repeat):
        start = time.time()
        q_values = backend.calc_q(p_values, bh_mass=1.0)
        end = time.time()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    result.add_result("calc_q", avg_time, {
        'num_values': len(p_values),
        'mean_std': np.std(times)
    })
    print(f"  ✓ Average: {avg_time:.2f} ms (±{np.std(times):.2f})")

    # Benchmark 2: calc_k_squared function
    print(f"\n[2/5] Benchmarking calc_k_squared...")

    times = []
    for _ in range(repeat):
        start = time.time()
        k2_values = backend.calc_k_squared(p_values, bh_mass=1.0)
        end = time.time()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    result.add_result("calc_k_squared", avg_time, {
        'num_values': len(p_values),
        'mean_std': np.std(times)
    })
    print(f"  ✓ Average: {avg_time:.2f} ms (±{np.std(times):.2f})")

    # Benchmark 3: calc_sn (Jacobi elliptic) - Most expensive
    print(f"\n[3/5] Benchmarking calc_sn (Jacobi elliptic)...")
    p_test = np.array([3.5, 5, 10, 20, 50])
    alpha_test = np.linspace(0, 2*np.pi, 1000)
    total_tests = len(p_test) * len(alpha_test) * 2  # 2 orders

    times = []
    for _ in range(repeat):
        start = time.time()
        sn_values = []
        for p in p_test:
            for alpha in alpha_test:
                for order in [0, 1]:
                    sn = backend.calc_sn(p, alpha, 1.0, 1.4, order)
                    sn_values.append(sn)
        end = time.time()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    result.add_result("calc_sn", avg_time, {
        'num_calculations': total_tests,
        'mean_std': np.std(times)
    })
    print(f"  ✓ Average: {avg_time:.2f} ms (±{np.std(times):.2f})")
    print(f"  ✓ Speed: {total_tests/avg_time:.1f} calculations/sec")

    # Benchmark 4: solve_for_periastron (root finding)
    print(f"\n[4/5] Benchmarking solve_for_periastron...")
    radii = np.array([6.5, 10, 15, 20, 30, 50])
    alpha_test2 = np.linspace(0, 2*np.pi, 100)
    total_tests = len(radii) * len(alpha_test2) * 2  # 2 orders

    times = []
    for _ in range(repeat):
        start = time.time()
        periastron_values = []
        for r in radii:
            for alpha in alpha_test2:
                for order in [0, 1]:
                    p = backend.solve_for_periastron(r, 1.4, alpha, 1.0, order)
                    periastron_values.append(p)
        end = time.time()
        times.append((end - start) * 1000)

    avg_time = np.mean(times)
    result.add_result("solve_for_periastron", avg_time, {
        'num_solves': total_tests,
        'mean_std': np.std(times)
    })
    print(f"  ✓ Average: {avg_time:.2f} ms (±{np.std(times):.2f})")
    print(f"  ✓ Speed: {total_tests/avg_time:.1f} solves/sec")

    # Benchmark 5: Full Black Hole calculation
    print(f"\n[5/5] Benchmarking full Black Hole calculation...")

    for resolution in resolutions:
        times = []
        total_points = resolution * resolution

        for _ in range(repeat):
            start = time.time()

            # Simulate black hole calculation
            radii = np.linspace(6.0, 30.0, resolution)
            angles = np.linspace(0, 2*np.pi, resolution)

            impact_params = []
            for r in radii:
                for alpha in angles:
                    b = backend.solve_for_impact_parameter(r, 1.4, alpha, 1.0, 0)
                    impact_params.append(b)

            end = time.time()
            times.append((end - start) * 1000)

        avg_time = np.mean(times)
        result.add_result(f"blackhole_{resolution}x{resolution}", avg_time, {
            'resolution': resolution,
            'total_points': total_points,
            'mean_std': np.std(times)
        })
        print(f"  ✓ Resolution {resolution}x{resolution}: {avg_time:.2f} ms (±{np.std(times):.2f})")
        print(f"    Speed: {total_points/avg_time:.1f} points/sec")

    return result


def compare_backends(backend_names: list = None, resolutions: list = [50, 100, 200]):
    """Compare performance between multiple backends.

    Args:
        backend_names: List of backend names to compare
        resolutions: Resolutions to test
    """
    if backend_names is None:
        backend_names = list_available_backends()

    print("\n" + "="*70)
    print(f"COMPARING BACKENDS: {', '.join(backend_names)}")
    print("="*70)

    # Run benchmarks for each backend
    all_results = {}
    for backend_name in backend_names:
        result = benchmark_backend(backend_name, resolutions=resolutions)
        if result:
            all_results[backend_name] = result

    # Generate comparison table
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    # Compare full black hole calculations at different resolutions
    print("\nBlack Hole Rendering (ms):")
    print(f"{'Resolution':<15}", end="")
    for backend_name in backend_names:
        print(f"{backend_name.upper():<15}", end="")
    print()

    for resolution in resolutions:
        print(f"{resolution}x{resolution:<13}", end="")
        for backend_name in backend_names:
            if backend_name in all_results:
                key = f"blackhole_{resolution}x{resolution}"
                if key in all_results[backend_name].results:
                    time_ms = all_results[backend_name].results[key]['time_ms']
                    print(f"{time_ms:<15.2f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()

    # Calculate speedup
    if len(all_results) >= 2 and 'scipy' in all_results:
        print("\nSpeedup (relative to scipy):")
        scipy_result = all_results['scipy']

        for resolution in resolutions:
            key = f"blackhole_{resolution}x{resolution}"
            if key not in scipy_result.results:
                continue

            scipy_time = scipy_result.results[key]['time_ms']
            print(f"\n{resolution}x{resolution}:")

            for backend_name in backend_names:
                if backend_name == 'scipy':
                    print(f"  {backend_name.upper()}: 1.00x (baseline)")
                elif backend_name in all_results and key in all_results[backend_name].results:
                    time_ms = all_results[backend_name].results[key]['time_ms']
                    speedup = scipy_time / time_ms
                    print(f"  {backend_name.upper()}: {speedup:.2f}x faster")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBenchmarked backends: {len(all_results)}")
    print(f"Available backends: {', '.join(list_available_backends())}")
    print("\nKey findings:")

    if 'scipy' in all_results and 'taichi' in all_results:
        scipy_time = all_results['scipy'].results.get('blackhole_100x100', {}).get('time_ms', 0)
        taichi_time = all_results['taichi'].results.get('blackhole_100x100', {}).get('time_ms', 0)

        if scipy_time > 0 and taichi_time > 0:
            speedup = scipy_time / taichi_time
            print(f"  - Taichi is {speedup:.2f}x faster than scipy at 100x100")
        else:
            print(f"  - Could not calculate speedup (missing data)")

    print("\n" + "="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark luminet computational backends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --engine=scipy
  python benchmark.py --engine=taichi
  python benchmark.py --compare
  python benchmark.py --compare --resolutions 50 100 200
        """
    )

    parser.add_argument(
        '--engine',
        type=str,
        choices=['scipy', 'taichi', 'jax', 'numba', 'mojo'] + list_available_backends(),
        help='Backend to benchmark (default: compare all)'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all available backends'
    )

    parser.add_argument(
        '--resolutions',
        type=int,
        nargs='+',
        default=[50, 100, 200],
        help='Resolutions to test (default: 50 100 200)'
    )

    parser.add_argument(
        '--repeat',
        type=int,
        default=3,
        help='Number of times to repeat each test (default: 3)'
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "="*70)
    print("LUMINET BACKEND BENCHMARK")
    print("="*70)
    print(f"\nAvailable backends: {', '.join(list_available_backends())}")

    # Run benchmarks
    if args.compare:
        compare_backends(resolutions=args.resolutions)
    elif args.engine:
        result = benchmark_backend(args.engine, resolutions=args.resolutions, repeat=args.repeat)
        if result:
            result.print_summary()
    else:
        # Default to compare
        compare_backends(resolutions=args.resolutions)


if __name__ == "__main__":
    main()
