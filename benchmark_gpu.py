"""Benchmark Taichi GPU vs CPU vs Scipy performance."""
import sys
sys.path.insert(0, '/home/betolink/hackweek/luminet')

import time
import numpy as np
from luminet.backends import get_backend

print("=" * 70)
print("Backend Performance Benchmark")
print("=" * 70)

# Test parameters
test_sizes = [100, 1000, 10000]
p_val = 5.0
bh_mass = 1.0

backends = [
    ('scipy', {}),
    ('numba', {}),
    ('taichi', {'arch': 'cpu'}),
    ('taichi', {'arch': 'gpu'}),
]

results = {}

for backend_name, kwargs in backends:
    label = f"{backend_name}"
    if kwargs:
        label += f" ({kwargs.get('arch', 'default')})"
    
    print(f"\n{label}:")
    print("-" * 70)
    
    try:
        backend = get_backend(backend_name, **kwargs)
        results[label] = {}
        
        for size in test_sizes:
            # Create test data
            p_array = np.full(size, p_val)
            
            # Warmup
            _ = backend.calc_q(p_array, bh_mass)
            
            # Benchmark
            start = time.time()
            result = backend.calc_q(p_array, bh_mass)
            elapsed = time.time() - start
            
            results[label][size] = elapsed * 1000  # Convert to ms
            print(f"  Size {size:5d}: {elapsed*1000:7.2f} ms")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        continue

# Print summary table
print("\n" + "=" * 70)
print("Performance Summary (milliseconds)")
print("=" * 70)
print(f"{'Backend':<20}", end="")
for size in test_sizes:
    print(f"{size:>12}", end="")
print()
print("-" * 70)

baseline_label = list(results.keys())[0] if results else None
for label, timings in results.items():
    print(f"{label:<20}", end="")
    for size in test_sizes:
        if size in timings:
            ms = timings[size]
            print(f"{ms:12.2f}", end="")
    print()

# Print speedup vs scipy
if baseline_label and len(results) > 1:
    print("\n" + "=" * 70)
    print(f"Speedup vs {baseline_label}")
    print("=" * 70)
    print(f"{'Backend':<20}", end="")
    for size in test_sizes:
        print(f"{size:>12}", end="")
    print()
    print("-" * 70)
    
    baseline_timings = results[baseline_label]
    for label, timings in results.items():
        if label == baseline_label:
            continue
        print(f"{label:<20}", end="")
        for size in test_sizes:
            if size in timings and size in baseline_timings:
                speedup = baseline_timings[size] / timings[size]
                print(f"{speedup:12.2f}x", end="")
        print()

print("\n" + "=" * 70)
