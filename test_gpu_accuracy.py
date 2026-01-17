#!/usr/bin/env python
"""Test GPU (f32) backend accuracy and performance.

This script compares the Taichi GPU backend (f32 precision) against
the scipy baseline to measure:
1. Accuracy (relative error)
2. Performance (speedup)
3. Precision loss patterns
"""

import numpy as np
import time
import warnings
from luminet.backends import get_backend

# Suppress Taichi precision warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='taichi')

print("="*70)
print("TAICHI GPU (f32) ACCURACY & PERFORMANCE TEST")
print("="*70)
print()

# Initialize backends
print("Initializing backends...")
scipy = get_backend('scipy')
numba = get_backend('numba')

# Try to get GPU backend
try:
    taichi_gpu = get_backend('taichi', arch='vulkan')
    print(f"✓ Taichi GPU: {taichi_gpu.get_backend_name()}")
    print(f"  - GPU: {taichi_gpu.supports_gpu()}")
    print(f"  - Architecture: {taichi_gpu.get_arch()}")
    print(f"  - Precision: {'f32' if not taichi_gpu._use_f64 else 'f64'}")
except Exception as e:
    print(f"✗ Failed to initialize GPU backend: {e}")
    print("\nFalling back to CPU comparison...")
    taichi_gpu = get_backend('taichi', arch='cpu')
    print(f"✓ Taichi CPU: {taichi_gpu.get_backend_name()}")

print()

# =============================================================================
# TEST 1: Single Point Accuracy
# =============================================================================
print("-"*70)
print("TEST 1: Single Point Accuracy")
print("-"*70)

test_cases = [
    (10.0, 1.4, 1.0, 1.0, 0, "Standard case"),
    (15.0, 1.2, 2.5, 1.0, 0, "Different angle"),
    (20.0, 1.0, 0.5, 1.0, 0, "Low inclination"),
    (6.5, 1.4, 3.0, 1.0, 0, "Near photon sphere"),
    (50.0, 0.3, 1.0, 1.0, 0, "Far field, low incl"),
]

print(f"\n{'Case':<25} {'Scipy':<15} {'GPU (f32)':<15} {'Rel Error':<12} {'Status'}")
print("-"*70)

errors = []
for radius, incl, alpha, bh_mass, order, desc in test_cases:
    scipy_result = scipy.solve_for_impact_parameter(radius, incl, alpha, bh_mass, order)
    gpu_result = taichi_gpu.solve_for_impact_parameter(radius, incl, alpha, bh_mass, order)
    
    if not np.isnan(scipy_result) and not np.isnan(gpu_result):
        rel_error = abs(scipy_result - gpu_result) / abs(scipy_result)
        errors.append(rel_error)
        status = "✓ Good" if rel_error < 1e-4 else "⚠ Degraded" if rel_error < 1e-2 else "✗ Poor"
        print(f"{desc:<25} {scipy_result:<15.8f} {gpu_result:<15.8f} {rel_error:<12.2e} {status}")
    else:
        print(f"{desc:<25} {scipy_result:<15.8f} {gpu_result:<15.8f} {'NaN':<12} {'N/A'}")

if errors:
    print()
    print(f"Mean relative error:   {np.mean(errors):.2e}")
    print(f"Max relative error:    {np.max(errors):.2e}")
    print(f"Min relative error:    {np.min(errors):.2e}")
    print(f"Std relative error:    {np.std(errors):.2e}")

# =============================================================================
# TEST 2: Batch Processing Accuracy (Grid)
# =============================================================================
print()
print("-"*70)
print("TEST 2: Batch Processing Accuracy (50×50 Grid)")
print("-"*70)

# Create test grid
radii = np.linspace(6.5, 30.0, 50)
angles = np.linspace(0, 2*np.pi, 50)
radii_grid, angles_grid = np.meshgrid(radii, angles)
radii_flat = radii_grid.flatten().astype(np.float64)
angles_flat = angles_grid.flatten().astype(np.float64)

print(f"\nComputing {len(radii_flat)} points...")

# Scipy baseline (scalar loop)
print("  Computing scipy baseline...")
scipy_results = np.zeros(len(radii_flat))
scipy_start = time.time()
for i in range(len(radii_flat)):
    scipy_results[i] = scipy.solve_for_impact_parameter(
        radii_flat[i], 1.4, angles_flat[i], 1.0, 0
    )
scipy_time = (time.time() - scipy_start) * 1000
print(f"  ✓ Scipy: {scipy_time:.1f} ms")

# Taichi GPU (batch)
print("  Computing Taichi GPU (batch)...")
gpu_start = time.time()
gpu_results = taichi_gpu.solve_for_impact_parameter(
    radii_flat, 1.4, angles_flat, 1.0, 0
)
gpu_time = (time.time() - gpu_start) * 1000
print(f"  ✓ GPU: {gpu_time:.1f} ms")

# Numba for reference (batch)
print("  Computing Numba (batch)...")
_ = numba.solve_for_impact_parameter(radii_flat, 1.4, angles_flat, 1.0, 0)  # warmup
numba_start = time.time()
numba_results = numba.solve_for_impact_parameter(
    radii_flat, 1.4, angles_flat, 1.0, 0
)
numba_time = (time.time() - numba_start) * 1000
print(f"  ✓ Numba: {numba_time:.1f} ms")

# Calculate errors (excluding NaN values)
valid_mask = ~(np.isnan(scipy_results) | np.isnan(gpu_results))
valid_scipy = scipy_results[valid_mask]
valid_gpu = gpu_results[valid_mask]
valid_numba = numba_results[valid_mask]

if len(valid_scipy) > 0:
    gpu_errors = np.abs(valid_scipy - valid_gpu) / np.abs(valid_scipy)
    numba_errors = np.abs(valid_scipy - valid_numba) / np.abs(valid_scipy)
    
    print()
    print("Accuracy Statistics:")
    print(f"  Valid points: {len(valid_scipy)} / {len(scipy_results)}")
    print()
    print(f"  GPU (f32) vs Scipy:")
    print(f"    Mean error:   {np.mean(gpu_errors):.2e}")
    print(f"    Median error: {np.median(gpu_errors):.2e}")
    print(f"    Max error:    {np.max(gpu_errors):.2e}")
    print(f"    Min error:    {np.min(gpu_errors):.2e}")
    print(f"    Std error:    {np.std(gpu_errors):.2e}")
    print()
    print(f"  Numba (f64) vs Scipy:")
    print(f"    Mean error:   {np.mean(numba_errors):.2e}")
    print(f"    Max error:    {np.max(numba_errors):.2e}")
    
    print()
    print("Performance:")
    print(f"  Scipy:  {scipy_time:8.1f} ms  (baseline)")
    print(f"  Numba:  {numba_time:8.1f} ms  ({scipy_time/numba_time:5.1f}× speedup)")
    print(f"  GPU:    {gpu_time:8.1f} ms  ({scipy_time/gpu_time:5.1f}× speedup)")

# =============================================================================
# TEST 3: Error Distribution Analysis
# =============================================================================
print()
print("-"*70)
print("TEST 3: Error Distribution Analysis")
print("-"*70)

if len(valid_scipy) > 0:
    # Categorize errors
    excellent = np.sum(gpu_errors < 1e-6)
    good = np.sum((gpu_errors >= 1e-6) & (gpu_errors < 1e-4))
    acceptable = np.sum((gpu_errors >= 1e-4) & (gpu_errors < 1e-2))
    poor = np.sum(gpu_errors >= 1e-2)
    
    total = len(gpu_errors)
    print(f"\nError Distribution:")
    print(f"  Excellent (< 1e-6):   {excellent:6d} / {total} ({100*excellent/total:5.1f}%)")
    print(f"  Good      (< 1e-4):   {good:6d} / {total} ({100*good/total:5.1f}%)")
    print(f"  Acceptable(< 1e-2):   {acceptable:6d} / {total} ({100*acceptable/total:5.1f}%)")
    print(f"  Poor      (>= 1e-2):  {poor:6d} / {total} ({100*poor/total:5.1f}%)")
    
    # Percentiles
    print(f"\nError Percentiles:")
    for p in [50, 90, 95, 99, 99.9]:
        print(f"  {p:5.1f}th percentile: {np.percentile(gpu_errors, p):.2e}")

# =============================================================================
# SUMMARY & RECOMMENDATIONS
# =============================================================================
print()
print("="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

if taichi_gpu.supports_gpu():
    print(f"\n✓ GPU backend successfully tested: {taichi_gpu.get_backend_name()}")
    print(f"  Architecture: {taichi_gpu.get_arch()}")
    print(f"  Precision: f32 (single precision)")
    
    if len(valid_scipy) > 0:
        mean_error = np.mean(gpu_errors)
        max_error = np.max(gpu_errors)
        
        print()
        print("Accuracy Assessment:")
        if mean_error < 1e-6 and max_error < 1e-4:
            print("  ✓ EXCELLENT - Suitable for scientific computing")
            recommendation = "RECOMMENDED for production use"
        elif mean_error < 1e-4 and max_error < 1e-2:
            print("  ✓ GOOD - Minor precision loss, acceptable for most uses")
            recommendation = "ACCEPTABLE for scientific visualization"
        elif mean_error < 1e-2 and max_error < 0.1:
            print("  ⚠ ACCEPTABLE - Noticeable precision loss")
            recommendation = "USE ONLY for visualization, NOT for science"
        else:
            print("  ✗ POOR - Significant precision loss")
            recommendation = "NOT RECOMMENDED - Use CPU backend instead"
        
        print()
        print("Performance Assessment:")
        if gpu_time < scipy_time:
            speedup = scipy_time / gpu_time
            print(f"  ✓ {speedup:.1f}× faster than scipy baseline")
            if speedup > 5:
                print(f"  ✓ EXCELLENT speedup")
            elif speedup > 2:
                print(f"  ✓ GOOD speedup")
            else:
                print(f"  ⚠ Modest speedup")
        else:
            print(f"  ✗ SLOWER than scipy baseline")
        
        print()
        print(f"RECOMMENDATION: {recommendation}")
        
        print()
        print("For scientific computing, consider:")
        print(f"  1. Numba backend: {numba_time:.1f} ms ({scipy_time/numba_time:.1f}× speedup, f64 precision)")
        print(f"  2. Taichi CPU:    Similar to Numba (f64 precision)")
else:
    print("\n⚠ Running on CPU (no GPU detected or GPU failed)")
    print("  Consider using Numba backend for best CPU performance")

print()
print("="*70)
print("END OF TEST")
print("="*70)
