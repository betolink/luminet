#!/usr/bin/env python
"""Comprehensive accuracy comparison test for all backends.

This script tests all available backends against scipy (reference) and generates
a detailed accuracy report.
"""

import json
import numpy as np
import time
from pathlib import Path

import luminet.black_hole_math as bhmath
from luminet.backends import list_available_backends


def test_single_backend(backend_name, reference_values, test_config, **backend_kwargs):
    """Test a single backend against reference values.
    
    Args:
        backend_name: Name of backend to test
        reference_values: Reference values from scipy
        test_config: Dictionary with test parameters
        **backend_kwargs: Additional backend arguments
    
    Returns:
        dict: Test results including accuracy and timing
    """
    print(f"\n  Testing {backend_name}...")
    
    try:
        # Set backend
        bhmath.set_backend(backend_name, **backend_kwargs)
        backend = bhmath.get_current_backend()
        backend_full_name = backend.get_backend_name()
        
        # Run computation
        start = time.time()
        test_values = np.array([
            bhmath.solve_for_impact_parameter(r, incl, a, bh_mass, order)
            for r, incl, a, bh_mass, order in test_config['test_points']
        ])
        elapsed = time.time() - start
        
        # Compute errors
        valid_mask = ~(np.isnan(reference_values) | np.isnan(test_values))
        valid_ref = reference_values[valid_mask]
        valid_test = test_values[valid_mask]
        
        if len(valid_ref) == 0:
            return {
                'backend': backend_full_name,
                'status': 'error',
                'error': 'No valid test points',
            }
        
        abs_errors = np.abs(valid_test - valid_ref)
        rel_errors = abs_errors / np.abs(valid_ref)
        
        # Determine if precision loss is significant
        max_rel_error = float(np.max(rel_errors))
        is_acceptable = max_rel_error < 1e-6  # f64 precision threshold
        precision_estimate = 'f64' if max_rel_error < 1e-10 else 'f32' if max_rel_error < 1e-4 else 'low'
        
        result = {
            'backend': backend_full_name,
            'status': 'success',
            'timing_s': elapsed,
            'speedup': test_config['reference_time'] / elapsed if elapsed > 0 else 0,
            'accuracy': {
                'valid_points': int(np.sum(valid_mask)),
                'total_points': len(reference_values),
                'mean_abs_error': float(np.mean(abs_errors)),
                'median_abs_error': float(np.median(abs_errors)),
                'max_abs_error': float(np.max(abs_errors)),
                'mean_rel_error': float(np.mean(rel_errors)),
                'median_rel_error': float(np.median(rel_errors)),
                'max_rel_error': max_rel_error,
                'percentile_95': float(np.percentile(rel_errors, 95)),
                'percentile_99': float(np.percentile(rel_errors, 99)),
                'is_acceptable': is_acceptable,
                'estimated_precision': precision_estimate,
            }
        }
        
        print(f"    ✓ {backend_full_name}")
        print(f"      Time: {elapsed*1000:.1f}ms (speedup: {result['speedup']:.1f}×)")
        print(f"      Max rel error: {max_rel_error:.2e} ({precision_estimate})")
        
        return result
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return {
            'backend': backend_name,
            'status': 'error',
            'error': str(e),
        }


def run_accuracy_tests(output_file='accuracy_comparison.json', num_points=200):
    """Run comprehensive accuracy tests on all backends.
    
    Args:
        output_file: Output JSON file for results
        num_points: Number of test points to use
    """
    print("=" * 70)
    print("BACKEND ACCURACY COMPARISON TEST")
    print("=" * 70)
    print(f"\nTest parameters:")
    print(f"  Number of test points: {num_points}")
    print(f"  Reference backend: scipy")
    
    # Generate test points (varied parameters to test different code paths)
    np.random.seed(42)
    radii = np.linspace(6.5, 30.0, num_points)
    incls = np.random.uniform(0.5, np.pi/2 - 0.1, num_points)
    angles = np.random.uniform(0, 2*np.pi, num_points)
    bh_masses = np.ones(num_points)
    orders = np.zeros(num_points, dtype=int)
    
    # Add some ghost image tests
    orders[num_points//4:num_points//3] = 1
    
    test_points = list(zip(radii, incls, angles, bh_masses, orders))
    
    # Compute reference values with scipy
    print(f"\nComputing reference values with scipy...")
    bhmath.set_backend('scipy')
    
    start = time.time()
    reference_values = np.array([
        bhmath.solve_for_impact_parameter(r, incl, a, bh_mass, order)
        for r, incl, a, bh_mass, order in test_points
    ])
    reference_time = time.time() - start
    
    print(f"  Reference time: {reference_time*1000:.1f}ms")
    print(f"  Valid points: {np.sum(~np.isnan(reference_values))}/{len(reference_values)}")
    
    test_config = {
        'test_points': test_points,
        'reference_time': reference_time,
        'num_points': num_points,
    }
    
    # Test all available backends
    print(f"\nTesting backends...")
    print("-" * 70)
    
    available_backends = list_available_backends()
    results = []
    
    # Test each backend
    for backend_name in available_backends:
        if backend_name == 'scipy':
            continue  # Skip reference
        
        result = test_single_backend(backend_name, reference_values, test_config)
        results.append(result)
    
    # Test taichi with different architectures
    if 'taichi' in available_backends:
        result_cpu = test_single_backend('taichi', reference_values, test_config, arch='cpu')
        results.append(result_cpu)
    
    # Prepare output
    output_data = {
        'test_info': {
            'num_points': num_points,
            'reference_backend': 'scipy',
            'reference_time_s': reference_time,
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': results,
    }
    
    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        print(f"\nSuccessful backends: {len(successful)}/{len(results)}")
        print(f"\n{'Backend':<25} {'Speedup':<10} {'Max Rel Err':<15} {'Precision':<10}")
        print("-" * 70)
        
        for result in sorted(successful, key=lambda x: x['speedup'], reverse=True):
            backend = result['backend']
            speedup = result['speedup']
            max_err = result['accuracy']['max_rel_error']
            precision = result['accuracy']['estimated_precision']
            print(f"{backend:<25} {speedup:>6.1f}×    {max_err:>12.2e}    {precision:<10}")
        
        # Find fastest accurate backend
        accurate = [r for r in successful if r['accuracy']['is_acceptable']]
        if accurate:
            fastest = max(accurate, key=lambda x: x['speedup'])
            print(f"\n⭐ Recommended: {fastest['backend']} ({fastest['speedup']:.1f}× faster, {fastest['accuracy']['max_rel_error']:.2e} error)")
    
    print(f"\n✓ Results saved to: {output_path}")
    print("=" * 70)
    
    return output_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test accuracy of all backends')
    parser.add_argument('--points', type=int, default=200,
                       help='Number of test points (default: 200)')
    parser.add_argument('--output', default='accuracy_comparison.json',
                       help='Output JSON file (default: accuracy_comparison.json)')
    
    args = parser.parse_args()
    
    try:
        run_accuracy_tests(output_file=args.output, num_points=args.points)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
