#!/usr/bin/env python
"""Command-line tool for rendering black holes with different backends.

This tool allows you to:
1. Render black hole images with different computational backends
2. Compare accuracy between backends (--debug mode)
3. Benchmark performance across backends
4. Export debug statistics to JSON

Examples:
    # Basic rendering with default scipy backend
    python render.py --output=blackhole.png
    
    # 10× faster rendering with Numba
    python render.py --backend=numba --output=fast.png
    
    # High resolution scientific render
    python render.py --backend=numba --resolution=500 --output=hires.png
    
    # Accuracy comparison with debug output
    python render.py --backend=numba --debug --debug-file=stats.json
    
    # GPU acceleration (Taichi on NVIDIA CUDA or AMD Vulkan)
    python render.py --backend=taichi --hw=gpu --output=gpu.png
    
    # Specific GPU backend (cuda for NVIDIA, vulkan for AMD)
    python render.py --backend=taichi --hw=cuda --output=nvidia.png
    python render.py --backend=taichi --hw=vulkan --output=amd.png
    
    # Compare all backends (benchmark mode)
    python render.py --benchmark --output=benchmark.png
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from luminet.black_hole import BlackHole
from luminet.backends import list_available_backends, get_backend_info
import luminet.black_hole_math as bhmath


def format_time(seconds):
    """Format time duration in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"


def compute_accuracy_stats(backend_name, reference_backend='scipy', 
                           test_points=100, **backend_kwargs):
    """Compute accuracy statistics by comparing against reference backend.
    
    Args:
        backend_name: Backend to test
        reference_backend: Reference backend for comparison (default: scipy)
        test_points: Number of test points to compare
        **backend_kwargs: Additional backend arguments
    
    Returns:
        dict: Statistics including mean/median/max errors and timing
    """
    print(f"   Computing accuracy statistics against {reference_backend}...")
    
    # Create test grid
    radii = np.linspace(6.5, 30.0, test_points)
    angles = np.linspace(0, 2*np.pi, test_points)
    
    # Reference backend
    bhmath.set_backend(reference_backend)
    ref_backend = bhmath.get_current_backend()
    
    print(f"   Computing reference values with {ref_backend.get_backend_name()}...")
    start = time.time()
    ref_values = np.array([
        bhmath.solve_for_impact_parameter(r, 1.4, a, 1.0, 0)
        for r, a in zip(radii, angles)
    ])
    ref_time = time.time() - start
    
    # Test backend
    bhmath.set_backend(backend_name, **backend_kwargs)
    test_backend = bhmath.get_current_backend()
    
    print(f"   Computing test values with {test_backend.get_backend_name()}...")
    start = time.time()
    test_values = np.array([
        bhmath.solve_for_impact_parameter(r, 1.4, a, 1.0, 0)
        for r, a in zip(radii, angles)
    ])
    test_time = time.time() - start
    
    # Compute errors (only for valid values)
    valid_mask = ~(np.isnan(ref_values) | np.isnan(test_values))
    valid_ref = ref_values[valid_mask]
    valid_test = test_values[valid_mask]
    
    abs_errors = np.abs(valid_test - valid_ref)
    rel_errors = abs_errors / np.abs(valid_ref)
    
    stats = {
        'backend': test_backend.get_backend_name(),
        'reference': ref_backend.get_backend_name(),
        'test_points': test_points,
        'valid_points': int(np.sum(valid_mask)),
        'accuracy': {
            'mean_abs_error': float(np.mean(abs_errors)),
            'median_abs_error': float(np.median(abs_errors)),
            'max_abs_error': float(np.max(abs_errors)),
            'mean_rel_error': float(np.mean(rel_errors)),
            'median_rel_error': float(np.median(rel_errors)),
            'max_rel_error': float(np.max(rel_errors)),
            'percentile_99_rel_error': float(np.percentile(rel_errors, 99)),
        },
        'timing': {
            'reference_time_s': ref_time,
            'test_time_s': test_time,
            'speedup': ref_time / test_time if test_time > 0 else 0,
        }
    }
    
    return stats


def benchmark_backends(backends, resolution=100, **bh_kwargs):
    """Benchmark multiple backends and return comparison stats.
    
    Args:
        backends: List of backend names to benchmark
        resolution: Resolution for rendering
        **bh_kwargs: Additional BlackHole constructor arguments
    
    Returns:
        dict: Benchmark results for each backend
    """
    results = {}
    
    print(f"\nBenchmarking {len(backends)} backends at {resolution}×{resolution} resolution...")
    print("=" * 70)
    
    for backend_name in backends:
        try:
            print(f"\nBenchmark: {backend_name}")
            print("-" * 70)
            
            # Parse backend kwargs if needed
            bkwargs = {}
            if backend_name.startswith('taichi'):
                parts = backend_name.split('-')
                if len(parts) > 1 and parts[1] in ['cpu', 'gpu', 'vulkan', 'cuda']:
                    backend_name = 'taichi'
                    bkwargs['arch'] = parts[1]
            
            # Create black hole
            start = time.time()
            bh = BlackHole(
                angular_resolution=resolution,
                radial_resolution=resolution,
                backend=backend_name,
                **bkwargs,
                **bh_kwargs
            )
            init_time = time.time() - start
            
            # Render (just calculate, don't plot yet)
            start = time.time()
            radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, resolution)
            bh.calc_isoradials(direct_r=radii, ghost_r=[])
            render_time = time.time() - start
            
            total_time = init_time + render_time
            
            results[bh.backend_name] = {
                'init_time_s': init_time,
                'render_time_s': render_time,
                'total_time_s': total_time,
                'points_calculated': resolution * resolution,
                'points_per_second': (resolution * resolution) / render_time if render_time > 0 else 0,
            }
            
            print(f"   Init time:   {format_time(init_time)}")
            print(f"   Render time: {format_time(render_time)}")
            print(f"   Total time:  {format_time(total_time)}")
            print(f"   Throughput:  {results[bh.backend_name]['points_per_second']:.0f} points/sec")
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results[backend_name] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Render black hole accretion disk with different backends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Backend selection
    available = list_available_backends()
    parser.add_argument(
        '--backend', 
        default='scipy',
        choices=available + ['taichi-cpu', 'taichi-gpu'],
        help='Computational backend to use (default: scipy, recommended: numba)'
    )
    parser.add_argument(
        '--hw',
        choices=['cpu', 'gpu', 'vulkan', 'cuda'],
        help='Hardware architecture for taichi backend: '
             'gpu (auto-detect), cuda (NVIDIA), vulkan (AMD/Intel), cpu (default)'
    )
    
    # Black hole parameters
    parser.add_argument('--mass', type=float, default=1.0,
                       help='Black hole mass (default: 1.0)')
    parser.add_argument('--inclination', type=float, default=1.4,
                       help='Inclination angle in radians (default: 1.4)')
    parser.add_argument('--accretion', type=float, default=1.0,
                       help='Accretion rate (default: 1.0)')
    parser.add_argument('--outer-edge', type=float, default=None,
                       help='Outer edge of disk (default: 30*mass)')
    
    # Resolution
    parser.add_argument('--resolution', type=int, default=200,
                       help='Resolution NxN (default: 200)')
    parser.add_argument('--angular-res', type=int, default=None,
                       help='Angular resolution (default: same as --resolution)')
    parser.add_argument('--radial-res', type=int, default=None,
                       help='Radial resolution (default: same as --resolution)')
    
    # Output
    parser.add_argument('--output', '--out', default='blackhole.png',
                       help='Output filename (default: blackhole.png)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output image (default: 150)')
    parser.add_argument('--color-scheme', '--cmap', default='flux',
                       choices=['flux', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'],
                       help='Color scheme for black hole (default: flux, physical greyscale)')
    parser.add_argument('--bg-color', '--background', default='black',
                       choices=['black', 'white'],
                       help='Background color (default: black)')
    
    # Debug and benchmarking
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode: compute accuracy statistics')
    parser.add_argument('--debug-file', default='render_stats.json',
                       help='Output file for debug statistics (default: render_stats.json)')
    parser.add_argument('--debug-points', type=int, default=200,
                       help='Number of test points for accuracy comparison (default: 200)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark all available backends')
    
    # Display
    parser.add_argument('--show', action='store_true',
                       help='Display plot in addition to saving')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Prepare backend kwargs
    backend_kwargs = {}
    backend_name = args.backend
    
    # Handle taichi shorthand backends
    if backend_name.startswith('taichi-'):
        parts = backend_name.split('-')
        if len(parts) == 2 and parts[1] in ['cpu', 'gpu']:
            backend_name = 'taichi'
            backend_kwargs['arch'] = parts[1]
    
    if args.hw:
        backend_kwargs['arch'] = args.hw
    
    # Benchmark mode
    if args.benchmark:
        if not args.quiet:
            print("BENCHMARK MODE")
            print("=" * 70)
        
        # Benchmark all available backends
        backends_to_test = available.copy()
        if 'taichi' in backends_to_test:
            backends_to_test.remove('taichi')
            backends_to_test.extend(['taichi-cpu'])  # Skip GPU for benchmark
        
        benchmark_results = benchmark_backends(
            backends_to_test,
            resolution=args.resolution,
            mass=args.mass,
            incl=args.inclination,
            acc=args.accretion,
            outer_edge=args.outer_edge
        )
        
        # Save benchmark results
        output_file = Path(args.output).stem + '_benchmark.json'
        with open(output_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        if not args.quiet:
            print(f"\n✓ Benchmark results saved to: {output_file}")
        
        return 0
    
    # Regular rendering mode
    if not args.quiet:
        print("BLACK HOLE RENDERER")
        print("=" * 70)
        backend_info = get_backend_info(backend_name)
        if backend_info:
            print(f"\nBackend: {backend_info['name']}")
            print(f"  GPU: {backend_info['gpu']}")
            print(f"  Vectorized: {backend_info['vectorized']}")
            if backend_kwargs:
                print(f"  Options: {backend_kwargs}")
        else:
            print(f"\nBackend: {backend_name}")
    
    # Debug mode: compute accuracy statistics
    debug_stats = None
    if args.debug:
        if not args.quiet:
            print(f"\nDEBUG MODE: Computing accuracy statistics...")
            print("-" * 70)
        
        debug_stats = compute_accuracy_stats(
            backend_name, 
            reference_backend='scipy',
            test_points=args.debug_points,
            **backend_kwargs
        )
        
        if not args.quiet:
            print(f"\nAccuracy Statistics:")
            print(f"  Valid points: {debug_stats['valid_points']}/{debug_stats['test_points']}")
            print(f"  Mean relative error: {debug_stats['accuracy']['mean_rel_error']:.2e}")
            print(f"  Median relative error: {debug_stats['accuracy']['median_rel_error']:.2e}")
            print(f"  Max relative error: {debug_stats['accuracy']['max_rel_error']:.2e}")
            print(f"  99th percentile: {debug_stats['accuracy']['percentile_99_rel_error']:.2e}")
            print(f"\nPerformance:")
            print(f"  Speedup vs scipy: {debug_stats['timing']['speedup']:.1f}×")
    
    # Compute resolutions
    angular_res = args.angular_res or args.resolution
    radial_res = args.radial_res or args.resolution
    
    # Create black hole
    if not args.quiet:
        print(f"\nInitializing black hole...")
        print(f"  Mass: {args.mass}")
        print(f"  Inclination: {args.inclination} rad ({np.degrees(args.inclination):.1f}°)")
        print(f"  Accretion rate: {args.accretion}")
        print(f"  Resolution: {angular_res}×{radial_res}")
    
    start_total = time.time()
    
    bh = BlackHole(
        mass=args.mass,
        incl=args.inclination,
        acc=args.accretion,
        outer_edge=args.outer_edge,
        angular_resolution=angular_res,
        radial_resolution=radial_res,
        backend=backend_name,
        **backend_kwargs
    )
    
    init_time = time.time() - start_total
    
    # Render
    if not args.quiet:
        print(f"\nRendering...")
        print(f"  Using backend: {bh.backend_name}")
    
    start_render = time.time()
    
    # Create figure with polar projection for black hole visualization
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor(args.bg_color)
    ax.set_facecolor(args.bg_color)
    
    # Plot with color scheme
    if args.color_scheme == 'flux':
        bh.plot(ax=ax)
    else:
        bh.plot(ax=ax, cmap=args.color_scheme)
    
    # Clean layout
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    render_time = time.time() - start_render
    
    # Save
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight', 
               facecolor=args.bg_color, edgecolor='none')
    
    total_time = time.time() - start_total
    
    # Prepare output stats
    output_stats = {
        'backend': bh.backend_name,
        'parameters': {
            'mass': args.mass,
            'inclination_rad': args.inclination,
            'inclination_deg': float(np.degrees(args.inclination)),
            'accretion_rate': args.accretion,
            'outer_edge': args.outer_edge or 30 * args.mass,
            'angular_resolution': angular_res,
            'radial_resolution': radial_res,
        },
        'timing': {
            'init_time_s': init_time,
            'render_time_s': render_time,
            'total_time_s': total_time,
            'init_time_formatted': format_time(init_time),
            'render_time_formatted': format_time(render_time),
            'total_time_formatted': format_time(total_time),
        },
        'output': {
            'filename': args.output,
            'dpi': args.dpi,
        }
    }
    
    # Add debug stats if available
    if debug_stats:
        output_stats['accuracy'] = debug_stats['accuracy']
        output_stats['comparison'] = {
            'reference_backend': debug_stats['reference'],
            'speedup': debug_stats['timing']['speedup'],
        }
    
    # Save stats to file
    if args.debug:
        with open(args.debug_file, 'w') as f:
            json.dump(output_stats, f, indent=2)
        
        if not args.quiet:
            print(f"\n✓ Debug statistics saved to: {args.debug_file}")
    
    if not args.quiet:
        print(f"\n" + "=" * 70)
        print(f"✓ Rendering completed!")
        print(f"  Init time:   {format_time(init_time)}")
        print(f"  Render time: {format_time(render_time)}")
        print(f"  Total time:  {format_time(total_time)}")
        print(f"  Output: {args.output}")
        
        if debug_stats:
            speedup = debug_stats['timing']['speedup']
            if speedup > 1.5:
                print(f"  ⚡ {speedup:.1f}× faster than scipy!")
    
    if args.show:
        plt.show()
    
    return 0


if __name__ == '__main__':
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
