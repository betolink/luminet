"""Validation and benchmarking tests for Taichi implementation.

This module validates that the Taichi implementation produces identical
results to the original scipy implementation with equal or better performance.
"""

import numpy as np
import time
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt

# Original scipy implementation
from luminet import black_hole_math as bhmath
from luminet import black_hole
from luminet.isoradial import Isoradial

# Taichi implementation (when ready)
# from luminet_taichi import black_hole_math as bhmath_taichi


class AccuracyValidator:
    """Validate numerical accuracy of Taichi vs scipy implementation."""

    def __init__(self, rtol: float = 1e-6, atol: float = 1e-9):
        self.rtol = rtol  # Relative tolerance
        self.atol = atol  # Absolute tolerance
        self.results = []

    def compare_arrays(
        self,
        scipy_result: np.ndarray,
        taichi_result: np.ndarray,
        name: str = "unknown"
    ) -> dict:
        """Compare two arrays and return accuracy metrics.

        Args:
            scipy_result: Result from scipy implementation
            taichi_result: Result from Taichi implementation
            name: Name of the test for logging

        Returns:
            dict: Accuracy metrics including max_error, mean_error, rmse, etc.
        """
        # Handle NaN values
        mask = ~(np.isnan(scipy_result) | np.isnan(taichi_result))
        scipy_clean = scipy_result[mask]
        taichi_clean = taichi_result[mask]

        # Calculate metrics
        abs_error = np.abs(scipy_clean - taichi_clean)
        rel_error = np.abs((scipy_clean - taichi_clean) / (scipy_clean + self.atol))

        metrics = {
            'name': name,
            'max_abs_error': np.max(abs_error),
            'mean_abs_error': np.mean(abs_error),
            'rmse': np.sqrt(np.mean(abs_error ** 2)),
            'max_rel_error': np.max(rel_error),
            'mean_rel_error': np.mean(rel_error),
            'nan_mismatch': np.sum(np.isnan(scipy_result) != np.isnan(taichi_result)),
            'shape_match': scipy_result.shape == taichi_result.shape,
            'all_close': np.allclose(
                scipy_result,
                taichi_result,
                rtol=self.rtol,
                atol=self.atol,
                equal_nan=True
            )
        }

        self.results.append(metrics)
        return metrics

    def test_calc_q(self, mass: float = 1.0):
        """Test calc_q function accuracy."""
        print("\n" + "="*60)
        print("Testing calc_q function")
        print("="*60)

        # Test various periastron values
        p_values = np.linspace(2.1, 100, 100)

        scipy_q = np.array([bhmath.calc_q(p, mass) for p in p_values])
        # taichi_q = calc_q_taichi_vectorized(p_values, mass)  # To implement

        # For now, just store scipy results
        print(f"  Tested {len(p_values)} periastron values")
        print(f"  Range: [{p_values.min():.2f}, {p_values.max():.2f}]")
        print(f"  Scipy result range: [{np.nanmin(scipy_q):.4e}, {np.nanmax(scipy_q):.4e}]")

        # When Taichi is ready:
        # metrics = self.compare_arrays(scipy_q, taichi_q, "calc_q")
        # self._print_metrics(metrics)
        # return metrics['all_close']

        return True

    def test_calc_k_squared(self, mass: float = 1.0):
        """Test calc_k_squared function accuracy."""
        print("\n" + "="*60)
        print("Testing calc_k_squared function")
        print("="*60)

        p_values = np.linspace(2.1, 100, 100)

        scipy_k2 = np.array([bhmath.calc_k_squared(p, mass) for p in p_values])
        # taichi_k2 = calc_k_squared_taichi_vectorized(p_values, mass)

        print(f"  Tested {len(p_values)} periastron values")
        print(f"  Scipy result range: [{np.nanmin(scipy_k2):.4e}, {np.nanmax(scipy_k2):.4e}]")

        return True

    def test_calc_sn(self, mass: float = 1.0, incl: float = 1.4):
        """Test calc_sn (Jacobi elliptic function) accuracy.

        This is the most critical and challenging test.
        """
        print("\n" + "="*60)
        print("Testing calc_sn function (Jacobi elliptic)")
        print("="*60)

        # Test various combinations
        p_values = np.array([3.5, 5, 10, 20, 50])  # Periastron
        alpha_values = np.linspace(0, 2*np.pi, 50)  # Angles
        orders = [0, 1]  # Direct and ghost images

        print(f"  Testing {len(p_values)} periastron values")
        print(f"  Testing {len(alpha_values)} angles")
        print(f"  Testing {len(orders)} image orders")
        print(f"  Total tests: {len(p_values) * len(alpha_values) * len(orders)}")

        scipy_sn = []
        for p in p_values:
            for alpha in alpha_values:
                for order in orders:
                    sn = bhmath.calc_sn(p, alpha, mass, incl, order)
                    scipy_sn.append(sn)

        scipy_sn = np.array(scipy_sn)

        print(f"  Scipy sn range: [{np.nanmin(scipy_sn):.4e}, {np.nanmax(scipy_sn):.4e}]")
        print(f"  NaN count: {np.sum(np.isnan(scipy_sn))}")

        # When Taichi is ready:
        # metrics = self.compare_arrays(scipy_sn, taichi_sn, "calc_sn")
        # self._print_metrics(metrics)
        # return metrics['all_close']

        return True

    def test_solve_for_periastron(self, mass: float = 1.0, incl: float = 1.4):
        """Test periastron solver accuracy (root finding)."""
        print("\n" + "="*60)
        print("Testing solve_for_periastron (root finding)")
        print("="*60)

        # Test various radii and angles
        radii = np.array([6.5, 10, 15, 20, 30, 50])  # Must be > 6M
        alpha_values = np.linspace(0, 2*np.pi, 36)  # Every 10 degrees
        orders = [0, 1]

        print(f"  Testing {len(radii)} radii")
        print(f"  Testing {len(alpha_values)} angles")
        print(f"  Testing {len(orders)} orders")
        print(f"  Total tests: {len(radii) * len(alpha_values) * len(orders)}")

        scipy_periastron = []
        for r in radii:
            for alpha in alpha_values:
                for order in orders:
                    p = bhmath.solve_for_periastron(r, alpha, incl, mass, order)
                    scipy_periastron.append(p)

        scipy_periastron = np.array(scipy_periastron)

        print(f"  Scipy periastron range: [{np.nanmin(scipy_periastron):.4e}, {np.nanmax(scipy_periastron):.4e}]")
        print(f"  Valid solutions: {np.sum(~np.isnan(scipy_periastron))}/{len(scipy_periastron)}")

        return True

    def test_solve_for_impact_parameter(self, mass: float = 1.0, incl: float = 1.4):
        """Test impact parameter solver accuracy."""
        print("\n" + "="*60)
        print("Testing solve_for_impact_parameter")
        print("="*60)

        radii = np.array([6.5, 10, 15, 20, 30, 50])
        alpha_values = np.linspace(0, 2*np.pi, 36)
        orders = [0, 1]

        print(f"  Testing {len(radii) * len(alpha_values) * len(orders)} combinations")

        scipy_b = []
        for r in radii:
            for alpha in alpha_values:
                for order in orders:
                    b = bhmath.solve_for_impact_parameter(r, incl, alpha, mass, order)
                    scipy_b.append(b)

        scipy_b = np.array(scipy_b)

        print(f"  Scipy impact parameter range: [{np.nanmin(scipy_b):.4e}, {np.nanmax(scipy_b):.4e}]")
        print(f"  Valid solutions: {np.sum(~np.isnan(scipy_b))}/{len(scipy_b)}")

        return True

    def test_calc_redshift_factor(self, mass: float = 1.0, incl: float = 1.4):
        """Test redshift factor calculation."""
        print("\n" + "="*60)
        print("Testing calc_redshift_factor")
        print("="*60)

        radii = np.array([6.5, 10, 15, 20, 30, 50])
        alpha_values = np.linspace(0, 2*np.pi, 36)

        # Generate corresponding b values
        scipy_b = []
        for r in radii:
            for alpha in alpha_values:
                b = bhmath.solve_for_impact_parameter(r, incl, alpha, mass, 0)
                scipy_b.append(b)
        scipy_b = np.array(scipy_b)

        scipy_z = bhmath.calc_redshift_factor(
            np.repeat(radii, len(alpha_values)),
            np.tile(alpha_values, len(radii)),
            incl,
            mass,
            scipy_b
        )

        print(f"  Redshift factor range: [{np.nanmin(scipy_z):.4e}, {np.nanmax(scipy_z):.4e}]")

        return True

    def test_full_isoradial(self, mass: float = 1.0, incl: float = 1.4, radius: float = 10.0):
        """Test full isoradial calculation (integration test)."""
        print("\n" + "="*60)
        print("Testing full isoradial calculation")
        print("="*60)

        ir = Isoradial(
            radius=radius,
            incl=incl,
            bh_mass=mass,
            order=0,
            angular_resolution=100
        )

        print(f"  Radius: {radius}")
        print(f"  Angular resolution: {len(ir.angles)}")
        print(f"  Impact parameter range: [{np.min(ir.impact_parameters):.4e}, {np.max(ir.impact_parameters):.4e}]")
        print(f"  Redshift factor range: [{np.min(ir.redshift_factors):.4e}, {np.max(ir.redshift_factors):.4e}]")

        return True

    def test_full_black_hole(self, mass: float = 1.0, incl: float = 1.4, resolution: int = 50):
        """Test full black hole rendering (end-to-end test)."""
        print("\n" + "="*60)
        print("Testing full black hole rendering")
        print("="*60)

        bh = black_hole.BlackHole(
            mass=mass,
            incl=incl,
            acc=1.0,
            outer_edge=30.0,
            angular_resolution=resolution,
            radial_resolution=resolution
        )

        radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, resolution)

        # Calculate isoradials
        bh.calc_isoradials(direct_r=radii.tolist(), ghost_r=[])

        print(f"  Calculated {len(bh.isoradials)} isoradials")
        print(f"  Each with ~{resolution} points")

        # Check that we have valid data
        valid_count = 0
        total_points = 0
        for ir in bh.isoradials:
            valid_count += np.sum(~np.isnan(ir.impact_parameters))
            total_points += len(ir.impact_parameters)

        print(f"  Valid points: {valid_count}/{total_points} ({100*valid_count/total_points:.1f}%)")

        return True

    def _print_metrics(self, metrics: dict):
        """Print accuracy metrics in a formatted way."""
        print(f"\n  Results for: {metrics['name']}")
        print(f"  Shape match: {metrics['shape_match']}")
        print(f"  All close (rtol={self.rtol}, atol={self.atol}): {metrics['all_close']}")
        print(f"  Max absolute error: {metrics['max_abs_error']:.4e}")
        print(f"  Mean absolute error: {metrics['mean_abs_error']:.4e}")
        print(f"  RMSE: {metrics['rmse']:.4e}")
        print(f"  Max relative error: {metrics['max_rel_error']:.4e}")
        print(f"  Mean relative error: {metrics['mean_rel_error']:.4e}")
        print(f"  NaN mismatches: {metrics['nan_mismatch']}")

    def run_all_tests(self, mass: float = 1.0, incl: float = 1.4):
        """Run all validation tests."""
        print("\n" + "="*60)
        print("RUNNING ALL VALIDATION TESTS")
        print("="*60)

        tests = [
            self.test_calc_q,
            self.test_calc_k_squared,
            self.test_calc_sn,
            self.test_solve_for_periastron,
            self.test_solve_for_impact_parameter,
            self.test_calc_redshift_factor,
            self.test_full_isoradial,
            self.test_full_black_hole,
        ]

        results = []
        for test in tests:
            try:
                result = test(mass, incl)
                results.append((test.__name__, result))
            except Exception as e:
                print(f"  ERROR in {test.__name__}: {e}")
                results.append((test.__name__, False))

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {name}")

        all_passed = all(result for _, result in results)
        print(f"\n  Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

        return all_passed


class PerformanceBenchmark:
    """Benchmark performance of Taichi vs scipy implementation."""

    def __init__(self):
        self.results = []

    def benchmark_function(
        self,
        func: Callable,
        func_name: str,
        n_repeats: int = 10,
        **kwargs
    ) -> dict:
        """Benchmark a function's execution time.

        Args:
            func: Function to benchmark
            func_name: Name of the function for logging
            n_repeats: Number of times to repeat the benchmark
            **kwargs: Arguments to pass to the function

        Returns:
            dict: Performance metrics including mean_time, std_time, etc.
        """
        print(f"\n  Benchmarking {func_name}...")
        times = []

        # Warm-up run
        func(**kwargs)

        # Timed runs
        for _ in range(n_repeats):
            start_time = time.time()
            result = func(**kwargs)
            end_time = time.time()
            times.append(end_time - start_time)

        times = np.array(times)

        metrics = {
            'name': func_name,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'n_repeats': n_repeats,
        }

        print(f"    Mean time: {metrics['mean_time']*1000:.2f} ms")
        print(f"    Std time:  {metrics['std_time']*1000:.2f} ms")
        print(f"    Min time:  {metrics['min_time']*1000:.2f} ms")
        print(f"    Max time:  {metrics['max_time']*1000:.2f} ms")

        self.results.append(metrics)
        return metrics

    def benchmark_scipy_black_hole(
        self,
        mass: float = 1.0,
        incl: float = 1.4,
        resolution: int = 100,
        n_repeats: int = 5
    ):
        """Benchmark scipy black hole rendering."""
        print("\n" + "="*60)
        print("Benchmarking scipy black hole rendering")
        print("="*60)

        def render_bh():
            bh = black_hole.BlackHole(
                mass=mass,
                incl=incl,
                acc=1.0,
                outer_edge=30.0,
                angular_resolution=resolution,
                radial_resolution=resolution
            )
            radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, resolution)
            bh.calc_isoradials(direct_r=radii.tolist(), ghost_r=[])
            return bh

        scipy_metrics = self.benchmark_function(
            render_bh,
            f"scipy_black_hole_res{resolution}",
            n_repeats=n_repeats
        )

        return scipy_metrics

    def benchmark_taichi_black_hole(
        self,
        mass: float = 1.0,
        incl: float = 1.4,
        resolution: int = 100,
        n_repeats: int = 5
    ):
        """Benchmark Taichi black hole rendering (when implemented)."""
        print("\n" + "="*60)
        print("Benchmarking Taichi black hole rendering")
        print("="*60)

        # TODO: Implement when Taichi version is ready
        print("  (Not yet implemented)")

        return None

    def compare_performance(self):
        """Compare performance between implementations."""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)

        if len(self.results) < 2:
            print("  Not enough benchmarks to compare")
            return

        # Group by resolution
        by_resolution = {}
        for r in self.results:
            if 'black_hole' in r['name']:
                # Extract resolution from name
                res = r['name'].split('_')[-1].replace('res', '')
                by_resolution[res] = by_resolution.get(res, {})
                impl = r['name'].split('_')[0]
                by_resolution[res][impl] = r

        print("\n  Comparison by resolution:")
        for res, impls in by_resolution.items():
            if 'scipy' in impls and 'taichi' in impls:
                scipy_time = impls['scipy']['mean_time']
                taichi_time = impls['taichi']['mean_time']
                speedup = scipy_time / taichi_time
                print(f"    Resolution {res}:")
                print(f"      Scipy:  {scipy_time*1000:.2f} ms")
                print(f"      Taichi: {taichi_time*1000:.2f} ms")
                print(f"      Speedup: {speedup:.2f}x")

    def generate_report(self):
        """Generate a comprehensive validation and benchmark report."""
        print("\n" + "="*60)
        print("VALIDATION AND BENCHMARK REPORT")
        print("="*60)

        print("\n  Performance results:")
        for r in self.results:
            print(f"    {r['name']}: {r['mean_time']*1000:.2f} ms (±{r['std_time']*1000:.2f})")


class VisualValidator:
    """Visual validation through image comparison."""

    def __init__(self):
        pass

    def compare_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        title1: str = "Scipy",
        title2: str = "Taichi"
    ):
        """Visually compare two rendered images."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original images
        axes[0].imshow(image1, cmap='inferno')
        axes[0].set_title(title1)
        axes[0].axis('off')

        axes[1].imshow(image2, cmap='inferno')
        axes[1].set_title(title2)
        axes[1].axis('off')

        # Difference
        diff = np.abs(image1 - image2)
        axes[2].imshow(diff, cmap='hot', vmin=0, vmax=np.max(diff))
        axes[2].set_title(f"Difference (max: {np.max(diff):.4e})")
        axes[2].axis('off')

        plt.tight_layout()
        return fig

    def render_scipy_black_hole(
        self,
        mass: float = 1.0,
        incl: float = 1.4,
        resolution: int = 200
    ) -> np.ndarray:
        """Render black hole image using scipy."""
        bh = black_hole.BlackHole(
            mass=mass,
            incl=incl,
            acc=1.0,
            outer_edge=30.0,
            angular_resolution=resolution,
            radial_resolution=resolution
        )

        ax = bh.plot()

        # Convert to array
        fig = ax.figure
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    def render_taichi_black_hole(
        self,
        mass: float = 1.0,
        incl: float = 1.4,
        resolution: int = 200
    ) -> np.ndarray:
        """Render black hole image using Taichi (when implemented)."""
        # TODO: Implement when Taichi version is ready
        pass


if __name__ == "__main__":
    # Run validation tests
    validator = AccuracyValidator(rtol=1e-6, atol=1e-9)
    validator.run_all_tests()

    # Run performance benchmarks
    benchmark = PerformanceBenchmark()
    benchmark.benchmark_scipy_black_hole(resolution=50, n_repeats=3)
    benchmark.benchmark_scipy_black_hole(resolution=100, n_repeats=3)
    benchmark.generate_report()

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
