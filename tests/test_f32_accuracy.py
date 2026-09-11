"""
Test f32 vs f64 precision accuracy for black hole calculations.

This test measures the accuracy loss when using 32-bit floating point
precision compared to 64-bit baseline (scipy). We focus purely on
accuracy, not performance.

The test uses CPU-based calculations to isolate precision effects
from hardware limitations.
"""

import numpy as np
from luminet.backends import get_backend
import sys


def test_single_point_accuracy():
    """Test accuracy on individual test cases."""
    print("=" * 70)
    print("F32 vs F64 PRECISION ACCURACY TEST")
    print("=" * 70)
    print()
    print("Initializing backends...")
    
    # Get backends
    scipy_backend = get_backend('scipy')
    print(f"✓ Scipy baseline: {scipy_backend.get_backend_name()}")
    
    # Try to get Taichi CPU backend with f32
    try:
        # Force CPU mode and use default precision (should try f32 first)
        taichi_backend = get_backend('taichi', arch='cpu')
        print(f"✓ Taichi CPU: {taichi_backend.get_backend_name()}")
        print(f"  - Precision: {'f64' if taichi_backend._use_f64 else 'f32'}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize Taichi: {e}")
        return
    
    # Test cases covering different parameter ranges
    test_cases = [
        {
            'name': 'Standard case',
            'radius': 10.0,
            'incl': 1.4,
            'alpha': 1.0,
            'bh_mass': 1.0,
            'order': 0
        },
        {
            'name': 'Small radius (near horizon)',
            'radius': 3.0,
            'incl': 1.4,
            'alpha': 0.5,
            'bh_mass': 1.0,
            'order': 0
        },
        {
            'name': 'Large radius',
            'radius': 100.0,
            'incl': 1.4,
            'alpha': 2.0,
            'bh_mass': 1.0,
            'order': 0
        },
        {
            'name': 'High inclination',
            'radius': 10.0,
            'incl': 1.5,
            'alpha': 1.0,
            'bh_mass': 1.0,
            'order': 0
        },
        {
            'name': 'Low inclination',
            'radius': 10.0,
            'incl': 0.1,
            'alpha': 1.0,
            'bh_mass': 1.0,
            'order': 0
        },
        {
            'name': 'Extreme parameters',
            'radius': 5.0,
            'incl': 1.55,
            'alpha': 0.1,
            'bh_mass': 1.0,
            'order': 1
        },
    ]
    
    print("-" * 70)
    print("TEST 1: Single Point Accuracy")
    print("-" * 70)
    print()
    print(f"{'Case':<30} {'Scipy (f64)':<16} {'Taichi':<16} {'Rel Error':<12} {'Status'}")
    print("-" * 70)
    
    max_error = 0.0
    errors = []
    
    for case in test_cases:
        name = case['name']
        radius = case['radius']
        incl = case['incl']
        alpha = case['alpha']
        bh_mass = case['bh_mass']
        order = case['order']
        
        # Compute with scipy (f64 baseline)
        scipy_result = scipy_backend.solve_for_impact_parameter(
            radius, incl, alpha, bh_mass, order
        )
        
        # Compute with Taichi
        taichi_result = taichi_backend.solve_for_impact_parameter(
            radius, incl, alpha, bh_mass, order
        )
        
        # Calculate relative error
        if abs(scipy_result) > 1e-10:
            rel_error = abs(scipy_result - taichi_result) / abs(scipy_result)
        else:
            rel_error = abs(scipy_result - taichi_result)
        
        errors.append(rel_error)
        max_error = max(max_error, rel_error)
        
        # Status indicator
        if rel_error < 1e-6:
            status = "✓ Excellent"
        elif rel_error < 1e-4:
            status = "✓ Good"
        elif rel_error < 1e-2:
            status = "⚠ Acceptable"
        else:
            status = "✗ Poor"
        
        print(f"{name:<30} {scipy_result:<16.10f} {taichi_result:<16.10f} {rel_error:<12.2e} {status}")
    
    print("-" * 70)
    print()
    print(f"Maximum relative error: {max_error:.2e}")
    print(f"Mean relative error:    {np.mean(errors):.2e}")
    print(f"Median relative error:  {np.median(errors):.2e}")
    print()


def test_batch_accuracy():
    """Test accuracy on a batch of points."""
    print("-" * 70)
    print("TEST 2: Batch Accuracy Statistics")
    print("-" * 70)
    print()
    
    # Get backends
    scipy_backend = get_backend('scipy')
    taichi_backend = get_backend('taichi', arch='cpu')
    
    # Create test grid
    n = 50
    radii = np.linspace(3.5, 20.0, n)
    alphas = np.linspace(0.1, 2.0, n)
    
    print(f"Testing on {n}×{n} = {n*n} points...")
    print()
    
    scipy_results = np.zeros((n, n))
    taichi_results = np.zeros((n, n))
    
    # Compute with scipy
    for i, radius in enumerate(radii):
        for j, alpha in enumerate(alphas):
            try:
                scipy_results[i, j] = scipy_backend.solve_for_impact_parameter(
                    radius, 1.4, alpha, 1.0, 0
                )
            except Exception as e:
                print(f"  Warning: scipy failed at radius={radius:.2f}, alpha={alpha:.2f}: {e}")
                scipy_results[i, j] = np.nan
    
    # Compute with Taichi
    for i, radius in enumerate(radii):
        for j, alpha in enumerate(alphas):
            try:
                taichi_results[i, j] = taichi_backend.solve_for_impact_parameter(
                    radius, 1.4, alpha, 1.0, 0
                )
            except Exception as e:
                print(f"  Warning: taichi failed at radius={radius:.2f}, alpha={alpha:.2f}: {e}")
                taichi_results[i, j] = np.nan
    
    # Calculate errors
    abs_errors = np.abs(scipy_results - taichi_results)
    rel_errors = abs_errors / (np.abs(scipy_results) + 1e-15)
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(scipy_results) | np.isnan(taichi_results))
    abs_errors_valid = abs_errors[valid_mask]
    rel_errors_valid = rel_errors[valid_mask]
    
    n_valid = np.sum(valid_mask)
    n_total = scipy_results.size
    n_failed = n_total - n_valid
    
    print(f"Precision: {'f64' if taichi_backend._use_f64 else 'f32'}")
    print(f"Backend:   {taichi_backend.get_backend_name()}")
    print(f"Valid points: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    if n_failed > 0:
        print(f"Failed points: {n_failed} ({100*n_failed/n_total:.1f}%)")
    print()
    
    if n_valid > 0:
        print("Absolute Error Statistics:")
        print(f"  Mean:   {np.mean(abs_errors_valid):.6e}")
        print(f"  Median: {np.median(abs_errors_valid):.6e}")
        print(f"  Max:    {np.max(abs_errors_valid):.6e}")
        print(f"  Min:    {np.min(abs_errors_valid):.6e}")
        print(f"  Std:    {np.std(abs_errors_valid):.6e}")
        print()
        print("Relative Error Statistics:")
        print(f"  Mean:   {np.mean(rel_errors_valid):.6e}")
        print(f"  Median: {np.median(rel_errors_valid):.6e}")
        print(f"  Max:    {np.max(rel_errors_valid):.6e}")
        print(f"  Min:    {np.min(rel_errors_valid):.6e}")
        print(f"  Std:    {np.std(rel_errors_valid):.6e}")
        print()
        print("Percentiles (relative error):")
        for p in [50, 75, 90, 95, 99, 99.9]:
            print(f"  {p:5.1f}%: {np.percentile(rel_errors_valid, p):.6e}")
        print()
    else:
        print("  ERROR: No valid points computed!")
        print()


def test_elliptic_functions():
    """Test accuracy of the taichi elliptic integrals vs the scipy (f64) reference.

    The taichi backend ships native GPU-compatible elliptic kernels. We drive the
    scalar-only ones (complete integral K(m) and Jacobi sn(u,m)) through small test
    kernels and compare against scipy.special, the f64 baseline.

    Note: the backend's incomplete-integral kernel (F) and its ``calc_sn`` method use
    a physics argument signature (impact parameter / angle / mass); they are covered
    by ``test_single_point_accuracy`` / ``test_batch_accuracy`` rather than here.
    """
    import taichi as ti
    import scipy.special as sp
    from luminet.backends.taichi_backend import ti_ellipk_agm, ti_ellipj_sn

    # f64 baseline so the comparison measures algorithmic agreement, not precision.
    ti.init(arch=ti.cpu, default_fp=ti.f64, log_level="error")

    @ti.kernel
    def _K(m: ti.types.ndarray(), out: ti.types.ndarray()):
        for i in range(m.shape[0]):
            out[i] = ti_ellipk_agm(m[i])

    @ti.kernel
    def _Sn(u: ti.types.ndarray(), mm: ti.types.ndarray(), out: ti.types.ndarray()):
        for i in range(u.shape[0]):
            out[i] = ti_ellipj_sn(u[i], mm[i])

    print("-" * 70)
    print("TEST 3: Elliptic Function Accuracy (taichi native vs scipy.special)")
    print("-" * 70)
    print()

    # --- Complete elliptic integral K(m) -------------------------------------
    print("Complete Elliptic Integral K(m):")
    print(f"{'m':<8} {'Scipy (f64)':<18} {'Taichi':<18} {'Rel Error':<12}")
    print("-" * 56)

    m_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    m_arr = np.array(m_values, dtype=np.float64)
    taichi_k = np.zeros_like(m_arr)
    _K(m_arr, taichi_k)
    scipy_k = sp.ellipk(m_arr)
    k_rel = np.abs(taichi_k - scipy_k) / np.abs(scipy_k)
    for mv, sk, tk, re in zip(m_values, scipy_k, taichi_k, k_rel):
        print(f"{mv:<8.2f} {sk:<18.12f} {tk:<18.12f} {re:<12.2e}")
    print(f"\nK(m) - Max relative error: {np.max(k_rel):.2e}")
    assert np.all(np.isfinite(taichi_k)), "taichi K(m) returned non-finite values"
    assert np.max(k_rel) < 1e-6, f"K(m) rel error too large: {np.max(k_rel):.2e}"

    # --- Jacobi elliptic function sn(u, m) -----------------------------------
    print("\nJacobi Elliptic sn(u, m):")
    print(f"{'u':<8} {'m':<8} {'Scipy (f64)':<18} {'Taichi':<18} {'Rel Error':<12}")
    print("-" * 56)

    u_values = [0.5, 1.0, 1.5]
    m_values_sn = [0.2, 0.5, 0.8]
    u_arr = np.array(u_values, dtype=np.float64)
    m_arr_sn = np.array(m_values_sn, dtype=np.float64)
    taichi_sn = np.zeros(len(u_values))
    _Sn(u_arr, m_arr_sn, taichi_sn)
    # scipy.special.ellipj returns (sn, cn, dn, ph); sn is index 0.
    scipy_sn = sp.ellipj(u_arr, m_arr_sn)[0]
    sn_rel = np.abs(taichi_sn - scipy_sn) / np.abs(scipy_sn)
    for uv, mv, ss, ts, re in zip(u_values, m_values_sn, scipy_sn, taichi_sn, sn_rel):
        print(f"{uv:<8.2f} {mv:<8.2f} {ss:<18.12f} {ts:<18.12f} {re:<12.2e}")
    print(f"\nsn(u,m) - Max relative error: {np.max(sn_rel):.2e}")
    assert np.all(np.isfinite(taichi_sn)), "taichi sn(u,m) returned non-finite values"
    assert np.max(sn_rel) < 1e-6, f"sn(u,m) rel error too large: {np.max(sn_rel):.2e}"

    ti.reset()
    print("\nAll elliptic-function checks passed.")


def print_summary():
    """Print final summary and recommendations."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    taichi_backend = get_backend('taichi', arch='cpu')
    precision = 'f64' if taichi_backend._use_f64 else 'f32'
    
    print(f"Taichi Backend Precision: {precision}")
    print()
    
    if precision == 'f32':
        print("F32 PRECISION LIMITATIONS:")
        print("  - ~6-7 significant decimal digits")
        print("  - Relative errors typically 1e-6 to 1e-7")
        print("  - Errors accumulate in iterative algorithms (AGM, bisection)")
        print("  - May cause numerical instability near singularities")
        print()
        print("RECOMMENDATION:")
        print("  ✗ NOT suitable for scientific computing")
        print("  ✗ NOT suitable for accurate black hole physics")
        print("  ? Possibly acceptable for visualization (if errors < 1%)")
        print()
        print("PREFER:")
        print("  ✓ Use 'numba' backend (f64, 10× faster than scipy)")
        print("  ✓ Use 'taichi' with arch='cpu' (should use f64 by default)")
        print("  ✓ Use 'scipy' backend (f64, reference implementation)")
    else:
        print("F64 PRECISION:")
        print("  - ~15-17 significant decimal digits")
        print("  - Machine precision for double-precision arithmetic")
        print("  - Suitable for scientific computing")
        print()
        print("RECOMMENDATION:")
        print("  ✓ Suitable for all black hole physics calculations")
        print("  ✓ Taichi CPU backend provides good performance with f64")
        print("  ✓ Numba backend recommended for best performance")
    print()


if __name__ == '__main__':
    try:
        test_single_point_accuracy()
        test_batch_accuracy()
        print_summary()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
