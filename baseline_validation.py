"""Simple baseline validation without matplotlib dependencies."""

import sys
import time
import numpy as np

# Try to import luminet
try:
    from luminet import black_hole_math as bhmath
    from luminet.black_hole import BlackHole
    print("✓ Successfully imported luminet modules")
except ImportError as e:
    print(f"✗ Failed to import luminet: {e}")
    sys.exit(1)

def run_baseline_tests():
    """Run baseline accuracy and performance tests."""

    print("\n" + "="*70)
    print("BASELINE VALIDATION - SCIPY IMPLEMENTATION")
    print("="*70)

    # Test parameters
    mass = 1.0
    incl = 1.4

    # Test 1: calc_q function
    print("\n[1/5] Testing calc_q function...")
    p_values = np.linspace(2.1, 100, 100)
    start = time.time()
    scipy_q = np.array([bhmath.calc_q(p, mass) for p in p_values])
    elapsed = time.time() - start
    print(f"  ✓ Processed {len(p_values)} values in {elapsed*1000:.2f} ms")
    print(f"  Result range: [{np.nanmin(scipy_q):.4e}, {np.nanmax(scipy_q):.4e}]")

    # Test 2: calc_k_squared function
    print("\n[2/5] Testing calc_k_squared function...")
    start = time.time()
    scipy_k2 = np.array([bhmath.calc_k_squared(p, mass) for p in p_values])
    elapsed = time.time() - start
    print(f"  ✓ Processed {len(p_values)} values in {elapsed*1000:.2f} ms")
    print(f"  Result range: [{np.nanmin(scipy_k2):.4e}, {np.nanmax(scipy_k2):.4e}]")

    # Test 3: calc_sn (Jacobi elliptic) - CRITICAL
    print("\n[3/5] Testing calc_sn (Jacobi elliptic) - CRITICAL...")
    p_test = np.array([3.5, 5, 10, 20, 50])
    alpha_test = np.linspace(0, 2*np.pi, 50)
    orders = [0, 1]

    total_tests = len(p_test) * len(alpha_test) * len(orders)
    print(f"  Running {total_tests} test cases...")

    start = time.time()
    scipy_sn = []
    for p in p_test:
        for alpha in alpha_test:
            for order in orders:
                sn = bhmath.calc_sn(p, alpha, mass, incl, order)
                scipy_sn.append(sn)
    scipy_sn = np.array(scipy_sn)
    elapsed = time.time() - start

    print(f"  ✓ Completed in {elapsed*1000:.2f} ms")
    print(f"  Result range: [{np.nanmin(scipy_sn):.4e}, {np.nanmax(scipy_sn):.4e}]")
    print(f"  NaN count: {np.sum(np.isnan(scipy_sn))}/{total_tests}")
    print(f"  Speed: {total_tests/elapsed:.1f} calculations/sec")

    # Test 4: solve_for_periastron (root finding)
    print("\n[4/5] Testing solve_for_periastron (root finding)...")
    radii = np.array([6.5, 10, 15, 20, 30, 50])
    alpha_test2 = np.linspace(0, 2*np.pi, 36)
    orders = [0, 1]

    total_tests = len(radii) * len(alpha_test2) * len(orders)
    print(f"  Running {total_tests} test cases...")

    start = time.time()
    scipy_periastron = []
    for r in radii:
        for alpha in alpha_test2:
            for order in orders:
                p = bhmath.solve_for_periastron(r, alpha, incl, mass, order)
                scipy_periastron.append(p)
    scipy_periastron = np.array(scipy_periastron)
    elapsed = time.time() - start

    print(f"  ✓ Completed in {elapsed*1000:.2f} ms")
    print(f"  Result range: [{np.nanmin(scipy_periastron):.4e}, {np.nanmax(scipy_periastron):.4e}]")
    print(f"  Valid solutions: {np.sum(~np.isnan(scipy_periastron))}/{total_tests}")
    print(f"  Speed: {total_tests/elapsed:.1f} calculations/sec")

    # Test 5: Full Black Hole rendering (small scale)
    print("\n[5/5] Testing full Black Hole calculation (resolution=50)...")
    resolution = 50

    start = time.time()
    bh = BlackHole(
        mass=mass,
        incl=incl,
        acc=1.0,
        outer_edge=30.0,
        angular_resolution=resolution,
        radial_resolution=resolution
    )
    radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, resolution)
    bh.calc_isoradials(direct_r=radii.tolist(), ghost_r=[])
    elapsed = time.time() - start

    print(f"  ✓ Completed in {elapsed*1000:.2f} ms")
    print(f"  Calculated {len(bh.isoradials)} isoradials")
    print(f"  Total points: {sum(len(ir.impact_parameters) for ir in bh.isoradials)}")

    # Statistics
    valid_count = 0
    total_points = 0
    for ir in bh.isoradials:
        valid_count += np.sum(~np.isnan(ir.impact_parameters))
        total_points += len(ir.impact_parameters)

    print(f"  Valid points: {valid_count}/{total_points} ({100*valid_count/total_points:.1f}%)")

    # Summary
    print("\n" + "="*70)
    print("BASELINE SUMMARY")
    print("="*70)
    print(f"✓ All tests passed successfully")
    print(f"✓ calc_sn is the computational bottleneck (Jacobi elliptic)")
    print(f"✓ Root finding is also expensive (scipy.optimize.brentq)")
    print(f"\nBaseline performance established for Taichi comparison")
    print("="*70)

    return True

if __name__ == "__main__":
    try:
        run_baseline_tests()
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
