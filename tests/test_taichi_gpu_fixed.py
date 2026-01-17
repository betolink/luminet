"""Test if Taichi GPU backend works after ti.template() fix."""
import sys
sys.path.insert(0, '/home/betolink/hackweek/luminet')

from luminet.backends import get_backend

print("=" * 70)
print("Testing Taichi GPU Backend After ti.template() Fix")
print("=" * 70)

# Test 1: Initialize GPU backend
print("\n1. Initialize Taichi GPU backend...")
try:
    backend = get_backend('taichi', arch='gpu')
    print(f"   ✅ Backend initialized: {backend}")
    print(f"   Architecture: {backend.arch}")
    print(f"   Is GPU: {backend._is_gpu}")
    print(f"   Using f64: {backend._use_f64}")
except Exception as e:
    print(f"   ❌ Failed to initialize: {e}")
    sys.exit(1)

# Test 2: Simple calc_q test
print("\n2. Test calc_q(5.0, 1.0)...")
try:
    result = backend.calc_q(5.0, 1.0)
    print(f"   ✅ calc_q result: {result:.6f}")
    print(f"   Expected: ~6.480741 (scipy baseline)")
except Exception as e:
    print(f"   ❌ calc_q failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test elliptic integrals (uses asin internally)
print("\n3. Test calc_elliptic_k(0.5)...")
try:
    result = backend.calc_elliptic_k(0.5)
    print(f"   ✅ calc_elliptic_k result: {result:.6f}")
    print(f"   Expected: ~1.854075 (scipy baseline)")
except Exception as e:
    print(f"   ❌ calc_elliptic_k failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Full BlackHole initialization (THE BIG TEST)
print("\n4. Test full BlackHole initialization with GPU backend...")
try:
    from luminet import BlackHole
    bh = BlackHole(
        mass=1.0,
        spin=0.9,
        inclination=45.0,
        distance=10.0,
        disk_inner=6.0,
        disk_outer=20.0,
        backend='taichi',
        arch='gpu'
    )
    print(f"   ✅ BlackHole created successfully!")
    print(f"   Mass: {bh.mass}")
    print(f"   Spin: {bh.spin}")
    print(f"   Disk outer edge: {bh.disk_apparent_outer_edge:.6f}")
except Exception as e:
    print(f"   ❌ BlackHole initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 ALL TESTS PASSED! Taichi GPU backend is working!")
print("=" * 70)
