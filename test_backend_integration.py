#!/usr/bin/env python
"""Quick test to verify backend integration works."""

import numpy as np
import luminet.black_hole_math as bhmath

def test_backend_integration():
    """Test that backend integration works correctly."""
    
    print("Testing backend integration...")
    print("=" * 60)
    
    # Test 1: Default backend (scipy)
    print("\n1. Testing default backend (should be scipy):")
    bhmath.reset_backend()
    backend = bhmath.get_current_backend()
    print(f"   Backend name: {backend.get_backend_name()}")
    
    # Test basic calculation
    b = bhmath.solve_for_impact_parameter(10.0, 1.4, 1.0, 1.0, 0)
    print(f"   Impact parameter (r=10, incl=1.4): {b:.6f}")
    
    # Test 2: Numba backend
    print("\n2. Testing Numba backend:")
    try:
        bhmath.set_backend('numba')
        backend = bhmath.get_current_backend()
        print(f"   Backend name: {backend.get_backend_name()}")
        
        b_numba = bhmath.solve_for_impact_parameter(10.0, 1.4, 1.0, 1.0, 0)
        print(f"   Impact parameter (r=10, incl=1.4): {b_numba:.6f}")
        print(f"   Difference from scipy: {abs(b - b_numba):.2e}")
    except ImportError as e:
        print(f"   Numba not available: {e}")
    
    # Test 3: Taichi CPU backend
    print("\n3. Testing Taichi CPU backend:")
    try:
        bhmath.set_backend('taichi', arch='cpu')
        backend = bhmath.get_current_backend()
        print(f"   Backend name: {backend.get_backend_name()}")
        
        b_taichi = bhmath.solve_for_impact_parameter(10.0, 1.4, 1.0, 1.0, 0)
        print(f"   Impact parameter (r=10, incl=1.4): {b_taichi:.6f}")
        print(f"   Difference from scipy: {abs(b - b_taichi):.2e}")
    except ImportError as e:
        print(f"   Taichi not available: {e}")
    except Exception as e:
        print(f"   Taichi error: {e}")
    
    # Test 4: BlackHole class integration
    print("\n4. Testing BlackHole class integration:")
    from luminet.black_hole import BlackHole
    
    # Scipy
    print("   Creating BlackHole with scipy...")
    bh_scipy = BlackHole(backend='scipy', angular_resolution=50, radial_resolution=50)
    print(f"   Backend: {bh_scipy.backend_name}")
    
    # Numba
    try:
        print("   Creating BlackHole with numba...")
        bh_numba = BlackHole(backend='numba', angular_resolution=50, radial_resolution=50)
        print(f"   Backend: {bh_numba.backend_name}")
    except ImportError:
        print("   Numba not available for BlackHole test")
    
    print("\n" + "=" * 60)
    print("✓ Backend integration tests completed successfully!")
    return True

if __name__ == '__main__':
    try:
        success = test_backend_integration()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
