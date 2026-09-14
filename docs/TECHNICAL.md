# Technical Documentation

This document covers technical aspects of the luminet library implementation, including precision analysis, backend architecture, and performance characteristics.

## Table of Contents

- [Precision Analysis](#precision-analysis)
- [Backend Architecture](#backend-architecture)
- [Numerical Methods](#numerical-methods)
- [Performance Considerations](#performance-considerations)

---

## Precision Analysis

### Overview

Luminet simulates black hole accretion disks using numerical methods that require careful handling of floating-point precision. The codebase supports both f32 (single-precision, 32-bit) and f64 (double-precision, 64-bit) floating-point arithmetic.

### Precision Comparison

| Aspect | F32 (Single Precision) | F64 (Double Precision) |
|---------|------------------------|------------------------|
| **Significand bits** | 23 bits | 52 bits |
| **Decimal digits** | ~6-7 | ~15-17 |
| **Machine epsilon** | ~1.2e-7 | ~2.2e-16 |
| **Typical error** | 1e-6 to 1e-7 | 1e-15 to 1e-16 |
| **Range (exponent)** | ±10^±38 | ±10^±308 |
| **Memory per value** | 4 bytes | 8 bytes |
| **GPU support** | Universal | Professional GPUs only |

### Why F64 is Required

**F64 (double precision) is the default** for black hole physics calculations due to:

1. **Iterative algorithms where errors accumulate**:
   - Arithmetic-Geometric Mean (AGM) for elliptic integrals
   - Bisection and Brent's method for root finding
   - Descending Landen transformations

2. **Sensitivity near singularities**:
   - Photon sphere (r ≈ 3M): Light rays extremely bent
   - ISCO (r ≈ 6M): Disk inner edge, high curvature
   - Extreme inclinations (i → π/2): Edge-on view, maximum lensing

3. **Elliptic integral calculations**:
   - K(m) = ∫₀^(π/2) dt / √(1 - m·sin²(t))
   - F(φ,m) = ∫₀^φ dt / √(1 - m·sin²(t))
   - Requires high precision for convergence

### Error Propagation

For a typical calculation (r=10M, α=1.0, i=1.4 rad):

| Step | F32 Error | F64 Error | Error Growth |
|------|-----------|-----------|--------------|
| Compute k²(p) | 1e-7 | 1e-15 | Base error |
| Compute K(k²) via AGM | 5e-7 | 5e-15 | 5× (10 iterations) |
| Compute F(ζ,k²) via quadrature | 1e-5 | 1e-13 | 20× (20 terms) |
| Compute sn(u,k²) | 5e-5 | 5e-13 | 5× (backward recurrence) |
| Solve for p via Brent's method | **2e-4** | **2e-12** | 4× (bisection steps) |

**Final Error**:
- **F32**: ~0.02% relative error (unacceptable for science)
- **F64**: ~0.0000002% relative error (machine precision)

### GPU Precision Considerations

**Taichi GPU with f32** works for visualization purposes:
- Accuracy: ~1e-6 to 1e-8 relative error
- Use case: Real-time video generation, educational demos
- **Not recommended** for: Scientific publications, research

**When to use f32**:
- Real-time visualization where speed > accuracy
- GPU acceleration on consumer hardware (no f64 support)
- Educational demonstrations where artifacts are acceptable

**When to use f64**:
- Scientific research and publications
- Accuracy validation and verification
- All production-grade rendering

### Memory and Performance

**Memory Usage** (for 1000×1000 image grid):
- **F32**: 1000×1000×4 bytes = 4 MB
- **F64**: 1000×1000×8 bytes = 8 MB
- **Impact**: Negligible on modern systems

**CPU Performance**:

| Operation | F32 | F64 | Ratio |
|-----------|-----|-----|-------|
| Addition | 0.3 ns | 0.3 ns | 1.0× |
| Multiplication | 0.3 ns | 0.3 ns | 1.0× |
| Division | 3 ns | 3 ns | 1.0× |
| `sqrt()` | 5 ns | 6 ns | 1.2× |
| `sin()`/`cos()` | 15 ns | 20 ns | 1.3× |

**GPU Performance**:

| GPU Type | F32 Throughput | F64 Throughput | Ratio |
|----------|----------------|----------------|-------|
| Consumer (RTX 3090) | 35.6 TFLOPS | 0.55 TFLOPS | 64× slower |
| Professional (A100) | 19.5 TFLOPS | 9.7 TFLOPS | 2× slower |
| Consumer (AMD RADV) | ~10 TFLOPS | **Not available** | ∞ |

**Conclusion**: Consumer GPUs are not viable for f64 black hole physics.

---

## Backend Architecture

### Overview

Luminet uses a **pluggable backend system** that allows choosing between different computational implementations:

```
luminet/
├── backends/
│   ├── __init__.py              # Factory functions and exports
│   ├── base.py                  # Abstract base class
│   ├── scipy_backend.py          # Original implementation
│   ├── taichi_backend.py        # GPU-accelerated (CPU + GPU)
│   ├── jax_backend.py           # JAX with auto-vectorization
│   └── numba_backend.py         # Numba JIT compilation
```

### Available Backends

| Backend | Status | GPU | Vectorization | Precision |
|---------|---------|------|--------------|-----------|
| **scipy** | ✅ Production | No | No | f64 |
| **taichi** | ✅ Production | Yes (CPU) / Yes (GPU) | Yes | f64 (CPU) / f32 (GPU) |
| **jax** | ✅ Production | Yes (CPU) | Yes (GPU, CUDA only) | f64 |
| **numba** | ✅ Production | No | Yes | f64 |

### Backend Selection

**Python API**:
```python
from luminet.black_hole import BlackHole

# Default scipy backend
bh = BlackHole(mass=1.0, incl=1.4)

# Numba backend (4.7× faster)
bh = BlackHole(mass=1.0, incl=1.4, backend='numba')

# Taichi CPU backend (9× faster)
bh = BlackHole(mass=1.0, incl=1.4, backend='taichi')

# Taichi GPU backend (up to 50× faster)
bh = BlackHole(mass=1.0, incl=1.4, backend='taichi', arch='gpu')
```

**Command Line** (render.py):
```bash
# Scipy (baseline)
python render.py --backend=scipy

# Numba (recommended for CPU)
python render.py --backend=numba

# Taichi CPU
python render.py --backend=taichi --hw=cpu

# Taichi GPU (auto-detects CUDA/Vulkan)
python render.py --backend=taichi --hw=gpu
```

### Backend Implementation Details

#### Scipy Backend
- **Implementation**: Wraps existing scipy functions
- **Functions**: `brentq`, `ellipj`, `ellipe`, `solve_ivp`
- **Performance**: Baseline (1×)
- **Precision**: f64 (double)
- **Use case**: Maximum compatibility, reference implementation

#### Numba Backend
- **Implementation**: JIT-compiled Python with `@njit` decorator
- **Functions**: All backends functions compiled with Numba
- **Performance**: 4-7× faster than scipy
- **Precision**: f64 (double)
- **Use case**: CPU acceleration, interactive rendering

#### Taichi Backend
- **Implementation**: Taichi JIT-compiled kernels
- **GPU Support**:
  - CUDA (NVIDIA GPUs) - both f32 and f64
  - Vulkan (AMD/Intel/NVIDIA) - f32 only
- **CPU Support**: f32 and f64
- **Performance**:
  - CPU: 4-10× faster than scipy
  - GPU (CUDA): 10-50× faster than scipy
  - GPU (Vulkan): 4-11× faster than scipy
- **Precision**: f32 (GPU) / f64 (CPU)
- **Use case**: Real-time video generation, large-scale renders

**Taichi GPU Precision Notes**:
- Uses `ti.template()` for generic type support
- Automatically detects GPU capabilities:
  1. Try CUDA f32 (most compatible)
  2. Try CUDA f64 (professional GPUs)
  3. Try Vulkan f32 (AMD/Intel)
  4. Try Vulkan f64 (rare)
  5. Fallback to CPU f64
- F32 precision on GPU: ~1e-6 to 1e-8 relative error
- Recommended for visualization, not scientific accuracy

#### JAX Backend
- **Implementation**: JAX with automatic vectorization (`vmap`)
- **GPU Support**: CUDA only (NVIDIA GPUs)
- **CPU Support**: Yes, with XLA compilation
- **Performance**: 5-20× faster than scipy
- **Precision**: f64 (double)
- **Use case**: GPU acceleration on NVIDIA, CPU with vectorization

---

## Numerical Methods

### Elliptic Integrals

The core of black hole simulation uses elliptic integrals from Luminet (1979):

**Complete Elliptic Integral K(m)**:
```
K(m) = ∫₀^(π/2) dt / √(1 - m·sin²(t))
```
Computed using Arithmetic-Geometric Mean (AGM) iteration.

**Incomplete Elliptic Integral F(φ,m)**:
```
F(φ,m) = ∫₀^φ dt / √(1 - m·sin²(t))
```
Computed using 20-point Gauss-Legendre quadrature.

**Jacobi Elliptic Function sn(u,m)**:
```
sn(u,m) = sin(amplitude)
```
Computed using Descending Landen Transformation (12 iterations).

### Root Finding

Solves for impact parameter using Brent's method:
```
f(p) = 4Mp - r(term₁ + term₂) = 0
```
Where term₁ and term₂ involve elliptic integrals.

### Periastron Calculation

Orbital mechanics for black hole photon trajectories:
```
γ = 2nπ + 2arcsin(...)
```
Requires high precision for convergence near critical impact parameters.

---

## Performance Considerations

### Recommended Backends by Use Case

#### Scientific Research & Publications
- **Backend**: Numba or Taichi CPU (f64)
- **Reason**: Full double precision, significant speedup
- **Expected**: 4-7× faster than scipy

#### Educational Demonstrations
- **Backend**: Numba (CPU) or Taichi (CPU)
- **Reason**: Interactive performance, full accuracy
- **Expected**: 4-10× faster than scipy

#### Real-Time Visualization (Video, Interactive)
- **Backend**: Taichi GPU (f32)
- **Reason**: Maximum speed, acceptable ~1e-6 error
- **Expected**: 10-50× faster than scipy
- **Hardware**: NVIDIA CUDA or AMD Vulkan

#### High-Resolution Rendering (4K+)
- **Backend**: Taichi GPU (f32)
- **Reason**: GPU acceleration essential for large renders
- **Expected**: 11× speedup at 4K resolution (8M pixels)

### Resolution Recommendations

**For smooth color gradients**:
- **Default**: `radial_resolution=400`, `angular_resolution=200`
- **Color banding**: Caused by too few radial rings (< 200)
- **Increase to**: 800+ for 4K resolution

**For performance**:
- **CPU backends**: 100-200 resolution (real-time)
- **GPU backends**: 200-400 resolution (real-time)
- **Final renders**: 400+ resolution (quality over speed)

### Memory Limits

**GPU Memory** (approximate):
```
memory ≈ resolution × resolution × 4 bytes × 2 (direct + ghost)
```

| Resolution | GPU Memory |
|------------|-------------|
| 720p (1280×720) | ~7 MB |
| 1080p (1920×1080) | ~16 MB |
| 1440p (2560×1440) | ~30 MB |
| 4K (3840×2160) | ~66 MB |

All are well within modern GPU limits (2GB+).

---

## References

1. Luminet, J.-P. (1979): "Image of a spherical black hole with thin accretion disk"
2. IEEE 754-2008: "IEEE Standard for Floating-Point Arithmetic"
3. Numerical Recipes (Press et al.): Chapters on root finding and special functions
4. [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
5. Taichi Documentation: [Precision and Type System](https://docs.taichi-lang.org/docs/type)
