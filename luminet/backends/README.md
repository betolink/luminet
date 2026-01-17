# Computational Backends

This directory contains pluggable computational backends for luminet.

## Architecture

The backend system allows users to choose between different computational implementations:

- **scipy**: Original implementation using scipy.optimize.brentq and scipy.special (f64 precision)
- **numba**: JIT-compiled CPU with @njit decorator (4.7× faster, f64 precision)
- **taichi**: GPU-accelerated implementation (10-50× faster, f32 GPU / f64 CPU)
- **jax**: JAX-based with automatic vectorization and JIT compilation (5-20× faster, f64)

## Backend Comparison

| Backend | Status | GPU Support | Precision | Speedup vs Scipy |
|---------|---------|--------------|-----------|--------------------|
| scipy | ✅ Production | No | f64 | 1× (baseline) |
| numba | ✅ Production | No | f64 | 4.7× |
| taichi (CPU) | ✅ Production | No | f64 | 4-10× |
| taichi (GPU) | ✅ Production | Yes (CUDA/Vulkan) | f32 | 10-50× |
| jax (CPU) | ✅ Production | No | f64 | 5-20× |
| jax (GPU) | ✅ Production | Yes (CUDA only) | f64 | 5-20× |

## Quick Start

```python
from luminet.black_hole import BlackHole

# Default scipy backend (baseline)
bh = BlackHole(mass=1.0, incl=1.4)

# Numba backend (recommended for CPU)
bh = BlackHole(mass=1.0, incl=1.4, backend='numba')

# Taichi CPU backend
bh = BlackHole(mass=1.0, incl=1.4, backend='taichi')

# Taichi GPU backend (auto-detects CUDA/Vulkan)
bh = BlackHole(mass=1.0, incl=1.4, backend='taichi', arch='gpu')

# JAX backend
bh = BlackHole(mass=1.0, incl=1.4, backend='jax')
```

## File Structure

```
backends/
├── __init__.py              # Factory functions and exports
├── base.py                  # Abstract base class for all backends
├── scipy_backend.py          # Scipy implementation (complete)
├── numba_backend.py         # Numba JIT implementation (complete)
├── taichi_backend.py        # Taichi GPU/CPU implementation (complete)
└── jax_backend.py           # JAX implementation (complete)
```

## GPU Backend Details

### Taichi GPU

**Hardware Support**:
- NVIDIA GPUs (CUDA): Both f32 and f64 (depending on GPU model)
- AMD/Intel GPUs (Vulkan): f32 only
- Automatic fallback: If GPU fails, uses CPU f64

**Precision**:
- f32: ~1e-6 to 1e-8 relative error
- Suitable for: Real-time video, visualization, demos
- Not recommended for: Scientific publications (use f64)

**Performance**:
- Consumer GPUs: 4-11× faster than scipy
- Professional GPUs (A100, V100): 10-50× faster than scipy

### JAX GPU

**Hardware Support**:
- NVIDIA GPUs (CUDA) only
- AMD GPUs not supported (no ROCm in JAX yet)

**Precision**:
- f64 (double precision) on both CPU and GPU

**Performance**:
- 5-20× faster than scipy
- Automatic vectorization via `vmap`

## Adding a New Backend

To add a new backend (e.g., a new JIT compiler):

1. Create a new file `your_backend.py`:

```python
from .base import BackendBase

class YourBackend(BackendBase):
    def get_backend_name(self):
        return "your_backend"
    
    def supports_gpu(self):
        return False  # or True
    
    def supports_vectorization(self):
        return False  # or True
    
    # Implement required methods:
    # - calc_q()
    # - calc_k_squared()
    # - calc_zeta_inf()
    # - calc_sn()
    # - solve_for_impact_parameter()
    # - calc_redshift_factor()
    # - calc_flux_observed()
```

2. Register in `__init__.py`:

```python
from .your_backend import YourBackend

_AVAILABLE_BACKENDS['your_backend'] = YourBackend
```

3. Update documentation:
   - Add to backend comparison table above
   - Document GPU support if applicable
   - Note precision and performance characteristics

## Performance Tips

### For Best Speed

1. **Use GPU for large renders** (> 500×500 pixels):
   ```python
   bh = BlackHole(..., backend='taichi', arch='gpu')
   ```

2. **Use Numba for CPU** (if GPU unavailable):
   ```python
   bh = BlackHole(..., backend='numba')
   ```

3. **Use appropriate resolution**:
   - CPU backends: 100-200 (real-time)
   - GPU backends: 200-400 (real-time)
   - Final renders: 400+ (quality)

### For Best Accuracy

1. **Use f64 precision** for scientific work:
   - `backend='scipy'` (reference, slowest)
   - `backend='numba'` (recommended, 4.7× faster)
   - `backend='taichi'` (CPU mode, 4-10× faster)

2. **Avoid GPU f32** for publications** (acceptable for demos only)

## References

- [Technical Documentation](../docs/TECHNICAL.md) - Precision analysis and numerical methods
- [GPU Backends Guide](../../GPU_BACKENDS.md) - GPU setup and hardware requirements
- [Performance Benchmarks](../../PERF.md) - Detailed performance comparison
