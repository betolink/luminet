# Computational Backends

This directory contains pluggable computational backends for luminet.

## Architecture

The backend system allows users to choose between different computational implementations:

- **scipy**: Original implementation using scipy.optimize.brentq and scipy.special
- **taichi**: GPU-accelerated implementation (work-in-progress)

## File Structure

```
backends/
├── __init__.py              # Factory functions and exports
├── base.py                  # Abstract base class for all backends
├── scipy_backend.py          # Scipy implementation (complete)
└── taichi_backend.py        # Taichi implementation (skeleton)
```

## Adding a New Backend

To add a new backend (e.g., JAX, Numba):

1. Create a new file `your_backend.py`:
```python
from luminet.backends.base import BaseBackend

class YourBackend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.name = "your_backend"

    def calc_q(self, p, bh_mass):
        # Your implementation
        pass

    # Implement all other abstract methods...
```

2. Update `backends/__init__.py` to import and register your backend:
```python
try:
    from luminet.backends.your_backend import YourBackend
    YOUR_BACKEND_AVAILABLE = True
except ImportError:
    YOUR_BACKEND_AVAILABLE = False
```

3. Update `get_backend()` factory to handle your backend:
```python
elif backend_name == "your_backend":
    if not YOUR_BACKEND_AVAILABLE:
        raise ImportError("Your backend not installed")
    return YourBackend(**kwargs)
```

## Backend API

All backends must implement these methods:

### Core Math Functions
- `calc_q(p, bh_mass)` - Convert periastron to Q
- `calc_k_squared(p, bh_mass)` - Calculate elliptic modulus squared
- `calc_zeta_inf(p, bh_mass)` - Calculate zeta_infinity
- `calc_sn(p, angle, bh_mass, incl, order)` - Jacobi elliptic function

### Optimization Functions
- `periastron_cost(p, radius, angle, bh_mass, incl, order)` - Cost function
- `solve_for_periastron(radius, incl, alpha, bh_mass, order)` - Root finding
- `solve_for_impact_parameter(radius, incl, alpha, bh_mass, order)` - Get impact parameter

### Physics Functions
- `calc_redshift_factor(radius, angle, incl, bh_mass, b)` - Redshift calculation
- `calc_flux_intrinsic_swarzschild(radius, acc, bh_mass)` - Intrinsic flux
- `calc_flux_observed(radius, acc, bh_mass, redshift_factor)` - Observed flux

### Metadata
- `get_backend_name()` - Return backend name
- `supports_vectorization()` - Can process arrays?
- `supports_gpu()` - Can run on GPU?

## Usage Examples

### Python API
```python
from luminet import get_backend, list_available_backends

# List available backends
available = list_available_backends()
print(f"Available backends: {available}")

# Get specific backend
backend = get_backend("scipy")  # or "taichi"

# Use backend
q = backend.calc_q(p=10.0, bh_mass=1.0)
b = backend.solve_for_impact_parameter(
    radius=10, incl=1.4, alpha=0.5, bh_mass=1.0
)
```

### Command Line
```bash
# Benchmark scipy backend
python benchmark.py --engine=scipy

# Benchmark taichi backend
python benchmark.py --engine=taichi

# Compare all backends
python benchmark.py --compare

# Custom resolutions
python benchmark.py --compare --resolutions 50 100 200 500
```

## Testing

Run validation tests for all backends:
```bash
# Test scipy backend (baseline)
python tests/test_validation.py --backend scipy

# Test taichi backend (when implemented)
python tests/test_validation.py --backend taichi
```

## Performance

Expected speedup for Taichi backend (when complete):

| Resolution | Scipy (ms) | Taichi CPU (ms) | Taichi GPU (ms) |
|------------|--------------|------------------|------------------|
| 50x50      | ~100         | ~20              | ~5               |
| 100x100     | ~400         | ~80              | ~15              |
| 200x200     | ~1600        | ~320             | ~40              |
| 500x500     | ~10000       | ~2000            | ~200             |

*Note: These are estimated values. Actual results will vary based on hardware.*

## Current Status

| Backend | Status | Complete | GPU Support |
|---------|---------|-----------|--------------|
| scipy   | ✅ Done | 100%      | ❌ No        |
| taichi  | 🚧 WIP  | 10%       | ✅ Yes       |

### Scipy Backend
- ✅ All core math functions implemented
- ✅ Root finding with scipy.optimize.brentq
- ✅ Full backward compatibility
- ✅ Well-tested

### Taichi Backend
- ✅ Base class structure
- ✅ Factory integration
- ✅ Skeleton methods (fall back to scipy)
- ⏳ GPU kernels to be implemented
- ⏳ Elliptic integrals (Jacobi functions)
- ⏳ Vectorized root finding

## References

- [Base Backend Class](base.py)
- [Scipy Implementation](scipy_backend.py)
- [Taichi Implementation](taichi_backend.py)
- [Benchmark Script](../benchmark.py)
- [Validation Tests](../tests/test_validation.py)
