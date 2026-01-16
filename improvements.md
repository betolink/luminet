# Luminet Performance Optimization Plan

**Quick Links:**
- [Validation Q&A](validation_qa.md) - How to ensure mathematical accuracy and performance
- [Benchmark Script](benchmark.py) - Compare backend performance
- [Backend Architecture](luminet/backends/) - Pluggable backend system

## Backend Architecture (NEW)

The codebase now supports **pluggable computational backends**:

### Available Backends
1. **scipy** - Original implementation (default, backward compatible)
2. **taichi** - GPU-accelerated implementation (work-in-progress)

### Usage
```python
from luminet import get_backend, list_available_backends

# List available backends
print(list_available_backends())  # ['scipy', 'taichi']

# Get a specific backend
backend = get_backend("scipy")  # or "taichi"

# Use backend API
q = backend.calc_q(p=10.0, bh_mass=1.0)
b = backend.solve_for_impact_parameter(radius=10, incl=1.4, alpha=0.5, bh_mass=1.0)
```

### Command Line Benchmarking
```bash
# Benchmark specific backend
python benchmark.py --engine=scipy
python benchmark.py --engine=taichi

# Compare all backends
python benchmark.py --compare

# Custom resolutions
python benchmark.py --compare --resolutions 50 100 200 500
```

### Benefits
- ✅ **Backward compatibility**: Original scipy code unchanged
- ✅ **Easy testing**: Switch backends with one line
- ✅ **Performance comparison**: Built-in benchmarking tools
- ✅ **Extensible**: Easy to add JAX, numba, or other backends

## Current State Analysis

### Computational Bottlenecks

1. **Scalar root finding** (`black_hole_math.py:36`)
   - `scipy.optimize.brentq` is called per-pixel (scalar, not vectorized)
   - Each isoradial calculation requires ~200 angle evaluations
   - Each angle evaluation requires root finding
   - This is the primary bottleneck

2. **Serial elliptic integral computation** (`black_hole_math.py:180`)
   - `calc_sn()` computes `ellipkinc`, `ellipk`, `ellipj` for each point
   - These are expensive transcendental functions
   - Called thousands of times per black hole image

3. **Multiprocessing overhead** (`black_hole.py:219`)
   - Uses Python's `multiprocessing.Pool` for parallelization
   - High overhead due to pickling/unpickling
   - Limited to CPU, no GPU utilization
   - Process spawning costs for each batch

4. **No spatial caching**
   - Recomputes elliptic integrals and root finding for similar parameters
   - No memoization or interpolation tables
   - Repeated calculations for adjacent points

### Large Files Analysis

**Identified blobs:**
- `assets/1979A+A____75__228L.pdf` (1.1M) - Original Luminet (1979) paper
- `pixi.lock` (371K) - Package lock file (standard, keep)
- `.git/objects/pack/*.pack` (83M) - Normal git history (expected)

**Recommendation:** Move PDF to external reference (GitHub LFS or external link)

---

## Taichi Optimization Plan (Recommended)

### Why Taichi?

1. Native GPU kernels without JIT compilation overhead
2. Built-in automatic differentiation (for future optimization)
3. Explicit memory management for better control
4. Pythonic syntax similar to NumPy
5. Can target CUDA, Metal, Vulkan, OpenGL
6. No need to reimplement scipy solvers - can use numerical methods
7. Better performance than JAX for this use case (more explicit control)

### Phase 1: Core Math Kernels (GPU Acceleration)

Create `luminet_taichi/core.pyti`:

```python
import taichi as ti

ti.init(arch=ti.cuda)  # or ti.gpu for auto-selection

@ti.func
def calc_q_taichi(p: float, bh_mass: float) -> float:
    if p < 2.0 * bh_mass:
        return 0.0
    return ti.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))

@ti.func
def calc_k_squared_taichi(p: float, bh_mass: float) -> float:
    q = calc_q_taichi(p, bh_mass)
    if q == 0.0:
        return 0.0
    return (q - p + 6 * bh_mass) / (2 * q)

@ti.func
def calc_sn_taichi(p: float, angle: float, bh_mass: float, 
                   incl: float, order: int) -> float:
    # Vectorized elliptic integral computation
    q = calc_q_taichi(p, bh_mass)
    if q == 0.0:
        return 0.0
    
    m = calc_k_squared_taichi(p, bh_mass)
    z_inf = ti.asin(ti.sqrt((q - p + 2 * bh_mass) / (q - p + 6 * bh_mass)))
    ell_inf = ti.math.ellipkinc(z_inf, m)
    
    g = ti.acos(ti.cos(angle) / ti.sqrt(ti.cos(angle)**2 + 1 / (ti.tan(incl)**2)))
    
    if order == 0:
        ellips_arg = g / (2.0 * ti.sqrt(p / q)) + ell_inf
    else:
        ell_k = ti.math.ellipk(m)
        ellips_arg = (g - 2.0 * order * np.pi) / (2.0 * ti.sqrt(p / q)) - ell_inf + 2.0 * ell_k
    
    # Use Taichi's built-in jacobi_elliptic or implement numerically
    return ti.math.jacobi_elliptic(ellips_arg, m)[0]

@ti.func
def periastron_cost(p: float, radius: float, angle: float, 
                    bh_mass: float, incl: float, order: int) -> float:
    q = calc_q_taichi(p, bh_mass)
    if q == 0.0:
        return 0.0
    sn = calc_sn_taichi(p, angle, bh_mass, incl, order)
    term1 = -(q - p + 2.0 * bh_mass)
    term2 = (q - p + 6.0 * bh_mass) * sn * sn
    return 4.0 * bh_mass * p - radius * (term1 + term2)

@ti.kernel
def solve_periastron_vectorized(
    radii: ti.types.ndarray(),
    angles: ti.types.ndarray(),
    bh_mass: float,
    incl: float,
    order: int,
    results: ti.types.ndarray()
):
    # Parallel root finding using bisection or Newton
    for i in range(radii.shape[0]):
        for j in range(angles.shape[0]):
            p_min = 3.0 * bh_mass + order * 1e-5
            p_max = radii[i]
            
            # Bisection method (can be optimized with Newton)
            for iter in range(50):
                p_mid = (p_min + p_max) / 2
                f_mid = periastron_cost(p_mid, radii[i], angles[j], bh_mass, incl, order)
                f_min = periastron_cost(p_min, radii[i], angles[j], bh_mass, incl, order)
                
                if f_mid * f_min < 0:
                    p_max = p_mid
                else:
                    p_min = p_mid
            
            results[i, j] = (p_min + p_max) / 2

@ti.kernel
def calc_isoradial_vectorized(
    radius: float,
    bh_mass: float,
    incl: float,
    order: int,
    angles: ti.types.ndarray(),
    impact_parameters: ti.types.ndarray()
):
    for i in range(angles.shape[0]):
        # Vectorized impact parameter calculation
        p = 3.0 * bh_mass  # Initial guess
        
        # Find periastron
        p_min = 3.0 * bh_mass + order * 1e-5
        p_max = radius
        
        for iter in range(50):
            p_mid = (p_min + p_max) / 2
            f_mid = periastron_cost(p_mid, radius, angles[i], bh_mass, incl, order)
            f_min = periastron_cost(p_min, radius, angles[i], bh_mass, incl, order)
            
            if f_mid * f_min < 0:
                p_max = p_mid
            else:
                p_min = p_mid
        
        p = (p_min + p_max) / 2
        impact_parameters[i] = ti.sqrt(p**3 / (p - 2.0 * bh_mass))
```

### Phase 2: High-Level API

Create `luminet_taichi/black_hole.py`:

```python
import numpy as np
import taichi as ti

class BlackHoleTaichi:
    def __init__(self, mass=1.0, incl=1.4, acc=1.0, outer_edge=None,
                 angular_resolution=200, radial_resolution=200):
        self.mass = mass
        self.incl = incl
        self.acc = acc
        self.angular_resolution = angular_resolution
        self.radial_resolution = radial_resolution
        
    def calc_isoradials(self, direct_r, ghost_r):
        """Vectorized isoradial computation on GPU"""
        all_r = np.array(direct_r + ghost_r)
        n_radial = len(all_r)
        
        angles = np.linspace(0, 2*np.pi, self.angular_resolution)
        
        # Allocate GPU memory
        impact_params = np.zeros((n_radial, self.angular_resolution))
        
        for i, radius in enumerate(all_r):
            order = 1 if radius in ghost_r else 0
            calc_isoradial_vectorized(
                radius, self.mass, self.incl, order,
                angles, impact_params[i]
            )
        
        return all_r, angles, impact_params

    def plot(self, **kwargs):
        """GPU-accelerated plotting"""
        radii = np.linspace(6*self.mass, self.disk_outer_edge, self.radial_resolution)
        r_grid, angles, b_grid = self.calc_isoradials(radii.tolist(), [])
        
        # Vectorized flux calculation
        # ... use matplotlib as before
        
        return ax
```

### Jacobi Elliptic Functions: The Hardest Part

**Challenge:** Taichi doesn't have built-in `ellipj` (Jacobi elliptic functions), while scipy has highly optimized implementations.

**Solutions (ordered by preference):**

1. **Numerical AGM Implementation** (Recommended)
   ```python
   @ti.func
   def ellipj_taichi(u: float, m: float) -> tuple:
       # Arithmetic-Geometric Mean method
       # Reference: Abramowitz & Stegun 16.11-16.17
       a = 1.0
       b = ti.sqrt(1.0 - m)
       for i in range(20):
           a, b = (a + b) / 2.0, ti.sqrt(a * b)
       phi = a * u
       return (ti.sin(phi), ti.cos(phi), ...)
   ```

2. **Lookup Table Interpolation** (Fastest, Practical)
   - Precompute `ellipj` on CPU with scipy
   - Interpolate on GPU
   - Error < 1e-8 with good sampling

3. **Use Taichi's Built-in Functions** (Check availability)
   - `ti.math.ellipk(m)` - Complete elliptic integral
   - May need to derive `ellipj` from `ellipk`

**Validation:** This is the most critical test. See `test_calc_sn()` in validation tests.

### Phase 3: Performance Optimizations

1. **Memoization tables** for frequently computed elliptic integrals
   - Cache results for (p, bh_mass) pairs
   - LRU cache with size limit
   - Interpolate between cached values

2. **Adaptive resolution** based on curvature
   - Higher resolution near photon sphere
   - Lower resolution in flat regions
   - Dynamic angular resolution per isoradial

3. **Root finding acceleration** with Newton-Raphson after initial bisection
   - Use bisection for first 10 iterations
   - Switch to Newton for convergence
   - Analytical derivatives available

4. **Batch processing** for large computations
   - Process multiple isoradials in single kernel launch
   - Reduce kernel launch overhead
   - Better GPU utilization

---

## Alternative: JAX Implementation

Create `luminet_jax/core.py`:

```python
import jax
import jax.numpy as jnp
from jax import jit, vmap

@jit
def calc_q_jax(p: float, bh_mass: float) -> float:
    return jnp.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))

@jit
def calc_k_squared_jax(p: float, bh_mass: float) -> float:
    q = calc_q_jax(p, bh_mass)
    return (q - p + 6 * bh_mass) / (2 * q)

@jit
def calc_redshift_factor_jax(radius, angle, incl, bh_mass, b):
    return (1.0 + jnp.sqrt(bh_mass / radius**3) * b * jnp.sin(incl) * jnp.sin(angle)) * \
           (1 - 3.0 * bh_mass / radius) ** -0.5

# Vectorized root finding
@jit
def solve_periastron_bisection(radius, alpha, bh_mass, incl, order):
    p_min = 3.0 * bh_mass + order * 1e-5
    p_max = radius
    
    def body(state):
        p_min, p_max = state
        p_mid = (p_min + p_max) / 2
        f_mid = periastron_cost_jax(p_mid, radius, alpha, bh_mass, incl, order)
        f_min = periastron_cost_jax(p_min, radius, alpha, bh_mass, incl, order)
        new_min = jnp.where(f_mid * f_min < 0, p_min, p_mid)
        new_max = jnp.where(f_mid * f_min < 0, p_mid, p_max)
        return (new_min, new_max)
    
    (p_min, p_max) = jax.lax.fori_loop(0, 50, body, (p_min, p_max))
    return (p_min + p_max) / 2

# Vectorize across angles
solve_periastron_vectorized = vmap(solve_periastron_bisection, in_axes=(None, 0, None, None, None))
```

**Pros of JAX:**
- Easier to get started
- Good ecosystem
- Automatic vectorization
- Works well with research workflows

**Cons vs Taichi:**
- Less explicit control over GPU memory
- JIT compilation overhead
- Higher memory usage
- Less optimal for this specific use case

---

## Expected Performance Gains

| Optimization | Speedup | Notes |
|-------------|---------|-------|
| **Taichi GPU** | 10-50x | Depends on GPU, best for high-res |
| **Taichi CPU (multithreaded)** | 5-15x | Better than multiprocessing |
| **JAX (GPU)** | 8-30x | Good for research workflows |
| **JAX (CPU)** | 3-10x | Similar to Taichi CPU |
| **Current optimized** | 1x | Baseline |

**Benchmarks to run:**
- Image generation time for 200x200 resolution
- Memory usage comparison
- Scaling with resolution (100x100, 500x500, 1000x1000)
- GPU vs CPU comparison

---

## Blob Reduction Plan

### Immediate Actions

1. **Move PDF to external reference**
   ```bash
   git rm --cached assets/1979A+A____75__228L.pdf
   echo "assets/*.pdf" >> .gitignore
   # Update README.md to use URL
   ```

2. **Update documentation**
   - Keep reference to Luminet (1979) paper
   - Use arXiv or ADS link
   - Keep DOI in bibliography

3. **Potential future optimizations**
   - Consider using `git-lfs` for large assets if needed in future
   - Compress image assets in assets/ directory
   - Review if any test data can be generated on-the-fly

### File Size Impact

- **Before:** ~1.1M PDF + normal git objects
- **After:** ~0M PDF (external reference) + normal git objects
- **Savings:** ~1.1M per clone

---

## Implementation Roadmap (Updated)

### Phase 0: Refactoring (COMPLETED) ✅
- [x] Create `BaseBackend` abstract class
- [x] Implement `ScipyBackend` (wraps existing code)
- [x] Create `TaichiBackend` skeleton (fallback to scipy for now)
- [x] Implement backend factory pattern
- [x] Create `benchmark.py` with CLI support
- [x] Update documentation

### Phase 1: Taichi Core (Week 1-2)

- [ ] Implement `calc_q`, `calc_k_squared` kernels
- [ ] Implement `calc_sn` with elliptic integrals
- [ ] Implement vectorized root finding
- [ ] Unit tests against scipy implementation
- [ ] Verify numerical accuracy (< 1e-6 error)

### Phase 2: High-Level API (Week 3)

- [ ] Complete `TaichiBackend` implementation
- [ ] Port `BlackHole` class to use backends
- [ ] Port `Isoradial` class to use backends
- [ ] Integrate with matplotlib for visualization
- [ ] Performance benchmarks vs scipy backend
- [ ] Accuracy validation

### Phase 3: Optimization (Week 4)

- [ ] Implement memoization for elliptic integrals
- [ ] Add adaptive resolution based on curvature
- [ ] Optimize GPU memory usage
- [ ] Add Newton-Raphson acceleration
- [ ] Batch processing optimization

### Phase 4: Documentation & Cleanup (Week 5)

- [ ] Migration guide from scipy to Taichi
- [ ] Performance comparison charts
- [ ] Blob cleanup (remove PDF)
- [ ] Update README with backend architecture
- [ ] Add installation instructions for Taichi

### Phase 5: Testing & Release (Week 6)

- [ ] Full test suite with all backends
- [ ] CI/CD integration
- [ ] Performance regression tests
- [ ] Documentation updates
- [ ] Release preparation

---

## Current Optimization Assessment

### What's Already Good

**The code is reasonably well-optimized for pure Python/NumPy:**
- ✓ Uses NumPy vectorization where possible
- ✓ Parallel processing with multiprocessing
- ✓ Efficient data structures
- ✓ Clean, modular architecture
- ✓ Good separation of concerns

### Main Limitation

**Scalar `scipy.optimize.brentq` cannot be vectorized**, which is the primary bottleneck. Moving to Taichi or JAX is the correct approach for GPU acceleration.

### No Immediate Optimizations Needed

The current implementation is solid for:
- Low-resolution visualizations
- Educational purposes
- Small-scale research
- Systems without GPU

---

## Technical Notes

### Elliptic Integral Implementation

Taichi provides `ti.math.ellipk` and `ti.math.ellipkinc` but not `ellipj`. Options:

1. Use numerical approximation (Jacobi elliptic functions)
2. Port scipy's implementation
3. Use relationship between elliptic integrals and theta functions
4. Implement using arithmetic-geometric mean

### Root Finding Strategy

For GPU implementation:
- Bisection: Robust, slower (50 iterations)
- Newton-Raphson: Fast (5-10 iterations), needs derivative
- Secant: Fast (10-15 iterations), no derivative needed
- Hybrid: Bisection first, then Newton (recommended)

### Memory Management

Taichi provides:
- `ti.ndarray` for GPU arrays
- Explicit memory allocation
- Control over data transfers
- Avoids Python object overhead

---

## References

- Luminet (1979) - Original paper: https://ui.adsabs.harvard.edu/abs/1979A%26A....75..228L/abstract
- Taichi documentation: https://docs.taichi-lang.org/
- JAX documentation: https://jax.readthedocs.io/
- Scipy elliptic integrals: https://docs.scipy.org/doc/scipy/reference/special.html#elliptic-integrals
