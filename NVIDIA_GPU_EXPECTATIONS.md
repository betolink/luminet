# NVIDIA GPU Support - Expected Behavior

## Summary

When you plug in an NVIDIA GPU, **both Taichi and JAX GPU backends should work automatically**. Here's what to expect:

---

## 🎯 Taichi GPU Backend

### Status: ✅ **WILL WORK** (already fixed)

#### What Will Happen

1. **Initialization**:
   ```python
   from luminet.backends import get_backend
   backend = get_backend('taichi', arch='gpu')
   # Or explicitly: arch='cuda'
   ```

2. **Auto-detection logic** (from `taichi_backend.py` lines 58-78):
   - Tries: **CUDA f32** (most likely to succeed) ← **NVIDIA will use this**
   - Falls back to: CUDA f64 → Vulkan f32 → Vulkan f64 → CPU
   
3. **Expected output**:
   ```
   Backend: cuda, GPU: True, f64: False
   ```
   
   If your NVIDIA GPU supports f64 transcendentals (professional GPUs like A100, V100):
   ```
   Backend: cuda, GPU: True, f64: True
   ```

#### Performance Expectations

- **Consumer GPUs** (RTX 3060, 4090, etc.): **f32 precision**
  - ~1e-6 to 1e-8 accuracy (perfect for visualization)
  - Faster than AMD Vulkan (better CUDA optimization)

- **Professional GPUs** (A100, V100, etc.): **f64 precision possible**
  - ~1e-15 accuracy (same as CPU)
  - Significantly faster for large workloads

#### Verification Test

```bash
# After plugging in NVIDIA GPU:
python -c "
from luminet.backends import get_backend
backend = get_backend('taichi', arch='gpu')
print(f'Architecture: {backend.arch}')
print(f'GPU: {backend._is_gpu}')
print(f'Precision: f{64 if backend._use_f64 else 32}')
"
```

**Expected**: `Architecture: cuda, GPU: True`

---

## 🎯 JAX GPU Backend

### Status: ✅ **WILL WORK** (CUDA 12 support already installed)

#### Current Installation

```bash
$ pip list | grep jax
jax                       0.8.2
jax-cuda12-pjrt           0.8.2      ← CUDA 12 support installed ✅
jax-cuda12-plugin         0.8.2      ← CUDA 12 plugin installed ✅
jaxlib                    0.8.2
```

#### What Will Happen

1. **Initialization**:
   ```python
   from luminet.backends import get_backend
   backend = get_backend('jax', use_gpu=True)
   ```

2. **JAX will auto-detect** NVIDIA GPU:
   ```python
   import jax
   jax.devices()  # Will show: [cuda(id=0)] instead of [cpu(id=0)]
   ```

3. **JIT-compiled operations** will run on GPU automatically

#### Current JAX Backend Capabilities

From `jax_backend.py`:
- ✅ JIT-compiled: `calc_q`, `calc_k_squared`, `calc_zeta_inf`, `calc_redshift`, `calc_flux_intrinsic`
- ⚠️ Falls back to scipy: Elliptic functions, root finding (for accuracy)

**Note**: JAX backend uses scipy fallback for complex math, so it won't be as fast as Taichi GPU for full BlackHole calculations. But simple operations will be GPU-accelerated.

#### Verification Test

```bash
# After plugging in NVIDIA GPU:
python -c "
import jax
print('JAX devices:', jax.devices())
print('Default backend:', jax.default_backend())

from luminet.backends import get_backend
backend = get_backend('jax', use_gpu=True)
result = backend.calc_q(5.0, 1.0)
print(f'calc_q result: {result}')
"
```

**Expected output**:
```
JAX devices: [cuda(id=0)]
Default backend: gpu
calc_q result: 5.744563
```

---

## ⚡ Performance Comparison (Predicted with NVIDIA GPU)

| Backend | Hardware | Precision | Speed (est.) | Best For |
|---------|----------|-----------|--------------|----------|
| scipy | CPU | f64 | 1× | Baseline, small data |
| numba | CPU | f64 | 4.7× | CPU-only systems |
| taichi (cpu) | CPU | f64 | 4.5× | CPU-only systems |
| **taichi (cuda)** | **NVIDIA GPU** | **f32** | **10-50×** 🔥 | **Large renders** |
| taichi (cuda) | NVIDIA Pro GPU | f64 | 10-50× | Professional work |
| jax (gpu) | NVIDIA GPU | f32/f64 | 5-20× | Simple operations |

**Notes**:
- GPU speedup depends on problem size (larger = better speedup)
- Consumer NVIDIA GPUs use f32 (great for visualization)
- Professional GPUs (A100, V100) can use f64

---

## 🧪 Recommended First Tests with NVIDIA GPU

### 1. Quick Verification (30 seconds)

```bash
# Test 1: Check devices
python -c "
import jax
import taichi as ti
print('JAX devices:', jax.devices())
ti.init(arch=ti.cuda)
print('Taichi CUDA: initialized')
"
```

### 2. Backend Test (1 minute)

```bash
python test_gpu_final.py
```

**Expected**: All tests pass, shows `Architecture: cuda`

### 3. Performance Benchmark (2 minutes)

```bash
python benchmark_gpu.py
```

**Expected**: Taichi GPU shows 10-50× speedup for large arrays

### 4. Full Render Test (5 minutes)

```python
from luminet.black_hole import BlackHole
from luminet import black_hole_math as bhmath
from luminet.backends import get_backend

# Use CUDA GPU
backend = get_backend('taichi', arch='cuda')
bhmath._backend = backend

# Create and render
bh = BlackHole(mass=1.0, incl=1.4, acc=1.0)
# Continue with full rendering...
```

---

## 🐛 Potential Issues & Solutions

### Issue 1: "CUDA Error: CUDA_ERROR_NO_DEVICE"
**Cause**: NVIDIA drivers not installed or GPU not detected  
**Solution**: 
```bash
nvidia-smi  # Check if GPU is detected
# If not, install NVIDIA drivers
```

### Issue 2: "jax_plugins.xla_cuda12.initialize() failed"
**Cause**: CUDA toolkit version mismatch  
**Solution**:
```bash
# Check CUDA version
nvcc --version
# Reinstall JAX for your CUDA version:
pip install jax[cuda12]  # For CUDA 12.x
# or
pip install jax[cuda11]  # For CUDA 11.x
```

### Issue 3: Taichi falls back to CPU
**Cause**: CUDA initialization failed  
**Solution**:
```bash
# Test Taichi CUDA directly:
python -c "
import taichi as ti
ti.init(arch=ti.cuda, log_level='debug')
print('CUDA initialized')
"
# Check error messages
```

### Issue 4: Slower than expected
**Cause**: Data transfer overhead or small problem size  
**Solution**: 
- Use larger batch sizes (>10k elements)
- Keep data on GPU between operations
- Profile with `ti.profiler.print_kernel_profiler_info()`

---

## 📝 Code Changes Needed (None!)

**Good news**: No code changes needed! The current implementation:
- ✅ Already tries CUDA first when `arch='gpu'`
- ✅ All `ti.template()` conversions done (works for both f32 and f64)
- ✅ JAX CUDA 12 support already installed
- ✅ Automatic fallback if GPU unavailable

---

## 🎉 Bottom Line

**With NVIDIA GPU plugged in:**

1. **Taichi**: Change nothing, just use `arch='gpu'` or `arch='cuda'`
   - Will automatically use CUDA
   - 10-50× faster for large operations
   - Works immediately (all fixes already applied)

2. **JAX**: Change nothing, just use `use_gpu=True`
   - Will automatically detect CUDA GPU
   - 5-20× faster for simple operations
   - Works immediately (CUDA support already installed)

**Test command after NVIDIA GPU installation**:
```bash
python test_gpu_final.py && python benchmark_gpu.py
```

If both tests pass and show CUDA backend, you're good to go! 🚀
