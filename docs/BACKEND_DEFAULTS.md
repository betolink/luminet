# Backend and Hardware Defaults Reference

Quick reference for default settings across different tools in Luminet.

## Default Settings

### generate_video.py (Video Generation)

```bash
--backend taichi     # GPU-optimized for video rendering
--hw gpu             # Auto-detect GPU (Vulkan or CUDA)
--frames 60          # 2 seconds at 30fps
--fps 30             # Standard framerate
--resolution 1080p   # Full HD
```

**Why these defaults?**
- `taichi` + `gpu`: Provides 4-11× speedup for video rendering
- Auto-detect GPU works on AMD (Vulkan) and NVIDIA (CUDA)

### render.py (Single Image Rendering)

```bash
--backend scipy      # Most stable, widely compatible
--hw (not set)       # CPU by default
--resolution 200     # 200×200 pixels
```

**Why these defaults?**
- `scipy`: Rock-solid stability for single images
- Small resolution (200×200) renders quickly for testing

## Command Line Options

### Backend Selection

Both tools support:

```bash
--backend {scipy,numba,taichi,jax}
```

| Backend | Best For | Speed | Precision |
|---------|----------|-------|-----------|
| **scipy** | Single images, stability | 1× (baseline) | f64 |
| **numba** | CPU-only, medium arrays | 2-3× | f64 |
| **taichi** | GPU/CPU, large arrays | 2-11× | f32 (GPU), f64 (CPU) |
| **jax** | NVIDIA GPU only | 5-20× (GPU) | f32/f64 |

### Hardware Selection (for taichi/jax)

```bash
--hw {cpu,gpu,cuda,vulkan}
```

| Hardware | Description | Auto-Detects |
|----------|-------------|--------------|
| **cpu** | Force CPU execution | - |
| **gpu** | Auto-detect GPU | ✅ CUDA or Vulkan |
| **cuda** | Force NVIDIA CUDA | - |
| **vulkan** | Force Vulkan | Works on AMD/Intel/NVIDIA |

## Common Usage Patterns

### Quick Single Image (render.py)

```bash
# Default: scipy (stable, medium speed)
python render.py --output test.png

# Fast: numba (2-3× faster, CPU only)
python render.py --backend numba --output test.png

# Fastest: GPU (4-11× faster for large resolutions)
python render.py --backend taichi --hw gpu --resolution 500 --output test.png
```

### Video Generation (generate_video.py)

```bash
# Default: GPU-accelerated (recommended)
python generate_video.py --type rotation

# Explicitly specify GPU
python generate_video.py --type rotation --backend taichi --hw gpu

# Force CPU (slower, but works without GPU)
python generate_video.py --type rotation --backend scipy

# Force AMD Vulkan
python generate_video.py --type rotation --backend taichi --hw vulkan

# Force NVIDIA CUDA
python generate_video.py --type rotation --backend taichi --hw cuda
```

## When to Override Defaults

### Use scipy (CPU) when:
- ✅ Generating single images (render.py default)
- ✅ Need guaranteed f64 precision
- ✅ GPU not available
- ✅ Small images (< 200×200)

### Use numba (CPU) when:
- ✅ Need f64 precision but faster than scipy
- ✅ Medium-sized images (200-500 pixels)
- ✅ GPU not available
- ✅ Repeated renders (amortize JIT compilation)

### Use taichi + gpu when:
- ✅ Generating videos (generate_video.py default)
- ✅ Large images (> 300×300)
- ✅ High-resolution renders (1080p, 4K)
- ✅ GPU available
- ✅ Speed matters more than precision

### Use taichi + cpu when:
- ✅ Testing GPU code on CPU
- ✅ Medium arrays without GPU
- ✅ Want portability (same code for CPU/GPU)

## Environment-Specific Defaults

### On Systems with AMD GPU

**Recommended for videos**:
```bash
python generate_video.py --type rotation
# Auto-detects Vulkan, 4-11× speedup
```

**Recommended for single images**:
```bash
# Small images: use default
python render.py --output test.png

# Large images (>300×300): use GPU
python render.py --backend taichi --hw gpu --resolution 500 --output test.png
```

### On Systems with NVIDIA GPU

**Recommended for videos**:
```bash
python generate_video.py --type rotation
# Auto-detects CUDA, 10-50× speedup expected
```

**Can also use JAX**:
```bash
python generate_video.py --type rotation --backend jax
# JAX has XLA optimization for NVIDIA
```

### On Systems without GPU

**For videos** (use CPU backend):
```bash
python generate_video.py --type rotation --backend scipy
# OR
python generate_video.py --type rotation --backend numba
```

**For images** (use default or numba):
```bash
python render.py --output test.png  # scipy default
# OR
python render.py --backend numba --output test.png  # 2-3× faster
```

## Performance Comparison

### Single Image (500×500 pixels)

| Command | Backend | Hardware | Time |
|---------|---------|----------|------|
| `render.py` (default) | scipy | CPU | ~2s |
| `render.py --backend numba` | numba | CPU | ~1s |
| `render.py --backend taichi --hw gpu` | taichi | AMD GPU | ~0.3s |

### Video (60 frames, 1080p)

| Command | Backend | Hardware | Time |
|---------|---------|----------|------|
| `generate_video.py --backend scipy` | scipy | CPU | ~20min |
| `generate_video.py --backend numba` | numba | CPU | ~8min |
| `generate_video.py` (default) | taichi | AMD GPU | ~2min |
| `generate_video.py` (NVIDIA) | taichi | NVIDIA GPU | ~1min (est.) |

## Configuration Summary

### render.py
```
Default: --backend scipy (stable, CPU)
Override for speed: --backend taichi --hw gpu
Override for precision: --backend scipy (already default)
```

### generate_video.py
```
Default: --backend taichi --hw gpu (fastest)
Override for CPU: --backend scipy or --backend numba
Override for NVIDIA: --backend taichi --hw cuda (or auto-detects)
```

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│ LUMINET BACKEND QUICK REFERENCE                             │
├─────────────────────────────────────────────────────────────┤
│ Single Image (render.py):                                   │
│   Default:     scipy (stable)                               │
│   Fast:        --backend numba                              │
│   Fastest:     --backend taichi --hw gpu                    │
│                                                              │
│ Video (generate_video.py):                                  │
│   Default:     taichi + gpu ✅ (4-11× faster)               │
│   CPU only:    --backend scipy                              │
│   Force AMD:   --backend taichi --hw vulkan                 │
│   Force NVIDIA: --backend taichi --hw cuda                  │
│                                                              │
│ Hardware Options:                                           │
│   --hw cpu       Force CPU                                  │
│   --hw gpu       Auto-detect (RECOMMENDED)                  │
│   --hw cuda      Force NVIDIA CUDA                          │
│   --hw vulkan    Force Vulkan (AMD/Intel)                   │
└─────────────────────────────────────────────────────────────┘
```

## See Also

- `PERF.md` - Detailed performance benchmarks
- `GPU_BACKENDS.md` - GPU backend setup
- `VIDEO_GENERATION.md` - Video generation guide
- `README.md` - General usage documentation
