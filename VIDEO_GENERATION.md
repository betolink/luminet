# Black Hole Video Generation

Generate stunning animations of black holes with GPU acceleration!

## Quick Start

### 1. Install ffmpeg (Required for video creation)

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg
```

### 2. Generate Your First Video

```bash
# Quick 480p preview (fast, ~10 seconds on GPU)
python generate_video.py --type rotation --frames 30 --resolution 480p --backend taichi --hw gpu

# High-quality 1080p rotation (2-3 minutes on GPU)
python generate_video.py --type rotation --frames 60 --resolution 1080p --backend taichi --hw gpu

# Ultra-smooth 720p orbit (30 seconds on GPU)
python generate_video.py --type orbit --frames 120 --resolution 720p --fps 60
```

## Animation Types

### 1. Rotation (`--type rotation`)

Sweeps the viewing angle from face-on (0°) to edge-on (90°), showing how the black hole appearance changes with inclination.

**Visual effect**: Accretion disk goes from circular (face-on) to elliptical (edge-on), photon ring morphs

**Best for**: Educational videos, parameter exploration

**Recommended settings**:
```bash
python generate_video.py --type rotation --frames 60 --resolution 1080p --fps 30
```

### 2. Orbit (`--type orbit`)

Simulates orbiting around the black hole with smooth sinusoidal inclination variation.

**Visual effect**: Smooth "flying around" the black hole

**Best for**: Cinematic shots, presentations

**Recommended settings**:
```bash
python generate_video.py --type orbit --frames 120 --resolution 1080p --fps 60
```

### 3. Zoom (`--type zoom`)

Zooms in and out on the accretion disk, varying the outer disk radius.

**Visual effect**: Smooth zoom revealing disk structure

**Best for**: Detail exploration, dramatic reveal

**Recommended settings**:
```bash
python generate_video.py --type zoom --frames 60 --resolution 1080p --fps 30
```

## Performance Guide

Based on actual benchmarks (AMD Radeon Phoenix APU with Vulkan):

### GPU Acceleration (Recommended)

| Resolution | Frames | Backend | Hardware | Est. Time | File Size |
|------------|--------|---------|----------|-----------|-----------|
| **480p** | 30 | taichi | gpu | ~10s | ~2 MB |
| **720p** | 60 | taichi | gpu | ~30s | ~5 MB |
| **1080p** | 60 | taichi | gpu | ~2min | ~10 MB |
| **1080p** | 120 | taichi | gpu | ~4min | ~20 MB |
| **4K** | 60 | taichi | gpu | ~10min | ~40 MB |

**GPU provides 4-11× speedup!** (See PERF.md for details)

### CPU Baseline (scipy)

| Resolution | Frames | Backend | Est. Time |
|------------|--------|---------|-----------|
| 480p | 30 | scipy | ~1min |
| 720p | 60 | scipy | ~8min |
| 1080p | 60 | scipy | ~20min |

**Recommendation**: Always use GPU (`--backend taichi --hw gpu`) for video generation!

## Command Reference

### Basic Options

```bash
--type {rotation,orbit,zoom}    # Animation type (default: rotation)
--frames FRAMES                  # Number of frames (default: 60)
--fps FPS                        # Frames per second (default: 30)
--resolution {480p,720p,1080p,1440p,4k}  # Output resolution
--output FILENAME                # Output video path
--keep-frames                    # Keep individual PNG frames
```

### Backend Options

```bash
--backend {scipy,numba,taichi,jax}  # Computational backend (default: taichi)
--hw {cpu,gpu,cuda,vulkan}           # Hardware for taichi (default: gpu)
```

## Examples

### Quick Preview (Test GPU Setup)

```bash
# Fast 480p test with GPU - completes in ~10 seconds
python generate_video.py \
    --type rotation \
    --frames 30 \
    --resolution 480p \
    --backend taichi \
    --hw gpu \
    --output test_gpu.mp4
```

### Publication-Quality Video

```bash
# High-quality 1080p with smooth framerate
python generate_video.py \
    --type rotation \
    --frames 60 \
    --resolution 1080p \
    --fps 30 \
    --backend taichi \
    --hw gpu \
    --output bh_rotation_1080p.mp4
```

### Cinematic Orbit

```bash
# Ultra-smooth 60fps orbital motion
python generate_video.py \
    --type orbit \
    --frames 120 \
    --resolution 1080p \
    --fps 60 \
    --backend taichi \
    --hw gpu \
    --output bh_orbit_cinematic.mp4
```

### 4K for Presentations

```bash
# High-resolution 4K (slow, but stunning)
python generate_video.py \
    --type rotation \
    --frames 60 \
    --resolution 4k \
    --fps 30 \
    --backend taichi \
    --hw gpu \
    --output bh_4k.mp4
```

### CPU-Only (No GPU)

```bash
# If you don't have GPU, use scipy (slower)
python generate_video.py \
    --type rotation \
    --frames 30 \
    --resolution 720p \
    --backend scipy \
    --output bh_cpu.mp4
```

## Troubleshooting

### Error: "ffmpeg not found"

Install ffmpeg (see Quick Start above). If you can't install ffmpeg, use `--keep-frames` to save individual PNG images:

```bash
python generate_video.py --type rotation --frames 30 --keep-frames
```

Then manually create video:
```bash
ffmpeg -framerate 30 -i frames_rotation_1080p/frame_%04d.png \
       -c:v libx264 -pix_fmt yuv420p output.mp4
```

### GPU Not Working

Test GPU backend first:
```bash
python tests/test_gpu_final.py
```

If GPU doesn't work, fall back to CPU:
```bash
python generate_video.py --backend scipy
```

### Out of Memory

Reduce resolution or use CPU backend:
```bash
# Lower resolution
python generate_video.py --resolution 720p

# Or use CPU (slower but uses less GPU memory)
python generate_video.py --backend scipy
```

### Slow Rendering

GPU should make this much faster. Check:

1. **Are you using GPU?** Add `--backend taichi --hw gpu`
2. **Is Taichi using GPU?** Check output for "Architecture: vulkan" or "cuda"
3. **Try lower resolution first**: Start with 480p to verify setup

## Advanced: Custom Animations

You can modify `generate_video.py` to create custom animations:

### Example: Vary Accretion Rate

```python
# In generate_video.py, create new function:
def generate_accretion_frames(...):
    # Vary accretion rate from 0.1 to 2.0
    accretion_rates = np.linspace(0.1, 2.0, n_frames)
    
    for i, acc in enumerate(accretion_rates):
        bh = BlackHole(
            mass=1.0,
            incl=1.4,
            acc=acc,  # Variable accretion
            ...
        )
        bh.plot(ax=ax)
        # ... save frame
```

### Example: Combined Motion

```python
# Vary multiple parameters simultaneously
for i in range(n_frames):
    incl = base_incl + 0.3 * np.sin(2 * np.pi * i / n_frames)
    acc = 1.0 + 0.5 * np.cos(2 * np.pi * i / n_frames)
    
    bh = BlackHole(mass=1.0, incl=incl, acc=acc, ...)
```

## Output Format

Videos are created with:
- **Codec**: H.264 (widely compatible)
- **Quality**: CRF 18 (high quality, ~95% visually lossless)
- **Preset**: slow (better compression, higher quality)
- **Pixel format**: yuv420p (compatible with all players)

## Tips for Best Results

1. **Start small**: Test with 480p/30 frames first
2. **Use GPU**: Always specify `--backend taichi --hw gpu` for 4-11× speedup
3. **Smooth motion**: Use higher frame counts (120+) with 60fps for ultra-smooth animations
4. **File size**: 1080p/60fps ~= 10MB, 4K/60fps ~= 40MB
5. **Quality vs speed**: 720p is sweet spot for quality/speed balance

## Performance Comparison

**Example: 60-frame 1080p rotation**

| Backend | Hardware | Time | Speedup |
|---------|----------|------|---------|
| scipy | CPU | ~20 min | 1× (baseline) |
| taichi | CPU | ~8 min | 2.5× |
| **taichi** | **GPU (AMD)** | **~2 min** | **10×** |
| taichi | GPU (NVIDIA) | ~1 min (est.) | 20× (estimated) |

**GPU makes video generation practical!** 🚀

## See Also

- `PERF.md` - Detailed performance benchmarks
- `GPU_BACKENDS.md` - GPU backend setup and troubleshooting
- `render.py` - Single image rendering tool
- `benchmark_breakeven_direct.py` - Performance analysis
