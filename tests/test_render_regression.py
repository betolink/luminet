"""Image-level regression tests for the luminet renderers.

These guard against the class of bug where a backend change makes the *image*
look different even though unit-level numeric tests still pass.  We render a
small image with two backends and compare pixel-by-pixel.

Run with:
    PYTHONPATH=. python -m pytest tests/test_render_regression.py -q
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

import luminet.black_hole_math as bhmath
from render_raster import render_raster


def _render_small(backend, size=120, n_radius=80, n_angle=120, seed=None):
    """Render a tiny deterministic image with a given backend."""
    if seed is not None:
        np.random.seed(seed)
    rgb, _, _ = render_raster(
        mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
        size=size, n_radius=n_radius, n_angle=n_angle,
        orders=(0, 1), backend=backend, cmap="gray",
        gamma=1.0, percentile=100.0, bg_color="black",
    )
    return rgb[..., 0]  # single channel


def test_raster_numba_matches_scipy():
    """The numba raster must match the scipy raster to within display tolerance."""
    ref = _render_small("scipy")
    test = _render_small("numba")
    assert ref.shape == test.shape
    # Relative mean absolute difference on lit pixels.
    lit = ref > 0
    assert lit.mean() > 0.05, "reference image has no flux; renderer is broken"
    # Mean abs diff normalized by peak.
    peak = ref.max()
    mean_abs = float(np.abs(ref[lit] - test[lit]).mean())
    rel = mean_abs / peak
    # numba solves to ~1e-9 vs scipy; after histogram splatting + tone map this
    # should be well under a few percent of the peak.
    assert rel < 0.05, f"numba raster diverges from scipy: rel mean abs err {rel:.4f}"


def test_raster_has_shadow_and_ring():
    """A correct render has a dark central shadow and a brighter lensed ring."""
    img = _render_small("numba", size=160, n_radius=120, n_angle=180)
    c = img.shape[0] // 2
    shadow = img[c - 12:c + 12, c - 12:c + 12].mean()
    yy, xx = np.indices(img.shape)
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    rbins = np.linspace(0, c, 30)
    idx = np.digitize(rr.ravel(), rbins)
    prof = np.array([img.ravel()[idx == i].mean() for i in range(1, len(rbins))])
    ring = prof.max()
    # At high inclination (e.g. 80deg) the front of the disk legitimately
    # fills in front of the shadow, so we only require the centre to be
    # noticeably dimmer than the ring, not empty.
    assert shadow < ring * 0.6, (
        f"shadow not darker than ring: shadow={shadow:.3f} ring={ring:.3f}")
    assert ring > 0.1, f"no bright ring found; ring={ring:.3f}"
