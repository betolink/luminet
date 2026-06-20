#!/usr/bin/env python
r"""Physically-based raster renderer for a Schwarzschild thin-disk black hole.

Instead of drawing colored isoradial *lines* (the original Luminet-1979 style)
or forward-splatting disk samples (which can neither occlude nor stay smooth),
this is a **backward per-pixel ray tracer**:

    for each observer-plane pixel (x, y):
        b = sqrt(x^2 + y^2)              # impact parameter
        alpha = atan2(x, -y)             # disk azimuth (alpha=0 at the south)
        for order in [0 (direct), 1 (ghost)]:
            invert the lensing map  b = lens(r, alpha, order)
            to find the disk radius  r  that emits the photon reaching here
        visible emission = the *direct* image if it exists (it is in front),
                           else the *ghost* image if it exists,
                           else shadow (black).
        shade the pixel with the observed flux F_o = F_s(r) / (1+z)^4.

The lensing inversion is well-posed because ``b(r; alpha, order)`` is monotonic
in ``r`` for every ``alpha`` (verified numerically).  We precompute the forward
map on a dense ``(alpha, r)`` polar grid, invert it column-by-column with
``np.interp`` (vectorised), shade in polar, then resample to the Cartesian
pixel grid with ``scipy.interpolate.RegularGridInterpolator``.  The whole
pipeline is vectorised numpy/scipy -- no per-pixel Python loop -- and is
trivially portable to a GPU kernel later.

Occlusion is physical: the direct geodesic reaches the camera before any
higher-order wraparound, so where the direct image has a solution it occludes
the ghost.  The shadow (b < 3*sqrt(3)*M, no disk geodesic reaches the camera)
stays black, and the front of the disk (Newtonian-ellipse region, small b at
alpha~0) correctly covers the lower part of the shadow.

Usage
-----
    python render_raster.py --backend=numba --size=900 --output=bh_raster.png
    python render_raster.py --inclination=80 --cmap=inferno --size=1200
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import luminet.black_hole_math as bhmath


# ---------------------------------------------------------------------------
# Realism helpers: disk-space density noise, blackbody color, filmic tone-map
# ---------------------------------------------------------------------------

def _disk_noise(a_grid, r_grid, octaves=5, anisotropy=4.0, seed=0,
                  direction="azimuthal", ridged=False):
    """Periodic, anisotropic fractal noise in *disk* coordinates (alpha, r).

    The noise is generated in disk polar coordinates so that when it is
    carried through the lensing map it is naturally distorted by spacetime
    curvature (clumps get stretched around the photon sphere, etc.) -- this
    is what makes the texture look physical rather than painted on the screen.

    direction="azimuthal" suppresses high *alpha* frequencies relative to
    radial ones, producing features elongated tangentially (differential-
    rotation shear).  direction="radial" does the opposite: it suppresses high
    *radial* frequencies, producing radially-stretched streaks -- the look of
    matter streaming inward / spiralling into the hole at relativistic speed.

    Returns an array of shape (len(a_grid), len(r_grid)) with values roughly
    in [-1, 1] and no seam at alpha = 0/2*pi.
    """
    n_a, n_r = len(a_grid), len(r_grid)
    rng = np.random.default_rng(seed)
    # Complex white noise with random phase.
    field = (rng.standard_normal((n_a, n_r))
             + 1j * rng.standard_normal((n_a, n_r)))
    spec = np.fft.fft2(field)
    fa = np.fft.fftfreq(n_a)[:, None] * n_a     # cycles per full alpha turn
    fr = np.fft.fftfreq(n_r)[None, :] * n_r     # cycles across r span
    # 1/f^beta fractal spectrum.
    beta = 2.0 + 0.4 * octaves
    if direction == "radial":
        # Stretch features along r: suppress high radial frequencies.
        denom = 1.0 + fa ** 2 + (fr * anisotropy) ** 2
    else:
        # Stretch features along alpha (tangential): suppress high alpha freqs.
        denom = 1.0 + (fa * anisotropy) ** 2 + fr ** 2
    spec *= denom ** (-beta / 2.0)
    # Low-pass the very highest frequencies (both axes) to avoid aliasing once
    # the noise is lensed (the photon sphere magnifies small scales a lot).
    spec *= np.exp(-(fr / (0.45 * n_r)) ** 2)
    spec *= np.exp(-(fa / (0.45 * n_a)) ** 2)
    noise = np.real(np.fft.ifft2(spec))
    if ridged:
        # Ridged multifractal: sharp bright filaments at the zero-crossings of
        # the noise.  ridge = 1 - |n| is large (bright) only along thin lines,
        # giving the particle/filament look of discrete matter strands rather
        # than smooth clumps.  Map back to ~[-1, 1] for the lognormal modulator.
        noise = 1.0 - np.abs(noise)
        noise = 2.0 * (noise - noise.mean())
    else:
        noise -= noise.mean(axis=0, keepdims=True)
    # Detrend each radial band so the lognormal modulation exp(amp*N) has no
    # systematic azimuthal bias (which would otherwise amplify Doppler beaming
    # one-sidedly and look like a camera artifact rather than disk texture).
    if not ridged:
        noise -= noise.mean(axis=0, keepdims=True)
    noise /= (np.abs(noise).max() + 1e-12)
    return noise


# Blackbody sRGB approximation (Tanner Helland), vectorized over T in Kelvin.
_BB_COEFFS = np.array([
    # (T breakpoint, R, G, B) -- piecewise-linear in log-T
    (1000, 1.000, 0.392, 0.000),
    (1500, 1.000, 0.566, 0.148),
    (2000, 1.000, 0.685, 0.312),
    (2500, 1.000, 0.761, 0.443),
    (3000, 1.000, 0.817, 0.547),
    (4000, 1.000, 0.880, 0.678),
    (5000, 1.000, 0.913, 0.763),
    (6000, 1.000, 0.937, 0.830),
    (8000, 1.000, 0.964, 0.905),
    (10000, 1.000, 0.977, 0.945),
    (15000, 1.000, 0.991, 0.980),
])
_BB_T = _BB_COEFFS[:, 0]
_BB_RGB = _BB_COEFFS[:, 1:]

def _blackbody_rgb(T):
    """sRGB color of a blackbody at temperature T (Kelvin), shape (..., 3)."""
    T = np.clip(np.asarray(T, dtype=np.float64), 800.0, 40000.0)
    rgb = np.empty(T.shape + (3,), dtype=np.float64)
    for c in range(3):
        rgb[..., c] = np.interp(T, _BB_T, _BB_RGB[:, c])
    return rgb


def _filmic(rgb_lin):
    """ACES filmic approximate tone-map, per-channel. Input is linear HDR in
    [0, inf); output is sRGB-ish [0, 1] with rolled-off highlights."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    rgb_lin = np.clip(rgb_lin, 0.0, None)
    num = rgb_lin * (a * rgb_lin + b)
    den = rgb_lin * (c * rgb_lin + d) + e
    return np.clip(num / den, 0.0, 1.0)


# A light-blue / white palette (cold blue-white -> deep blue) for a colormap
# mode that matches the reference aesthetic.  Control points in luminance.
_BLUE_WHITE = np.array([
    # (t, R, G, B)
    (0.00, 0.02, 0.05, 0.18),   # near-black deep blue
    (0.15, 0.08, 0.16, 0.42),   # dim blue
    (0.35, 0.25, 0.40, 0.72),   # mid blue
    (0.55, 0.55, 0.70, 0.92),   # light blue
    (0.75, 0.80, 0.88, 0.99),   # pale blue-white
    (0.90, 0.93, 0.96, 1.00),   # near white, faint blue tint
    (1.00, 1.00, 1.00, 1.00),   # pure white (hottest)
])
_BW_T = _BLUE_WHITE[:, 0]
_BW_RGB = _BLUE_WHITE[:, 1:]

def _bluewhite_rgb(t):
    """Light-blue/white palette color for normalized intensity t in [0,1]."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    rgb = np.empty(t.shape + (3,), dtype=np.float64)
    for c in range(3):
        rgb[..., c] = np.interp(t, _BW_T, _BW_RGB[:, c])
    return rgb


def _sample_particles(a_grid, r_grid, noise, flux_r, n_particles, seed=0):
    """Sample disk-space particles concentrated on bright ridges x flux.

    Particles are drawn with probability density proportional to
    ``max(0, noise)^2 * flux(r)`` so they cluster on the bright filament ridges
    and where the disk is luminous (inner disk), giving discrete strand-like
    points instead of a uniform scatter.  Returns arrays (r, alpha) of length
    ~n_particles (rejection sampling -> approximate count).
    """
    rng = np.random.default_rng(seed)
    n_a, n_r = noise.shape
    # Density field on the polar grid (>= 0).
    dens = np.clip(noise, 0.0, None) ** 2
    dens = dens * flux_r[None, :]
    dens /= dens.sum() + 1e-30
    # Inverse-CDF sampling on the flattened grid.
    flat = dens.ravel()
    cdf = np.cumsum(flat)
    u = rng.random(n_particles)
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 0, len(flat) - 1)
    ja = idx // n_r
    jr = idx % n_r
    # jitter within the cell so particles don't sit on grid nodes
    da_step = (a_grid[1] - a_grid[0])
    alpha = (a_grid[ja] + rng.random(n_particles) * da_step) % (2.0 * np.pi)
    # logspace jitter in r
    log_r = np.log(r_grid[jr]) + (rng.random(n_particles) - 0.5) * (
        np.log(r_grid[1]) - np.log(r_grid[0]))
    r = np.exp(log_r)
    return r, alpha


def _particle_colors(r, z_factor, t_inner, inner, redshift_tint,
                    color="blackbody", L=None):
    """Per-particle sRGB color by the chosen color mode, with a red tint on
    strongly redshifted (receding) particles.

    L (optional) is a normalized luminance per particle for the blue/cmap
    palette modes; blackbody uses T(r) instead.
    """
    if color == "blackbody":
        T = t_inner * (r / inner) ** (-0.75)
        rgb = _blackbody_rgb(T)            # (N, 3) in [0,1]
    elif color == "blue":
        rgb = _bluewhite_rgb(np.clip(L, 0.0, 1.0))
    else:  # fallback: white
        rgb = np.ones((len(r), 3))
    if redshift_tint > 0.0:
        dz = np.clip(((z_factor - 1.0) / redshift_tint) ** 6, 0.0, 1.0)
        red = np.array([1.0, 0.30, 0.20])
        rgb = rgb * (1.0 - dz[:, None]) + red * dz[:, None]
    return rgb


def _donor_stream(a_grid, r_grid, inner, outer, incl, M,
                  impact_alpha=0.0, strength=1.0, width=0.35,
                  r_attach=None, falloff=6.0):
    """A luminous stream of matter from a donor star arcing into the disk.

    The stream is modelled in *disk* coordinates (alpha, r) so that it is
    lensed with the geodesics like the rest of the disk, instead of being
    painted on the screen.  It is a Gaussian tube in alpha centred on the
    impact azimuth, brightening toward the attachment radius (where it meets
    the disk) and fading outward (toward the off-screen donor), plus a soft
    glow at the attachment point (the hot spot where infalling matter shocks
    the disk).  Returns an additive luminance field in (n_a, n_r) >= 0.

    Args:
      impact_alpha: disk azimuth (radians) where the stream hits the disk.
      strength: peak luminance of the stream (relative to disk flux scale).
      width: azimuthal Gaussian width of the stream (radians).
      r_attach: disk radius where the stream attaches (default = outer edge).
      falloff: how sharply the stream fades past the attachment radius.
    """
    if r_attach is None:
        r_attach = outer * 0.92
    n_a, n_r = len(a_grid), len(r_grid)
    # angular tube, periodic in alpha
    da = ((a_grid[:, None] - impact_alpha + np.pi) % (2 * np.pi)) - np.pi
    tube = np.exp(-0.5 * (da / width) ** 2)               # (n_a, 1)
    # radial profile: rises to a sharp peak at r_attach then falls off outward,
    # dimmer inward (the stream lives outside the disk proper, feeding it).
    ln_r = np.log(r_grid[None, :])
    ln_a = np.log(r_attach)
    ln_out = np.log(outer)
    # bright at r_attach, fades outward toward the donor
    outward = np.exp(-falloff * np.clip((ln_r - ln_a) / (ln_out - ln_a + 1e-9),
                                        0.0, 2.0))
    # small inward tail (hot spot bleeding into the outer disk)
    inward = np.exp(-3.0 * np.clip((ln_a - ln_r) / (ln_a - np.log(inner) + 1e-9),
                                   0.0, 2.0)) * 0.35
    radial = outward + inward                              # (1, n_r)
    field = strength * tube * radial
    # bright compact hot-spot at the attachment point
    hot = strength * 1.6 * tube * np.exp(-40.0 * (ln_r - ln_a) ** 2)
    return field + hot


def _forward_map(r_grid, a_grid, incl, M, order):
    """Compute b(r, alpha; order) on the polar grid.

    Returns B with shape (len(a_grid), len(r_grid)), monotonic increasing in r.
    """
    B = np.empty((len(a_grid), len(r_grid)), dtype=np.float64)
    for j, alpha in enumerate(a_grid):
        row = np.empty(len(r_grid))
        for i, r in enumerate(r_grid):
            row[i] = bhmath.solve_for_impact_parameter(r, incl, alpha, M, order)
        B[j] = row
    return B


def _invert_and_shade(B, r_grid, a_grid, b_grid, incl, M, acc, noise=None,
                      noise_amp=0.0, boost_ramp=None, stream=None,
                      stream_scale=1.0, ridged=False):
    """Invert b->r per alpha column and shade with observed flux.

    Returns (F_polar, R_polar, valid) of shape (len(a_grid), len(b_grid)).
    F_polar is the observed flux, R_polar is the disk radius that emits it
    (used downstream for blackbody coloring), and valid marks real solutions.

    If ``noise`` (shape (n_a, n_r)) is given, the flux is multiplied by a
    lognormal density modulation ``exp(amp * N)`` sampled in disk coordinates
    so the texture is lensed with the geodesics rather than painted on screen.

    If ``boost_ramp`` (shape (n_r,)) is given, the flux is additionally
    scaled by that per-radius factor (inner-edge brightness ramp).

    If ``stream`` (shape (n_a, n_r), >= 0) is given, it is *added* to the
    flux (sampled at r_of_b) to model a luminous donor stream feeding the
    disk; the stream is lensed with the geodesics like everything else.
    ``stream_scale`` sets the absolute flux units of the stream field.
    """
    n_a, n_r = B.shape
    n_b = len(b_grid)
    F_polar = np.zeros((n_a, n_b), dtype=np.float64)
    R_polar = np.zeros((n_a, n_b), dtype=np.float64)
    Z_polar = np.ones((n_a, n_b), dtype=np.float64)  # redshift factor (1+z)
    valid = np.zeros((n_a, n_b), dtype=bool)

    for j in range(n_a):
        row = B[j]
        good = np.isfinite(row) & (row > 0)
        if not np.any(good):
            continue
        b_sort = row[good]
        r_sort = r_grid[good]
        in_range = (b_grid >= b_sort.min()) & (b_grid <= b_sort.max())
        r_of_b = np.interp(b_grid[in_range], b_sort, r_sort)
        a_col = a_grid[j]
        z = bhmath.calc_redshift_factor(r_of_b, a_col, incl, M, b_grid[in_range])
        F = bhmath.calc_flux_observed(r_of_b, acc, M, z)
        F = np.where(np.isfinite(F), F, 0.0)
        F = np.clip(F, 0.0, None)
        if noise is not None and noise_amp > 0.0:
            # Sample the disk-noise field at (alpha_j, r_of_b) -- a 1-D interp
            # along r within this alpha column. r_grid is increasing (logspace).
            n_col = np.interp(r_of_b, r_grid, noise[j])
            if ridged:
                # Filament mode: a sharp power-threshold modulation so dim
                # regions go to (near) dark and only ridge peaks stay bright --
                # this produces discrete bright strands with dark gaps between
                # them instead of a smooth modulated disk.  N is ~[0,1] on ridges.
                mod = np.clip(n_col, 0.0, None) ** (1.0 + 2.0 * noise_amp)
                # keep a faint floor so gaps aren't pure black (hot floor gas)
                mod = 0.08 + 0.92 * mod
            else:
                mod = np.exp(noise_amp * n_col)
            F = F * mod
        if boost_ramp is not None:
            # inner-edge brightness ramp, interpolated at r_of_b
            F = F * np.interp(r_of_b, r_grid, boost_ramp)
        if stream is not None:
            # additive donor-stream luminance, lensed with the geodesics
            s_col = np.interp(r_of_b, r_grid, stream[j]) * stream_scale
            F = F + np.clip(s_col, 0.0, None)
        F_polar[j, in_range] = F
        R_polar[j, in_range] = r_of_b
        Z_polar[j, in_range] = z
        valid[j, in_range] = True
    return F_polar, R_polar, Z_polar, valid


def render_raster(
    mass=1.0,
    incl=1.4,
    acc=1.0,
    outer_edge=30.0,
    size=900,
    n_radius=400,
    n_angle=720,
    n_b=None,
    orders=(0, 1),
    backend="numba",
    cmap="inferno",
    gamma=0.45,
    percentile=99.5,
    bg_color="black",
    texture="none",
    noise_amp=0.7,
    noise_octaves=5,
    noise_anisotropy=4.0,
    noise_direction="azimuthal",
    noise_seed=0,
    inner_boost=0.0,
    color="cmap",
    tone="gamma",
    t_inner=9000.0,
    exposure=1.0,
    palette="default",
    stream_alpha=None,
    stream_strength=1.0,
    stream_width=0.35,
    stream_attach=None,
    stream_scale=None,
    redshift_tint=0.0,
    particles=0,
    particle_size=0.8,
    particle_brightness=1.0,
    particle_seed=1,
):
    """Render a filled black-hole image and return (rgb, stats).

    Realism levers (each can be combined):
      texture: 'none', 'noise' (isotropic-ish clumps), or 'infall'
               (radially-streaked clumps = matter streaming inward).
      inner_boost: extra brightness ramp toward the inner edge (r->r_in),
                   modelling the real disk's sharp luminosity rise near ISCO.
      color:   'cmap' (single colormap), 'blackbody' (T(r) blackbody hues),
               or 'blue' (light-blue/white palette matching a cold aesthetic).
      tone:    'gamma' (simple power) or 'filmic' (ACES highlight roll-off).
      palette: 'default' or 'blue' -- overrides color with a blue-white palette
               for the reference light-blue/white look.
      stream_*: a luminous donor stream feeding the disk at a given azimuth
               (modelled in disk coordinates so it lenses with the geodesics).
    """
    bhmath.set_backend(backend)
    M = float(mass)
    incl = float(incl)
    acc = float(acc)
    inner = 6.0 * M
    outer = float(outer_edge)
    if n_b is None:
        n_b = size

    # Polar grids. Log-space r concentrates resolution near the bright inner
    # disk; alpha is periodic so we duplicate the 0 column at 2*pi for the
    # Cartesian resampler to wrap correctly.
    r_grid = np.logspace(np.log10(inner), np.log10(outer), n_radius)
    a_grid = np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False)

    # Image plane bounds.  Use a wide canvas when a donor stream is requested
    # so the disk keeps its horizontal aspect while the stream has vertical
    # headroom above it; otherwise a square canvas.
    if stream_alpha is not None:
        b_max = 1.12 * outer          # disk fills the width
        y_margin = 0.55 * outer       # extra vertical room for the stream
        x_range = (-b_max, b_max)
        y_range = (-b_max, b_max + y_margin)
    else:
        b_max = 1.15 * outer
        x_range = (-b_max, b_max)
        y_range = (-b_max, b_max)
    b_grid = np.linspace(0.0, b_max, n_b)

    # Disk-space density texture (lensed with the geodesics).
    noise = None
    if texture in ("noise", "infall", "filament"):
        direction = "radial" if texture == "infall" else noise_direction
        ridged = (texture == "filament")
        noise = _disk_noise(a_grid, r_grid, octaves=noise_octaves,
                            anisotropy=noise_anisotropy, seed=noise_seed,
                            direction=direction, ridged=ridged)

    # Inner-edge brightness ramp (rises toward ISCO).  Computed in disk coords
    # and applied with the flux so it is lensed consistently.
    # u in [0,1]: 0 at outer edge, 1 at inner edge; ramp = 1 + boost*u^2.
    r_norm = (np.log(outer) - np.log(r_grid)) / (np.log(outer) - np.log(inner))
    boost_ramp = 1.0 + inner_boost * np.clip(r_norm, 0.0, 1.0) ** 2

    # Donor stream (lensed with the geodesics).  Built in disk coords and added
    # to the flux.  Its absolute flux scale defaults to the disk's peak flux so
    # it reads at a comparable brightness; tune with stream_strength.
    stream = None
    stream_scale_eff = 0.0
    if stream_alpha is not None:
        stream = _donor_stream(a_grid, r_grid, inner, outer, incl, M,
                               impact_alpha=stream_alpha,
                               strength=stream_strength,
                               width=stream_width,
                               r_attach=stream_attach)
        if stream_scale is None:
            # Reference disk flux at r~inner (the brightest part), order 0.
            r_ref = inner * 1.05
            z_ref = bhmath.calc_redshift_factor(r_ref, 0.0, incl, M,
                                                bhmath.solve_for_impact_parameter(
                                                    r_ref, incl, 0.0, M, 0))
            stream_scale_eff = bhmath.calc_flux_observed(r_ref, acc, M, z_ref)
        else:
            stream_scale_eff = stream_scale

    t0 = time.time()
    # Build the observed-flux field in polar coordinates for each order.
    F_orders = {}
    R_orders = {}
    Z_orders = {}
    valid_orders = {}
    for order in orders:
        B = _forward_map(r_grid, a_grid, incl, M, order)
        F_polar, R_polar, Z_polar, valid = _invert_and_shade(
            B, r_grid, a_grid, b_grid, incl, M, acc,
            noise=noise,
            noise_amp=noise_amp if texture in ("noise", "infall", "filament") else 0.0,
            boost_ramp=boost_ramp if inner_boost > 0.0 else None,
            stream=stream, stream_scale=stream_scale_eff,
            ridged=(texture == "filament"))
        F_orders[order] = F_polar
        R_orders[order] = R_polar
        Z_orders[order] = Z_polar
        valid_orders[order] = valid
    t_solve = time.time() - t0

    # Occlusion by priority: direct (order 0) is in front of the ghost.
    # Take the lowest order that has a valid solution at each polar pixel.
    F_final = np.zeros_like(F_orders[orders[0]])
    R_final = np.zeros_like(F_final)
    Z_final = np.ones_like(F_final)
    lit = np.zeros_like(F_final, dtype=bool)
    for order in orders:
        v = valid_orders[order]
        # only fill pixels not already claimed by a lower (front) order
        new = v & (~lit)
        F_final = np.where(new, F_orders[order], F_final)
        R_final = np.where(new, R_orders[order], R_final)
        Z_final = np.where(new, Z_orders[order], Z_final)
        lit = lit | new

    # Wrap alpha: append the alpha=0 column as alpha=2*pi for the interpolator.
    a_grid_w = np.concatenate([a_grid, [2.0 * np.pi]])
    F_final_w = np.vstack([F_final, F_final[:1]])
    R_final_w = np.vstack([R_final, R_final[:1]])
    Z_final_w = np.vstack([Z_final, Z_final[:1]])
    lit_w = np.vstack([lit, lit[:1]])

    # Resample polar -> Cartesian pixel grid.
    # Use a non-square canvas when stream headroom is requested.
    if stream_alpha is not None:
        x_lo, x_hi = x_range
        y_lo, y_hi = y_range
        nx = size
        ny = int(round(size * (y_hi - y_lo) / (x_hi - x_lo)))
    else:
        x_lo, x_hi = x_range
        y_lo, y_hi = y_range
        nx = ny = size
    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)
    X, Y = np.meshgrid(xs, ys)  # X[col], Y[row]; imshow origin=lower maps row->y
    Bpix = np.sqrt(X ** 2 + Y ** 2)
    # alpha = 0 at the south (y<0): screen y = -b cos(alpha), x = b sin(alpha)
    Apix = np.arctan2(X, -Y) % (2.0 * np.pi)

    rgi_F = RegularGridInterpolator(
        (a_grid_w, b_grid), F_final_w,
        bounds_error=False, fill_value=0.0)
    rgi_R = RegularGridInterpolator(
        (a_grid_w, b_grid), R_final_w,
        bounds_error=False, fill_value=0.0)
    rgi_Z = RegularGridInterpolator(
        (a_grid_w, b_grid), Z_final_w,
        bounds_error=False, fill_value=1.0)
    rgi_lit = RegularGridInterpolator(
        (a_grid_w, b_grid), lit_w.astype(np.float64),
        bounds_error=False, fill_value=0.0)
    pts = np.stack([Apix.ravel(), Bpix.ravel()], axis=-1)
    flux_map = rgi_F(pts).reshape(ny, nx)
    r_map = rgi_R(pts).reshape(ny, nx)
    z_map = rgi_Z(pts).reshape(ny, nx)   # redshift factor (1+z)
    lit_map = rgi_lit(pts).reshape(ny, nx) > 0.5

    # Screen-space donor stream: a luminous arc from the off-screen donor down
    # to the disk attachment point.  Drawn directly in the image plane (b,alpha)
    # because the part beyond the disk outer edge has no lensing-map solution.
    # It is a Gaussian tube along the radial line at alpha_stream, from r_attach
    # out to b_max, with a bright hot-spot at the attachment and a soft glow.
    if stream_alpha is not None:
        sa = float(stream_alpha)
        r_attach_eff = stream_attach if stream_attach is not None else outer * 0.92
        # angular distance of each pixel from the stream azimuth (periodic)
        da = ((Apix - sa + np.pi) % (2.0 * np.pi)) - np.pi
        tube = np.exp(-0.5 * (da / stream_width) ** 2)
        # radial profile: bright at r_attach, fading outward toward the donor
        rfrac = np.clip((Bpix - r_attach_eff) / (b_max - r_attach_eff + 1e-9),
                        0.0, 1.0)
        outward = np.exp(-3.0 * rfrac) * (Bpix >= r_attach_eff)
        # compact hot spot at the attachment radius
        hot = np.exp(-40.0 * ((Bpix - r_attach_eff) / outer) ** 2)
        stream_screen = stream_strength * stream_scale_eff * (
            tube * outward + 1.5 * tube * hot)
        # only add where we are not already lit by the disk (avoid double-bright);
        # and never let the stream claim the black-hole shadow as 'lit'.
        add = stream_screen * (~lit_map) * (Bpix > 3.0 * np.sqrt(3.0) * M * 0.9)
        flux_map = flux_map + add
        lit_map = lit_map | (add > 1e-6 * stream_scale_eff)
    t_resample = time.time() - t0 - t_solve

    # ---- Display mapping -------------------------------------------------
    finite = flux_map[flux_map > 0]
    if finite.size == 0:
        raise RuntimeError("No flux landed in the image; check parameters.")
    white = np.percentile(finite, percentile)
    L = exposure * flux_map / white  # linear luminance, ~[0, exposure+]

    # palette overrides color
    color_eff = "blue" if palette == "blue" else color

    if color_eff == "blackbody":
        # Spectral radiance ~ blackbody(T(r)) * bolometric flux.  T(r) follows
        # the Shakura-Sunyaev profile T ~ r^(-3/4): white-hot inside, orange out.
        T = t_inner * (np.where(r_map > 0, r_map / inner, 1.0)) ** (-0.75)
        T = np.where(lit_map, T, 1.0)
        bb = _blackbody_rgb(T)
        rgb_lin = bb * L[..., None]
        if tone == "filmic":
            rgb = _filmic(rgb_lin)
        else:
            rgb = np.clip(rgb_lin, 0.0, 1.0) ** gamma
    elif color_eff == "blue":
        # Light-blue / white palette: color is a function of (lensed, tone-mapped)
        # luminance, with blue in the dim/mid tones and white at the hot peaks.
        if tone == "filmic":
            img = _filmic(L)
        else:
            img = np.clip(L, 0.0, 1.0) ** gamma
        rgb = _bluewhite_rgb(img)
    else:  # cmap
        if tone == "filmic":
            img = _filmic(L)
        else:
            img = np.clip(L, 0.0, 1.0) ** gamma
        cmap_obj = plt.get_cmap(cmap)
        rgb = cmap_obj(img)[..., :3]

    # Redshift red-tint: only the *most* redshifted filaments go red (sharp
    # threshold + high power), matching the paper's "a few red bits" rather
    # than the whole receding side.  redshift_tint is the z threshold.
    if redshift_tint > 0.0:
        dz = np.clip(((z_map - 1.0) / redshift_tint) ** 6, 0.0, 1.0)
        red = np.array([1.0, 0.30, 0.20])
        rgb = rgb * (1.0 - dz[..., None]) + red * dz[..., None]

    mask = lit_map
    if bg_color == "black":
        rgb = rgb * mask[..., None]
    else:
        rgb = rgb * mask[..., None] + (1.0 - mask[..., None])

    # ---- Particle overlay (discrete filament points on the smooth base) -----
    # Particles are sampled in disk space, lensed to the screen, and colored by
    # blackbody T(r) with a redshift red-tint on receding matter.  They cluster
    # on bright ridges of a ridged noise field so they read as filaments rather
    # than a uniform scatter, and concentrate where the disk flux is high.
    particle_layer = None
    if particles > 0:
        pnoise = _disk_noise(a_grid, r_grid, octaves=noise_octaves,
                             anisotropy=noise_anisotropy, seed=particle_seed,
                             direction=noise_direction, ridged=True)
        # per-radius flux profile (order 0, alpha=0 reference)
        flux_r = np.array([
            bhmath.calc_flux_observed(ri, acc, M,
                bhmath.calc_redshift_factor(ri, 0.0, incl, M,
                    bhmath.solve_for_impact_parameter(ri, incl, 0.0, M, 0)))
            for ri in r_grid])
        flux_r = np.where(np.isfinite(flux_r), flux_r, 0.0)
        pr, pa = _sample_particles(a_grid, r_grid, pnoise, flux_r,
                                    n_particles=particles, seed=particle_seed)
        # lens each particle (order 0 = direct, in front) and ghost (order 1)
        layers = []
        for order, size_scale, bright_scale in ((0, 1.0, 1.0), (1, 0.7, 0.45)):
            pb = np.array([bhmath.solve_for_impact_parameter(pr[i], incl, pa[i],
                                                             M, order)
                           for i in range(len(pr))])
            ok = np.isfinite(pb) & (pb > 0)
            if not np.any(ok):
                continue
            r_ok = pr[ok]; a_ok = pa[ok]; b_ok = pb[ok]
            z_ok = bhmath.calc_redshift_factor(r_ok, a_ok, incl, M, b_ok)
            fmag = np.clip(bhmath.calc_flux_observed(r_ok, acc, M, z_ok), 0, None)
            fmag_n = fmag / (np.percentile(fmag, 95) + 1e-30)
            color_mode = "blue" if palette == "blue" else color
            cols = _particle_colors(r_ok, z_ok, t_inner, inner, redshift_tint,
                                    color=color_mode, L=fmag_n)
            # screen coords + tangential streak direction (orbital shear)
            sx = b_ok * np.sin(a_ok)
            sy = -b_ok * np.cos(a_ok)
            # size: brighter where flux is high (inner); shrink with redshift dim
            sizes = particle_size * size_scale * (0.5 + 1.5 * fmag_n)
            alphas = np.clip(0.3 + 0.7 * fmag_n, 0.0, 1.0) * bright_scale
            layers.append((sx, sy, cols, sizes, alphas))
        particle_layer = {
            "layers": layers,
            "extent": (x_lo, x_hi, y_lo, y_hi),
            "nx": nx, "ny": ny,
        }

    stats = {
        "backend": bhmath.get_current_backend().get_backend_name(),
        "n_polar_samples": len(a_grid) * len(r_grid) * len(orders),
        "solve_time_s": t_solve,
        "resample_time_s": t_resample,
        "flux_max": float(flux_map.max()),
        "flux_whitepoint": float(white),
        "lit_fraction": float(mask.mean()),
        "texture": texture,
        "color": color,
        "tone": tone,
        "n_particles": particles,
    }
    return rgb, stats, particle_layer


def main():
    p = argparse.ArgumentParser(
        description="Raster black hole renderer (backward ray tracing)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--inclination", type=float, default=80.0,
                   help="degrees (default 80)")
    p.add_argument("--accretion", type=float, default=1.0)
    p.add_argument("--outer-edge", type=float, default=30.0)
    p.add_argument("--size", type=int, default=900,
                   help="output image side in pixels")
    p.add_argument("--n-radius", type=int, default=400)
    p.add_argument("--n-angle", type=int, default=720)
    p.add_argument("--n-b", type=int, default=None)
    p.add_argument("--orders", default="0,1",
                   help="comma list of image orders (front-to-back priority)")
    p.add_argument("--backend", default="numba")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--gamma", type=float, default=0.45)
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--bg-color", default="black", choices=["black", "white"])
    # Realism levers
    p.add_argument("--texture", default="none",
                   choices=["none", "noise", "infall", "filament"],
                   help="disk-space density: none, noise (clumps), infall "
                        "(radial streaks), filament (ridged thin strands)")
    p.add_argument("--noise-amp", type=float, default=0.7,
                   help="lognormal density modulation strength")
    p.add_argument("--noise-octaves", type=int, default=5)
    p.add_argument("--noise-anisotropy", type=float, default=4.0,
                   help="stretch factor for the texture direction")
    p.add_argument("--noise-direction", default="azimuthal",
                   choices=["azimuthal", "radial"],
                   help="stretch direction for --texture=noise")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--inner-boost", type=float, default=0.0,
                   help="extra brightness ramp toward ISCO (try 1-3)")
    p.add_argument("--color", default="cmap", choices=["cmap", "blackbody"],
                   help="blackbody: T(r) hues (white-hot inside, orange out)")
    p.add_argument("--tone", default="gamma", choices=["gamma", "filmic"],
                   help="filmic: ACES highlight roll-off (less cartoonish)")
    p.add_argument("--t-inner", type=float, default=9000.0,
                   help="inner-disk blackbody temperature in K (color only)")
    p.add_argument("--exposure", type=float, default=1.0,
                   help="linear exposure before tone map (try 3-6 with filmic)")
    p.add_argument("--palette", default="default",
                   choices=["default", "blue"],
                   help="blue: light-blue/white palette (overrides --color)")
    # Donor stream (matter feeding the disk from a companion star)
    p.add_argument("--stream-alpha", type=float, default=None,
                   help="disk azimuth (deg) where the donor stream hits; "
                        "enables the stream (e.g. 270 = top)")
    p.add_argument("--stream-strength", type=float, default=1.0,
                   help="donor stream peak luminance (relative to disk peak)")
    p.add_argument("--stream-width", type=float, default=0.35,
                   help="stream azimuthal width in radians")
    p.add_argument("--stream-attach", type=float, default=None,
                   help="disk radius where the stream attaches (default outer)")
    p.add_argument("--redshift-tint", type=float, default=0.0,
                   help="red tint for strongly redshifted matter; value is the "
                        "z threshold (try 0.25-0.35; 0 = off)")
    p.add_argument("--particles", type=int, default=0,
                   help="number of discrete filament particles to overlay "
                        "on the smooth base (try 4000-15000; 0 = off)")
    p.add_argument("--particle-size", type=float, default=0.8,
                   help="base particle marker size (points)")
    p.add_argument("--particle-brightness", type=float, default=1.0,
                   help="particle alpha/brightness scale")
    p.add_argument("--particle-seed", type=int, default=1)
    p.add_argument("--output", default="bh_raster.png")
    args = p.parse_args()

    incl = np.radians(args.inclination)
    orders = tuple(int(o) for o in args.orders.split(",") if o.strip())

    print("RASTER RENDERER (backward ray tracing)")
    print("=" * 64)
    print(f"  mass={args.mass}  incl={args.inclination:.1f}deg  acc={args.accretion}")
    print(f"  outer_edge={args.outer_edge}M  image={args.size}px  orders={orders}")
    print(f"  polar grid: {args.n_radius} radii x {args.n_angle} angles")
    print(f"  backend={args.backend}  cmap={args.cmap}  gamma={args.gamma}")
    print(f"  texture={args.texture}  color={args.color}  tone={args.tone}")

    rgb, stats, particle_layer = render_raster(
        mass=args.mass, incl=incl, acc=args.accretion, outer_edge=args.outer_edge,
        size=args.size, n_radius=args.n_radius, n_angle=args.n_angle,
        n_b=args.n_b, orders=orders, backend=args.backend, cmap=args.cmap,
        gamma=args.gamma, percentile=args.percentile, bg_color=args.bg_color,
        texture=args.texture, noise_amp=args.noise_amp,
        noise_octaves=args.noise_octaves, noise_anisotropy=args.noise_anisotropy,
        noise_direction=args.noise_direction, noise_seed=args.noise_seed,
        inner_boost=args.inner_boost, color=args.color, tone=args.tone,
        t_inner=args.t_inner, exposure=args.exposure,
        palette=args.palette,
        stream_alpha=(np.radians(args.stream_alpha)
                     if args.stream_alpha is not None else None),
        stream_strength=args.stream_strength,
        stream_width=args.stream_width,
        stream_attach=args.stream_attach,
        redshift_tint=args.redshift_tint,
        particles=args.particles,
        particle_size=args.particle_size,
        particle_brightness=args.particle_brightness,
        particle_seed=args.particle_seed,
    )

    plt.figure(figsize=(8, 8), dpi=120)
    plt.imshow(rgb, origin="lower", interpolation="bilinear")
    # Particle overlay: discrete filament points colored by T(r) + redshift.
    if particle_layer is not None:
        x_lo, x_hi, y_lo, y_hi = particle_layer["extent"]
        nx, ny = particle_layer["nx"], particle_layer["ny"]
        # map disk-plane coords -> pixel coords consistent with imshow origin
        for sx, sy, cols, sizes, alphas in particle_layer["layers"]:
            px = (np.asarray(sx) - x_lo) / (x_hi - x_lo) * nx
            py = (np.asarray(sy) - y_lo) / (y_hi - y_lo) * ny
            rgba = np.concatenate([np.clip(cols, 0, 1),
                                   np.clip(alphas, 0, 1)[:, None]], axis=1)
            plt.scatter(px, py, s=sizes, c=rgba, marker=",",
                        edgecolors="none", linewidths=0)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(args.output, dpi=150, bbox_inches="tight",
                facecolor=args.bg_color, edgecolor="none", pad_inches=0)
    plt.close()

    print("-" * 64)
    print(f"  solve time    : {stats['solve_time_s']:.2f}s")
    print(f"  resample time : {stats['resample_time_s']:.2f}s")
    print(f"  flux max      : {stats['flux_max']:.3e}")
    print(f"  lit fraction  : {stats['lit_fraction']:.3%}")
    print(f"  output        : {args.output}")


if __name__ == "__main__":
    main()
