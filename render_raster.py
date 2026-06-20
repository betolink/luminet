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


def _invert_and_shade(B, r_grid, a_grid, b_grid, incl, M, acc):
    """Invert b->r per alpha column and shade with observed flux.

    Returns (F_polar, valid) of shape (len(a_grid), len(b_grid)).
    F_polar is the observed flux; valid marks pixels with a real disk solution.
    """
    n_a, n_r = B.shape
    n_b = len(b_grid)
    F_polar = np.zeros((n_a, n_b), dtype=np.float64)
    valid = np.zeros((n_a, n_b), dtype=bool)

    for j in range(n_a):
        row = B[j]
        good = np.isfinite(row) & (row > 0)
        if not np.any(good):
            continue
        b_lo, b_hi = row[good].min(), row[good].max()
        # b(r) is monotonic increasing in r -> np.interp inverts it.
        # xp must be increasing; row[good] is increasing because b is monotonic.
        b_sort = row[good]
        r_sort = r_grid[good]
        # in-range mask on the polar b grid
        in_range = (b_grid >= b_sort.min()) & (b_grid <= b_sort.max())
        r_of_b = np.interp(b_grid[in_range], b_sort, r_sort)
        a_col = a_grid[j]
        z = bhmath.calc_redshift_factor(r_of_b, a_col, incl, M, b_grid[in_range])
        F = bhmath.calc_flux_observed(r_of_b, acc, M, z)
        F = np.where(np.isfinite(F), F, 0.0)
        F_polar[j, in_range] = np.clip(F, 0.0, None)
        valid[j, in_range] = True
    return F_polar, valid


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
):
    """Render a filled black-hole image and return (rgb, stats).

    Args:
        mass: Black hole mass (G=c=1).
        incl: Observer inclination in radians.
        acc: Accretion rate.
        outer_edge: Outer disk radius in units of M.
        size: Output image side length in pixels.
        n_radius: Number of disk radii in the forward-map grid.
        n_angle: Number of disk azimuths in the forward-map grid.
        n_b: Number of polar impact-parameter samples (default = size).
        orders: Image orders to consider (0 = direct, 1 = first ghost).
        backend: Computational backend for the lensing math.
        cmap: Matplotlib colormap for display.
        gamma: Tone-curve exponent applied to normalised intensity.
        percentile: Percentile of the flux used as the display white-point.
        bg_color: Background color around the disk.

    Returns:
        (rgb, stats) where rgb is a (size, size, 3) uint8 array.
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

    b_max = 1.15 * outer
    b_grid = np.linspace(0.0, b_max, n_b)

    t0 = time.time()
    # Build the observed-flux field in polar coordinates for each order.
    F_orders = {}
    valid_orders = {}
    for order in orders:
        B = _forward_map(r_grid, a_grid, incl, M, order)
        F_polar, valid = _invert_and_shade(
            B, r_grid, a_grid, b_grid, incl, M, acc)
        F_orders[order] = F_polar
        valid_orders[order] = valid
    t_solve = time.time() - t0

    # Occlusion by priority: direct (order 0) is in front of the ghost.
    # Take the lowest order that has a valid solution at each polar pixel.
    F_final = np.zeros_like(F_orders[orders[0]])
    lit = np.zeros_like(F_final, dtype=bool)
    for order in orders:
        v = valid_orders[order]
        # only fill pixels not already claimed by a lower (front) order
        new = v & (~lit)
        F_final = np.where(new, F_orders[order], F_final)
        lit = lit | new

    # Wrap alpha: append the alpha=0 column as alpha=2*pi for the interpolator.
    a_grid_w = np.concatenate([a_grid, [2.0 * np.pi]])
    F_final_w = np.vstack([F_final, F_final[:1]])
    lit_w = np.vstack([lit, lit[:1]])

    # Resample polar -> Cartesian pixel grid.
    xs = np.linspace(-b_max, b_max, size)
    X, Y = np.meshgrid(xs, xs)  # X[col], Y[row]; imshow origin=lower maps row->y
    Bpix = np.sqrt(X ** 2 + Y ** 2)
    # alpha = 0 at the south (y<0): screen y = -b cos(alpha), x = b sin(alpha)
    Apix = np.arctan2(X, -Y) % (2.0 * np.pi)

    rgi_F = RegularGridInterpolator(
        (a_grid_w, b_grid), F_final_w,
        bounds_error=False, fill_value=0.0)
    rgi_lit = RegularGridInterpolator(
        (a_grid_w, b_grid), lit_w.astype(np.float64),
        bounds_error=False, fill_value=0.0)
    pts = np.stack([Apix.ravel(), Bpix.ravel()], axis=-1)
    flux_map = rgi_F(pts).reshape(size, size)
    lit_map = rgi_lit(pts).reshape(size, size) > 0.5
    t_resample = time.time() - t0 - t_solve

    # ---- Display mapping -------------------------------------------------
    finite = flux_map[flux_map > 0]
    if finite.size == 0:
        raise RuntimeError("No flux landed in the image; check parameters.")
    white = np.percentile(finite, percentile)
    img = np.clip(flux_map / white, 0.0, 1.0)
    img = img ** gamma
    cmap_obj = plt.get_cmap(cmap)
    rgb = cmap_obj(img)[..., :3]
    mask = lit_map
    if bg_color == "black":
        rgb = rgb * mask[..., None]
    else:
        rgb = rgb * mask[..., None] + (1.0 - mask[..., None])

    stats = {
        "backend": bhmath.get_current_backend().get_backend_name(),
        "n_polar_samples": len(a_grid) * len(r_grid) * len(orders),
        "solve_time_s": t_solve,
        "resample_time_s": t_resample,
        "flux_max": float(flux_map.max()),
        "flux_whitepoint": float(white),
        "lit_fraction": float(mask.mean()),
    }
    return rgb, stats


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

    rgb, stats = render_raster(
        mass=args.mass, incl=incl, acc=args.accretion, outer_edge=args.outer_edge,
        size=args.size, n_radius=args.n_radius, n_angle=args.n_angle,
        n_b=args.n_b, orders=orders, backend=args.backend, cmap=args.cmap,
        gamma=args.gamma, percentile=args.percentile, bg_color=args.bg_color,
    )

    plt.figure(figsize=(8, 8), dpi=120)
    plt.imshow(rgb, origin="lower", interpolation="bilinear")
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
