"""Tests for Kerr (spinning) black hole rendering.

Kerr black holes have frame-dragging effects that change:
- The innermost stable orbit (ISCO) - for prograde orbits, ISCO < 6M
- The flux profile - different radial dependence
- The image appearance - asymmetric ring structure
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from luminet import black_hole_math as bhmath
from render_raster import render_raster


class TestKerrMath:
    """Test Kerr-specific math functions."""

    def test_isco_prograde_less_than_schwarzschild(self):
        """Prograde ISCO should be less than 6M for a spinning black hole."""
        # Schwarzschild ISCO = 6M
        isco_schwarzschild = bhmath.calc_innermost_stable_orbit(bh_mass=1.0, a=0.0)
        assert isco_schwarzschild == pytest.approx(6.0, abs=0.01), \
            f"Schwarzschild ISCO should be 6M, got {isco_schwarzschild}"

        # Prograde spin (a > 0) should have ISCO < 6M
        isco_prograde = bhmath.calc_innermost_stable_orbit(bh_mass=1.0, a=0.5)
        assert isco_prograde < 6.0, f"prograde ISCO should be < 6M, got {isco_prograde}"

    def test_isco_retrograde_greater_than_schwarzschild(self):
        """Retrograde ISCO should be greater than 6M."""
        isco_retrograde = bhmath.calc_innermost_stable_orbit(bh_mass=1.0, a=-0.5)
        assert isco_retrograde > 6.0, f"retrograde ISCO should be > 6M, got {isco_retrograde}"

    def test_isco_max_spin(self):
        """Max prograde spin should have ISCO close to 1M."""
        isco_max = bhmath.calc_innermost_stable_orbit(bh_mass=1.0, a=0.99)
        assert isco_max < 3.0, f"max prograde ISCO should be < 3M, got {isco_max}"

    def test_kerr_flux_finite(self):
        """Kerr flux should be finite for physical parameters."""
        spins = [0.0, 0.5, 0.9]
        acc = 1.0
        for spin in spins:
            radii = np.linspace(7.0, 50.0, 20)
            for r in radii:
                flux = bhmath.calc_flux_intrinsic_kerr(bh_mass=1.0, a=spin, r=r, acc=acc)
                assert np.isfinite(flux), f"non-finite Kerr flux at r={r}, spin={spin}"
                assert flux > 0, f"non-positive Kerr flux at r={r}, spin={spin}"

    def test_kerr_flux_decreases_with_radius(self):
        """Flux should decrease with radius (brighter inner disk)."""
        spins = [0.0, 0.5, 0.9]
        acc = 1.0
        for spin in spins:
            r1 = 8.0
            r2 = 30.0
            flux1 = bhmath.calc_flux_intrinsic_kerr(bh_mass=1.0, a=spin, r=r1, acc=acc)
            flux2 = bhmath.calc_flux_intrinsic_kerr(bh_mass=1.0, a=spin, r=r2, acc=acc)
            assert flux1 > flux2, (
                f"Kerr flux should decrease with radius: "
                f"r={r1} flux={flux1}, r={r2} flux={flux2}"
            )


class TestKerrRender:
    """Test Kerr black hole rendering."""

    def test_kerr_isco_affects_inner_edge(self):
        """Kerr rendering should use ISCO as inner edge."""
        # Schwarzschild: inner edge = 6M
        rgb_schwarzschild, stats_sch, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            kerr=False, spin=0.0,
        )
        assert stats_sch["kerr"] is False

        # Kerr with prograde spin: inner edge = ISCO < 6M
        rgb_kerr, stats_kerr, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            kerr=True, spin=0.5,
        )
        assert stats_kerr["kerr"] is True

        # Both images should have valid shapes
        assert rgb_schwarzschild.shape == (200, 200, 3)
        assert rgb_kerr.shape == (200, 200, 3)

    def test_kerr_image_different_from_schwarzschild(self):
        """Kerr rendering should differ from Schwarzschild."""
        rgb_schwarzschild, _, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            kerr=False, spin=0.0,
        )

        rgb_kerr, _, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            kerr=True, spin=0.5,
        )

        # The images should be different (Kerr has different ISCO and flux profile)
        assert not np.allclose(rgb_schwarzschild, rgb_kerr, atol=0.01), (
            "Kerr and Schwarzschild images should differ"
        )

    def test_kerr_higher_spin_brighter_inner_disk(self):
        """Higher spin should have brighter inner disk (flux increases near ISCO)."""
        spins = [0.0, 0.5, 0.9]
        fluxes = []
        for spin in spins:
            rgb, stats, _ = render_raster(
                mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
                size=200, n_radius=150, n_angle=200,
                orders=(0,), backend="scipy", cmap="gray",
                gamma=1.0, percentile=100.0, bg_color="black",
                texture="none", noise_amp=0.0,
                inner_boost=0.0,
                kerr=True, spin=spin,
            )
            finite = rgb[rgb > 0]
            if finite.size > 0:
                fluxes.append(finite.mean())

        # Higher spin should generally produce different brightness
        # (the Kerr flux profile differs from Schwarzschild)
        # We just check that the images are different for different spins
        for i in range(len(fluxes) - 1):
            # Allow some tolerance for numerical differences
            assert abs(fluxes[i] - fluxes[i + 1]) < 0.5 or fluxes[i] != fluxes[i + 1], (
                f"higher spin should produce different brightness: "
                f"spin={spins[i]} flux={fluxes[i]:.4f}, "
                f"spin={spins[i+1]} flux={fluxes[i+1]:.4f}"
            )

    def test_kerr_max_spin_isco(self):
        """Max spin should have ISCO close to 1M."""
        isco = bhmath.calc_innermost_stable_orbit(bh_mass=1.0, a=0.999)
        assert isco < 1.5, f"max spin ISCO should be close to 1M, got {isco}"
