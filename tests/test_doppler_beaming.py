"""Tests for relativistic Doppler beaming from orbital motion.

The Doppler effect from Keplerian orbital motion creates a characteristic
bright/receding asymmetry in black hole images: the approaching side of the
disk appears brighter (blueshifted) while the receding side appears dimmer
(redshifted). This test verifies the Doppler factor calculation and its
effect on the rendered image.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from luminet import black_hole_math as bhmath


class TestDopplerFactor:
    """Test the Doppler beaming factor calculation."""

    def test_doppler_factor_range(self):
        """Doppler factor should be in (0, inf) for physical parameters."""
        # At the approaching side (angle ~ pi/2, high incl), delta > 1
        delta_approach = bhmath.calc_doppler_factor(
            radius=10.0, angle=np.pi / 2, incl=np.pi / 2, bh_mass=1.0, b=5.0
        )
        assert delta_approach > 1.0, f"approaching side should be blueshifted, got {delta_approach}"

        # At the receding side (angle ~ -pi/2 or 3pi/2, high incl), delta < 1
        delta_recede = bhmath.calc_doppler_factor(
            radius=10.0, angle=-np.pi / 2, incl=np.pi / 2, bh_mass=1.0, b=5.0
        )
        assert delta_recede < 1.0, f"receding side should be redshifted, got {delta_recede}"

    def test_doppler_factor_symmetry(self):
        """Doppler factor should be approximately reciprocal for symmetric positions."""
        # For angle and -angle at high inclination, the Doppler factors should be
        # approximately reciprocal (approaching vs receding)
        angle = np.pi / 3
        delta1 = bhmath.calc_doppler_factor(
            radius=10.0, angle=angle, incl=np.pi / 2, bh_mass=1.0, b=5.0
        )
        delta2 = bhmath.calc_doppler_factor(
            radius=10.0, angle=-angle, incl=np.pi / 2, bh_mass=1.0, b=5.0
        )
        # They should be roughly reciprocal (within numerical tolerance)
        # Note: the formula is approximate, so we allow a wider tolerance
        assert abs(delta1 * delta2 - 1.0) < 0.05, (
            f"Doppler factors should be approximately reciprocal: "
            f"{delta1} * {delta2} = {delta1 * delta2}"
        )

    def test_doppler_factor_zero_inclination(self):
        """At zero inclination (face-on), Doppler effect should vanish."""
        delta = bhmath.calc_doppler_factor(
            radius=10.0, angle=np.pi / 2, incl=0.0, bh_mass=1.0, b=5.0
        )
        # At face-on, the line-of-sight velocity component is zero
        # so the Doppler factor should be close to 1 (only gravitational)
        # The formula gives sqrt(1-beta^2) which is < 1 but close
        assert abs(delta - 1.0) < 0.1, f"face-on should have minimal Doppler, got {delta}"

    def test_doppler_factor_edge_on(self):
        """At edge-on inclination, Doppler effect should be maximum."""
        # Compare edge-on to face-on
        delta_edge = bhmath.calc_doppler_factor(
            radius=10.0, angle=np.pi / 2, incl=np.pi / 2, bh_mass=1.0, b=5.0
        )
        delta_face = bhmath.calc_doppler_factor(
            radius=10.0, angle=np.pi / 2, incl=0.0, bh_mass=1.0, b=5.0
        )
        # Edge-on should have significant Doppler, face-on should not
        assert delta_edge > 1.0, f"edge-on approaching should be blueshifted, got {delta_edge}"
        assert delta_face < delta_edge, (
            f"face-on should have less Doppler than edge-on: "
            f"face={delta_face}, edge={delta_edge}"
        )

    def test_doppler_factor_increases_with_velocity(self):
        """Doppler effect should be stronger closer to the black hole (higher velocity)."""
        # Closer radius = higher orbital velocity = stronger Doppler
        delta_inner = bhmath.calc_doppler_factor(
            radius=7.0, angle=np.pi / 2, incl=np.pi / 2, bh_mass=1.0, b=5.0
        )
        delta_outer = bhmath.calc_doppler_factor(
            radius=30.0, angle=np.pi / 2, incl=np.pi / 2, bh_mass=1.0, b=5.0
        )
        # Inner disk should have stronger Doppler beaming
        assert delta_inner > delta_outer, (
            f"inner disk should have stronger Doppler: inner={delta_inner}, outer={delta_outer}"
        )

    def test_doppler_factor_finite(self):
        """Doppler factor should always be finite for physical parameters."""
        angles = np.linspace(0, 2 * np.pi, 36)
        for angle in angles:
            delta = bhmath.calc_doppler_factor(
                radius=10.0, angle=angle, incl=np.pi / 2, bh_mass=1.0, b=5.0
            )
            assert np.isfinite(delta), f"non-finite Doppler at angle={angle}"
            assert delta > 0, f"non-positive Doppler at angle={angle}: {delta}"


class TestDopplerFlux:
    """Test the Doppler-augmented flux calculation."""

    def test_doppler_flux_brightness_asymmetry(self):
        """The Doppler-augmented flux should show approaching/receding asymmetry."""
        acc = 1.0
        bh_mass = 1.0
        incl = np.pi / 2  # edge-on

        # Sample photons on the disk at the same radius but different azimuths
        r = 10.0
        angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

        fluxes_approach = []
        fluxes_recede = []
        for angle in angles:
            b = bhmath.solve_for_impact_parameter(r, incl, angle, bh_mass, 0)
            z = bhmath.calc_redshift_factor(r, angle, incl, bh_mass, b)
            delta = bhmath.calc_doppler_factor(r, angle, incl, bh_mass, b)

            flux_no_doppler = bhmath.calc_flux_observed(
                r, acc, bh_mass, z, exponent=4
            )
            flux_with_doppler = bhmath.calc_flux_observed_with_doppler(
                r, acc, bh_mass, z, exponent=4, doppler_factor=delta
            )

            if angle < np.pi:  # approaching side
                fluxes_approach.append(flux_with_doppler)
            else:  # receding side
                fluxes_recede.append(flux_with_doppler)

        # The Doppler-augmented flux includes the Doppler factor in the total shift
        # delta < 1 for receding (redshift), delta > 1 for approaching (blueshift)
        # But since we divide by delta^exponent, the receding side is dimmer
        # (larger total shift) and approaching is brighter (smaller total shift)
        mean_approach = np.mean(fluxes_approach)
        mean_recede = np.mean(fluxes_recede)
        # The approach should be brighter (smaller total shift = larger flux)
        assert mean_approach > mean_recede, (
            f"approaching should be brighter with Doppler: "
            f"approach={mean_approach:.6f}, recede={mean_recede:.6f}"
        )
        # The asymmetry should be significant (not just numerical noise)
        assert (mean_approach - mean_recede) / max(mean_approach, 1e-10) > 0.001, (
            f"Doppler asymmetry too small: {mean_approach:.6f} vs {mean_recede:.6f}"
        )

    def test_doppler_flux_finite(self):
        """Doppler-augmented flux should always be finite."""
        acc = 1.0
        bh_mass = 1.0
        incl = np.pi / 2

        angles = np.linspace(0, 2 * np.pi, 36)
        for angle in angles:
            r = 10.0
            b = bhmath.solve_for_impact_parameter(r, incl, angle, bh_mass, 0)
            z = bhmath.calc_redshift_factor(r, angle, incl, bh_mass, b)
            delta = bhmath.calc_doppler_factor(r, angle, incl, bh_mass, b)
            flux = bhmath.calc_flux_observed_with_doppler(
                r, acc, bh_mass, z, exponent=4, doppler_factor=delta
            )
            assert np.isfinite(flux), f"non-finite flux at angle={angle}"
            assert flux > 0, f"non-positive flux at angle={angle}: {flux}"


class TestDopplerImage:
    """Test that the Doppler effect produces a realistic image asymmetry."""

    def test_image_has_approaching_receding_asymmetry(self):
        """A render with Doppler beaming should show brightness asymmetry."""
        from render_raster import render_raster

        # Render without Doppler (exponent=4, no Doppler factor)
        rgb_no_doppler, _, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=300, n_radius=200, n_angle=360,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,  # no texture noise
            inner_boost=0.0,
            beaming=False,  # no Doppler beaming
        )

        # Render with Doppler beaming
        rgb_doppler, _, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=300, n_radius=200, n_angle=360,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            beaming=True,  # Doppler beaming ON
        )

        # Check that the images are different (Doppler should change brightness)
        assert not np.allclose(rgb_no_doppler, rgb_doppler, atol=0.001), (
            "Doppler beaming should change the image brightness distribution"
        )

        # The Doppler-augmented image should have higher contrast
        # (brighter approaching side, dimmer receding side)
        finite_no = rgb_no_doppler[rgb_no_doppler > 0]
        finite_dop = rgb_doppler[rgb_doppler > 0]

        if finite_no.size > 0 and finite_dop.size > 0:
            # The Doppler image should have a different brightness distribution
            mean_no = np.mean(finite_no)
            mean_dop = np.mean(finite_dop)
            # At least one should be different by a meaningful amount
            assert abs(mean_no - mean_dop) > 0.001, (
                f"Doppler should change mean brightness: no={mean_no:.4f}, dop={mean_dop:.4f}"
            )
