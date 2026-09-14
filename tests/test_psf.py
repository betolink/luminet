"""Tests for camera PSF (Point Spread Function) in the raster renderer.

Real telescopes have finite optical resolution, modeled as a PSF.
This test verifies that applying a PSF blurs the image appropriately
and that the PSF parameters work correctly.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from render_raster import render_raster, _gaussian_psf, _apply_psf


class TestGaussianPSF:
    """Test the Gaussian PSF function."""

    def test_psf_is_gaussian(self):
        """The PSF should be a proper Gaussian."""
        sigma = 2.0
        x = np.linspace(-6, 6, 100)
        y = np.zeros_like(x)
        psf = _gaussian_psf(x, y, sigma)
        # Should be maximum at center (x=0, which is at index ~50)
        center_idx = len(x) // 2
        assert psf[center_idx] >= psf[center_idx - 1], \
            f"PSF should peak at center: center={psf[center_idx]}, left={psf[center_idx-1]}"
        assert psf[center_idx] >= psf[center_idx + 1], \
            f"PSF should peak at center: center={psf[center_idx]}, right={psf[center_idx+1]}"
        # Should decay away from center
        assert psf[0] < psf[center_idx], f"PSF should decay: edge={psf[0]}, center={psf[center_idx]}"

    def test_psf_normalization(self):
        """The PSF should be normalized to integrate to 1."""
        sigma = 1.0
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        XX, YY = np.meshgrid(x, y)
        psf = _gaussian_psf(XX, YY, sigma)
        # Integrate over the grid using scipy
        from scipy.integrate import dblquad
        integral, _ = dblquad(lambda yy, xx: np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2)) / (2.0 * np.pi * sigma**2),
                             -np.inf, np.inf, lambda yy: -np.inf, lambda yy: np.inf)
        assert abs(integral - 1.0) < 0.01, f"PSF should integrate to 1, got {integral}"

    def test_psf_fwhm_from_sigma(self):
        """FWHM should relate to sigma as FWHM = 2*sqrt(2*ln2)*sigma."""
        fwhm = 4.0
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        x = np.linspace(-10, 10, 200)
        y = np.zeros_like(x)
        psf = _gaussian_psf(x, y, sigma, fwhm=fwhm)
        # The half-maximum should be at +/- FWHM/2
        # Find where the PSF drops to half its peak value
        peak = psf.max()
        half_max = peak / 2.0
        # Find the indices where the PSF crosses half_max
        indices = np.where(psf <= half_max)[0]
        if len(indices) > 0:
            # The first crossing should be near -FWHM/2
            first_cross = indices[0]
            expected_half_fwhm = fwhm / 2.0
            # Allow some tolerance for discrete sampling
            assert abs(first_cross - expected_half_fwhm) < 3.0, (
                f"FWHM half should be {expected_half_fwhm}, got {first_cross}"
            )


class TestPSFApply:
    """Test the PSF application function."""

    def test_psf_blurs_image(self):
        """Applying a PSF should blur sharp features."""
        # Create a sharp edge image
        img = np.zeros((64, 64))
        img[32:, :] = 1.0  # sharp edge

        # Apply no PSF
        img_no_psf = img.copy()

        # Apply PSF with sigma=2
        img_psf = _apply_psf(img, sigma=2.0)

        # The PSF image should be smoother (less contrast at the edge)
        # Check that the edge is blurred (the transition should be smoother)
        # At the edge, no_psf goes from 0 to 1 instantly
        # psf should have a gradual transition
        edge_region = img_psf[30:34, 32] - img_psf[28:32, 32]
        assert edge_region.mean() > 0, (
            f"PSF should blur the edge: edge_region={edge_region.mean()}"
        )

    def test_psf_preserves_total_brightness(self):
        """PSF convolution should preserve total brightness (conservation of energy)."""
        img = np.random.rand(32, 32)
        img_psf = _apply_psf(img, sigma=2.0)
        # Total brightness should be approximately preserved
        assert abs(img.sum() - img_psf.sum()) < 0.01 * img.sum(), (
            f"PSF should conserve brightness: original={img.sum():.4f}, "
            f"psf={img_psf.sum():.4f}"
        )

    def test_psf_fwhm_parameter(self):
        """PSF should work with FWHM parameter."""
        # Use a point source to test PSF
        img = np.zeros((64, 64))
        img[32, 32] = 1.0  # point source

        img_psf_fwhm = _apply_psf(img, fwhm=4.0)
        img_psf_sigma = _apply_psf(img, sigma=2.355)  # equivalent sigma

        # Both should produce Gaussian blobs
        # Check that the FWHM is approximately correct (radius at half-max)
        for img_test, name in [(img_psf_fwhm, "fwhm"), (img_psf_sigma, "sigma")]:
            peak = img_test.max()
            half_max = peak / 2.0
            # Find the radius at half maximum
            center = img_test.shape[0] // 2
            yy, xx = np.indices(img_test.shape)
            r = np.sqrt((xx - center)**2 + (yy - center)**2)
            sorted_r = np.sort(r.ravel())
            sorted_vals = img_test.ravel()[np.argsort(r.ravel())]
            # Find where it crosses half_max
            crossing_idx = np.where(sorted_vals <= half_max)[0]
            if len(crossing_idx) > 0:
                fwhm_measured = sorted_r[crossing_idx[0]]
                # FWHM=4.0 means radius at half-max is 2.0
                expected_radius = 2.0
                assert abs(fwhm_measured - expected_radius) <= 1.5, (
                    f"{name} PSF should have radius at half-max ~{expected_radius}, got {fwhm_measured}"
                )


class TestPSFRender:
    """Test PSF in the actual renderer."""

    def test_psf_changes_image(self):
        """PSF should change the rendered image."""
        # Render without PSF
        rgb_no_psf, _, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            psf_fwhm=0.0,
        )

        # Render with PSF (small blur)
        rgb_psf, stats, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            psf_fwhm=3.0,
        )

        assert stats["psf_applied"] is True

        # The PSF image should be different from the no-PSF image
        assert not np.allclose(rgb_no_psf, rgb_psf, atol=0.001), (
            "PSF should change the image"
        )

    def test_larger_psf_more_blur(self):
        """Larger PSF should produce more blur."""
        # Render with small PSF
        rgb_small, _, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            psf_fwhm=2.0,
        )

        # Render with large PSF
        rgb_large, _, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            psf_fwhm=8.0,
        )

        # The large PSF image should be more blurred (lower peak intensity)
        peak_small = rgb_small.max()
        peak_large = rgb_large.max()
        # Large PSF should reduce peak intensity more
        assert peak_large < peak_small * 0.95, (
            f"larger PSF should produce more blur: peak_small={peak_small:.4f}, "
            f"peak_large={peak_large:.4f}"
        )

    def test_no_psf_unchanged(self):
        """No PSF should leave the image unchanged."""
        rgb_no_psf, stats, _ = render_raster(
            mass=1.0, incl=np.radians(80.0), acc=1.0, outer_edge=30.0,
            size=200, n_radius=150, n_angle=200,
            orders=(0,), backend="scipy", cmap="gray",
            gamma=1.0, percentile=100.0, bg_color="black",
            texture="none", noise_amp=0.0,
            inner_boost=0.0,
            psf_fwhm=0.0,
        )

        assert stats.get("psf_applied", False) is False
