"""Taichi backend implementation (GPU-accelerated).

This backend provides GPU acceleration using Taichi for
vectorized computations.

Note: This is a work-in-progress. Not all functions are fully implemented yet.
"""

import numpy as np

try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False

from luminet.backends.base import BaseBackend


class TaichiBackend(BaseBackend):
    """Taichi-based computational backend with GPU support."""

    def __init__(self, arch="cuda"):
        """Initialize Taichi backend.

        Args:
            arch: Architecture to use ('cuda', 'gpu', 'cpu', 'vulkan', etc.)
        """
        super().__init__()

        if not TAICHI_AVAILABLE:
            raise ImportError("Taichi is not installed. Install with: pip install taichi")

        # Initialize Taichi
        ti.init(arch=arch, print_ir=False)
        self.name = "taichi"
        self.arch = arch

    def calc_q(self, p, bh_mass):
        """Convert periastron P to Q."""
        if isinstance(p, np.ndarray):
            return self._calc_q_vectorized(p, bh_mass)
        else:
            return self._calc_q_scalar(p, bh_mass)

    def _calc_q_scalar(self, p, bh_mass):
        """Scalar implementation of calc_q."""
        if p < 2.0 * bh_mass:
            return np.nan
        return np.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))

    def _calc_q_vectorized(self, p, bh_mass):
        """Vectorized implementation of calc_q."""
        q = np.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
        q[p < 2.0 * bh_mass] = np.nan
        return q

    def calc_k_squared(self, p, bh_mass):
        """Calculate squared modulus of elliptic integral."""
        q = self.calc_q(p, bh_mass)
        if isinstance(p, np.ndarray):
            result = (q - p + 6 * bh_mass) / (2 * q)
            result[np.isnan(q)] = np.nan
            return result
        else:
            if q is np.nan:
                return np.nan
            return (q - p + 6 * bh_mass) / (2 * q)

    def calc_zeta_inf(self, p, bh_mass):
        """Calculate zeta_infinity."""
        q = self.calc_q(p, bh_mass)
        if isinstance(p, np.ndarray):
            arg = (q - p + 2 * bh_mass) / (q - p + 6 * bh_mass)
            z_inf = np.arcsin(np.sqrt(arg))
            z_inf[np.isnan(arg)] = np.nan
            return z_inf
        else:
            if q is np.nan:
                return np.nan
            arg = (q - p + 2 * bh_mass) / (q - p + 6 * bh_mass)
            return np.arcsin(np.sqrt(arg))

    def calc_sn(self, p, angle, bh_mass, incl, order=0):
        """Calculate Jacobi elliptic function sn.

        TODO: Implement GPU-accelerated version.
        Currently falls back to scipy.
        """
        # Fallback to scipy implementation
        from scipy.special import ellipj, ellipk, ellipkinc

        q = self.calc_q(p, bh_mass)
        if np.isnan(q):
            return np.nan

        z_inf = self.calc_zeta_inf(p, bh_mass)
        m = self.calc_k_squared(p, bh_mass)
        ell_inf = ellipkinc(z_inf, m)

        cos_gamma = np.cos(angle) / np.sqrt(np.cos(angle) ** 2 + 1 / (np.tan(incl) ** 2))
        g = np.arccos(cos_gamma)

        if order == 0:
            ellips_arg = g / (2.0 * np.sqrt(p / q)) + ell_inf
        else:
            ell_k = ellipk(m)
            ellips_arg = (g - 2.0 * order * np.pi) / (2.0 * np.sqrt(p / q)) - ell_inf + 2.0 * ell_k

        sn, _, _, _ = ellipj(ellips_arg, m)
        return sn

    def periastron_cost(self, p, radius, angle, bh_mass, incl, order=0):
        """Cost function for periastron optimization."""
        q = self.calc_q(p, bh_mass)
        if np.isnan(q):
            return np.nan

        sn = self.calc_sn(p, angle, bh_mass, incl, order)
        term1 = -(q - p + 2.0 * bh_mass)
        term2 = (q - p + 6.0 * bh_mass) * sn * sn
        zero_opt = 4.0 * bh_mass * p - radius * (term1 + term2)
        return zero_opt

    def solve_for_periastron(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for periastron given black hole coordinates.

        Uses scipy for accuracy (GPU implementation TODO).
        """
        from luminet.solver import improve_solutions

        if radius <= 3 * bh_mass:
            return np.nan

        min_periastron = 3.0 * bh_mass + order * 1e-5
        periastron_initial_guess = np.linspace(min_periastron, radius, 2)

        y = np.array([
            self.periastron_cost(periastron_guess, radius, alpha, bh_mass, incl, order)
            for periastron_guess in periastron_initial_guess
        ])

        if any(np.isnan(y)):
            return np.nan

        if np.sign(y[0]) == np.sign(y[1]):
            return np.nan

        kwargs_eq13 = {
            "radius": radius,
            "angle": alpha,
            "bh_mass": bh_mass,
            "incl": incl,
            "order": order,
        }

        periastron = improve_solutions(
            func=self.periastron_cost,
            x=periastron_initial_guess,
            y=y,
            kwargs=kwargs_eq13,
        )

        return periastron

    def solve_for_impact_parameter(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for impact parameter b."""
        if order % 2 == 1:
            alpha = (alpha + np.pi) % (2 * np.pi)

        periastron_solution = self.solve_for_periastron(radius, incl, alpha, bh_mass, order)

        if np.isnan(periastron_solution):
            if order == 0 and ((alpha < np.pi / 2) or (alpha > 3 * np.pi / 2)):
                return self._ellipse(radius, alpha, incl)
            else:
                return np.nan

        b = np.sqrt(periastron_solution**3 / (periastron_solution - 2.0 * bh_mass))
        return b

    def _ellipse(self, r, a, incl):
        """Equation of an ellipse (Newtonian limit)."""
        a = (a + np.pi / 2) % (2 * np.pi)
        major_axis = r
        minor_axis = abs(major_axis * np.cos(incl))
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        return minor_axis / np.sqrt((1 - (eccentricity * np.cos(a)) ** 2))

    def calc_redshift_factor(self, radius, angle, incl, bh_mass, b):
        """Calculate gravitational redshift factor (1+z)."""
        z_factor = (
            1.0 + np.sqrt(bh_mass / (radius**3)) * b * np.sin(incl) * np.sin(angle)
        ) * (1 - 3.0 * bh_mass / radius) ** -0.5
        return z_factor

    def calc_flux_intrinsic_swarzschild(self, radius, acc, bh_mass):
        """Calculate intrinsic flux for Schwarzschild black hole."""
        r_ = radius / bh_mass
        log_arg = (np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3)) / ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
        A = 3 * bh_mass * acc / (8 * np.pi) / ((r_ - 3) * r_**2.5)
        f = A * (np.sqrt(r_) - np.sqrt(6) + (np.sqrt(3) / 2) * np.log(log_arg))
        return f

    def calc_flux_observed(self, radius, acc, bh_mass, redshift_factor):
        """Calculate observed flux with redshift correction."""
        flux_intr = self.calc_flux_intrinsic_swarzschild(radius=radius, acc=acc, bh_mass=bh_mass)
        flux_observed = flux_intr / redshift_factor**4
        return flux_observed

    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        return "taichi"

    def supports_vectorization(self) -> bool:
        """Check if backend supports vectorized operations."""
        return True

    def supports_gpu(self) -> bool:
        """Check if backend can run on GPU."""
        return self.arch in ["cuda", "gpu", "vulkan", "metal"]
