"""Numba backend implementation.

This backend provides Numba JIT compilation for CPU acceleration
with vectorization support.
"""

import numpy as np

try:
    from numba import jit, vectorize
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from luminet.backends.base import BaseBackend


class NumbaBackend(BaseBackend):
    """Numba-based computational backend.

    Numba provides:
    - JIT compilation for CPU
    - Vectorization via @vectorize decorator
    - Good performance without GPU requirements
    """

    def __init__(self, fastmath=True):
        """Initialize Numba backend.

        Args:
            fastmath: Whether to use fast math optimizations (default: True)
        """
        super().__init__()

        if not NUMBA_AVAILABLE:
            raise ImportError(
                "Numba is not installed. "
                "Install with: pip install numba"
            )

        self.name = "numba"
        self.fastmath = fastmath

        # JIT compile core functions
        self._calc_q_jit = jit(nopython=True, fastmath=fastmath)(self._calc_q_impl)
        self._calc_k_squared_jit = jit(nopython=True, fastmath=fastmath)(self._calc_k_squared_impl)
        self._calc_zeta_inf_jit = jit(nopython=True, fastmath=fastmath)(self._calc_zeta_inf_impl)
        self._calc_redshift_jit = jit(nopython=True, fastmath=fastmath)(self._calc_redshift_impl)
        self._calc_flux_intrinsic_jit = jit(nopython=True, fastmath=fastmath)(self._calc_flux_intrinsic_impl)
        self._calc_flux_observed_jit = jit(nopython=True, fastmath=fastmath)(self._calc_flux_observed_impl)

        # Vectorize for array operations
        self._calc_q_vec = vectorize(['f8', 'f8'], nopython=True, fastmath=fastmath)(self._calc_q_impl)
        self._calc_k_squared_vec = vectorize(['f8', 'f8'], nopython=True, fastmath=fastmath)(self._calc_k_squared_impl)
        self._calc_redshift_vec = vectorize(['f8', 'f8', 'f8', 'f8', 'f8'], nopython=True, fastmath=fastmath)(self._calc_redshift_impl)

    def calc_q(self, p, bh_mass):
        """Convert periastron P to Q.

        Automatically chooses JIT (scalar) or vectorized (array) implementation.
        """
        # Check if input is array
        is_array = isinstance(p, np.ndarray) or (hasattr(p, 'shape') and len(p.shape) > 0)

        if is_array:
            # Vectorized computation
            p_flat = np.array(p, dtype=np.float64)
            result = self._calc_q_vec(p_flat, bh_mass)
            return np.array(result)
        else:
            # Scalar JIT computation
            p_scalar = float(p)
            result = self._calc_q_jit(p_scalar, bh_mass)
            return float(result)

    def _calc_q_impl(self, p, bh_mass):
        """Internal Numba implementation of calc_q."""
        if p < 2.0 * bh_mass:
            return np.nan
        return np.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))

    def calc_k_squared(self, p, bh_mass):
        """Calculate squared modulus of elliptic integral."""
        q = self.calc_q(p, bh_mass)

        if isinstance(q, np.ndarray) or (hasattr(q, 'shape') and len(q.shape) > 0):
            q_flat = np.array(q, dtype=np.float64)
            result = self._calc_k_squared_vec(q_flat, p, bh_mass)
            return np.array(result)
        else:
            if q is np.nan or np.isnan(q):
                return np.nan
            return self._calc_k_squared_jit(q, p, bh_mass)

    def _calc_k_squared_impl(self, q, p, bh_mass):
        """Internal Numba implementation of calc_k_squared."""
        if np.isnan(q):
            return np.nan
        return (q - p + 6 * bh_mass) / (2 * q)

    def calc_zeta_inf(self, p, bh_mass):
        """Calculate zeta_infinity."""
        q = self.calc_q(p, bh_mass)

        if isinstance(q, np.ndarray) or (hasattr(q, 'shape') and len(q.shape) > 0):
            # Use scalar version for each element
            result = np.empty_like(q)
            for i in range(len(q)):
                result[i] = self._calc_zeta_inf_jit(q[i], p[i], bh_mass)
            return result
        else:
            if q is np.nan or np.isnan(q):
                return np.nan
            return self._calc_zeta_inf_jit(q, p, bh_mass)

    def _calc_zeta_inf_impl(self, q, p, bh_mass):
        """Internal Numba implementation of calc_zeta_inf."""
        if np.isnan(q):
            return np.nan
        arg = (q - p + 2 * bh_mass) / (q - p + 6 * bh_mass)
        return np.arcsin(np.sqrt(arg))

    def calc_sn(self, p, angle, bh_mass, incl, order=0):
        """Calculate Jacobi elliptic function sn.

        Note: Numba doesn't have elliptic functions.
        Falls back to scipy.special.
        """
        from scipy.special import ellipj, ellipk, ellipkinc

        q = self.calc_q(p, bh_mass)
        if q is np.nan or np.isnan(q):
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
        if q is np.nan or np.isnan(q):
            return np.nan

        sn = self.calc_sn(p, angle, bh_mass, incl, order)
        term1 = -(q - p + 2.0 * bh_mass)
        term2 = (q - p + 6.0 * bh_mass) * sn * sn
        zero_opt = 4.0 * bh_mass * p - radius * (term1 + term2)
        return zero_opt

    def solve_for_periastron(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for periastron given black hole coordinates.

        TODO: Implement Numba-based root finding.
        Currently falls back to scipy.optimize.
        """
        from functools import partial
        import scipy.optimize as opt

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

        periastron = opt.brentq(partial(self.periastron_cost, **kwargs_eq13),
                               periastron_initial_guess[0],
                               periastron_initial_guess[1])

        return periastron

    def solve_for_impact_parameter(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for impact parameter b."""
        if order % 2 == 1:
            alpha = (alpha + np.pi) % (2 * np.pi)

        periastron_solution = self.solve_for_periastron(radius, incl, alpha, bh_mass, order)

        if periastron_solution is np.nan or np.isnan(periastron_solution):
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
        """Calculate gravitational redshift factor (1+z).

        Automatically chooses JIT (scalar) or vectorized (array) implementation.
        """
        # Check if inputs are arrays
        is_array = isinstance(radius, np.ndarray) or isinstance(b, np.ndarray)

        if is_array:
            # Vectorized computation
            radius_flat = np.array(radius, dtype=np.float64)
            angle_flat = np.array(angle, dtype=np.float64)
            b_flat = np.array(b, dtype=np.float64)
            z_factor = self._calc_redshift_vec(radius_flat, angle_flat, incl, bh_mass, b_flat)
            return np.array(z_factor)
        else:
            # Scalar JIT computation
            z_factor = self._calc_redshift_jit(radius, angle, incl, bh_mass, b)
            return float(z_factor)

    def _calc_redshift_impl(self, radius, angle, incl, bh_mass, b):
        """Internal Numba implementation of calc_redshift."""
        return (
            1.0 + np.sqrt(bh_mass / (radius**3)) * b * np.sin(incl) * np.sin(angle)
        ) * (1 - 3.0 * bh_mass / radius) ** -0.5

    def calc_flux_intrinsic_swarzschild(self, radius, acc, bh_mass):
        """Calculate intrinsic flux for Schwarzschild black hole.

        Automatically chooses JIT (scalar) or vectorized (array) implementation.
        """
        # Check if input is array
        is_array = isinstance(radius, np.ndarray)

        if is_array:
            # Vectorized computation
            radius_flat = np.array(radius, dtype=np.float64)
            flux = np.empty_like(radius_flat)
            for i in range(len(radius_flat)):
                flux[i] = self._calc_flux_intrinsic_jit(radius_flat[i], acc, bh_mass)
            return np.array(flux)
        else:
            # Scalar JIT computation
            return self._calc_flux_intrinsic_jit(radius, acc, bh_mass)

    def _calc_flux_intrinsic_impl(self, radius, acc, bh_mass):
        """Internal Numba implementation of calc_flux_intrinsic."""
        r_ = radius / bh_mass
        log_arg = (np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3)) / ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
        A = 3 * bh_mass * acc / (8 * np.pi) / ((r_ - 3) * r_**2.5)
        f = A * (np.sqrt(r_) - np.sqrt(6) + (np.sqrt(3) / 2) * np.log(log_arg))
        return f

    def calc_flux_observed(self, radius, acc, bh_mass, redshift_factor):
        """Calculate observed flux with redshift correction.

        Automatically chooses JIT (scalar) or vectorized (array) implementation.
        """
        # Check if input is array
        is_array = isinstance(radius, np.ndarray) or isinstance(redshift_factor, np.ndarray)

        if is_array:
            # Vectorized computation
            radius_flat = np.array(radius, dtype=np.float64)
            redshift_flat = np.array(redshift_factor, dtype=np.float64)
            flux = np.empty_like(radius_flat)
            for i in range(len(radius_flat)):
                flux[i] = self._calc_flux_observed_jit(radius_flat[i], acc, bh_mass, redshift_flat[i])
            return np.array(flux)
        else:
            # Scalar JIT computation
            return self._calc_flux_observed_jit(radius, acc, bh_mass, redshift_factor)

    def _calc_flux_observed_impl(self, radius, acc, bh_mass, redshift_factor):
        """Internal Numba implementation of calc_flux_observed."""
        flux_intr = self._calc_flux_intrinsic_impl(radius, acc, bh_mass)
        return flux_intr / (redshift_factor ** 4)

    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        return "numba"

    def supports_vectorization(self) -> bool:
        """Check if backend supports vectorized operations.

        Numba supports vectorization via @vectorize decorator.
        """
        return True

    def supports_gpu(self) -> bool:
        """Check if backend can run on GPU.

        Numba is CPU-only (no GPU support for this code).
        """
        return False
