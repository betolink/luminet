"""JAX backend implementation.

This backend provides JAX-based computations with automatic
vectorization and JIT compilation.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from luminet.backends.base import BaseBackend


class JAXBackend(BaseBackend):
    """JAX-based computational backend.

    JAX provides:
    - Automatic vectorization via vmap
    - JIT compilation for performance
    - GPU/TPU support (via XLA)
    """

    def __init__(self, use_jit=True, use_gpu=False):
        """Initialize JAX backend.

        Args:
            use_jit: Whether to use JIT compilation (default: True)
            use_gpu: Whether to use GPU (default: False)
        """
        super().__init__()

        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. "
                "Install with: pip install jax jaxlib"
            )

        self.name = "jax"
        self.use_jit = use_jit
        self.use_gpu = use_gpu

        # Configure JAX
        if use_gpu:
            jax.config.update('jax_platform_name', 'gpu')
            print("JAX configured for GPU")
        else:
            jax.config.update('jax_platform_name', 'cpu')
            print("JAX configured for CPU")

        if use_jit:
            # JIT compile core functions
            self._calc_q_jit = jit(self._calc_q_impl)
            self._calc_k_squared_jit = jit(self._calc_k_squared_impl)
            self._calc_sn_jit = jit(self._calc_sn_impl)
        else:
            self._calc_q_jit = self._calc_q_impl
            self._calc_k_squared_jit = self._calc_k_squared_impl
            self._calc_sn_jit = self._calc_sn_impl

    def calc_q(self, p, bh_mass):
        """Convert periastron P to Q.

        Uses numpy/scipy for fallback, JAX for actual computation.
        """
        # Check if input is array
        is_array = isinstance(p, np.ndarray) or (hasattr(p, 'shape') and len(p.shape) > 0)

        if is_array:
            # Vectorized computation
            p_array = jnp.array(p)
            result = self._calc_q_jit(p_array, bh_mass)
            return np.array(result)
        else:
            # Scalar computation
            p_scalar = float(p)
            result = self._calc_q_jit(p_scalar, bh_mass)
            return float(result)

    def _calc_q_impl(self, p, bh_mass):
        """Internal JAX implementation of calc_q."""
        mask = p < 2.0 * bh_mass
        q = jnp.where(
            mask,
            jnp.nan,
            jnp.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
        )
        return q

    def calc_k_squared(self, p, bh_mass):
        """Calculate squared modulus of elliptic integral."""
        q = self.calc_q(p, bh_mass)
        if isinstance(q, np.ndarray):
            q_jax = jnp.array(q)
            result = (q_jax - p + 6 * bh_mass) / (2 * q_jax)
            return np.array(result)
        else:
            if q is np.nan or np.isnan(q):
                return np.nan
            q_jax = jnp.array(q)
            result = (q_jax - p + 6 * bh_mass) / (2 * q_jax)
            return float(result)

    def _calc_k_squared_impl(self, p, q, bh_mass):
        """Internal JAX implementation of calc_k_squared."""
        mask = jnp.isnan(q)
        result = jnp.where(
            mask,
            jnp.nan,
            (q - p + 6 * bh_mass) / (2 * q)
        )
        return result

    def calc_zeta_inf(self, p, bh_mass):
        """Calculate zeta_infinity."""
        q = self.calc_q(p, bh_mass)

        if isinstance(q, np.ndarray):
            q_jax = jnp.array(q)
            arg = (q_jax - p + 2 * bh_mass) / (q_jax - p + 6 * bh_mass)
            z_inf = jnp.arcsin(jnp.sqrt(arg))
            return np.array(z_inf)
        else:
            if q is np.nan or np.isnan(q):
                return np.nan
            q_jax = jnp.array(q)
            arg = (q_jax - p + 2 * bh_mass) / (q_jax - p + 6 * bh_mass)
            z_inf = jnp.arcsin(jnp.sqrt(arg))
            return float(z_inf)

    def calc_sn(self, p, angle, bh_mass, incl, order=0):
        """Calculate Jacobi elliptic function sn.

        Note: JAX doesn't have elliptic functions yet.
        Falls back to scipy.special for now.
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

    def _calc_sn_impl(self, p, angle, bh_mass, incl, order):
        """Internal JAX implementation of calc_sn.

        TODO: Implement when JAX has elliptic functions.
        For now, this is a placeholder.
        """
        raise NotImplementedError(
            "JAX doesn't have elliptic functions yet. "
            "Falling back to scipy in calc_sn()."
        )

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

        TODO: Implement JAX-based root finding.
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

        This can be vectorized in JAX.
        """
        # Check if inputs are arrays
        is_array = isinstance(radius, np.ndarray) or isinstance(b, np.ndarray)

        if is_array:
            radius_jax = jnp.array(radius)
            angle_jax = jnp.array(angle)
            b_jax = jnp.array(b)
            z_factor = (
                1.0 + jnp.sqrt(bh_mass / radius_jax**3) * b_jax * jnp.sin(incl) * jnp.sin(angle_jax)
            ) * (1.0 - 3.0 * bh_mass / radius_jax) ** -0.5
            return np.array(z_factor)
        else:
            z_factor = (
                1.0 + np.sqrt(bh_mass / radius**3) * b * np.sin(incl) * np.sin(angle)
            ) * (1.0 - 3.0 * bh_mass / radius) ** -0.5
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
        return "jax"

    def supports_vectorization(self) -> bool:
        """Check if backend supports vectorized operations.

        JAX has excellent vectorization support via vmap.
        """
        return True

    def supports_gpu(self) -> bool:
        """Check if backend can run on GPU.

        JAX supports GPU via XLA (CUDA, ROCm, Metal).
        """
        return self.use_gpu
