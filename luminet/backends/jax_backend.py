"""JAX backend implementation.

This backend provides JAX-based computations with automatic
vectorization and JIT compilation.

Note: For complex elliptic functions and root finding, this backend
currently uses scipy as a fallback for accuracy. Pure JAX implementation
can be added later for GPU acceleration.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

from luminet.backends.base import BaseBackend


class JAXBackend(BaseBackend):
    """JAX-based computational backend.
    
    Features:
    - JIT compilation via jax.jit for simple operations
    - Automatic vectorization via jax.vmap
    - GPU/TPU support via XLA (for vectorized operations)
    
    Note: Root finding and elliptic functions use scipy for accuracy.
    """

    def __init__(self, use_gpu=False, gpu_device=None):
        """Initialize JAX backend.
        
        Args:
            use_gpu: Whether to use GPU (deprecated, use gpu_device instead)
            gpu_device: GPU device selection - 'nvidia', 'amd', or 'auto' (default: auto)
        """
        super().__init__()
        
        # Map gpu_device to JAX platform names
        if gpu_device is not None:
            self.gpu_device = gpu_device.lower()
        elif use_gpu:
            self.gpu_device = 'auto'
        else:
            self.gpu_device = 'cpu'
        
        gpu_device_map = {
            'nvidia': 'gpu',
            'amd': 'cpu',  # JAX doesn't support AMD GPU
            'auto': 'gpu',  # Auto-detect CUDA
            'cpu': 'cpu',
        }
        
        self.jax_platform = gpu_device_map.get(self.gpu_device, 'cpu')
        
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. "
                "Install with: pip install jax jaxlib"
        )
        
        self.name = "jax"
        self.use_gpu = use_gpu
        
        # Configure JAX platform (GPU/CPU)
        if use_gpu:
            jax.config.update('jax_platform_name', self.jax_platform)
        else:
            jax.config.update('jax_platform_name', 'cpu')
        
        # JIT compile simple functions
        self._calc_q_jit = jit(self._calc_q_impl)
        self._calc_k_squared_jit = jit(self._calc_k_squared_impl)
        self._calc_zeta_inf_jit = jit(self._calc_zeta_inf_impl)
        self._calc_redshift_jit = jit(self._calc_redshift_impl)
        self._calc_flux_intrinsic_jit = jit(self._calc_flux_intrinsic_impl)

    def _calc_q_impl(self, p, bh_mass):
        """Calculate Q from periastron P."""
        result = jnp.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
        return jnp.where(p < 2.0 * bh_mass, jnp.nan, result)

    def _calc_k_squared_impl(self, p, bh_mass):
        """Calculate k^2."""
        q = self._calc_q_impl(p, bh_mass)
        return (q - p + 6.0 * bh_mass) / (2.0 * q)

    def _calc_zeta_inf_impl(self, p, bh_mass):
        """Calculate zeta_infinity."""
        q = self._calc_q_impl(p, bh_mass)
        arg = (q - p + 2.0 * bh_mass) / (q - p + 6.0 * bh_mass)
        return jnp.arcsin(jnp.sqrt(arg))

    def _calc_redshift_impl(self, radius, angle, incl, bh_mass, b):
        """Calculate redshift factor (1+z)."""
        return (1.0 + jnp.sqrt(bh_mass / (radius ** 3)) * b * jnp.sin(incl) * jnp.sin(angle)) * \
               (1.0 - 3.0 * bh_mass / radius) ** -0.5

    def _calc_flux_intrinsic_impl(self, radius, acc, bh_mass):
        """Calculate intrinsic flux."""
        r_ = radius / bh_mass
        sqrt_r = jnp.sqrt(r_)
        sqrt_3 = jnp.sqrt(3.0)
        sqrt_6 = jnp.sqrt(6.0)
        
        log_arg = (sqrt_r + sqrt_3) * (sqrt_6 - sqrt_3) / \
                  ((sqrt_r - sqrt_3) * (sqrt_6 + sqrt_3))
        
        A = 3.0 * bh_mass * acc / (8.0 * jnp.pi) / ((r_ - 3.0) * r_ ** 2.5)
        return A * (sqrt_r - sqrt_6 + (sqrt_3 / 2.0) * jnp.log(log_arg))

    # ========================================================================
    # Public API
    # ========================================================================

    def calc_q(self, p, bh_mass):
        """Calculate Q from periastron P."""
        if isinstance(p, np.ndarray):
            result = self._calc_q_jit(jnp.array(p), float(bh_mass))
            return np.array(result)
        return float(self._calc_q_jit(float(p), float(bh_mass)))

    def calc_k_squared(self, p, bh_mass):
        """Calculate k^2 (elliptic modulus squared)."""
        if isinstance(p, np.ndarray):
            result = self._calc_k_squared_jit(jnp.array(p), float(bh_mass))
            return np.array(result)
        return float(self._calc_k_squared_jit(float(p), float(bh_mass)))

    def calc_zeta_inf(self, p, bh_mass):
        """Calculate zeta_infinity."""
        if isinstance(p, np.ndarray):
            result = self._calc_zeta_inf_jit(jnp.array(p), float(bh_mass))
            return np.array(result)
        return float(self._calc_zeta_inf_jit(float(p), float(bh_mass)))

    def calc_sn(self, p, angle, bh_mass, incl, order=0):
        """Calculate Jacobi elliptic function sn.
        
        Uses scipy for accuracy (JAX doesn't have elliptic functions).
        """
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
        return 4.0 * bh_mass * p - radius * (term1 + term2)

    def solve_for_periastron(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for periastron using scipy (for accuracy)."""
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
        """Newtonian ellipse approximation."""
        a = (a + np.pi / 2) % (2 * np.pi)
        major_axis = r
        minor_axis = abs(major_axis * np.cos(incl))
        
        if abs(major_axis) < 1e-15:
            return 0.0
        
        ratio = minor_axis / major_axis
        if ratio >= 1.0:
            return minor_axis
        
        eccentricity = np.sqrt(1 - ratio ** 2)
        denom = 1 - (eccentricity * np.cos(a)) ** 2
        if denom < 1e-15:
            return minor_axis
        
        return minor_axis / np.sqrt(denom)

    def calc_redshift_factor(self, radius, angle, incl, bh_mass, b):
        """Calculate gravitational redshift factor (1+z)."""
        if isinstance(radius, np.ndarray):
            result = self._calc_redshift_jit(
                jnp.array(radius), jnp.array(angle), 
                float(incl), float(bh_mass), jnp.array(b)
            )
            return np.array(result)
        return float(self._calc_redshift_jit(
            float(radius), float(angle), float(incl), float(bh_mass), float(b)
        ))

    def calc_flux_intrinsic_swarzschild(self, radius, acc, bh_mass):
        """Calculate intrinsic flux for Schwarzschild black hole."""
        if isinstance(radius, np.ndarray):
            result = self._calc_flux_intrinsic_jit(jnp.array(radius), float(acc), float(bh_mass))
            return np.array(result)
        return float(self._calc_flux_intrinsic_jit(float(radius), float(acc), float(bh_mass)))

    def calc_flux_observed(self, radius, acc, bh_mass, redshift_factor):
        """Calculate observed flux with redshift correction."""
        flux_intr = self.calc_flux_intrinsic_swarzschild(radius, acc, bh_mass)
        return flux_intr / (np.asarray(redshift_factor) ** 4)

    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        return "jax"

    def supports_vectorization(self) -> bool:
        """Check if backend supports vectorized operations."""
        return True

    def supports_gpu(self) -> bool:
        """Check if backend can run on GPU."""
        return self.use_gpu
