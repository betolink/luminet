"""Numba backend implementation.

This backend provides Numba JIT compilation for CPU acceleration
with vectorization support. Implements Jacobi elliptic functions
using the AGM algorithm for full Numba compatibility.
"""

import numpy as np
from numba import jit, prange
import math

from luminet.backends.base import BaseBackend

# ============================================================================
# JIT-compiled helper functions (must be defined at module level for Numba)
# ============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def _calc_q_scalar(p: float, bh_mass: float) -> float:
    """Calculate Q from periastron P."""
    if p < 2.0 * bh_mass:
        return np.nan
    return math.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))


@jit(nopython=True, fastmath=True, cache=True)
def _calc_k_squared_scalar(p: float, bh_mass: float) -> float:
    """Calculate k^2 (elliptic modulus squared)."""
    if p < 2.0 * bh_mass:
        return np.nan
    q = math.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
    return (q - p + 6.0 * bh_mass) / (2.0 * q)


@jit(nopython=True, fastmath=True, cache=True)
def _calc_zeta_inf_scalar(p: float, bh_mass: float) -> float:
    """Calculate zeta_infinity."""
    if p < 2.0 * bh_mass:
        return np.nan
    q = math.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
    arg = (q - p + 2.0 * bh_mass) / (q - p + 6.0 * bh_mass)
    return math.asin(math.sqrt(arg))


@jit(nopython=True, fastmath=True, cache=True)
def _ellipk_agm(m: float) -> float:
    """Complete elliptic integral K(m) using AGM algorithm.
    
    K(m) = pi / (2 * AGM(1, sqrt(1-m)))
    
    Args:
        m: Parameter (0 <= m < 1)
    
    Returns:
        K(m) value
    """
    if m < 0.0 or m >= 1.0:
        return np.nan
    if m == 0.0:
        return math.pi / 2.0
    
    a = 1.0
    b = math.sqrt(1.0 - m)
    
    # AGM iteration
    for _ in range(25):  # Usually converges in ~10 iterations
        a_new = (a + b) / 2.0
        b_new = math.sqrt(a * b)
        if abs(a_new - b_new) < 1e-15:
            break
        a = a_new
        b = b_new
    
    return math.pi / (2.0 * a)


@jit(nopython=True, fastmath=True, cache=True)
def _ellipkinc_agm(phi: float, m: float) -> float:
    """Incomplete elliptic integral F(phi, m) using Gauss-Legendre quadrature.
    
    F(phi, m) = integral from 0 to phi of 1/sqrt(1 - m*sin^2(t)) dt
    
    Uses 20-point Gauss-Legendre quadrature for accuracy.
    
    Args:
        phi: Amplitude (radians)
        m: Parameter (0 <= m < 1)
    
    Returns:
        F(phi, m) value
    """
    if m < 0.0 or m >= 1.0:
        return np.nan
    if phi == 0.0:
        return 0.0
    if m == 0.0:
        return phi
    
    # Handle negative phi
    sign = 1.0
    if phi < 0.0:
        sign = -1.0
        phi = -phi
    
    # Handle large phi by splitting into complete periods + remainder
    # F(n*pi + phi_rem, m) = 2*n*K(m) + F(phi_rem, m)
    K = _ellipk_agm(m)
    n_periods = int(phi / math.pi)
    phi_rem = phi - n_periods * math.pi
    
    # Now phi_rem is in [0, pi]
    # If phi_rem > pi/2, use F(pi - x, m) = 2K - F(x, m)
    if phi_rem > math.pi / 2.0:
        phi_rem = math.pi - phi_rem
        result = 2.0 * K - _ellipkinc_gauss(phi_rem, m)
    else:
        result = _ellipkinc_gauss(phi_rem, m)
    
    # Add contribution from complete periods
    result += 2.0 * n_periods * K
    
    return sign * result


@jit(nopython=True, fastmath=True, cache=True)
def _ellipkinc_gauss(phi: float, m: float) -> float:
    """Incomplete elliptic integral F(phi, m) for phi in [0, pi/2].
    
    Uses 20-point Gauss-Legendre quadrature.
    """
    if phi == 0.0:
        return 0.0
    
    # Gauss-Legendre 20-point quadrature nodes and weights for [-1, 1]
    x = np.array([
        -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
        -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
        -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
        -0.0765265211334973,  0.0765265211334973,  0.2277858511416451,
         0.3737060887154195,  0.5108670019508271,  0.6360536807265150,
         0.7463319064601508,  0.8391169718222188,  0.9122344282513259,
         0.9639719272779138,  0.9931285991850949
    ])
    w = np.array([
        0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
        0.0832767415767047, 0.1019301198172404, 0.1181945319615184,
        0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
        0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
        0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
        0.1019301198172404, 0.0832767415767047, 0.0626720483341091,
        0.0406014298003869, 0.0176140071391521
    ])
    
    # Transform [−1, 1] to [0, phi]
    result = 0.0
    half_phi = phi / 2.0
    
    for i in range(20):
        t = half_phi * (x[i] + 1.0)
        sin_t = math.sin(t)
        integrand = 1.0 / math.sqrt(1.0 - m * sin_t * sin_t)
        result += w[i] * integrand
    
    return result * half_phi


@jit(nopython=True, fastmath=True, cache=True)
def _ellipj_sn(u: float, m: float) -> float:
    """Jacobi elliptic function sn(u, m) using descending Landen transformation.
    
    This is the key function needed for the black hole simulation.
    
    Args:
        u: Argument
        m: Parameter (0 <= m < 1)
    
    Returns:
        sn(u, m) value
    """
    if m < 0.0 or m > 1.0:
        return np.nan
    if m == 0.0:
        return math.sin(u)
    if m == 1.0:
        return math.tanh(u)
    
    # Store transformation parameters
    MAX_ITER = 25
    a = np.empty(MAX_ITER)
    c = np.empty(MAX_ITER)
    
    a[0] = 1.0
    b = math.sqrt(1.0 - m)
    c[0] = math.sqrt(m)
    
    n = 0
    for i in range(MAX_ITER - 1):
        if abs(c[i]) < 1e-15:
            n = i
            break
        a[i + 1] = (a[i] + b) / 2.0
        c[i + 1] = (a[i] - b) / 2.0
        b = math.sqrt(a[i] * b)
        n = i + 1
    
    # Backward recurrence
    phi = u * a[n] * (2.0 ** n)
    
    for i in range(n, 0, -1):
        sin_phi = math.sin(phi)
        phi = (phi + math.asin(c[i] * sin_phi / a[i])) / 2.0
    
    return math.sin(phi)


@jit(nopython=True, fastmath=True, cache=True)
def _calc_sn_scalar(p: float, angle: float, bh_mass: float, incl: float, order: int) -> float:
    """Calculate Jacobi elliptic sn for given parameters."""
    if p < 2.0 * bh_mass:
        return np.nan
    
    q = math.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
    
    # zeta_inf
    arg = (q - p + 2.0 * bh_mass) / (q - p + 6.0 * bh_mass)
    z_inf = math.asin(math.sqrt(arg))
    
    # k^2 (modulus squared)
    m = (q - p + 6.0 * bh_mass) / (2.0 * q)
    
    # F(zeta_inf, m)
    ell_inf = _ellipkinc_agm(z_inf, m)
    
    # cos(gamma)
    cos_angle = math.cos(angle)
    tan_incl = math.tan(incl)
    cos_gamma = cos_angle / math.sqrt(cos_angle ** 2 + 1.0 / (tan_incl ** 2))
    g = math.acos(cos_gamma)
    
    # elliptic argument
    sqrt_p_q = math.sqrt(p / q)
    
    if order == 0:
        ellips_arg = g / (2.0 * sqrt_p_q) + ell_inf
    else:
        ell_k = _ellipk_agm(m)
        ellips_arg = (g - 2.0 * order * math.pi) / (2.0 * sqrt_p_q) - ell_inf + 2.0 * ell_k
    
    return _ellipj_sn(ellips_arg, m)


@jit(nopython=True, fastmath=True, cache=True)
def _periastron_cost_scalar(p: float, radius: float, angle: float, 
                            bh_mass: float, incl: float, order: int) -> float:
    """Cost function for periastron root finding."""
    if p < 2.0 * bh_mass:
        return np.nan
    
    q = math.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
    sn = _calc_sn_scalar(p, angle, bh_mass, incl, order)
    
    if math.isnan(sn):
        return np.nan
    
    term1 = -(q - p + 2.0 * bh_mass)
    term2 = (q - p + 6.0 * bh_mass) * sn * sn
    return 4.0 * bh_mass * p - radius * (term1 + term2)


@jit(nopython=True, fastmath=False, cache=True)
def _bisect_periastron(radius: float, angle: float, bh_mass: float, 
                       incl: float, order: int, tol: float = 1e-10, 
                       max_iter: int = 100) -> float:
    """Bisection root finder for periastron.
    
    This replaces scipy.optimize.brentq with a pure Numba implementation.
    """
    if radius <= 3.0 * bh_mass:
        return np.nan
    
    # Bracket: [3M + epsilon, radius]
    a = 3.0 * bh_mass + order * 1e-5 + 1e-10
    b = radius
    
    fa = _periastron_cost_scalar(a, radius, angle, bh_mass, incl, order)
    fb = _periastron_cost_scalar(b, radius, angle, bh_mass, incl, order)
    
    if math.isnan(fa) or math.isnan(fb):
        return np.nan
    
    # Check if root is bracketed
    if fa * fb > 0:
        return np.nan
    
    # Ensure fa < 0 and fb > 0
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa
    
    # Bisection with Brent's method speedup
    c = a
    fc = fa
    
    for _ in range(max_iter):
        if abs(b - a) < tol:
            return (a + b) / 2.0
        
        # Try inverse quadratic interpolation
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
        
        # Check if s is in bounds and making progress
        if s < min(a, b) or s > max(a, b) or abs(s - b) > abs(b - a) / 2:
            # Fall back to bisection
            s = (a + b) / 2.0
        
        fs = _periastron_cost_scalar(s, radius, angle, bh_mass, incl, order)
        
        if math.isnan(fs):
            s = (a + b) / 2.0
            fs = _periastron_cost_scalar(s, radius, angle, bh_mass, incl, order)
        
        c = b
        fc = fb
        
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
    
    return (a + b) / 2.0


@jit(nopython=True, fastmath=False, cache=True)
def _ellipse_scalar(r: float, a: float, incl: float) -> float:
    """Newtonian ellipse approximation."""
    a_shifted = (a + math.pi / 2.0) % (2.0 * math.pi)
    major_axis = r
    minor_axis = abs(major_axis * math.cos(incl))
    
    # Handle edge case where minor_axis == major_axis (incl = pi/2)
    if abs(major_axis) < 1e-15:
        return 0.0
    
    ratio = minor_axis / major_axis
    if ratio >= 1.0:
        # Circle case
        return minor_axis
    
    eccentricity = math.sqrt(1.0 - ratio ** 2)
    denom = 1.0 - (eccentricity * math.cos(a_shifted)) ** 2
    if denom < 1e-15:
        return minor_axis  # Avoid division by zero
    
    return minor_axis / math.sqrt(denom)


@jit(nopython=True, fastmath=False, cache=True)
def _solve_impact_parameter_scalar(radius: float, incl: float, alpha: float, 
                                   bh_mass: float, order: int) -> float:
    """Solve for impact parameter b."""
    alpha_adj = alpha
    if order % 2 == 1:
        alpha_adj = (alpha + math.pi) % (2.0 * math.pi)
    
    periastron = _bisect_periastron(radius, alpha_adj, bh_mass, incl, order)
    
    if math.isnan(periastron):
        if order == 0 and (alpha < math.pi / 2.0 or alpha > 3.0 * math.pi / 2.0):
            return _ellipse_scalar(radius, alpha, incl)
        else:
            return np.nan
    
    return math.sqrt(periastron ** 3 / (periastron - 2.0 * bh_mass))


@jit(nopython=True, fastmath=True, cache=True)
def _calc_redshift_scalar(radius: float, angle: float, incl: float, 
                          bh_mass: float, b: float) -> float:
    """Calculate redshift factor (1+z)."""
    return (1.0 + math.sqrt(bh_mass / (radius ** 3)) * b * math.sin(incl) * math.sin(angle)) * \
           (1.0 - 3.0 * bh_mass / radius) ** -0.5


@jit(nopython=True, fastmath=True, cache=True)
def _calc_flux_intrinsic_scalar(radius: float, acc: float, bh_mass: float) -> float:
    """Calculate intrinsic flux."""
    r_ = radius / bh_mass
    sqrt_r = math.sqrt(r_)
    sqrt_3 = math.sqrt(3.0)
    sqrt_6 = math.sqrt(6.0)
    
    log_arg = (sqrt_r + sqrt_3) * (sqrt_6 - sqrt_3) / \
              ((sqrt_r - sqrt_3) * (sqrt_6 + sqrt_3))
    
    if log_arg <= 0:
        return 0.0
    
    A = 3.0 * bh_mass * acc / (8.0 * math.pi) / ((r_ - 3.0) * r_ ** 2.5)
    return A * (sqrt_r - sqrt_6 + (sqrt_3 / 2.0) * math.log(log_arg))


# ============================================================================
# Vectorized functions using parallel loops
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _calc_q_array(p_arr: np.ndarray, bh_mass: float) -> np.ndarray:
    """Vectorized calc_q."""
    n = len(p_arr)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _calc_q_scalar(p_arr[i], bh_mass)
    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _calc_k_squared_array(p_arr: np.ndarray, bh_mass: float) -> np.ndarray:
    """Vectorized calc_k_squared."""
    n = len(p_arr)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _calc_k_squared_scalar(p_arr[i], bh_mass)
    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _calc_sn_array(p_arr: np.ndarray, angle_arr: np.ndarray, 
                   bh_mass: float, incl: float, order: int) -> np.ndarray:
    """Vectorized calc_sn."""
    n = len(p_arr)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _calc_sn_scalar(p_arr[i], angle_arr[i], bh_mass, incl, order)
    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _solve_impact_parameter_array(radius_arr: np.ndarray, incl: float, 
                                   alpha_arr: np.ndarray, bh_mass: float, 
                                   order: int) -> np.ndarray:
    """Vectorized solve_for_impact_parameter."""
    n = len(radius_arr)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _solve_impact_parameter_scalar(radius_arr[i], incl, alpha_arr[i], bh_mass, order)
    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _calc_redshift_array(radius_arr: np.ndarray, angle_arr: np.ndarray, 
                         incl: float, bh_mass: float, b_arr: np.ndarray) -> np.ndarray:
    """Vectorized calc_redshift."""
    n = len(radius_arr)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _calc_redshift_scalar(radius_arr[i], angle_arr[i], incl, bh_mass, b_arr[i])
    return result


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _calc_flux_intrinsic_array(radius_arr: np.ndarray, acc: float, bh_mass: float) -> np.ndarray:
    """Vectorized calc_flux_intrinsic."""
    n = len(radius_arr)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _calc_flux_intrinsic_scalar(radius_arr[i], acc, bh_mass)
    return result


# ============================================================================
# NumbaBackend class
# ============================================================================

class NumbaBackend(BaseBackend):
    """Numba-based computational backend.
    
    Features:
    - Full JIT compilation (no Python interpreter overhead)
    - Parallel execution via prange
    - Native Jacobi elliptic functions (no scipy dependency in hot path)
    - Bisection root finding (no scipy.optimize dependency)
    """

    def __init__(self):
        """Initialize Numba backend."""
        super().__init__()
        self.name = "numba"
        
        # Warm up JIT compilation
        self._warmup()
    
    def _warmup(self):
        """Trigger JIT compilation of all functions."""
        # Small test values to trigger compilation
        _ = _calc_q_scalar(10.0, 1.0)
        _ = _calc_sn_scalar(10.0, 0.5, 1.0, 1.2, 0)
        _ = _bisect_periastron(10.0, 0.5, 1.0, 1.2, 0)
        
        # Array warmup
        test_arr = np.array([10.0, 15.0], dtype=np.float64)
        _ = _calc_q_array(test_arr, 1.0)

    def calc_q(self, p, bh_mass):
        """Calculate Q from periastron P."""
        if isinstance(p, np.ndarray):
            return _calc_q_array(np.asarray(p, dtype=np.float64), float(bh_mass))
        return _calc_q_scalar(float(p), float(bh_mass))

    def calc_k_squared(self, p, bh_mass):
        """Calculate k^2 (elliptic modulus squared)."""
        if isinstance(p, np.ndarray):
            return _calc_k_squared_array(np.asarray(p, dtype=np.float64), float(bh_mass))
        return _calc_k_squared_scalar(float(p), float(bh_mass))

    def calc_zeta_inf(self, p, bh_mass):
        """Calculate zeta_infinity."""
        if isinstance(p, np.ndarray):
            result = np.empty(len(p), dtype=np.float64)
            for i in range(len(p)):
                result[i] = _calc_zeta_inf_scalar(float(p[i]), float(bh_mass))
            return result
        return _calc_zeta_inf_scalar(float(p), float(bh_mass))

    def calc_sn(self, p, angle, bh_mass, incl, order=0):
        """Calculate Jacobi elliptic function sn."""
        if isinstance(p, np.ndarray):
            return _calc_sn_array(
                np.asarray(p, dtype=np.float64),
                np.asarray(angle, dtype=np.float64),
                float(bh_mass), float(incl), int(order)
            )
        return _calc_sn_scalar(float(p), float(angle), float(bh_mass), float(incl), int(order))

    def periastron_cost(self, p, radius, angle, bh_mass, incl, order=0):
        """Cost function for periastron optimization."""
        return _periastron_cost_scalar(float(p), float(radius), float(angle), 
                                        float(bh_mass), float(incl), int(order))

    def solve_for_periastron(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for periastron using JIT-compiled bisection."""
        return _bisect_periastron(float(radius), float(alpha), float(bh_mass), 
                                   float(incl), int(order))

    def solve_for_impact_parameter(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for impact parameter b."""
        if isinstance(radius, np.ndarray):
            return _solve_impact_parameter_array(
                np.asarray(radius, dtype=np.float64),
                float(incl),
                np.asarray(alpha, dtype=np.float64),
                float(bh_mass), int(order)
            )
        return _solve_impact_parameter_scalar(float(radius), float(incl), float(alpha), 
                                               float(bh_mass), int(order))

    def calc_redshift_factor(self, radius, angle, incl, bh_mass, b):
        """Calculate gravitational redshift factor (1+z)."""
        if isinstance(radius, np.ndarray):
            return _calc_redshift_array(
                np.asarray(radius, dtype=np.float64),
                np.asarray(angle, dtype=np.float64),
                float(incl), float(bh_mass),
                np.asarray(b, dtype=np.float64)
            )
        return _calc_redshift_scalar(float(radius), float(angle), float(incl), 
                                      float(bh_mass), float(b))

    def calc_flux_intrinsic_swarzschild(self, radius, acc, bh_mass):
        """Calculate intrinsic flux for Schwarzschild black hole."""
        if isinstance(radius, np.ndarray):
            return _calc_flux_intrinsic_array(
                np.asarray(radius, dtype=np.float64),
                float(acc), float(bh_mass)
            )
        return _calc_flux_intrinsic_scalar(float(radius), float(acc), float(bh_mass))

    def calc_flux_observed(self, radius, acc, bh_mass, redshift_factor):
        """Calculate observed flux with redshift correction."""
        flux_intr = self.calc_flux_intrinsic_swarzschild(radius, acc, bh_mass)
        return flux_intr / (np.asarray(redshift_factor) ** 4)

    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        return "numba"

    def supports_vectorization(self) -> bool:
        """Check if backend supports vectorized operations."""
        return True

    def supports_gpu(self) -> bool:
        """Check if backend can run on GPU."""
        return False
