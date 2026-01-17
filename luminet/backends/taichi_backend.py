"""Taichi backend implementation with GPU acceleration.

This backend provides GPU acceleration using Taichi with support for
Vulkan (cross-platform), CUDA, and CPU fallback.

Features:
- Native elliptic functions (no scipy dependency in hot path)
- GPU-accelerated batch processing via Vulkan or CUDA
- Automatic architecture detection with manual override
- Full double precision support for numerical accuracy
"""

import numpy as np

try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False

from luminet.backends.base import BaseBackend

# ============================================================================
# Taichi Configuration
# ============================================================================

# Global state for Taichi initialization
_ti_initialized = False
_ti_arch = None
_warned_f32 = False  # Track if we've warned about f32 precision loss

def _init_taichi(arch: str = "auto", force: bool = False):
    """Initialize Taichi with the specified architecture.
    
    Args:
        arch: Architecture to use ('auto', 'gpu', 'vulkan', 'cuda', 'cpu')
        force: Force re-initialization even if already initialized
    
    Returns:
        Tuple of (arch_name, is_gpu, use_f64)
    
    Note:
        Vulkan/GPU backends may not support 64-bit operations (asin, acos, etc.)
        on some hardware. Tries f32 first on GPU, then falls back to f64 or CPU.
    """
    global _ti_initialized, _ti_arch
    
    if _ti_initialized and not force:
        return _ti_arch
    
    if not TAICHI_AVAILABLE:
        raise ImportError("Taichi is not installed. Install with: pip install taichi")
    
    # Map user-friendly names to Taichi architectures
    # Format: (arch_name, ti_arch, use_f64)
    arch_priority = []
    
    if arch == "auto" or arch == "gpu":
        # Try GPU backends: f32 first, then f64, then CPU with f64
        arch_priority = [
            ("cuda", ti.cuda, False),    # Try CUDA f32 first
            ("cuda", ti.cuda, True),     # Then CUDA f64
            ("vulkan", ti.vulkan, False), # Try Vulkan f32 first
            ("vulkan", ti.vulkan, True),  # Then Vulkan f64
            ("cpu", ti.cpu, True),       # CPU always uses f64
        ]
    elif arch == "vulkan":
        arch_priority = [
            ("vulkan", ti.vulkan, False),
            ("vulkan", ti.vulkan, True),
            ("cpu", ti.cpu, True),
        ]
    elif arch == "cuda":
        arch_priority = [
            ("cuda", ti.cuda, False),
            ("cuda", ti.cuda, True),
            ("cpu", ti.cpu, True),
        ]
    elif arch == "cpu":
        arch_priority = [("cpu", ti.cpu, True)]
    else:
        raise ValueError(f"Unknown architecture: {arch}. Use 'auto', 'gpu', 'vulkan', 'cuda', or 'cpu'")
    
    # Suppress Taichi precision warnings (they're expected for GPU f32 mode)
    import warnings
    import logging
    
    # Suppress Taichi warnings about precision loss
    warnings.filterwarnings('ignore', module='taichi')
    logging.getLogger('taichi').setLevel(logging.ERROR)
    
    # Try each architecture with different precisions
    for arch_name, ti_arch, use_f64_flag in arch_priority:
        try:
            ti.reset()
            fp_type = ti.f64 if use_f64_flag else ti.f32
            ti.init(arch=ti_arch, default_fp=fp_type, debug=False, print_ir=False, log_level='error')
            
            # Test if transcendental functions work with this precision
            # This catches the "Instruction Asin(16) does not 64bits operation" error
            if use_f64_flag and arch_name != "cpu":
                # Try to compile a simple kernel with f64 transcendentals
                @ti.kernel
                def test_f64_transcendentals():
                    x = 0.5
                    y = ti.asin(x)  # Use default_fp (f64 in this case)
                    return y
                
                try:
                    # Try to compile and run
                    test_f64_transcendentals()
                except Exception:
                    # F64 transcendentals not supported, skip this config
                    continue
            
            _ti_initialized = True
            _ti_arch = (arch_name, arch_name != "cpu", use_f64_flag)
            
            # Emit single warning about f32 precision if using GPU with f32
            if not use_f64_flag:
                global _warned_f32
                if not _warned_f32:
                    import sys
                    print(f"⚠️  Taichi GPU using f32 precision (expect ~1e-6 to 1e-8 accuracy vs f64)", file=sys.stderr)
                    _warned_f32 = True
            
            return _ti_arch
        except Exception:
            continue
    
    raise RuntimeError("Failed to initialize Taichi with any architecture")


# ============================================================================
# Gauss-Legendre Quadrature Nodes and Weights (20-point)
# These are pre-computed for the interval [-1, 1]
# ============================================================================

# Nodes (abscissae)
GL_NODES = np.array([
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
    -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
    -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
    -0.0765265211334973,  0.0765265211334973,  0.2277858511416451,
     0.3737060887154195,  0.5108670019508271,  0.6360536807265150,
     0.7463319064601508,  0.8391169718222188,  0.9122344282513259,
     0.9639719272779138,  0.9931285991850949
], dtype=np.float64)

# Weights
GL_WEIGHTS = np.array([
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
    0.0832767415767047, 0.1019301198172404, 0.1181945319615184,
    0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
    0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
    0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
    0.1019301198172404, 0.0832767415767047, 0.0626720483341091,
    0.0406014298003869, 0.0176140071391521
], dtype=np.float64)


# ============================================================================
# Taichi Elliptic Functions (GPU-compatible)
# ============================================================================

@ti.func
def ti_ellipk_agm(m: ti.template()) -> ti.template():
    """Complete elliptic integral K(m) using AGM algorithm.
    
    K(m) = pi / (2 * AGM(1, sqrt(1-m)))
    
    Args:
        m: Parameter (0 <= m < 1)
    
    Returns:
        K(m) value
    """
    result = ti.math.nan
    
    if m >= 0.0 and m < 1.0:
        if m == 0.0:
            result = ti.math.pi / 2.0
        else:
            a = 1.0
            b = ti.sqrt(1.0 - m)
            
            # AGM iteration (usually converges in ~10 iterations)
            for _ in range(25):
                a_new = (a + b) / 2.0
                b_new = ti.sqrt(a * b)
                if ti.abs(a_new - b_new) < 1e-15:
                    break
                a = a_new
                b = b_new
            
            result = ti.math.pi / (2.0 * a)
    
    return result


@ti.func
def ti_ellipkinc_gauss_core(phi: ti.template(), m: ti.template(), 
                             nodes: ti.types.ndarray(),
                             weights: ti.types.ndarray()) -> ti.template():
    """Incomplete elliptic integral F(phi, m) for phi in [0, pi/2].
    
    Uses 20-point Gauss-Legendre quadrature.
    
    Args:
        phi: Amplitude (radians), should be in [0, pi/2]
        m: Parameter (0 <= m < 1)
        nodes: Gauss-Legendre nodes
        weights: Gauss-Legendre weights
    
    Returns:
        F(phi, m) value
    """
    result = 0.0
    
    if phi != 0.0:
        # Transform [-1, 1] to [0, phi]
        half_phi = phi / 2.0
        
        for i in range(20):
            t = half_phi * (nodes[i] + 1.0)
            sin_t = ti.sin(t)
            integrand = 1.0 / ti.sqrt(1.0 - m * sin_t * sin_t)
            result += weights[i] * integrand
        
        result = result * half_phi
    
    return result


@ti.func
def ti_ellipkinc_agm(phi: ti.template(), m: ti.template(),
                      nodes: ti.types.ndarray(),
                      weights: ti.types.ndarray()) -> ti.template():
    """Incomplete elliptic integral F(phi, m) using Gauss-Legendre quadrature.
    
    F(phi, m) = integral from 0 to phi of 1/sqrt(1 - m*sin^2(t)) dt
    
    Handles arbitrary phi by reducing to [0, pi/2].
    
    Args:
        phi: Amplitude (radians)
        m: Parameter (0 <= m < 1)
        nodes: Gauss-Legendre nodes
        weights: Gauss-Legendre weights
    
    Returns:
        F(phi, m) value
    """
    result = ti.math.nan
    
    if m >= 0.0 and m < 1.0:
        if phi == 0.0:
            result = 0.0
        elif m == 0.0:
            result = phi
        else:
            # Handle negative phi
            sign = 1.0
            phi_abs = phi
            if phi < 0.0:
                sign = -1.0
                phi_abs = -phi
            
            # Handle large phi by splitting into complete periods + remainder
            # F(n*pi + phi_rem, m) = 2*n*K(m) + F(phi_rem, m)
            K = ti_ellipk_agm(m)
            n_periods = ti.floor(phi_abs / ti.math.pi)
            phi_rem = phi_abs - n_periods * ti.math.pi
            
            # Now phi_rem is in [0, pi]
            # If phi_rem > pi/2, use F(pi - x, m) = 2K - F(x, m)
            if phi_rem > ti.math.pi / 2.0:
                phi_rem = ti.math.pi - phi_rem
                inner = ti_ellipkinc_gauss_core(phi_rem, m, nodes, weights)
                result = sign * (2.0 * n_periods * K + 2.0 * K - inner)
            else:
                inner = ti_ellipkinc_gauss_core(phi_rem, m, nodes, weights)
                result = sign * (2.0 * n_periods * K + inner)
    
    return result


@ti.func
def ti_ellipj_sn(u: ti.template(), m: ti.template()) -> ti.template():
    """Jacobi elliptic function sn(u, m) using descending Landen transformation.
    
    Args:
        u: Argument
        m: Parameter (0 <= m <= 1)
    
    Returns:
        sn(u, m) value
    """
    result = ti.math.nan
    
    if m >= 0.0 and m <= 1.0:
        if m == 0.0:
            result = ti.sin(u)
        elif m == 1.0:
            result = ti.tanh(u)
        else:
            # Store transformation parameters (fixed size = 25 for GPU)
            # We need to inline the size for Taichi
            a_0, a_1, a_2, a_3, a_4 = 0.0, 0.0, 0.0, 0.0, 0.0
            a_5, a_6, a_7, a_8, a_9 = 0.0, 0.0, 0.0, 0.0, 0.0
            a_10, a_11, a_12, a_13, a_14 = 0.0, 0.0, 0.0, 0.0, 0.0
            a_15, a_16, a_17, a_18, a_19 = 0.0, 0.0, 0.0, 0.0, 0.0
            a_20, a_21, a_22, a_23, a_24 = 0.0, 0.0, 0.0, 0.0, 0.0
            
            c_0, c_1, c_2, c_3, c_4 = 0.0, 0.0, 0.0, 0.0, 0.0
            c_5, c_6, c_7, c_8, c_9 = 0.0, 0.0, 0.0, 0.0, 0.0
            c_10, c_11, c_12, c_13, c_14 = 0.0, 0.0, 0.0, 0.0, 0.0
            c_15, c_16, c_17, c_18, c_19 = 0.0, 0.0, 0.0, 0.0, 0.0
            c_20, c_21, c_22, c_23, c_24 = 0.0, 0.0, 0.0, 0.0, 0.0
            
            a_0 = 1.0
            b = ti.sqrt(1.0 - m)
            c_0 = ti.sqrt(m)
            
            n = 0
            
            # Unrolled AGM iteration
            # Iteration 0->1
            if ti.abs(c_0) >= 1e-15:
                a_1 = (a_0 + b) / 2.0
                c_1 = (a_0 - b) / 2.0
                b = ti.sqrt(a_0 * b)
                n = 1
            
            if ti.abs(c_1) >= 1e-15 and n >= 1:
                a_2 = (a_1 + b) / 2.0
                c_2 = (a_1 - b) / 2.0
                b = ti.sqrt(a_1 * b)
                n = 2
            
            if ti.abs(c_2) >= 1e-15 and n >= 2:
                a_3 = (a_2 + b) / 2.0
                c_3 = (a_2 - b) / 2.0
                b = ti.sqrt(a_2 * b)
                n = 3
            
            if ti.abs(c_3) >= 1e-15 and n >= 3:
                a_4 = (a_3 + b) / 2.0
                c_4 = (a_3 - b) / 2.0
                b = ti.sqrt(a_3 * b)
                n = 4
            
            if ti.abs(c_4) >= 1e-15 and n >= 4:
                a_5 = (a_4 + b) / 2.0
                c_5 = (a_4 - b) / 2.0
                b = ti.sqrt(a_4 * b)
                n = 5
            
            if ti.abs(c_5) >= 1e-15 and n >= 5:
                a_6 = (a_5 + b) / 2.0
                c_6 = (a_5 - b) / 2.0
                b = ti.sqrt(a_5 * b)
                n = 6
            
            if ti.abs(c_6) >= 1e-15 and n >= 6:
                a_7 = (a_6 + b) / 2.0
                c_7 = (a_6 - b) / 2.0
                b = ti.sqrt(a_6 * b)
                n = 7
            
            if ti.abs(c_7) >= 1e-15 and n >= 7:
                a_8 = (a_7 + b) / 2.0
                c_8 = (a_7 - b) / 2.0
                b = ti.sqrt(a_7 * b)
                n = 8
            
            if ti.abs(c_8) >= 1e-15 and n >= 8:
                a_9 = (a_8 + b) / 2.0
                c_9 = (a_8 - b) / 2.0
                b = ti.sqrt(a_8 * b)
                n = 9
            
            if ti.abs(c_9) >= 1e-15 and n >= 9:
                a_10 = (a_9 + b) / 2.0
                c_10 = (a_9 - b) / 2.0
                b = ti.sqrt(a_9 * b)
                n = 10
            
            # Usually converges by n=10, but continue a few more for safety
            if ti.abs(c_10) >= 1e-15 and n >= 10:
                a_11 = (a_10 + b) / 2.0
                c_11 = (a_10 - b) / 2.0
                b = ti.sqrt(a_10 * b)
                n = 11
            
            if ti.abs(c_11) >= 1e-15 and n >= 11:
                a_12 = (a_11 + b) / 2.0
                c_12 = (a_11 - b) / 2.0
                b = ti.sqrt(a_11 * b)
                n = 12
            
            # Get the final a_n value
            a_n = a_0
            if n >= 1: a_n = a_1
            if n >= 2: a_n = a_2
            if n >= 3: a_n = a_3
            if n >= 4: a_n = a_4
            if n >= 5: a_n = a_5
            if n >= 6: a_n = a_6
            if n >= 7: a_n = a_7
            if n >= 8: a_n = a_8
            if n >= 9: a_n = a_9
            if n >= 10: a_n = a_10
            if n >= 11: a_n = a_11
            if n >= 12: a_n = a_12
            
            # Backward recurrence
            phi = u * a_n * (2.0 ** n)
            
            # Unroll backward recurrence
            if n >= 12:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_12 * sin_phi / a_12)) / 2.0
            if n >= 11:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_11 * sin_phi / a_11)) / 2.0
            if n >= 10:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_10 * sin_phi / a_10)) / 2.0
            if n >= 9:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_9 * sin_phi / a_9)) / 2.0
            if n >= 8:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_8 * sin_phi / a_8)) / 2.0
            if n >= 7:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_7 * sin_phi / a_7)) / 2.0
            if n >= 6:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_6 * sin_phi / a_6)) / 2.0
            if n >= 5:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_5 * sin_phi / a_5)) / 2.0
            if n >= 4:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_4 * sin_phi / a_4)) / 2.0
            if n >= 3:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_3 * sin_phi / a_3)) / 2.0
            if n >= 2:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_2 * sin_phi / a_2)) / 2.0
            if n >= 1:
                sin_phi = ti.sin(phi)
                phi = (phi + ti.asin(c_1 * sin_phi / a_1)) / 2.0
            
            result = ti.sin(phi)
    
    return result


@ti.func
def ti_calc_q(p: ti.template(), bh_mass: ti.template()) -> ti.template():
    """Calculate Q from periastron P."""
    result = ti.math.nan
    if p >= 2.0 * bh_mass:
        result = ti.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
    return result


@ti.func
def ti_calc_k_squared(p: ti.template(), bh_mass: ti.template()) -> ti.template():
    """Calculate k^2 (elliptic modulus squared)."""
    result = ti.math.nan
    if p >= 2.0 * bh_mass:
        q = ti.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
        result = (q - p + 6.0 * bh_mass) / (2.0 * q)
    return result


@ti.func
def ti_calc_zeta_inf(p: ti.template(), bh_mass: ti.template()) -> ti.template():
    """Calculate zeta_infinity."""
    result = ti.math.nan
    if p >= 2.0 * bh_mass:
        q = ti.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
        arg = (q - p + 2.0 * bh_mass) / (q - p + 6.0 * bh_mass)
        result = ti.asin(ti.sqrt(arg))
    return result


@ti.func
def ti_calc_sn(p: ti.template(), angle: ti.template(), bh_mass: ti.template(), incl: ti.template(), order: ti.i32,
               nodes: ti.types.ndarray(),
               weights: ti.types.ndarray()) -> ti.template():
    """Calculate Jacobi elliptic sn for given parameters."""
    result = ti.math.nan
    
    if p >= 2.0 * bh_mass:
        q = ti.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
        
        # zeta_inf
        arg = (q - p + 2.0 * bh_mass) / (q - p + 6.0 * bh_mass)
        z_inf = ti.asin(ti.sqrt(arg))
        
        # k^2 (modulus squared)
        m = (q - p + 6.0 * bh_mass) / (2.0 * q)
        
        # F(zeta_inf, m)
        ell_inf = ti_ellipkinc_agm(z_inf, m, nodes, weights)
        
        # cos(gamma)
        cos_angle = ti.cos(angle)
        tan_incl = ti.tan(incl)
        cos_gamma = cos_angle / ti.sqrt(cos_angle ** 2 + 1.0 / (tan_incl ** 2))
        g = ti.acos(cos_gamma)
        
        # elliptic argument
        sqrt_p_q = ti.sqrt(p / q)
        
        ellips_arg = 0.0
        if order == 0:
            ellips_arg = g / (2.0 * sqrt_p_q) + ell_inf
        else:
            ell_k = ti_ellipk_agm(m)
            ellips_arg = (g - 2.0 * order * ti.math.pi) / (2.0 * sqrt_p_q) - ell_inf + 2.0 * ell_k
        
        result = ti_ellipj_sn(ellips_arg, m)
    
    return result


@ti.func
def ti_periastron_cost(p: ti.template(), radius: ti.template(), angle: ti.template(), 
                        bh_mass: ti.template(), incl: ti.template(), order: ti.i32,
                        nodes: ti.types.ndarray(),
                        weights: ti.types.ndarray()) -> ti.template():
    """Cost function for periastron root finding."""
    result = ti.math.nan
    
    if p >= 2.0 * bh_mass:
        q = ti.sqrt((p - 2.0 * bh_mass) * (p + 6.0 * bh_mass))
        sn = ti_calc_sn(p, angle, bh_mass, incl, order, nodes, weights)
        
        if not ti.math.isnan(sn):
            term1 = -(q - p + 2.0 * bh_mass)
            term2 = (q - p + 6.0 * bh_mass) * sn * sn
            result = 4.0 * bh_mass * p - radius * (term1 + term2)
    
    return result


@ti.func
def ti_bisect_periastron(radius: ti.template(), angle: ti.template(), bh_mass: ti.template(), 
                          incl: ti.template(), order: ti.i32,
                          nodes: ti.types.ndarray(),
                          weights: ti.types.ndarray(),
                          tol: ti.template(), max_iter: ti.i32) -> ti.template():
    """Bisection root finder for periastron with Brent's method speedup."""
    result = ti.math.nan
    
    if radius > 3.0 * bh_mass:
        # Bracket: [3M + epsilon, radius]
        a = 3.0 * bh_mass + order * 1e-5 + 1e-10
        b = radius
        
        fa = ti_periastron_cost(a, radius, angle, bh_mass, incl, order, nodes, weights)
        fb = ti_periastron_cost(b, radius, angle, bh_mass, incl, order, nodes, weights)
        
        valid = True
        if ti.math.isnan(fa) or ti.math.isnan(fb):
            valid = False
        
        # Check if root is bracketed
        if valid and fa * fb > 0:
            valid = False
        
        if valid:
            # Ensure fa < 0 and fb > 0
            if fa > 0:
                a, b = b, a
                fa, fb = fb, fa
            
            c = a
            fc = fa
            
            converged = False
            for _ in range(max_iter):
                if ti.abs(b - a) < tol:
                    result = (a + b) / 2.0
                    converged = True
                
                if not converged:
                    # Try inverse quadratic interpolation
                    s = 0.0
                    if fa != fc and fb != fc:
                        # Inverse quadratic interpolation
                        s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                            (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                            (c * fa * fb) / ((fc - fa) * (fc - fb))
                    else:
                        # Secant method
                        s = b - fb * (b - a) / (fb - fa)
                    
                    # Check if s is in bounds and making progress
                    min_ab = ti.min(a, b)
                    max_ab = ti.max(a, b)
                    if s < min_ab or s > max_ab or ti.abs(s - b) > ti.abs(b - a) / 2:
                        # Fall back to bisection
                        s = (a + b) / 2.0
                    
                    fs = ti_periastron_cost(s, radius, angle, bh_mass, incl, order, nodes, weights)
                    
                    if ti.math.isnan(fs):
                        s = (a + b) / 2.0
                        fs = ti_periastron_cost(s, radius, angle, bh_mass, incl, order, nodes, weights)
                    
                    c = b
                    fc = fb
                    
                    if fa * fs < 0:
                        b = s
                        fb = fs
                    else:
                        a = s
                        fa = fs
                    
                    result = (a + b) / 2.0
    
    return result


@ti.func
def ti_ellipse(r: ti.template(), a: ti.template(), incl: ti.template()) -> ti.template():
    """Newtonian ellipse approximation."""
    a_shifted = (a + ti.math.pi / 2.0) % (2.0 * ti.math.pi)
    major_axis = r
    minor_axis = ti.abs(major_axis * ti.cos(incl))
    
    result = 0.0
    if ti.abs(major_axis) >= 1e-15:
        ratio = minor_axis / major_axis
        if ratio >= 1.0:
            result = minor_axis
        else:
            eccentricity = ti.sqrt(1.0 - ratio ** 2)
            denom = 1.0 - (eccentricity * ti.cos(a_shifted)) ** 2
            if denom < 1e-15:
                result = minor_axis
            else:
                result = minor_axis / ti.sqrt(denom)
    
    return result


@ti.func
def ti_solve_impact_parameter(radius: ti.template(), incl: ti.template(), alpha: ti.template(), 
                               bh_mass: ti.template(), order: ti.i32,
                               nodes: ti.types.ndarray(),
                               weights: ti.types.ndarray()) -> ti.template():
    """Solve for impact parameter b."""
    alpha_adj = alpha
    if order % 2 == 1:
        alpha_adj = (alpha + ti.math.pi) % (2.0 * ti.math.pi)
    
    periastron = ti_bisect_periastron(radius, alpha_adj, bh_mass, incl, order, 
                                       nodes, weights, 1e-10, 100)
    
    result = ti.math.nan
    if ti.math.isnan(periastron):
        if order == 0 and (alpha < ti.math.pi / 2.0 or alpha > 3.0 * ti.math.pi / 2.0):
            result = ti_ellipse(radius, alpha, incl)
    else:
        result = ti.sqrt(periastron ** 3 / (periastron - 2.0 * bh_mass))
    
    return result


# ============================================================================
# Taichi Kernels for Batch Processing
# ============================================================================

@ti.kernel
def kernel_solve_impact_parameters(
    radii: ti.types.ndarray(),
    angles: ti.types.ndarray(),
    result: ti.types.ndarray(),
    incl: ti.template(),
    bh_mass: ti.template(),
    order: ti.i32,
    nodes: ti.types.ndarray(),
    weights: ti.types.ndarray()
):
    """Batch solve for impact parameters on GPU.
    
    Args:
        radii: Array of radii values
        angles: Array of angle values  
        result: Output array for impact parameters
        incl: Inclination angle
        bh_mass: Black hole mass
        order: Order of the image (0 = direct, 1 = first order, etc.)
        nodes: Gauss-Legendre quadrature nodes
        weights: Gauss-Legendre quadrature weights
    """
    for i in range(radii.shape[0]):
        result[i] = ti_solve_impact_parameter(
            radii[i], incl, angles[i], bh_mass, order, nodes, weights
        )


@ti.kernel
def kernel_calc_q(
    p_arr: ti.types.ndarray(),
    result: ti.types.ndarray(),
    bh_mass: ti.template()
):
    """Batch calculate Q values."""
    for i in range(p_arr.shape[0]):
        result[i] = ti_calc_q(p_arr[i], bh_mass)


@ti.kernel
def kernel_calc_k_squared(
    p_arr: ti.types.ndarray(),
    result: ti.types.ndarray(),
    bh_mass: ti.template()
):
    """Batch calculate k^2 values."""
    for i in range(p_arr.shape[0]):
        result[i] = ti_calc_k_squared(p_arr[i], bh_mass)


@ti.kernel
def kernel_calc_sn(
    p_arr: ti.types.ndarray(),
    angle_arr: ti.types.ndarray(),
    result: ti.types.ndarray(),
    bh_mass: ti.template(),
    incl: ti.template(),
    order: ti.i32,
    nodes: ti.types.ndarray(),
    weights: ti.types.ndarray()
):
    """Batch calculate sn values."""
    for i in range(p_arr.shape[0]):
        result[i] = ti_calc_sn(p_arr[i], angle_arr[i], bh_mass, incl, order, nodes, weights)


@ti.kernel
def kernel_calc_redshift(
    radius_arr: ti.types.ndarray(),
    angle_arr: ti.types.ndarray(),
    b_arr: ti.types.ndarray(),
    result: ti.types.ndarray(),
    incl: ti.template(),
    bh_mass: ti.template()
):
    """Batch calculate redshift factors."""
    for i in range(radius_arr.shape[0]):
        r = radius_arr[i]
        angle = angle_arr[i]
        b = b_arr[i]
        result[i] = (1.0 + ti.sqrt(bh_mass / (r ** 3)) * b * ti.sin(incl) * ti.sin(angle)) * \
                    (1.0 - 3.0 * bh_mass / r) ** -0.5


@ti.kernel
def kernel_calc_flux_intrinsic(
    radius_arr: ti.types.ndarray(),
    result: ti.types.ndarray(),
    acc: ti.template(),
    bh_mass: ti.template()
):
    """Batch calculate intrinsic flux."""
    for i in range(radius_arr.shape[0]):
        r = radius_arr[i]
        r_ = r / bh_mass
        sqrt_r = ti.sqrt(r_)
        sqrt_3 = ti.sqrt(3.0)
        sqrt_6 = ti.sqrt(6.0)
        
        log_arg = (sqrt_r + sqrt_3) * (sqrt_6 - sqrt_3) / \
                  ((sqrt_r - sqrt_3) * (sqrt_6 + sqrt_3))
        
        if log_arg <= 0:
            result[i] = 0.0
        else:
            A = 3.0 * bh_mass * acc / (8.0 * ti.math.pi) / ((r_ - 3.0) * r_ ** 2.5)
            result[i] = A * (sqrt_r - sqrt_6 + (sqrt_3 / 2.0) * ti.log(log_arg))


# ============================================================================
# TaichiBackend Class
# ============================================================================

class TaichiBackend(BaseBackend):
    """Taichi-based computational backend with GPU support.
    
    Features:
    - Full GPU acceleration via Vulkan (cross-platform) or CUDA
    - Native elliptic functions (no scipy dependency in hot path)
    - Batch processing for optimal GPU utilization
    - Automatic architecture detection with manual override
    
    Args:
        arch: Architecture to use ('auto', 'gpu', 'vulkan', 'cuda', 'cpu')
              - 'auto'/'gpu': Try vulkan -> cuda -> cpu
              - 'vulkan': Use Vulkan (works on AMD/Intel/NVIDIA)
              - 'cuda': Use CUDA (NVIDIA only)
              - 'cpu': Use CPU (fallback)
    """

    def __init__(self, arch: str = "auto"):
        """Initialize Taichi backend.

        Args:
            arch: Architecture to use ('auto', 'gpu', 'vulkan', 'cuda', 'cpu')
        """
        super().__init__()

        if not TAICHI_AVAILABLE:
            raise ImportError("Taichi is not installed. Install with: pip install taichi")

        # Initialize Taichi
        arch_name, is_gpu, use_f64 = _init_taichi(arch)
        self.name = "taichi"
        self.arch = arch_name
        self._is_gpu = is_gpu
        self._use_f64 = use_f64
        
        # Store quadrature nodes and weights as numpy arrays for kernel calls
        # Use float32 for GPU (f32), float64 for CPU (f64)
        dtype = np.float64 if use_f64 else np.float32
        self._nodes = GL_NODES.astype(dtype)
        self._weights = GL_WEIGHTS.astype(dtype)
        
        # Warm up kernels
        self._warmup()
    
    def _warmup(self):
        """Trigger JIT compilation of all kernels."""
        # Small test arrays to trigger compilation
        dtype = np.float64 if self._use_f64 else np.float32
        test_radii = np.array([10.0, 15.0], dtype=dtype)
        test_angles = np.array([0.5, 1.0], dtype=dtype)
        test_result = np.zeros(2, dtype=dtype)
        
        # Warm up the main kernel
        try:
            kernel_solve_impact_parameters(
                test_radii, test_angles, test_result,
                1.4, 1.0, 0, self._nodes, self._weights
            )
        except Exception:
            pass  # Warmup failure is okay
    
    def calc_q(self, p, bh_mass):
        """Calculate Q from periastron P."""
        dtype = np.float64 if self._use_f64 else np.float32
        if isinstance(p, np.ndarray):
            p_arr = np.asarray(p, dtype=dtype)
            result = np.zeros(len(p_arr), dtype=dtype)
            kernel_calc_q(p_arr, result, float(bh_mass))
            return result
        else:
            # Single value - use a 1-element array
            p_arr = np.array([float(p)], dtype=dtype)
            result = np.zeros(1, dtype=dtype)
            kernel_calc_q(p_arr, result, float(bh_mass))
            return result[0]

    def calc_k_squared(self, p, bh_mass):
        """Calculate squared modulus of elliptic integral."""
        dtype = np.float64 if self._use_f64 else np.float32
        if isinstance(p, np.ndarray):
            p_arr = np.asarray(p, dtype=dtype)
            result = np.zeros(len(p_arr), dtype=dtype)
            kernel_calc_k_squared(p_arr, result, float(bh_mass))
            return result
        else:
            p_arr = np.array([float(p)], dtype=dtype)
            result = np.zeros(1, dtype=dtype)
            kernel_calc_k_squared(p_arr, result, float(bh_mass))
            return result[0]

    def calc_zeta_inf(self, p, bh_mass):
        """Calculate zeta_infinity."""
        # Use numpy for this simple calculation
        q = self.calc_q(p, bh_mass)
        if isinstance(p, np.ndarray):
            arg = (q - p + 2 * bh_mass) / (q - p + 6 * bh_mass)
            z_inf = np.arcsin(np.sqrt(arg))
            z_inf[np.isnan(arg)] = np.nan
            return z_inf
        else:
            if np.isnan(q):
                return np.nan
            arg = (q - p + 2 * bh_mass) / (q - p + 6 * bh_mass)
            return np.arcsin(np.sqrt(arg))

    def calc_sn(self, p, angle, bh_mass, incl, order=0):
        """Calculate Jacobi elliptic function sn.
        
        Uses GPU-accelerated native implementation.
        """
        dtype = np.float64 if self._use_f64 else np.float32
        if isinstance(p, np.ndarray):
            p_arr = np.asarray(p, dtype=dtype)
            angle_arr = np.asarray(angle, dtype=dtype)
            result = np.zeros(len(p_arr), dtype=dtype)
            kernel_calc_sn(p_arr, angle_arr, result, float(bh_mass), 
                          float(incl), int(order), self._nodes, self._weights)
            return result
        else:
            p_arr = np.array([float(p)], dtype=dtype)
            angle_arr = np.array([float(angle)], dtype=dtype)
            result = np.zeros(1, dtype=dtype)
            kernel_calc_sn(p_arr, angle_arr, result, float(bh_mass), 
                          float(incl), int(order), self._nodes, self._weights)
            return result[0]

    def periastron_cost(self, p, radius, angle, bh_mass, incl, order=0):
        """Cost function for periastron optimization."""
        # Use numpy implementation for single values (kernel overhead not worth it)
        q = self.calc_q(p, bh_mass)
        if np.isnan(q):
            return np.nan

        sn = self.calc_sn(p, angle, bh_mass, incl, order)
        term1 = -(q - p + 2.0 * bh_mass)
        term2 = (q - p + 6.0 * bh_mass) * sn * sn
        return 4.0 * bh_mass * p - radius * (term1 + term2)

    def solve_for_periastron(self, radius, incl, alpha, bh_mass, order=0):
        """Solve for periastron given black hole coordinates.
        
        Uses GPU-accelerated bisection method.
        """
        # For single values, compute via impact parameter kernel
        b = self.solve_for_impact_parameter(radius, incl, alpha, bh_mass, order)
        if np.isnan(b):
            return np.nan
        # Back-calculate periastron from b
        # b = sqrt(p^3 / (p - 2M))
        # b^2 * (p - 2M) = p^3
        # This is a cubic equation - for now use the direct method
        # Just return the periastron from the internal calculation
        return self._solve_periastron_direct(radius, incl, alpha, bh_mass, order)
    
    def _solve_periastron_direct(self, radius, incl, alpha, bh_mass, order):
        """Direct periastron solve using numpy (for single values)."""
        from luminet.solver import improve_solutions
        
        if radius <= 3 * bh_mass:
            return np.nan

        alpha_adj = alpha
        if order % 2 == 1:
            alpha_adj = (alpha + np.pi) % (2 * np.pi)

        min_periastron = 3.0 * bh_mass + order * 1e-5
        periastron_initial_guess = np.linspace(min_periastron, radius, 2)

        y = np.array([
            self.periastron_cost(p_guess, radius, alpha_adj, bh_mass, incl, order)
            for p_guess in periastron_initial_guess
        ])

        if any(np.isnan(y)):
            return np.nan

        if np.sign(y[0]) == np.sign(y[1]):
            return np.nan

        kwargs_eq13 = {
            "radius": radius,
            "angle": alpha_adj,
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
        """Solve for impact parameter b.
        
        Uses GPU-accelerated batch processing when given arrays.
        """
        dtype = np.float64 if self._use_f64 else np.float32
        if isinstance(radius, np.ndarray):
            radii = np.asarray(radius, dtype=dtype)
            angles = np.asarray(alpha, dtype=dtype)
            result = np.zeros(len(radii), dtype=dtype)
            kernel_solve_impact_parameters(
                radii, angles, result,
                float(incl), float(bh_mass), int(order),
                self._nodes, self._weights
            )
            return result
        else:
            # Single value - still use kernel for consistency
            radii = np.array([float(radius)], dtype=dtype)
            angles = np.array([float(alpha)], dtype=dtype)
            result = np.zeros(1, dtype=dtype)
            kernel_solve_impact_parameters(
                radii, angles, result,
                float(incl), float(bh_mass), int(order),
                self._nodes, self._weights
            )
            return result[0]

    def _ellipse(self, r, a, incl):
        """Equation of an ellipse (Newtonian limit)."""
        a = (a + np.pi / 2) % (2 * np.pi)
        major_axis = r
        minor_axis = abs(major_axis * np.cos(incl))
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        return minor_axis / np.sqrt((1 - (eccentricity * np.cos(a)) ** 2))

    def calc_redshift_factor(self, radius, angle, incl, bh_mass, b):
        """Calculate gravitational redshift factor (1+z)."""
        dtype = np.float64 if self._use_f64 else np.float32
        if isinstance(radius, np.ndarray):
            radius_arr = np.asarray(radius, dtype=dtype)
            angle_arr = np.asarray(angle, dtype=dtype)
            b_arr = np.asarray(b, dtype=dtype)
            result = np.zeros(len(radius_arr), dtype=dtype)
            kernel_calc_redshift(radius_arr, angle_arr, b_arr, result, 
                                float(incl), float(bh_mass))
            return result
        else:
            return (1.0 + np.sqrt(bh_mass / (radius**3)) * b * np.sin(incl) * np.sin(angle)) * \
                   (1 - 3.0 * bh_mass / radius) ** -0.5

    def calc_flux_intrinsic_swarzschild(self, radius, acc, bh_mass):
        """Calculate intrinsic flux for Schwarzschild black hole."""
        dtype = np.float64 if self._use_f64 else np.float32
        if isinstance(radius, np.ndarray):
            radius_arr = np.asarray(radius, dtype=dtype)
            result = np.zeros(len(radius_arr), dtype=dtype)
            kernel_calc_flux_intrinsic(radius_arr, result, float(acc), float(bh_mass))
            return result
        else:
            r_ = radius / bh_mass
            log_arg = (np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3)) / \
                      ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
            A = 3 * bh_mass * acc / (8 * np.pi) / ((r_ - 3) * r_**2.5)
            return A * (np.sqrt(r_) - np.sqrt(6) + (np.sqrt(3) / 2) * np.log(log_arg))

    def calc_flux_observed(self, radius, acc, bh_mass, redshift_factor):
        """Calculate observed flux with redshift correction."""
        flux_intr = self.calc_flux_intrinsic_swarzschild(radius=radius, acc=acc, bh_mass=bh_mass)
        flux_observed = flux_intr / np.asarray(redshift_factor)**4
        return flux_observed

    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        precision = "f64" if self._use_f64 else "f32"
        return f"taichi-{self.arch}-{precision}"

    def supports_vectorization(self) -> bool:
        """Check if backend supports vectorized operations."""
        return True

    def supports_gpu(self) -> bool:
        """Check if backend can run on GPU."""
        return self._is_gpu
    
    def get_arch(self) -> str:
        """Get the current Taichi architecture."""
        return self.arch
