"""Backend factory and module initialization.

This module provides a factory pattern for creating computational backends.
"""

from luminet.backends.base import BaseBackend
from luminet.backends.scipy_backend import ScipyBackend

# Try to import optional backends
try:
    from luminet.backends.taichi_backend import TaichiBackend
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False

try:
    from luminet.backends.jax_backend import JAXBackend
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    from luminet.backends.numba_backend import NumbaBackend
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from luminet.backends.mojo_backend import MojoBackend
    MOJO_AVAILABLE = True
except ImportError:
    MOJO_AVAILABLE = False


def get_backend(backend_name: str, **kwargs) -> BaseBackend:
    """Factory function to create a backend instance.
    
    Args:
        backend_name: Name of backend ('scipy', 'taichi', 'jax', 'numba', 'mojo')
        **kwargs: Additional arguments to pass to backend constructor (e.g., arch, gpu_device)
    
    Returns:
        Backend instance
    
    Raises:
        ValueError: If backend name is not recognized
        ImportError: If backend dependencies are not installed
    """
    backend_name = backend_name.lower()

    if backend_name == "scipy":
        return ScipyBackend(**kwargs)

    elif backend_name == "taichi":
        if not TAICHI_AVAILABLE:
            raise ImportError(
                "Taichi is not installed. "
                "Install with: pip install taichi"
            )
        return TaichiBackend(**kwargs)

    elif backend_name == "jax":
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. "
                "Install with: pip install jax jaxlib"
            )
        return JAXBackend(**kwargs)

    elif backend_name == "numba":
        if not NUMBA_AVAILABLE:
            raise ImportError(
                "Numba is not installed. "
                "Install with: pip install numba"
            )
        return NumbaBackend(**kwargs)

    elif backend_name == "mojo":
        if not MOJO_AVAILABLE:
            raise ImportError(
                "Mojo is not installed or Python FFI is not available. "
                "Note: Mojo backend is experimental."
            )
        return MojoBackend(**kwargs)

    else:
        available = list_available_backends()
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            f"Available backends: {available}"
        )


def list_available_backends() -> list:
    """List all available backends.

    Returns:
        List of backend names
    """
    available = ["scipy"]
    if TAICHI_AVAILABLE:
        available.append("taichi")
    if JAX_AVAILABLE:
        available.append("jax")
    if NUMBA_AVAILABLE:
        available.append("numba")
    if MOJO_AVAILABLE:
        available.append("mojo")
    return available


def get_backend_info(backend_name: str = None) -> dict:
    """Get information about a specific backend or all backends.

    Args:
        backend_name: Name of backend (None = all backends)

    Returns:
        Dictionary with backend information
    """
    backends_info = {
        "scipy": {
            "name": "Scipy",
            "available": True,
            "vectorized": False,
            "gpu": False,
            "description": "Original implementation using scipy.optimize.brentq and scipy.special"
        },
        "taichi": {
            "name": "Taichi",
            "available": TAICHI_AVAILABLE,
            "vectorized": True,
            "gpu": True,
            "description": "GPU-accelerated implementation (work-in-progress)"
        },
        "jax": {
            "name": "JAX",
            "available": JAX_AVAILABLE,
            "vectorized": True,
            "gpu": True,
            "description": "JAX-based with automatic vectorization and JIT compilation"
        },
        "numba": {
            "name": "Numba",
            "available": NUMBA_AVAILABLE,
            "vectorized": True,
            "gpu": False,
            "description": "Numba JIT compilation for CPU with vectorization"
        },
        "mojo": {
            "name": "Mojo",
            "available": MOJO_AVAILABLE,
            "vectorized": False,
            "gpu": False,
            "description": "Experimental Mojo backend (requires Python FFI)"
        }
    }

    if backend_name:
        backend_name_lower = backend_name.lower()
        if backend_name_lower in backends_info:
            return backends_info[backend_name_lower]
        else:
            return None
    else:
        return backends_info


__all__ = [
    "BaseBackend",
    "ScipyBackend",
    "TaichiBackend",
    "JAXBackend",
    "NumbaBackend",
    "MojoBackend",
    "get_backend",
    "list_available_backends",
    "get_backend_info"
]
