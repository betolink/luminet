"""Backend factory and module initialization.

This module provides a factory pattern for creating computational backends.
"""

from luminet.backends.base import BaseBackend
from luminet.backends.scipy_backend import ScipyBackend

try:
    from luminet.backends.taichi_backend import TaichiBackend
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False


def get_backend(backend_name: str, **kwargs) -> BaseBackend:
    """Factory function to create a backend instance.

    Args:
        backend_name: Name of backend ('scipy', 'taichi', 'jax')
        **kwargs: Additional arguments to pass to backend constructor

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
        raise NotImplementedError(
            "JAX backend is not yet implemented. "
            "Use 'scipy' or 'taichi' instead."
        )

    else:
        available = ["scipy"]
        if TAICHI_AVAILABLE:
            available.append("taichi")

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
    return available


__all__ = ["BaseBackend", "ScipyBackend", "TaichiBackend", "get_backend", "list_available_backends"]
