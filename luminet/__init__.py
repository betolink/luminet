"""
     Calculate and plot Swarzschild black holes with a thin accretion disk
 """

try:
    from importlib.metadata import version, metadata

    __version__ = version("luminet")
    _meta = metadata("luminet")

    # Parse author-email field (format: "Name <email@example.com>")
    author_email = _meta.get("Author-email", "")
    if "<" in author_email and ">" in author_email:
        __author__ = author_email.split("<")[0].strip()
        __email__ = author_email.split("<")[1].split(">")[0].strip()
    else:
        __author__ = _meta.get("Author", "unknown")
        __email__ = author_email or "unknown"

    __license__ = _meta.get("License", "unknown")

except Exception:
    __version__ = "unknown"
    __author__ = "unknown"
    __email__ = "unknown"
    __license__ = "unknown"

# Export backends (imported lazily to avoid circular imports)
def get_backend(backend_name: str = "scipy", **kwargs):
    """Get a computational backend instance.

    Args:
        backend_name: Name of backend ('scipy', 'taichi')
        **kwargs: Additional arguments for backend

    Returns:
        Backend instance

    Example:
        backend = get_backend("scipy")
        backend = get_backend("taichi", arch="cuda")
    """
    from luminet.backends import get_backend as _get_backend
    return _get_backend(backend_name, **kwargs)


def list_available_backends() -> list:
    """List all available computational backends.

    Returns:
        List of backend names

    Example:
        >>> list_available_backends()
        ['scipy', 'taichi']
    """
    from luminet.backends import list_available_backends as _list
    return _list()
