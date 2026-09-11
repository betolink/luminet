"""Verify the Taichi backend works (CPU, and GPU when a Vulkan backend exists).

Replaces the old ad-hoc diagnostic script (which used ``sys.exit`` at import time
and a hardcoded path). These are proper parametrised pytest tests that compare the
Taichi backend against the scipy reference implementation. GPU (arch='gpu') tests
skip automatically when no Vulkan/GPU backend is available.
"""
import numpy as np
import pytest
from luminet.backends import get_backend

_VULKAN_AVAILABLE = None


def _vulkan_available():
    """Cache a one-time probe of Vulkan/GPU availability."""
    global _VULKAN_AVAILABLE
    if _VULKAN_AVAILABLE is None:
        import taichi as ti
        try:
            _VULKAN_AVAILABLE = bool(ti.has_vulkan())
        except AttributeError:
            try:
                ti.init(arch=ti.vulkan, log_level="error")
                ti.reset()
                _VULKAN_AVAILABLE = True
            except Exception:
                _VULKAN_AVAILABLE = False
    return _VULKAN_AVAILABLE


@pytest.fixture(autouse=True)
def _reset_taichi_between_tests():
    yield
    import taichi as ti

    # A BlackHole created with backend="taichi" leaves the global
    # black_hole_math "current backend" pointing at a TaichiBackend. After
    # ti.reset() its runtime program is torn down (prog is None), so any later
    # test that routes through get_current_backend() would crash. Reset the
    # global backend so this file doesn't leak state into other test modules.
    import luminet.black_hole_math as bhmath

    bhmath.set_backend("scipy")
    ti.reset()


def _taichi_cpu_backend():
    """Fresh Taichi CPU/f64 backend, forcing a re-init regardless of any cached state."""
    import taichi as ti
    import luminet.backends.taichi_backend as _tb

    # The backend caches Taichi's global init; clear it so we always get a clean CPU/f64 context.
    _tb._ti_initialized = False
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, log_level="error")
    return get_backend("taichi", arch="cpu")


def test_taichi_cpu_backend_initializes():
    """The Taichi CPU backend initializes and reports f64 by default."""
    backend = _taichi_cpu_backend()
    assert backend.get_backend_name().startswith("taichi")
    assert backend.arch == "cpu"
    assert backend._use_f64 is True


def test_taichi_calc_q_matches_scipy():
    """calc_q agrees with the scipy reference across a range of p and mass."""
    taichi = _taichi_cpu_backend()
    scipy = get_backend("scipy")

    checked = 0
    for mass in (1.0, 2.0, 5.0):
        for p in (1.0, 3.0, 5.0, 10.0):
            ref = scipy.calc_q(p, mass)
            if not np.isfinite(ref) or ref == 0.0:
                continue
            got = taichi.calc_q(p, mass)
            assert np.isfinite(got)
            assert abs(got - ref) / abs(ref) < 1e-9
            checked += 1
    assert checked > 0


def test_taichi_calc_k_squared_matches_scipy():
    """calc_k_squared agrees with the scipy reference."""
    taichi = _taichi_cpu_backend()
    scipy = get_backend("scipy")

    checked = 0
    for mass in (1.0, 0.5):
        for p in (1.0, 3.0, 8.0):
            ref = scipy.calc_k_squared(p, mass)
            if not np.isfinite(ref):
                continue
            got = taichi.calc_k_squared(p, mass)
            assert np.isfinite(got)
            assert abs(got - ref) < 1e-9 * (1.0 + abs(ref))
            checked += 1
    assert checked > 0


def test_taichi_calc_sn_matches_scipy():
    """calc_sn agrees with the scipy reference (comparing only finite cases)."""
    taichi = _taichi_cpu_backend()
    scipy = get_backend("scipy")

    checked = 0
    for mass in (1.0, 0.5):
        for p in (5.0, 8.0, 15.0):
            for angle in (0.5, 1.0, 1.4):
                for incl in (1.0, 1.3):
                    ref = scipy.calc_sn(p, angle, mass, incl)
                    if not np.isfinite(ref):
                        continue
                    got = taichi.calc_sn(p, angle, mass, incl)
                assert np.isfinite(got)
                assert abs(got - ref) < 1e-6 * (1.0 + abs(ref))
                checked += 1
    assert checked > 0


def test_taichi_solve_for_impact_parameter_matches_scipy():
    """solve_for_impact_parameter agrees with the scipy reference."""
    taichi = _taichi_cpu_backend()
    scipy = get_backend("scipy")

    for radius in (6.0, 10.0, 20.0, 50.0):
        for alpha in (0.5, 1.0, 1.5):
            ref = scipy.solve_for_impact_parameter(radius, 1.3, alpha, 1.0, 0)
            got = taichi.solve_for_impact_parameter(radius, 1.3, alpha, 1.0, 0)
            assert abs(got - ref) < 1e-6 * (1.0 + abs(ref))


def test_blackhole_init_with_taichi_cpu():
    """A full BlackHole can be created on the Taichi CPU backend."""
    from luminet.black_hole import BlackHole

    _taichi_cpu_backend()  # ensure taichi is initialised
    bh = BlackHole(
        mass=1.0,
        incl=1.0,
        outer_edge=20.0,
        backend="taichi",
        arch="cpu",
    )
    assert bh.mass == 1.0
    ip = np.asarray(bh.disk_apparent_outer_edge.impact_parameters)
    assert float(ip.max()) > 0.0


@pytest.mark.skipif(not _vulkan_available(), reason="no Vulkan/GPU backend available")
def test_taichi_gpu_backend_initializes():
    """The Taichi GPU (Vulkan) backend initializes on f32 and computes correctly.

    Force-resets Taichi's cached global init (a prior CPU init would otherwise make
    the backend report arch='cpu'), then runs a real GPU kernel.
    """
    import taichi as ti
    import luminet.backends.taichi_backend as _tb

    # Reset cached state so the backend re-initialises on the GPU (f32).
    _tb._ti_initialized = False
    ti.reset()

    try:
        ti.init(arch=ti.vulkan, default_fp=ti.f32, log_level="error")
        backend = get_backend("taichi", arch="gpu")
    except Exception as exc:  # pragma: no cover - depends on hardware/drivers
        pytest.skip(f"GPU backend could not be initialised: {exc}")

    assert backend.arch in ("gpu", "vulkan")
    assert backend._is_gpu is True
    assert backend._use_f64 is False  # GPU runs in f32

    # A real GPU computation, checked against the scipy reference.
    ref = get_backend("scipy").calc_q(5.0, 1.0)
    got = backend.calc_q(5.0, 1.0)
    assert np.isfinite(got)
    assert abs(got - ref) / abs(ref) < 1e-4
