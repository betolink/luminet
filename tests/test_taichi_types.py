"""Test Taichi type flexibility for GPU f32 support.

These exercise Taichi's own type system (``ti.template()`` vs hardcoded
``ti.f64``/``ti.f32``) rather than the luminet physics. GPU (Vulkan) cases are
skipped automatically when no Vulkan backend is available.
"""
import numpy as np
import pytest
import taichi as ti

_VULKAN_AVAILABLE = None


def _vulkan_available():
    """Cache a one-time probe of Vulkan/GPU availability."""
    global _VULKAN_AVAILABLE
    if _VULKAN_AVAILABLE is None:
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
    """Reset Taichi's global state after every test."""
    yield
    ti.reset()


# A generic kernel-internal function usable by both precisions.
@ti.func
def generic_func(x: ti.template()) -> ti.template():
    """Generic function that works with any numeric type."""
    return ti.sqrt(x * 2.0)


def test_ti_template_works_for_f32_and_f64():
    """``ti.template()`` lets one func be called from both f32 and f64 kernels."""
    ti.init(arch=ti.cpu, default_fp=ti.f64, log_level="error")

    @ti.kernel
    def kernel_f32(x: ti.f32) -> ti.f32:
        return generic_func(x)

    @ti.kernel
    def kernel_f64(x: ti.f64) -> ti.f64:
        return generic_func(x)

    expected = float(np.sqrt(16.0 * 2.0))

    r32 = float(kernel_f32(16.0))
    r64 = float(kernel_f64(16.0))

    # f32 is only ~6-7 digits; f64 should match to machine precision.
    assert abs(r32 - expected) / expected < 1e-5
    assert abs(r64 - expected) / expected < 1e-12


def test_mixed_f32_kernel_with_f64_func_runs():
    """A kernel declared f32 that calls an f64-typed func: Taichi upcasts and the
    run completes with a finite result (~sqrt(32))."""
    ti.init(arch=ti.cpu, default_fp=ti.f32, log_level="error")

    @ti.func
    def hardcoded_f64_func(x: ti.f64) -> ti.f64:
        return ti.sqrt(x * 2.0)

    @ti.kernel
    def mixed_kernel(x: ti.f32) -> ti.f32:
        return hardcoded_f64_func(x)

    result = float(mixed_kernel(16.0))
    expected = float(np.sqrt(16.0 * 2.0))
    assert np.isfinite(result)
    assert abs(result - expected) / expected < 1e-5


@pytest.mark.skipif(not _vulkan_available(), reason="no Vulkan/GPU backend available")
def test_gpu_vulkan_f32_with_template():
    """Vulkan GPU with f32 + generic ``ti.template()`` functions."""
    ti.init(arch=ti.vulkan, default_fp=ti.f32, log_level="error")

    @ti.func
    def gpu_generic_func(x: ti.template()) -> ti.template():
        return ti.sqrt(x) + ti.sin(x)

    @ti.kernel
    def gpu_test_kernel(x: ti.f32) -> ti.f32:
        return gpu_generic_func(x)

    result = float(gpu_test_kernel(2.0))
    expected = float(np.sqrt(2.0) + np.sin(2.0))
    assert abs(result - expected) / expected < 1e-4
