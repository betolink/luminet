"""Test the specific Taichi f32 behaviour with transcendental functions (asin).

Verifies that ``ti.asin`` (and complex nested kernels using it) work on CPU, and
documents the recommended ``ti.template()`` approach for GPU f32. GPU (Vulkan)
cases are skipped when no Vulkan backend is available.
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


def test_cpu_f32_asin_runs():
    """A CPU f32 kernel using ti.asin completes and returns a finite value."""
    ti.init(arch=ti.cpu, default_fp=ti.f32, log_level="error")

    @ti.func
    def calc_with_asin(x: ti.template()) -> ti.template():
        return ti.asin(ti.sqrt(x))

    @ti.kernel
    def cpu_asin_kernel(x: ti.f32) -> ti.f32:
        return calc_with_asin(x)

    result = float(cpu_asin_kernel(0.25))
    expected = float(np.arcsin(np.sqrt(0.25)))  # asin(0.5) = pi/6
    assert np.isfinite(result)
    assert abs(result - expected) / expected < 1e-4


def test_cpu_complex_nested_asin():
    """A deeply nested CPU kernel (sqrt/asin/sin) mimicking backend math runs."""
    ti.init(arch=ti.cpu, default_fp=ti.f64, log_level="error")

    @ti.func
    def inner_math(a: ti.template(), b: ti.template()) -> ti.template():
        return ti.sqrt(a * a + b * b)

    @ti.func
    def middle_layer(x: ti.template()) -> ti.template():
        temp = inner_math(x, x * 2.0)
        return ti.asin(ti.sqrt(temp / 10.0))

    @ti.func
    def outer_calc(val: ti.template()) -> ti.template():
        result = middle_layer(val)
        for _ in range(5):
            result = result + ti.sin(result) * 0.1
        return result

    @ti.kernel
    def complex_kernel(x: ti.f64) -> ti.f64:
        return outer_calc(x)

    result = float(complex_kernel(2.0))
    assert np.isfinite(result)


@pytest.mark.skipif(not _vulkan_available(), reason="no Vulkan/GPU backend available")
def test_gpu_vulkan_f32_asin_template():
    """Vulkan GPU f32 with asin using ti.template() (the recommended form)."""
    ti.init(arch=ti.vulkan, default_fp=ti.f32, log_level="error")

    @ti.func
    def calc_with_asin_template(x: ti.template()) -> ti.template():
        return ti.asin(ti.sqrt(x))

    @ti.kernel
    def gpu_asin_kernel(x: ti.f32) -> ti.f32:
        return calc_with_asin_template(x)

    result = float(gpu_asin_kernel(0.25))
    expected = float(np.arcsin(np.sqrt(0.25)))
    assert abs(result - expected) / expected < 1e-3
