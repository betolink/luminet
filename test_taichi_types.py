"""Test Taichi type flexibility for GPU f32 support."""
import taichi as ti

# Test 1: ti.template() for generic typing
ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.func
def generic_func(x: ti.template()) -> ti.template():
    """Generic function that works with any type."""
    return ti.sqrt(x * 2.0)

@ti.kernel
def test_kernel_f32(x: ti.f32) -> ti.f32:
    return generic_func(x)

@ti.kernel  
def test_kernel_f64(x: ti.f64) -> ti.f64:
    return generic_func(x)

print("Test 1: ti.template() for generic types")
try:
    result_f32 = test_kernel_f32(16.0)
    result_f64 = test_kernel_f64(16.0)
    print(f"  f32 result: {result_f32:.6f}")
    print(f"  f64 result: {result_f64:.6f}")
    print("  ✅ ti.template() works!")
except Exception as e:
    print(f"  ❌ Failed: {e}")

ti.reset()

# Test 2: Explicit f32 initialization with f64 functions
print("\nTest 2: f32 init with f64 function signatures (current bug)")
ti.init(arch=ti.cpu, default_fp=ti.f32)

@ti.func
def hardcoded_f64_func(x: ti.f64) -> ti.f64:
    """Function with hardcoded f64 types."""
    return ti.sqrt(x * 2.0)

@ti.kernel
def test_mixed_types(x: ti.f32) -> ti.f32:
    # This should fail - calling f64 function with f32 value
    return hardcoded_f64_func(x)

try:
    result = test_mixed_types(16.0)
    print(f"  Unexpected success: {result}")
except Exception as e:
    print(f"  ❌ Expected failure: {type(e).__name__}")

ti.reset()

# Test 3: GPU Vulkan with f32
print("\nTest 3: Vulkan GPU with f32 + generic functions")
try:
    ti.init(arch=ti.vulkan, default_fp=ti.f32, log_level='error')
    
    @ti.func
    def gpu_generic_func(x: ti.template()) -> ti.template():
        """GPU-compatible generic function."""
        return ti.sqrt(x) + ti.sin(x)
    
    @ti.kernel
    def gpu_test_kernel(x: ti.f32) -> ti.f32:
        return gpu_generic_func(x)
    
    result = gpu_test_kernel(2.0)
    print(f"  Vulkan f32 result: {result:.6f}")
    print("  ✅ GPU with ti.template() works!")
    ti.reset()
except Exception as e:
    print(f"  ❌ GPU test failed: {e}")
    ti.reset()

print("\nConclusion: Use ti.template() instead of ti.f64 for GPU f32 support")
