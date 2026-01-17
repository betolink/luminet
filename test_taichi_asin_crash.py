"""Test the specific Taichi GPU crash with asin and complex math."""
import taichi as ti
import sys

print("Testing Taichi GPU f32 with transcendental functions...\n")

# Test 1: CPU f32 with hardcoded f64 functions using asin
print("Test 1: CPU f32 with asin (hardcoded ti.f64)")
ti.init(arch=ti.cpu, default_fp=ti.f32, log_level='error')

@ti.func
def calc_with_asin_f64(x: ti.f64) -> ti.f64:
    """Use asin with hardcoded f64 type."""
    return ti.asin(ti.sqrt(x))

@ti.kernel
def test_cpu_asin(x: ti.f32) -> ti.f32:
    return calc_with_asin_f64(x)

try:
    result = test_cpu_asin(0.25)
    print(f"  ✅ CPU result: {result:.6f}")
except Exception as e:
    print(f"  ❌ CPU failed: {e}")

ti.reset()

# Test 2: GPU Vulkan f32 with hardcoded f64 asin
print("\nTest 2: GPU Vulkan f32 with asin (hardcoded ti.f64)")
try:
    ti.init(arch=ti.vulkan, default_fp=ti.f32, log_level='error')
    
    @ti.func
    def calc_with_asin_f64_gpu(x: ti.f64) -> ti.f64:
        """Use asin with hardcoded f64 type."""
        return ti.asin(ti.sqrt(x))
    
    @ti.kernel
    def test_gpu_asin_f64(x: ti.f32) -> ti.f32:
        return calc_with_asin_f64_gpu(x)
    
    result = test_gpu_asin_f64(0.25)
    print(f"  ✅ GPU result: {result:.6f}")
except Exception as e:
    print(f"  ❌ GPU failed: {type(e).__name__}: {str(e)[:100]}")

ti.reset()

# Test 3: GPU Vulkan f32 with ti.template() asin  
print("\nTest 3: GPU Vulkan f32 with asin (ti.template)")
try:
    ti.init(arch=ti.vulkan, default_fp=ti.f32, log_level='error')
    
    @ti.func
    def calc_with_asin_template(x: ti.template()) -> ti.template():
        """Use asin with template type."""
        return ti.asin(ti.sqrt(x))
    
    @ti.kernel
    def test_gpu_asin_template(x: ti.f32) -> ti.f32:
        return calc_with_asin_template(x)
    
    result = test_gpu_asin_template(0.25)
    print(f"  ✅ GPU result: {result:.6f}")
except Exception as e:
    print(f"  ❌ GPU failed: {type(e).__name__}: {str(e)[:100]}")

ti.reset()

# Test 4: Complex nested functions like the actual backend
print("\nTest 4: Complex nested functions (simulating BlackHole init)")
try:
    ti.init(arch=ti.vulkan, default_fp=ti.f32, log_level='error')
    
    @ti.func
    def inner_math(a: ti.f64, b: ti.f64) -> ti.f64:
        return ti.sqrt(a * a + b * b)
    
    @ti.func  
    def middle_layer(x: ti.f64) -> ti.f64:
        temp = inner_math(x, x * 2.0)
        return ti.asin(ti.sqrt(temp / 10.0))
    
    @ti.func
    def outer_calc(val: ti.f64) -> ti.f64:
        result = middle_layer(val)
        for i in range(5):
            result = result + ti.sin(result) * 0.1
        return result
    
    @ti.kernel
    def complex_kernel(x: ti.f32) -> ti.f32:
        return outer_calc(x)
    
    result = complex_kernel(2.0)
    print(f"  ✅ Complex GPU result: {result:.6f}")
except Exception as e:
    error_msg = str(e)
    if 'does not 64bits operation' in error_msg or 'Asin' in error_msg:
        print(f"  ❌ GPU CRASH (this is our bug!): {error_msg[:200]}")
    else:
        print(f"  ❌ GPU failed: {type(e).__name__}: {error_msg[:100]}")

ti.reset()

print("\n" + "="*70)
print("CONCLUSION:")
print("  If Test 2 or 4 failed with 'does not 64bits operation',")
print("  then we MUST replace all ti.f64 with ti.template()")
print("="*70)
