#!/usr/bin/env python3
"""MLX/Metal Verification Script for Unsloth.

Tests pure MLX functionality without model downloads.
Use this to verify MLX/Metal is working correctly on Apple Silicon.
"""

import sys
import platform
import argparse


def print_header(title: str):
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print("-" * 60)


def test_system():
    print_header("1. System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("❌ MLX requires macOS on Apple Silicon (arm64)")
        return False
    print("✅ System compatible.")
    return True


def test_mlx_installation():
    print_header("2. MLX Installation")
    try:
        import importlib.metadata
        import mlx.core as mx
        
        try:
            mlx_version = importlib.metadata.version("mlx")
            print(f"MLX Version: {mlx_version}")
        except Exception:
            print("MLX Version: (unknown)")
        
        print(f"MLX Core: Available")
        
        default = mx.default_device()
        print(f"Default Device: {default}")
        
        if hasattr(mx, 'metal'):
            metal_available = mx.metal.is_available()
            print(f"Metal Backend: {'✅ Yes' if metal_available else '❌ No'}")
        else:
            print("Metal Backend: ❌ Not available")
        
        return True
    except ImportError as e:
        print(f"❌ MLX not installed: {e}")
        print("   Install with: pip install mlx")
        return False


def test_mlx_operations():
    print_header("3. MLX Core Operations")
    try:
        import mlx.core as mx
        
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = a + b
        mx.eval(c)
        print("✅ Basic array operations work")
        
        x = mx.random.uniform(shape=(256, 256))
        y = mx.matmul(x, x)
        mx.eval(y)
        print("✅ Matrix multiplication works")
        
        return True
    except Exception as e:
        print(f"❌ MLX operations failed: {e}")
        return False


def test_mlx_fast_ops():
    print_header("4. MLX Fast Operations (Metal)")
    try:
        import mlx.core as mx
        
        has_fast = hasattr(mx, 'fast')
        print(f"mx.fast available: {'✅ Yes' if has_fast else '❌ No'}")
        
        if has_fast:
            ops = ['rms_norm', 'layer_norm', 'rope', 'scaled_dot_product_attention']
            for op in ops:
                available = hasattr(mx.fast, op)
                print(f"  mx.fast.{op}: {'✅' if available else '❌'}")
        
        return has_fast
    except Exception as e:
        print(f"❌ MLX fast ops check failed: {e}")
        return False


def test_unsloth_mlx_kernels():
    print_header("5. Unsloth MLX Kernels")
    try:
        from unsloth.kernels.mlx.utils import is_mlx_available, get_mlx_version
        print(f"is_mlx_available(): {is_mlx_available()}")
        print(f"get_mlx_version(): {get_mlx_version()}")
    except ImportError as e:
        print(f"⚠️  Unsloth MLX utils not available: {e}")
    
    try:
        from unsloth.kernels.mlx.fast_ops import is_mlx_fast_available
        print(f"is_mlx_fast_available(): {is_mlx_fast_available()}")
        return True
    except Exception as e:
        print(f"⚠️  Fast ops check failed: {e}")
        return False


def test_mlx_lora():
    print_header("6. MLX LoRA Components")
    try:
        import mlx.core as mx
        from unsloth.kernels.mlx.lora import LoRALinear, LoRAConfig
        
        config = LoRAConfig(r=8, lora_alpha=16)
        print(f"✅ LoRAConfig created: r={config.r}, alpha={config.lora_alpha}")
        
        lora = LoRALinear(in_features=64, out_features=64, r=8, lora_alpha=16)
        print(f"✅ LoRALinear created")
        
        x = mx.random.uniform(shape=(2, 64))
        y = lora(x)
        mx.eval(y)
        print(f"✅ LoRA forward pass works, output shape: {y.shape}")
        
        return True
    except ImportError as e:
        print(f"❌ LoRA components not available: {e}")
        return False
    except Exception as e:
        print(f"❌ LoRA test failed: {e}")
        return False


def test_mlx_optimizers():
    print_header("7. MLX Optimizers")
    try:
        import mlx.core as mx
        from unsloth.kernels.mlx.optimizers import AdamW, SGDM
        
        
        params = {'w': mx.random.uniform(shape=(10, 10))}
        grads = {'w': mx.random.uniform(shape=(10, 10))}
        
        optimizer = AdamW(learning_rate=1e-4)
        print(f"✅ AdamW optimizer created")
        
        updated = optimizer(grads, params)
        print(f"✅ Optimizer step works")
        
        return True
    except ImportError as e:
        print(f"❌ Optimizers not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        return False


def test_mlx_bridge():
    print_header("8. MLX-Torch Bridge")
    try:
        import torch
        import mlx.core as mx
        from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch
        
        t = torch.randn(3, 4)
        mx_arr = torch_to_mlx(t)
        print(f"✅ torch_to_mlx: {t.shape} -> {mx_arr.shape}")
        
        t_back = mlx_to_torch(mx_arr)
        print(f"✅ mlx_to_torch: {mx_arr.shape} -> {t_back.shape}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Bridge not available (optional): {e}")
        return None
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        return False


def test_memory():
    print_header("9. Memory Info")
    try:
        import mlx.core as mx
        
        if hasattr(mx, 'metal') and hasattr(mx.metal, 'memory_info'):
            info = mx.metal.memory_info()
            print(f"Peak Memory: {info.get('peak_memory', 'N/A')}")
            print(f"Recommended Max: {info.get('recommended_max_memory', 'N/A')}")
        else:
            print("⚠️  Metal memory info not available in this MLX version")
        
        return True
    except Exception as e:
        print(f"⚠️  Memory info check failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Verify MLX/Metal support for Unsloth")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("     Unsloth MLX/Metal Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("System Check", test_system()))
    if not results[0][1]:
        print("\n❌ System incompatible with MLX")
        sys.exit(1)
    
    results.append(("MLX Installation", test_mlx_installation()))
    if not results[1][1]:
        print("\n❌ Please install MLX: pip install mlx")
        sys.exit(1)
    
    results.append(("MLX Operations", test_mlx_operations()))
    results.append(("MLX Fast Ops", test_mlx_fast_ops()))
    results.append(("Unsloth MLX Kernels", test_unsloth_mlx_kernels()))
    results.append(("MLX LoRA", test_mlx_lora()))
    results.append(("MLX Optimizers", test_mlx_optimizers()))
    results.append(("MLX-Torch Bridge", test_mlx_bridge()))
    results.append(("Memory Info", test_memory()))
    
    print_header("Summary")
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    for name, result in results:
        if result is True:
            print(f"✅ {name}")
        elif result is False:
            print(f"❌ {name}")
        else:
            print(f"⚠️  {name} (skipped/optional)")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Check output above for details.")
        sys.exit(1)
    else:
        print("\n✅ All MLX tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
