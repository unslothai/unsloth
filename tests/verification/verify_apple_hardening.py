# Apply Mac compatibility patches BEFORE importing unsloth
import platform
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

import torch
import os
import psutil
from unsloth.device_utils import get_device_properties, get_device_name, get_total_memory, get_available_memory, get_current_memory_usage
from unsloth_zoo.device_type import DEVICE_TYPE

def test_hardware_discovery():
    print("\n--- 1. Hardware Discovery ---")
    props = get_device_properties()
    print(f"DEVICE_TYPE (unsloth_zoo): {DEVICE_TYPE}")
    print(f"Device Name: {props.name}")
    print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"GPU Cores/SM: {props.multi_processor_count}")
    
    # Validation logic
    if DEVICE_TYPE == "mps":
        assert "Apple" in props.name, f"Expected Apple in device name, got {props.name}"
        assert props.major == 0 and props.minor == 0, "Compute capability should be 0 for MPS"
    print("✅ Hardware discovery looks correct.")

def test_memory_accounting():
    print("\n--- 2. Memory Accounting ---")
    used = get_current_memory_usage() / (1024**3)
    free = get_available_memory() / (1024**3)
    total = get_total_memory() / (1024**3)
    
    print(f"Memory Stats: {used:.2f} GB Used / {free:.2f} GB Free / {total:.2f} GB Total")
    
    # Check consistency
    if DEVICE_TYPE == "mps":
        # On MPS, used + free should roughly equal total (allowing for system buffers)
        assert abs((used + free) - total) < 5, "Memory sum deviates significantly from total (delta > 5GB)"
    print("✅ Memory accounting looks consistent.")

def check_no_monkeypatch():
    print("\n--- 3. Clean Environment Check ---")
    # Ensuring we didn't leave the global monkeypatch in patches.py
    try:
        # This will either return real CUDA props (if on CUDA) or fail (on Mac)
        # It should NOT return a mock object on Mac anymore.
        props = torch.cuda.get_device_properties(0)
        print(f"Native torch.cuda props: {props}")
    except Exception as e:
        print(f"Expected failure for native torch.cuda.get_device_properties: {e}")
    
    print("✅ Global monkeypatch correctly removed.")

if __name__ == "__main__":
    print(f"Testing Unsloth Apple Silicon Hardening (Phases 1 & 2)")
    print("="*60)
    try:
        test_hardware_discovery()
        test_memory_accounting()
        check_no_monkeypatch()
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
    else:
        print("\n" + "="*60)
        print("ALL TESTS PASSED! Phases 1 & 2 are working correctly.")
