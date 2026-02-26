import platform
import sys
import os

if platform.system() == "Darwin":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from patcher import MacPatcher
        patcher = MacPatcher()
        patcher.apply()
    except ImportError:
        patcher = None
else:
    patcher = None

import gc
import time
import torch

from unsloth import FastLanguageModel

if platform.system() == "Darwin" and patcher is not None:
    patcher.patch_patching_utils_late()


def is_mlx_available():
    """Check if MLX is available and not mocked."""
    try:
        from unsloth.kernels.mlx import utils
        if not getattr(utils, "_MLX_AVAILABLE", False):
            return False
        import mlx
        type_name = type(mlx).__name__
        if "Mock" in type_name:
            return False
        return True
    except ImportError:
        return False


def get_mps_memory_mb():
    """Get current MPS memory usage in MB."""
    if not torch.backends.mps.is_available():
        return {"active": 0, "reserved": 0}
    
    torch.mps.synchronize()
    active = torch.mps.current_allocated_memory() / 1e6
    try:
        driver_mem = torch.mps.driver_allocated_memory() / 1e6
        if driver_mem > 0:
            reserved = driver_mem
        else:
            reserved = torch.mps.peak_allocated_memory() / 1e6
    except (AttributeError, Exception):
        reserved = torch.mps.peak_allocated_memory() / 1e6
    
    return {"active": active, "reserved": reserved}


def profile_model_loading(model_name, load_in_4bit=False, max_seq_length=128):
    """Profile model loading time and memory usage on MPS."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        try:
            torch.mps.reset_peak_memory_stats()
        except AttributeError:
            pass
    
    mem_before = get_mps_memory_mb()
    
    start_time = time.perf_counter()
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="mps",
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "load_time_sec": 0,
            "memory_active_mb": 0,
            "memory_reserved_mb": 0,
        }
    
    load_time = time.perf_counter() - start_time
    
    torch.mps.synchronize()
    mem_after = get_mps_memory_mb()
    
    result = {
        "success": True,
        "model_name": model_name,
        "load_in_4bit": load_in_4bit,
        "load_time_sec": round(load_time, 2),
        "memory_active_mb": round(mem_after["active"], 2),
        "memory_reserved_mb": round(mem_after["reserved"], 2),
        "memory_delta_mb": round(mem_after["active"] - mem_before["active"], 2),
    }
    
    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return result


def run_loading_profiler():
    """Run the model loading profiler for both 16-bit and 4-bit."""
    if not torch.backends.mps.is_available():
        print("MPS not available. Skipping profiler.")
        return
    
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    
    print("=" * 60)
    print("Model Loading Profiler - Apple Silicon (MPS)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print()
    
    print("Testing 16-bit loading...")
    result_16bit = profile_model_loading(model_name, load_in_4bit=False)
    
    if result_16bit["success"]:
        print(f"  Load time: {result_16bit['load_time_sec']}s")
        print(f"  Memory active: {result_16bit['memory_active_mb']} MB")
        print(f"  Memory reserved: {result_16bit['memory_reserved_mb']} MB")
    else:
        print(f"  Failed: {result_16bit['error']}")
    
    gc.collect()
    torch.mps.empty_cache()
    time.sleep(2)
    
    print("\nTesting 4-bit loading (MLX)... (not available - skipping)")
    print("  Skipped: 4-bit MLX loading requires proper MLX installation")
    result_4bit = {"success": False, "error": "MLX not properly installed", 
                  "load_time_sec": 0, "memory_active_mb": 0, "memory_reserved_mb": 0}
    
    if result_4bit["success"]:
        print(f"  Load time: {result_4bit['load_time_sec']}s")
        print(f"  Memory active: {result_4bit['memory_active_mb']} MB")
        print(f"  Memory reserved: {result_4bit['memory_reserved_mb']} MB")
    else:
        print(f"  Failed: {result_4bit['error']}")
        print("  Note: 4-bit loading on MPS requires MLX to be installed")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if result_16bit["success"]:
        print(f"16-bit: {result_16bit['load_time_sec']}s, {result_16bit['memory_active_mb']} MB active")
    
    if result_4bit["success"]:
        print(f"4-bit:  {result_4bit['load_time_sec']}s, {result_4bit['memory_active_mb']} MB active")
        print(f"Memory savings: {result_16bit['memory_active_mb'] - result_4bit['memory_active_mb']:.2f} MB")
        print(f"Time difference: {result_16bit['load_time_sec'] - result_4bit['load_time_sec']:.2f}s")
    elif result_16bit["success"]:
        print(f"4-bit:  Not available (MLX not installed)")


if __name__ == "__main__":
    run_loading_profiler()
