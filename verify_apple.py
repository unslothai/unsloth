#!/usr/bin/env python
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import os
import sys
import platform
import time

def print_header(text):
    print(f"\n{'-'*60}")
    print(f" {text}")
    print(f"{'-'*60}")

def check_system():
    print_header("1. System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    
    if platform.system() != "Darwin":
        print("❌ CRITICAL: This script is intended for macOS (Apple Silicon).")
        return False
    
    if platform.machine() != "arm64":
        print("❌ CRITICAL: Unsloth optimized kernels require Apple Silicon (M1/M2/M3/M4).")
        return False
    
    print("✅ System compatible.")
    return True

def check_pytorch():
    print_header("2. PyTorch & MPS")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        mps_available = torch.backends.mps.is_available()
        print(f"MPS Available: {'✅ Yes' if mps_available else '❌ No'}")
        
        if not mps_available:
            print("❌ ERROR: MPS backend not available. Ensure you are on macOS 12.3+.")
            return False
            
        # Test basic MPS operation
        x = torch.ones(1024, device="mps")
        y = x + x
        torch.mps.synchronize()
        print("✅ Basic MPS operation successful.")
        
        # Check bfloat16 support
        try:
            bf16_test = torch.tensor([1.0], dtype=torch.bfloat16, device="mps")
            print("✅ bfloat16 supported on this hardware.")
        except:
            print("⚠️  bfloat16 not natively supported (Hardware limitation M1/M2?). Fallback to float16/float32.")
            
    except ImportError:
        print("❌ ERROR: PyTorch not found. Install via 'pip install torch'.")
        return False
    return True

def check_unsloth():
    print_header("3. Unsloth & Patches")
    try:
        import unsloth
        from unsloth.patches import is_patched
        from unsloth_zoo.device_type import DEVICE_TYPE
        
        print(f"Unsloth Version: {unsloth.__version__}")
        print(f"Device Type (Patched): {DEVICE_TYPE}")
        print(f"MPS Patch Applied: {'✅ Yes' if is_patched() else '❌ No (Check initialization order)'}")
        
        if DEVICE_TYPE != "mps":
            print("❌ ERROR: DEVICE_TYPE is not 'mps'. The patch failed to apply.")
            return False
            
    except ImportError:
        print("❌ ERROR: Unsloth not found. Install via 'pip install unsloth'.")
        return False
    return True

def smoke_test_model():
    print_header("4. Model Smoke Test (Llama-3.1-8B)")
    print("Running a quick forward pass test...")
    try:
        from unsloth import FastLanguageModel
        import torch
        
        model_name = "unsloth/Llama-3.1-8B-bnb-4bit" # Uses 4-bit MLX loading
        
        start_time = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 1024,
            load_in_4bit = True,
            device_map = "mps",
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        inputs = tokenizer(["What is the capital of France?"], return_tensors = "pt").to("mps")
        
        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens = 5)
        gen_time = time.time() - start_time
        
        print(f"✅ Generation successful in {gen_time:.2f}s")
        print(f"Output: {tokenizer.batch_decode(outputs)[0]}")
        
    except Exception as e:
        print(f"❌ ERROR during smoke test: {str(e)}")
        print("\nNote: This test requires internet access to download a small model segment.")
        return False
    return True

def main():
    print("="*60)
    print("           Unsloth Apple Silicon Verification")
    print("="*60)
    
    if not check_system(): sys.exit(1)
    if not check_pytorch(): sys.exit(1)
    if not check_unsloth(): sys.exit(1)
    
    print_header("Verification Summary")
    print("🎉 Your Mac is ready for Unsloth!")
    print("\nNext steps:")
    print("1. Check out the guide: /docs/apple-silicon-guide.md")
    print("2. Run a finetuning script from /examples")
    
    # Optional smoke test - skip by default as it downloads 5GB+
    # smoke_test_model()

if __name__ == "__main__":
    main()
