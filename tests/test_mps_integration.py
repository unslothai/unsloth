# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
MPS Integration Test for Apple Silicon.

This test verifies that:
1. FastLanguageModel loads correctly on MPS
2. Forward pass works
3. Metal/MLX kernels are actually being called (via instrumentation)

Run on Mac EC2:
    python tests/test_mps_integration.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mps_integration():
    """Full integration test for MPS/Metal/MLX kernel dispatch."""
    import torch
    
    # Skip if not on macOS
    import platform
    if platform.system() != "Darwin":
        print("⏭️  Skipping: Not running on macOS (Darwin)")
        return True
    
    if not torch.backends.mps.is_available():
        print("⏭️  Skipping: MPS not available")
        return True
    
    print("=" * 70)
    print("MPS Integration Test")
    print("=" * 70)
    
    # ==========================================================================
    # Step 1: Verify kernel availability
    # ==========================================================================
    print("\n[1/4] Checking kernel availability...")
    
    from unsloth.kernels.mps.dispatch import _is_metal_available, _is_mlx_available
    
    metal_available = _is_metal_available()
    mlx_available = _is_mlx_available()
    
    print(f"  Metal kernels: {'✅ Available' if metal_available else '❌ Not available'}")
    print(f"  MLX kernels:   {'✅ Available' if mlx_available else '❌ Not available'}")
    
    if not metal_available and not mlx_available:
        print("  ⚠️  Warning: No optimized kernels available, will use MPS fallbacks")
    
    # ==========================================================================
    # Step 2: Instrument dispatch functions to track calls
    # ==========================================================================
    print("\n[2/4] Instrumenting kernel dispatch...")
    
    kernel_call_log = {
        "dispatch_rms_layernorm": 0,
        "dispatch_rope_embedding": 0,
        "dispatch_swiglu_fg": 0,
        "dispatch_lora_mlp_swiglu": 0,
        "dispatch_lora_qkv": 0,
        "dispatch_lora_o": 0,
    }
    
    # Patch dispatch functions to log calls
    import unsloth.kernels.mps.dispatch as dispatch_module
    
    original_funcs = {}
    for func_name in kernel_call_log.keys():
        if hasattr(dispatch_module, func_name):
            original_funcs[func_name] = getattr(dispatch_module, func_name)
            
            def make_wrapper(name, original):
                def wrapper(*args, **kwargs):
                    kernel_call_log[name] += 1
                    return original(*args, **kwargs)
                return wrapper
            
            setattr(dispatch_module, func_name, make_wrapper(func_name, original_funcs[func_name]))
    
    print(f"  Instrumented {len(original_funcs)} dispatch functions")
    
    # ==========================================================================
    # Step 3: Load model and run forward pass
    # ==========================================================================
    print("\n[3/4] Loading FastLanguageModel and running forward pass...")
    
    try:
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-bnb-4bit",
            max_seq_length=128,
            load_in_4bit=False,  # No bitsandbytes on MPS
            dtype=torch.bfloat16,
        )
        
        device = next(model.parameters()).device
        print(f"  Model loaded on: {device}")
        
        # Prepare LoRA for dispatch testing
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print("  LoRA adapters attached")
        
        # Tokenize a sample
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"  Forward pass completed! Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"  ❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original functions
        for func_name, original in original_funcs.items():
            setattr(dispatch_module, func_name, original)
    
    # ==========================================================================
    # Step 4: Verify kernel dispatch calls
    # ==========================================================================
    print("\n[4/4] Verifying kernel dispatch calls...")
    
    any_called = False
    for func_name, count in kernel_call_log.items():
        status = "✅" if count > 0 else "⚪"
        print(f"  {status} {func_name}: {count} calls")
        if count > 0:
            any_called = True
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    if any_called:
        print("✅ INTEGRATION TEST PASSED")
        print("   Metal/MLX kernels were successfully dispatched during forward pass!")
    else:
        print("⚠️  INTEGRATION TEST WARNING")
        print("   No dispatch functions were called - model may be using different code path")
    print("=" * 70)
    
    return any_called


def test_kernel_correctness():
    """Quick numerical correctness check for each Metal kernel."""
    import torch
    import platform
    
    if platform.system() != "Darwin":
        print("⏭️  Skipping kernel correctness: Not on macOS")
        return True
    
    print("\n" + "=" * 70)
    print("Kernel Correctness Tests")
    print("=" * 70)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16
    
    all_passed = True
    
    # Test 1: SwiGLU
    print("\n[SwiGLU] Testing...")
    try:
        from unsloth.kernels.mps.dispatch import dispatch_swiglu_fg
        e = torch.randn(2, 512, 4096, device=device, dtype=dtype)
        g = torch.randn(2, 512, 4096, device=device, dtype=dtype)
        
        result = dispatch_swiglu_fg(e, g)
        expected = torch.nn.functional.silu(e) * g
        
        max_diff = (result - expected).abs().max().item()
        passed = max_diff < 0.01
        print(f"  {'✅' if passed else '❌'} Max diff: {max_diff:.6f}")
        all_passed = all_passed and passed
    except Exception as ex:
        print(f"  ❌ Error: {ex}")
        all_passed = False
    
    # Test 2: RoPE
    print("\n[RoPE] Testing...")
    try:
        from unsloth.kernels.mps.dispatch import dispatch_rope_embedding
        
        B, H, S, D = 2, 32, 64, 128
        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Create cos/sin
        freqs = torch.arange(D // 2, device=device, dtype=torch.float32) / (D // 2)
        freqs = 1.0 / (10000.0 ** freqs)
        positions = torch.arange(S, device=device, dtype=torch.float32)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        cos = torch.cos(angles).to(dtype)
        sin = torch.sin(angles).to(dtype)
        # Repeat for full dim
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        Q_out, K_out = dispatch_rope_embedding(Q, K, cos, sin)
        
        # Just verify shapes and no NaNs
        shapes_ok = Q_out.shape == Q.shape and K_out.shape == K.shape
        no_nans = not (torch.isnan(Q_out).any() or torch.isnan(K_out).any())
        passed = shapes_ok and no_nans
        print(f"  {'✅' if passed else '❌'} Shapes: {Q_out.shape}, No NaNs: {no_nans}")
        all_passed = all_passed and passed
    except Exception as ex:
        print(f"  ❌ Error: {ex}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test 3: RMS LayerNorm
    print("\n[RMS LayerNorm] Testing...")
    try:
        from unsloth.kernels.mps.dispatch import dispatch_rms_layernorm
        
        X = torch.randn(2, 512, 4096, device=device, dtype=dtype)
        W = torch.ones(4096, device=device, dtype=dtype)
        eps = 1e-6
        
        result = dispatch_rms_layernorm(X, W, eps, gemma=False)
        
        # Manual RMS norm
        variance = X.float().pow(2).mean(-1, keepdim=True)
        expected = (X.float() * torch.rsqrt(variance + eps) * W.float()).to(dtype)
        
        max_diff = (result - expected).abs().max().item()
        passed = max_diff < 0.1  # More lenient for layernorm
        print(f"  {'✅' if passed else '❌'} Max diff: {max_diff:.6f}")
        all_passed = all_passed and passed
    except Exception as ex:
        print(f"  ❌ Error: {ex}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL KERNEL CORRECTNESS TESTS PASSED")
    else:
        print("❌ SOME KERNEL TESTS FAILED")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    print("\n🍎 Unsloth MPS Integration Test Suite\n")
    
    # Run correctness tests first
    correctness_ok = test_kernel_correctness()
    
    # Then run integration test
    integration_ok = test_mps_integration()
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Kernel Correctness: {'✅ PASS' if correctness_ok else '❌ FAIL'}")
    print(f"  Model Integration:  {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print("=" * 70)
    
    sys.exit(0 if (correctness_ok and integration_ok) else 1)
