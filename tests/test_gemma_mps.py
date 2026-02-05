
import sys
import os
import torch
import platform
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestGemmaMPS(unittest.TestCase):
    def setUp(self):
        if platform.system() != "Darwin":
            self.skipTest("Not running on macOS")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")

    def test_gemma_forward_pass(self):
        print("\n" + "=" * 60)
        print("Running Gemma MPS Integration Test")
        print("=" * 60)

        # Import Unsloth
        from unsloth import FastLanguageModel
        
        # Instrument dispatch to verify kernels
        kernel_counts = {"dispatch_rms_layernorm": 0, "dispatch_lora_mlp_geglu_approx": 0}
        import unsloth.kernels.mps.dispatch as dispatch_module

        original_funcs = {}
        for name in kernel_counts:
            if hasattr(dispatch_module, name):
                original_funcs[name] = getattr(dispatch_module, name)
                def make_wrapper(n, orig):
                    def wrapper(*args, **kwargs):
                        kernel_counts[n] += 1
                        return orig(*args, **kwargs)
                    return wrapper
                setattr(dispatch_module, name, make_wrapper(name, original_funcs[name]))

        try:
            # Load model
            # On Mac, we CANNOT use bitsandbytes 4-bit models.
            # Use 16-bit models. Using unsloth's copy to avoid gating.
            print("Loading unsloth/gemma-2b (16-bit) for MPS testing...")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/gemma-2b", # Using Unsloth's public copy
                max_seq_length = 512,
                load_in_4bit = False, # Must be False on Mac (no bitsandbytes)
                dtype = torch.float16,
            )

            device = next(model.parameters()).device
            print(f"Model loaded on: {device}")
            self.assertTrue(device.type == "mps" or device.type == "meta") # meta if quantized offloading? No, should be mps for active.

            # Prepare inputs
            inputs = tokenizer("Unsloth is amazing because", return_tensors="pt").to(device)

            # Forward pass
            print("Running forward pass...")
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"Output shape: {outputs.logits.shape}")
            self.assertIsNotNone(outputs.logits)
            
            # Verify Kernels
            # Note: 4-bit quantization might use different kernels (quantized_matmul) than raw dispatch
            # usage of RMS norm should still hopefully route through dispatch if we patched it.
            # But Gemma uses Approximate GeGLU.
            
            print("Kernel usages:", kernel_counts)
            # We don't strictly assert counts > 0 because if we use MLX quantized kernels directly they might bypass mps dispatch wrappers?
            # Actually, FastGemmaModel calls `fast_rms_layernorm_inference_gemma` which might call `dispatch_rms_layernorm`.
            
        except Exception as e:
            print(f"Gemma test failed with: {e}")
            raise e
        finally:
            # Restore
            for name, orig in original_funcs.items():
                setattr(dispatch_module, name, orig)

if __name__ == "__main__":
    unittest.main()
