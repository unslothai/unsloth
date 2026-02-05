
import sys
import os
import torch
import platform
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestQwen2MPS(unittest.TestCase):
    def setUp(self):
        if platform.system() != "Darwin":
            self.skipTest("Not running on macOS")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")

    def test_qwen2_forward_pass(self):
        print("\n" + "=" * 60)
        print("Running Qwen2 MPS Integration Test")
        print("=" * 60)

        try:
            from unsloth import FastLanguageModel
            import unsloth.kernels.mps.dispatch as dispatch_module
        except Exception as e:
            self.fail(f"Failed to import Unsloth: {e}")

        # Instrument dispatch
        # Qwen2 uses LlamaStructure, so it should use:
        # - dispatch_rms_layernorm
        # - dispatch_rope_embedding
        # - dispatch_lora_mlp_swiglu (standard Qwen2 uses SwiGLU)
        # - dispatch_lora_qkv (if LoRA is tested or if we use the fast attention path that calls standard qkv logic? actually fast_qkv usually)
        
        kernel_counts = {
            "dispatch_rms_layernorm": 0,
            "dispatch_rope_embedding": 0,
            "dispatch_swiglu_fg": 0 # Qwen2 uses SwiGLU
        }

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
            print("Loading Qwen/Qwen2-0.5B (16-bit) for MPS testing...")
            model_name = "Qwen/Qwen2-0.5B" # Small model for faster testing
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = 512,
                load_in_4bit = False, # Must be False on Mac (no bitsandbytes)
                dtype = torch.float16,
            )

            device = next(model.parameters()).device
            print(f"Model loaded on: {device}")

            # Prepare inputs
            inputs = tokenizer("Qwen2 is compatible with Unsloth because", return_tensors="pt").to(device)

            # Forward pass
            print("Running forward pass...")
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"Output shape: {outputs.logits.shape}")
            self.assertIsNotNone(outputs.logits)
            
            print("Kernel usages:", kernel_counts)
            
        except Exception as e:
            print(f"Qwen2 test failed with: {e}")
            raise e
        finally:
            # Restore
            for name, orig in original_funcs.items():
                setattr(dispatch_module, name, orig)

if __name__ == "__main__":
    unittest.main()
