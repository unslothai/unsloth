
import sys
import os
import torch
import platform
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMistralMPS(unittest.TestCase):
    def setUp(self):
        if platform.system() != "Darwin":
            self.skipTest("Not running on macOS")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")

    def test_mistral_forward_pass(self):
        print("\n" + "=" * 60)
        print("Running Mistral MPS Integration Test")
        print("=" * 60)

        # Import Unsloth
        try:
            from unsloth import FastLanguageModel
            import unsloth.kernels.mps.dispatch as dispatch_module
        except Exception as e:
            self.fail(f"Failed to import Unsloth: {e}")

        # Instrument dispatch to verify kernels
        # Mistral uses RoPE and potentially SwiGLU/GeGLU depending on config, but standard Mistral is SwiGLU.
        # Mistral uses SwiGLU: https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
        kernel_counts = {
            "dispatch_rms_layernorm": 0, 
            "dispatch_rope_embedding": 0,
            "dispatch_lora_mlp_swiglu": 0
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
            print("Loading mistralai/Mistral-7B-v0.1 (16-bit) for MPS testing...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "mistralai/Mistral-7B-v0.1",
                max_seq_length = 512,
                load_in_4bit = False, # Must be False on Mac (no bitsandbytes)
                dtype = torch.float16,
            )

            device = next(model.parameters()).device
            print(f"Model loaded on: {device}")
            # Ensure it's on MPS (or meta if offloaded, but active tensors should allow mps)
            
            # Prepare inputs
            inputs = tokenizer("Unsloth makes Mistral faster because", return_tensors="pt").to(device)

            # Forward pass
            print("Running forward pass...")
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"Output shape: {outputs.logits.shape}")
            self.assertIsNotNone(outputs.logits)
            
            print("Kernel usages:", kernel_counts)
            
            # Check for silent failures or fallbacks
            # If dispatch_rope_embedding is 0, it might mean we fell back to eager implementation 
            # OR we are using MLX directly without the dispatcher wrapper (if implemented that way).
            # But currently `MistralAttention_fast_forward` calls `dispatch_rope_embedding`.
            
        except Exception as e:
            print(f"Mistral test failed with: {e}")
            raise e
        finally:
            # Restore
            for name, orig in original_funcs.items():
                setattr(dispatch_module, name, orig)

if __name__ == "__main__":
    unittest.main()
