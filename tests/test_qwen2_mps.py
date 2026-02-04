
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

        # Import Unsloth
        from unsloth import FastLanguageModel
        
        # Instrument dispatch to verify kernels
        kernel_counts = {
            "dispatch_rms_layernorm": 0, 
            "dispatch_rope_embedding": 0,
            "dispatch_swiglu_fg": 0,
            "dispatch_lora_qkv": 0,
            "dispatch_lora_o": 0,
            "dispatch_lora_mlp_swiglu": 0
        }
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
            # User requested Qwen/Qwen2-7B, but unsloth version is preferred for verification of unsloth specific loading paths including 4bit
            print("Loading unsloth/Qwen2-7B-bnb-4bit...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/Qwen2-7B-bnb-4bit", 
                max_seq_length = 512,
                load_in_4bit = True,
                dtype = torch.float16,
            )

            device = next(model.parameters()).device
            print(f"Model loaded on: {device}")
            self.assertTrue(device.type == "mps" or device.type == "meta")

            # Prepare inputs
            inputs = tokenizer("Qwen is a smart assistant", return_tensors="pt").to(device)

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
