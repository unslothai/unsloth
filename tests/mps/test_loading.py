import os
import torch
from unsloth import FastLanguageModel
import mlx.core as mx
from unsloth.kernels.mlx.quantization import MLXQuantizedWeight


def test_mps_4bit_loading():
    print("Testing Seamless 4-bit Loading on MPS...")

    # Use a small model if possible, but Llama-3.2-1B-Instruct is the default.
    model_name = "unsloth/Llama-3.2-1B-Instruct"

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=128,
            dtype=None,
            load_in_4bit=True,  # This triggers our new path
            device_map="mps",
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ Failed to load model: {e}")
        return

    print("✅ Model loaded successfully!")

    # Verify Quantization
    print("Verifying Quantization...")
    layer = model.model.layers[0].mlp.gate_proj

    if not hasattr(layer.weight, "_mlx_cache"):
        print("❌ Layer missing _mlx_cache! Quantization hook failed.")
        return

    cache = layer.weight._mlx_cache
    if isinstance(cache, MLXQuantizedWeight):
        print(
            f"✅ Weight is quantized! (Group size: {cache.group_size}, Bits: {cache.bits})"
        )
    else:
        print(f"❌ Cache is not MLXQuantizedWeight: {type(cache)}")
        return

    # Run Inference
    print("Running Inference (Generation)...")
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("mps")

    # Should use the quantized kernels automatically
    outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)

    text = tokenizer.decode(outputs[0])
    print(f"Generated: {text}")
    print("✅ Inference completed without crash.")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        test_mps_4bit_loading()
    else:
        print("Skipping MPS test on non-MPS device.")
