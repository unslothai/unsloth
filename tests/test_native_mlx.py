import torch
from unsloth import FastLanguageModel
import time
import os

# Test with a community model that HAS a standard HF shell but MLX weights
# mlx-community/Llama-3.2-1B-4bit usually has the standard config.json
model_name = "mlx-community/Llama-3.2-1B-4bit"


def test_native_mlx_loading():
    print(f"\n--- Testing Native MLX Loading: {model_name} ---")

    # This should:
    # 1. Load the model architecture (shell)
    # 2. Detect the MLX weights on the Hub
    # 3. Download and map them directly
    # 4. Skip "Fast 4-bit quantization" progress bar

    start_time = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")

    # Check if weights are mapped
    mapped_count = 0
    for name, param in model.named_parameters():
        if hasattr(param, "_mlx_cache"):
            mapped_count += 1

    if mapped_count > 0:
        print(
            f"✅ Success: {mapped_count} weights were natively mapped from MLX safetensors."
        )
    else:
        print(
            f"❌ Failure: No weights were natively mapped. Check logs for mapping errors."
        )

    # Simple generation test
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(["The capital of France is"], return_tensors = "pt").to("mps")
    outputs = model.generate(**inputs, max_new_tokens = 20)
    print("Output:", tokenizer.batch_decode(outputs)[0])


if __name__ == "__main__":
    test_native_mlx_loading()
