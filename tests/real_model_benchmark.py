import torch
from unsloth import FastLanguageModel
import time
import os

# Results for Llama-3-8B 
model_name = "unsloth/llama-3-8b-bnb-4bit" # Or any small model

def benchmark_model(load_in_4bit=True):
    print(f"\n--- Benchmarking Model (load_in_4bit={load_in_4bit}) ---")
    
    start_time = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")

    # Simple prompt
    inputs = tokenizer(["What is the capital of France?"], return_tensors = "pt").to("mps")
    
    # Warmup
    _ = model.generate(**inputs, max_new_tokens = 1)
    
    # Benchmark generation
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens = 50)
    generation_time = time.time() - start
    
    # Decode
    print(tokenizer.batch_decode(outputs)[0])
    print(f"Generation time: {generation_time:.2f} seconds.")
    
    # Memory
    if torch.backends.mps.is_available():
        mem = torch.mps.current_allocated_memory() / 1e9
        print(f"MPS Allocated Memory: {mem:.2f} GB")

if __name__ == "__main__":
    # Note: On Mac, load_in_4bit=True will automatically use the new MLX backend
    benchmark_model(load_in_4bit=True)
