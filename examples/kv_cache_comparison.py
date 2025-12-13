import torch
from unsloth import FastLanguageModel
import time

def main():
    # 1. Load Model
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    # 2. Setup Inputs
    # We simulate a conversation history
    # History: User says "Hello", Assistant says "Hi there! How can I help you?"
    # New Input: User says "What is 2+2?"
    
    # We want to compare generating the response to "What is 2+2?"
    # Case A: Baseline - Pass full history + new input.
    # Case B: KV Cache - Pass full history + new input, BUT provide KV cache for the history.

    messages_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
    ]
    messages_new = [
        {"role": "user", "content": "What is 2+2?"},
    ]

    # Prepare text for history and full conversation
    # Note: apply_chat_template returns a string
    text_history = tokenizer.apply_chat_template(messages_history, tokenize=False, add_generation_prompt=False)
    text_full = tokenizer.apply_chat_template(messages_history + messages_new, tokenize=False, add_generation_prompt=True)

    print(f"\nHistory Length (chars): {len(text_history)}")
    print(f"Full Prompt Length (chars): {len(text_full)}")

    # 3. Pre-compute KV Cache for the history (Simulating that we have it from previous turns)
    print("\nPre-computing KV cache for history...")
    inputs_history = tokenizer(text_history, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # Run a forward pass to get the cache
        # We use the underlying model to get past_key_values
        outputs_history = model(**inputs_history, use_cache=True)
        past_key_values_history = outputs_history.past_key_values
    print("KV Cache computed.")

    # 4. Baseline Generation (No external KV passed)
    print("\n--- Baseline Generation (No KV passed) ---")
    inputs_full = tokenizer(text_full, return_tensors="pt").to("cuda")
    
    # Warmup
    model.generate(**inputs_full, max_new_tokens=1)
    torch.cuda.synchronize()
    
    start_time = time.time()
    output_baseline = model.generate(
        **inputs_full, 
        max_new_tokens=50, 
        use_cache=True,
        do_sample=False, # Deterministic for comparison
    )
    torch.cuda.synchronize()
    end_time = time.time()
    
    time_baseline = end_time - start_time
    text_baseline = tokenizer.decode(output_baseline[0], skip_special_tokens=True)
    # Extract just the new response
    response_baseline = text_baseline[len(tokenizer.decode(inputs_full.input_ids[0], skip_special_tokens=True)):]
    # Actually decoding the whole thing and splitting is safer for display, but let's just show the last part
    # Or better, just decode the new tokens if we can find where they start. 
    # output_baseline includes input_ids.
    new_tokens_baseline = output_baseline[0][inputs_full.input_ids.shape[1]:]
    response_text_baseline = tokenizer.decode(new_tokens_baseline, skip_special_tokens=True)

    print(f"Time: {time_baseline:.4f}s")
    print(f"Output: {response_text_baseline.strip()}")

    # 5. KV Cache Generation
    print("\n--- KV Cache Generation (Passing custom KV) ---")
    
    # Warmup (optional, but good for fairness if we did it for baseline)
    # model.generate(**inputs_full, past_key_values=past_key_values_history, max_new_tokens=1)
    
    torch.cuda.synchronize()
    start_time = time.time()
    output_kv = model.generate(
        **inputs_full, 
        max_new_tokens=50, 
        past_key_values=past_key_values_history,
        use_cache=True,
        do_sample=False,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    
    time_kv = end_time - start_time
    
    # output_kv also includes input_ids. 
    # Note: When passing past_key_values, the model might return the full sequence or just new tokens depending on implementation,
    # but standard HF generate returns full sequence (input + new).
    new_tokens_kv = output_kv[0][inputs_full.input_ids.shape[1]:]
    response_text_kv = tokenizer.decode(new_tokens_kv, skip_special_tokens=True)

    print(f"Time: {time_kv:.4f}s")
    print(f"Output: {response_text_kv.strip()}")

    # 6. Comparison
    print("\n--- Comparison ---")
    print(f"Baseline Time: {time_baseline:.4f}s")
    print(f"KV Cache Time: {time_kv:.4f}s")
    print(f"Speedup: {time_baseline / time_kv:.2f}x")
    
    if response_text_baseline == response_text_kv:
        print("SUCCESS: Outputs match perfectly.")
    else:
        print("WARNING: Outputs do not match.")
        print(f"Baseline: {response_text_baseline}")
        print(f"KV Cache: {response_text_kv}")

if __name__ == "__main__":
    main()
