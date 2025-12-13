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
    messages_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
    ]
    messages_new = [
        {"role": "user", "content": "What is 2+2?"},
    ]

    # Prepare text
    text_history = tokenizer.apply_chat_template(messages_history, tokenize=False, add_generation_prompt=False)
    # We want to ensure the full prompt is exactly history + new.
    # Some tokenizers might add a space or merge tokens at the boundary if we just concat strings.
    # Let's tokenize history first.
    inputs_history = tokenizer(text_history, return_tensors="pt").to("cuda")
    len_history_tokens = inputs_history.input_ids.shape[1]

    # Now tokenize the full conversation
    text_full = tokenizer.apply_chat_template(messages_history + messages_new, tokenize=False, add_generation_prompt=True)
    inputs_full = tokenizer(text_full, return_tensors="pt").to("cuda")
    len_full_tokens = inputs_full.input_ids.shape[1]

    print(f"\nHistory Length (tokens): {len_history_tokens}")
    print(f"Full Prompt Length (tokens): {len_full_tokens}")

    # Verify Prefix Match
    prefix_match = torch.equal(inputs_full.input_ids[:, :len_history_tokens], inputs_history.input_ids)
    print(f"Prefix Match: {prefix_match}")

    if not prefix_match:
        print("WARNING: Tokenization mismatch! Attempting to fix...")
        # If they don't match, we must construct inputs_full carefully.
        # But for chat templates, we usually want to respect the template.
        # Let's see WHERE they differ.
        diff_idx = (inputs_full.input_ids[:, :len_history_tokens] != inputs_history.input_ids).nonzero()
        if len(diff_idx) > 0:
            print(f"First difference at index: {diff_idx[0].tolist()}")
            print(f"History token: {inputs_history.input_ids[0, diff_idx[0, 1]]}")
            print(f"Full token:    {inputs_full.input_ids[0, diff_idx[0, 1]]}")
        
        # Force alignment for the sake of the test if needed, but better to understand why.
        # Often it's BOS token or space.
        # Let's try to just use inputs_full for everything to be consistent.
        inputs_history_fixed = inputs_full.input_ids[:, :len_history_tokens]
        # But wait, if we use inputs_full prefix, we must ensure the KV cache generated from it 
        # is what we want.
        # Actually, if we pass `past_key_values`, the model expects the cache to match the prefix of `input_ids`.
        # So we should use the prefix of `inputs_full` to generate the cache.
        print("Re-generating inputs_history from inputs_full prefix to ensure alignment.")
        inputs_history = tokenizer(text_full, return_tensors="pt").to("cuda") # Just a dummy container
        inputs_history.input_ids = inputs_full.input_ids[:, :len_history_tokens]
        inputs_history.attention_mask = inputs_full.attention_mask[:, :len_history_tokens]

    # 3. Pre-compute KV Cache for the history
    print("\nPre-computing KV cache for history...")
    with torch.no_grad():
        outputs_history = model(**inputs_history, use_cache=True)
        past_key_values_history = outputs_history.past_key_values
    print("KV Cache computed.")

    # 4. Baseline Generation
    print("\n--- Baseline Generation (No KV passed) ---")
    
    # Warmup
    model.generate(**inputs_full, max_new_tokens=1)
    torch.cuda.synchronize()
    
    start_time = time.time()
    output_baseline = model.generate(
        **inputs_full, 
        max_new_tokens=50, 
        use_cache=True,
        do_sample=False,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    
    time_baseline = end_time - start_time
    new_tokens_baseline = output_baseline[0][len_full_tokens:]
    response_text_baseline = tokenizer.decode(new_tokens_baseline, skip_special_tokens=True)

    print(f"Time: {time_baseline:.4f}s")
    print(f"Output: {response_text_baseline.strip()}")

    # 5. KV Cache Generation
    print("\n--- KV Cache Generation (Passing custom KV) ---")
    
    # We pass the SAME inputs_full. Unsloth/HF should detect we passed past_key_values 
    # and slice input_ids to just the new tokens automatically?
    # Wait, standard HF `generate` with `past_key_values` expects `input_ids` to be ONLY the new tokens 
    # IF the model is not prepared to handle full inputs. 
    # But Unsloth's `_fast_prepare_inputs_for_generation` handles slicing!
    # It checks `past_length` vs `input_ids.shape[1]`.
    
    # Let's verify what we are passing.
    # inputs_full has length `len_full_tokens`.
    # past_key_values_history has length `len_history_tokens`.
    # So `input_ids.shape[1] (full) > past_length (history)` is True.
    # Unsloth should slice it: `input_ids = input_ids[:, past_length:]`.
    
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
    
    # output_kv includes input_ids? Usually yes.
    # But if we passed past_key_values, does it return full sequence?
    # Yes, generate usually returns full sequence.
    
    # However, if Unsloth sliced the input internally, `generate` might be confused about what "input" was?
    # No, `generate` manages the loop.
    
    # Let's decode carefully.
    # If output_kv is just new tokens (unlikely), or full.
    if output_kv.shape[1] > len_full_tokens:
        new_tokens_kv = output_kv[0][len_full_tokens:]
    else:
        # Fallback if something weird happened
        new_tokens_kv = output_kv[0]
        
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
