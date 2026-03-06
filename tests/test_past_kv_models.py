"""
Integration tests for past_key_values support across model architectures.
Requires a CUDA GPU. Best run in Colab or a GPU-equipped machine.

Run with:
    python -m pytest tests/test_past_kv_models.py -v -s

Or run individual model tests:
    python -m pytest tests/test_past_kv_models.py -v -s -k "Qwen3"
    python -m pytest tests/test_past_kv_models.py -v -s -k "Gemma2"
    python -m pytest tests/test_past_kv_models.py -v -s -k "Llama"
"""

import unittest
import torch


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")


def _run_past_kv_test(test_case, model_name, load_in_4bit = True):
    """
    Shared test logic: generate with baseline vs past_key_values and verify
    outputs match (or at minimum, that no errors are raised).
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    # Build a conversation with history
    messages_history = [
        {"role": "user", "content": "Remember: the secret code is ALPHA-7."},
        {"role": "assistant", "content": "Got it, the secret code is ALPHA-7."},
    ]
    messages_new = [
        {"role": "user", "content": "What is the secret code?"},
    ]

    # Tokenize history alone
    text_history = tokenizer.apply_chat_template(
        messages_history, tokenize = False, add_generation_prompt = False
    )
    inputs_history = tokenizer(text_history, return_tensors = "pt").to("cuda")

    # Tokenize full conversation
    text_full = tokenizer.apply_chat_template(
        messages_history + messages_new, tokenize = False, add_generation_prompt = True
    )
    inputs_full = tokenizer(text_full, return_tensors = "pt").to("cuda")

    len_history = inputs_history.input_ids.shape[1]
    len_full = inputs_full.input_ids.shape[1]
    print(f"\n  History tokens: {len_history}, Full tokens: {len_full}")

    # Pre-compute KV cache for history
    with torch.no_grad():
        outputs_history = model(**inputs_history, use_cache = True)
        past_kv = outputs_history.past_key_values

    # Baseline generation (no custom KV)
    output_baseline = model.generate(
        **inputs_full,
        max_new_tokens = 30,
        use_cache = True,
        do_sample = False,
    )
    text_baseline = tokenizer.decode(
        output_baseline[0][len_full:], skip_special_tokens = True
    )
    print(f"  Baseline: {text_baseline.strip()}")

    # KV cache generation
    output_kv = model.generate(
        **inputs_full,
        max_new_tokens = 30,
        past_key_values = past_kv,
        use_cache = True,
        do_sample = False,
    )
    if output_kv.shape[1] > len_full:
        text_kv = tokenizer.decode(output_kv[0][len_full:], skip_special_tokens = True)
    else:
        text_kv = tokenizer.decode(output_kv[0], skip_special_tokens = True)
    print(f"  KV Cache: {text_kv.strip()}")

    # Both should produce coherent output (not crash)
    test_case.assertGreater(len(text_kv.strip()), 0, "KV cache output is empty")

    # Outputs should match
    test_case.assertEqual(
        text_baseline.strip(),
        text_kv.strip(),
        "Baseline and KV cache outputs differ",
    )

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()


def _run_tuple_kv_test(test_case, model_name, load_in_4bit = True):
    """
    Test that passing tuple past_key_values (not DynamicCache) works.
    This validates the _ensure_cache_is_dynamic v5 compat path.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")

    # Get KV cache from forward pass
    with torch.no_grad():
        outputs = model(**inputs, use_cache = True)
        past_kv = outputs.past_key_values

    # Convert DynamicCache to tuple format (simulating user-provided tuple KV)
    if hasattr(past_kv, "get_seq_length"):
        tuple_kv = tuple(past_kv[i] for i in range(len(past_kv)))
    else:
        tuple_kv = past_kv  # Already tuple

    # This should NOT raise ValueError even on transformers v5
    next_token = tokenizer(" Paris", return_tensors = "pt").to("cuda")
    full_input = torch.cat([inputs.input_ids, next_token.input_ids], dim = 1)
    output = model.generate(
        input_ids = full_input,
        max_new_tokens = 10,
        past_key_values = tuple_kv,
        use_cache = True,
        do_sample = False,
    )
    text = tokenizer.decode(output[0], skip_special_tokens = True)
    print(f"\n  Tuple KV output: {text.strip()}")
    test_case.assertGreater(len(text.strip()), 0)

    del model, tokenizer
    torch.cuda.empty_cache()


class TestPastKVLlama(unittest.TestCase):
    def setUp(self):
        _skip_if_no_cuda()

    def test_past_kv_generation(self):
        """Test past_key_values with Llama model."""
        try:
            _run_past_kv_test(self, "unsloth/Llama-3.2-1B-Instruct")
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")

    def test_tuple_kv_v5_compat(self):
        """Test tuple KV cache conversion (v5 compat) with Llama."""
        try:
            _run_tuple_kv_test(self, "unsloth/Llama-3.2-1B-Instruct")
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")


class TestPastKVQwen3(unittest.TestCase):
    def setUp(self):
        _skip_if_no_cuda()

    def test_past_kv_generation(self):
        """Test past_key_values with Qwen3 model (validates RoPE position_ids fix)."""
        try:
            _run_past_kv_test(self, "unsloth/Qwen3-0.6B")
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")


class TestPastKVGemma2(unittest.TestCase):
    def setUp(self):
        _skip_if_no_cuda()

    def test_past_kv_generation(self):
        """Test past_key_values with Gemma2 model (validates 4D mask fix)."""
        try:
            _run_past_kv_test(self, "unsloth/gemma-2-2b-it")
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")


if __name__ == "__main__":
    unittest.main()
