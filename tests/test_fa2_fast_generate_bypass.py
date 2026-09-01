"""Regression coverage for the FlashAttention generation fallback."""

import ast
import inspect
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


VISION_PATH = Path(__file__).parents[1] / "unsloth" / "models" / "vision.py"


def _load_function(name, namespace):
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body = [function], type_ignores = []), str(VISION_PATH), "exec"), namespace)
    return namespace[name]


uses_flash_attention = _load_function(
    "_uses_flash_attention_for_generation",
    {
        "_config_get": lambda config, field, default = None: (
            config.get(field, default)
            if isinstance(config, dict)
            else getattr(config, field, default)
        ),
        "_is_flash_attention_requested": lambda value: (
            isinstance(value, str) and value.startswith("flash_attention")
        ),
    },
)
clear_generation_caches = _load_function("_clear_generation_caches", {})


def test_top_level_flash_attention_is_detected():
    config = SimpleNamespace(_attn_implementation = "flash_attention_2")
    assert uses_flash_attention(config)


def test_per_backbone_text_flash_attention_is_detected():
    private_config = SimpleNamespace(
        _attn_implementation = {
            "vision_config": "sdpa",
            "text_config": "flash_attention_2",
        }
    )
    public_config = SimpleNamespace(
        attn_implementation = {
            "vision_config": "sdpa",
            "text_config": "flash_attention_2",
        }
    )
    assert uses_flash_attention(private_config)
    assert uses_flash_attention(public_config)


def test_per_backbone_llm_flash_attention_is_detected():
    config = SimpleNamespace(
        _attn_implementation = {
            "vision_config": "sdpa",
            "llm_config": "flash_attention_2",
        }
    )
    assert uses_flash_attention(config)


def test_default_backbone_flash_attention_is_detected():
    config = SimpleNamespace(
        _attn_implementation = {
            "": "flash_attention_2",
            "vision_config": "sdpa",
        }
    )
    assert uses_flash_attention(config)


def test_explicit_language_backend_overrides_default_backend():
    config = SimpleNamespace(
        _attn_implementation = {
            "": "flash_attention_2",
            "text_config": "sdpa",
        }
    )
    assert not uses_flash_attention(config)


def test_nested_language_backend_overrides_normalized_default_backend():
    config = SimpleNamespace(
        _attn_implementation = "flash_attention_2",
        text_config = SimpleNamespace(_attn_implementation = "sdpa"),
    )
    assert not uses_flash_attention(config)

    nested_text = SimpleNamespace(_attn_implementation = "sdpa")
    thinker_config = SimpleNamespace(
        _attn_implementation = "flash_attention_2",
        sub_configs = {"text_config": object},
        text_config = nested_text,
        get_text_config = lambda: nested_text,
    )
    assert not uses_flash_attention(SimpleNamespace(thinker_config = thinker_config))


def test_nested_text_and_decoder_configs_are_detected():
    nested_text = SimpleNamespace(attn_implementation = "flash_attention_2")
    assert uses_flash_attention(
        SimpleNamespace(_attn_implementation = "sdpa", text_config = nested_text)
    )
    assert uses_flash_attention(
        SimpleNamespace(decoder_config = {"_attn_implementation": "flash_attention_2"})
    )


def test_nested_llm_config_is_detected():
    config = SimpleNamespace(llm_config = SimpleNamespace(_attn_implementation = "flash_attention_2"))
    assert uses_flash_attention(config)


def test_get_text_config_is_detected():
    nested_text = SimpleNamespace(_attn_implementation = "flash_attention_2")
    config = SimpleNamespace(get_text_config = lambda: nested_text)
    assert uses_flash_attention(config)


def test_declared_custom_generation_subconfig_is_detected():
    nested_text = SimpleNamespace(_attn_implementation = "flash_attention_2")
    custom_generation = SimpleNamespace(
        sub_configs = {"text_config": object},
        text_config = nested_text,
    )
    config = SimpleNamespace(
        sub_configs = {"custom_generation_config": object},
        custom_generation_config = custom_generation,
    )
    assert uses_flash_attention(config)
    assert uses_flash_attention(
        SimpleNamespace(
            _attn_implementation = {
                "thinker_config": "flash_attention_2",
                "vision_config": "sdpa",
            }
        )
    )


def test_vision_only_flash_attention_does_not_bypass_text_generation():
    config = SimpleNamespace(
        _attn_implementation = {
            "vision_config": "flash_attention_2",
            "text_config": "sdpa",
        }
    )
    assert not uses_flash_attention(config)


def test_non_flash_attention_does_not_bypass_fast_generation():
    assert not uses_flash_attention(SimpleNamespace(_attn_implementation = "sdpa"))
    assert not uses_flash_attention(SimpleNamespace())


def test_wrapper_dispatch_preserves_normalization_and_selects_expected_path():
    events = []

    class FakeTensor:
        shape = (1, 3)

        def __init__(self):
            self.converted_to = None

        def to(self, dtype):
            self.converted_to = dtype
            return self

    class FailIfUsed:
        def __getattr__(self, name):
            raise AssertionError(f"fast-generation path unexpectedly used torch._dynamo.{name}")

    fake_torch = SimpleNamespace(
        Tensor = FakeTensor,
        bfloat16 = "bfloat16",
        float16 = "float16",
        _dynamo = FailIfUsed(),
        inference_mode = nullcontext,
        autocast = lambda **kwargs: nullcontext(),
    )

    class FakeFastBaseModel:
        @staticmethod
        def for_inference(model):
            events.append("for_inference")

    architecture = "Qwen3VLForConditionalGeneration"
    namespace = {
        "torch": fake_torch,
        "os": os,
        "inspect": inspect,
        "FastBaseModel": FakeFastBaseModel,
        "dtype_from_config": lambda config: "bfloat16",
        "_get_dtype": lambda dtype: dtype,
        "_unsloth_generate_accepts_kwarg": lambda model, name: False,
        "NUM_LOGITS_TO_KEEP": {architecture: None},
        "DEVICE_TYPE_TORCH": "cuda",
        "_uses_flash_attention_for_generation": uses_flash_attention,
        "_clear_generation_caches": clear_generation_caches,
    }
    fast_generate = _load_function("unsloth_base_fast_generate", namespace)

    captured = {}
    cache_module = SimpleNamespace(_flex_attention_cache = object())

    class Model:
        config = SimpleNamespace(
            architectures = [architecture],
            eos_token_id = 2,
            text_config = SimpleNamespace(_attn_implementation = "flash_attention_2"),
        )

        def forward(self, input_ids = None):
            return input_ids

        def named_modules(self):
            return [("cache", cache_module)]

        def _old_generate(self, *args, **kwargs):
            assert not hasattr(cache_module, "_flex_attention_cache")
            captured.update(kwargs)
            cache_module._flex_attention_cache = object()
            return "fallback-result"

    input_ids = FakeTensor()
    pixel_values = FakeTensor()
    result = fast_generate(
        Model(),
        input_ids = input_ids,
        pixel_values = pixel_values,
        mm_token_type_ids = FakeTensor(),
    )

    assert result == "fallback-result"
    assert events == ["for_inference"]
    assert "mm_token_type_ids" not in captured
    assert captured["pixel_values"] is pixel_values
    assert pixel_values.converted_to == "bfloat16"
    assert not hasattr(cache_module, "_flex_attention_cache")

    class FastPathReached(Exception):
        pass

    class ExpectFastPath:
        @staticmethod
        def mark_static(*args, **kwargs):
            raise FastPathReached

    fake_torch._dynamo = ExpectFastPath()
    Model.config._attn_implementation = "flash_attention_2"
    Model.config.text_config._attn_implementation = "sdpa"
    captured.clear()
    try:
        fast_generate(Model(), input_ids = FakeTensor())
    except FastPathReached:
        pass
    else:
        raise AssertionError("non-FlashAttention generation did not enter the fast path")
    assert captured == {}


def test_flash_attention_fallback_pins_a_dynamic_cache():
    # Delegating is not enough on its own:
    namespace = {
        "torch": SimpleNamespace(
            Tensor = type("FakeTensor", (), {"shape": (1, 3)}),
            bfloat16 = "bfloat16",
            float16 = "float16",
            inference_mode = nullcontext,
            autocast = lambda **kwargs: nullcontext(),
        ),
        "os": os,
        "inspect": inspect,
        "FastBaseModel": SimpleNamespace(for_inference = lambda model: None),
        "dtype_from_config": lambda config: "bfloat16",
        "_get_dtype": lambda dtype: dtype,
        "_unsloth_generate_accepts_kwarg": lambda model, name: False,
        "NUM_LOGITS_TO_KEEP": {"Qwen3VLForConditionalGeneration": None},
        "DEVICE_TYPE_TORCH": "cuda",
        "_uses_flash_attention_for_generation": uses_flash_attention,
        "_clear_generation_caches": clear_generation_caches,
    }
    fast_generate = _load_function("unsloth_base_fast_generate", namespace)

    captured = {}

    class Model:
        config = SimpleNamespace(
            architectures = ["Qwen3VLForConditionalGeneration"],
            eos_token_id = 2,
            _attn_implementation = "flash_attention_2",
        )

        def forward(self, input_ids = None):
            return input_ids

        def named_modules(self):
            return []

        def _old_generate(self, *args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return "fallback-result"

    input_ids = namespace["torch"].Tensor()

    fast_generate(Model(), input_ids = input_ids)
    assert captured["cache_implementation"] == "dynamic"

    # The kwarg wins over a supplied generation_config, since update() applies it last.
    generation_config = SimpleNamespace(cache_implementation = "static")
    fast_generate(Model(), input_ids = input_ids, generation_config = generation_config)
    assert captured["cache_implementation"] == "dynamic"

    fast_generate(Model(), input_ids = input_ids, cache_implementation = "static")
    assert captured["cache_implementation"] == "dynamic"

    # generate() rejects a caller cache combined with any cache_implementation.
    cache = object()
    fast_generate(Model(), input_ids = input_ids, past_key_values = cache)
    assert "cache_implementation" not in captured
    assert captured["past_key_values"] is cache


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"OK: {len(tests)} FA2 fallback regression tests passed")
