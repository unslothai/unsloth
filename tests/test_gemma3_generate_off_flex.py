# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""generate() of a Gemma 3 decoder that Unsloth put on flex_attention: prefill stays on flex, one-token
decode steps run the non-flex path."""

import inspect
import types

import pytest
import torch

import unsloth.models.vision as vision
from unsloth.models._utils import _EXPLICIT_FLEX_ATTR
from unsloth.models.vision import _NonFlexGenerate, _flex_generate_configs


@pytest.fixture(autouse = True)
def _no_opt_out(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FLEX_ATTENTION_GENERATE", raising = False)


def _text_model(model_type = "gemma3_text", attn = "flex_attention"):
    from transformers import Gemma3TextConfig, LlamaConfig

    cls = Gemma3TextConfig if model_type == "gemma3_text" else LlamaConfig
    config = cls(
        num_hidden_layers = 1,
        hidden_size = 32,
        intermediate_size = 64,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        vocab_size = 64,
    )
    config._attn_implementation = attn
    return _with_prepare(types.SimpleNamespace(config = config))


def _with_prepare(model):
    """A stand-in for GenerationMixin.prepare_inputs_for_generation: records the decoder config each
    call reads (the static cache builds the step mask there), returns that step's input_ids."""
    model.seen = []

    def prepare_inputs_for_generation(
        input_ids,
        cache_position = None,
        **kwargs,
    ):
        text = model.config.get_text_config()
        model.seen.append(text._attn_implementation)
        if cache_position is not None:
            input_ids = input_ids[:, -cache_position.shape[0] :]
        return {"input_ids": input_ids, "cache_position": cache_position}

    model.prepare_inputs_for_generation = prepare_inputs_for_generation
    return model


def _steps(
    model,
    prompt_len = 5,
    new_tokens = 3,
    cache_position = True,
):
    """Prefill then one-token steps, as generate() calls prepare_inputs_for_generation."""
    for step in range(new_tokens):
        n = prompt_len + step
        ids = torch.zeros(2, n, dtype = torch.long)
        kw = {}
        if cache_position:
            kw["cache_position"] = torch.arange(n) if step == 0 else torch.tensor([n - 1])
        elif step > 0:
            ids = ids[:, -1:]
        model.prepare_inputs_for_generation(ids, **kw)


def _vlm_model():
    from transformers import Gemma3Config

    config = Gemma3Config()
    config.text_config._attn_implementation = "flex_attention"
    config.vision_config._attn_implementation = "sdpa"
    return _with_prepare(types.SimpleNamespace(config = config))


def test_prefill_on_flex_decode_off_flex_and_restores():
    model = _text_model()
    prepare = model.prepare_inputs_for_generation
    with _NonFlexGenerate(model):
        _steps(model)
    assert model.seen == ["flex_attention", "eager", "eager"]
    assert model.config._attn_implementation == "flex_attention"
    assert model.prepare_inputs_for_generation is prepare
    assert not [k for k in model.__dict__ if k.startswith("_unsloth_non_flex")]


def test_step_kind_without_cache_position():
    # No cache_position: the first call guesses prefill, then the prepared length decides.
    model = _text_model()
    with _NonFlexGenerate(model):
        _steps(model, cache_position = False)
        # a one-token "prefill" (e.g. a 1-token prompt) is rebuilt on the decode path
        model.prepare_inputs_for_generation(torch.zeros(1, 1, dtype = torch.long))
    assert model.seen == ["flex_attention", "eager", "eager", "eager"]
    assert model.config._attn_implementation == "flex_attention"


def test_one_token_prompt_decodes_off_flex():
    model = _text_model()
    with _NonFlexGenerate(model):
        _steps(model, prompt_len = 1)
    assert model.seen == ["eager", "eager", "eager"]


def test_second_generate_prefills_on_flex_again():
    model = _text_model()
    for _ in range(2):
        with _NonFlexGenerate(model):
            _steps(model)
    assert model.seen == ["flex_attention", "eager", "eager"] * 2


def test_instance_prepare_is_restored():
    # PEFT puts its own prepare_inputs_for_generation on the base model during generate().
    model = _text_model()
    mine = model.prepare_inputs_for_generation
    with _NonFlexGenerate(model):
        assert model.prepare_inputs_for_generation is not mine
    assert model.__dict__["prepare_inputs_for_generation"] is mine


def test_overlapping_thread_waits_for_the_running_generate():
    # Thread B starting while A's flex prefill mask is built but not yet run must not switch the
    # shared config under it: B waits until A exits, then runs its own steps.
    import threading

    model = _text_model()
    outer = _NonFlexGenerate(model).__enter__()
    model.prepare_inputs_for_generation(torch.zeros(1, 8, dtype = torch.long))
    entered = threading.Event()

    def other():
        with _NonFlexGenerate(model):
            entered.set()
            _steps(model)

    thread = threading.Thread(target = other)
    thread.start()
    assert not entered.wait(0.2)
    assert model.config._attn_implementation == "flex_attention"
    outer.__exit__(None, None, None)
    thread.join(5)
    assert entered.is_set() and not thread.is_alive()
    assert model.seen == ["flex_attention", "flex_attention", "eager", "eager"]
    assert model.config._attn_implementation == "flex_attention"
    assert not [k for k in model.__dict__ if k.startswith("_unsloth_non_flex")]


def test_model_stays_copyable_after_generate():
    # The serialising lock lives outside the model: deepcopy / pickle must keep working.
    import copy, pickle

    model = _text_model()
    with _NonFlexGenerate(model):
        _steps(model)
    copy.deepcopy(model)
    pickle.dumps(model.config)
    assert not [k for k in model.__dict__ if "flex_generate" in k]


def test_restored_when_generate_raises():
    model = _text_model()
    prepare = model.prepare_inputs_for_generation
    with pytest.raises(RuntimeError):
        with _NonFlexGenerate(model):
            _steps(model)
            raise RuntimeError("boom")
    assert model.config._attn_implementation == "flex_attention"
    assert model.prepare_inputs_for_generation is prepare


def test_vlm_moves_only_the_decoder():
    model = _vlm_model()
    top = model.config._attn_implementation
    with _NonFlexGenerate(model):
        _steps(model)
        assert model.config.text_config._attn_implementation == "eager"
        assert model.config.vision_config._attn_implementation == "sdpa"
    assert model.seen == ["flex_attention", "eager", "eager"]
    assert model.config.text_config._attn_implementation == "flex_attention"
    assert model.config.vision_config._attn_implementation == "sdpa"
    assert model.config._attn_implementation == top


def test_nested_calls_restore_once():
    model = _text_model()
    with _NonFlexGenerate(model):
        with _NonFlexGenerate(model):
            _steps(model)
            assert model.config._attn_implementation == "eager"
        assert model.config._attn_implementation == "eager"
        _steps(model)
    assert model.seen == ["flex_attention", "eager", "eager"] * 2
    assert model.config._attn_implementation == "flex_attention"


@pytest.mark.parametrize(
    "model",
    [
        _text_model(attn = "eager"),
        _text_model(attn = "sdpa"),
        _text_model(model_type = "llama"),
    ],
    ids = ["gemma3-eager", "gemma3-sdpa", "llama-flex"],
)
def test_untouched_unless_unsloth_chose_flex_for_gemma3(model):
    before = model.config._attn_implementation
    assert _flex_generate_configs(model) == []
    with _NonFlexGenerate(model):
        _steps(model)
        assert model.config._attn_implementation == before
    assert set(model.seen) == {before}
    assert model.config._attn_implementation == before


def test_explicit_flex_request_is_kept():
    model = _text_model()
    setattr(model, _EXPLICIT_FLEX_ATTR, True)
    with _NonFlexGenerate(model):
        _steps(model)
    assert model.seen == ["flex_attention"] * 3


def test_opt_out(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FLEX_ATTENTION_GENERATE", "1")
    model = _text_model()
    with _NonFlexGenerate(model):
        _steps(model)
    assert model.seen == ["flex_attention"] * 3


def test_generate_enters_the_scope():
    source = inspect.getsource(vision.unsloth_base_fast_generate)
    assert "_NonFlexGenerate(self)" in source


def _tiny_gemma3():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        num_hidden_layers = 2,
        hidden_size = 64,
        intermediate_size = 128,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        vocab_size = 128,
        sliding_window = 8,
        max_position_embeddings = 256,
    )
    config._attn_implementation = "flex_attention"
    return Gemma3ForCausalLM(config).eval()


def _record_paths(model, layer):
    seen = []

    def hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        seen.append((model.config._attn_implementation, hidden.shape[1], type(mask).__name__))

    layer.self_attn.register_forward_pre_hook(hook, with_kwargs = True)
    return seen


def _generate_kwargs(new_tokens = 5):
    ids = torch.randint(0, 128, (2, 20))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    return dict(
        input_ids = ids,
        attention_mask = mask,
        max_new_tokens = new_tokens,
        do_sample = False,
        cache_implementation = "static",
    )


def test_real_generate_prefills_on_flex_and_decodes_off_flex():
    # A tiny Gemma 3 under transformers' generate() with a static cache: the prefill forward gets a
    # flex BlockMask, every decode forward a dense tensor mask on the non-flex path, and the ids match
    # SDPA end to end (left padding, prompt longer than the sliding window), so the decode steps read
    # the KV cache the flex prefill wrote.
    model = _tiny_gemma3()
    seen = _record_paths(model, model.model.layers[0])
    kw = _generate_kwargs()
    try:
        with _NonFlexGenerate(model):
            out = model.generate(**kw)
    except Exception as e:  # flex_attention unavailable on this CPU build
        pytest.skip(f"flex_attention generate failed on CPU: {e!r}")
    assert seen[0] == ("flex_attention", 20, "BlockMask")
    assert seen[1:] == [("eager", 1, "Tensor")] * 4
    assert model.config._attn_implementation == "flex_attention"
    model.config._attn_implementation = "sdpa"
    assert torch.equal(out, model.generate(**kw))


@pytest.mark.parametrize("base_patched", [False, True], ids = ["peft-only", "peft-and-base"])
def test_lora_generate_routes_each_step(base_patched):
    # Unsloth patches generate() on the PeftModel. PeftModelForCausalLM.generate hands off to the base
    # model, whose loop calls the base model's prepare_inputs_for_generation, so the scope must route
    # there too, with or without a second (nested) scope on the base model's own generate().
    peft = pytest.importorskip("peft")
    model = _tiny_gemma3()
    seen = _record_paths(model, model.model.layers[0])
    lora = peft.get_peft_model(model, peft.LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"]))

    def scoped(target):
        original = target.generate

        def generate(self, *args, **kwargs):
            with _NonFlexGenerate(self):
                return original(*args, **kwargs)

        target.generate = types.MethodType(generate, target)

    if base_patched:
        scoped(model)
    scoped(lora)
    kw = _generate_kwargs()
    try:
        out = lora.generate(**kw)
    except Exception as e:  # flex_attention unavailable on this CPU build
        pytest.skip(f"flex_attention generate failed on CPU: {e!r}")
    assert seen[0] == ("flex_attention", 20, "BlockMask")
    assert seen[1:] == [("eager", 1, "Tensor")] * 4
    assert model.config._attn_implementation == "flex_attention"
    for m in (lora, model, lora.base_model):
        assert "prepare_inputs_for_generation" not in m.__dict__
        assert not [k for k in m.__dict__ if k.startswith("_unsloth_non_flex")]
    model.config._attn_implementation = "sdpa"
    del lora.__dict__["generate"]
    assert torch.equal(out, lora.generate(**kw))
