# SPDX-License-Identifier: AGPL-3.0-or-later
"""The large-head-dim flex routing: decoder-only scoping, opt-outs, config-driven detection,
and when Unsloth's mask wrapper skips the causal mask."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _text_only():
    return _Cfg(model_type = "fake", head_dim = 256, num_attention_heads = 8)


def _multimodal():
    text = _Cfg(model_type = "fake", head_dim = 256, num_attention_heads = 8)
    return _Cfg(
        model_type = "fake_vl",
        text_config = text,
        vision_config = _Cfg(model_type = "fake_vision", head_dim = 64, num_attention_heads = 8),
    )


@pytest.fixture(autouse = True)
def _reset_probe_cache(monkeypatch):
    monkeypatch.setattr(u, "_flex_kernels_fit_large_head_dim", lambda: True)
    u._ATTN_IMPL_MAPPING_SUPPORTED.clear()
    yield
    u._ATTN_IMPL_MAPPING_SUPPORTED.clear()


def test_no_mapping_support_refuses_flex_on_multimodal():
    u._ATTN_IMPL_MAPPING_SUPPORTED.append(False)
    assert u._flex_attn_impl_for(_multimodal(), "sdpa") is None


def test_no_mapping_support_still_allows_flex_on_text_only():
    u._ATTN_IMPL_MAPPING_SUPPORTED.append(False)
    assert u._flex_attn_impl_for(_text_only(), "sdpa") == "flex_attention"


def test_mapping_support_scopes_flex_to_the_decoder():
    u._ATTN_IMPL_MAPPING_SUPPORTED.append(True)
    got = u._flex_attn_impl_for(_multimodal(), "sdpa")
    assert isinstance(got, dict)
    assert got[""] == "sdpa"
    assert got["text_config"] == "flex_attention"
    assert "vision_config" not in got


def test_mapping_support_text_only_is_a_plain_string():
    u._ATTN_IMPL_MAPPING_SUPPORTED.append(True)
    assert u._flex_attn_impl_for(_text_only(), "sdpa") == "flex_attention"


# getattr cannot tell an inherited False (qwen3_5) from a deliberate one (T5Gemma2).


def _real_model_class(module_path, class_name):
    pytest.importorskip("transformers")
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except Exception:
        pytest.skip(f"{module_path} not available in this transformers")
    cls = getattr(mod, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not available in this transformers")
    return cls


def test_qwen3_5_only_inherits_the_base_default():
    cls = _real_model_class(
        "transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5ForConditionalGeneration"
    )
    assert u._declares_flex_support(cls) is None


def test_t5gemma2_declares_its_own_opt_out():
    cls = _real_model_class(
        "transformers.models.t5gemma2.modeling_t5gemma2", "T5Gemma2ForConditionalGeneration"
    )
    assert u._declares_flex_support(cls) is False


def test_force_enable_refuses_an_explicit_opt_out():
    cls = _real_model_class(
        "transformers.models.t5gemma2.modeling_t5gemma2", "T5Gemma2ForConditionalGeneration"
    )
    u._FLEX_SUPPORT_FORCED.clear()
    assert u._enable_flex_attention_support(cls, "t5gemma2") is False
    # and it must not have mutated the class on the way out
    assert u._declares_flex_support(cls) is False


def test_force_enable_still_opts_in_qwen3_5():
    cls = _real_model_class(
        "transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5ForConditionalGeneration"
    )
    u._FLEX_SUPPORT_FORCED.clear()
    assert u._enable_flex_attention_support(cls, "qwen3_5") is True


@pytest.fixture
def _no_env(monkeypatch):
    monkeypatch.delenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, raising = False)


def test_large_head_dim_is_detected_from_config_by_default(_no_env):
    assert u._prefers_flex_for_head_dim(_text_only()) is True


def test_small_head_dim_is_left_on_sdpa_by_default(_no_env):
    assert (
        u._prefers_flex_for_head_dim(_Cfg(model_type = "llama", head_dim = 128, num_attention_heads = 8))
        is False
    )


def test_head_dim_derived_from_hidden_size_when_absent(_no_env):
    # older configs omit head_dim; hidden_size / num_attention_heads is the same quantity
    assert (
        u._prefers_flex_for_head_dim(
            _Cfg(model_type = "fake", hidden_size = 2048, num_attention_heads = 8)
        )
        is True
    )
    assert (
        u._prefers_flex_for_head_dim(
            _Cfg(model_type = "fake", hidden_size = 1024, num_attention_heads = 8)
        )
        is False
    )


def test_per_layer_head_dims_take_the_maximum(_no_env):
    # 5.x per_layer_config: the largest layer decides.
    cfg = _Cfg(
        model_type = "fake",
        head_dim = 128,
        num_attention_heads = 8,
        per_layer_config = [
            _Cfg(head_dim = 128),
            _Cfg(head_dim = 128),
            _Cfg(head_dim = 512),
        ],
    )
    assert u._text_attention_head_dim(cfg) == 512
    assert u._prefers_flex_for_head_dim(cfg) is True


def test_a_homogeneous_small_config_is_unaffected_by_the_per_layer_read(_no_env):
    # 4.x configs have no per_layer_config at all; the global head_dim must still decide.
    assert (
        u._prefers_flex_for_head_dim(_Cfg(model_type = "fake", head_dim = 128, num_attention_heads = 8))
        is False
    )


def test_excluded_model_stays_on_sdpa_even_at_large_head_dim(_no_env):
    assert (
        u._prefers_flex_for_head_dim(_Cfg(model_type = "gemma2", head_dim = 256, num_attention_heads = 8))
        is False
    )


def test_a_vision_tower_alone_never_turns_it_on(_no_env):
    cfg = _Cfg(
        model_type = "fake_vl",
        text_config = _Cfg(model_type = "fake", head_dim = 64, num_attention_heads = 8),
        vision_config = _Cfg(model_type = "fake_vision", head_dim = 256, num_attention_heads = 8),
    )
    assert u._prefers_flex_for_head_dim(cfg) is False


def test_missing_head_dim_is_not_a_guess(_no_env):
    assert u._prefers_flex_for_head_dim(_Cfg(model_type = "fake")) is False


@pytest.mark.parametrize("value", ["0", " 0 "])
def test_env_var_zero_forces_sdpa(monkeypatch, value):
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, value)
    assert u._prefers_flex_for_head_dim(_text_only()) is False


@pytest.mark.parametrize("value", ["1", "true", " 1 "])
def test_env_var_nonzero_forces_flex(monkeypatch, value):
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, value)
    assert (
        u._prefers_flex_for_head_dim(_Cfg(model_type = "llama", head_dim = 64, num_attention_heads = 8))
        is True
    )


def test_empty_env_var_falls_back_to_the_config(monkeypatch):
    # an exported-but-empty variable is the shell's "unset", not a request to disable
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "")
    assert u._prefers_flex_for_head_dim(_text_only()) is True
    assert (
        u._prefers_flex_for_head_dim(_Cfg(model_type = "llama", head_dim = 64, num_attention_heads = 8))
        is False
    )


@pytest.mark.parametrize("env, expected", [("1", "flex_attention"), (None, "flash_attention_2")])
def test_forcing_the_env_var_outranks_flash_attention(monkeypatch, env, expected):
    import transformers as T

    monkeypatch.setattr(u, "HAS_FLASH_ATTENTION", True)
    if env is None:
        monkeypatch.delenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, raising = False)
    else:
        monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, env)
    cfg = T.LlamaConfig(
        hidden_size = 512,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        head_dim = 128,
        num_hidden_layers = 1,
        intermediate_size = 64,
        vocab_size = 128,
    )
    resolved = u.resolve_attention_implementation(T.LlamaForCausalLM, cfg)
    assert resolved == expected


@pytest.mark.parametrize(
    "mapping, expected",
    [(False, "flash_attention_2"), (True, {"": "sdpa", "text_config": "flex_attention"})],
)
def test_forcing_flex_that_cannot_be_scoped_keeps_flash_attention(monkeypatch, mapping, expected):
    import transformers as T

    monkeypatch.setattr(u, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "1")
    u._ATTN_IMPL_MAPPING_SUPPORTED.append(mapping)
    resolved = u.resolve_attention_implementation(T.LlamaForCausalLM, _multimodal())
    assert resolved == expected


def test_forcing_the_env_var_cannot_override_an_architecture_opt_out(monkeypatch):
    # a deliberate _supports_flex_attn = False still wins over the env var
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "1")
    cls = _real_model_class(
        "transformers.models.t5gemma2.modeling_t5gemma2", "T5Gemma2ForConditionalGeneration"
    )
    u._FLEX_SUPPORT_FORCED.clear()
    assert u._enable_flex_attention_support(cls, "t5gemma2") is False


# Upstream skips the mask for an unpadded batch, but `_ignore_causal_mask_sdpa` returns False
# while tracing, so a compiled create_causal_mask always builds one. unsloth-zoo#1575 restores the
# skip in the wrapper: an unpadded, cache-free SDPA call runs the original eagerly and gets None,
# while padding, packed position_ids and UNSLOTH_SKIP_CAUSAL_MASK=0 keep the mask. Pins that
# contract, and that with UNSLOTH_COMPILE_DISABLE=1 the uncompiled wrapper keeps upstream's skip.


def _mask_for(
    create,
    attention_mask,
    q_len = 64,
    head_dim = 256,
    bsz = 2,
    position_ids = "arange",
):
    import inspect

    import torch
    import transformers as T
    from transformers import masking_utils

    cfg = T.LlamaConfig(
        hidden_size = 2048,
        num_attention_heads = 8,
        num_key_value_heads = 8,
        num_hidden_layers = 1,
        head_dim = head_dim,
        vocab_size = 128,
    )
    cfg._attn_implementation = "sdpa"
    # Read the upstream signature, not the wrapper's (*args, **kwargs): transformers <= 5.1 names
    # it `input_embeds` and requires `cache_position`, 5.2+ renamed it and 5.9 dropped the cache.
    original = getattr(
        masking_utils, "_unsloth_original_create_causal_mask", masking_utils.create_causal_mask
    )
    parameters = inspect.signature(original).parameters
    embeds_name = "input_embeds" if "input_embeds" in parameters else "inputs_embeds"
    if isinstance(position_ids, str):
        position_ids = torch.arange(q_len).unsqueeze(0).expand(bsz, -1)
    kwargs = {
        "config": cfg,
        embeds_name: torch.zeros(bsz, q_len, cfg.hidden_size, dtype = torch.bfloat16),
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    if "cache_position" in parameters:
        kwargs["cache_position"] = torch.arange(q_len)
    return create(**kwargs)


def _padded_masks():
    import torch

    right = torch.ones(2, 64, dtype = torch.long)
    right[0, -8:] = 0
    left = torch.ones(2, 64, dtype = torch.long)
    left[0, :8] = 0
    return right, left


def _packed_position_ids():
    import torch

    # Two sequences of 32 packed into each row: position ids reset mid-row.
    return torch.arange(32).repeat(2).unsqueeze(0).expand(2, -1)


def _uncompiled_create_causal_mask():
    from transformers import masking_utils

    original = getattr(masking_utils, "_unsloth_original_create_causal_mask", None)
    if original is None:
        pytest.skip("this transformers/unsloth pair does not stash the original")
    return original


def _wrapper_skips_unpadded_masks():
    """Whether the installed unsloth_zoo has the unpadded causal-mask skip (unsloth-zoo#1575)."""
    try:
        from unsloth_zoo.temporary_patches import misc
    except Exception:
        return False
    return hasattr(misc, "CAUSAL_MASK_SKIP_STATS") and hasattr(misc, "_maskless_causal_arguments")


def _skip_count():
    from unsloth_zoo.temporary_patches.misc import CAUSAL_MASK_SKIP_STATS
    return CAUSAL_MASK_SKIP_STATS["skipped"]


@pytest.fixture
def _skip_on(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SKIP_CAUSAL_MASK", raising = False)


def test_upstream_skips_the_mask_for_an_unpadded_batch():
    import torch

    create = _uncompiled_create_causal_mask()
    # No position_ids with no mask: transformers 4.x treats any position_ids as possibly packed
    # (find_packed_sequence_indices never returns None there) and then keeps the mask.
    assert _mask_for(create, None, position_ids = None) is None
    assert _mask_for(create, torch.ones(2, 64, dtype = torch.long)) is None


def test_upstream_still_materialises_a_mask_when_padded():
    create = _uncompiled_create_causal_mask()
    right, left = _padded_masks()
    assert _mask_for(create, right) is not None
    assert _mask_for(create, left) is not None


def _mask_wrapper_is_compiled():
    """Whether the installed create_causal_mask wrapper calls a compiled function.

    Read off the wrapper rather than UNSLOTH_COMPILE_DISABLE: unsloth_zoo decides once, when it
    patches at import, and a test module can flip the variable later in the same process. With
    the switch set, zoo still installs the wrapper (its keyword fixes apply) around the
    uncompiled original, which is what it stashes.
    """
    import inspect

    from transformers import masking_utils

    original = _uncompiled_create_causal_mask()
    try:
        inner = inspect.getclosurevars(masking_utils.create_causal_mask).nonlocals.get("f")
    except TypeError:
        inner = None
    if inner is None:
        pytest.skip("cannot see which function the mask wrapper calls")
    return inner is not original


def test_compiled_wrapper_skips_the_mask_for_an_unpadded_batch(_skip_on):
    """An unpadded, cache-free SDPA batch gets no mask, so SDPA can take its is_causal kernels."""
    import torch
    from transformers import masking_utils

    if not _mask_wrapper_is_compiled():
        pytest.skip("the mask wrapper calls the uncompiled original in this run")
    create = masking_utils.create_causal_mask
    if not _wrapper_skips_unpadded_masks():
        # Older unsloth_zoo: tracing makes `_ignore_causal_mask_sdpa` refuse, so the mask is built.
        assert _mask_for(create, None) is not None
        return
    before = _skip_count()
    assert _mask_for(create, None) is None
    assert _mask_for(create, torch.ones(2, 64, dtype = torch.long)) is None
    # The skip came from the wrapper's eager path, not from a silent fallback.
    assert _skip_count() == before + 2


def test_compiled_wrapper_keeps_the_mask_when_padded_or_packed(_skip_on):
    from transformers import masking_utils

    create = masking_utils.create_causal_mask
    right, left = _padded_masks()
    before = _skip_count() if _wrapper_skips_unpadded_masks() else None
    assert _mask_for(create, right) is not None
    assert _mask_for(create, left) is not None
    assert _mask_for(create, None, position_ids = _packed_position_ids()) is not None
    if before is not None:
        assert _skip_count() == before


def test_kill_switch_keeps_the_mask_for_an_unpadded_batch(monkeypatch):
    import torch
    from transformers import masking_utils

    if not _wrapper_skips_unpadded_masks():
        pytest.skip("this unsloth_zoo has no unpadded causal-mask skip")
    if not _mask_wrapper_is_compiled():
        pytest.skip("uncompiled, the original skips the mask itself")
    monkeypatch.setenv("UNSLOTH_SKIP_CAUSAL_MASK", "0")
    create = masking_utils.create_causal_mask
    before = _skip_count()
    assert _mask_for(create, None) is not None
    assert _mask_for(create, torch.ones(2, 64, dtype = torch.long)) is not None
    assert _skip_count() == before


def test_an_uncompiled_wrapper_keeps_the_upstream_skip(_skip_on):
    """Without compilation the wrapper agrees with upstream on unpadded and padded batches.

    unsloth-zoo#1335 made the wrapper install under UNSLOTH_COMPILE_DISABLE=1 too, around the
    uncompiled original. The mask is then skipped exactly as upstream skips it, and kept exactly
    where upstream keeps it.
    """
    import torch
    from transformers import masking_utils

    if _mask_wrapper_is_compiled():
        pytest.skip("the mask wrapper is compiled in this run")
    create = masking_utils.create_causal_mask
    assert _mask_for(create, None, position_ids = None) is None
    assert _mask_for(create, torch.ones(2, 64, dtype = torch.long)) is None
    right, left = _padded_masks()
    assert _mask_for(create, right) is not None
    assert _mask_for(create, left) is not None
