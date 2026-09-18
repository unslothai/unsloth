# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gemma 4 stays on SDPA: all its layers share one _attn_implementation, and flex would
disable unsloth_zoo's gemma4_flash_sliding router ("sdpa" key) for its 25 sliding layers."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, raising = False)
    monkeypatch.setattr(u, "_sdpa_reaches_cudnn_at_head_dim_256", lambda: False)


def _gemma4():
    text = _Cfg(
        model_type = "gemma4_text",
        head_dim = 256,
        global_head_dim = 512,
        sliding_window = 1024,
        num_attention_heads = 16,
    )
    return _Cfg(
        model_type = "gemma4",
        text_config = text,
        vision_config = _Cfg(model_type = "gemma4_vision", head_dim = 72, num_attention_heads = 16),
    )


def test_gemma4_is_not_routed_to_flex():
    assert u._prefers_flex_for_head_dim(_gemma4()) is False


def test_gemma4_text_only_is_not_routed_either():
    # A text-only load presents gemma4_text at the top.
    text_only = _Cfg(
        model_type = "gemma4_text",
        head_dim = 256,
        global_head_dim = 512,
        num_attention_heads = 16,
    )
    assert u._prefers_flex_for_head_dim(text_only) is False


def test_the_head_dim_is_still_large_so_this_really_is_the_exclusion_talking():
    assert u._text_attention_head_dim(_gemma4()) == 512


def test_gemma2_stays_excluded():
    cfg = _Cfg(model_type = "gemma2", head_dim = 256, num_attention_heads = 8)
    assert u._prefers_flex_for_head_dim(cfg) is False


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_next"])
def test_the_models_this_routing_is_for_are_unaffected(model_type):
    cfg = _Cfg(model_type = model_type, head_dim = 256, num_attention_heads = 8)
    assert u._prefers_flex_for_head_dim(cfg) is True


def test_an_explicit_request_can_still_force_flex_on_gemma4(monkeypatch):
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "1")
    assert u._prefers_flex_for_head_dim(_gemma4()) is True


class _Proxy:
    """A forwarding proxy with no instance dict, like unsloth_zoo's Gemma 4 one."""

    __slots__ = ("_real",)

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


def test_a_proxied_text_config_is_still_recognised_by_name():
    # unsloth_zoo's proxy means get_text_config() is not `config.text_config`.
    u._ATTN_IMPL_MAPPING_SUPPORTED.clear()
    u._ATTN_IMPL_MAPPING_SUPPORTED.append(True)
    try:
        text = _Cfg(model_type = "fake_text", head_dim = 256, num_attention_heads = 8)
        cfg = _Cfg(
            model_type = "fake_vl",
            text_config = text,
            vision_config = _Cfg(model_type = "fake_vision", head_dim = 64, num_attention_heads = 8),
        )
        cfg.get_text_config = lambda: _Proxy(text)
        got = u._flex_attn_impl_for(cfg, "sdpa")
        assert got == {"": "sdpa", "text_config": "flex_attention"}
    finally:
        u._ATTN_IMPL_MAPPING_SUPPORTED.clear()


def test_an_unnameable_text_config_declines_rather_than_flexing_everything():
    u._ATTN_IMPL_MAPPING_SUPPORTED.clear()
    u._ATTN_IMPL_MAPPING_SUPPORTED.append(True)
    try:
        stranger = _Cfg(model_type = "fake_text", head_dim = 256, num_attention_heads = 8)
        cfg = _Cfg(
            model_type = "fake_vl",
            vision_config = _Cfg(model_type = "fake_vision", head_dim = 64, num_attention_heads = 8),
        )
        cfg.get_text_config = lambda: stranger
        assert u._flex_attn_impl_for(cfg, "sdpa") is None
    finally:
        u._ATTN_IMPL_MAPPING_SUPPORTED.clear()
