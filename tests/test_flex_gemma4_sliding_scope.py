# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gemma 4 keeps SDPA, because its sliding layers already have something better than flex.

Transformers has no per-LAYER attention implementation: every one of Gemma 4's 30 decoder
layers reads the same `config._attn_implementation` (modeling_gemma4.py, the
`ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, ...)` call). 25 of
those layers are sliding (window 1024, head_dim 256) and 5 are full attention
(global_head_dim 512), so routing the decoder to flex routes the sliding layers too.

unsloth_zoo's gemma4_flash_sliding installs a flash-attn-2 windowed router under the "sdpa"
key. Measured at the Gemma 4 sliding shape, fwd+bwd, T=8192: FA2 with window_size=(w-1, 0)
2.285 ms against flex's 3.283 ms, break-even 1 step against flex's 78. Switching to flex does
not just lose that, it disables the router outright, because those layers never look up "sdpa"
again -- and the banded SDPA fallback and the UNSLOTH_GEMMA4_FLASH_SLIDING knob go with it.

These pin the exclusion, and that it does not leak onto the models the routing is for.
"""

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
    # Shape of the real models/gemma-4-26B-A4B-it/config.json: head_dim 256 on the 25 sliding
    # layers, global_head_dim 512 on the 5 full-attention ones.
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
    # The exclusion has to match on the text sub-config's model_type as well as the top-level
    # one, because a text-only load presents gemma4_text at the top.
    text_only = _Cfg(
        model_type = "gemma4_text",
        head_dim = 256,
        global_head_dim = 512,
        num_attention_heads = 16,
    )
    assert u._prefers_flex_for_head_dim(text_only) is False


def test_the_head_dim_is_still_large_so_this_really_is_the_exclusion_talking():
    # Guard against the test passing for the wrong reason: if the head-dim probe stopped seeing
    # 512 the assertion above would hold with no exclusion at all.
    assert u._text_attention_head_dim(_gemma4()) == 512


def test_gemma2_stays_excluded():
    cfg = _Cfg(model_type = "gemma2", head_dim = 256, num_attention_heads = 8)
    assert u._prefers_flex_for_head_dim(cfg) is False


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_next"])
def test_the_models_this_routing_is_for_are_unaffected(model_type):
    cfg = _Cfg(model_type = model_type, head_dim = 256, num_attention_heads = 8)
    assert u._prefers_flex_for_head_dim(cfg) is True


def test_an_explicit_request_can_still_force_flex_on_gemma4(monkeypatch):
    # The exclusion is a default, not a ban: someone with a long run who wants the global
    # layers on flex must still be able to ask.
    monkeypatch.setenv(u._FLEX_LARGE_HEAD_DIM_ENV_VAR, "1")
    assert u._prefers_flex_for_head_dim(_gemma4()) is True


# ---------------------------------------------------------------------------------------------
# The vision tower must never be dragged along, even behind a proxied text config
# ---------------------------------------------------------------------------------------------


class _Proxy:
    """A forwarding proxy with no instance dict, like unsloth_zoo's Gemma 4 one."""

    __slots__ = ("_real",)

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


def test_a_proxied_text_config_is_still_recognised_by_name():
    # unsloth_zoo wraps Gemma 4's text config to hide num_kv_shared_layers when it is 0, so
    # get_text_config() returns something that is NOT `config.text_config`. A plain `is` test
    # fails to name the field, and the old fall-through then applied flex as a plain string to
    # the whole model -- vision tower included, the exact regression _flex_attn_impl_for exists
    # to prevent.
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
        # A text config that is not reachable from any *_config field at all.
        cfg.get_text_config = lambda: stranger
        assert u._flex_attn_impl_for(cfg, "sdpa") is None
    finally:
        u._ATTN_IMPL_MAPPING_SUPPORTED.clear()
