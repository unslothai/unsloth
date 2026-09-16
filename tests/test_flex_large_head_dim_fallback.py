"""The large-head-dim flex opt-in must never move a VLM vision tower.

On Transformers < 4.57 there is no per-sub-config attn_implementation mapping, so a plain
"flex_attention" string would apply to the vision tower as well as the decoder. That is a
regression rather than a fallback: the tower has a small head dim so SDPA already reaches a
fused kernel, and it attends over a different length every call, which would make flex
recompile per shape. These pin that the unofferable case degrades to sdpa instead.
"""
import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


def _text_only():
    return _Cfg(model_type = "fake", head_dim = 256, num_attention_heads = 8)


def _multimodal():
    text = _Cfg(model_type = "fake", head_dim = 256, num_attention_heads = 8)
    return _Cfg(model_type = "fake_vl", text_config = text,
                vision_config = _Cfg(model_type = "fake_vision", head_dim = 64,
                                     num_attention_heads = 8))


@pytest.fixture(autouse = True)
def _reset_probe_cache():
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
