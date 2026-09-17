# SPDX-License-Identifier: AGPL-3.0-or-later
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


# --- explicit opt-outs must survive the force-enable path ------------------------------
# PreTrainedModel sets _supports_flex_attn = False for every model, so getattr() cannot tell
# "never mentioned it" (qwen3_5) from "deliberately off" (T5Gemma2, whose custom masks cannot
# merge under flex). Hence the vars() read with the MRO walk stopping at the generic base.


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
