# SPDX-License-Identifier: Apache-2.0
"""#4037: bnb loads of hybrid Mamba models must fail fast with a clear message.

The quantizer crash (AttributeError: JambaAttention has no attribute `feed_forward`)
is opaque; the helpers below turn it into an actionable error before shards are
downloaded (preflight) or with context after the fact (translation).
"""

from types import SimpleNamespace

import pytest

from unsloth.models.vision import (
    _hybrid_mamba_bnb_preflight,
    _raise_quantizer_error,
)


def test_preflight_blocks_jamba_4bit():
    cfg = SimpleNamespace(model_type="jamba", architectures=["JambaForCausalLM"])
    with pytest.raises(ValueError, match="not supported yet"):
        _hybrid_mamba_bnb_preflight(cfg, load_in_4bit=True, load_in_8bit=False)


def test_preflight_blocks_zamba2_8bit():
    cfg = SimpleNamespace(model_type="zamba2", architectures=["Zamba2ForCausalLM"])
    with pytest.raises(ValueError, match="not supported yet"):
        _hybrid_mamba_bnb_preflight(cfg, load_in_4bit=False, load_in_8bit=True)


def test_preflight_allows_nemotron_h_4bit():
    # Nemotron-H is now handled in unsloth/models/loader.py and ships official
    # bnb-4bit repos — the preflight must NOT block it. #4037 review
    cfg = SimpleNamespace(model_type="nemotron_h", architectures=["NemotronHForCausalLM"])
    _hybrid_mamba_bnb_preflight(cfg, load_in_4bit=True, load_in_8bit=False)  # must not raise


def test_preflight_allows_llama_4bit():
    cfg = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
    _hybrid_mamba_bnb_preflight(cfg, load_in_4bit=True, load_in_8bit=False)  # must not raise


def test_preflight_allows_jamba_16bit():
    """16-bit loading stays allowed for hybrid models — only bnb is blocked."""
    cfg = SimpleNamespace(model_type="jamba", architectures=["JambaForCausalLM"])
    _hybrid_mamba_bnb_preflight(cfg, load_in_4bit=False, load_in_8bit=False)  # must not raise


def test_translation_adds_context_to_quantizer_error():
    raw = AttributeError("JambaAttention has no attribute `feed_forward`")
    with pytest.raises(ValueError, match="bitsandbytes quantizer"):
        _raise_quantizer_error(raw, model_name="some/Jamba2-Mini")


def test_translation_passes_other_attribute_errors_through():
    """Non-quantizer AttributeErrors keep their original type (no masking of real bugs)."""
    raw = AttributeError("something else entirely")
    with pytest.raises(AttributeError):
        _raise_quantizer_error(raw, model_name="x")
