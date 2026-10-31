# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Only architectures with a KV filter may drop nextn blocks; the rest still allocate KV for them."""

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# structlog stub needs get_logger: freshness_flow.py calls it at import.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
if "structlog" not in sys.modules:
    _structlog_stub = _types.ModuleType("structlog")
    _structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    sys.modules["structlog"] = _structlog_stub

import pytest  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from test_kv_cache_estimation import _backend_from_gguf  # noqa: E402


def _gqa_backend(**overrides):
    """A plain-GQA header: no SSM, no MLA, no SWA, so Path 4 prices it."""
    defaults = {
        "_n_layers": 47,
        "_n_kv_heads": 8,
        "_n_heads": 96,
        "_embedding_length": 4096,
        "_kv_key_length": 128,
        "_kv_value_length": 128,
    }
    defaults.update(overrides)
    b = LlamaCppBackend()
    for k, v in defaults.items():
        setattr(b, k, v)
    return b


def test_glm4_moe_nextn_block_stays_in_target_kv():
    """GLM-4.5 MoE has no KV filter, so its nextn block keeps target KV: count all 47 layers."""
    b = _gqa_backend(_nextn_predict_layers = 1)
    cells = 4096
    per_layer = cells * 8 * (128 + 128) * 2

    assert b._estimate_kv_cache_bytes(4096, "f16") == 47 * per_layer


def test_glm4_moe_target_kv_does_not_move_when_the_head_is_declared():
    """Declaring the MTP head must not shrink the target reserve by a layer."""
    with_nextn = _gqa_backend(_nextn_predict_layers = 1)
    without = _gqa_backend()

    missing = without._estimate_kv_cache_bytes(4096, "f16") - with_nextn._estimate_kv_cache_bytes(
        4096, "f16"
    )
    assert missing == 0, f"target KV dropped by {missing} bytes ({missing / 1024**2:.1f} MiB)"


def test_gemma4_assistant_shaped_header_does_not_collapse_to_one_layer():
    """gemma4-assistant has nextn equal to all layers, so a max(1, ...) floor would price one layer."""
    b = _gqa_backend(_n_layers = 12, _nextn_predict_layers = 12)
    cells = 4096
    per_layer = cells * 8 * (128 + 128) * 2

    result = b._estimate_kv_cache_bytes(4096, "f16")
    assert result != 1 * per_layer, "estimate collapsed to the max(1, ...) floor"
    assert result == 12 * per_layer


def test_qwen35_hybrid_nextn_subtraction_is_correct():
    """Qwen3.5 hybrids filter nextn upstream, so subtracting it is right; the arch gate must keep that."""
    b = LlamaCppBackend()
    for k, v in {
        "_n_layers": 65,
        "_nextn_predict_layers": 1,
        "_n_kv_heads": 4,
        "_n_heads": 24,
        "_embedding_length": 5120,
        "_kv_key_length": 256,
        "_kv_value_length": 256,
        "_full_attention_interval": 4,
        "_ssm_inner_size": 6144,
        "_ssm_state_size": 128,
        "_ssm_group_count": 16,
        "_ssm_conv_kernel": 4,
    }.items():
        setattr(b, k, v)

    per_slot = 48 * ((4 - 1) * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    kv_only = 16 * 4096 * 4 * (256 + 256) * 2
    assert b._estimate_kv_cache_bytes(4096, "f16") == kv_only + per_slot


# arch -> does llama.cpp's TARGET context leave the nextn block out? Unknown archs fail closed.
ARCH_TRUTH_TABLE = [
    ("qwen35", True),
    ("qwen35moe", True),
    ("qwen3next", True),
    ("minimax-01", True),
    ("nemotron_h", True),
    ("nemotron_h_moe", True),
    ("glm-dsa", True),
    ("deepseek32", True),
    ("step35", True),
    ("hy_v3", True),
    ("mimo2", True),
    ("deepseek2", False),
    ("glm4moe", False),
    ("glm4", False),
    ("bailingmoe2", False),
    ("cohere2moe", False),
    ("exaone4", False),
    ("granite-switch", False),
    ("some_future_arch", False),
]

_GQA_FIELDS = {
    "block_count": 47,
    "attention.head_count_kv": 8,
    "attention.head_count": 96,
    "embedding_length": 4096,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "context_length": 131072,
}


@pytest.mark.parametrize("arch,excludes", ARCH_TRUTH_TABLE, ids = [a for a, _ in ARCH_TRUTH_TABLE])
def test_target_kv_nextn_policy_matches_llama_cpp_per_arch(arch, excludes):
    """One layer of 47 is the whole question; get it right per architecture."""
    b = _backend_from_gguf(arch, {**_GQA_FIELDS, "nextn_predict_layers": 1})

    assert b._nextn_predict_layers == 1, "the header did not parse"
    assert b._target_kv_excludes_nextn() is excludes

    per_layer = 4096 * 8 * (128 + 128) * 2
    expected = (46 if excludes else 47) * per_layer
    assert b._estimate_kv_cache_bytes(4096, "f16") == expected


@pytest.mark.parametrize("arch,_excludes", ARCH_TRUTH_TABLE, ids = [a for a, _ in ARCH_TRUTH_TABLE])
def test_no_nextn_key_is_never_reduced(arch, _excludes):
    """Backwards compat: a GGUF with no MTP head is priced exactly as before."""
    b = _backend_from_gguf(arch, dict(_GQA_FIELDS))

    assert not b._nextn_predict_layers
    assert b._target_kv_excludes_nextn() is False
    assert b._estimate_kv_cache_bytes(4096, "f16") == 47 * 4096 * 8 * (128 + 128) * 2


def test_a_hybrid_header_is_evidence_even_for_an_unknown_arch():
    """A hybrid header's ssm dimensions count as evidence for subtraction, even for an arch not named."""
    b = _backend_from_gguf(
        "qwen39_hypothetical",
        {
            "block_count": 65,
            "nextn_predict_layers": 1,
            "attention.head_count_kv": 4,
            "attention.head_count": 24,
            "embedding_length": 5120,
            "attention.key_length": 256,
            "attention.value_length": 256,
            "full_attention_interval": 4,
            "ssm.inner_size": 6144,
            "ssm.state_size": 128,
            "ssm.group_count": 16,
            "ssm.conv_kernel": 4,
            "context_length": 131072,
        },
    )

    assert b._target_kv_excludes_nextn() is True
    per_slot = 48 * ((4 - 1) * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    assert b._estimate_kv_cache_bytes(4096, "f16") == 16 * 4096 * 4 * (256 + 256) * 2 + per_slot


def test_the_glm4_moe_regression_is_gone():
    """Shipped GLM-4.5-Air GGUFs count the nextn block in block_count, so layer 47 must stay priced."""
    with_head = _backend_from_gguf("glm4moe", {**_GQA_FIELDS, "nextn_predict_layers": 1})
    without = _backend_from_gguf("glm4moe", dict(_GQA_FIELDS))

    assert with_head._estimate_kv_cache_bytes(4096, "f16") == without._estimate_kv_cache_bytes(
        4096, "f16"
    )


def test_a_nextn_equal_to_block_count_does_not_collapse():
    """A nextn count equal to block_count must not collapse to one layer; the arch gate excludes it."""
    b = _backend_from_gguf(
        "gemma4-assistant",
        {**_GQA_FIELDS, "block_count": 12, "nextn_predict_layers": 12},
    )

    assert b._target_kv_excludes_nextn() is False
    assert b._estimate_kv_cache_bytes(4096, "f16") == 12 * 4096 * 8 * (128 + 128) * 2


def test_recurrent_state_is_independent_of_the_arch_gate():
    """Recurrent state always subtracts nextn, since its memory is sized on n_layer() for every arch."""
    b = _backend_from_gguf(
        "qwen35",
        {
            "block_count": 65,
            "nextn_predict_layers": 1,
            "attention.head_count_kv": 4,
            "attention.head_count": 24,
            "embedding_length": 5120,
            "attention.key_length": 256,
            "attention.value_length": 256,
            "full_attention_interval": 4,
            "ssm.inner_size": 6144,
            "ssm.state_size": 128,
            "ssm.group_count": 16,
            "ssm.conv_kernel": 4,
            "context_length": 131072,
        },
    )

    expected = 48 * ((4 - 1) * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    assert b._mamba_recurrent_state_bytes() == expected


def test_the_status_field_stays_an_optional_string():
    """An already-installed client parses the new reason without a schema bump."""
    from models.inference import InferenceStatusResponse

    field = InferenceStatusResponse.model_fields["spec_fallback_reason"]
    assert field.default is None
    parsed = InferenceStatusResponse.model_validate({"spec_fallback_reason": "mtp_partial_offload"})
    assert parsed.spec_fallback_reason == "mtp_partial_offload"
