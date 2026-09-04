# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which architectures may drop the embedded MTP blocks from target KV.

GGUF ``block_count`` includes the trailing NextN/MTP blocks, and
``llama_hparams::n_layer()`` subtracts them (llama-hparams.cpp:297) -- but the
target KV cache walks ``n_layer_all`` (llama-kv-cache.cpp:100) and drops blocks
only through an optional per-architecture filter (:169). So "carries a
nextn_predict_layers key" and "its MTP blocks are outside the target context"
are different questions, and only the second one may reduce the estimate.

Filters, ggml-org/llama.cpp @ adb55e5:
  llama-model.cpp:2289   hybrids -- qwen3next / qwen35 / qwen35moe / minimax-01,
                         and nemotron_h via is_recr
  llama-model.cpp:2129   glm-dsa / deepseek32
  llama-model.cpp:2356   step35 / hy_v3 / mimo2
Everything else (deepseek2, glm4, glm4moe, bailingmoe2, cohere2moe, exaone4)
gets ``filter == nullptr``, so its trailing MTP block DOES get target KV and
subtracting it would under-reserve.

Recurrent state is the exception that always subtracts: llama_memory_recurrent
sizes on ``n_layer()`` directly (llama-memory-recurrent.cpp:29).
"""

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Same dependency stubs the neighbouring estimation tests install, but the
# structlog stand-in carries get_logger: a bare module here is what the rest of
# the suite finds under setdefault, and utils/prebuilt/freshness_flow.py calls
# structlog.get_logger at import time.
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
    """GLM-4.5-Air shape: block_count 47 = 46 trunk + 1 nextn.

    conversion/glm.py:116 writes block_count INCLUDING the nextn block and
    :154 writes nextn_predict_layers, so a shipped GLM-4.5/4.6 MoE GGUF carries
    both. src/models/glm4-moe.cpp:23 reduces n_layer(), but the target context
    for LLM_ARCH_GLM4_MOE is a plain llama_kv_cache with filter == nullptr, so
    llama-kv-cache.cpp:100 still walks all 47 blocks and allocates KV for the
    nextn block. The estimate must therefore cover 47 layers, not 46.
    """
    b = _gqa_backend(_nextn_predict_layers = 1)
    cells = 4096  # already 256-aligned, so cells == n_ctx at one unified slot
    per_layer = cells * 8 * (128 + 128) * 2

    assert b._estimate_kv_cache_bytes(4096, "f16") == 47 * per_layer


def test_glm4_moe_target_kv_does_not_move_when_the_head_is_declared():
    """Declaring the MTP head must not shrink the target reserve by a layer."""
    with_nextn = _gqa_backend(_nextn_predict_layers = 1)
    without = _gqa_backend()

    missing = without._estimate_kv_cache_bytes(4096, "f16") - with_nextn._estimate_kv_cache_bytes(
        4096, "f16"
    )
    # One layer of 47 is ~2.1% of the reserve, and it is llama.cpp's to allocate.
    assert missing == 0, f"target KV dropped by {missing} bytes ({missing / 1024**2:.1f} MiB)"


def test_gemma4_assistant_shaped_header_does_not_collapse_to_one_layer():
    """src/models/gemma4-assistant.cpp:15 asserts n_layer_nextn == n_layer_all.

    block_count - nextn is 0 there, so an unconditional subtraction behind a
    max(1, ...) floor would price a 12-layer model as one layer. The arch gate
    keeps it out of the subtraction entirely. Reachable through the
    separate-drafter call at _mtp_draft_kv_bytes, which sizes a drafter GGUF's
    own KV with this function, and by loading the assistant model directly.
    """
    b = _gqa_backend(_n_layers = 12, _nextn_predict_layers = 12)
    cells = 4096
    per_layer = cells * 8 * (128 + 128) * 2

    result = b._estimate_kv_cache_bytes(4096, "f16")
    assert result != 1 * per_layer, "estimate collapsed to the max(1, ...) floor"
    assert result == 12 * per_layer


def test_qwen35_hybrid_nextn_subtraction_is_correct():
    """The control: the PR's own target case, which upstream DOES filter.

    llama-model.cpp:2289 filters both the attention and recurrent halves to
    il < n_layer() for QWEN35/QWEN35MOE/QWEN3NEXT, and
    llama-memory-recurrent.cpp:29 sizes on n_layer() too, so subtracting is right
    here. This is what the cases above must not be allowed to break: the arch
    gate has to keep the hybrid saving while dropping the rest.
    """
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

    # 64 trunk blocks -> ceil(64/4) = 16 attention layers, 48 recurrent.
    per_slot = 48 * ((4 - 1) * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    kv_only = 16 * 4096 * 4 * (256 + 256) * 2
    assert b._estimate_kv_cache_bytes(4096, "f16") == kv_only + per_slot


# ─────────── per-architecture truth table (real GGUF headers) ───────────

# arch -> does llama.cpp's TARGET context leave the nextn block out?
ARCH_TRUTH_TABLE = [
    # Hybrid attention + recurrent, llama-model.cpp:2289
    ("qwen35", True),
    ("qwen35moe", True),
    ("qwen3next", True),
    ("minimax-01", True),
    ("nemotron_h", True),
    ("nemotron_h_moe", True),
    # MLA / DSA trunk with a dense MTP head, llama-model.cpp:2129
    ("glm-dsa", True),
    ("deepseek32", True),
    # Plain attention trunk with an explicit nextn filter, llama-model.cpp:2356
    ("step35", True),
    ("hy_v3", True),
    ("mimo2", True),
    # filter == nullptr: the trailing MTP block still gets target KV
    ("deepseek2", False),
    ("glm4moe", False),
    ("glm4", False),
    ("bailingmoe2", False),
    ("cohere2moe", False),
    ("exaone4", False),
    ("granite-switch", False),  # nextn=1 leaks for a ROUTER layer that needs KV
    # An arch this Unsloth has never heard of must fail closed.
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
    """Forwards compat: a future hybrid Mamba arch must still get the subtraction.

    Every hybrid llama.cpp takes a nextn key from is filtered at
    llama-model.cpp:2289 and its recurrent half is sized on n_layer() regardless
    (llama-memory-recurrent.cpp:29), so the ssm dims answer without the name.
    """
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
    # 64 trunk blocks -> 16 attention layers + 48 recurrent.
    per_slot = 48 * ((4 - 1) * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    assert b._estimate_kv_cache_bytes(4096, "f16") == 16 * 4096 * 4 * (256 + 256) * 2 + per_slot


def test_the_glm4_moe_regression_is_gone():
    """The concrete case: a shipped GLM-4.5-Air GGUF keeps its 47th layer.

    conversion/glm.py:116 writes block_count INCLUDING the nextn block and :154
    writes the key, so this is what a real file looks like.
    """
    with_head = _backend_from_gguf("glm4moe", {**_GQA_FIELDS, "nextn_predict_layers": 1})
    without = _backend_from_gguf("glm4moe", dict(_GQA_FIELDS))

    assert with_head._estimate_kv_cache_bytes(4096, "f16") == without._estimate_kv_cache_bytes(
        4096, "f16"
    )


def test_a_nextn_equal_to_block_count_does_not_collapse():
    """gemma4-assistant asserts n_layer_nextn == n_layer_all (gemma4-assistant.cpp:15).

    block_count - nextn is 0 there, and a max(1, ...) floor would price a 12-layer
    model as one layer. The arch gate keeps it out of the subtraction entirely.
    """
    b = _backend_from_gguf(
        "gemma4-assistant",
        {**_GQA_FIELDS, "block_count": 12, "nextn_predict_layers": 12},
    )

    assert b._target_kv_excludes_nextn() is False
    assert b._estimate_kv_cache_bytes(4096, "f16") == 12 * 4096 * 8 * (128 + 128) * 2


# ───────────────────── old / unusual llama.cpp builds ─────────────────────


def test_recurrent_state_is_independent_of_the_arch_gate():
    """llama_memory_recurrent sizes on n_layer() for EVERY arch.

    llama-memory-recurrent.cpp:29 takes hparams.n_layer(), which subtracts nextn
    unconditionally, so _mamba_recurrent_state_bytes must keep subtracting even
    where the attention half does not. Guards against a fix that over-corrects.
    """
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

    # 48 recurrent blocks out of 64 trunk, not 49 out of 65.
    expected = 48 * ((4 - 1) * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    assert b._mamba_recurrent_state_bytes() == expected


def test_the_status_field_stays_an_optional_string():
    """An already-installed client parses the new reason without a schema bump."""
    from models.inference import InferenceStatusResponse

    field = InferenceStatusResponse.model_fields["spec_fallback_reason"]
    assert field.default is None
    # Free-form string, so "mtp_partial_offload" needs no enum change anywhere.
    parsed = InferenceStatusResponse.model_validate({"spec_fallback_reason": "mtp_partial_offload"})
    assert parsed.spec_fallback_reason == "mtp_partial_offload"
