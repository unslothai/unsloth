# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An MLA model rejects different K and V cache types.

llama-context.cpp gained this check, and it sits ABOVE the V-quantization
check, so it decides first:

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4)
            && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("model does not support different K (%s) and V (%s)
                         cache types");
        return nullptr;
    }

is_mla() covers DeepSeek V2/V3/R1, Kimi K2 and GLM-4.7/5.x, which Studio
already recognises through kv_lora_rank.

The flash-attention-off retry resets a quantized V cache to f16 and
deliberately leaves K quantized, because a quantized K needs no FA and
resetting it enlarges the cache. On an MLA model that produces K=q8_0 V=f16,
which is a hard abort for a DIFFERENT reason than the one being avoided: the
retry that exists to recover from an FA crash fails on the K/V mismatch
instead of recovering.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

QUANT_KV = [
    "llama-server",
    "-m",
    "ds.gguf",
    "--flash-attn",
    "on",
    "--cache-type-k",
    "q8_0",
    "--cache-type-v",
    "q8_0",
]


def _types_of(
    cmd,
    k_flag = "--cache-type-k",
    v_flag = "--cache-type-v",
):
    return cmd[cmd.index(k_flag) + 1], cmd[cmd.index(v_flag) + 1]


class TestTheRetryKeepsKAndVEqualOnMla:
    def test_both_axes_come_down_together(self):
        out = LlamaCppBackend._with_flash_attn_off(list(QUANT_KV), mla = True)
        k, v = _types_of(out)
        assert k == v == "f16", (k, v)

    def test_the_inline_equals_spelling_is_handled(self):
        cmd = ["llama-server", "--flash-attn", "on", "--cache-type-k=q8_0", "--cache-type-v=q8_0"]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert "--cache-type-k=f16" in out and "--cache-type-v=f16" in out

    def test_the_draft_pair_is_handled_too(self):
        cmd = [*QUANT_KV, "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0"]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert _types_of(out) == ("f16", "f16")
        assert _types_of(out, "--spec-draft-type-k", "--spec-draft-type-v") == ("f16", "f16")

    def test_an_unquantized_kv_is_left_alone(self):
        """Nothing to reset, so nothing moves and no cache grows."""
        cmd = [
            "llama-server",
            "--flash-attn",
            "on",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
        ]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert _types_of(out) == ("f16", "f16")

    def test_a_quantized_k_alone_does_not_drag_anything(self):
        """No V reset happened, so the MLA branch must not fire either: K and V
        were already equal in llama.cpp's eyes before we touched anything."""
        cmd = ["llama-server", "--flash-attn", "on", "--cache-type-k", "q8_0"]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert out[out.index("--cache-type-k") + 1] == "q8_0"


class TestNonMlaBehaviourIsUnchanged:
    """The size argument still holds everywhere else: resetting K needlessly
    enlarges it and can OOM a memory-constrained config."""

    def test_k_stays_quantized(self):
        out = LlamaCppBackend._with_flash_attn_off(list(QUANT_KV), mla = False)
        assert _types_of(out) == ("q8_0", "f16")

    def test_the_default_is_the_non_mla_behaviour(self):
        """mla defaults False, so an un-updated caller keeps today's answer."""
        out = LlamaCppBackend._with_flash_attn_off(list(QUANT_KV))
        assert _types_of(out) == ("q8_0", "f16")


class TestTheSignalStudioAlreadyHas:
    def test_kv_lora_rank_is_the_mla_marker(self):
        """Both call sites pass `self._kv_lora_rank is not None`, which is the
        same signal _can_estimate_kv uses to take its MLA branch."""
        import inspect

        src = inspect.getsource(LlamaCppBackend.load_model)
        assert src.count("mla = self._kv_lora_rank is not None") >= 2
