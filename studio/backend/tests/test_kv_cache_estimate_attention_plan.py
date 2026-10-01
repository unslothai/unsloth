# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/models/kv-cache-estimate answers for a named attention plan, not for the defaults.

The route took the estimator's signature defaults while the loader resolved the same three
knobs differently, so one model at one cache type put three numbers on screen (#10489).
The arithmetic itself is covered by tests/test_kv_cache_estimation.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from test_kv_cache_estimation import _make_gguf_bytes  # noqa: E402

import routes.models as models_routes  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

N_CTX = 32768

# Equal K/V widths, so the only flash-attention term that can move the total is the f16
# floor on V.
_PLAIN_GQA = {
    "context_length": 131072,
    "block_count": 32,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "embedding_length": 4096,
    "attention.key_length": 128,
    "attention.value_length": 128,
}

# Gemma-class: narrower V on the sliding-window layers, padded to the model-wide maximum
# when flash attention is off, so even an f16 cache moves.
_RAGGED_SWA = {
    "context_length": 131072,
    "block_count": 30,
    "attention.head_count": 16,
    "attention.head_count_kv": 8,
    "embedding_length": 2816,
    "attention.key_length": 512,
    "attention.value_length": 512,
    "attention.key_length_swa": 256,
    "attention.value_length_swa": 256,
    "attention.sliding_window": 1024,
    "attention.sliding_window_pattern": 6,
}


def _write_gguf(
    path: Path,
    fields: dict,
    arch: str = "testarch",
) -> Path:
    kv = {"general.architecture": arch}
    for key, value in fields.items():
        kv[f"{arch}.{key}"] = value
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(_make_gguf_bytes(arch, kv))
    return path


def _call_route(
    monkeypatch,
    *,
    path: Path,
    caps: dict | None = None,
    **overrides,
):
    """All passed explicitly: called in process an omitted one arrives as its ``Query``
    object rather than the None FastAPI would have resolved."""
    monkeypatch.setattr(
        models_routes,
        "_resolve_quant_gguf",
        lambda _repo, _quant, _local: (str(path), 4096),
    )
    monkeypatch.setattr(models_routes, "is_local_path", lambda _p: False, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, *a, **k: dict(caps if caps is not None else {"mtp_token": True})),
    )
    kwargs = dict(
        repo_id = "org/repo",
        quant = "Q4_K_M",
        n_ctx = N_CTX,
        cache_type_kv = None,
        n_parallel = 1,
        speculative_type = None,
        spec_draft_n_max = None,
        spec_draft_cache_type = None,
        ctx_checkpoints = None,
        disable_vision = False,
        n_batch = None,
        n_ubatch = None,
        tensor_parallel = False,
        flash_attn = None,
        kv_unified = None,
        swa_full = None,
        no_mmproj_offload = None,
        request = None,
        current_subject = "test",
    )
    kwargs.update(overrides)
    return asyncio.run(models_routes.get_kv_cache_estimate(**kwargs))


@pytest.fixture
def gqa(tmp_path):
    return _write_gguf(tmp_path / "gqa-Q4_K_M.gguf", _PLAIN_GQA)


@pytest.fixture
def ragged(tmp_path):
    return _write_gguf(tmp_path / "ragged-Q4_K_M.gguf", _RAGGED_SWA)


class TestFlashAttention:
    def test_an_explicit_off_prices_the_padded_cache(self, monkeypatch, ragged):
        on = _call_route(monkeypatch, path = ragged, flash_attn = True)
        off = _call_route(monkeypatch, path = ragged, flash_attn = False)
        assert off["kv_bytes"] > on["kv_bytes"]

    def test_omitting_it_resolves_to_what_the_launch_emits(self, monkeypatch, ragged):
        default = _call_route(monkeypatch, path = ragged)
        assert (
            default["kv_bytes"]
            == _call_route(monkeypatch, path = ragged, flash_attn = True)["kv_bytes"]
        )

    def test_a_build_without_the_flag_prices_the_padded_cache(self, monkeypatch, ragged):
        blind = _call_route(monkeypatch, path = ragged)
        unsupported = _call_route(
            monkeypatch,
            path = ragged,
            caps = {"found": True, "supports_flash_attn": False},
        )
        assert unsupported["kv_bytes"] > blind["kv_bytes"]

    def test_a_quantized_v_cache_is_not_priced_without_it(self, monkeypatch, gqa):
        """An explicit off is not a launch that can happen: llama.cpp forces it on."""
        forced = _call_route(monkeypatch, path = gqa, cache_type_kv = "q8_0", flash_attn = False)
        resolved = _call_route(monkeypatch, path = gqa, cache_type_kv = "q8_0")
        assert forced["kv_bytes"] == resolved["kv_bytes"]

    def test_the_quantized_v_answer_is_the_narrow_one(self, monkeypatch, gqa):
        """2.125 bytes per element, not the 3.0625 the loader used to contradict it with."""
        q8 = _call_route(monkeypatch, path = gqa, cache_type_kv = "q8_0")
        f16 = _call_route(monkeypatch, path = gqa, cache_type_kv = "f16")
        assert q8["kv_bytes"] == pytest.approx(f16["kv_bytes"] * (2.125 / 4.0), rel = 1e-6)


class TestTheOtherTwoLayoutKnobs:
    def test_swa_full_collapses_the_two_cache_sizes(self, monkeypatch, ragged):
        # Net of checkpoints: --swa-full zeroes the checkpoint share, so the totals can move
        # the other way while the attention cache itself still grows.
        compact = _call_route(monkeypatch, path = ragged)
        full = _call_route(monkeypatch, path = ragged, swa_full = True)
        # The share is None, not 0, when nothing is reserved.
        compact_attn = compact["kv_bytes"] - (compact["kv_checkpoint_bytes"] or 0)
        full_attn = full["kv_bytes"] - (full["kv_checkpoint_bytes"] or 0)
        assert full_attn > compact_attn

    def test_swa_full_drops_the_checkpoint_share(self, monkeypatch, ragged):
        """No sliding window left to snapshot, so any reported share is memory nobody
        reserves."""
        compact = _call_route(monkeypatch, path = ragged, ctx_checkpoints = 8)
        full = _call_route(monkeypatch, path = ragged, ctx_checkpoints = 8, swa_full = True)
        assert compact["kv_checkpoint_bytes"]
        assert not full["kv_checkpoint_bytes"]

    def test_the_unified_cache_is_an_input(self, monkeypatch, ragged):
        unified = _call_route(monkeypatch, path = ragged, n_parallel = 4, kv_unified = True)
        split = _call_route(monkeypatch, path = ragged, n_parallel = 4, kv_unified = False)
        assert unified["kv_bytes"] != split["kv_bytes"]

    def test_a_single_slot_resolves_to_the_launch_default(self, monkeypatch, ragged):
        """Unsloth asks for a unified cache only to serve more than one slot."""
        default = _call_route(monkeypatch, path = ragged, n_parallel = 1)
        assert (
            default["kv_bytes"]
            == _call_route(monkeypatch, path = ragged, n_parallel = 1, kv_unified = False)["kv_bytes"]
        )


class TestTheContract:
    def test_the_new_parameters_are_all_optional(self):
        import inspect
        signature = inspect.signature(models_routes.get_kv_cache_estimate)
        for name in ("flash_attn", "kv_unified", "swa_full", "no_mmproj_offload"):
            assert name in signature.parameters, f"{name} is not on the route"
            assert signature.parameters[name].default is not inspect.Parameter.empty

    def test_the_response_shape_did_not_change(self, monkeypatch, gqa):
        """No response_model, so a strict client breaks on a new key."""
        from test_memory_estimate_contract_freeze import _KV_CACHE_ESTIMATE_KEYS
        assert set(_call_route(monkeypatch, path = gqa)) == set(_KV_CACHE_ESTIMATE_KEYS)

    def test_an_omitted_plan_does_not_move_the_answer_for_a_plain_model(self, monkeypatch, gqa):
        answer = _call_route(monkeypatch, path = gqa)
        assert (
            answer["kv_bytes"]
            == _call_route(monkeypatch, path = gqa, flash_attn = True, kv_unified = True, swa_full = False)[
                "kv_bytes"
            ]
        )
