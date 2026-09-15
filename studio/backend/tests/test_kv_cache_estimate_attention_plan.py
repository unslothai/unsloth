# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/models/kv-cache-estimate answers for a named attention plan, not for the defaults.

The route took the estimator's signature defaults for the three knobs that decide the cache
layout: flash attention on, no ``--swa-full``, a unified cache. It had no way to be told
otherwise, so it could not answer for the load the user was about to start, and the loader
resolved the same knobs differently. One model, one cache type, three numbers on screen:
the memory panel's, the loader's own log line, and the context warning's ceiling (#10489,
2.37 GiB against 9.4 GB at q8_0 / 262,144).

So the plan is now an input, resolved the way the launch resolves it when the caller says
nothing, and each figure in one response is priced against the same plan -- including the
planner's aggregate, which is handed the plan in the vocabulary it already understands (the
extra arguments a load would carry).

The estimator is the same function the loader calls; the arithmetic itself is covered by
tests/test_kv_cache_estimation.py. What is asserted here is that the route can be told, that
it resolves rather than defaults, and that it cannot be told something the launch would
overrule.
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

# The real loggers package FIRST. The GGUF builder below installs a process-wide `loggers`
# stub that is a module rather than a package, and routes.models reaches
# loggers.media_progress through it, so the import order decides whether this file can be
# collected on its own rather than only beside a sibling that got there first.
try:
    import loggers.media_progress  # noqa: F401,E402
except Exception:  # pragma: no cover - a stub already won, as in a shared shard
    pass

# Installs the process-wide loggers/structlog/httpx stubs and the GGUF builder, as the
# sibling route suite does.
from test_kv_cache_estimation import _make_gguf_bytes  # noqa: E402

import routes.models as models_routes  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

N_CTX = 32768

# A plain GQA header: equal K/V widths, so the only flash-attention term that can move the
# total is the f16 floor on the V axis, which is the one #9697 measured.
_PLAIN_GQA = {
    "context_length": 131072,
    "block_count": 32,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "embedding_length": 4096,
    "attention.key_length": 128,
    "attention.value_length": 128,
}

# Gemma-class: the sliding-window layers carry a narrower V than the full-attention ones,
# which is the other flash-attention term (with it off, llama.cpp pads every layer's V to
# the model-wide maximum, so an f16 cache moves too).
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
    """Drive the real handler with the quant already resolved to *path*.

    Every parameter is passed explicitly, including the new ones: called in process the
    omitted ones arrive as their ``Query`` objects rather than as the None FastAPI would
    have resolved, and this route's own tests are the first consumer of that.
    """
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
        """With flash attention off llama.cpp cannot use a ragged V cache and pads every
        layer to the model-wide maximum, so even an f16 cache costs more."""
        on = _call_route(monkeypatch, path = ragged, flash_attn = True)
        off = _call_route(monkeypatch, path = ragged, flash_attn = False)
        assert off["kv_bytes"] > on["kv_bytes"]

    def test_omitting_it_resolves_to_what_the_launch_emits(self, monkeypatch, ragged):
        """The managed default is --flash-attn on, so an omitted plan must match the
        explicit on, not the conservative off."""
        default = _call_route(monkeypatch, path = ragged)
        assert (
            default["kv_bytes"]
            == _call_route(monkeypatch, path = ragged, flash_attn = True)["kv_bytes"]
        )

    def test_a_build_without_the_flag_prices_the_padded_cache(self, monkeypatch, ragged):
        """Nothing is emitted on such a build, so the launch runs without it and the
        estimate has to say so. This is what the route could not express before."""
        blind = _call_route(monkeypatch, path = ragged)
        unsupported = _call_route(
            monkeypatch,
            path = ragged,
            caps = {"found": True, "supports_flash_attn": False},
        )
        assert unsupported["kv_bytes"] > blind["kv_bytes"]

    def test_a_quantized_v_cache_is_not_priced_without_it(self, monkeypatch, gqa):
        """llama.cpp enables flash attention itself for a quantized V cache, so an
        explicit off is not a launch that can happen and must not be priced as one."""
        forced = _call_route(monkeypatch, path = gqa, cache_type_kv = "q8_0", flash_attn = False)
        resolved = _call_route(monkeypatch, path = gqa, cache_type_kv = "q8_0")
        assert forced["kv_bytes"] == resolved["kv_bytes"]

    def test_the_quantized_v_answer_is_the_narrow_one(self, monkeypatch, gqa):
        """And it is the smaller of the two prices, which is the panel figure the loader
        used to contradict: 2.125 bytes per element, not 3.0625."""
        q8 = _call_route(monkeypatch, path = gqa, cache_type_kv = "q8_0")
        f16 = _call_route(monkeypatch, path = gqa, cache_type_kv = "f16")
        assert q8["kv_bytes"] == pytest.approx(f16["kv_bytes"] * (2.125 / 4.0), rel = 1e-6)


class TestTheOtherTwoLayoutKnobs:
    def test_swa_full_collapses_the_two_cache_sizes(self, monkeypatch, ragged):
        compact = _call_route(monkeypatch, path = ragged)
        full = _call_route(monkeypatch, path = ragged, swa_full = True)
        assert full["kv_bytes"] > compact["kv_bytes"]

    def test_swa_full_drops_the_checkpoint_share(self, monkeypatch, ragged):
        """--swa-full has no sliding window left to snapshot, so --ctx-checkpoints
        allocates nothing. Reporting a host share there is memory nobody reserves."""
        compact = _call_route(monkeypatch, path = ragged, ctx_checkpoints = 8)
        full = _call_route(monkeypatch, path = ragged, ctx_checkpoints = 8, swa_full = True)
        assert compact["kv_checkpoint_bytes"]
        assert not full["kv_checkpoint_bytes"]

    def test_the_unified_cache_is_an_input(self, monkeypatch, ragged):
        """Per slot when unified, one window otherwise: on an SWA model served by more
        than one slot the two layouts are different amounts of memory."""
        unified = _call_route(monkeypatch, path = ragged, n_parallel = 4, kv_unified = True)
        split = _call_route(monkeypatch, path = ragged, n_parallel = 4, kv_unified = False)
        assert unified["kv_bytes"] != split["kv_bytes"]

    def test_a_single_slot_resolves_to_the_launch_default(self, monkeypatch, ragged):
        """Unsloth asks for a unified cache only to serve more than one slot, so a
        one-slot load is priced as the launch runs it."""
        default = _call_route(monkeypatch, path = ragged, n_parallel = 1)
        assert (
            default["kv_bytes"]
            == _call_route(monkeypatch, path = ragged, n_parallel = 1, kv_unified = False)["kv_bytes"]
        )


class TestTheContract:
    def test_the_new_parameters_are_all_optional(self):
        """An old caller sends none of them, so every one must have a default. The compat
        suite asserts the same thing for the parameters that shipped before."""
        import inspect

        signature = inspect.signature(models_routes.get_kv_cache_estimate)
        for name in ("flash_attn", "kv_unified", "swa_full", "no_mmproj_offload"):
            assert name in signature.parameters, f"{name} is not on the route"
            assert signature.parameters[name].default is not inspect.Parameter.empty

    def test_the_response_shape_did_not_change(self, monkeypatch, gqa):
        """The route has no response_model, so a strict client breaks on a new key. The
        plan is an input, not an output."""
        from test_memory_estimate_contract_freeze import _KV_CACHE_ESTIMATE_KEYS
        assert set(_call_route(monkeypatch, path = gqa)) == set(_KV_CACHE_ESTIMATE_KEYS)

    def test_an_omitted_plan_does_not_move_the_answer_for_a_plain_model(self, monkeypatch, gqa):
        """The regression guard for every caller that exists today: on a model with no
        ragged V and no sliding window, resolving the plan is arithmetically identical to
        the defaults the route used to take."""
        answer = _call_route(monkeypatch, path = gqa)
        assert (
            answer["kv_bytes"]
            == _call_route(monkeypatch, path = gqa, flash_attn = True, kv_unified = True, swa_full = False)[
                "kv_bytes"
            ]
        )
