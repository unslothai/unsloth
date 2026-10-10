# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discrete-GPU placement must not charge --ctx-checkpoints against VRAM.

llama-server copies each snapshot device-to-host into a std::vector, so the N per
slot live in host RAM. Charging them against the card shrank the context of an
explicit request (Gemma 3 27B on 24 GiB: 87552 blank, 8192 at 32) for memory the
card never holds (#8988).

The target is Gemma-3 shaped, the SWA case the old fit charged.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

MIB = 1024 * 1024
NATIVE_CTX = 131072
CARD_MIB = 12 * 1024

SWA = {
    "_architecture": "gemma3",
    "_vocab_size": 262144,
    "_n_layers": 62,
    "_n_kv_heads": 4,
    "_n_heads": 16,
    "_embedding_length": 3840,
    "_kv_key_length": 256,
    "_kv_value_length": 256,
    "_key_length_mla": None,
    "_context_length": NATIVE_CTX,
    "_sliding_window": 1024,
}


def _plan(
    tmp_path,
    *,
    weights_mib,
    n_parallel,
    ctx_checkpoints,
    vram_mib = CARD_MIB,
    cache_type_kv = "q8_0",
    ctx_checkpoints_flag = "--ctx-checkpoints",
):
    """Return the generated plan plus what its own context really costs."""
    memory = [(0, vram_mib, vram_mib)]
    backend, gguf = _backend(tmp_path, vulkan = False, memory = memory)

    def read(_path):
        for key, value in SWA.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: weights_mib * MIB
    del backend._can_estimate_kv  # the real one, now that the dims are set
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_ctx_checkpoints": ctx_checkpoints_flag is not None,
        "ctx_checkpoints_flag": ctx_checkpoints_flag,
    }
    launched = _launch(
        backend,
        gguf,
        speculative_type = "off",
        n_ctx = 0,
        n_parallel = n_parallel,
        cache_type_kv = cache_type_kv,
        ctx_checkpoints = ctx_checkpoints,
    )
    cmd = launched["cmd"]

    def flag(name, default = None):
        return cmd[cmd.index(name) + 1] if name in cmd else default

    ctx = int(flag("-c", 0))
    slots = int(flag("--parallel", 1))
    _cp = int(ctx_checkpoints or 0)
    kv_kwargs = dict(
        n_parallel = slots,
        swa_full = False,
        kv_unified = True,
        n_ubatch = None,
        flash_attn = True,
    )
    return {
        "ctx": ctx,
        "slots": slots,
        "fit": flag("--fit", "off"),
        "checkpoints": flag("--ctx-checkpoints"),
        # What the launch reserves beyond the plain cache.
        "reserve_bytes": (
            backend._estimate_kv_cache_bytes(ctx, cache_type_kv, ctx_checkpoints = _cp, **kv_kwargs)
            - backend._estimate_kv_cache_bytes(ctx, cache_type_kv, ctx_checkpoints = 0, **kv_kwargs)
        ),
    }


def _prime(backend):
    """Set every field the KV estimator reads on a bare backend."""
    for key, value in SWA.items():
        setattr(backend, key, value)
    backend._kv_key_length_swa = None
    backend._kv_value_length_swa = None
    backend._sliding_window_pattern = None
    backend._kv_lora_rank = None
    backend._nextn_predict_layers = 0
    backend._ssm_inner_size = None
    backend._ssm_state_size = None
    backend._ssm_group_count = None
    backend._ssm_conv_kernel = None
    backend._full_attention_interval = None
    backend._shared_kv_layers = None


class TestTheReserveIsHostMemory:
    def test_the_reserve_is_not_free_on_this_fixture(self):
        """Guards the equality tests below from passing on a zero-cost shape."""
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        _prime(backend)
        kv = dict(n_parallel = 4, swa_full = False, kv_unified = True, flash_attn = True)
        assert backend._estimate_kv_cache_bytes(
            8192, "q8_0", ctx_checkpoints = 32, **kv
        ) > backend._estimate_kv_cache_bytes(8192, "q8_0", ctx_checkpoints = 0, **kv)


class TestThePlanIgnoresTheReserve:
    @pytest.mark.parametrize("checkpoints", [4, 16, 32])
    def test_a_checkpointed_launch_plans_like_an_uncheckpointed_one(self, tmp_path, checkpoints):
        free = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = 0)
        asked = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = checkpoints)
        assert asked["reserve_bytes"] > 0
        assert asked["checkpoints"] == str(checkpoints)
        assert (asked["ctx"], asked["slots"], asked["fit"]) == (
            free["ctx"],
            free["slots"],
            free["fit"],
        )

    def test_a_single_slot_keeps_its_context(self, tmp_path):
        free = _plan(tmp_path, weights_mib = 6_800, n_parallel = 1, ctx_checkpoints = 0)
        asked = _plan(tmp_path, weights_mib = 6_800, n_parallel = 1, ctx_checkpoints = 32)
        assert asked["fit"] == "off"
        assert asked["reserve_bytes"] > 0
        assert asked["ctx"] == free["ctx"] > 0

    def test_a_build_without_the_flag_is_not_charged(self, tmp_path):
        skipped = _plan(
            tmp_path,
            weights_mib = 9_200,
            n_parallel = 4,
            ctx_checkpoints = 32,
            ctx_checkpoints_flag = None,
        )
        none_asked = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = 0)
        assert skipped["checkpoints"] is None
        assert (skipped["ctx"], skipped["slots"]) == (none_asked["ctx"], none_asked["slots"])

    def test_no_checkpoints_is_unchanged(self, tmp_path):
        default = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = None)
        zero = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = 0)
        assert default["ctx"] == zero["ctx"]
        assert default["slots"] == zero["slots"]
        assert zero["reserve_bytes"] == 0
