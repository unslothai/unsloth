# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The slot/context fit predicate has to charge the --ctx-checkpoints reserve.

``--ctx-checkpoints N`` allocates N SWA/recurrent snapshots PER SLOT.
``_slots_that_fit_on_gpu`` priced its candidates without it: survivable while the
only consumer was the slot count, but the post-reduction re-fit uses the same
predicate to pick the launched ``-c``, so it spent bytes already promised.

The target is Gemma-3 shaped, since the reserve is charged only on SWA layers.
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


class TestThePredicateChargesTheReserve:
    """Straight at the helper, the way the include_requested case is tested."""

    @staticmethod
    def _fit(ctx_checkpoints):
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        _prime(backend)
        return backend._slots_that_fit_on_gpu(
            8,
            8192,
            [(0, CARD_MIB)],
            {0: CARD_MIB},
            6_000 * MIB,
            "q8_0",
            LlamaCppBackend._GPU_PIN_VRAM_FRACTION,
            0,
            1,
            n_ubatch = 512,
            ctx_checkpoints = ctx_checkpoints,
            include_requested = True,
        )

    def test_charging_the_reserve_costs_slots(self):
        """More memory per slot can only buy the same count or fewer."""
        free = self._fit(0)
        charged = self._fit(32)
        assert free[2] > charged[2], (free, charged)

    def test_the_reserve_is_not_free_on_this_fixture(self):
        """Guards the two tests above from passing on a zero-cost shape."""
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        _prime(backend)
        kv = dict(n_parallel = 4, swa_full = False, kv_unified = True, flash_attn = True)
        assert backend._estimate_kv_cache_bytes(
            8192, "q8_0", ctx_checkpoints = 32, **kv
        ) > backend._estimate_kv_cache_bytes(8192, "q8_0", ctx_checkpoints = 0, **kv)


class TestTheRefitDoesNotSpendTheReserve:
    """End to end: the context the re-fit publishes has to leave room for it."""

    @pytest.mark.parametrize("checkpoints", [4, 16, 32])
    def test_a_checkpointed_launch_gets_less_context_than_an_uncheckpointed_one(
        self, tmp_path, checkpoints
    ):
        """Unpriced, the two stay identical however large --ctx-checkpoints gets,
        while the child allocates it anyway.

        The claim is that the reserve is CHARGED, not that context specifically is
        what pays. There are three ways to pay, and which one applies depends on how
        big the reserve is relative to the budget: give up context at the same slot
        count, give up a slot, or give up residency and offload. At 16 and 32
        checkpoints this fixture already takes the third -- `--fit on` at the offload
        fallback -- and `charged["ctx"] < free["ctx"]` only held there by arithmetic
        coincidence, because the fallback happens to be shorter than the resident
        plan's context. Naming the three keeps a real regression (nothing was
        charged: same slots, same context, same residency) distinguishable from the
        planner picking a different axis, which a raised fit floor can do on its own.
        """
        free = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = 0)
        charged = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = checkpoints)
        assert charged["reserve_bytes"] > 0
        assert charged["checkpoints"] == str(checkpoints)
        if charged["fit"] != free["fit"]:
            assert charged["fit"] == "on", (free, charged)
        elif charged["slots"] != free["slots"]:
            assert charged["slots"] < free["slots"], (free, charged)
        else:
            assert charged["ctx"] < free["ctx"], (free, charged)

    def test_a_plan_with_room_for_it_still_stays_on_gpu(self, tmp_path):
        """The reserve costs context, not the GPU pin, while there is room."""
        got = _plan(tmp_path, weights_mib = 6_800, n_parallel = 8, ctx_checkpoints = 16)
        assert got["fit"] == "off"
        assert got["ctx"] > 0
        assert got["reserve_bytes"] > 0

    def test_a_build_without_the_flag_is_not_charged(self, tmp_path):
        """The argv builder drops the request, so the child allocates nothing."""
        supported = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = 32)
        skipped = _plan(
            tmp_path,
            weights_mib = 9_200,
            n_parallel = 4,
            ctx_checkpoints = 32,
            ctx_checkpoints_flag = None,
        )
        none_asked = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = 0)
        assert skipped["checkpoints"] is None  # not emitted
        assert (skipped["ctx"], skipped["slots"]) == (none_asked["ctx"], none_asked["slots"])
        assert skipped["ctx"] > supported["ctx"]

    def test_no_checkpoints_is_unchanged(self, tmp_path):
        """The default (0) has to plan exactly as it did before."""
        default = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = None)
        zero = _plan(tmp_path, weights_mib = 9_200, n_parallel = 4, ctx_checkpoints = 0)
        assert default["ctx"] == zero["ctx"]
        assert default["slots"] == zero["slots"]
        assert zero["reserve_bytes"] == 0
