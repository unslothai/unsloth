# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows does not overcommit: a GGUF load whose private host memory (an unmapped spill past
free VRAM, plus the host share of the KV cache) exceeds min(available RAM, commit headroom) is
refused before llama-server starts, instead of failing part way through the load."""

from __future__ import annotations

import inspect
import sys

import pytest

from core.inference import diffusion_memory
from core.inference.llama_cpp import LlamaCppBackend

_GB = 1024**3
_MIB = 1024
_message = LlamaCppBackend._windows_host_commit_refusal_message

# One 32 GB card with 30 GB free.
_CARD = [(0, 30 * _MIB)]


def _backend(
    model_gb = 70.0,
    kv_gb = 8.0,
    context_length = 262144,
    seen = None,
):
    backend = object.__new__(LlamaCppBackend)
    swa = backend._seen_swa_full = []
    backend._gguf_read_bytes = lambda _path: int(model_gb * _GB)
    backend._lazy_read_host_bytes = lambda *a, **k: 0
    backend.probe_server_capabilities = lambda *a, **k: {}

    def kv(
        n_ctx,
        cache_type = None,
        **kwargs,
    ):
        if seen is not None:
            seen.append((n_ctx, cache_type, kwargs.get("n_parallel")))
        swa.append(kwargs.get("swa_full"))
        return int(kv_gb * _GB) if n_ctx else 0

    backend._estimate_kv_cache_bytes = kv
    backend._context_length = context_length
    return backend


def _cmd(*extra):
    return ["llama-server", "-m", "model.gguf", "-c", "262144", *extra]


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")


class TestMessage:
    def test_fits_under_both_limits(self):
        assert _message(10 * _GB, 64 * _MIB, 40 * _MIB) is None

    def test_commit_headroom_is_the_binding_limit(self):
        msg = _message(40 * _GB, 64 * _MIB, 20 * _MIB)
        assert msg is not None
        assert "commit" in msg and "40 GB" in msg and "18 GB" in msg
        assert "smaller or more quantized GGUF" in msg and "shorter context" in msg

    def test_available_ram_is_the_binding_limit(self):
        msg = _message(40 * _GB, 20 * _MIB, 64 * _MIB)
        assert msg is not None and "system RAM" in msg

    def test_unknown_readings_never_refuse(self):
        assert _message(400 * _GB, None, None) is None

    def test_nothing_private_never_refuses(self):
        assert _message(0, 1, 1) is None

    def test_boundary_keeps_the_headroom(self):
        assert _message(18 * _GB, 64 * _MIB, 20 * _MIB) is None
        assert _message(19 * _GB, 64 * _MIB, 20 * _MIB) is not None


class TestPreLaunch:
    def test_unmapped_spill_past_commit_headroom_is_refused(self, windows):
        # 70 GB GGUF on a 32 GB card copies about 40 GB into private host memory.
        msg = _backend()._windows_host_commit_refusal(
            _cmd("--no-mmap", "--n-cpu-moe", "40"),
            _CARD,
            {},
            avail_mib = 96 * _MIB,
            commit_headroom_mib = 24 * _MIB,
        )
        assert msg is not None and "commit" in msg

    def test_same_load_with_more_commit_goes_ahead(self, windows):
        assert (
            _backend()._windows_host_commit_refusal(
                _cmd("--no-mmap", "--n-cpu-moe", "40"),
                _CARD,
                {},
                avail_mib = 96 * _MIB,
                commit_headroom_mib = 80 * _MIB,
            )
            is None
        )

    def test_fake_commit_headroom_is_read_when_not_passed(self, windows, monkeypatch):
        monkeypatch.setattr(diffusion_memory, "_windows_commit_headroom_mib", lambda: 8 * _MIB)
        msg = _backend()._windows_host_commit_refusal(
            _cmd("--no-mmap"), _CARD, {}, avail_mib = 96 * _MIB
        )
        assert msg is not None and "commit" in msg

    def test_mapped_spill_charges_no_commit(self, windows):
        # Mapped weights are file-backed; the advisory, not this check, speaks for them.
        assert (
            _backend(kv_gb = 0)._windows_host_commit_refusal(
                _cmd(), _CARD, {}, avail_mib = 4 * _MIB, commit_headroom_mib = 4 * _MIB
            )
            is None
        )

    def test_host_kv_counts_even_when_weights_are_mapped(self, windows):
        msg = _backend(kv_gb = 24)._windows_host_commit_refusal(
            _cmd("--no-kv-offload"),
            _CARD,
            {},
            avail_mib = 96 * _MIB,
            commit_headroom_mib = 16 * _MIB,
        )
        assert msg is not None

    def test_layer_split_puts_its_share_of_kv_on_the_host(self, windows):
        # 40 of 70 GB spill: about 57% of a 32 GB cache, 18 GB, sits in host RAM.
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 16 * _MIB)
        assert _backend(kv_gb = 32)._windows_host_commit_refusal(_cmd(), _CARD, {}, **kwargs)
        # Experts on the CPU keep the attention layers, and their KV, on the GPU.
        assert (
            _backend(kv_gb = 32)._windows_host_commit_refusal(
                _cmd("--n-cpu-moe", "40"), _CARD, {}, **kwargs
            )
            is None
        )

    def test_lazy_tables_neither_spill_nor_pull_kv_to_the_host(self, windows):
        # 120 GB file, 100 GB of it lazy-mapped tables: 20 GB resident fits 30 GB free with its KV.
        backend = _backend(model_gb = 120, kv_gb = 32)
        backend._lazy_read_host_bytes = lambda *a, **k: 100 * _GB
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 8 * _MIB)
        assert backend._windows_host_commit_refusal(_cmd("--no-mmap"), _CARD, {}, **kwargs) is None
        # Without the discount the same file spills 90 GB.
        backend._lazy_read_host_bytes = lambda *a, **k: 0
        assert backend._windows_host_commit_refusal(_cmd("--no-mmap"), _CARD, {}, **kwargs)

    @pytest.mark.parametrize(
        "env",
        [
            {"LLAMA_ARG_CPU_MOE": "1"},
            {"LLAMA_ARG_N_CPU_MOE": "40"},
            {"LLAMA_ARG_OVERRIDE_TENSOR": "exps=CPU"},
        ],
    )
    def test_expert_placement_from_the_environment_counts_too(self, windows, env):
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 16 * _MIB)
        assert _backend(kv_gb = 32)._windows_host_commit_refusal(_cmd(), _CARD, env, **kwargs) is None

    def test_kv_settings_come_from_argv_then_the_environment(self, windows):
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 96 * _MIB)
        env = {
            "LLAMA_ARG_CTX_SIZE": "32768",
            "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_CACHE_TYPE_V": "q8_0",
            "LLAMA_ARG_N_PARALLEL": "4",
        }
        seen = []
        base = ["llama-server", "-m", "model.gguf", "--no-kv-offload"]
        _backend(seen = seen)._windows_host_commit_refusal(base, _CARD, env, **kwargs)
        assert seen[-1] == (32768, "q8_0", 4)
        _backend(seen = seen)._windows_host_commit_refusal(
            [*base, "-c", "8192", "-ctk", "f16", "-np", "2"], _CARD, env, **kwargs
        )
        # K from argv (f16), V from the environment (q8_0): each priced at its own type.
        assert seen[-2:] == [(8192, "f16", 2), (8192, "q8_0", 2)]

    def test_k_and_v_are_priced_at_their_own_types(self, windows):
        # A 32 GB f16 cache with q4_0 K costs about 20.5 GB: not refused with 24 GB usable.
        backend = _backend()
        backend._kv_lora_rank = None
        per_axis = {"f16": 32 * _GB, "q4_0": 9 * _GB}
        backend._estimate_kv_cache_bytes = lambda n_ctx, cache_type = None, **k: per_axis[cache_type]
        cmd = ["llama-server", "-m", "model.gguf", "--no-kv-offload", "-c", "8192", "-ctk", "q4_0"]
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 26 * _MIB)
        assert backend._windows_host_commit_refusal(cmd, _CARD, {}, **kwargs) is None
        # Both axes in f16 (32 GB) is refused at the same headroom.
        cmd_f16 = ["llama-server", "-m", "model.gguf", "--no-kv-offload", "-c", "8192"]
        assert backend._windows_host_commit_refusal(cmd_f16, _CARD, {}, **kwargs)
        # MLA caches K only, so V's type does not enter.
        backend._kv_lora_rank = 512
        assert backend._windows_host_commit_refusal(cmd, _CARD, {}, **kwargs) is None

    @pytest.mark.parametrize(
        "extra, env",
        [
            (("-ngl", "1"), {}),
            (("--n-gpu-layers", "0"), {}),
            ((), {"LLAMA_ARG_N_GPU_LAYERS": "1"}),
        ],
    )
    def test_a_hand_set_layer_count_caps_the_vram_credit(self, windows, extra, env):
        # 20 GB fits 30 GB free, but -ngl 1 of 40 layers leaves almost all of it on the CPU.
        backend = _backend(model_gb = 20, kv_gb = 0)
        backend._n_layers = 40
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 12 * _MIB)
        cmd = _cmd("--no-mmap", *extra)
        assert backend._windows_host_commit_refusal(cmd, _CARD, env, **kwargs)
        # -1 (every layer) and no count keep the free-VRAM credit.
        cmd = _cmd("--no-mmap", "-ngl", "-1")
        assert backend._windows_host_commit_refusal(cmd, _CARD, {}, **kwargs) is None
        assert backend._windows_host_commit_refusal(_cmd("--no-mmap"), _CARD, {}, **kwargs) is None

    def test_a_layer_count_keeps_host_kv_beside_experts_on_the_cpu(self, windows):
        # --cpu-moe keeps attention on the GPU only for the layers -ngl puts there.
        backend = _backend(model_gb = 20, kv_gb = 32)
        backend._n_layers = 40
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 24 * _MIB)
        cmd = _cmd("--cpu-moe", "-ngl", "4")
        assert backend._windows_host_commit_refusal(cmd, _CARD, {}, **kwargs)
        assert backend._windows_host_commit_refusal(_cmd("--cpu-moe"), _CARD, {}, **kwargs) is None

    def test_mixed_cache_types_follow_the_key_and_value_widths(self, windows):
        # K 192 wide at f16, V 128 wide at q4_0: 32 GB if both were f16, 9 GB if both q4_0.
        backend = _backend()
        backend._kv_lora_rank = None
        backend._kv_key_length, backend._kv_value_length = 192, 128
        per_axis = {"f16": 32 * _GB, "q4_0": 9 * _GB}
        backend._estimate_kv_cache_bytes = lambda n_ctx, cache_type = None, **k: per_axis[cache_type]
        cmd = ["llama-server", "-m", "model.gguf", "--no-kv-offload", "-c", "8", "-ctv", "q4_0"]
        # 0.6 * 32 + 0.4 * 9 = 22.8 GB: fits 26 GB of headroom, not 24.
        assert (
            backend._windows_host_commit_refusal(
                cmd, _CARD, {}, avail_mib = 96 * _MIB, commit_headroom_mib = 26 * _MIB
            )
            is None
        )
        assert backend._windows_host_commit_refusal(
            cmd, _CARD, {}, avail_mib = 96 * _MIB, commit_headroom_mib = 24 * _MIB
        )

    @pytest.mark.parametrize(
        "extra, env, want",
        [((), {}, False), (("--swa-full",), {}, True), ((), {"LLAMA_ARG_SWA_FULL": "1"}, True)],
    )
    def test_full_swa_reaches_the_kv_estimate(self, windows, extra, env, want):
        kwargs = dict(avail_mib = 96 * _MIB, commit_headroom_mib = 96 * _MIB)
        backend = _backend()
        cmd = _cmd("--no-kv-offload", *extra)
        backend._windows_host_commit_refusal(cmd, _CARD, env, **kwargs)
        assert backend._seen_swa_full[-1] is want

    def test_model_that_fits_the_card_is_never_refused(self, windows):
        assert (
            _backend(model_gb = 20)._windows_host_commit_refusal(
                _cmd("--no-mmap"), _CARD, {}, avail_mib = 1 * _MIB, commit_headroom_mib = 1 * _MIB
            )
            is None
        )

    def test_unread_gpu_probe_and_unified_memory_are_left_alone(self, windows):
        kwargs = dict(avail_mib = 4 * _MIB, commit_headroom_mib = 4 * _MIB)
        backend = _backend()
        assert backend._windows_host_commit_refusal(_cmd("--no-mmap"), [], {}, **kwargs) is None
        assert (
            backend._windows_host_commit_refusal(
                _cmd("--no-mmap"), _CARD, {}, shared_gpu_ids = [0], **kwargs
            )
            is None
        )

    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    def test_other_platforms_are_unchanged(self, monkeypatch, platform):
        monkeypatch.setattr(sys, "platform", platform)
        assert (
            _backend()._windows_host_commit_refusal(
                _cmd("--no-mmap", "--no-kv-offload"),
                _CARD,
                {},
                avail_mib = 1 * _MIB,
                commit_headroom_mib = 1 * _MIB,
            )
            is None
        )

    def test_unreadable_model_never_refuses(self, windows):
        backend = _backend()

        def boom(_path):
            raise OSError("unreadable")

        backend._gguf_read_bytes = boom
        assert (
            backend._windows_host_commit_refusal(
                _cmd("--no-mmap"), _CARD, {}, avail_mib = 1 * _MIB, commit_headroom_mib = 1 * _MIB
            )
            is None
        )


def test_load_model_refuses_before_the_spawn():
    src = inspect.getsource(LlamaCppBackend.load_model)
    check = src.index("self._windows_host_commit_refusal(")
    assert check < src.index("launched_with_mmproj = ")
    assert check < src.index("_last_spawn_cmd = list(cmd)")
    # After the pageable rewrite, so it judges the load mode that actually launches.
    assert src.index("self._page_an_oversized_unmapped_load(") < check
