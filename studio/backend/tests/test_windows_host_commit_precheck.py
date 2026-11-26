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
):
    backend = object.__new__(LlamaCppBackend)
    backend._gguf_read_bytes = lambda _path: int(model_gb * _GB)
    backend._lazy_read_host_bytes = lambda *a, **k: 0
    backend.probe_server_capabilities = lambda *a, **k: {}
    backend._estimate_kv_cache_bytes = lambda n_ctx, *a, **k: int(kv_gb * _GB) if n_ctx else 0
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
