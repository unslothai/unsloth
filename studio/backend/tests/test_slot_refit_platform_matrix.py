# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which platforms the post-reduction context re-fit is allowed to touch.

Its guard -- ``use_fit and n_parallel > 1 and gpus and self._can_estimate_kv()
and effective_ctx > 0`` -- encodes a hardware claim that is easy to lose in a
refactor. ``gpus`` is empty on Metal (no torch.cuda device is enumerated, which
is why the Apple arm exists) and on any CPU-only host; tensor-parallel clears
``use_fit`` first; manual memory mode empties ``gpus``. All are excluded.

These watch the predicate, not the argv, so a cell that starts entering the
block fails here even when its numbers happen not to move.
"""

from __future__ import annotations

import platform as _platform
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

import core.inference.llama_cpp as llama_mod  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from test_llama_cpp_placement import _backend, _launch  # noqa: E402

MIB = 1024 * 1024
NATIVE_CTX = 262144
CARD_MIB = 12 * 1024

DENSE = {
    "_architecture": "qwen3",
    "_vocab_size": 248320,
    "_n_layers": 64,
    "_n_kv_heads": 8,
    "_n_heads": 32,
    "_embedding_length": 5120,
    "_kv_key_length": 128,
    "_kv_value_length": 128,
    "_key_length_mla": None,
    "_context_length": NATIVE_CTX,
}

# (sys.platform, platform.system(), apple_silicon)
OS_CELLS = {
    "linux": ("linux", "Linux", False),
    "wsl": ("linux", "Linux", False),
    "windows": ("win32", "Windows", False),
    "macos_arm": ("darwin", "Darwin", True),
    "macos_intel": ("darwin", "Darwin", False),
}
# (vulkan, enumerates_a_gpu)
VENDOR_CELLS = {
    "nvidia": (False, True),
    "amd": (False, True),
    "vulkan": (True, True),
    "cpu": (False, False),
}

REACHABLE = {
    (os_key, vendor)
    for os_key in ("linux", "wsl", "windows")
    for vendor in ("nvidia", "amd", "vulkan")
}
ALL_CELLS = [(o, v) for o in OS_CELLS for v in VENDOR_CELLS]


class _RefitSpy:
    """Counts re-fit entries: only ``_slots_hold`` passes ``include_requested``."""

    def __init__(self):
        self.calls = 0

    def __enter__(self):
        real, spy = LlamaCppBackend._slots_that_fit_on_gpu, self

        def wrapper(backend_self, *args, **kwargs):
            if kwargs.get("include_requested"):
                spy.calls += 1
            return real(backend_self, *args, **kwargs)

        self._patch = patch.object(LlamaCppBackend, "_slots_that_fit_on_gpu", wrapper)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def _plan(
    tmp_path,
    *,
    os_key,
    vendor,
    weights_mib = 10_200,
    n_parallel = 4,
    vram_mib = CARD_MIB,
    n_ctx = 0,
    tensor_parallel = False,
    gpu_memory_mode = None,
):
    """Drive the real planner under a spoofed host. Returns (plan, refit_entries)."""
    sys_platform, system, apple_silicon = OS_CELLS[os_key]
    vulkan, enumerates_gpu = VENDOR_CELLS[vendor]

    with ExitStack() as stack:
        # Never os.name: it swaps pathlib's flavour and the temp GGUF stops opening.
        stack.enter_context(patch.object(llama_mod.sys, "platform", sys_platform))
        stack.enter_context(patch.object(_platform, "system", lambda: system))
        # The Metal budget selects the Apple arm; 0 elsewhere keeps it inert.
        stack.enter_context(
            patch.object(
                LlamaCppBackend,
                "_apple_metal_memory_budget_bytes",
                staticmethod(lambda: (48 * 1024 * MIB) if apple_silicon else 0),
            )
        )
        stack.enter_context(patch.object(llama_mod, "_metal_device_is_paravirtual", lambda: False))

        # macOS enumerates no torch.cuda device; a CPU-only host has none either.
        cards = [] if (not enumerates_gpu or sys_platform == "darwin") else [vram_mib]
        memory = [(i, mib, mib) for i, mib in enumerate(cards)]
        backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)

        def read(_path):
            for key, value in DENSE.items():
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
        }

        kwargs = {"speculative_type": "off", "n_ctx": n_ctx, "n_parallel": n_parallel}
        if tensor_parallel:
            kwargs["tensor_parallel"] = True
        if gpu_memory_mode is not None:
            kwargs["gpu_memory_mode"] = gpu_memory_mode

        with _RefitSpy() as spy:
            launched = _launch(backend, gguf, **kwargs)
        cmd = launched["cmd"]

        def flag(name, default = None):
            return cmd[cmd.index(name) + 1] if name in cmd else default

        return {
            "ctx": int(flag("-c", 0)),
            "slots": int(flag("--parallel", 1)),
            "fit": flag("--fit", "off"),
            "ngl": flag("-ngl"),
            "threads": flag("--threads"),
            "ceiling": backend._max_context_length,
        }, spy.calls


class TestWhoTheRefitIsAllowedToTouch:
    @pytest.mark.parametrize("os_key,vendor", ALL_CELLS, ids = [f"{o}-{v}" for o, v in ALL_CELLS])
    def test_only_a_gpu_host_enters_the_refit(self, tmp_path, os_key, vendor):
        """Metal and CPU-only hosts must not reach the new block at all."""
        _, entries = _plan(tmp_path, os_key = os_key, vendor = vendor)
        if (os_key, vendor) in REACHABLE:
            assert entries > 0, f"{os_key}/{vendor} should re-fit"
        else:
            assert entries == 0, f"{os_key}/{vendor} must not re-fit"

    @pytest.mark.parametrize("os_key", ["macos_arm", "macos_intel"])
    def test_macos_plans_exactly_as_it_did(self, tmp_path, os_key):
        """No enumerated GPU means the Apple arm owns the plan, untouched."""
        got, entries = _plan(tmp_path, os_key = os_key, vendor = "nvidia")
        assert entries == 0
        assert got["ngl"] is None  # never pinned to a device that does not exist

    def test_tensor_parallel_is_excluded(self, tmp_path):
        """The tensor arm has no --fit valve, so the re-fit must stay out."""
        _, entries = _plan(
            tmp_path,
            os_key = "linux",
            vendor = "nvidia",
            tensor_parallel = True,
            vram_mib = 24 * 1024,
        )
        assert entries == 0

    def test_manual_memory_mode_is_excluded(self, tmp_path):
        """Manual mode is the caller taking the budget over."""
        _, entries = _plan(tmp_path, os_key = "linux", vendor = "nvidia", gpu_memory_mode = "manual")
        assert entries == 0

    def test_a_single_slot_request_is_excluded(self, tmp_path):
        """Nothing to reduce, so nothing to re-fit."""
        _, entries = _plan(tmp_path, os_key = "linux", vendor = "nvidia", n_parallel = 1)
        assert entries == 0


class TestWindowsPlansLikeLinux:
    """The re-fit must not newly arm the Windows full-offload thread cap."""

    def test_the_thread_cap_tracks_the_slot_reduction_not_the_refit(self, tmp_path):
        """--threads 2 rides fully_gpu_offloaded, which the reduction already set."""
        win, win_entries = _plan(tmp_path, os_key = "windows", vendor = "nvidia")
        linux, linux_entries = _plan(tmp_path, os_key = "linux", vendor = "nvidia")
        assert win_entries == linux_entries > 0
        # Same plan either way; only the Windows-only thread pin differs.
        assert (win["ctx"], win["slots"], win["fit"]) == (
            linux["ctx"],
            linux["slots"],
            linux["fit"],
        )
        assert linux["threads"] is None
