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

# Weights that actually REACH the re-fit on a GPU host, which every cell here depends
# on: the block sits behind `if not _uf_slots:`, so it runs only when the slot search
# found a count that fits. On this 12 GiB card that is a band -- under ~8,000 MiB the
# load holds all four slots and never reduces, over ~9,400 no count fits and it
# offloads -- and the band moves with the fit floor, because the search is priced at
# it. 10,200 was inside it at a 4096 floor and is past the top at 8192, which took
# every cell in this file to zero entries: the nine REACHABLE ones failed, and the
# four exclusion cells kept passing while asserting nothing, since "excluded" and
# "never got there" both read as entries == 0. test_the_scenario_still_reaches_the
# _refit_at_all pins both edges so that cannot recur silently.
REFIT_BAND_WEIGHTS_MIB = 8_800

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
    weights_mib = REFIT_BAND_WEIGHTS_MIB,
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
    @pytest.mark.parametrize("weights_mib,enters", [(7_600, False), (8_800, True), (9_600, False)])
    def test_the_scenario_still_reaches_the_refit_at_all(self, tmp_path, weights_mib, enters):
        """Brackets the band ``REFIT_BAND_WEIGHTS_MIB`` has to stay inside.

        Every ``entries == 0`` assertion in this file is satisfied by a scenario that
        never reaches the re-fit, so without this the whole matrix can go quietly
        vacuous: the exclusion cells keep passing and only the nine REACHABLE ones
        report, which is what a moving fit floor did to it. The two outer rows are the
        two ways out of the band -- 7,600 holds all four slots so nothing reduces,
        9,600 holds no count so it offloads -- and both must stay OUT while the middle
        one stays IN.
        """
        _, entries = _plan(tmp_path, os_key = "linux", vendor = "nvidia", weights_mib = weights_mib)
        assert (entries > 0) is enters

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
        # The control: the same weights on an enumerating host DO re-fit, so the zero
        # above is macOS being excluded rather than the scenario never arriving.
        _, on_gpu = _plan(tmp_path, os_key = "linux", vendor = "nvidia")
        assert on_gpu > 0

    def test_tensor_parallel_is_excluded(self, tmp_path):
        """The tensor arm has no --fit valve, so the re-fit must stay out."""
        kwargs = dict(os_key = "linux", vendor = "nvidia", vram_mib = 24 * 1024)
        _, entries = _plan(tmp_path, tensor_parallel = True, **kwargs)
        assert entries == 0
        # Same card without the tensor flag must reach the re-fit, or the exclusion is
        # untested. The 24 GiB card has its own band (20,000-21,000 MiB here), so this
        # weight is scaled to the card rather than shared with the 12 GiB default.
        _, layer_split = _plan(tmp_path, weights_mib = 20_500, **kwargs)
        assert layer_split > 0

    def test_manual_memory_mode_is_excluded(self, tmp_path):
        """Manual mode is the caller taking the budget over."""
        _, entries = _plan(tmp_path, os_key = "linux", vendor = "nvidia", gpu_memory_mode = "manual")
        assert entries == 0
        _, auto = _plan(tmp_path, os_key = "linux", vendor = "nvidia")
        assert auto > 0, "the Auto control stopped re-fitting; this proves nothing"

    def test_a_single_slot_request_is_excluded(self, tmp_path):
        """Nothing to reduce, so nothing to re-fit."""
        _, entries = _plan(tmp_path, os_key = "linux", vendor = "nvidia", n_parallel = 1)
        assert entries == 0
        _, many = _plan(tmp_path, os_key = "linux", vendor = "nvidia", n_parallel = 4)
        assert many > 0, "the multi-slot control stopped re-fitting; this proves nothing"


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
