# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server's host RAM after load: the pinned-load guard, the growth it prices, the
single-slot default on a small card the weights overflow, and the --cache-ram auto probe.

Reported case: an 8 GB card and a 16 GB laptop, a MoE whose weights overflow the card. The
default plan pinned the host-side experts (--load-mode none) and kept 4 slots with the 8192 MiB
prompt cache default; llama-server then grew to 10-11 GB and the machine hung."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as placement  # noqa: E402
import utils.hardware as hardware  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB
FIT_MODE = LlamaCppBackend._FIT_LOAD_MODE


class _Host:
    """Just the host-RAM readers and the fit the guard calls."""

    def __init__(self, avail_mib, total_mib):
        self._avail_mib = avail_mib
        self._total_mib = total_mib

    def _available_system_memory_mib(self):
        return self._avail_mib

    def _host_memory_capacity_mib(self):
        return self._total_mib

    def _amd_apu_wants_unified_memory(self, gpu_indices = None):
        return False

    _fits_without_paging = LlamaCppBackend._fits_without_paging
    _unified_pool_gpu_ids = LlamaCppBackend._unified_pool_gpu_ids
    _fit_derived_load_mode = LlamaCppBackend._fit_derived_load_mode
    _pinned_load_mode_keeps = LlamaCppBackend._pinned_load_mode_keeps
    _host_peak_bytes = LlamaCppBackend._host_peak_bytes
    _FIT_LOAD_MODE = LlamaCppBackend._FIT_LOAD_MODE


def _kwargs(
    model_gib,
    free_mib,
    margin_mib = 1097,
    host_only_gib = 0.0,
):
    return dict(
        model_size = int(model_gib * GIB),
        gpus = [(0, free_mib)] if free_mib else [],
        fit_margin_mib = margin_mib,
        host_only_bytes = int(host_only_gib * GIB),
    )


@pytest.fixture(autouse = True)
def _not_apple(monkeypatch):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)


def test_growth_is_the_configured_cache_cap_plus_the_server():
    server = lc._LLAMA_SERVER_HOST_MIB * MIB
    grow = LlamaCppBackend._llama_server_growth_bytes
    # The cache keeps earlier prompts until its size cap is full: the cap is the price, however
    # small one saved state is.
    assert grow(8192 * MIB, 1 * GIB) == 8192 * MIB + server
    assert grow(1804 * MIB, 4 * GIB) == 1804 * MIB + server
    assert grow(0, 4 * GIB) == server
    # Unbounded (-1) keeps at most n_ctx tokens in all: one full-context state.
    assert grow(None, 1 * GIB) == 1 * GIB + server


def test_the_reported_laptop_keeps_mmap(monkeypatch):
    """13.4 GB load, 7.6 GB free on an 8 GB card, 11.5 GB available of 15.8 GB."""
    monkeypatch.setattr(sys, "platform", "linux")
    host = _Host(11500, 15800)
    kw = _kwargs(13.4, 7600, host_only_gib = 1.0)
    # The fit alone says none: VRAM plus available RAM holds the load.
    assert LlamaCppBackend._fit_derived_load_mode(host, **kw, announce = False) == FIT_MODE
    growth = LlamaCppBackend._llama_server_growth_bytes(8192 * MIB, int(1.4 * GIB))
    assert not host._pinned_load_mode_keeps(kw, growth, announce = False)


def test_a_big_host_keeps_the_pinned_load(monkeypatch):
    """Same model on a 24 GB card and a 64 GB host: nothing changes."""
    monkeypatch.setattr(sys, "platform", "linux")
    host = _Host(58000, 64000)
    kw = _kwargs(13.4, 23000)
    growth = LlamaCppBackend._llama_server_growth_bytes(8192 * MIB, int(1.4 * GIB))
    assert host._pinned_load_mode_keeps(kw, growth, announce = False)


def test_a_big_spill_on_a_big_host_still_pins_under_the_safe_share(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    host = _Host(58000, 64000)
    # 20 GB on the host side + 9 GB growth = 29 GB, under half of 62.5 GiB.
    kw = _kwargs(44.0, 24000, margin_mib = 0)
    growth = LlamaCppBackend._llama_server_growth_bytes(8192 * MIB, 4 * GIB)
    assert host._pinned_load_mode_keeps(kw, growth, announce = False)


@pytest.mark.parametrize("platform, keeps", [("linux", True), ("win32", False)])
def test_windows_keeps_a_smaller_share(platform, keeps, monkeypatch):
    """7 GiB pinned plus 1.5 GiB growth on 20 GiB: 42%, between the two shares."""
    monkeypatch.setattr(sys, "platform", platform)
    host = _Host(18 * 1024, 20 * 1024)
    kw = _kwargs(15.0, 8 * 1024, margin_mib = 0)
    growth = int(0.5 * GIB) + lc._LLAMA_SERVER_HOST_MIB * MIB
    assert host._pinned_load_mode_keeps(kw, growth, announce = False) is keeps


def test_growth_that_no_longer_fits_available_ram_keeps_mmap(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    # Plenty of total RAM, little of it available: the share passes, the fit does not.
    host = _Host(9000, 128 * 1024)
    kw = _kwargs(12.0, 7600)
    assert LlamaCppBackend._fit_derived_load_mode(host, **kw, announce = False) == FIT_MODE
    growth = LlamaCppBackend._llama_server_growth_bytes(8192 * MIB, 4 * GIB)
    assert not host._pinned_load_mode_keeps(kw, growth, announce = False)


def test_a_cpu_only_load_is_not_held_to_the_pinned_share(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    host = _Host(30 * 1024, 32 * 1024)
    kw = _kwargs(20.0, 0)
    growth = lc._LLAMA_SERVER_HOST_MIB * MIB
    assert host._pinned_load_mode_keeps(kw, growth, announce = False)


def test_the_host_peak_estimate_is_the_spill_plus_growth(monkeypatch):
    host = _Host(64 * 1024, 64 * 1024)
    kw = _kwargs(10.0, 4 * 1024, margin_mib = 0, host_only_gib = 1.0)
    # Host-only terms sit on top of the weights: 10 GiB less 4 GiB of VRAM, plus 1 + 2 GiB.
    assert host._host_peak_bytes(kw, 2 * GIB) == 6 * GIB + 1 * GIB + 2 * GIB


@pytest.mark.parametrize(
    "gpus, totals, model_gib, n, from_default, expected",
    [
        ([(0, 7600)], {0: 8188}, 11.0, 4, True, 8188),
        # The weights fit the card: slots stay.
        ([(0, 7600)], {0: 8188}, 4.0, 4, True, None),
        # A card above 12 GB: slots stay.
        ([(0, 15000)], {0: 16384}, 20.0, 4, True, None),
        # Two small cards that together hold the weights: slots stay.
        ([(0, 7600), (1, 7600)], {0: 8188, 1: 8188}, 11.0, 4, True, None),
        # A slot count the user chose, or already one slot.
        ([(0, 7600)], {0: 8188}, 11.0, 4, False, None),
        ([(0, 7600)], {0: 8188}, 11.0, 1, True, None),
        # No discrete card: a shared-memory row reports total 0.
        ([(0, 7600)], {0: 0}, 11.0, 4, True, None),
        ([], {}, 11.0, 4, True, None),
    ],
)
def test_one_slot_only_when_the_weights_overflow_a_small_card(
    gpus, totals, model_gib, n, from_default, expected
):
    got = LlamaCppBackend._small_card_overflow_total_mib(
        gpus,
        totals,
        int(model_gib * GIB),
        n_parallel = n,
        slots_from_default = from_default,
    )
    assert got == expected


def test_apple_silicon_keeps_its_slots(monkeypatch):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
    assert (
        LlamaCppBackend._small_card_overflow_total_mib(
            [(0, 7600)], {0: 8188}, 11 * GIB, n_parallel = 4, slots_from_default = True
        )
        is None
    )


def test_an_auto_cache_is_priced_at_most_the_default():
    assert lc._prompt_cache_host_bytes(
        None, ["--cache-ram", "auto"], {}, supports_cache_ram = True
    ) == (lc._LLAMA_CACHE_RAM_DEFAULT_MIB * MIB)
    assert lc._prompt_cache_host_bytes(
        None, None, {"LLAMA_ARG_CACHE_RAM": "auto"}, supports_cache_ram = True
    ) == (lc._LLAMA_CACHE_RAM_DEFAULT_MIB * MIB)
    assert lc._prompt_cache_host_bytes(None, ["-cram", "-1"], {}, supports_cache_ram = True) is None


@pytest.mark.parametrize(
    "extra_help, auto",
    [
        ("", False),
        (
            "-cram, --cache-ram N                   set the maximum cache size in MiB "
            "(default: 8192, 0 for decision models, -1 - no limit, 0 - disable)\n",
            False,
        ),
        (
            "-cram, --cache-ram N|auto               set the maximum cache size in MiB ...\n"
            "                                        auto sizes it from host RAM available\n"
            "                                        after the model loads (default: 8192)\n",
            True,
        ),
        # Only the value spelling counts: "auto" in a description is not the option.
        (
            "-cram, --cache-ram N                   set the maximum cache size in MiB, "
            "see --load-mode auto (default: 8192)\n",
            False,
        ),
    ],
    ids = ["no_flag", "upstream", "fork_auto", "auto_in_prose_only"],
)
def test_the_probe_reads_cache_ram_auto(tmp_path, monkeypatch, extra_help, auto):
    monkeypatch.setattr(LlamaCppBackend, "_capability_cache", {})
    binary = placement._write_help_binary(
        tmp_path, f"llama-server-cram-{auto}-{bool(extra_help)}", placement._HELP_BASE + extra_help
    )
    caps = LlamaCppBackend.probe_server_capabilities(binary)
    assert caps["help_probe_ok"] is True
    assert caps["supports_cache_ram"] is bool(extra_help)
    assert caps["supports_cache_ram_auto"] is auto


def test_a_failed_probe_advertises_no_cache_ram_auto():
    caps = LlamaCppBackend.probe_server_capabilities("/nonexistent/llama-server")
    assert caps["supports_cache_ram_auto"] is False


def test_another_servers_auto_cache_is_held_back_at_the_default(monkeypatch):
    from types import SimpleNamespace

    other = SimpleNamespace(
        is_active = True, _process = None, _sized_cache_ram_flags = ["--cache-ram", "auto"]
    )
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    monkeypatch.setattr(lc, "_serving_backends_snapshot", lambda: [backend, other])
    assert backend._other_sized_cache_mib() == lc._LLAMA_CACHE_RAM_DEFAULT_MIB


def test_the_slot_clamp_carries_the_lint_marker():
    import inspect
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "n_parallel = 1  # allow-slot-clamp: weights spill to host RAM" in src


def _loaded(n_parallel: int, cut: bool) -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._process = object()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "UD-IQ3_XXS"
    backend._requested_n_ctx = 8192
    backend._requested_spec_mode = "auto"
    backend._requested_n_parallel = n_parallel
    backend._slots_cut_for_host_ram = cut
    return backend


def _slots_intent(n_parallel: int, from_default: bool) -> lc.GgufLoadIntent:
    return lc.GgufLoadIntent(
        model_identifier = "owner/repo",
        hf_variant = "UD-IQ3_XXS",
        n_ctx = 8192,
        speculative_type = "auto",
        n_parallel = n_parallel,
        n_parallel_from_default = from_default,
    )


@pytest.mark.parametrize(
    "cut, from_default, matches",
    [
        # Auto cut the default 4 slots to 1: the same Auto request reuses the server ...
        (True, True, True),
        # ... but 4 slots set in the UI must reload instead of getting the 1-slot server.
        (True, False, False),
        # No cut: an explicit 4 matches a server that runs 4.
        (False, False, True),
    ],
)
def test_a_host_ram_slot_cut_reloads_for_an_explicit_slot_count(cut, from_default, matches):
    backend = _loaded(4, cut)
    assert backend._runtime_matches_intent(_slots_intent(4, from_default), None) is matches


def test_the_load_records_the_host_ram_slot_cut():
    src = inspect.getsource(LlamaCppBackend.load_model)
    # The cut keeps the asked count, and the commit compares the launched count against it.
    assert "_host_ram_slots_asked = n_parallel\n" in src
    assert (
        "self._slots_cut_for_host_ram = (\n"
        "                    _host_ram_slots_asked is not None and n_parallel < _host_ram_slots_asked"
    ) in src
    backend = LlamaCppBackend()
    backend._slots_cut_for_host_ram = True
    backend._reset_effective_parallel_slots()
    assert backend._slots_cut_for_host_ram is False


def test_a_load_on_a_small_card_marks_the_slot_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    backend, gguf = placement._backend(tmp_path, vulkan = False, memory = [(0, 7600, 8188)])
    backend._get_gguf_size_bytes = lambda _path: 11 * GIB
    backend._available_system_memory_mib = lambda: 11500
    backend._total_system_memory_mib = lambda: 15800
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_cache_ram": True,
        "supports_kv_unified": True,
    }
    out = placement._launch(backend, gguf, n_ctx = 8192, n_parallel = 4, n_parallel_from_default = True)
    cmd = out["cmd"]
    assert cmd[cmd.index("--parallel") + 1] == "1"
    assert backend._slots_cut_for_host_ram is True
    assert (
        backend._runtime_matches_intent(
            lc.GgufLoadIntent(
                model_identifier = "test", n_ctx = 8192, n_parallel = 4, n_parallel_from_default = False
            ),
            None,
        )
        is False
    )
