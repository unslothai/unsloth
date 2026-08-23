# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Platform matrix for the fit-driven ``--load-mode`` pick.

Walks [Linux, Windows, WSL, macOS] x [NVIDIA, AMD discrete, AMD APU, Vulkan iGPU,
CPU only] x [fits in VRAM, fits in VRAM plus RAM, does not fit], then the whole
policy chain the pick feeds into, then the fallback paths that have to take it back
out again. The point is the negative half: on every host where the fit abstains, the
argv has to be exactly what it was before this existed.
"""

from __future__ import annotations

import itertools

import pytest

import utils.hardware as hardware
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_server_args import (
    apply_load_mode_policy,
    apply_model_memory_policy,
)

GIB = 1024**3
# What a proven fit picks; the constant is deliberately the only switch.
FIT_MODE = LlamaCppBackend._FIT_LOAD_MODE
MIB = 1024 * 1024

# (label, gpu rows, vulkan?, unified-memory APU?, shared ordinals)
HARDWARE = {
    "nvidia_discrete": ([(0, 24 * 1024)], False, False, set()),
    "nvidia_multi": ([(0, 12 * 1024), (1, 12 * 1024)], False, False, set()),
    "amd_discrete": ([(0, 24 * 1024)], False, False, set()),
    "amd_apu": ([(0, 24 * 1024)], False, True, set()),
    "vulkan_igpu": ([(0, 24 * 1024)], True, False, {0}),
    "cpu_only": ([], False, False, set()),
}

PLATFORMS = ["linux", "windows", "wsl", "macos"]

# The cartesian product the matrix walks, materialised once.
_OS_GPU = list(itertools.product(PLATFORMS, HARDWARE))


class _Stub:
    """Only what the predicate touches: host RAM and the APU question."""

    def __init__(
        self,
        avail_mib,
        is_apu = False,
    ):
        self._avail_mib = avail_mib
        self._is_apu = is_apu

    def _available_system_memory_mib(self):
        return self._avail_mib

    def _amd_apu_wants_unified_memory(self, gpu_indices = None):
        return self._is_apu

    _fits_without_paging = LlamaCppBackend._fits_without_paging
    # The mode a proven fit picks, read off the backend so flipping the
    # constant moves the whole matrix with it rather than failing 100 cases.
    _FIT_LOAD_MODE = LlamaCppBackend._FIT_LOAD_MODE


def _mode(platform, hw, footprint, avail_mib, monkeypatch, **kwargs):
    rows, vulkan, apu, shared = HARDWARE[hw]
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: platform == "macos")
    stub = _Stub(avail_mib, is_apu = apu)
    return LlamaCppBackend._fit_derived_load_mode(
        stub,
        model_size = footprint,
        gpus = rows,
        shared_gpu_ids = shared,
        is_vulkan_backend = vulkan,
        avail_mib = avail_mib,
        **kwargs,
    )


# ---------------------------------------------------------------- the matrix


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_a_load_that_fits_in_vram_takes_none(platform, hw, monkeypatch):
    """8 GiB into a 24 GiB card, with host RAM deliberately too small to help."""
    rows, _vulkan, apu, _shared = HARDWARE[hw]
    got = _mode(platform, hw, 8 * GIB, 4 * 1024, monkeypatch)
    if platform == "macos":
        # Metal keeps mmap: buffer_from_host_ptr makes the mapping zero copy there.
        assert got is None
    elif not rows or apu or hw == "vulkan_igpu":
        # No usable dedicated VRAM to fit into, and 4 GiB of RAM cannot hold it.
        assert got is None
    else:
        assert got == FIT_MODE


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_a_load_that_fits_in_vram_plus_ram_takes_none(platform, hw, monkeypatch):
    """32 GiB against 24 GiB of VRAM and 64 GiB of RAM."""
    rows, _vulkan, apu, _shared = HARDWARE[hw]
    got = _mode(platform, hw, 32 * GIB, 64 * 1024, monkeypatch)
    if platform == "macos":
        assert got is None
    else:
        # Unified memory is priced against RAM alone, which still holds 32 GiB here.
        assert got == FIT_MODE, (platform, hw, apu, rows)


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_a_load_that_fits_nowhere_keeps_auto(platform, hw, monkeypatch):
    """400 GiB against 24 GiB of VRAM and 32 GiB of RAM: mmap is the only way."""
    assert _mode(platform, hw, 400 * GIB, 32 * 1024, monkeypatch) is None


@pytest.mark.parametrize("platform", PLATFORMS)
def test_unified_memory_is_never_counted_twice(platform, monkeypatch):
    """40 GiB, an APU reporting 32 GiB of "VRAM", and 32 GiB of RAM.

    Counted on both sides that is 64 GiB and the model "fits". It is one 32 GiB
    pool, and it does not.
    """
    assert _mode(platform, "amd_apu", 40 * GIB, 32 * 1024, monkeypatch) is None
    assert _mode(platform, "vulkan_igpu", 40 * GIB, 32 * 1024, monkeypatch) is None


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_an_unsized_model_keeps_auto(platform, hw, monkeypatch):
    for unsized in (None, 0):
        assert _mode(platform, hw, unsized, 64 * 1024, monkeypatch) is None


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_an_unsized_kv_keeps_auto(platform, hw, monkeypatch):
    assert _mode(platform, hw, 8 * GIB, 64 * 1024, monkeypatch, kv_sized = False) is None


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_an_unsized_drafter_keeps_auto(platform, hw, monkeypatch):
    assert _mode(platform, hw, 8 * GIB, 64 * 1024, monkeypatch, mtp_unsized = True) is None


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_unreadable_host_ram_keeps_auto_when_vram_alone_will_not_do(platform, hw, monkeypatch):
    assert _mode(platform, hw, 32 * GIB, None, monkeypatch) is None


def test_every_footprint_term_is_charged(monkeypatch):
    """Each term alone can push a load over the edge, so none may be dropped."""
    rows_free = 8 * GIB
    base = dict(
        platform = "linux",
        hw = "nvidia_discrete",
        avail_mib = 0,
        monkeypatch = monkeypatch,
    )
    # 8 GiB of weights into 24 GiB, with each extra term sized to blow past it.
    for term in (
        "mmproj_pinned_bytes",
        "kv_cache_bytes",
        "mtp_bytes",
        "compute_buffer_flat",
        "compute_buffer_ctx",
        "soft_overhead",
    ):
        assert _mode(footprint = rows_free, **base, **{term: 0}) == FIT_MODE
        assert _mode(footprint = rows_free, **base, **{term: 20 * GIB}) is None


# ------------------------------------------------------- the policy chain


@pytest.fixture
def toggles(monkeypatch):
    """Drive the Model Memory settings the two policies read lazily."""

    def _set(keep_resident, no_ram_reserve):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(
            mm, "get_model_memory_settings", lambda: (keep_resident, no_ram_reserve)
        )
        monkeypatch.setattr(mm, "get_keep_resident", lambda: keep_resident)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_ram_reserve)
        monkeypatch.setattr(mm, "should_mlock", lambda: keep_resident)

    return _set


def _chain(extras, *, user_mode, fit_mode, supports, host_resident):
    """What load_model does, in the same order."""
    managed, rest = apply_model_memory_policy(
        extras,
        supports_load_mode = supports,
        weights_in_host_memory = host_resident,
    )
    lm_managed, rest = apply_load_mode_policy(
        rest,
        supports_load_mode = supports,
        weights_in_host_memory = host_resident,
        requested_load_mode = user_mode or fit_mode,
    )
    return list(managed) + list(lm_managed) + list(rest)


_CHAIN_AXES = list(
    itertools.product(
        [(False, False), (True, False), (False, True), (True, True)],  # toggles
        [None, "none", "mmap", "mlock", "dio", "auto"],  # user pick
        [None, "none"],  # fit pick
        [True, False],  # supports --load-mode
        [True, False],  # weights in host memory
    )
)


@pytest.mark.parametrize("axes", _CHAIN_AXES)
def test_the_chain_never_emits_an_unknown_or_duplicate_mode(axes, toggles):
    (keep, no_reserve), user, fit, supports, host = axes
    toggles(keep, no_reserve)
    argv = _chain([], user_mode = user, fit_mode = fit, supports = supports, host_resident = host)
    # At most one managed mode selector reaches the child.
    assert argv.count("--load-mode") <= 1
    if "--load-mode" in argv:
        value = argv[argv.index("--load-mode") + 1]
        assert value in {"none", "mmap", "mlock", "mmap+mlock", "dio"}
        # A build without the flag must never be handed it.
        assert supports
    if not supports:
        assert "-lm" not in argv


@pytest.mark.parametrize("axes", _CHAIN_AXES)
def test_the_fit_changes_nothing_a_user_pick_did_not_already_decide(axes, toggles):
    """With a user pick present, the fit is invisible: same argv either way."""
    (keep, no_reserve), user, _fit, supports, host = axes
    if user is None:
        pytest.skip("no user pick to defer to")
    toggles(keep, no_reserve)
    with_fit = _chain([], user_mode = user, fit_mode = "none", supports = supports, host_resident = host)
    without = _chain([], user_mode = user, fit_mode = None, supports = supports, host_resident = host)
    assert with_fit == without


def _chain_before_this_change(extras, *, user_mode, supports, host_resident):
    """The chain exactly as it was: the per-model pick, and nothing else.

    Kept as its own literal copy rather than calling ``_chain`` with no fit, so a
    future edit to the live chain cannot quietly redefine what "before" means.
    """
    managed, rest = apply_model_memory_policy(
        extras,
        supports_load_mode = supports,
        weights_in_host_memory = host_resident,
    )
    lm_managed, rest = apply_load_mode_policy(
        rest,
        supports_load_mode = supports,
        weights_in_host_memory = host_resident,
        requested_load_mode = user_mode,
    )
    return list(managed) + list(lm_managed) + list(rest)


@pytest.mark.parametrize("axes", _CHAIN_AXES)
@pytest.mark.parametrize("extras", [[], ["--mlock"], ["--no-mmap"], ["-ngl", "10"]])
def test_an_abstaining_fit_reproduces_the_old_argv_exactly(axes, extras, toggles):
    """The upgrade-safety property, against a literal copy of the old chain.

    Every host where the fit cannot prove a fit -- unreadable RAM, an unsized
    model, Apple Silicon, a load too big for the machine -- has to come out of
    this byte-identical to a Studio that never had the feature.
    """
    (keep, no_reserve), user, _fit, supports, host = axes
    toggles(keep, no_reserve)
    assert _chain(
        list(extras), user_mode = user, fit_mode = None, supports = supports, host_resident = host
    ) == _chain_before_this_change(
        list(extras), user_mode = user, supports = supports, host_resident = host
    )


@pytest.mark.parametrize("toggle_pair", [(True, False), (False, True), (True, True)])
def test_the_model_memory_settings_still_win_over_the_fit(toggle_pair, toggles):
    keep, no_reserve = toggle_pair
    toggles(keep, no_reserve)
    argv = _chain([], user_mode = None, fit_mode = "none", supports = True, host_resident = True)
    if no_reserve:
        # "Don't reserve system RAM" vetoes none outright.
        assert "none" not in argv
    if keep and not no_reserve:
        # "Keep model in GPU memory" owns the mode.
        assert argv[:2] == ["--load-mode", "mmap+mlock"]
        assert "none" not in argv


def test_the_fit_applies_when_both_toggles_are_off(toggles):
    toggles(False, False)
    assert _chain([], user_mode = None, fit_mode = "none", supports = True, host_resident = True) == [
        "--load-mode",
        "none",
    ]


def test_an_old_binary_gets_the_pre_enum_spelling(toggles):
    toggles(False, False)
    assert _chain([], user_mode = None, fit_mode = "none", supports = False, host_resident = True) == [
        "--no-mmap"
    ]


def test_a_hand_typed_flag_still_wins_by_last_arg(toggles):
    toggles(False, False)
    argv = _chain(
        ["--load-mode", "mmap"],
        user_mode = None,
        fit_mode = "none",
        supports = True,
        host_resident = True,
    )
    # Managed block first, the user's copy after it, so llama.cpp's last-wins
    # parsing lands on the user's.
    assert argv == ["--load-mode", "none", "--load-mode", "mmap"]
    assert argv[-1] == "mmap"


# ------------------------------------------- the paths that must take it back


def test_the_cpu_fallback_drops_the_fits_load_mode(monkeypatch):
    """A CPU replay runs on no GPU, so the VRAM half of the fit is void and the
    whole model has to come out of host RAM. mmap is what makes that survivable."""
    from unittest import mock

    from core.inference import llama_cpp as lc

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._fit_load_mode_flags = ["--load-mode", "none"]
    replay = [
        "llama-server",
        "-m",
        "model.gguf",
        "--load-mode",
        "none",
        "--gpu-layers",
        "0",
        "--device",
        "none",
    ]
    with (
        mock.patch.object(lc.LlamaCppBackend, "_is_vulkan_backend", return_value = True),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_replay", return_value = list(replay)),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_binary", return_value = "cpu-server"),
        mock.patch.object(
            lc.LlamaCppBackend,
            "_llama_server_env_for_binary",
            return_value = {lc._loader_path_var(): "/staged"},
        ),
    ):
        out, _reason = backend._prepare_cpu_fallback_launch("llama-server", replay, {}, {})
    assert "--load-mode" not in out
    assert "none" not in out[: out.index("--gpu-layers")]
    # The placement flags the replay itself adds are untouched.
    assert out[-2:] == ["--device", "none"]


def test_the_cpu_fallback_keeps_a_load_mode_the_user_asked_for(monkeypatch):
    """Only Unsloth's own tokens are recorded, so a user's pick survives."""
    from unittest import mock

    from core.inference import llama_cpp as lc

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._fit_load_mode_flags = []  # user pick: nothing recorded
    replay = ["llama-server", "-m", "model.gguf", "--load-mode", "none"]
    with (
        mock.patch.object(lc.LlamaCppBackend, "_is_vulkan_backend", return_value = True),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_replay", return_value = list(replay)),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_binary", return_value = "cpu-server"),
        mock.patch.object(
            lc.LlamaCppBackend,
            "_llama_server_env_for_binary",
            return_value = {lc._loader_path_var(): "/staged"},
        ),
    ):
        out, _reason = backend._prepare_cpu_fallback_launch("llama-server", replay, {}, {})
    assert out[-2:] == ["--load-mode", "none"]


def test_only_the_recorded_subsequence_is_removed():
    """A user's own --load-mode after ours must survive the strip."""
    from core.inference.llama_cpp import _without_subsequence

    argv = ["-m", "m.gguf", "--load-mode", "none", "--load-mode", "mmap"]
    assert _without_subsequence(argv, ["--load-mode", "none"]) == [
        "-m",
        "m.gguf",
        "--load-mode",
        "mmap",
    ]


def test_the_fit_on_retry_drops_the_fits_load_mode():
    """The retry exists because the fit did NOT hold, so the conclusion it drew
    from that fit cannot ride along. Checked at the source, like the other
    ordering invariants in this launch path, because the retry only runs behind a
    real startup crash."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    retry = src[src.index("retrying once with --fit on so it can offload") :]
    retry = retry[: retry.index("_did_fit_retry = True")]
    assert "_fit_load_mode_flags" in retry
    assert "_without_subsequence" in retry
