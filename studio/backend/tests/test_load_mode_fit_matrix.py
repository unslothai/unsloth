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


# ------------------------------------------------- the CPU-pinned drafter


class _DraftStub:
    """Only the two readers _cpu_resident_draft_bytes consults."""

    def __init__(self, weights, kv):
        self._weights = weights
        self._kv = kv

    def _get_gguf_size_bytes(self, path):
        if self._weights is None:
            raise OSError(path)
        return self._weights

    def _mtp_draft_kv_bytes(self, n_ctx, **kwargs):
        return self._kv

    _cpu_resident_draft_bytes = LlamaCppBackend._cpu_resident_draft_bytes


def test_no_cpu_pinned_drafter_charges_nothing():
    stub = _DraftStub(3 * GIB, 512 * MIB)
    assert stub._cpu_resident_draft_bytes(8192, drafter_path = None) == 0


def test_a_cpu_pinned_drafter_is_charged_weights_plus_kv():
    """``-ngld 0`` takes the drafter off the GPU. It does not delete it: those
    bytes are in host RAM, and ``none`` allocates them anonymously there."""
    stub = _DraftStub(3 * GIB, 512 * MIB)
    assert stub._cpu_resident_draft_bytes(8192, drafter_path = "d.gguf") == 3 * GIB + 512 * MIB


@pytest.mark.parametrize("weights,kv", [(None, 512 * MIB), (0, 512 * MIB), (3 * GIB, None)])
def test_an_unpriceable_cpu_drafter_abstains(weights, kv):
    """A drafter that is there but cannot be sized is exactly the case that must
    not be silently charged as zero."""
    stub = _DraftStub(weights, kv)
    assert stub._cpu_resident_draft_bytes(8192, drafter_path = "d.gguf") is None


def test_the_cpu_drafter_flips_a_fit_that_only_looked_like_one(monkeypatch):
    """20 GiB of target onto an 8 GiB card with 16 GiB of RAM: the 12 GiB spill
    clears the 2 GiB headroom and the fit reads as real. Add the 2.5 GiB drafter
    ``-ngld 0`` leaves resident in host RAM and it does not."""
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    stub = _Stub(16 * 1024)

    def _fit(mtp_bytes):
        return LlamaCppBackend._fit_derived_load_mode(
            stub,
            model_size = 20 * GIB,
            mtp_bytes = mtp_bytes,
            gpus = [(0, 8 * 1024)],
            avail_mib = 16 * 1024,
        )

    assert _fit(0) == FIT_MODE
    assert _fit(int(2.5 * GIB)) is None


def test_the_launch_charges_the_cpu_pinned_drafter_to_the_fit():
    """The budget nulls the drafter path before its weights are sized, so the
    fit has to keep its own copy. Checked at the source, like the other launch
    ordering invariants here: the call site sits inside load_model's fit try."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    # Captured BEFORE the VRAM budget drops it.
    assert "_cpu_draft_path=_mtp_draft_for_budgetif_draft_on_cpuelseNone" in compact
    assert compact.index("_cpu_draft_path=_mtp_draft_for_budget") < compact.index(
        "if_draft_on_cpu:_mtp_draft_for_budget=None"
    )
    # ... and charged to the footprint, abstaining when it cannot be priced.
    assert "mtp_bytes=_mtp_bytes(effective_ctx)+(_cpu_draft_fit_bytesor0)" in compact
    assert "or_cpu_draft_fit_bytesisNone" in compact


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


def test_the_arch_crash_retry_voids_the_fit_the_weights_only_floor_still_allows():
    """The premise behind the strip below, priced on real numbers.

    A 40 GB card and a 4 GB one, 20 GB of RAM. The fit was proved against the card
    the launch PINNED; the arch-crash retry moves to the survivor, and there the
    same footprint no longer fits. The retry's own guard is a weights-only floor by
    design, so it passes and cannot re-establish the proof.
    """
    rows = [(0, 40_000), (1, 4_000)]
    footprint = 30 * GIB  # weights + KV + scratch
    weights = 20 * GIB
    stub = _Stub(20_000)

    # The retry really does move off the fitted card.
    assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0, 1]) == [1]

    assert stub._fits_without_paging(footprint, rows, gpu_indices = [0]) is True
    assert stub._fits_without_paging(footprint, rows, gpu_indices = [1]) is False
    # ...while the weights-only refusal the retry runs stays silent, so nothing on
    # that path would take the fit's "none" back out on its own.
    assert LlamaCppBackend._host_offload_shortfall_message(weights - 4_000 * MIB, 20_000) is None


def test_the_arch_crash_retry_drops_the_fits_load_mode():
    """The retry respawns from `cmd` on a device set the fit was never proved
    against (cards the crashed launch never touched, or the discrete survivors of a
    narrowing), so the mode that fit concluded cannot ride along. Checked at the
    source, like the --fit on retry above, because this arm only runs behind a real
    kernel-image crash."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    retry = src[src.index("the llama.cpp build has no kernels") :]
    # Bounded at the respawn, so the strip is proved to happen BEFORE it.
    retry = retry[: retry.index('label = "-archfallback"')]
    assert "_without_subsequence(cmd, self._fit_load_mode_flags)" in retry
    assert "self._fit_load_mode_flags = []" in retry


# ------------------------------------ pass-through args that move the weights


# Every spelling llama.cpp accepts for "run this somewhere the planner did not
# put it", each of which is appended AFTER Unsloth's own placement flags and so
# wins by last-arg (common/arg.cpp assigns n_gpu_layers / devices per occurrence).
PLACEMENT_OVERRIDES = [
    ["-ngl", "0"],
    ["--gpu-layers", "0"],
    ["--n-gpu-layers", "0"],
    ["--gpu-layers=0"],
    # A partial count is the same problem: the planner priced full offload.
    ["-ngl", "12"],
    ["--device", "none"],
    ["-dev", "cpu"],
    ["--n-cpu-moe", "24"],
    ["-ot", r".ffn_.*_exps.=CPU"],
]


@pytest.mark.parametrize("extras", PLACEMENT_OVERRIDES)
def test_a_pass_through_placement_override_voids_the_vram_credit(extras, monkeypatch):
    """8 GiB into a 24 GiB card, 4 GiB of host RAM: the fit that says "none" is
    the VRAM one, and these flags run the weights out of RAM instead. Charging
    that fit's VRAM anyway would disable mmap on a load RAM cannot hold, which is
    an OOM kill where llama.cpp's own default would have demand-paged."""
    assert _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch) == FIT_MODE
    assert (
        _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch, extra_args = extras) is None
    )


@pytest.mark.parametrize("extras", PLACEMENT_OVERRIDES)
def test_an_override_still_takes_none_when_host_ram_holds_the_whole_load(extras, monkeypatch):
    """The credit is dropped, not the answer: 8 GiB against 64 GiB of RAM is
    resident wherever these flags put it, so the pick stands."""
    assert (
        _mode("linux", "nvidia_discrete", 8 * GIB, 64 * 1024, monkeypatch, extra_args = extras)
        == FIT_MODE
    )


@pytest.mark.parametrize(
    "extras",
    [
        [],
        None,
        ["-c", "8192"],
        ["--flash-attn", "on"],
        # -ncmoe 0 and -otd place nothing on this model's CPU side.
        ["--n-cpu-moe", "0"],
        ["-otd", r".*=CPU"],
        ["--device", "CUDA0"],
    ],
)
def test_extras_that_leave_placement_alone_keep_the_fit(extras, monkeypatch):
    assert (
        _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch, extra_args = extras)
        == FIT_MODE
    )


@pytest.mark.parametrize(
    "var,value",
    [
        ("LLAMA_ARG_OVERRIDE_TENSOR", r".ffn_.*_exps.=CPU"),
        ("LLAMA_ARG_CPU_MOE", "1"),
        ("LLAMA_ARG_N_CPU_MOE", "24"),
    ],
)
def test_inherited_cpu_placement_env_voids_the_vram_credit(var, value, monkeypatch):
    """The child inherits these, so they outlive any token stripping."""
    monkeypatch.setenv(var, value)
    assert _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch) is None


def test_the_launch_hands_the_fit_the_extras_the_child_will_get():
    """The predicate is only worth anything if the call site feeds it. Checked at
    the source, like the fallback ordering tests above, because reaching this call
    needs a real GPU probe."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    call = src[src.index("_fit_load_mode = self._fit_derived_load_mode(") :]
    call = call[: call.index("except Exception as e:")]
    assert "extra_args" in call
    # A gpu_ids pin drops the device flags from the emitted extras, so the fit has
    # to classify on the stripped copy rather than the request.
    assert "_strip_device_extra_args(extra_args)" in call
