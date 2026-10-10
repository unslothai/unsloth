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
from types import SimpleNamespace

import pytest

import utils.hardware as hardware
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_server_args import (
    apply_load_mode_policy,
    apply_model_memory_policy,
    split_policy_starves_devices,
)

GIB = 1024**3
FIT_MODE = LlamaCppBackend._FIT_LOAD_MODE
MIB = 1024 * 1024

HARDWARE = {
    "nvidia_discrete": ([(0, 24 * 1024)], False, False, set()),
    "nvidia_multi": ([(0, 12 * 1024), (1, 12 * 1024)], False, False, set()),
    "amd_discrete": ([(0, 24 * 1024)], False, False, set()),
    "amd_apu": ([(0, 24 * 1024)], False, True, set()),
    "vulkan_igpu": ([(0, 24 * 1024)], True, False, {0}),
    "cpu_only": ([], False, False, set()),
}

PLATFORMS = ["linux", "windows", "wsl", "macos"]

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


@pytest.mark.parametrize("platform,hw", _OS_GPU)
def test_a_load_that_fits_in_vram_takes_none(platform, hw, monkeypatch):
    """8 GiB into a 24 GiB card, with host RAM deliberately too small to help."""
    rows, _vulkan, apu, _shared = HARDWARE[hw]
    got = _mode(platform, hw, 8 * GIB, 4 * 1024, monkeypatch)
    if platform == "macos":
        # Metal keeps mmap: buffer_from_host_ptr makes it zero copy.
        assert got is None
    elif not rows or apu or hw == "vulkan_igpu":
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
    for term in (
        "mmproj_pinned_bytes",
        "kv_cache_bytes",
        "mtp_bytes",
        "compute_buffer_flat",
        "compute_buffer_ctx",
        "pipeline_overhead_bytes",
        "soft_overhead",
    ):
        assert _mode(footprint = rows_free, **base, **{term: 0}) == FIT_MODE
        assert _mode(footprint = rows_free, **base, **{term: 20 * GIB}) is None


def test_the_extra_devices_pipeline_overhead_is_charged(monkeypatch):
    """A layer split allocates a fixed CUDA context and scratch on every card, and
    the placement reserves 1 GiB per EXTRA device for it (_subset_model_size). The
    load-mode footprint has to carry the same term: 23 GiB across two 12 GiB cards
    with no host RAM to spill into is a fit until the second card's share is
    priced, and claiming that fit would hand the load a loader that cannot page."""
    base = dict(platform = "linux", hw = "nvidia_multi", avail_mib = 0, monkeypatch = monkeypatch)
    assert _mode(footprint = 23 * GIB, **base) == FIT_MODE
    assert _mode(footprint = 23 * GIB, pipeline_overhead_bytes = 2 * GIB, **base) is None
    assert (
        _mode(
            platform = "linux",
            hw = "nvidia_discrete",
            footprint = 23 * GIB,
            avail_mib = 0,
            monkeypatch = monkeypatch,
            pipeline_overhead_bytes = 0,
        )
        == FIT_MODE
    )


def test_the_launch_charges_the_pipeline_overhead_per_extra_device():
    """max(0, n - 1), so a single-GPU load adds nothing, and ungated by whether
    llama.cpp keeps the pipeline -- the per-device context is there either way,
    which is why the placement's own term is ungated too."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "pipeline_overhead_bytes=(max(0,_fit_devices-1)*_pipeline_overhead_bytes)" in compact


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
    _MTP_DRAFT_COMPUTE_BYTES = LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES


def test_no_cpu_pinned_drafter_charges_nothing():
    stub = _DraftStub(3 * GIB, 512 * MIB)
    assert stub._cpu_resident_draft_bytes(8192, drafter_path = None) == 0


def test_a_cpu_pinned_drafter_is_charged_weights_plus_kv_plus_its_graph():
    """``-ngld 0`` takes the drafter off the GPU. It does not delete it: those
    bytes are in host RAM, and ``none`` allocates them anonymously there.

    The decode graph goes with them. llama.cpp gives the drafter a context of its
    own and decodes through it (common/speculative.cpp), and _soft_overhead only
    charges _MTP_DRAFT_COMPUTE_BYTES while _mtp_reserves_gpu, which is False for
    exactly this placement -- so left out here it is nowhere in the footprint."""
    stub = _DraftStub(3 * GIB, 512 * MIB)
    assert stub._cpu_resident_draft_bytes(8192, drafter_path = "d.gguf") == (
        3 * GIB + 512 * MIB + LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES
    )


def test_the_cpu_drafters_decode_graph_is_charged_to_host_ram(monkeypatch):
    """And it is enough to move the answer on its own: 8 GiB of target against a
    24 GiB card, with host RAM sized so the drafter's weights and KV clear the
    headroom by less than the graph. Charged to RAM alone, like the rest of a
    host-only term."""
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    stub = _DraftStub(2 * GIB, 0)
    draft = stub._cpu_resident_draft_bytes(8192, drafter_path = "d.gguf")
    avail_mib = (2 * GIB + LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES // 2) // MIB + 2 * 1024

    def _fit(host_only):
        return LlamaCppBackend._fit_derived_load_mode(
            _Stub(avail_mib),
            model_size = 8 * GIB,
            host_only_bytes = host_only,
            gpus = [(0, 24 * 1024)],
            avail_mib = avail_mib,
        )

    assert _fit(2 * GIB) == FIT_MODE
    assert _fit(draft) is None


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
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "_cpu_draft_path=_mtp_draft_for_budgetif_draft_on_cpuelseNone" in compact
    assert compact.index("_cpu_draft_path=_mtp_draft_for_budget") < compact.index(
        "if_draft_on_cpu:_mtp_draft_for_budget=None"
    )
    assert "host_only_bytes=(_cpu_draft_fit_bytesor0)+_ckpt_host_bytes," in compact
    assert "or_cpu_draft_fit_bytesisNone" in compact


def test_a_weights_only_drafter_reserve_counts_as_unsized():
    """_flat_mtp_engages is "callback missing OR _mtp_kv_unsized", and the second
    arm is the one this call site has to keep: _estimate_mtp_overhead_bytes returns
    weights (plus any MLA/Mamba target copy) rather than None when the draft KV
    cannot be sized, so the callback exists and prices no draft KV at all. Charging
    that as a sized drafter understates a ctx-linear term the placement only covers
    with a flat cushion the footprint has no room for. Re-narrowing to
    "mtp_overhead_fn is None" absorbs the arm away, which is what it used to do."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert (
        "mtp_unsized=bool(_flat_mtp_engagesor_cpu_draft_fit_bytesisNone"
        "or_draft_split_across_host)"
    ) in compact
    assert "_flat_mtp_engagesandmtp_overhead_fnisNone" not in compact


def test_a_cpu_pinned_drafter_is_not_paid_for_out_of_vram(monkeypatch):
    """8 GiB target and a 3 GiB CPU-pinned drafter against a 24 GiB card with 4 GiB
    of RAM. Pooled, the card covers all 11 GiB and the fit reads as real; but
    ``-ngld 0`` means those 3 GiB can only be allocated in host RAM, where 4 GiB
    minus the 2 GiB headroom does not hold them, and ``none`` would make them
    anonymous instead of a mapping the OS can page."""
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    stub = _Stub(4 * 1024)

    def _fit(**kwargs):
        return LlamaCppBackend._fit_derived_load_mode(
            stub,
            model_size = 8 * GIB,
            gpus = [(0, 24 * 1024)],
            avail_mib = 4 * 1024,
            **kwargs,
        )

    assert _fit() == FIT_MODE
    assert _fit(host_only_bytes = 3 * GIB) is None
    assert _fit(mtp_bytes = 3 * GIB) == FIT_MODE
    stub._avail_mib = 16 * 1024
    assert (
        LlamaCppBackend._fit_derived_load_mode(
            stub,
            model_size = 8 * GIB,
            gpus = [(0, 24 * 1024)],
            avail_mib = 16 * 1024,
            host_only_bytes = 3 * GIB,
        )
        == FIT_MODE
    )


def test_the_fitters_margin_is_not_credited_to_the_fit(monkeypatch):
    """23 GiB free on a 24 GiB card and a 22.5 GiB footprint, with 2 GiB of RAM.

    Raw free VRAM covers it, but a launch that leaves ``--fit on`` runs llama.cpp's
    fitter, which keeps ``--fit-target`` (default 1024 MiB, "target margin per device
    for --fit") free on every device and spills the rest to host RAM instead. Those
    weights would then be anonymous under ``none``, on RAM nobody priced.
    """
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    stub = _Stub(2 * 1024)
    rows = [(0, 23 * 1024)]
    footprint = int(22.5 * GIB)

    def _fit(margin):
        return LlamaCppBackend._fit_derived_load_mode(
            stub,
            model_size = footprint,
            gpus = rows,
            fit_margin_mib = margin,
            avail_mib = 2 * 1024,
        )

    assert _fit(0.0) == FIT_MODE
    assert _fit(1024.0) is None


def test_the_margin_is_charged_per_device():
    """--fit-target is per device, so two cards keep two margins."""
    stub = _Stub(None)
    rows = [(0, 12 * 1024), (1, 12 * 1024)]
    assert stub._fits_without_paging(23 * GIB, rows, avail_mib = None) is True
    assert stub._fits_without_paging(23 * GIB, rows, vram_margin_mib = 1024.0, avail_mib = None) is None
    assert stub._fits_without_paging(21 * GIB, rows, vram_margin_mib = 1024.0, avail_mib = None) is True


@pytest.mark.parametrize(
    "auto_fit,delta,supports,expected",
    [
        (False, 0.0, True, 1024.0),
        (True, 0.0, True, 512.0),
        (False, 2048.0, True, 3072.0),
        (True, -4096.0, True, 512.0),
        (True, 0.0, False, 1024.0),
    ],
)
def test_the_fit_and_the_flag_read_the_same_margin(auto_fit, delta, supports, expected):
    got = LlamaCppBackend._fit_target_margin_mib(
        auto_fit = auto_fit,
        fit_target_delta_mib = delta,
        supports_fit_target = supports,
    )
    assert got == expected
    flags = LlamaCppBackend._ctx_integrity_flags(
        1,
        True,
        auto_fit,
        0,
        0,
        {"supports_fit_target": supports},
    )
    if "--fit-target" in flags:
        assert flags[flags.index("--fit-target") + 1] == str(int(expected))


def test_the_launch_charges_the_fitters_margin_only_when_fit_stays_on():
    """Checked at the source: reaching this call needs a real GPU probe."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    call = compact[compact.index("_fit_margin_mib=(") :]
    call = call[: call.index("exceptExceptionase:")]
    assert "max(self._fit_target_margin_mib(" in call
    # Extras land after --fit and llama.cpp is last-wins, so use the effective state.
    assert "or0.0,)if_fitter_runselse0.0" in call
    assert "fit_target_delta_mib=_fit_target_delta_mibifuse_fitelse0.0" in call
    assert "supports_fit_target=use_fitand" in call
    assert "fit_target_margin_in(_fit_extras,_fit_env)or0.0" in call
    assert "fit_margin_mib=_fit_margin_mib" in call


def test_the_effective_fitter_state_reads_the_launchs_own_fit_flag():
    """`--fit on` in the extras beats the proved path's `--fit off` by last-arg.

    llama.cpp assigns `params.fit_params` on every occurrence (common/arg.cpp) and
    `-ngl -1` is its own default, which the fitter is free to lower (common/fit.cpp
    aborts only on a count the user really set). So the margin has to be charged.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    from core.inference.llama_server_args import fit_is_effectively_on

    compact = "".join(inspect.getsource(B.load_model).split())
    assert '_fitter_runs=fit_is_effectively_on(["--fit","on"ifuse_fitelse"off",' in compact

    def _runs(
        use_fit,
        extras,
        env = None,
    ):
        return fit_is_effectively_on(["--fit", "on" if use_fit else "off", *extras], env)

    assert _runs(False, []) is False
    assert _runs(True, []) is True
    assert _runs(False, ["--fit", "on"]) is True
    assert _runs(True, ["--fit", "off"]) is False
    # llama.cpp reads env BEFORE argv, and this launch always emits --fit.
    assert _runs(False, [], {"LLAMA_ARG_FIT": "1"}) is False


@pytest.mark.parametrize(
    "extras,env,expected",
    [
        ([], None, None),
        (["--fit-target", "4096"], None, 4096.0),
        (["-fitt", "8192"], None, 8192.0),
        # Per-device margins: the largest is the only safe price.
        (["--fit-target", "512,4096,1024"], None, 4096.0),
        # Upstream splits on ',' and '/' (common/arg.cpp).
        (["--fit-target", "4096/4096"], None, 4096.0),
        (["-fitt", "512/4096/1024"], None, 4096.0),
        (["--fit-target", "512,4096/1024"], None, 4096.0),
        ([], {"LLAMA_ARG_FIT_TARGET": "1024/8192"}, 8192.0),
        (["--fit-target", "512/lots"], None, None),
        (["--fit-target", "4096", "--fit-target", "512"], None, 512.0),
        ([], {"LLAMA_ARG_FIT_TARGET": "2048"}, 2048.0),
        (["--fit-target", "512"], {"LLAMA_ARG_FIT_TARGET": "8192"}, 512.0),
        (["--fit-target", "lots"], None, None),
        (["--fit-target", "512,lots"], None, None),
    ],
)
def test_a_pass_through_fit_target_is_the_margin_the_child_really_keeps(extras, env, expected):
    from core.inference.llama_server_args import fit_target_margin_in
    assert fit_target_margin_in(extras, env) == expected


@pytest.fixture
def toggles(monkeypatch):
    """Drive the Model Memory settings the two policies read lazily."""
    import utils.model_memory_settings as mm

    def _set(keep_resident, no_ram_reserve):
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
        [(False, False), (True, False), (False, True), (True, True)],
        [None, "none", "mmap", "mlock", "dio", "auto"],
        [None, "none"],
        [True, False],
        [True, False],
    )
)


@pytest.mark.parametrize("axes", _CHAIN_AXES)
def test_the_chain_never_emits_an_unknown_or_duplicate_mode(axes, toggles):
    (keep, no_reserve), user, fit, supports, host = axes
    toggles(keep, no_reserve)
    argv = _chain([], user_mode = user, fit_mode = fit, supports = supports, host_resident = host)
    assert argv.count("--load-mode") <= 1
    if "--load-mode" in argv:
        value = argv[argv.index("--load-mode") + 1]
        assert value in {"none", "mmap", "mlock", "mmap+mlock", "dio"}
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
    this byte-identical to an Unsloth instance that never had the feature.
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
        assert "none" not in argv
    if keep and not no_reserve:
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
    # User's copy after the managed block so last-wins picks it.
    assert argv == ["--load-mode", "none", "--load-mode", "mmap"]
    assert argv[-1] == "mmap"


def test_the_cpu_fallback_drops_the_fits_load_mode(monkeypatch):
    """A CPU replay runs on no GPU, so the VRAM half of the fit is void and the
    whole model has to come out of host RAM. mmap is what makes that survivable."""
    from core.inference import llama_cpp as lc
    from unittest import mock

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._memory_dio_flags = []
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
        out, _reason, _note = backend._prepare_cpu_fallback_launch("llama-server", replay, {}, {})
    assert "--load-mode" not in out
    assert "none" not in out[: out.index("--gpu-layers")]
    assert out[-2:] == ["--device", "none"]


def test_the_cpu_fallback_keeps_a_load_mode_the_user_asked_for(monkeypatch):
    """Only Unsloth's own tokens are recorded, so a user's pick survives."""
    from core.inference import llama_cpp as lc
    from unittest import mock

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._memory_dio_flags = []
    backend._fit_load_mode_flags = []
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
        out, _reason, _note = backend._prepare_cpu_fallback_launch("llama-server", replay, {}, {})
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
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    retry = src[src.index("retrying once with --fit on so it can offload") :]
    retry = retry[: retry.index("_did_fit_retry = True")]
    assert "_fit_load_mode_flags" in retry
    assert "_without_subsequence" in retry
    # The record stays: the arch-crash respawn below still strips the tokens `cmd` carries.
    assert "self._fit_load_mode_flags = []" not in retry


def test_the_arch_crash_retry_voids_the_fit_the_weights_only_floor_still_allows():
    """The premise behind the strip below, priced on real numbers.

    A 40 GB card and a 4 GB one, 20 GB of RAM. The fit was proved against the card
    the launch PINNED; the arch-crash retry moves to the survivor, and there the
    same footprint no longer fits. The retry's own guard is a weights-only floor by
    design, so it passes and cannot re-establish the proof.
    """
    rows = [(0, 40_000), (1, 4_000)]
    footprint = 30 * GIB
    weights = 20 * GIB
    stub = _Stub(20_000)

    assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0, 1]) == [1]

    assert stub._fits_without_paging(footprint, rows, gpu_indices = [0]) is True
    assert stub._fits_without_paging(footprint, rows, gpu_indices = [1]) is False
    assert LlamaCppBackend._host_offload_shortfall_message(weights - 4_000 * MIB, 20_000) is None


def test_the_arch_crash_retry_drops_the_fits_load_mode():
    """The retry respawns from `cmd` on a device set the fit was never proved
    against (cards the crashed launch never touched, or the discrete survivors of a
    narrowing), so the mode that fit concluded cannot ride along. Checked at the
    source, like the --fit on retry above, because this arm only runs behind a real
    kernel-image crash."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    retry = src[src.index("the llama.cpp build has no kernels") :]
    retry = retry[: retry.index('label = "-archfallback"')]
    assert "_without_subsequence(cmd, self._fit_load_mode_flags)" in retry
    assert "self._fit_load_mode_flags = []" in retry
    # Drop the record ONLY where `cmd` was rewritten, else a later respawn is not stripped.
    assert src.count("self._fit_load_mode_flags = []") == 1


def test_the_no_flash_retry_drops_the_fits_load_mode():
    """Both --flash-attn off respawns, which rewrite the footprint the fit priced.

    The mode is NOT gated on full offload: a partially offloaded "--fit on" launch
    that fits VRAM plus RAM carries it too, and on that launch the --fit on retry
    (gated on fully_gpu_offloaded) is not a second net. Checked at the source, like
    the retries above, because this arm only runs behind a real signal crash.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    for label in ('label = "-noflash"', 'label = "-noflash-mtp"'):
        arm = src[: src.index(label)]
        arm = arm[arm.rindex("self._with_flash_attn_off(") :]
        assert "_drop_fit_load_mode_for_no_flash(_fa_cmd)" in arm
    assert src.count("_drop_fit_load_mode_for_no_flash(_fa_cmd)") == 2
    helper = src[src.index("def _drop_fit_load_mode_for_no_flash(") :]
    helper = helper[: helper.index("def _spawn_and_wait(")]
    assert "_without_subsequence(fa_cmd, self._fit_load_mode_flags)" in helper
    assert "self._fit_load_mode_flags = []" not in helper


def test_the_no_flash_rewrite_really_grows_the_footprint_the_fit_priced():
    """The premise behind the strip above, on the two terms the estimator misses.

    The FA-off rewrite takes a quantized V to f16, and on an MLA model K goes with
    it, because llama.cpp rejects a split K/V there before it rejects a quantized V
    without flash attention. The MLA branch of the KV estimate prices that latent
    cache at the K width alone, so the upcast is unbudgeted.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B

    cmd = ["llama-server", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    out = B._with_flash_attn_off(cmd, mla = True)
    assert out is not None
    assert "on" not in out[out.index("--flash-attn") : out.index("--flash-attn") + 2]
    assert out[out.index("--cache-type-v") + 1] == "f16"
    assert out[out.index("--cache-type-k") + 1] == "f16"


# Appended after Unsloth's placement flags, so they win by last-arg.
PLACEMENT_OVERRIDES = [
    ["-ngl", "0"],
    ["--gpu-layers", "0"],
    ["--n-gpu-layers", "0"],
    ["--gpu-layers=0"],
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


@pytest.mark.parametrize("value", ["0", "12", " 12 ", "all", "garbage"])
def test_an_inherited_gpu_layer_count_voids_the_vram_credit(value, monkeypatch):
    """The env twin of -ngl, and the sharper of the two: the fitting path emits
    "--fit on" and no layer flag at all, so an inherited count is the ONLY layer
    policy the child sees, and llama.cpp's fitter refuses to lower a count it did
    not set (common/fit.cpp: "n_gpu_layers already set by user, abort", downgraded
    to a warning). Nothing clears it on an automatic load -- only Manual mode owns
    it -- so the weights the fit credited to a card load into host RAM instead."""
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", value)
    assert _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch) is None
    assert _mode("linux", "nvidia_discrete", 8 * GIB, 64 * 1024, monkeypatch) == FIT_MODE


@pytest.mark.parametrize("value", ["-1", "auto", "AUTO", "", "   "])
def test_an_inherited_default_gpu_layer_count_keeps_the_fit(value, monkeypatch):
    """-1 / "auto" IS llama.cpp's default (common/common.h, llama-model.cpp), so
    the fitter still runs and still owns the placement this fit priced. An empty
    value never reaches placement at all: std::stoi throws and the child dies."""
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", value)
    assert _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch) == FIT_MODE


def test_the_launch_hands_the_fit_the_extras_the_child_will_get():
    """The predicate is only worth anything if the call site feeds it. Checked at
    the source, like the fallback ordering tests above, because reaching this call
    needs a real GPU probe."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    call = src[src.index("_fit_extras = (") :]
    call = call[: call.index("except Exception as e:")]
    assert "extra_args = _fit_extras" in call
    assert "_strip_device_extra_args(extra_args)" in call
    # These overrides have env twins read BEFORE argv; the pin drops both.
    assert "env = _fit_env" in call
    assert "_fit_env = dict(os.environ)" in call
    assert "self._clear_device_placement_env(_fit_env)" in call
    assert "self._clear_manual_placement_env(_fit_env)" in call


def test_an_inherited_device_selection_voids_the_vram_credit(monkeypatch):
    """Nothing clears LLAMA_ARG_DEVICE on an automatic load, so the child gets it.

    Only an explicit gpu_ids pin calls _clear_device_placement_env; unpinned, an
    inherited "none" runs the whole load out of host RAM while the argv says
    nothing at all. llama.cpp applies the env before argv (common/arg.cpp).
    """
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "none")
    assert _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch) is None
    assert (
        _mode(
            "linux",
            "nvidia_discrete",
            8 * GIB,
            4 * 1024,
            monkeypatch,
            extra_args = ["--device", "CUDA0"],
        )
        == FIT_MODE
    )


@pytest.mark.parametrize(
    "hw,extras,gpu_indices,expected",
    [
        ("nvidia_discrete", ["--device", "CUDA0"], None, FIT_MODE),
        ("nvidia_multi", ["--device", "CUDA0"], None, None),
        ("nvidia_multi", ["-dev", "CUDA1"], None, None),
        ("nvidia_multi", ["--device", "CUDA0,CUDA1"], None, FIT_MODE),
        # parse_device_list never deduplicates: both entries are one VRAM pool.
        ("nvidia_multi", ["--device", "CUDA0,CUDA0"], None, None),
        ("nvidia_multi", ["-dev", "CUDA1,CUDA1,CUDA1"], None, None),
        ("nvidia_discrete", ["--device", "CUDA0,CUDA0"], None, FIT_MODE),
        ("nvidia_multi", ["--device", "CUDA0", "--device", "CUDA0,CUDA1"], None, FIT_MODE),
        ("nvidia_multi", ["--device", "CUDA0,CUDA1", "--device", "CUDA0"], None, None),
        ("nvidia_multi", [], None, FIT_MODE),
    ],
)
def test_a_narrowing_device_pass_through_voids_the_vram_credit(
    hw, extras, gpu_indices, expected, monkeypatch
):
    """`--device` is opt-in under auto-select (stripped only when gpu_ids is set).

    llama.cpp REPLACES the device list on every occurrence and offloads to nothing
    else (common/arg.cpp parse_device_list), so a name list shorter than what this
    fit charges leaves the credit paying for cards the child never opens.
    """
    assert (
        _mode(
            "linux",
            hw,
            20 * GIB,
            2 * 1024,
            monkeypatch,
            extra_args = extras,
            gpu_indices = gpu_indices,
        )
        == expected
    )


def test_the_device_narrowing_is_counted_against_what_the_fit_credits(monkeypatch):
    """Not against what was DETECTED: a launch already pinned to one card is not
    narrowed by an extras `--device` naming one, and voiding there would abstain on
    a fit that still holds."""
    for extras, expected in ((["--device", "CUDA0"], FIT_MODE), ([], FIT_MODE)):
        assert (
            _mode(
                "linux",
                "nvidia_multi",
                10 * GIB,
                2 * 1024,
                monkeypatch,
                extra_args = extras,
                gpu_indices = [0],
            )
            == expected
        )


def _multi(footprint, avail_mib, monkeypatch, **kwargs):
    """Two 12 GiB cards, both credited, so a starved split is observable."""
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    return LlamaCppBackend._fit_derived_load_mode(
        _Stub(avail_mib),
        model_size = footprint,
        gpus = [(0, 12 * 1024), (1, 12 * 1024)],
        shared_gpu_ids = set(),
        is_vulkan_backend = False,
        avail_mib = avail_mib,
        **kwargs,
    )


@pytest.mark.parametrize(
    "extras",
    [
        ["--split-mode", "none"],
        ["-sm", "none"],
        ["--tensor-split", "1,0"],
        ["-ts", "1,0"],
        # A short list zero-fills the tail upstream.
        ["--tensor-split", "1"],
        ["--split-mode", "layer", "--split-mode", "none"],
    ],
)
def test_a_starving_split_voids_the_pooled_vram_credit(extras, monkeypatch):
    """18 GiB across 2x12 GiB fits the pool but not one card, and RAM is far too
    small to hold the spill. Crediting both cards anyway would emit the no-mmap
    flag for weights that then have nowhere pageable to live."""
    assert _multi(18 * GIB, 4 * 1024, monkeypatch, extra_args = extras) is None


@pytest.mark.parametrize(
    "env_value",
    [{"LLAMA_ARG_SPLIT_MODE": "none"}, {"LLAMA_ARG_TENSOR_SPLIT": "1,0"}],
)
def test_an_inherited_starving_split_voids_it_too(env_value, monkeypatch):
    """The child inherits these exactly as it inherits LLAMA_ARG_DEVICE."""
    assert _multi(18 * GIB, 4 * 1024, monkeypatch, env = env_value) is None


@pytest.mark.parametrize(
    "extras",
    [
        ["--split-mode", "layer"],
        ["--split-mode", "row"],
        ["--split-mode", "tensor"],
        ["--tensor-split", "1,1"],
        ["--tensor-split", "3,1"],
        ["--tensor-split", "nonsense"],
    ],
)
def test_a_split_that_starves_nobody_keeps_the_fit(extras, monkeypatch):
    assert _multi(18 * GIB, 4 * 1024, monkeypatch, extra_args = extras) == FIT_MODE


def test_a_starving_split_is_a_no_op_on_a_single_gpu(monkeypatch):
    """--split-mode none confines the model to one card, which is where a
    single-GPU load already put it."""
    assert (
        _mode(
            "linux",
            "nvidia_discrete",
            8 * GIB,
            4 * 1024,
            monkeypatch,
            extra_args = ["--split-mode", "none"],
        )
        == FIT_MODE
    )


def test_a_starving_split_still_takes_none_when_ram_holds_the_load(monkeypatch):
    """The credit is dropped, not the answer, exactly as for the other overrides."""
    assert _multi(18 * GIB, 64 * 1024, monkeypatch, extra_args = ["-sm", "none"]) == FIT_MODE


@pytest.mark.parametrize(
    "raw,n_credited,expected",
    [
        ("none", 2, True),
        ("NONE", 2, True),
        ("none", 1, False),
        ("layer", 2, False),
        ("row", 2, False),
        ("tensor", 2, False),
        (None, 2, False),
    ],
)
def test_split_mode_starvation_is_value_aware(raw, n_credited, expected):
    args = None if raw is None else ["--split-mode", raw]
    assert split_policy_starves_devices(args, n_credited) is expected


@pytest.mark.parametrize(
    "raw,n_credited,expected",
    [
        ("1,0", 2, True),
        ("0,1", 2, True),
        ("1", 2, True),
        ("1/0", 2, True),  # upstream splits on "/" too
        ("1,1", 2, False),
        ("3,1", 2, False),
        ("1,1,1", 2, False),
        ("1,0", 1, False),
    ],
)
def test_tensor_split_starvation_matches_upstream_parsing(raw, n_credited, expected):
    assert split_policy_starves_devices(["-ts", raw], n_credited) is expected


def _vulkan(footprint, avail_mib, monkeypatch, **kwargs):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    return LlamaCppBackend._fit_derived_load_mode(
        _Stub(avail_mib),
        model_size = footprint,
        gpus = [(0, 12 * 1024), (1, 12 * 1024), (2, 12 * 1024)],
        shared_gpu_ids = set(),
        is_vulkan_backend = True,
        avail_mib = avail_mib,
        gpu_indices = [0, 1],
        **kwargs,
    )


@pytest.mark.parametrize(
    "extras",
    [
        ["--device", "Vulkan1,Vulkan2"],
        ["-dev", "Vulkan0,Vulkan2"],
        ["--device", "Vulkan0,Vulkan1,Vulkan2"],
    ],
)
def test_a_replaced_vulkan_pin_voids_the_credit(extras, monkeypatch):
    """Unsloth pins Vulkan0,Vulkan1 from the credited ordinals; a pass-through
    --device lands after it and last-wins."""
    assert _vulkan(18 * GIB, 4 * 1024, monkeypatch, extra_args = extras) is None


@pytest.mark.parametrize(
    "extras",
    [
        ["--device", "Vulkan0,Vulkan1"],
        ["--device", "vulkan0,vulkan1"],  # ggml name-matching is not case sensitive
        ["--device", "Vulkan1,Vulkan0"],
    ],
)
def test_a_restated_vulkan_pin_keeps_the_fit(extras, monkeypatch):
    assert _vulkan(18 * GIB, 4 * 1024, monkeypatch, extra_args = extras) == FIT_MODE


def test_a_replaced_cuda_pin_is_still_judged_by_count(monkeypatch):
    """Only Vulkan gets the exact match: CUDA/ROCm ordinals are assigned after a
    visibility mask this launch has not written, so there is no name to compare."""
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    assert (
        LlamaCppBackend._fit_derived_load_mode(
            _Stub(4 * 1024),
            model_size = 18 * GIB,
            gpus = [(0, 12 * 1024), (1, 12 * 1024), (2, 12 * 1024)],
            shared_gpu_ids = set(),
            is_vulkan_backend = False,
            avail_mib = 4 * 1024,
            gpu_indices = [0, 1],
            extra_args = ["--device", "CUDA1,CUDA2"],
        )
        == FIT_MODE
    )


def test_a_cpu_pinned_projector_is_charged_to_host_ram(monkeypatch):
    """8 GiB of weights and a 10 GiB projector pinned to the CPU by
    --no-mmproj-offload. The card has 24 GiB, so surplus VRAM would happily
    cover all 18 GiB and answer yes without ever asking RAM -- but the projector
    can only live in the 4 GiB of RAM this host has."""
    assert (
        _mode(
            "linux",
            "nvidia_discrete",
            8 * GIB,
            4 * 1024,
            monkeypatch,
            mmproj_pinned_bytes = 10 * GIB,
        )
        is None
    )


def test_a_cpu_pinned_projector_fits_when_ram_really_holds_it(monkeypatch):
    """The same load with RAM that can take the projector still picks the mode."""
    assert (
        _mode(
            "linux",
            "nvidia_discrete",
            8 * GIB,
            64 * 1024,
            monkeypatch,
            mmproj_pinned_bytes = 10 * GIB,
        )
        == FIT_MODE
    )


class _MixedRocmStub(_Stub):
    """A ROCm host where only SOME of the visible devices are unified-memory APUs.

    The plain _Stub answers the APU question blanket, which is exactly the shape
    that cannot tell a mixed host from a pure one.
    """

    def __init__(self, avail_mib, unified):
        super().__init__(avail_mib)
        self._unified = set(unified)

    def _amd_apu_wants_unified_memory(self, gpu_indices = None):
        if gpu_indices is None:
            return bool(self._unified)
        return any(idx in self._unified for idx in gpu_indices)


def _mixed_rocm(footprint, avail_mib, unified, rows, monkeypatch, **kwargs):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    return LlamaCppBackend._fit_derived_load_mode(
        _MixedRocmStub(avail_mib, unified),
        model_size = footprint,
        gpus = rows,
        shared_gpu_ids = set(),
        is_vulkan_backend = False,
        avail_mib = avail_mib,
        gpu_indices = [idx for idx, _free in rows],
        **kwargs,
    )


def test_a_discrete_rocm_card_keeps_its_vram_next_to_an_apu(monkeypatch):
    """An 8 GiB APU beside a 24 GiB discrete Radeon, 20 GiB to place.

    The APU's "VRAM" IS the host RAM the spill would come from, so it is dropped
    from the credit -- but the discrete card's 24 GiB is its own pool and holds
    the whole load. Marking every row shared because one of them is an APU would
    forfeit a fit that is really there: 4 GiB of RAM (2 after headroom) is
    nowhere near enough.
    """
    rows = [(0, 8 * 1024), (1, 24 * 1024)]
    assert _mixed_rocm(20 * GIB, 4 * 1024, {0}, rows, monkeypatch) == FIT_MODE


def test_an_apu_only_rocm_host_still_loses_the_whole_credit(monkeypatch):
    """The case the union exists for: counting the shared pool as VRAM AND as the
    RAM the spill comes from would fit the model twice into memory holding it once.
    """
    rows = [(0, 24 * 1024)]
    assert _mixed_rocm(20 * GIB, 4 * 1024, {0}, rows, monkeypatch) is None


def test_every_apu_on_a_mixed_host_loses_its_credit(monkeypatch):
    """Per device, not "the first one": two APUs beside one discrete card leaves
    only the discrete card's 12 GiB, which cannot hold 20 GiB."""
    rows = [(0, 24 * 1024), (1, 24 * 1024), (2, 12 * 1024)]
    assert _mixed_rocm(20 * GIB, 4 * 1024, {0, 1}, rows, monkeypatch) is None


@pytest.mark.parametrize(
    "extras,env",
    [
        (["--no-kv-offload"], None),
        (["-nkvo"], None),
        ([], {"LLAMA_ARG_KV_OFFLOAD": "0"}),
        (["-kvo", "-nkvo"], None),
        # The env twin parses BEFORE argv.
        ([], {"LLAMA_ARG_KV_OFFLOAD": "false"}),
        # get_value_from_env checks the negative spelling first, falsey on presence.
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "1"}),
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "0"}),
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": ""}),
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "0", "LLAMA_ARG_KV_OFFLOAD": "1"}),
    ],
)
def test_a_cpu_only_kv_cache_is_charged_to_host_ram_alone(extras, env, monkeypatch):
    """8 GiB of weights and a 12 GiB cache on a 24 GiB card with 4 GiB of RAM.

    Pooled, the card's surplus covers the cache and the fit answers yes without
    ever asking RAM. But --no-kv-offload allocates every layer's K and V on the
    CPU buffer whatever the layer placement says (llama-kv-cache.cpp upgrades the
    buffer type only inside `if (offload)`), so those 12 GiB can only come out of
    the 4 GiB this host has.
    """
    assert (
        _mode(
            "linux",
            "nvidia_discrete",
            8 * GIB,
            4 * 1024,
            monkeypatch,
            kv_cache_bytes = 12 * GIB,
            extra_args = extras,
            env = env or {},
        )
        is None
    )


@pytest.mark.parametrize(
    "extras,env",
    [
        ([], None),
        (["--kv-offload"], None),
        (["-kvo"], {"LLAMA_ARG_KV_OFFLOAD": "0"}),
        (["-nkvo", "-kvo"], None),
        (["-kvo"], {"LLAMA_ARG_NO_KV_OFFLOAD": "1"}),
    ],
)
def test_an_offloaded_kv_cache_is_still_pooled(extras, env, monkeypatch):
    """The same load with the cache where llama.cpp puts it by default. Charging
    it to host RAM anyway would abstain on a fit that is really there."""
    assert (
        _mode(
            "linux",
            "nvidia_discrete",
            8 * GIB,
            4 * 1024,
            monkeypatch,
            kv_cache_bytes = 12 * GIB,
            extra_args = extras,
            env = env or {},
        )
        == FIT_MODE
    )


def test_a_cpu_only_kv_cache_is_not_charged_twice(monkeypatch):
    """Host-only, like the pinned projector: counted ONCE in the footprint and
    charged to RAM, not added on top of it. 8 + 12 GiB into 32 GiB of RAM (30
    after headroom) fits; double-counting the cache would make it 32 and fail."""
    assert (
        _mode(
            "linux",
            "cpu_only",
            8 * GIB,
            32 * 1024,
            monkeypatch,
            kv_cache_bytes = 12 * GIB,
            extra_args = ["-nkvo"],
        )
        == FIT_MODE
    )


def test_a_tensor_split_charges_the_replicated_compute_buffer():
    """Check that the fit charges the same per-device buffer as _plan_tensor_parallel.
    This source check avoids requiring a multi-GPU probe."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "_compute_buffer_tensor=self._estimate_compute_buffer_bytes(" in compact
    assert "per_device_tensor=True,)" in compact
    assert "if_compute_buffer_tensor<=0:" in compact
    # Gated on _tp_planned: tp_tensor_split stays None when llama.cpp sizes the split.
    assert "_tp_planned=False" in compact
    assert "use_fit=False_tp_planned=True" in compact
    assert (
        "compute_buffer_flat=(_fit_devices*_compute_buffer_tensor"
        "if_tp_plannedelse_compute_buffer_pipeline)"
    ) in compact


def test_a_tensor_split_pays_the_buffer_on_every_device():
    """Each device of a tensor split reserves the whole single-GPU compute buffer
    (Qwen3 8B across two GPUs: 128 MiB on each at ubatch 512, 512 MiB at 2048), so
    the fit's per-device multiplication is the whole difference from a layer lump."""
    b = LlamaCppBackend()
    b._vocab_size = 151936
    b._embedding_length = 4096
    b._feed_forward_length = 12288
    layer = b._estimate_compute_buffer_bytes(n_ubatch = 2048, n_parallel = 1)
    tensor = b._estimate_compute_buffer_bytes(n_ubatch = 2048, n_parallel = 1, per_device_tensor = True)
    assert tensor == layer
    assert 4 * tensor - layer > GIB


def test_the_replicated_buffer_can_decide_the_fit(monkeypatch):
    """23 GiB across two 12 GiB cards with no host RAM is a fit until the second
    device's copy of the compute buffer is priced."""
    base = dict(platform = "linux", hw = "nvidia_multi", avail_mib = 0, monkeypatch = monkeypatch)
    assert _mode(footprint = 22 * GIB, compute_buffer_flat = GIB, **base) == FIT_MODE
    assert _mode(footprint = 22 * GIB, compute_buffer_flat = 3 * GIB, **base) is None


@pytest.mark.parametrize(
    "extras,env,n_draft_layers,expected",
    [
        ([], None, 28, False),
        (["-ngld", "-1"], None, 28, False),
        (["-ngld", "0"], None, 28, False),
        (["-ngld", "1"], None, 28, True),
        (["--gpu-layers-draft", "14"], None, 28, True),
        (["--spec-draft-ngl=4"], None, 28, True),
        (["-ngld", "1"], None, None, True),
        # Equal to block count is still a split: i_gpu_start is 1, block 0 stays on host.
        (["-ngld", "28"], None, 28, True),
        (["-ngld", "29"], None, 28, False),
        (["-ngld", "999"], None, 28, False),
        ([], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "1"}, 28, True),
        (["-ngld", "-1"], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "1"}, 28, False),
        (["-ngld", "1", "-ngld", "-1"], None, 28, False),
        (["-ngld", "-1", "-ngld", "1"], None, 28, True),
        (["-ngld", "lots"], None, 28, False),
    ],
)
def test_a_partial_draft_offload_is_recognised(extras, env, n_draft_layers, expected):
    from core.inference.llama_cpp import _draft_is_split_across_host
    assert _draft_is_split_across_host(extras, env or {}, n_draft_layers = n_draft_layers) is expected


def test_the_draft_whole_offload_threshold_matches_the_main_model():
    """``-ngld <block count>`` is a split, on llama.cpp's own off-by-one.

    ``i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`` (llama-model.cpp), and
    ``n_layer_all`` is the GGUF block count read straight off the key, so a count
    EQUAL to the block count leaves ``i_gpu_start == 1`` and hands block 0 the CPU
    buffer list. The main model already draws the line there
    (``_partially_offloads_layers``); the drafter has to agree, or the fit credits
    VRAM for a block that never reaches the card.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B

    from core.inference.llama_cpp import _draft_is_split_across_host

    assert _draft_is_split_across_host(["-ngld", "28"], {}, n_draft_layers = 28) is True
    assert _draft_is_split_across_host(["--spec-draft-ngl=12"], {}, n_draft_layers = 12) is True
    assert (
        _draft_is_split_across_host([], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "28"}, n_draft_layers = 28)
        is True
    )
    assert _draft_is_split_across_host(["-ngld", "29"], {}, n_draft_layers = 28) is False
    stub = SimpleNamespace(n_layers = 28)
    assert B._partially_offloads_layers(stub, ["-ngl", "28"], {}) is True
    assert B._partially_offloads_layers(stub, ["-ngl", "29"], {}) is False


def test_a_partial_draft_offload_leaves_the_drafter_unsized():
    """Checked at the source: reaching this call needs a real drafter GGUF.

    -ngld 1 is not _extra_args_draft_offloaded_to_cpu (that is the -ngld 0 case), so
    _cpu_draft_path is None and mtp_bytes carries the drafter WHOLE in the
    GPU-eligible term while llama.cpp pins the layers past the count, and their KV,
    in host RAM. Nothing at the call site can say which slice stays on the card, so
    it has to abstain rather than credit VRAM for bytes that never reach it.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert (
        "_draft_split_across_host=bool(_mtp_draft_for_budgetand_draft_is_split_across_host("
        in compact
    )
    assert "_draft_is_split_across_host(_fit_extras,_fit_env," in compact
    assert (
        'n_draft_layers=getattr(self._draft_backend_for(_mtp_draft_for_budget),"_n_layers",None,)'
        in compact
    )
    assert "or_draft_split_across_host" in compact


def test_the_cpu_projector_retry_voids_the_fit_it_was_proved_against():
    """The premise behind the strip below, priced on real numbers.

    A 24 GiB card and 2 GiB of free host RAM. As launched the projector is on the
    card and the whole load fits VRAM alone, so ``_fits_without_paging`` answers
    True without ever consulting RAM. The CPU-projector retry moves exactly those
    bytes into host RAM (--no-mmproj-offload clears mmproj_use_gpu, and clip.cpp
    gates its whole GPU backend on it), and there the same footprint does not fit.
    """
    stub = _Stub(2 * 1024)
    rows = [(0, 24 * 1024)]
    weights, projector = 16 * GIB, 4 * GIB

    assert stub._fits_without_paging(weights + projector, rows) is True
    assert stub._fits_without_paging(weights + projector, rows, host_only_bytes = projector) is False


def test_the_cpu_projector_retry_drops_the_fits_load_mode():
    """The projector respawn changes placement, so the fit's conclusion goes with
    it, like the --fit on, arch-crash and no-flash retries. Checked at the source,
    like those, because this arm only runs behind a real projector startup crash."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    arm = src[: src.index('label = "-mmproj-cpu"')]
    arm = arm[arm.rindex("_cpu_projector_cmd = self._with_mmproj_offload_disabled(") :]
    assert "self._fit_load_mode_flags" in arm
    assert "_without_subsequence(" in arm
    assert "self._fit_load_mode_flags = []" not in arm
    assert src.count("self._fit_load_mode_flags = []") == 1


def _checkpoint_swa_backend():
    """A Gemma-shaped SWA model, the one architecture --ctx-checkpoints charges."""
    b = LlamaCppBackend()
    for name, value in {
        "_n_layers": 62,
        "_n_kv_heads": 16,
        "_n_heads": 32,
        "_embedding_length": 5376,
        "_kv_key_length": 128,
        "_kv_value_length": 128,
        "_sliding_window": 1024,
    }.items():
        setattr(b, name, value)
    return b


def test_context_checkpoint_snapshots_move_the_fit_verdict(monkeypatch):
    """The premise: the snapshots are GiBs, not a rounding error.

    62 SWA layers at 8192 context: 1.5 GiB of KV becomes 4.4 GiB with 8 snapshots
    per slot. A card sized for the first answers "none" for a load that really
    needs the second, and with the host holding nothing that is an OOM where the
    mapping would only have paged.
    """
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    backend = _checkpoint_swa_backend()
    ctx = 8192
    base = backend._estimate_kv_cache_bytes(ctx, "f16")
    snapshotted = backend._estimate_kv_cache_bytes(ctx, "f16", ctx_checkpoints = 8)
    assert snapshotted - base > 2 * GIB

    model = 16 * GIB
    stub = _Stub(1024)
    rows = [(0, (model + base) // MIB)]

    def _verdict(kv_bytes):
        return LlamaCppBackend._fit_derived_load_mode(
            stub,
            model_size = model,
            kv_cache_bytes = kv_bytes,
            gpus = rows,
            avail_mib = 1024,
        )

    assert _verdict(base) == FIT_MODE
    assert _verdict(snapshotted) is None


def test_the_fit_prices_the_effective_checkpoint_count():
    """The load-mode footprint prices checkpoints as host-only bytes."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "kv_cache_bytes=_kv_bytes(effective_ctx,0)," in compact
    assert "_kv_bytes(effective_ctx,_effective_ctx_checkpoints)-_kv_bytes(effective_ctx,0)," in (
        compact
    )
    assert "host_only_bytes=(_cpu_draft_fit_bytesor0)+_ckpt_host_bytes," in compact
    assert "def_kv_bytes(ctx:int,ctx_checkpoints:int=0)->int:" in compact


def test_an_inherited_loader_mode_wins_over_the_fits_pick():
    """llama.cpp applies LLAMA_ARG_LOAD_MODE before argv, so the managed flag would
    beat an operator's inherited choice silently. The fit's mode stands aside for
    it, the way it stands aside for the per-model pick; a per-model pick still
    wins, and so does a hand-typed flag, by last-arg."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    arm = compact[
        compact.index("_fit_load_mode_env_view=dict(_mem_env)") : compact.index(
            "_resolved_load_mode=load_modeor_fit_load_mode"
        )
    ]
    assert "scrub_memory_env(_fit_load_mode_env_view,_mem_settings)" in arm
    # Assert each clause separately: the formatter may wrap the call.
    assert "_fit_load_mode" in arm and "notload_mode" in arm
    assert "memory_env_selects_load_mode(_fit_load_mode_env_view)" in arm
    assert "_fit_load_mode=None" in arm


def test_an_unpriced_projector_is_the_difference_between_a_fit_and_an_oom(monkeypatch):
    """The premise behind pricing an inherited projector: 18 GiB of weights fit a
    20 GiB card, and the same load with a 3 GiB projector the child loads from
    LLAMA_ARG_MMPROJ does not."""
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    stub = _Stub(1024)
    rows = [(0, 20 * 1024)]
    weights, projector = 18 * GIB, 3 * GIB

    def _verdict(model_size):
        return LlamaCppBackend._fit_derived_load_mode(
            stub, model_size = model_size, gpus = rows, avail_mib = 1024
        )

    assert _verdict(weights) == FIT_MODE
    assert _verdict(weights + projector) is None


def test_the_fit_prices_a_projector_inherited_through_the_environment():
    """model_size is gated on launch_mmproj_path, so a projector that arrives only
    through LLAMA_ARG_MMPROJ is resident but uncharged. Sized when it can be, and
    abstaining when it cannot: a URL names a download that has not happened, and an
    unreadable path cannot be sized."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert 'else(_fit_env.get("LLAMA_ARG_MMPROJ")or"").strip()' in compact
    assert 'else(_fit_env.get("LLAMA_ARG_MMPROJ_URL")or"").strip()' in compact
    assert "_fit_env_mmproj_unsized=bool(_fit_env_mmproj_url)or(" in compact
    assert "if_fit_env_mmproj_unsized:" in compact
    assert "_fit_model_size=None" in compact
    assert "model_size=_fit_model_size," in compact
    assert (
        "mmproj_pinned_bytes=_mmproj_pinned_bytes+(_fit_env_mmproj_bytesif_fit_env_mmproj_on_hostelse0),"
        in compact
    )
    # The vision switch and paravirtual pin scrub these vars from the child env.
    assert (
        "_fit_env_mmproj_scrubbed=bool(_pv_mmproj_unpinnableor_paravirtual_cpu_forced)" in compact
    )


def _allowance(mmproj_bytes, *, on_host):
    """The projector allowance, resolved at call time.

    Bound here rather than in a class body so a missing method fails the tests that
    rely on it instead of erroring out the whole module at import.
    """
    stub = SimpleNamespace(_MMPROJ_VRAM_SAFETY = LlamaCppBackend._MMPROJ_VRAM_SAFETY)
    return LlamaCppBackend._inherited_mmproj_soft_overhead(stub, mmproj_bytes, on_host = on_host)


def test_an_inherited_projector_carries_the_same_safety_allowance_as_a_resolved_one():
    """The resolved projector's allowance rides in _soft_overhead, which is gated on
    effective_is_vision -> launch_mmproj_path, so an env-only projector was charged its
    file size and nothing for the buffers the encoder really allocates.

    Same formula as the three resolved call sites, read off the constant so moving it
    moves both together.
    """
    projector = 8 * GIB
    expected = int(projector * (LlamaCppBackend._MMPROJ_VRAM_SAFETY - 1.0))

    assert _allowance(projector, on_host = False) == expected
    assert expected > 0
    assert _allowance(projector, on_host = True) == 0
    assert _allowance(0, on_host = False) == 0


def test_the_projector_allowance_flips_a_fit_that_only_looked_like_one(monkeypatch):
    """22 GiB of weights-plus-projector into a 24 GiB card looks like a fit while the
    8 GiB projector is charged raw. With the allowance the real footprint is 25.2 GiB,
    which does not fit, and understating it is the direction that hands the load a
    loader that cannot page. Host RAM is deliberately too small to cover the gap.
    """
    projector = 8 * GIB
    footprint = 22 * GIB
    allowance = _allowance(projector, on_host = False)

    assert _mode("linux", "nvidia_discrete", footprint, 512, monkeypatch) == FIT_MODE
    assert (
        _mode(
            "linux",
            "nvidia_discrete",
            footprint,
            512,
            monkeypatch,
            soft_overhead = allowance,
        )
        is None
    )


def test_the_launch_wires_the_projector_allowance_into_the_fit():
    """Reachability only: driving load_model this far needs a real GPU probe."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "_inherited_mmproj_soft_overhead" in compact
    assert "soft_overhead=_fit_soft_overhead" in compact


def test_pass_through_adapter_paths_follow_llama_cpp_parsing():
    from core.inference.llama_cpp import _sidecar_adapter_paths

    assert _sidecar_adapter_paths(["--lora", "/a.gguf,/b.gguf"]) == ["/a.gguf", "/b.gguf"]
    assert _sidecar_adapter_paths(["--lora-scaled=/a.gguf:0.5"]) == ["/a.gguf"]
    assert _sidecar_adapter_paths(["--control-vector", "/cv.gguf"]) == ["/cv.gguf"]
    assert _sidecar_adapter_paths(["--control-vector-scaled", "/cv.gguf:2"]) == ["/cv.gguf"]
    # Accumulated, not last-wins: every --lora handler push_back()s.
    assert _sidecar_adapter_paths(["--lora", "/a.gguf", "--lora", "/b.gguf"]) == [
        "/a.gguf",
        "/b.gguf",
    ]
    # Upstream string_split(item, ':') needs exactly two parts, so these throw.
    assert _sidecar_adapter_paths(["--lora-scaled", "/a.gguf"]) == []
    assert _sidecar_adapter_paths(["--lora-scaled", "C:/a.gguf:0.5"]) == []
    assert _sidecar_adapter_paths(["--lora-scaled", "/a.gguf:half"]) == []
    assert _sidecar_adapter_paths(["--lora-init-without-apply", "-ngl", "-1"]) == []
    assert _sidecar_adapter_paths([]) == []
    assert _sidecar_adapter_paths(None) == []


def test_the_fit_prices_pass_through_adapter_weights(tmp_path, monkeypatch):
    """A 3 GiB LoRA is the difference between an 18 GiB load fitting a 20 GiB card
    and not. llama.cpp allocates the adapter on the base tensor's own buffer type
    (llama-adapter.cpp) and never mmaps it, so those bytes are resident on top of a
    placement neither this fit nor common/fit.cpp ever charged for.
    """
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"")
    import os as _os

    with open(adapter, "wb") as fh:  # sparse, so the test costs no disk
        fh.truncate(3 * GIB)
    assert _os.stat(adapter).st_size == 3 * GIB

    stub = _Stub(1024)
    rows = [(0, 20 * 1024)]

    def _verdict(extras):
        return LlamaCppBackend._fit_derived_load_mode(
            stub,
            model_size = 18 * GIB,
            gpus = rows,
            avail_mib = 1024,
            extra_args = extras,
            env = {},
        )

    assert _verdict([]) == FIT_MODE
    assert _verdict(["--lora", str(adapter)]) is None
    assert _verdict([f"--lora-scaled={adapter}:0.5"]) is None
    assert _verdict(["--lora", f"{adapter},{adapter}"]) is None
    small = tmp_path / "small.gguf"
    small.write_bytes(b"x" * 4096)
    assert _verdict(["--lora", str(small)]) == FIT_MODE


def test_the_fit_prices_the_legacy_two_token_scaled_adapter(tmp_path, monkeypatch):
    """``--lora-scaled FNAME SCALE`` is the spelling older llama-servers declared,
    and Unsloth runs whatever binary it is pointed at. Reading only FNAME as the
    operand and dropping it for want of a colon prices the adapter at zero, which is
    the optimistic direction: a load that needs 21 GiB on a 20 GiB card would be
    handed a loader that cannot page.
    """
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    adapter = tmp_path / "adapter.gguf"
    with open(adapter, "wb") as fh:  # sparse, so the test costs no disk
        fh.truncate(3 * GIB)

    stub = _Stub(1024)

    def _verdict(extras):
        return LlamaCppBackend._fit_derived_load_mode(
            stub,
            model_size = 18 * GIB,
            gpus = [(0, 20 * 1024)],
            avail_mib = 1024,
            extra_args = extras,
            env = {},
        )

    assert _verdict([]) == FIT_MODE
    assert _verdict(["--lora-scaled", str(adapter), "0.5"]) is None
    assert _verdict(["--control-vector-scaled", str(adapter), "1"]) is None
    from core.inference.llama_cpp import _sidecar_adapter_paths

    assert _sidecar_adapter_paths(["--lora-scaled", "C:/a.gguf", "0.5"]) == ["C:/a.gguf"]
    assert _sidecar_adapter_paths(["--lora-scaled", "/a.gguf:0.5"]) == ["/a.gguf"]
    assert _sidecar_adapter_paths(["--lora-scaled", "/a.gguf"]) == []
    assert _sidecar_adapter_paths(["--lora-scaled", "/a.gguf", "--lora", "/b.gguf"]) == [
        "/b.gguf",
    ]


def test_an_unreadable_adapter_abstains(tmp_path, monkeypatch):
    """Engaged but unsized, the same answer every other unreadable term gives."""
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    stub = _Stub(1024)
    got = LlamaCppBackend._fit_derived_load_mode(
        stub,
        model_size = 8 * GIB,
        gpus = [(0, 24 * 1024)],
        avail_mib = 1024,
        extra_args = ["--lora", str(tmp_path / "missing.gguf")],
        env = {},
    )
    assert got is None


def test_adapter_flags_really_reach_the_child():
    """Premise test (holds either way): the fit only has to price these because the
    pass-through layer lets them through. It is a denylist, and none of the four
    adapter flags is on it."""
    from core.inference.llama_server_args import validate_extra_args
    for flag in ("--lora", "--lora-scaled", "--control-vector", "--control-vector-scaled"):
        assert validate_extra_args([flag, "/a.gguf:0.5"]) == [flag, "/a.gguf:0.5"]


def test_the_extras_own_load_mode_stands_the_fit_down():
    from core.inference.llama_server_args import extra_args_select_load_mode

    assert extra_args_select_load_mode(["--load-mode", "none"]) is True
    assert extra_args_select_load_mode(["-lm", "mmap"]) is True
    assert extra_args_select_load_mode(["--load-mode=mlock"]) is True
    assert extra_args_select_load_mode(["--no-mmap"]) is True
    assert extra_args_select_load_mode(["--mmap"]) is True
    assert extra_args_select_load_mode(["-ngl", "-1", "--temp", "0.7"]) is False
    assert extra_args_select_load_mode([]) is False
    assert extra_args_select_load_mode(None) is False


def test_a_chained_retry_cannot_eat_the_users_own_load_mode():
    """_without_subsequence removes the FIRST value match, so two identical pairs
    in one argv are two removals across two retry rungs -- the second taking the
    user's. The fit standing aside when the extras pick a mode is what keeps the
    strips the no-ops their docstrings claim to be.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B
    from core.inference.llama_cpp import _without_subsequence
    import inspect

    from core.inference.llama_server_args import (
        apply_load_mode_policy,
        extra_args_select_load_mode,
    )

    user = ["--load-mode", "none"]
    assert extra_args_select_load_mode(user)

    def two_rungs(requested):
        """The real policy call and the real strip, twice, as the retry does."""
        managed, extras = apply_load_mode_policy(
            list(user),
            supports_load_mode = True,
            weights_in_host_memory = True,
            requested_load_mode = requested,
        )
        fit_flags = list(managed) if requested else []
        argv = [*managed, "--temp", "0.7", *extras]
        return _without_subsequence(_without_subsequence(argv, fit_flags), fit_flags)

    # Emitting the fit's pair beside an identical user pair loses the user's flag.
    assert two_rungs("none") == ["--temp", "0.7"]
    assert two_rungs(None) == ["--temp", "0.7", "--load-mode", "none"]

    # Single call expression, so a reformat cannot break the whitespace-stripped pin.
    compact = "".join(inspect.getsource(B.load_model).split())
    assert "extra_args_select_load_mode(_mem_extras)" in compact
    assert "_fit_load_mode" in compact and "notload_mode" in compact
    assert "_fit_load_mode=None" in compact
    # Not pinned on apply_load_mode_policy: a multi-token run a reformat can wrap.


def test_the_fit_still_emits_when_the_extras_pick_nothing():
    """The guard is scoped to a real pick, not to the presence of any extras."""
    from core.inference.llama_server_args import apply_load_mode_policy, extra_args_select_load_mode

    extras = ["-ngl", "-1", "--temp", "0.7"]
    assert extra_args_select_load_mode(extras) is False
    managed, passed = apply_load_mode_policy(
        extras, supports_load_mode = True, requested_load_mode = FIT_MODE
    )
    assert managed == ["--load-mode", FIT_MODE]
    assert passed == extras


def test_a_stripped_load_mode_changes_what_the_reload_predicate_answers(monkeypatch):
    """The premise behind the two recomputes below, on the real predicates.

    Both rungs take the fit's ``--load-mode none`` back out, which returns the
    child to llama.cpp's auto, and auto maps (llama-model-loader.cpp derives
    use_mmap from mmap/mmap+mlock/auto). A record still describing the pre-retry
    argv therefore says "reserves RAM" about a child that maps, and under "Don't
    reserve system RAM" that is a full model reload the running server already
    satisfies.
    """
    from core.inference.llama_cpp import _without_subsequence
    import utils.model_memory_settings as mm

    from core.inference.llama_server_args import (
        memory_state_satisfies_settings,
        resolve_effective_memory_state,
    )

    launched = ["llama-server", "-m", "m.gguf", "--load-mode", FIT_MODE, "--flash-attn", "off"]
    stripped = _without_subsequence(launched, ["--load-mode", FIT_MODE])

    stale = resolve_effective_memory_state(launched, {})
    fresh = resolve_effective_memory_state(stripped, {})
    assert stale == (False, True)
    assert fresh == (False, False)

    monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: True)
    assert memory_state_satisfies_settings(stale, True) is False
    assert memory_state_satisfies_settings(fresh, True) is True


def test_the_recompute_reads_the_argv_not_the_managed_block():
    """Why both rungs recompute from the argv they are about to spawn.

    They descend from _last_spawn_cmd, which carries any page-lock
    _spawn_and_wait's --fit on retry appended to its own argv. Rebuilding from
    the Model Memory block instead would drop that lock and record an unlocked
    child that is in fact locked, which is the optimistic direction: no-reserve
    would read as satisfied while the reservation stands.
    """
    from core.inference.llama_cpp import _without_subsequence

    from core.inference.llama_server_args import resolve_effective_memory_state

    retried = ["llama-server", "-m", "m.gguf", "--load-mode", FIT_MODE, "--load-mode", "mmap+mlock"]
    stripped = _without_subsequence(retried, ["--load-mode", FIT_MODE])
    assert resolve_effective_memory_state(stripped, {}) == (True, False)
    assert resolve_effective_memory_state([], {}) == (False, False)


def test_the_no_flash_rung_recomputes_the_memory_record():
    """Reachability at the source, like the strip it sits next to: the arm only
    runs behind a real signal crash. Whitespace stripped, so a reformat that
    wraps the call cannot break the pin."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    # Anchor on the definition: the bare name also appears in earlier comments.
    helper = src[src.index("def _drop_fit_load_mode_for_no_flash(") :]
    helper = helper[: helper.index("def _spawn_and_wait")]
    assert "self._record_memory_state" in helper
    compact = "".join(helper.split())
    assert "self._record_memory_state(stripped" in compact


def test_the_cpu_projector_rung_recomputes_the_memory_record():
    """Same, for the projector respawn."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    arm = src[: src.index('"-mmproj-cpu"')]
    arm = arm[arm.rindex("_with_mmproj_offload_disabled") :]
    assert "self._record_memory_state" in arm
    compact = "".join(arm.split())
    assert "self._record_memory_state(_stripped_cpu_projector_cmd" in compact


def test_the_arch_crash_rung_records_from_the_argv_not_the_parts():
    """The record has to be recomputed from the stripped argv on this rung too.

    By the time the arch-crash retry strips the fit's pair, `cmd` may have been
    rebuilt as the CPU, device-gated, no-tensor-split or arch-retry argv, so
    _mem_managed + _mem_extras no longer add up to it. Rebuilding from the parts
    drops whatever those paths added, and dropping a page-lock records an
    unlocked child that is in fact locked -- the optimistic direction.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    from core.inference.llama_server_args import resolve_effective_memory_state

    parts = ["--mlock", "--temp", "0.7"]
    argv_after_retry = ["--mlock", "--temp", "0.7", "--no-mmap"]
    assert resolve_effective_memory_state(parts, {}) != resolve_effective_memory_state(
        argv_after_retry, {}
    )

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "self._record_memory_state(cmd,env)" in compact


def test_the_cpu_replay_and_the_launch_record_disagree(monkeypatch):
    """The premise, on the real functions rather than a hand-written argv.

    _prepare_cpu_fallback_launch takes the fit's pair out, so a record taken off
    the launch argv describes a reservation the replay does not make. Under
    "Don't reserve system RAM" that stale record is a full model reload the CPU
    server already satisfies.
    """
    from core.inference import llama_cpp as lc
    from unittest import mock
    import utils.model_memory_settings as mm

    from core.inference.llama_server_args import (
        memory_state_satisfies_settings,
        resolve_effective_memory_state,
    )

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._memory_dio_flags = []
    backend._fit_load_mode_flags = ["--load-mode", FIT_MODE]
    launched = ["llama-server", "-m", "model.gguf", "--load-mode", FIT_MODE]
    with (
        mock.patch.object(lc.LlamaCppBackend, "_is_vulkan_backend", return_value = True),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_replay", return_value = list(launched)),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_binary", return_value = "cpu-server"),
        mock.patch.object(
            lc.LlamaCppBackend,
            "_llama_server_env_for_binary",
            return_value = {lc._loader_path_var(): "/staged"},
        ),
    ):
        replay, _reason, _note = backend._prepare_cpu_fallback_launch(
            "llama-server", launched, {}, {}
        )

    stale = resolve_effective_memory_state(launched, {})
    fresh = resolve_effective_memory_state(replay, {})
    assert stale == (False, True)
    assert fresh == (False, False)

    monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: True)
    assert memory_state_satisfies_settings(stale, True) is False
    assert memory_state_satisfies_settings(fresh, True) is True


def test_the_crash_path_cpu_fallback_recomputes_the_memory_record():
    """Reachability at the source, like the rungs beside it: the arm runs only
    behind a real signal crash on an auto-selected Vulkan backend.

    From _last_spawn_cmd, not from the replay handed to the spawn: that is the
    argv that really started, so a page-lock _spawn_and_wait's own --fit retry
    appended is kept rather than recorded away.
    """
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = "".join(inspect.getsource(B.load_model).split())
    arm = src[src.index("_try_auto_vulkan_cpu_fallback") :]
    arm = arm[: arm.index("_apply_cpu_fallback_state")]
    assert "self._record_memory_state(_last_spawn_cmd,env)" in arm


def test_the_replayed_cpu_fallback_recomputes_the_memory_record():
    """Same, for a request that carries cpu_fallback and rebuilds the replay
    before anything spawns."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = "".join(inspect.getsource(B.load_model).split())
    arm = src[src.index("allow_manual_cpu=True") :]
    arm = arm[: arm.index("_apply_cpu_fallback_state")]
    assert "self._record_memory_state(cmd,env)" in arm


def _no_flash_fit_rewriter(
    extra_args,
    *,
    manual_gpu_layers = None,
    env = None,
):
    """The nested `_enable_managed_fit_for_no_flash` as a callable: it closes over
    load_model's locals, so compiling its source is the only way to run the REAL rewrite."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    from core.inference import llama_cpp as B_module
    from core.inference.llama_cpp import logger, _flag_name
    import inspect, textwrap

    src = inspect.getsource(B.load_model)
    start = src.index("                def _enable_managed_fit_for_no_flash(")
    end = src.index("                def ", start + 1)
    namespace = {
        "logger": logger,
        "_flag_name": _flag_name,
        "_placement_is_fitter_proof": B_module._placement_is_fitter_proof,
        "_user_fit_disabled": B_module._user_fit_disabled,
        "extra_args": extra_args,
        "_manual_gpu_layers": manual_gpu_layers,
        "env": {} if env is None else env,
    }
    exec(textwrap.dedent(src[start:end]), namespace)
    return namespace["_enable_managed_fit_for_no_flash"]


def test_the_no_flash_retry_re_enables_unsloths_own_fitter():
    """A managed `--fit off` is flipped back on for the respawn.

    The reserve is only safe because the respawn is re-placed, and it inherits the auto
    placement's `--fit off` that `_fit_off_retry_eligible` will not override, so without the
    flip the retry re-lands on the smaller-cache placement and OOMs.
    """
    rewrite = _no_flash_fit_rewriter(None)
    out = rewrite(["llama-server", "--fit", "off", "--flash-attn", "off"])
    assert out[out.index("--fit") + 1] == "on"
    on = ["llama-server", "--fit", "on"]
    assert rewrite(on) == on
    bare = ["llama-server", "--flash-attn", "off"]
    assert rewrite(bare) == bare


def test_the_no_flash_fit_flip_leaves_a_user_fit_alone():
    """The user's own `--fit off` survives the respawn, in both spellings: theirs wins by
    last-arg anyway, except where the only `--fit` present is theirs."""
    for user_tokens in (["--fit", "off"], ["--fit=off"], ["-fit", "off"]):
        rewrite = _no_flash_fit_rewriter(user_tokens)
        both = rewrite(["llama-server", "--fit", "off", *user_tokens])
        assert both == ["llama-server", "--fit", "off", *user_tokens]
        theirs = rewrite(["llama-server", *user_tokens])
        assert theirs == ["llama-server", *user_tokens]


def test_an_inherited_fit_off_survives_the_no_flash_retry():
    """LLAMA_ARG_FIT=off is the user's `--fit off`, and only manual mode scrubs it from the
    child env, so outside manual the flip would put `--fit on` on the CLI and beat it.

    `_user_fit_disabled` reads the same variable and already reserved the padded V cache for
    this load, so the respawn lands on a placement priced for it and needs no re-place.
    """
    off = ["llama-server", "--fit", "off", "--flash-attn", "off"]
    for value in ("off", "0", "false", "no", "disabled", " OFF "):
        rewrite = _no_flash_fit_rewriter(None, env = {"LLAMA_ARG_FIT": value})
        assert rewrite(off) == off, value

    for value in ("on", "1", ""):
        rewrite = _no_flash_fit_rewriter(None, env = {"LLAMA_ARG_FIT": value})
        assert rewrite(off)[off.index("--fit") + 1] == "on", value

    theirs = _no_flash_fit_rewriter(["--fit", "on"], env = {"LLAMA_ARG_FIT": "off"})
    assert theirs(off) == off


def test_the_no_flash_retry_re_places_at_both_respawns():
    """Both --flash-attn off arms re-enable the fitter, each before its own spawn. Checked
    at the source because these arms only run behind a real crash."""
    from core.inference.llama_cpp import LlamaCppBackend as B
    import inspect

    src = inspect.getsource(B.load_model)
    for label in ('label = "-noflash"', 'label = "-noflash-mtp"'):
        arm = src[: src.index(label)]
        arm = arm[arm.rindex("self._with_flash_attn_off(") :]
        assert "_enable_managed_fit_for_no_flash(_fa_cmd)" in arm
    assert src.count("_enable_managed_fit_for_no_flash(_fa_cmd)") == 2


def test_a_fixed_layer_count_keeps_the_no_flash_retry_on_its_placement():
    """Manual mode's `--fit off` is not an auto placement's, and must not be flipped.

    Manual emits `--gpu-layers N --fit off`, token for token what an auto placement emits,
    but `common_params_fit_impl` throws "n_gpu_layers already set by user"
    (common/fit.cpp:377) for any count but -1, so the retry keeps the fixed placement and
    the flip would only buy a reserve priced for a re-placement that cannot happen.
    """
    rewrite = _no_flash_fit_rewriter(None, manual_gpu_layers = 20)
    fixed = ["llama-server", "--gpu-layers", "20", "--fit", "off"]
    assert rewrite(fixed) == fixed

    assert _no_flash_fit_rewriter(None)(fixed) == fixed
    inherited = _no_flash_fit_rewriter(None, env = {"LLAMA_ARG_N_GPU_LAYERS": "20"})
    auto = ["llama-server", "--fit", "off"]
    assert inherited(auto) == auto
    default_count = ["llama-server", "--gpu-layers", "-1", "--fit", "off"]
    assert _no_flash_fit_rewriter(None)(default_count)[-1] == "on"


def test_the_reserve_holds_for_a_placement_the_fitter_will_not_move():
    """The other half of the same case, on the estimate rather than the argv: with no
    re-placement available the reserve prices the padded, f16-floored V."""
    from core.inference.llama_cpp import _placement_is_fitter_proof, _reserved_flash_attn_state

    assert _reserved_flash_attn_state(True, None, gpu_layers = 20, env = {}) is False
    assert _reserved_flash_attn_state(True, ["-ngl", "20"], env = {}) is False
    assert _reserved_flash_attn_state(True, None, env = {"LLAMA_ARG_N_GPU_LAYERS": "20"}) is False
    assert _reserved_flash_attn_state(True, None, gpu_layers = -1, env = {}) is True
    assert _reserved_flash_attn_state(True, ["-ngl", "-1"], env = {}) is True
    assert _reserved_flash_attn_state(True, None, env = {}) is True

    assert _placement_is_fitter_proof(["--gpu-layers=20"], env = {}) is True
    assert _placement_is_fitter_proof(["--n-gpu-layers", "0"], env = {}) is True
    assert _placement_is_fitter_proof(["-ngl", "auto"], env = {}) is False
    assert _placement_is_fitter_proof(None, gpu_layers = 0, env = {}) is True
    assert _placement_is_fitter_proof(None, env = {}) is False
