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
    # Read off the backend, so flipping the constant moves the whole matrix.
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
    # Single GPU: the launch's max(0, n - 1) is 0 there, so the term never moves it.
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "pipeline_overhead_bytes=(max(0,_fit_devices-1)*_pipeline_overhead_bytes)" in compact


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
    # Room for the weights and the headroom, but not for the graph on top.
    avail_mib = (2 * GIB + LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES // 2) // MIB + 2 * 1024

    def _fit(host_only):
        return LlamaCppBackend._fit_derived_load_mode(
            _Stub(avail_mib),
            model_size = 8 * GIB,
            host_only_bytes = host_only,
            gpus = [(0, 24 * 1024)],
            avail_mib = avail_mib,
        )

    # Weights and KV alone still read as a fit; the graph is what tips it.
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    # Captured BEFORE the VRAM budget drops it.
    assert "_cpu_draft_path=_mtp_draft_for_budgetif_draft_on_cpuelseNone" in compact
    assert compact.index("_cpu_draft_path=_mtp_draft_for_budget") < compact.index(
        "if_draft_on_cpu:_mtp_draft_for_budget=None"
    )
    # ... and charged to the footprint as a HOST-ONLY term (free VRAM cannot pay for
    # an allocation the child only ever makes in RAM), abstaining when unpriceable.
    assert "host_only_bytes=_cpu_draft_fit_bytesor0" in compact
    assert "or_cpu_draft_fit_bytesisNone" in compact


def test_a_weights_only_drafter_reserve_counts_as_unsized():
    """_flat_mtp_engages is "callback missing OR _mtp_kv_unsized", and the second
    arm is the one this call site has to keep: _estimate_mtp_overhead_bytes returns
    weights (plus any MLA/Mamba target copy) rather than None when the draft KV
    cannot be sized, so the callback exists and prices no draft KV at all. Charging
    that as a sized drafter understates a ctx-linear term the placement only covers
    with a flat cushion the footprint has no room for. Re-narrowing to
    "mtp_overhead_fn is None" absorbs the arm away, which is what it used to do."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

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
    # Same bytes on the GPU side of the ledger DO fit the card, which is the
    # difference this term draws.
    assert _fit(mtp_bytes = 3 * GIB) == FIT_MODE
    # And with RAM that can hold them, the host-only term is satisfied too.
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


# ------------------------------------------- the fitter's own VRAM margin


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

    # --fit off (Unsloth proved the placement): no fitter, no margin, the card pays.
    assert _fit(0.0) == FIT_MODE
    # --fit on: the last 1 GiB is the fitter's, so 0.5 GiB of weights spill, and
    # 2 GiB of RAM is exactly the headroom, so nothing is left to hold them.
    assert _fit(1024.0) is None


def test_the_margin_is_charged_per_device():
    """--fit-target is per device, so two cards keep two margins."""
    stub = _Stub(None)  # host RAM unreadable: the VRAM term has to settle it
    rows = [(0, 12 * 1024), (1, 12 * 1024)]
    # 23 GiB fits 24 GiB of raw free, but not 24 - 2 x 1 GiB.
    assert stub._fits_without_paging(23 * GIB, rows, avail_mib = None) is True
    assert stub._fits_without_paging(23 * GIB, rows, vram_margin_mib = 1024.0, avail_mib = None) is None
    assert stub._fits_without_paging(21 * GIB, rows, vram_margin_mib = 1024.0, avail_mib = None) is True


@pytest.mark.parametrize(
    "auto_fit,delta,supports,expected",
    [
        # Legacy auto path, untouched slider: llama.cpp's own default.
        (False, 0.0, True, 1024.0),
        # Manual + Auto starts from the tighter floor.
        (True, 0.0, True, 512.0),
        # A lowered VRAM budget raises the margin Unsloth asks the fitter to keep.
        (False, 2048.0, True, 3072.0),
        # ...and never below the 512 MiB floor.
        (True, -4096.0, True, 512.0),
        # Too old for the flag: the child still keeps its own 1024 MiB.
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
    # And the emitted flag is that same number, so the margin the child is told to
    # keep and the margin the fit refuses to spend cannot drift.
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    call = compact[compact.index("_fit_margin_mib=(") :]
    call = call[: call.index("exceptExceptionase:")]
    assert "max(self._fit_target_margin_mib(" in call
    # Not `use_fit`: the extras land after this launch's own --fit and llama.cpp is
    # last-wins, so only the effective state may zero the whole margin.
    assert "or0.0,)if_fitter_runselse0.0" in call
    # A fitter the extras turned back on gets llama.cpp's own default, not this
    # launch's budget, which _ctx_integrity_flags never emitted for it.
    assert "fit_target_delta_mib=_fit_target_delta_mibifuse_fitelse0.0" in call
    assert "supports_fit_target=use_fitand" in call
    # And a pass-through --fit-target raises the margin above either.
    assert "fit_target_margin_in(_fit_extras,_fit_env)or0.0" in call
    assert "fit_margin_mib=_fit_margin_mib" in call


def test_the_effective_fitter_state_reads_the_launchs_own_fit_flag():
    """`--fit on` in the extras beats the proved path's `--fit off` by last-arg.

    llama.cpp assigns `params.fit_params` on every occurrence (common/arg.cpp) and
    `-ngl -1` is its own default, which the fitter is free to lower (common/fit.cpp
    aborts only on a count the user really set). So the margin has to be charged.
    """
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B
    from core.inference.llama_server_args import fit_is_effectively_on

    compact = "".join(inspect.getsource(B.load_model).split())
    assert '_fitter_runs=fit_is_effectively_on(["--fit","on"ifuse_fitelse"off",' in compact

    # The synthetic prefix the call site builds, exercised directly.
    def _runs(
        use_fit,
        extras,
        env = None,
    ):
        return fit_is_effectively_on(["--fit", "on" if use_fit else "off", *extras], env)

    assert _runs(False, []) is False  # proved path: no fitter
    assert _runs(True, []) is True
    assert _runs(False, ["--fit", "on"]) is True  # the bug this closes
    assert _runs(True, ["--fit", "off"]) is False
    # The env twin cannot revive a fitter argv turned off: llama.cpp reads the
    # env BEFORE argv, and this launch always emits a --fit.
    assert _runs(False, [], {"LLAMA_ARG_FIT": "1"}) is False


@pytest.mark.parametrize(
    "extras,env,expected",
    [
        ([], None, None),
        (["--fit-target", "4096"], None, 4096.0),
        (["-fitt", "8192"], None, 8192.0),
        # A comma list is one margin per device; the fit credits every device it
        # counts, so the largest is the only safe price.
        (["--fit-target", "512,4096,1024"], None, 4096.0),
        # Upstream splits on both "," and "/" (common/arg.cpp), so a slash list is
        # a well-formed per-device margin, not an unreadable token.
        (["--fit-target", "4096/4096"], None, 4096.0),
        (["-fitt", "512/4096/1024"], None, 4096.0),
        (["--fit-target", "512,4096/1024"], None, 4096.0),
        ([], {"LLAMA_ARG_FIT_TARGET": "1024/8192"}, 8192.0),
        # Still unreadable when a slash-separated entry is: upstream's stoull
        # throws on the whole list.
        (["--fit-target", "512/lots"], None, None),
        # Last-wins over argv, like every other llama.cpp flag.
        (["--fit-target", "4096", "--fit-target", "512"], None, 512.0),
        # Env twin only when argv sets nothing.
        ([], {"LLAMA_ARG_FIT_TARGET": "2048"}, 2048.0),
        (["--fit-target", "512"], {"LLAMA_ARG_FIT_TARGET": "8192"}, 512.0),
        # Unreadable abstains: upstream would reject the whole list.
        (["--fit-target", "lots"], None, None),
        (["--fit-target", "512,lots"], None, None),
    ],
)
def test_a_pass_through_fit_target_is_the_margin_the_child_really_keeps(extras, env, expected):
    from core.inference.llama_server_args import fit_target_margin_in
    assert fit_target_margin_in(extras, env) == expected


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
        out, _reason, _note = backend._prepare_cpu_fallback_launch("llama-server", replay, {}, {})
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    retry = src[src.index("retrying once with --fit on so it can offload") :]
    retry = retry[: retry.index("_did_fit_retry = True")]
    assert "_fit_load_mode_flags" in retry
    assert "_without_subsequence" in retry
    # ...but the RECORD stays: this retry rewrites its own argv, not `cmd`, so the
    # arch-crash respawn below still has to be able to name the tokens `cmd` is
    # carrying. Clearing here would leave it stripping nothing.
    assert "self._fit_load_mode_flags = []" not in retry


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
    # The record is dropped ONLY where `cmd` itself was rewritten. Anywhere else it
    # would disarm this strip for a respawn that still carries the tokens.
    assert src.count("self._fit_load_mode_flags = []") == 1


def test_the_no_flash_retry_drops_the_fits_load_mode():
    """Both --flash-attn off respawns, which rewrite the footprint the fit priced.

    The mode is NOT gated on full offload: a partially offloaded "--fit on" launch
    that fits VRAM plus RAM carries it too, and on that launch the --fit on retry
    (gated on fully_gpu_offloaded) is not a second net. Checked at the source, like
    the retries above, because this arm only runs behind a real signal crash.
    """
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    # Both sites, each bounded at its own respawn so the strip is proved to happen
    # BEFORE it: the startup crash and the MTP first-decode crash.
    for label in ('label = "-noflash"', 'label = "-noflash-mtp"'):
        arm = src[: src.index(label)]
        arm = arm[arm.rindex("self._with_flash_attn_off(") :]
        assert "_drop_fit_load_mode_for_no_flash(_fa_cmd)" in arm
    # One helper, so the two arms cannot drift, and it strips without clearing:
    # the CPU fallback below still respawns from an argv carrying the tokens.
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
    # Flash attention really is off...
    assert "on" not in out[out.index("--flash-attn") : out.index("--flash-attn") + 2]
    # ...and BOTH axes were upcast, which is more KV than the fit charged.
    assert out[out.index("--cache-type-v") + 1] == "f16"
    assert out[out.index("--cache-type-k") + 1] == "f16"


# ------------------------------------ pass-through args that move the weights


# Every spelling llama.cpp accepts for "run this somewhere the planner did not put
# it", each appended AFTER Unsloth's own placement flags and so winning by last-arg.
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
    # Dropped credit, not a dropped answer: RAM that holds the whole load is a fit
    # wherever the count puts it.
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    call = src[src.index("_fit_extras = (") :]
    call = call[: call.index("except Exception as e:")]
    assert "extra_args = _fit_extras" in call
    # A gpu_ids pin drops the device flags from the emitted extras, so the fit has
    # to classify on the stripped copy rather than the request.
    assert "_strip_device_extra_args(extra_args)" in call
    # Every one of those overrides has an env twin read BEFORE argv, so the tokens
    # alone are half an answer, and the pin drops the twins with the flags.
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
    # 8 GiB against a 24 GiB card would fit VRAM outright, but 4 GiB of RAM
    # (minus the 2 GiB headroom) cannot hold it once the credit is void.
    assert _mode("linux", "nvidia_discrete", 8 * GIB, 4 * 1024, monkeypatch) is None
    # ... and the argv still wins over the env, so a GPU pick in the extras is
    # not condemned by a stale variable it overrides.
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
        # One card, one name: a restatement, not a narrowing.
        ("nvidia_discrete", ["--device", "CUDA0"], None, FIT_MODE),
        # Two cards, one name: the child opens one of them, so crediting both
        # pays for VRAM it never reaches.
        ("nvidia_multi", ["--device", "CUDA0"], None, None),
        ("nvidia_multi", ["-dev", "CUDA1"], None, None),
        # Both named: nothing is left out.
        ("nvidia_multi", ["--device", "CUDA0,CUDA1"], None, FIT_MODE),
        # Two entries, ONE card: parse_device_list never deduplicates, so both
        # resolve to the same device and one VRAM pool. A raw count would call it a
        # restatement and credit a card that is not there; distinct names do not.
        ("nvidia_multi", ["--device", "CUDA0,CUDA0"], None, None),
        ("nvidia_multi", ["-dev", "CUDA1,CUDA1,CUDA1"], None, None),
        # ...and on a one-card host the same list still names every credited
        # device, so it stays a restatement.
        ("nvidia_discrete", ["--device", "CUDA0,CUDA0"], None, FIT_MODE),
        # Last-wins, like every other llama.cpp flag.
        ("nvidia_multi", ["--device", "CUDA0", "--device", "CUDA0,CUDA1"], None, FIT_MODE),
        ("nvidia_multi", ["--device", "CUDA0,CUDA1", "--device", "CUDA0"], None, None),
        # No selection at all: unchanged, the credit stands.
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
    # Fits the pair (2 x 12 GiB) but not one card, and host RAM cannot hold it.
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
    # 10 GiB on one 12 GiB card of a two-card host, and host RAM cannot hold it,
    # so the answer turns entirely on whether the credit was voided.
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


# ------------------------------------------------- split policy overrides


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
        # One GPU holds everything, so the second card's credit is imaginary.
        ["--split-mode", "none"],
        ["-sm", "none"],
        # An explicit zero starves its device.
        ["--tensor-split", "1,0"],
        ["-ts", "1,0"],
        # A short list zero-fills the tail upstream, starving it just the same.
        ["--tensor-split", "1"],
        # Last-wins: the starving value is the one that reaches the child.
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
        # Every mode but "none" keeps all devices holding weights.
        ["--split-mode", "layer"],
        ["--split-mode", "row"],
        ["--split-mode", "tensor"],
        # A ratio that starves nobody is a restatement, not an override.
        ["--tensor-split", "1,1"],
        ["--tensor-split", "3,1"],
        # Upstream throws and the child never starts: nothing to misprice.
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
        ("1", 2, True),  # zero-filled tail
        ("1/0", 2, True),  # upstream splits on "/" too
        ("1,1", 2, False),
        ("3,1", 2, False),
        ("1,1,1", 2, False),  # only the credited prefix matters
        ("1,0", 1, False),  # single GPU: nothing to starve
    ],
)
def test_tensor_split_starvation_matches_upstream_parsing(raw, n_credited, expected):
    assert split_policy_starves_devices(["-ts", raw], n_credited) is expected


# ----------------------------------------------- vulkan device replacement


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
        # Same COUNT, different cards: a count cannot see this, a name can.
        ["--device", "Vulkan1,Vulkan2"],
        ["-dev", "Vulkan0,Vulkan2"],
        # Longer: adds a device this footprint charged no compute or overhead for.
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
        ["--device", "vulkan0,vulkan1"],  # ggml name-matching is not case sensitive here
        ["--device", "Vulkan1,Vulkan0"],  # order is not a placement change
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


# ------------------------------------------- CPU-pinned projector bytes


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


# ------------------------------------- mixed ROCm host: APU plus discrete card


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


# ---------------------------------------- a KV cache the launch pins to the host


@pytest.mark.parametrize(
    "extras,env",
    [
        (["--no-kv-offload"], None),
        (["-nkvo"], None),
        ([], {"LLAMA_ARG_KV_OFFLOAD": "0"}),
        # Last-wins, and the positive form is a real flag: -nkvo then -kvo is back on.
        (["-kvo", "-nkvo"], None),
        # The env twin parses BEFORE argv, so a false env survives an absent flag.
        ([], {"LLAMA_ARG_KV_OFFLOAD": "false"}),
        # And the NEGATIVE spelling, which get_value_from_env checks first and
        # forces falsey on PRESENCE, whatever it says (common/arg.cpp).
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "1"}),
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "0"}),
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": ""}),
        # It is read before the affirmative one and returns, so it wins outright.
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
        # A CLI re-enable beats a false env: llama.cpp reads the env first.
        (["-kvo"], {"LLAMA_ARG_KV_OFFLOAD": "0"}),
        (["-nkvo", "-kvo"], None),
        # Same for the negative spelling: env parses first, argv still wins.
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


# ---------------------------- the tensor-parallel compute buffer, replicated


def test_a_tensor_split_charges_the_replicated_compute_buffer():
    """Checked at the source: reaching this call needs a real multi-GPU probe.

    _plan_tensor_parallel admits devices against, and reserves per device,
    _estimate_compute_buffer_bytes(per_device_tensor = True). The fit has to charge
    the same thing: the layer-mode lump is ONE device's buffer at the smaller rate,
    and understating it on a pooled multi-GPU credit is the direction that claims a
    fit that is not there.
    """
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    # The tensor-mode rate is measured alongside the layer one...
    assert "_compute_buffer_tensor=self._estimate_compute_buffer_bytes(" in compact
    assert "per_device_tensor=True,)" in compact
    # ...and the same flat fallback covers missing dims.
    assert "if_compute_buffer_tensor<=0:" in compact
    # ...and it is what the fit charges per selected device, but only when the
    # tensor-parallel branch really planned this load (tp_tensor_split cannot say:
    # it stays None on a split llama.cpp is left to size itself).
    assert "_tp_planned=False" in compact
    assert "use_fit=False_tp_planned=True" in compact
    assert (
        "compute_buffer_flat=(_fit_devices*_compute_buffer_tensor"
        "if_tp_plannedelse_compute_buffer_pipeline)"
    ) in compact


def test_the_tensor_rate_really_exceeds_the_layer_one():
    """The under-charge the fix closes is not a rounding difference: at one slot
    the layer figure drops the vocab-width output buffer entirely."""

    class _Dims:
        # is_embedding_gguf is a read-only property on the real backend.
        _vocab_size = 151936
        _embedding_length = 5120
        is_embedding_gguf = False
        _DEFAULT_N_UBATCH = LlamaCppBackend._DEFAULT_N_UBATCH
        _COMPUTE_BUFFER_SAFETY = LlamaCppBackend._COMPUTE_BUFFER_SAFETY

    layer = LlamaCppBackend._estimate_compute_buffer_bytes(
        _Dims(), n_ubatch = 512, n_parallel = 1, per_device_tensor = False
    )
    tensor = LlamaCppBackend._estimate_compute_buffer_bytes(
        _Dims(), n_ubatch = 512, n_parallel = 1, per_device_tensor = True
    )
    assert tensor > layer
    # 4-GPU tensor split: what the fit used to charge vs what it now does.
    assert 4 * tensor - layer > GIB


def test_the_replicated_buffer_can_decide_the_fit(monkeypatch):
    """23 GiB across two 12 GiB cards with no host RAM is a fit until the second
    device's copy of the compute buffer is priced."""
    base = dict(platform = "linux", hw = "nvidia_multi", avail_mib = 0, monkeypatch = monkeypatch)
    assert _mode(footprint = 22 * GIB, compute_buffer_flat = GIB, **base) == FIT_MODE
    assert _mode(footprint = 22 * GIB, compute_buffer_flat = 3 * GIB, **base) is None


# ------------------------------- a drafter split between the GPU and the host


@pytest.mark.parametrize(
    "extras,env,n_draft_layers,expected",
    [
        # Nothing set: llama.cpp's own -1 default places the drafter whole.
        ([], None, 28, False),
        (["-ngld", "-1"], None, 28, False),
        # The whole-drafter case _extra_args_draft_offloaded_to_cpu already owns.
        (["-ngld", "0"], None, 28, False),
        # A real split: the layers past the count stay in host RAM.
        (["-ngld", "1"], None, 28, True),
        (["--gpu-layers-draft", "14"], None, 28, True),
        (["--spec-draft-ngl=4"], None, 28, True),
        (["-ngld", "1"], None, None, True),  # unreadable block count says nothing
        # Equal to the block count is still a split: i_gpu_start is 1 there, so
        # the first block and its KV stay on the host.
        (["-ngld", "28"], None, 28, True),
        # Only a count ABOVE the block count places the drafter whole.
        (["-ngld", "29"], None, 28, False),
        (["-ngld", "999"], None, 28, False),
        # The env twin the child reads before argv...
        ([], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "1"}, 28, True),
        # ...which argv last-wins over.
        (["-ngld", "-1"], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "1"}, 28, False),
        # Last-wins within argv too.
        (["-ngld", "1", "-ngld", "-1"], None, 28, False),
        (["-ngld", "-1", "-ngld", "1"], None, 28, True),
        # Upstream's std::stoi throws and the child never starts.
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

    # The boundary itself, from both spellings and the env twin.
    assert _draft_is_split_across_host(["-ngld", "28"], {}, n_draft_layers = 28) is True
    assert _draft_is_split_across_host(["--spec-draft-ngl=12"], {}, n_draft_layers = 12) is True
    assert (
        _draft_is_split_across_host([], {"LLAMA_ARG_N_GPU_LAYERS_DRAFT": "28"}, n_draft_layers = 28)
        is True
    )
    # One above it clears the whole drafter onto the card, so a real fit survives.
    assert _draft_is_split_across_host(["-ngld", "29"], {}, n_draft_layers = 28) is False
    # The same threshold the main model uses, asked of its own predicate over a
    # stub: a count equal to the block count is partial there too. A premise test,
    # here to pin the two thresholds together.
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    assert (
        "_draft_split_across_host=bool(_mtp_draft_for_budgetand_draft_is_split_across_host("
        in compact
    )
    # The same effective views the other overrides are classified on.
    assert "_draft_is_split_across_host(_fit_extras,_fit_env," in compact
    # Scoped by the drafter's own block count, off the backend the KV sizing above
    # already cached, so -ngld at or above it is not treated as a split.
    assert (
        'n_draft_layers=getattr(self._draft_backend_for(_mtp_draft_for_budget),"_n_layers",None,)'
        in compact
    )
    # And it lands on the abstain, not on a term.
    assert "or_draft_split_across_host" in compact


# ------------------------------------- round 7: terms and paths the fit missed


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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    # Bounded at the respawn, so the strip is proved to happen BEFORE it.
    arm = src[: src.index('label = "-mmproj-cpu"')]
    arm = arm[arm.rindex("_cpu_projector_cmd = self._with_mmproj_offload_disabled(") :]
    assert "self._fit_load_mode_flags" in arm
    assert "_without_subsequence(" in arm
    # The record stays: the fallbacks below respawn from an argv carrying these
    # tokens and have to be able to name them.
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
    stub = _Stub(1024)  # 1 GiB of RAM: nothing spills anywhere
    rows = [(0, (model + base) // MIB)]  # exactly holds the unsnapshotted load

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
    """...and the call site really passes it, while the closure default leaves the
    placement paths -- which price the snapshots by their own route -- unmoved."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "kv_cache_bytes=_kv_bytes(effective_ctx,_effective_ctx_checkpoints)," in compact
    assert "def_kv_bytes(ctx:int,ctx_checkpoints:int=0)->int:" in compact


def test_an_inherited_loader_mode_wins_over_the_fits_pick():
    """llama.cpp applies LLAMA_ARG_LOAD_MODE before argv, so the managed flag would
    beat an operator's inherited choice silently. The fit's mode stands aside for
    it, the way it stands aside for the per-model pick; a per-model pick still
    wins, and so does a hand-typed flag, by last-arg."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    arm = compact[
        compact.index("_fit_load_mode_env_view=dict(_mem_env)") : compact.index(
            "_resolved_load_mode=load_modeor_fit_load_mode"
        )
    ]
    # Asked of the child's environment AFTER the same scrub it gets, so a var a
    # Model Memory toggle drops vetoes nothing.
    assert "scrub_memory_env(_fit_load_mode_env_view)" in arm
    # Assert the CONDITIONS, not one rendering: the formatter is free to wrap this,
    # which breaks a single-string pin. Each clause is asserted on its own.
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    assert 'else(_fit_env.get("LLAMA_ARG_MMPROJ")or"").strip()' in compact
    assert 'else(_fit_env.get("LLAMA_ARG_MMPROJ_URL")or"").strip()' in compact
    # Unsized -> abstain, the same answer the other unreadable terms give.
    assert "_fit_env_mmproj_unsized=bool(_fit_env_mmproj_url)or(" in compact
    assert "if_fit_env_mmproj_unsized:" in compact
    assert "_fit_model_size=None" in compact
    assert "model_size=_fit_model_size," in compact
    # Host-resident when the resolved offload says CPU, pooled otherwise.
    assert (
        "mmproj_pinned_bytes=_mmproj_pinned_bytes+(_fit_env_mmproj_bytesif_fit_env_mmproj_on_hostelse0),"
        in compact
    )
    # Not charged where the child never sees it: the vision switch and the
    # paravirtual pin both scrub those vars out of the child environment.
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
    # Host arm: the resolved projector is priced raw there too, so no allowance.
    assert _allowance(projector, on_host = True) == 0
    # Nothing inherited, nothing to charge.
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    compact = "".join(inspect.getsource(B.load_model).split())
    assert "_inherited_mmproj_soft_overhead" in compact
    assert "soft_overhead=_fit_soft_overhead" in compact


# ------------------------- round 8: adapter sidecars the extras pass through


def test_pass_through_adapter_paths_follow_llama_cpp_parsing():
    from core.inference.llama_cpp import _sidecar_adapter_paths

    # parse_csv_row: one operand, many adapters.
    assert _sidecar_adapter_paths(["--lora", "/a.gguf,/b.gguf"]) == ["/a.gguf", "/b.gguf"]
    # The inline spelling, and the FNAME:SCALE tail the -scaled flags carry.
    assert _sidecar_adapter_paths(["--lora-scaled=/a.gguf:0.5"]) == ["/a.gguf"]
    assert _sidecar_adapter_paths(["--control-vector", "/cv.gguf"]) == ["/cv.gguf"]
    assert _sidecar_adapter_paths(["--control-vector-scaled", "/cv.gguf:2"]) == ["/cv.gguf"]
    # ACCUMULATED, not last-wins: every handler push_back()s, so a second --lora
    # adds to the first. Pricing only the last would understate the load.
    assert _sidecar_adapter_paths(["--lora", "/a.gguf", "--lora", "/b.gguf"]) == [
        "/a.gguf",
        "/b.gguf",
    ]
    # string_split(item, ':') demands exactly two parts, so these are upstream's own
    # throw: the child never starts and there is no placement to misprice.
    assert _sidecar_adapter_paths(["--lora-scaled", "/a.gguf"]) == []
    assert _sidecar_adapter_paths(["--lora-scaled", "C:/a.gguf:0.5"]) == []
    assert _sidecar_adapter_paths(["--lora-scaled", "/a.gguf:half"]) == []
    # Nothing else on the command line is an adapter.
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

    # The same load, with and without the adapter the extras pass through.
    assert _verdict([]) == FIT_MODE
    assert _verdict(["--lora", str(adapter)]) is None
    assert _verdict([f"--lora-scaled={adapter}:0.5"]) is None
    # Charged once per occurrence: two references to the same file is two loads.
    assert _verdict(["--lora", f"{adapter},{adapter}"]) is None
    # A small adapter still fits, so the fit is not simply refused on the flag.
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
    # A drive letter is a filename, not a scale, once the scale is its own token.
    from core.inference.llama_cpp import _sidecar_adapter_paths

    assert _sidecar_adapter_paths(["--lora-scaled", "C:/a.gguf", "0.5"]) == ["C:/a.gguf"]
    # Today's syntax is unchanged, and a lone FNAME with no scale after it is still
    # upstream's own throw rather than a load to price.
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


# ----------------- round 8: a user's own load mode across a retry CHAIN


def test_the_extras_own_load_mode_stands_the_fit_down():
    from core.inference.llama_server_args import extra_args_select_load_mode

    # The enum, both spellings, value and inline forms.
    assert extra_args_select_load_mode(["--load-mode", "none"]) is True
    assert extra_args_select_load_mode(["-lm", "mmap"]) is True
    assert extra_args_select_load_mode(["--load-mode=mlock"]) is True
    # The legacy flags the enum replaced, which is what this launch emits on a
    # pre-enum binary, so a user's own copy collides there the same way.
    assert extra_args_select_load_mode(["--no-mmap"]) is True
    assert extra_args_select_load_mode(["--mmap"]) is True
    # Nothing else on the command line picks a loader mode.
    assert extra_args_select_load_mode(["-ngl", "-1", "--temp", "0.7"]) is False
    assert extra_args_select_load_mode([]) is False
    assert extra_args_select_load_mode(None) is False


def test_a_chained_retry_cannot_eat_the_users_own_load_mode():
    """_without_subsequence removes the FIRST value match, so two identical pairs
    in one argv are two removals across two retry rungs -- the second taking the
    user's. The fit standing aside when the extras pick a mode is what keeps the
    strips the no-ops their docstrings claim to be.
    """
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B
    from core.inference.llama_cpp import _without_subsequence
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
        # What load_model records as the fit's own flags: only when the fit chose.
        fit_flags = list(managed) if requested else []
        argv = [*managed, "--temp", "0.7", *extras]
        return _without_subsequence(_without_subsequence(argv, fit_flags), fit_flags)

    # Emitting the fit's pair alongside an identical user pair is what loses it:
    # two rungs, two first-match removals, and the typed flag is the second.
    assert two_rungs("none") == ["--temp", "0.7"]
    # Standing aside leaves one pair, so both rungs are the no-op the docstrings
    # claim, and the flag the user typed is still there on the second respawn.
    assert two_rungs(None) == ["--temp", "0.7", "--load-mode", "none"]

    # Reachability of the guard in load_model. A single call expression, so a
    # reformat that wraps the `if` cannot break it once whitespace is stripped.
    compact = "".join(inspect.getsource(B.load_model).split())
    # Assert each clause on its own -- the formatter is free to wrap the `if`.
    assert "extra_args_select_load_mode(_mem_extras)" in compact
    assert "_fit_load_mode" in compact and "notload_mode" in compact
    assert "_fit_load_mode=None" in compact
    # The view is pinned by the call above (_mem_extras, as Model Memory left them)
    # and deliberately NOT by a second pin on the apply_load_mode_policy call: that
    # is a multi-token run a reformat can wrap, the shape that broke CI last round.


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


# ------------------ round 11: the record has to follow the argv on every rung


def test_a_stripped_load_mode_changes_what_the_reload_predicate_answers(monkeypatch):
    """The premise behind the two recomputes below, on the real predicates.

    Both rungs take the fit's ``--load-mode none`` back out, which returns the
    child to llama.cpp's auto, and auto maps (llama-model-loader.cpp derives
    use_mmap from mmap/mmap+mlock/auto). A record still describing the pre-retry
    argv therefore says "reserves RAM" about a child that maps, and under "Don't
    reserve system RAM" that is a full model reload the running server already
    satisfies.
    """
    import utils.model_memory_settings as mm
    from core.inference.llama_cpp import _without_subsequence
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
    # The stale record is the pessimistic direction: a reload that buys nothing,
    # never a reservation left standing.
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
    # The managed block alone knows nothing about that appended lock.
    assert resolve_effective_memory_state([], {}) == (False, False)


def test_the_no_flash_rung_recomputes_the_memory_record():
    """Reachability at the source, like the strip it sits next to: the arm only
    runs behind a real signal crash. Whitespace stripped, so a reformat that
    wraps the call cannot break the pin."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    helper = src[src.index("_drop_fit_load_mode_for_no_flash") :]
    # Bounded at the next definition, so this proves the recompute is in the
    # helper and not merely somewhere later in load_model.
    helper = helper[: helper.index("_spawn_and_wait")]
    assert "self._memory_state" in helper
    compact = "".join(helper.split())
    assert "resolve_effective_memory_state(stripped" in compact


def test_the_cpu_projector_rung_recomputes_the_memory_record():
    """Same, for the projector respawn."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = inspect.getsource(B.load_model)
    arm = src[: src.index('"-mmproj-cpu"')]
    arm = arm[arm.rindex("_with_mmproj_offload_disabled") :]
    assert "self._memory_state" in arm
    compact = "".join(arm.split())
    assert "resolve_effective_memory_state(_stripped_cpu_projector_cmd" in compact


def test_the_arch_crash_rung_records_from_the_argv_not_the_parts():
    """The record has to be recomputed from the stripped argv on this rung too.

    By the time the arch-crash retry strips the fit's pair, `cmd` may have been
    rebuilt as the CPU, device-gated, no-tensor-split or arch-retry argv, so
    _mem_managed + _mem_extras no longer add up to it. Rebuilding from the parts
    drops whatever those paths added, and dropping a page-lock records an
    unlocked child that is in fact locked -- the optimistic direction.
    """
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B
    from core.inference.llama_server_args import resolve_effective_memory_state

    # The two inputs genuinely disagree, so which one is used is observable.
    parts = ["--mlock", "--temp", "0.7"]
    argv_after_retry = ["--mlock", "--temp", "0.7", "--no-mmap"]
    assert resolve_effective_memory_state(parts, {}) != resolve_effective_memory_state(
        argv_after_retry, {}
    )

    # Reachability: single call expression, whitespace stripped, so a reformat
    # that wraps the call cannot break it.
    compact = "".join(inspect.getsource(B.load_model).split())
    assert "resolve_effective_memory_state(cmd,env)" in compact


# ------------------ round 12: the CPU fallback is a rung that strips the loader


def test_the_cpu_replay_and_the_launch_record_disagree(monkeypatch):
    """The premise, on the real functions rather than a hand-written argv.

    _prepare_cpu_fallback_launch takes the fit's pair out, so a record taken off
    the launch argv describes a reservation the replay does not make. Under
    "Don't reserve system RAM" that stale record is a full model reload the CPU
    server already satisfies.
    """
    from unittest import mock

    import utils.model_memory_settings as mm
    from core.inference import llama_cpp as lc
    from core.inference.llama_server_args import (
        memory_state_satisfies_settings,
        resolve_effective_memory_state,
    )

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
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
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    # Whitespace stripped before slicing, so a reformat that rewraps any of it
    # cannot break the pin.
    src = "".join(inspect.getsource(B.load_model).split())
    arm = src[src.index("_try_auto_vulkan_cpu_fallback") :]
    # Bounded at the normalisation that closes the recovery, so this proves the
    # recompute is in the arm and not merely somewhere later in load_model.
    arm = arm[: arm.index("_apply_cpu_fallback_state")]
    assert "resolve_effective_memory_state(_last_spawn_cmd,env)" in arm


def test_the_replayed_cpu_fallback_recomputes_the_memory_record():
    """Same, for a request that carries cpu_fallback and rebuilds the replay
    before anything spawns."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend as B

    src = "".join(inspect.getsource(B.load_model).split())
    # allow_manual_cpu is what distinguishes this eligibility check from the
    # crash path's, which reads the same helper without it.
    arm = src[src.index("allow_manual_cpu=True") :]
    arm = arm[: arm.index("_apply_cpu_fallback_state")]
    assert "resolve_effective_memory_state(cmd,env)" in arm
