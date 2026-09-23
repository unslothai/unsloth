# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Simulation matrix for the auto tensor-parallel split fallback (PR #10884).

The PR makes an AUTO-mode tensor-parallel load fall back to the user's
``tensor_split`` when the planner decides an even share fits and returns
``None``, and records the emitted ratio in ``_auto_tensor_split`` so a later
request with a different ratio reloads.

This file simulates the hardware the change can meet, on a CPU-only host, by
driving the real ``LlamaCppBackend`` against faked GPU inventories: the vendor
(CUDA / Vulkan, which is how AMD and Intel arrive), the card count, asymmetric
cards, cards whose driver reports no total, and the paravirtualised Metal
device macOS hands a VM. The reload-deduplication half is driven through the
real ``adopt_load_intent_if_matched``, because that is the method a second
``/api/inference/load`` actually reaches.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

from core.inference.llama_cpp import GgufLoadIntent  # noqa: E402

GIB = 1024**3


def _tp_backend(
    tmp_path: Path,
    *,
    memory,
    model_gib: float = 1.0,
    vulkan: bool = False,
    reserve_mib: int = 256,
):
    """A backend whose placement inputs are fully determined.

    KV and the context-linear compute buffer are zeroed so the only thing the
    planner and the new budget check weigh is the model size against the cards.
    That is what makes each case below a statement about ONE variable.
    """
    backend, gguf = _backend(tmp_path, vulkan=vulkan, memory=memory)
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: 0
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._get_gguf_size_bytes = lambda _path: int(model_gib * GIB)
    backend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB = reserve_mib
    return backend, gguf


def _auto_tp(backend, gguf, **kwargs):
    """An auto-mode tensor-parallel load, the shape the PR is about."""
    params = dict(
        gpu_memory_mode="auto",
        tensor_parallel=True,
        n_ctx=4096,
    )
    params.update(kwargs)
    return _launch(backend, gguf, **params)["cmd"]


def _flag(cmd, name):
    """The value of ``name`` in an argv, or None. Refuses a duplicate."""
    hits = [i for i, tok in enumerate(cmd) if tok == name]
    assert len(hits) <= 1, f"{name} appears {len(hits)} times in {cmd}"
    if not hits:
        return None
    return cmd[hits[0] + 1]


def _split_values(cmd):
    raw = _flag(cmd, "--tensor-split")
    if raw is None:
        return None
    return [float(part) for part in raw.split(",")]


def _normalized(values):
    total = sum(values)
    return tuple(round(v / total, 9) for v in values)


# --------------------------------------------------------------------------
# 1. The fix itself, across card counts, vendors and ratios.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("vulkan", [False, True], ids=["cuda", "vulkan"])
@pytest.mark.parametrize(
    "memory, ratio",
    [
        # Two 24GB cards, the ordinary desktop pair.
        ([(0, 24_000, 24_000), (1, 24_000, 24_000)], [3, 1]),
        # Two T4s, which is what a Kaggle session is.
        ([(0, 15_360, 15_360), (1, 15_360, 15_360)], [3, 1]),
        # Reversed, so an off-by-order bug cannot pass both.
        ([(0, 24_000, 24_000), (1, 24_000, 24_000)], [1, 3]),
        # An even ratio, explicitly asked for rather than defaulted.
        ([(0, 24_000, 24_000), (1, 24_000, 24_000)], [1, 1]),
        # Three and four cards.
        ([(i, 24_000, 24_000) for i in range(3)], [2, 1, 1]),
        ([(i, 24_000, 24_000) for i in range(4)], [4, 3, 2, 1]),
        # Mismatched cards, the #10355 complaint exactly: a slow small card
        # that the user wants to carry less.
        ([(0, 11_000, 11_000), (1, 24_000, 24_000)], [1, 3]),
    ],
)
def test_the_user_ratio_reaches_llama_server_in_auto_mode(tmp_path, memory, ratio, vulkan):
    """The regression in #10355: auto mode dropped the user's ratio."""
    backend, gguf = _tp_backend(tmp_path, memory=memory, vulkan=vulkan)
    cmd = _auto_tp(
        backend,
        gguf,
        tensor_split=list(ratio),
        gpu_ids=[idx for idx, *_ in memory],
    )

    assert _flag(cmd, "--split-mode") == "tensor"
    assert _split_values(cmd) is not None, "the user ratio was dropped again"
    assert _normalized(_split_values(cmd)) == pytest.approx(
        _normalized([float(x) for x in ratio]), abs=1e-6
    )


def test_the_emitted_token_is_plain_decimal(tmp_path):
    """llama.cpp splits --tensor-split on commas and parses each field. A token
    in scientific notation, or one carrying a locale decimal comma, would either
    be read as a different number or split into extra fields."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    cmd = _auto_tp(backend, gguf, tensor_split=[0.75, 0.25], gpu_ids=[0, 1])
    raw = _flag(cmd, "--tensor-split")
    assert raw is not None
    assert raw.count(",") == 1, f"{raw!r} would parse as more fields than there are GPUs"
    for part in raw.split(","):
        assert "e" not in part.lower(), f"{part!r} is scientific notation"
        float(part)


@pytest.mark.parametrize(
    "ratio",
    [
        [1_000_000, 1],  # a ratio is relative; this is legal and means 1e6:1
        [0.0000004, 0.0000001],  # ...and so is this, which means 4:1
        [1, 3],
    ],
)
def test_an_extreme_ratio_survives_formatting(tmp_path, ratio):
    """llama.cpp normalizes the list, so only the proportion has to survive.

    Two ways it can fail to: %g renders 1000000 as "1e+06", and a fixed-point
    rendering collapses a ratio of very small numbers to "0,0", which is a
    division by zero once llama.cpp normalizes it.
    """
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    cmd = _auto_tp(backend, gguf, tensor_split=list(ratio), gpu_ids=[0, 1])
    raw = _flag(cmd, "--tensor-split")
    if raw is None:
        pytest.skip("the budget check declined this ratio; formatting is not reached")

    assert raw.count(",") == len(ratio) - 1, f"{raw!r} parses as the wrong field count"
    values = [float(part) for part in raw.split(",")]
    assert sum(values) > 0, f"{raw!r} normalizes to a division by zero"
    assert _normalized(values) == pytest.approx(
        _normalized([float(x) for x in ratio]), abs=1e-6
    ), f"{raw!r} is not the proportion that was asked for"


# --------------------------------------------------------------------------
# 2. Ratios that must NOT be forwarded.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio, why",
    [
        ([3, 1, 1], "three weights for two cards"),
        ([3], "one weight for two cards"),
        ([0, 0], "a total of zero"),
        ([-1, -1], "negatives, which sanitize to zero"),
        ([float("nan"), 1], "a NaN weight"),
        ([float("inf"), 1], "an infinite weight"),
    ],
)
def test_a_degenerate_ratio_is_dropped_not_forwarded(tmp_path, ratio, why):
    """A bad ratio must leave llama.cpp on its own default, never reach it."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    cmd = _auto_tp(backend, gguf, tensor_split=list(ratio), gpu_ids=[0, 1])
    assert _flag(cmd, "--split-mode") == "tensor", why
    raw = _flag(cmd, "--tensor-split")
    if raw is not None:
        for part in raw.split(","):
            value = float(part)
            assert math.isfinite(value) and value >= 0.0, f"{raw!r} ({why})"


def test_a_ratio_that_overshoots_one_card_is_dropped(tmp_path):
    """The planner's own budget, applied to the user's ratio. A 25GB model at
    3:1 puts ~19GB on a 16GB card."""
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 16_000, 16_000), (1, 16_000, 16_000)],
        model_gib=25,
    )
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert _flag(cmd, "--split-mode") == "tensor"
    assert _flag(cmd, "--tensor-split") is None


def test_the_budget_check_agrees_with_the_planner_on_an_even_share(tmp_path):
    """The check's whole claim is that it applies "the same per-device budget
    the planner uses". Then an EVEN ratio must be accepted exactly when the
    planner's own even-share rule says the load fits -- otherwise a user who
    types 1,1 is told their ratio does not fit a machine the planner just
    decided to split evenly, which is #10355 wearing a different hat.

    Sized to sit in the gap: the context compute buffer is real here, and
    charging it once (the planner's rule) accepts while charging it twice
    rejects.
    """
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    mib = 1024 * 1024
    cc_per_device_mib = 1024
    backend._compute_buffer_ctx_bytes = lambda *a, **k: cc_per_device_mib * mib

    gpus = [(0, 24_000), (1, 24_000)]
    total_by_idx = {0: 24_000, 1: 24_000}
    # usable = 24000 - max(0.03*24000, min(512, 0.03*24000)) = 23280 MiB per card,
    # reserve = 256, so the planner's ceiling is 23024 MiB a card.
    model_mib = 43_452
    even_share = (model_mib + 2 * cc_per_device_mib) / 2
    assert 22_000 < even_share < 23_024, "the case has drifted out of the gap it tests"

    assert (
        backend._tensor_split_fits_budget(
            [1.0, 1.0],
            gpus,
            [0, 1],
            model_mib * mib,
            4096,
            n_ubatch=512,
            total_by_idx=total_by_idx,
            vram_fraction=0.97,
        )
        is True
    )


def test_the_context_buffer_is_charged_flat_not_by_the_ratio(tmp_path):
    """The context-linear compute buffer is REPLICATED on every device: each one
    allocates the whole thing whatever weight it carries. Distributing the
    aggregate by the ratio instead is the same arithmetic only at an even share,
    and away from it the high-weight card is charged nearly twice the buffer --
    so a ratio the planner's own rule accepts is refused, which for a user who
    typed one is #10355 again. Sized to sit exactly in that gap: the planner
    accepts 1:9 here, the aggregate form does not.
    """
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    mib = 1024 * 1024
    cc_per_device_mib = 6 * 1024
    backend._compute_buffer_ctx_bytes = lambda *a, **k: cc_per_device_mib * mib
    model_mib = 14 * 1024

    # usable - reserve = 23024 MiB a card, as in the even-share cell above.
    ceiling_mib = 23_024
    assert model_mib / 2 + cc_per_device_mib < ceiling_mib, "the planner would not have"
    assert 0.9 * model_mib < ceiling_mib - cc_per_device_mib, "charged flat, 1:9 fits"
    assert 0.9 * (model_mib + 2 * cc_per_device_mib) > ceiling_mib, "charged by ratio, it does not"

    assert (
        backend._tensor_split_fits_budget(
            [1.0, 9.0],
            [(0, 24_000), (1, 24_000)],
            [0, 1],
            model_mib * mib,
            4096,
            n_ubatch=512,
            total_by_idx={0: 24_000, 1: 24_000},
            vram_fraction=0.97,
        )
        is True
    )


def test_the_budget_check_still_refuses_what_does_not_fit(tmp_path):
    """The other side of the rule above: relaxing the double charge must not
    turn the check into one that accepts anything."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    mib = 1024 * 1024
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0

    assert (
        backend._tensor_split_fits_budget(
            [9.0, 1.0],
            [(0, 24_000), (1, 24_000)],
            [0, 1],
            40_000 * mib,
            4096,
            n_ubatch=512,
            total_by_idx={0: 24_000, 1: 24_000},
            vram_fraction=0.97,
        )
        is False
    )


def test_the_budget_check_fails_closed_on_an_unsurveyed_card(tmp_path):
    """gpu_indices and the tensor-parallel survey can disagree. Pricing one
    against the other used to raise KeyError out of load_model."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    assert (
        backend._tensor_split_fits_budget(
            [1.0, 1.0],
            [(0, 24_000)],
            [0, 1],
            1024,
            4096,
            total_by_idx={0: 24_000},
        )
        is False
    )


def test_manual_mode_is_untouched_by_the_new_branch(tmp_path):
    """Manual mode has always emitted the ratio through its own path. The new
    elif is guarded on `gpu_memory_mode != "manual"`, so this must be
    byte-identical to its pre-PR behaviour."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    cmd = _launch(
        backend,
        gguf,
        gpu_memory_mode="manual",
        gpu_layers=99,
        tensor_parallel=True,
        tensor_split=[3, 1],
        gpu_ids=[0, 1],
        n_ctx=4096,
    )["cmd"]
    assert _flag(cmd, "--split-mode") == "tensor"
    assert _flag(cmd, "--tensor-split") == "3,1"
    assert backend._auto_tensor_split is None, "manual mode must not record an auto ratio"


# --------------------------------------------------------------------------
# 3. Hardware shapes where tensor parallelism is not available at all.
# --------------------------------------------------------------------------


def test_a_single_gpu_never_gets_a_split(tmp_path):
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000)])
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0])
    assert _flag(cmd, "--tensor-split") is None


def test_no_gpu_at_all_does_not_raise(tmp_path):
    """CPU-only. The new code divides by len(gpu_indices) and by sum(split);
    neither may be reached with a zero."""
    backend, gguf = _tp_backend(tmp_path, memory=[])
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1])
    assert _flag(cmd, "--tensor-split") is None


def test_cards_whose_driver_reports_no_total(tmp_path):
    """Some Vulkan drivers report free memory without a total. The new budget
    check reads `total_by_idx.get(idx, 0)`, so this is the shape that exercises
    the zero-total branch of _vram_usable_mib."""
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 24_000, 0), (1, 24_000, 0)],
        vulkan=True,
    )
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert _flag(cmd, "--split-mode") == "tensor"


def test_paravirtual_metal_never_launches_a_tensor_split(tmp_path, monkeypatch):
    """macOS in a VM pins to CPU and overrides --split-mode. The PR must not
    reintroduce a split there."""
    import core.inference.llama_cpp as llama_cpp

    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: True)
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert _flag(cmd, "--tensor-split") is None


# --------------------------------------------------------------------------
# 3b. The placement planner's own recovery arm.
#
# `load_model` prices the load inside one long `try`, and its `except` arm is a
# designed degradation: it drops the plan, sets `--fit on` and launches anyway.
# It resets `tp_tensor_split` to None but deliberately NOT `tensor_parallel` --
# which is exactly the state the new fallback fires in, with `gpu_indices`
# rebuilt from a WIDER set than the `tp_gpus` the planner had filtered, or not
# rebuilt at all. Three shapes, all of which launched cleanly before the PR.
# --------------------------------------------------------------------------


def _planner_raises(backend):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated GPU selection failure")

    backend._plan_tensor_parallel = _boom


def test_a_failed_plan_still_launches_on_vulkan(tmp_path):
    """Vulkan rebuilds gpu_indices from every DETECTED device, so it can name a
    card the planner's reserve filter had dropped."""
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 24_000, 24_000), (1, 24_000, 24_000), (2, 200, 24_000)],
        vulkan=True,
    )
    _planner_raises(backend)
    cmd = _auto_tp(backend, gguf, tensor_split=[2, 1, 1])
    assert cmd, "the load must still launch, as it did before the fallback existed"


def test_a_failed_plan_still_launches_when_gpu_ids_were_pinned(tmp_path):
    """The CUDA / ROCm shape: gpu_indices comes back as sorted(gpu_ids), which
    can include a card the tensor-parallel reserve filter excluded."""
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 24_000, 24_000), (1, 24_000, 24_000), (2, 200, 24_000)],
    )
    _planner_raises(backend)
    cmd = _auto_tp(backend, gguf, tensor_split=[2, 1, 1], gpu_ids=[0, 1, 2])
    assert cmd


def test_a_plan_that_fails_before_the_gpu_survey_still_launches(tmp_path):
    """The earliest throw: nothing the fallback reads has been bound yet."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])

    def _boom(_path):
        raise RuntimeError("simulated GGUF read failure")

    backend._get_gguf_size_bytes = _boom
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert cmd


# --------------------------------------------------------------------------
# 4. Reload deduplication -- the half that can regress an already-working path.
# --------------------------------------------------------------------------


def _intent(gguf, **kwargs):
    params = dict(
        gguf_path=str(gguf),
        model_identifier="test",
        gpu_memory_mode="auto",
        tensor_parallel=True,
        n_ctx=4096,
    )
    params.update(kwargs)
    return GgufLoadIntent(**params)


def _reuses(backend, gguf, **kwargs):
    """Whether a second identical /load would reuse the running server."""
    return backend.adopt_load_intent_if_matched(_intent(gguf, **kwargs))


def test_an_identical_auto_request_with_a_ratio_reuses_the_server(tmp_path):
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert _reuses(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1]) is True


def test_a_changed_ratio_does_not_reuse_the_server(tmp_path):
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert _reuses(backend, gguf, tensor_split=[1, 3], gpu_ids=[0, 1]) is False


def test_an_identical_auto_request_with_no_ratio_reuses_the_server(tmp_path):
    """The plain auto tensor-parallel load: the user set no ratio at all."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    _auto_tp(backend, gguf, gpu_ids=[0, 1])
    assert _reuses(backend, gguf, gpu_ids=[0, 1]) is True


def test_a_planner_weighted_split_still_reuses_the_server(tmp_path):
    """THE REGRESSION THIS FILE EXISTS FOR.

    When the model does not fit evenly the planner returns its OWN weighted
    split, in MiB, and no user ratio is involved. If the recorded auto ratio is
    the planner's output while the comparison is against the user's request,
    every repeat of an identical request mismatches and reloads a
    multi-gigabyte model -- forever, because the next launch records the same
    planner output again. This is an ordinary, already-working, pre-PR path.
    """
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 24_000, 24_000), (1, 16_000, 16_000)],
        model_gib=30,
    )
    cmd = _auto_tp(backend, gguf, gpu_ids=[0, 1])
    assert (
        _flag(cmd, "--tensor-split") is not None
    ), "this case is only meaningful when the planner emitted its own split"
    assert _reuses(backend, gguf, gpu_ids=[0, 1]) is True


def test_a_rejected_user_ratio_still_reuses_the_server(tmp_path):
    """The other half of the same shape: the budget check declined the ratio,
    so nothing was emitted. Repeating the identical request must not reload --
    the second launch would decline it identically."""
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 16_000, 16_000), (1, 16_000, 16_000)],
        model_gib=25,
    )
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert _flag(cmd, "--tensor-split") is None
    assert _reuses(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1]) is True


def test_an_equivalent_ratio_reuses_the_server(tmp_path):
    """3:1 and 6:2 are the same instruction to llama.cpp."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert _reuses(backend, gguf, tensor_split=[6, 2], gpu_ids=[0, 1]) is True


def test_unload_clears_the_recorded_ratio(tmp_path):
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    assert backend._auto_tensor_split is not None
    backend.unload_model()
    assert backend._auto_tensor_split is None


def test_a_layer_split_load_clears_the_recorded_ratio(tmp_path):
    """Turning tensor parallelism off must not leave the ratio behind: the
    next auto tensor load would compare against a server that never had it."""
    backend, gguf = _tp_backend(tmp_path, memory=[(0, 24_000, 24_000), (1, 24_000, 24_000)])
    _auto_tp(backend, gguf, tensor_split=[3, 1], gpu_ids=[0, 1])
    _launch(
        backend,
        gguf,
        gpu_memory_mode="auto",
        tensor_parallel=False,
        gpu_ids=[0, 1],
        n_ctx=4096,
    )
    assert backend._auto_tensor_split is None


# --------------------------------------------------------------------------
# 6. What /status reports after a recovery rewrites the argv.
# --------------------------------------------------------------------------


def _crash_once_with_an_arch_error(backend):
    """Make the first spawn die the way a binary with no kernels for the pinned
    card does, so the arch-crash retry runs and the second spawn succeeds."""
    calls: list[int] = []

    def _health(timeout, **_kw):
        calls.append(1)
        return len(calls) > 1

    backend._wait_for_health = _health
    backend._kernel_image_invalid = lambda _text: True
    return calls


def test_the_arch_crash_retry_stops_reporting_the_ratio_it_dropped(tmp_path):
    """The emitted ratio is not just a record, it is what `tensor_split` reports.

    The retry re-masks the child onto a narrowed device set, so a split sized for
    the crashed one weights the wrong cards and `_without_tensor_split` takes it
    off the argv. Left recorded, /status answers with a ratio the live child does
    not have, and here the narrowing leaves ONE device, so it would answer with a
    ratio beside `tensor_parallel: false`. The third card is below the reserve, so
    the planner pins two of three and the retry has somewhere to go.
    """
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 24_000, 24_000), (1, 24_000, 24_000), (2, 400, 24_000)],
    )
    _crash_once_with_an_arch_error(backend)

    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1])

    assert _flag(cmd, "--tensor-split") is None, "the retry kept a split it re-indexed"
    assert backend.tensor_parallel is False
    assert backend.tensor_split is None, (
        "/status still reports the dropped ratio: " f"{backend.tensor_split}"
    )


def test_a_launch_that_does_not_crash_still_reports_its_ratio(tmp_path):
    """The control. Clearing on the recovery arm must not clear on the ordinary
    one, or the PR's own reporting is gone."""
    backend, gguf = _tp_backend(
        tmp_path,
        memory=[(0, 24_000, 24_000), (1, 24_000, 24_000), (2, 400, 24_000)],
    )
    cmd = _auto_tp(backend, gguf, tensor_split=[3, 1])
    assert _flag(cmd, "--tensor-split") == "3,1"
    assert backend.tensor_split == [0.75, 0.25]
