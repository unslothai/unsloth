# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The multi-card leg: a branch no pinned payload can reach.

``unsloth/kernels/utils.py:170``::

    if DEVICE_COUNT > 1:
        torch_gpu_device = torch.cuda.device      # a real device switch
    else:
        def torch_gpu_device(device): return nullcontext()

``build_kernel.py`` pins every ordinary payload with ``CUDA_VISIBLE_DEVICES``,
so **every unsloth kernel this CI has ever run took the nullcontext branch** --
and so did the ``DEVICE_COUNT``-sized ``CUDA_STREAMS`` / ``WEIGHT_BUFFERS`` /
``ABSMAX_BUFFERS`` arrays, the per-device rotary caches in
``unsloth/models/llama.py:1838``, and the ``temp_mlp`` device tuples at
``llama.py:1300``.

The leg is free because the divergence is triggered by VISIBILITY, not by using
both cards: the model still fits on one, so it co-tenants and the driver
reserves its 0.7 GB on each card rather than a whole one.

**What these rules deliberately do NOT assert** is that the parameters end up
spread across both cards. Whether accelerate shards or pins to ``cuda:0`` is
the open question the leg exists to answer, and a rule written before the
answer is a rule written to match whatever happens.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from kaggle_t4_ci import build_kernel, legs  # noqa: E402


def _payload():
    spec = importlib.util.spec_from_file_location(
        "_t4_smoke_multi_gpu", SMOKE_DIR / "run_t4_smoke.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SMOKE_DIR))
    spec.loader.exec_module(module)
    return module


GOOD = {
    "device_count": 2,
    "module_device_count": 2,
    "torch_gpu_device_is_real_switch": True,
    "torch_gpu_device_repr": "<class 'torch.cuda.device'>",
    "cuda_streams_len": 2,
    "weight_buffers_len": 2,
    "absmax_buffers_len": 2,
    "rotary_cache_slots": 2,
    "parameters_by_device": {"cuda:0": 596049920},
    "cuda_devices_holding_parameters": ["cuda:0"],
}


def test_the_happy_reading_passes():
    assert _payload().multi_gpu_failures(GOOD, expected_cards = 2) == []


def test_the_nullcontext_shim_is_the_failure_this_leg_exists_for():
    """The single most important rule here. A leg that ran on one card, or
    whose unsloth was imported before the cards were visible, gets the shim --
    which performs NO device switch -- and would otherwise report a pass for
    coverage a pinned leg already has."""
    facts = dict(
        GOOD,
        torch_gpu_device_is_real_switch = False,
        torch_gpu_device_repr = "<function torch_gpu_device at 0x7f>",
    )
    broken = _payload().multi_gpu_failures(facts, expected_cards = 2)
    assert broken and "nullcontext" in broken[0]


def test_one_visible_card_fails_and_says_only_that():
    """A pinned run measures the wrong machine, so every rule after the count
    is noise. Reporting eight failures about arrays that are correctly sized
    for the one card the leg was given sends the reader after the wrong bug."""
    facts = dict(
        GOOD,
        device_count = 1,
        module_device_count = 1,
        torch_gpu_device_is_real_switch = False,
        cuda_streams_len = 1,
        weight_buffers_len = 1,
        absmax_buffers_len = 1,
        rotary_cache_slots = 1,
    )
    broken = _payload().multi_gpu_failures(facts, expected_cards = 2)
    assert len(broken) == 1
    assert "the driver pinned it" in broken[0]


def test_an_import_before_the_cards_were_visible_is_caught_separately():
    """torch can see two cards while unsloth.kernels.utils was imported when
    only one was visible -- the bindings are made once, at import. The two are
    different failures with different fixes, so they are reported separately."""
    facts = dict(GOOD, module_device_count = 1)
    broken = _payload().multi_gpu_failures(facts, expected_cards = 2)
    assert broken and "imported before the cards were visible" in broken[0]


def test_short_stream_and_buffer_arrays_each_fail():
    payload = _payload()
    for key in ("cuda_streams_len", "weight_buffers_len", "absmax_buffers_len"):
        broken = payload.multi_gpu_failures(dict(GOOD, **{key: 1}), expected_cards = 2)
        assert broken, key
        assert "no stream or buffer of its own" in broken[0], key
        # None is not "fine": a missing array means the attribute is gone.
        assert payload.multi_gpu_failures(dict(GOOD, **{key: None}), expected_cards = 2)


def test_missing_facts_are_a_failure_rather_than_a_silence():
    """A cycle that never ran the reading reports null, and null must not read
    as "nothing wrong". This is the shape that let a payload be carried and
    never executed for two rounds."""
    payload = _payload()
    assert payload.multi_gpu_failures(None, expected_cards = 2)
    assert payload.multi_gpu_failures({}, expected_cards = 2)
    broken = payload.multi_gpu_failures({"error": "ImportError: no unsloth"}, expected_cards = 2)
    assert broken and "could not be read" in broken[0]


def test_a_model_entirely_off_the_gpu_fails():
    facts = dict(GOOD, parameters_by_device = {"cpu": 596049920}, cuda_devices_holding_parameters = [])
    assert _payload().multi_gpu_failures(facts, expected_cards = 2)


def test_the_spread_across_cards_is_RECORDED_and_not_required():
    """The open question, and the rule must not pre-empt it.

    Every two-card measurement this repo has is a LOAD and not a train
    (unsloth-probe-vision-recon-c76ea3 saw a model split 897.7/1017.1 MB and
    never called a trainer). So a single-card placement passes, a spread
    passes, and the report carries which happened. When a session answers it,
    tighten this deliberately rather than discovering the rule was already
    asserting an answer nobody had.
    """
    payload = _payload()
    one_card = dict(GOOD, cuda_devices_holding_parameters = ["cuda:0"])
    spread = dict(
        GOOD,
        parameters_by_device = {"cuda:0": 300000000, "cuda:1": 296049920},
        cuda_devices_holding_parameters = ["cuda:0", "cuda:1"],
    )
    assert payload.multi_gpu_failures(one_card, expected_cards = 2) == []
    assert payload.multi_gpu_failures(spread, expected_cards = 2) == []


def test_the_reading_is_taken_from_the_module_and_not_recomputed():
    """`torch.cuda.device_count()` on both sides of the comparison is how a
    rule ends up unable to fail: the binding is made once at import, so asking
    torch again answers a different question and answers it agreeably."""
    import ast

    source = (SMOKE_DIR / "run_t4_smoke.py").read_text(encoding = "utf-8")
    func = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "multi_gpu_facts"
    )
    body = ast.unparse(func)
    assert "from unsloth.kernels import utils" in body
    assert "binding is torch.cuda.device" in body
    assert "DEVICE_COUNT" in body


# ------------------------------------------------------------ the leg itself


def test_the_leg_asks_for_two_cards_and_the_BUILT_payload_enforces_it():
    """Asserted through the generated notebook, not off the dataclass. A field
    nothing emits reads like coverage and does nothing -- which is exactly how
    a payload was carried and never run for two rounds."""
    leg = legs.LEGS["multi_gpu"]
    assert leg.all_cards is True
    notebook = build_kernel.build_payload_notebook(
        SMOKE_DIR, leg, unsloth_ref = "main", zoo_ref = "main"
    )
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "device_count() == 2" in source, (
        "the built payload does not require two visible GPUs, so a run the "
        "driver pinned would measure the single-card branch and pass"
    )
    assert "--require-multi-gpu" in source
    assert "--expected-cards" in source


def test_every_OTHER_leg_still_requires_exactly_one():
    """The pin check is what catches a driver that failed to pin, and widening
    it for this leg must not widen it for the rest."""
    for name, leg in legs.LEGS.items():
        if name == "multi_gpu":
            continue
        notebook = build_kernel.build_payload_notebook(
            SMOKE_DIR, leg, unsloth_ref = "main", zoo_ref = "main"
        )
        source = "".join("".join(c["source"]) for c in notebook["cells"])
        assert "device_count() == 1" in source, name


def test_the_leg_is_small_enough_to_co_tenant():
    """The whole cost argument. At a whole card this leg would block gptoss
    (12.78 of the 13.0 budget) for its entire life and the kernel would get
    longer, which is the one thing this was built not to do."""
    leg = legs.LEGS["multi_gpu"]
    assert leg.vram_gb <= 1.0, leg.vram_gb
    assert leg.vram_gb + legs.LEGS["gptoss"].vram_gb > 13.0, (
        "gptoss must still be unable to share a card with this leg; if that "
        "stopped being true the co-tenancy argument changed"
    )


def test_it_does_not_export_a_gguf_and_the_reason_is_recorded():
    """The bundle install_llama_cpp fetches for the notebook legs is the CPU
    one -- on unsloth-probe-full-concurrent-417238 this model's llama-bench
    reports `backend CPU` -- so a two-card tensor-split assertion here could
    not fail. Adding the export anyway would cost ~40s of a thin margin and
    buy a claim four other legs already make."""
    leg = legs.LEGS["multi_gpu"]
    assert "--export-gguf" not in leg.args
    assert "backend CPU" in legs.UNWIRED["multi_gpu"] or "CPU bundle" in legs.UNWIRED["multi_gpu"]


# --------------------------------------------------------- driver scheduling


def _driver_source() -> str:
    payloads = {}
    for name in ("default", "gptoss", "multi_gpu"):
        leg = legs.LEGS[name]
        payloads[f"t4_{leg.name}.ipynb"] = build_kernel.build_payload_notebook(
            SMOKE_DIR, leg, unsloth_ref = "main", zoo_ref = "main"
        )
    driver = build_kernel.build_driver(
        payloads,
        per_run_timeout = 3600,
        vram_source = {
            f"t4_{legs.LEGS[n].name}.ipynb": legs.LEGS[n]
            for n in ("default", "gptoss", "multi_gpu")
        },
        all_card = ("t4_Multi_GPU.ipynb",),
    )
    return "".join("".join(c["source"]) for c in driver["cells"])


def test_the_all_card_payload_is_kept_out_of_the_card_queue():
    source = _driver_source()
    assert "ALL_CARD = (" in source
    assert "t4_Multi_GPU.ipynb" in source.split("ALL_CARD = ")[1][:200]
    assert "_queue = [(i, n) for i, n in enumerate(ORDER) if n not in ALL_CARD]" in source


def test_it_runs_UNPINNED_or_it_measures_the_single_card_branch():
    """`run_one(name, None, idx)`. An index here sets CUDA_VISIBLE_DEVICES and
    the leg would take the nullcontext branch under a multi-GPU name."""
    source = _driver_source()
    lane = source.split("def _all_card_lane")[1].split("threads = []")[0]
    assert "run_one(name, None, idx)" in lane
    assert "run_one(name, 0" not in lane


def test_the_reservation_is_all_or_nothing():
    """A partial reservation holds a seat on card 0 while waiting for card 1 --
    capacity nothing is using, and a deadlock against a big leg waiting for
    card 0. The refusal path must not have mutated anything."""
    source = _driver_source()
    body = source.split("def _admit_all(name):")[1].split("def _release_all")[0]
    check, commit = body.split("for g in range(N_GPU):")[1:3]
    # Every rejection happens in the FIRST loop, before any card is charged.
    assert "return False" in check
    assert "return False" not in commit
    assert "card_load[g] += want" in commit
    assert "card_load[g] += want" not in check


def test_the_all_card_lane_is_joined_with_the_card_workers():
    """It runs BESIDE the training legs, so anything concluding the cards are
    free has to wait for it too."""
    source = _driver_source()
    assert "for t in all_card_threads:" in source
    assert source.index("for t in all_card_threads:") < source.index(
        "if AFTER_GPU and not AFTER_GPU_CONCURRENT:"
    )


def test_a_single_leg_dispatch_still_stands_down_on_one_card():
    """`expected_gpus = min(len(payloads), SESSION_GPUS)` derives 1 for a
    one-leg kernel, which is right for every leg but this one.

    An all-card leg on a one-card allocation is an INFRASTRUCTURE fact, and
    without this it arrives as a payload failure on `device_count() == 2` --
    the exact confusion the shortfall guard's own comment says it exists to
    prevent.
    """
    notebooks = build_kernel.build_kernel(
        SMOKE_DIR,
        ("multi_gpu",),
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 3600,
    )
    source = "".join("".join(c["source"]) for c in notebooks["cells"])
    assert "EXPECTED_GPUS = 2" in source, (
        "a one-leg multi_gpu dispatch accepts a single card, so the leg would "
        "fail inside the payload instead of standing the kernel down"
    )
    # And an ordinary one-leg dispatch is unchanged.
    other = build_kernel.build_kernel(
        SMOKE_DIR,
        ("default",),
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 3600,
    )
    assert "EXPECTED_GPUS = 1" in "".join("".join(c["source"]) for c in other["cells"])
