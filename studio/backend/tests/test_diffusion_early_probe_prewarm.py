# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Early int8 probe warm-up: started with the load, joined before timing, reused at the swap. CPU only."""

from __future__ import annotations

import ast
import threading
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_convrot_quant as cq
from core.inference import diffusion_int8_gemm as gemm

_BACKEND = Path(__file__).resolve().parents[1]
_VERDICT_FILE_PRESENT = cq.verdict_file_present


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(cq.PROBE_PREWARM_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_GEMM", raising = False)
    monkeypatch.setattr(gemm, "_EARLY_PREWARM", {})
    monkeypatch.setattr(gemm, "_EARLY_DONE", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gemm, "is_rocm_torch", lambda t: False)
    monkeypatch.setattr(cq, "rotq_probes_pending", lambda index: True)
    monkeypatch.setattr(cq, "producer_probe_pending", lambda index: True)
    monkeypatch.setattr(gemm, "_gemm_probe_jobs", lambda index: [object(), object()])
    monkeypatch.setattr(cq, "verdict_file_present", lambda: False)


def _record(monkeypatch, gate = None):
    calls = []

    def prewarm(
        index,
        *,
        rotq,
        producer,
        extra,
        logger = None,
    ):
        if gate is not None:
            assert gate.wait(10)
        calls.append((index, rotq, producer, len(extra), threading.current_thread().name))
        return 7

    monkeypatch.setattr(cq, "prewarm_probe_kernels", prewarm)
    return calls


def test_the_early_warm_up_runs_on_its_own_thread_and_the_swap_reuses_it(monkeypatch):
    gate = threading.Event()
    calls = _record(monkeypatch, gate)
    assert gemm.start_early_probe_prewarm(0, rotated = True, producer = True)
    assert not gemm.start_early_probe_prewarm(0, rotated = True, producer = True)  # one per card
    gate.set()
    gemm.join_early_probe_prewarm()
    assert calls == [(0, True, True, 2, "unsloth-probe-prewarm-early")]
    monkeypatch.setattr(
        "core.inference.diffusion_flux_convrot_fused.producer_probe_expected", lambda t: True
    )
    assert gemm._prewarm_probes(0, object(), True, None) == 7
    assert len(calls) == 1


def test_a_swap_needing_more_than_the_early_warm_up_compiles_the_rest(monkeypatch):
    calls = _record(monkeypatch)
    gemm.start_early_probe_prewarm(0, rotated = False, producer = False)
    gemm.join_early_probe_prewarm(0)
    monkeypatch.setattr(
        "core.inference.diffusion_flux_convrot_fused.producer_probe_expected", lambda t: False
    )
    assert (
        gemm._prewarm_probes(0, object(), True, None) == 7
    )  # rotated Linears: rotq kernels still to compile
    assert [c[1] for c in calls] == [False, True]


def test_switches_and_non_cuda_cards_start_nothing(monkeypatch):
    calls = _record(monkeypatch)
    monkeypatch.setenv(cq.PROBE_PREWARM_ENV, "0")
    assert not gemm.start_early_probe_prewarm(0, rotated = True, producer = False)
    monkeypatch.delenv(cq.PROBE_PREWARM_ENV)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_GEMM", "0")
    assert not gemm.start_early_probe_prewarm(0, rotated = True, producer = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_GEMM")
    monkeypatch.setattr(gemm, "is_rocm_torch", lambda t: True)
    assert not gemm.start_early_probe_prewarm(0, rotated = True, producer = False)
    gemm.join_early_probe_prewarm()
    assert calls == []


def test_a_card_with_saved_verdicts_compiles_nothing_early(monkeypatch):
    # A saved verdict file, not the (too early) pending checks, decides on a returning card.
    calls = _record(monkeypatch)
    monkeypatch.setattr(cq, "verdict_file_present", lambda: True)
    assert not gemm.start_early_probe_prewarm(0, rotated = True, producer = True)
    gemm.join_early_probe_prewarm()
    assert calls == []


def test_the_load_kicks_it_for_hosted_int8_cuda_loads_only(monkeypatch):
    import core.inference.diffusion as D

    seen = []
    monkeypatch.setattr(
        gemm,
        "start_early_probe_prewarm",
        lambda index, rotated, producer, logger = None: seen.append((index, rotated, producer))
        or True,
    )
    from core.inference.diffusion_families import detect_family

    cuda = types.SimpleNamespace(device = "cuda", ordinal = 1)
    assert D._kick_early_probe_prewarm(detect_family("black-forest-labs/flux-3-image-klein"), cuda)
    assert D._kick_early_probe_prewarm(detect_family("Qwen/Qwen-Image-2.1"), cuda)
    assert seen == [(1, True, True), (1, True, False)]  # the producer probe is FLUX.2 / FLUX.3 only
    assert not D._kick_early_probe_prewarm(
        detect_family("Qwen/Qwen-Image-2.1"), types.SimpleNamespace(device = "mps", ordinal = None)
    )


def test_every_prefetch_stop_joins_the_early_warm_up(monkeypatch):
    import core.inference.diffusion as D

    joined = []
    monkeypatch.setattr(gemm, "join_early_probe_prewarm", lambda index = None: joined.append(index))
    me = types.SimpleNamespace(_load_prefetch = None)
    D.DiffusionBackend._stop_load_prefetch(me)
    assert joined == [None]


def test_the_pipeline_load_kicks_it_only_for_an_int8_seed_after_the_previous_stop():
    tree = ast.parse((_BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8"))
    kicks = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "pipeline_seed_scheme == 'int8'"
        and any(
            isinstance(c, ast.Call) and ast.unparse(c.func) == "_kick_early_probe_prewarm"
            for c in ast.walk(n)
        )
    ]
    assert len(kicks) == 1


def test_the_verdict_file_check_reads_the_convrot_file_next_to_the_probe_cache(
    tmp_path, monkeypatch
):
    import core.inference.diffusion_probe_cache as pc

    monkeypatch.setattr(pc, "_cache_file", lambda: tmp_path / "diffusion_quant_probe.json")
    (tmp_path / "diffusion_quant_probe.json").write_text("{}")
    assert not _VERDICT_FILE_PRESENT()
    (tmp_path / cq._VERDICT_FILE).write_text("{}")
    assert _VERDICT_FILE_PRESENT()
    monkeypatch.setenv(pc._ENV, "0")
    assert not _VERDICT_FILE_PRESENT()


def test_joining_never_blocks_inside_tracing_or_capture(monkeypatch):
    gate = threading.Event()
    calls = _record(monkeypatch, gate)
    assert gemm.start_early_probe_prewarm(0, rotated = True, producer = False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    gemm.join_early_probe_prewarm()  # returns at once, the thread is still waiting on the gate
    assert 0 in gemm._EARLY_PREWARM and calls == []
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    gemm.join_early_probe_prewarm()
    assert 0 in gemm._EARLY_PREWARM
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    gate.set()
    gemm.join_early_probe_prewarm()
    assert gemm._EARLY_PREWARM == {} and len(calls) == 1


def test_a_joiner_racing_the_start_waits_for_the_warm_up(monkeypatch):
    # A joiner that runs between publishing the entry and ``start()`` must still wait for the worker.
    calls = _record(monkeypatch)
    base = threading.Thread
    seen = {}

    def joiner():
        gemm.join_early_probe_prewarm(0)
        seen["calls_at_return"] = len(calls)

    class RacingThread(base):
        def start(self):
            other = base(target = joiner)
            other.start()
            other.join(0.5)  # on the racy order the joiner finishes here, before the worker starts
            super().start()
            seen["joiner"] = other

    monkeypatch.setattr(gemm.threading, "Thread", RacingThread)
    assert gemm.start_early_probe_prewarm(0, rotated = True, producer = True)
    seen["joiner"].join(10)
    assert not seen["joiner"].is_alive()
    assert seen["calls_at_return"] == 1
    assert 0 not in gemm._EARLY_PREWARM


def test_a_failed_thread_start_leaves_no_entry_and_allows_a_retry(monkeypatch):
    calls = _record(monkeypatch)
    base = threading.Thread

    class FailingThread(base):
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(gemm.threading, "Thread", FailingThread)
    assert not gemm.start_early_probe_prewarm(0, rotated = True, producer = True)
    assert 0 not in gemm._EARLY_PREWARM
    monkeypatch.setattr(gemm.threading, "Thread", base)
    assert gemm.start_early_probe_prewarm(0, rotated = True, producer = True)
    gemm.join_early_probe_prewarm(0)
    assert len(calls) == 1
