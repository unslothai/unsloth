# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""A plain transformers Trainer (and Seq2SeqTrainer) on a LoRA model with reentrant gradient
checkpointing must train under DDP: Unsloth picks find_unused_parameters = False as it does for its
TRL trainers, where transformers' True fails with "marked as ready twice". An explicit value is kept."""

from __future__ import annotations

import os
import queue as queue_module
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned ranks inherit sys.path
import _plain_trainer_ddp_gc_worker as worker  # noqa: E402

pytest.importorskip("peft")
pytest.importorskip("accelerate")
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "two gloo ranks over TCP localhost")


def _failed(results):
    """A rank that raised (in backward, in the save) can leave its peer blocked in a collective:
    once one has, wait only briefly for the other."""
    return any(
        out.get("error") or any(isinstance(c, dict) and c.get("error") for c in out.values())
        for out in results.values()
    )


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _run(cases):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target = worker.run_rank, args = (r, port, cases, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    try:
        while len(results) < len(procs):
            try:
                rank, out = queue.get(timeout = 60 if _failed(results) else 600)
            except queue_module.Empty:
                break
            results[rank] = out
    finally:
        for p in procs:
            p.join(5 if len(results) < len(procs) else 120)
            if p.is_alive():
                p.kill()
    assert results, "no rank reported"
    for out in results.values():
        assert "error" not in out, out.get("error")
    return results


def test_plain_trainers_train_under_ddp_with_reentrant_checkpointing():
    results = _run([("trainer", "Trainer", None), ("seq2seq", "Seq2SeqTrainer", None)])
    for rank in results:
        for name in ("trainer", "seq2seq"):
            case = results[rank].get(name)
            assert case is not None, (rank, name, results[rank])
            assert case["error"] is None, (rank, name, case["error"])
            assert case["is_ddp"] and case["find_unused"] is False
            assert case["args_value"] is False
            assert len(case["losses"]) == 2
    assert sorted(results) == [0, 1]
    # DDP averages the gradients: both ranks log the same losses.
    assert results[0]["trainer"]["losses"] == results[1]["trainer"]["losses"]


def test_explicit_user_value_is_kept():
    results = _run([("user_true", "Trainer", True)])
    for rank in results:
        case = results[rank]["user_true"]
        assert case["args_value"] is True
        assert case["find_unused"] is True
        # The user's choice is honoured, so the reentrant checkpointing failure is theirs to see.
        assert case["error"] is not None and "ready" in case["error"], case["error"]
