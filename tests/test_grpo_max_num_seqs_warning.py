# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO / RLOO with fast_inference warn once when one vLLM generate call has more rows than the engine's
max_num_seqs (from_pretrained sizes the cap from free KV memory before any trainer exists)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_vllm_batch_cap", ROOT / "unsloth" / "models" / "_vllm_batch_cap.py"
)
cap_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap_mod)


def _trainer(
    cap = 48,
    bs = 8,
    steps = 8,
    gen = None,
    mode = "colocate",
    procs = 1,
    tp = 1,
    where = "vllm_generation",
):
    engine = SimpleNamespace(
        vllm_config = SimpleNamespace(scheduler_config = SimpleNamespace(max_num_seqs = cap))
    )
    llm = SimpleNamespace(llm_engine = engine)
    args = SimpleNamespace(
        use_vllm = True,
        vllm_mode = mode,
        per_device_train_batch_size = bs,
        steps_per_generation = steps,
        generation_batch_size = gen,
        vllm_tensor_parallel_size = tp,
    )
    t = SimpleNamespace(args = args, accelerator = SimpleNamespace(num_processes = procs))
    if where == "vllm_generation":
        t.vllm_generation = SimpleNamespace(llm = llm)
    elif where == "llm":
        t.llm = llm
    else:
        t.model = SimpleNamespace(vllm_engine = llm)
    return t


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GRPO_MAX_NUM_SEQS_WARN", raising = False)


@pytest.mark.parametrize("where", ["vllm_generation", "llm", "model"])
def test_warns_once_when_the_batch_exceeds_the_cap(capsys, where):
    t = _trainer(where = where)
    assert cap_mod.grpo_check_max_num_seqs(t) == (64, 48)
    out = capsys.readouterr().out
    assert "64 rows" in out and "max_num_seqs = 48" in out and "max_num_seqs = 64" in out
    assert cap_mod.grpo_check_max_num_seqs(t) == (64, 48)
    assert capsys.readouterr().out == ""


def test_quiet_when_the_batch_fits(capsys):
    assert cap_mod.grpo_check_max_num_seqs(_trainer(cap = 64)) is None
    assert cap_mod.grpo_check_max_num_seqs(_trainer(bs = 4, steps = 2)) is None
    assert capsys.readouterr().out == ""


def test_rows_per_engine():
    args = _trainer(gen = 256, procs = 4).args
    assert cap_mod.grpo_generation_rows_per_engine(args, 4) == 64
    args = _trainer(gen = 256, tp = 2).args
    assert cap_mod.grpo_generation_rows_per_engine(args, 4) == 128
    args = _trainer(steps = None).args
    args.gradient_accumulation_steps = 4
    assert cap_mod.grpo_generation_rows_per_engine(args, 1) == 32


def test_server_mode_no_vllm_and_kill_switch(capsys, monkeypatch):
    assert cap_mod.grpo_check_max_num_seqs(_trainer(mode = "server")) is None
    t = _trainer()
    t.args.use_vllm = False
    assert cap_mod.grpo_check_max_num_seqs(t) is None
    monkeypatch.setenv("UNSLOTH_GRPO_MAX_NUM_SEQS_WARN", "0")
    assert cap_mod.grpo_check_max_num_seqs(_trainer()) is None
    assert capsys.readouterr().out == ""


def test_missing_engine_is_silent():
    t = _trainer()
    t.vllm_generation = SimpleNamespace(llm = None)
    assert cap_mod.grpo_check_max_num_seqs(t) is None


def test_rl_injects_the_check_for_grpo_and_rloo():
    src = (ROOT / "unsloth" / "models" / "rl.py").read_text(encoding = "utf-8")
    assert 'trainer_file in ("grpo_trainer", "rloo_trainer")' in src
    assert "_vllm_batch_cap import grpo_check_max_num_seqs" in src
