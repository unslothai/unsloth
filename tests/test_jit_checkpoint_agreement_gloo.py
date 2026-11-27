# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""enable_jit_checkpoint on two ranks (gloo, CPU): a SIGTERM that reaches the ranks a moment apart must still make
every rank save at the same check. transformers' own callback acts on each rank's flag alone (the control case
below saves at different checks, which hangs a real run inside the collective save). _jit_checkpoint.py is loaded
on its own (no GPU); the save itself is stubbed to record where it ran."""

import importlib.util
import multiprocessing
import os
import types
from pathlib import Path

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_layout_agree_gloo import _spawn  # noqa: E402

jit = pytest.importorskip("transformers.trainer_jit_checkpoint")

HOOKS = ("on_step_begin", "on_pre_optimizer_step", "on_step_end")


def _module():
    path = Path(__file__).resolve().parents[1] / "unsloth/models/_jit_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_test_jit_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trainer():
    callback = jit.JITCheckpointCallback()
    trainer = types.SimpleNamespace(args = types.SimpleNamespace(enable_jit_checkpoint = True))
    callback.set_trainer(trainer)
    trainer.callback_handler = types.SimpleNamespace(callbacks = [object(), callback])
    return trainer, callback


def _drive(
    callback,
    flip_at,
    checks = 12,
):
    """Returns check indices where a checkpoint was written; SIGTERM fires just before flip_at."""
    manager = callback.jit_manager
    saved = []

    def execute():
        manager.is_checkpoint_requested = False
        saved.append(index[0])

    manager.execute_jit_checkpoint = execute
    index = [0]
    control = types.SimpleNamespace(should_training_stop = False, should_save = False)
    for i in range(checks):
        index[0] = i
        if i == flip_at:
            manager._enable_checkpoint()
        getattr(callback, HOOKS[i % len(HOOKS)])(None, None, control)
    return saved


def _worker(rank, world, port, out, agree):
    import datetime

    import torch.distributed as dist

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        trainer, callback = _trainer()
        if agree:
            assert _module().install_jit_checkpoint_agreement(trainer)
        out.put(
            (rank, _drive(callback, flip_at = 4 + rank))
        )  # rank 1 hears the signal one check later
    finally:
        dist.destroy_process_group()


def _run(agree):
    results = dict(_spawn(_worker, agree))
    assert sorted(results) == [0, 1]
    return results


def test_ranks_save_at_the_same_check():
    out = _run(agree = True)
    assert out[0] == out[1] == [4]


def test_control_transformers_alone_saves_at_different_checks():
    out = _run(agree = False)
    assert out[0] == [4] and out[1] == [5]


def test_single_process_keeps_transformers_behaviour():
    trainer, callback = _trainer()
    assert _module().install_jit_checkpoint_agreement(trainer)
    assert _drive(callback, flip_at = 3) == [3]


def test_second_train_call_can_checkpoint_again():
    trainer, callback = _trainer()
    assert _module().install_jit_checkpoint_agreement(trainer)
    assert _drive(callback, flip_at = 2) == [2]
    assert _drive(callback, flip_at = 5) == []  # same run: already checkpointed
    callback.on_train_begin(None, None, None)
    assert _drive(callback, flip_at = 1) == [1]


def test_off_without_the_flag_or_by_env(monkeypatch):
    m = _module()
    trainer, _ = _trainer()
    trainer.args.enable_jit_checkpoint = False
    assert m.install_jit_checkpoint_agreement(trainer) is False
    trainer, _ = _trainer()
    monkeypatch.setenv("UNSLOTH_JIT_CHECKPOINT_AGREE", "0")
    assert m.install_jit_checkpoint_agreement(trainer) is False
