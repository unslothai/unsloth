# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 with a Trainer subclass that builds its own optimizer (gloo world of 2):
UnslothTrainer's create_optimizer with `embedding_learning_rate` (unsloth/trainer.py, run here from
its source) never calls the base Trainer.create_optimizer, so a group mixing a replicated trainable
parameter with sharded ones reached the LR scheduler unsplit (transformers 4.x
create_optimizer_and_scheduler builds the scheduler before Accelerator.prepare). The groups are now
split before the scheduler is built: one base LR per final group, and every group follows the
schedule during training."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import datetime  # noqa: E402
import functools  # noqa: E402
import inspect  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import (  # noqa: E402
    _install_wrapped_trainer_init,
    _lora,
    _targs,
)

_HELPERS = (
    "_create_unsloth_optimizer",
    "_embedding_weight_ids",
    "_install_legacy_resume",
    "_migrate_legacy_optimizer_state",
    "_migrate_previous_layout_state",
    "_super_create_optimizer_takes_model",
)
_CONSTANTS = (
    "_LEGACY_ROLE_ORDER",
    "_EMBEDDING_SPLIT_MARKER",
    "_SUPER_CREATE_OPTIMIZER_TAKES_MODEL",
)


def _unsloth_trainer_class():
    """UnslothTrainer with its real create_optimizer and the optimizer builder it calls, from
    unsloth/trainer.py (importing unsloth needs a GPU); SFTTrainer stands as transformers' Trainer."""
    from transformers import Trainer

    path = os.path.join(REPO, "unsloth", "trainer.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _HELPERS:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _CONSTANTS for t in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "UnslothTrainer":
            method = next(
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "create_optimizer"
            )
            node.body = [method]
            node.decorator_list = []
            body.append(node)
    ns = {
        "torch": torch,
        "inspect": inspect,
        "wraps": functools.wraps,
        "functools": functools,
        "logger": logging.getLogger("unsloth_test"),
        "SFTTrainer": Trainer,
        "__name__": "unsloth_trainer_source",
    }
    exec(compile(ast.Module(body = body, type_ignores = []), path, "exec"), ns)
    return ns["UnslothTrainer"]


def _mixed_model(native):
    model = _lora(1)
    first = next(p for n, p in model.named_parameters() if "lora_A" in n)
    cfg = native.FSDP2Config(param_dtype = torch.float32, ignored_params = {first})
    return native.shard_model(model, cfg)


def _case_scheduler(native, tmp):
    _install_wrapped_trainer_init()
    trainer = _unsloth_trainer_class()(model = _mixed_model(native), args = _targs(tmp))
    trainer.args.embedding_learning_rate = 1e-4
    trainer.create_optimizer_and_scheduler(num_training_steps = 10)
    optimizer = trainer.optimizer
    return {
        "mixed_left": native._mixed_param_groups(optimizer),
        "groups": len(optimizer.param_groups),
        "base_lrs": len(trainer.lr_scheduler.base_lrs),
        "lambdas": len(getattr(trainer.lr_scheduler, "lr_lambdas", [])),
    }


def _case_train(native, tmp):
    from datasets import Dataset

    _install_wrapped_trainer_init()
    ds = Dataset.from_list([{"input_ids": list(range(5, 21)), "labels": list(range(5, 21))}] * 4)
    args = _targs(
        tmp,
        max_steps = 3,
        learning_rate = 1e-3,
        per_device_train_batch_size = 1,
        lr_scheduler_type = "linear",
        save_strategy = "no",
        logging_steps = 1,
    )
    args.embedding_learning_rate = 1e-4
    trainer = _unsloth_trainer_class()(model = _mixed_model(native), args = args, train_dataset = ds)
    trainer.train()
    groups = trainer.optimizer.param_groups
    # Linear decay to 0 over 3 steps: after the last step every group sits at lr 0; a group the
    # scheduler never saw keeps its initial LR.
    return {
        "mixed_left": native._mixed_param_groups(trainer.optimizer),
        "lrs": [float(g["lr"]) for g in groups],
    }


def _worker(rank, world, port, out, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        native = _load_native()
        res = _case_scheduler(native, tmp) if case == "scheduler" else _case_train(native, tmp)
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    case,
    tmp_path,
    timeout = 240,
):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out, case, str(tmp_path)),
        nprocs = 2,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(
                    f"{case}: the ranks did not finish within {timeout} s (hung collective)"
                )
    except mp.ProcessExitedException as exc:
        pytest.fail(f"{case}: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ROUND12_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_subclass_optimizer_groups_are_split_before_the_scheduler(tmp_path):
    pytest.importorskip("peft")
    for rank, r in _run("scheduler", tmp_path).items():
        assert r["mixed_left"] == 0, (rank, r)
        assert r["base_lrs"] == r["groups"] and r["lambdas"] == r["groups"], (rank, r)


def test_subclass_optimizer_every_group_follows_the_schedule(tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    for rank, r in _run("train", tmp_path).items():
        assert r["mixed_left"] == 0, (rank, r)
        assert all(lr == 0.0 for lr in r["lrs"]), (rank, r)
