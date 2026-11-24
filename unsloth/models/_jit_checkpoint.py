# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TrainingArguments(enable_jit_checkpoint = True) on more than one rank: every rank saves at the same step.

transformers' JITCheckpointCallback (trainer_jit_checkpoint.py) arms a 3 s timer on SIGTERM and checks a per-rank
flag at on_step_begin / on_pre_optimizer_step / on_step_end / on_epoch_end. A scheduler signals each rank on its
own, so the timers fire at slightly different times. When one rank's flag flips before a check and another's after
it, the first rank enters _save_checkpoint (collective under DDP, FSDP2 and native FSDP2) while the second runs the
next step's collectives: the job hangs and leaves a checkpoint with only the incomplete sentinel. Measured on
2x T4 (native FSDP2): 21-36 ms between the two signals was enough.

Here the timer only marks the request as pending, and every check first takes the maximum of the pending flags
over all ranks (one int all_reduce on a CPU gloo group; every rank reaches the same checks in the same order), so
either every rank saves at that check or none does. A single process keeps transformers' behaviour.
"""

import os

import torch

__all__ = ["install_jit_checkpoint_agreement"]

_PENDING = "_unsloth_jit_pending"
_HOOKS = ("on_step_begin", "on_pre_optimizer_step", "on_step_end", "on_epoch_end")


def _world():
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return None
    return dist


def _agreement_group(dist):
    """A CPU (gloo) group: the vote is a host-side int, so it never waits for the device stream at a step
    boundary. Collective to create, which is fine at install time (every rank builds the Trainer)."""
    if "gloo" in str(dist.get_backend()).lower():
        return None
    try:
        return dist.new_group(backend = "gloo")
    except Exception:
        return None


def _any_rank(dist, flag, group):
    device = "cpu"
    if group is None:
        backend = str(dist.get_backend()).lower()
        if "nccl" in backend and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        elif "xccl" in backend and hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu", torch.xpu.current_device())
    t = torch.tensor([1 if flag else 0], dtype = torch.int32, device = device)
    dist.all_reduce(t, op = dist.ReduceOp.MAX, group = group)
    return bool(t.item())


def _agree(manager):
    if getattr(manager, "_unsloth_jit_done", False):
        # Every rank agreed on one checkpoint at the same check and training is stopping: a later timer (the
        # same signal heard later on another rank, or a repeat) must not start a second one. All ranks set
        # this flag at the same check, so all of them skip the vote from here on.
        manager.is_checkpoint_requested = False
        return
    pending = bool(getattr(manager, _PENDING, False))
    dist = _world()
    if dist is not None:
        pending = _any_rank(dist, pending, getattr(manager, "_unsloth_jit_group", None))
    if pending:
        setattr(manager, _PENDING, False)
        manager._unsloth_jit_done = True
        manager.is_checkpoint_requested = True
    else:
        manager.is_checkpoint_requested = False


def _patch_manager(manager):
    if getattr(manager, "_unsloth_jit_agreement", False):
        return
    setattr(manager, _PENDING, False)
    dist = _world()
    manager._unsloth_jit_group = _agreement_group(dist) if dist is not None else None

    def _enable_checkpoint():
        # The timer thread only marks the request; the main thread turns it into is_checkpoint_requested at a
        # check every rank reaches (_agree), so no rank can act on it alone.
        setattr(manager, _PENDING, True)

    manager._enable_checkpoint = _enable_checkpoint
    manager._unsloth_jit_agreement = True


def _wrap_hook(callback, name):
    original = getattr(callback, name, None)
    if original is None or getattr(original, "_unsloth_jit_agreement", False):
        return

    def hook(*args, **kwargs):
        manager = getattr(callback, "jit_manager", None)
        if manager is not None:
            _agree(manager)
        return original(*args, **kwargs)

    hook._unsloth_jit_agreement = True
    hook.__wrapped__ = original
    setattr(callback, name, hook)


def _wrap_train_begin(callback):
    original = callback.on_train_begin
    if getattr(original, "_unsloth_jit_agreement", False):
        return

    def on_train_begin(*args, **kwargs):
        # A later train() call on the same Trainer starts with no checkpoint taken.
        manager = getattr(callback, "jit_manager", None)
        if manager is not None:
            manager._unsloth_jit_done = False
            setattr(manager, _PENDING, False)
        return original(*args, **kwargs)

    on_train_begin._unsloth_jit_agreement = True
    setattr(callback, "on_train_begin", on_train_begin)


def install_jit_checkpoint_agreement(trainer):
    """Called after Trainer.__init__. True when a JIT checkpoint callback was made rank-agreed."""
    if os.environ.get("UNSLOTH_JIT_CHECKPOINT_AGREE", "1") == "0":
        return False
    args = getattr(trainer, "args", None)
    if not getattr(args, "enable_jit_checkpoint", False):
        return False
    try:
        from transformers.trainer_jit_checkpoint import JITCheckpointCallback
    except Exception:
        return False
    handler = getattr(trainer, "callback_handler", None)
    done = False
    for callback in list(getattr(handler, "callbacks", None) or []):
        if not isinstance(callback, JITCheckpointCallback):
            continue
        manager = getattr(callback, "jit_manager", None)
        if manager is None:
            continue
        _patch_manager(manager)
        for name in _HOOKS:
            _wrap_hook(callback, name)
        _wrap_train_begin(callback)
        done = True
    return done
