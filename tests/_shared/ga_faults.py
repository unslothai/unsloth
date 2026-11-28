# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Known loss normalisation / gradient / update bugs, injected on purpose, to prove a parity test can fail.

``UNSLOTH_TEST_GA_FAULT=<name>`` makes tests/test_parallel_ga_parity_gloo.py install one in every process it
spawns (the one-process oracle and every rank) before ``import unsloth``. Each fault is a real regression class;
the suite must fail under every one (the fault -> failing test map is in that file's docstring).

A fault patches one function of one source file, matched by path, as that file is executed: also when a test
loads the file under another module name. ``install_on_import`` arms it for files not imported yet and patches
the ones already imported. Never imported by unsloth itself.
"""

from __future__ import annotations

import functools
import importlib.abc
import os
import sys

ENV = "UNSLOTH_TEST_GA_FAULT"


def _wrap_attr(owner, name, make):
    original = getattr(owner, name)
    if getattr(original, "_ga_fault", False):
        return
    faulty = make(original)
    faulty._ga_fault = True
    setattr(owner, name, faulty)


def _wrap_get_batch_samples(transform):
    """unsloth.models._utils resolves _unsloth_get_batch_samples (unsloth_zoo's token count) at call time."""

    def patch(module):
        def make(original):
            @functools.wraps(original)
            def faulty(self, *args, **kwargs):
                return transform(original, self, *args, **kwargs)

            return faulty

        _wrap_attr(module, "_unsloth_get_batch_samples", make)

    return patch


def _drop_num_items(original, self, *args, **kwargs):
    # Every micro-batch is then its own mean, divided by the accumulation steps (HF's fallback).
    batches, _ = original(self, *args, **kwargs)
    return batches, None


def _per_rank_norm(original, self, *args, **kwargs):
    # average_tokens_across_devices off: no count gather and no x world, so DDP averages per-rank means.
    self.args.average_tokens_across_devices = False
    return original(self, *args, **kwargs)


def _skip_count_allreduce(original, self, *args, **kwargs):
    # The count stays this rank's own while the loss is still scaled by the world (the gather was lost).
    was = self.args.average_tokens_across_devices
    self.args.average_tokens_across_devices = False
    try:
        return original(self, *args, **kwargs)
    finally:
        self.args.average_tokens_across_devices = was


def _cp_double_count(original, self, *args, **kwargs):
    # Context parallel peers share their rows: a count not divided back by cp counts every token cp times.
    batches, n = original(self, *args, **kwargs)
    manager = getattr(self, "_context_parallel_manager", None)
    if manager is not None and n is not None:
        n = n * manager.size
    return batches, n


def _micro_mean(module):
    # The loss head ignores num_items_in_batch and returns its micro-batch's own mean, while the Trainer still
    # treats the loss as normalised by the window's count (the remote code head regression): a mean of
    # micro-batch means, GA times too large. Wraps every causal LM loss Unsloth installs.
    from transformers.loss import loss_utils

    def wrap_mapping():
        for key, fn in list(loss_utils.LOSS_MAPPING.items()):
            if getattr(fn, "_ga_fault", False):
                continue

            def own_mean(
                *args,
                _fn = fn,
                **kwargs,
            ):
                kwargs["num_items_in_batch"] = None
                return _fn(*args, **kwargs)

            own_mean._ga_fault = True
            loss_utils.LOSS_MAPPING[key] = own_mean

    def make(original):
        @functools.wraps(original)
        def patch_loss_functions(*args, **kwargs):
            out = original(*args, **kwargs)
            wrap_mapping()
            return out

        return patch_loss_functions

    wrap_mapping()
    _wrap_attr(module, "patch_loss_functions", make)


def _count_padding(module):
    # Counts every shifted position, padding and masked labels included, instead of real target tokens.
    def make(original):
        def count_batch_items(labels, *args, **kwargs):
            _, short, degenerate = original(labels, *args, **kwargs)
            return labels.new_tensor(labels[..., 1:].numel()), short, degenerate

        return count_batch_items

    _wrap_attr(module, "count_batch_items", make)


def _tp_double_count(original, self, *args, **kwargs):
    # Tensor parallel peers hold the same rows. The count is gathered over the world and the loss scaled by the
    # world, which cancels; counting the dp replicas only (pipeline parallel's convention) while still scaling
    # by the world weighs every token tp times.
    batches, n = original(self, *args, **kwargs)
    layout = self.accelerator.__dict__.get("_unsloth_tp_data")
    if layout is not None and n is not None:
        import torch.distributed as dist
        n = n * layout[0] // dist.get_world_size()
    return batches, n


def _world_gather(data):
    import torch
    import torch.distributed as dist
    from accelerate.utils import recursively_apply

    def one(t):
        t = t.clone()[None] if t.ndim == 0 else t.contiguous()
        out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(out, t)
        return torch.cat(out, dim = 0)

    return recursively_apply(one, data, error_on_other_type = True)


def _pp_world_count(module):
    # Pipeline stages hold the same rows: gathering over the world counts each token pp times.
    _wrap_attr(module, "_dp_gather", lambda original: lambda accelerator, stage: _world_gather)


def _sr_skip_dtensor(module):
    # Before stochastic rounding covered FSDP2 / TP shards: every DTensor weight rounds to nearest.
    def make(original):
        def _tracked(p, dtensor = False):
            return False if module._is_dtensor(p) else original(p, dtensor)

        return _tracked

    _wrap_attr(module, "_tracked", make)


def _sr_skip_bnb_dtensor(module):
    # The bitsandbytes route only (8-bit AdamW): DTensor shards step without rounding, plain weights still round.
    def make(original):
        def _wrap_bitsandbytes(optimizer, tracked, *args, **kwargs):
            shards = {
                id(p) for g in optimizer.param_groups for p in g["params"] if module._is_dtensor(p)
            }
            kept = {k: v for k, v in tracked.items() if k not in shards}
            return original(optimizer, kept, *args, **kwargs)

        return _wrap_bitsandbytes

    _wrap_attr(module, "_wrap_bitsandbytes", make)


def _ep_double_expert_grads(module):
    # An expert gradient counted twice (a token dispatched to its expert twice, or the 1 / world scale lost).
    def make(original):
        def _ep_params(model):
            out = list(original(model))
            for _, p in out:
                if p.requires_grad and not getattr(p, "_ga_fault_hooked", False):
                    p.register_hook(lambda g: g * 2)
                    p._ga_fault_hooked = True
            return out

        return _ep_params

    _wrap_attr(module, "_ep_params", make)


def _aux_unscaled(module):
    # The MoE router aux loss added once per micro-batch at full weight, not by its share of the window's tokens.
    _wrap_attr(module, "unsloth_ga_scale_aux_loss", lambda original: lambda aux, *a, **k: aux)


_UTILS = "unsloth/models/_utils.py"
# name -> (source file it patches, patch(module))
FAULTS = {
    "drop_num_items": (_UTILS, _wrap_get_batch_samples(_drop_num_items)),
    "micro_mean": ("unsloth_zoo/loss_utils.py", _micro_mean),
    "per_rank_norm": (_UTILS, _wrap_get_batch_samples(_per_rank_norm)),
    "skip_count_allreduce": (_UTILS, _wrap_get_batch_samples(_skip_count_allreduce)),
    "count_padding": ("unsloth_zoo/loss_utils.py", _count_padding),
    "cp_double_count": (_UTILS, _wrap_get_batch_samples(_cp_double_count)),
    "tp_double_count": (_UTILS, _wrap_get_batch_samples(_tp_double_count)),
    "pp_world_count": ("unsloth/models/pp_env.py", _pp_world_count),
    "sr_skip_dtensor": ("unsloth/optimizers/bf16_stochastic_rounding.py", _sr_skip_dtensor),
    "sr_skip_bnb_dtensor": ("unsloth/optimizers/bf16_stochastic_rounding.py", _sr_skip_bnb_dtensor),
    "ep_double_expert_grads": ("unsloth/models/ep_native.py", _ep_double_expert_grads),
    "aux_unscaled": ("unsloth_zoo/fused_losses/aux_loss.py", _aux_unscaled),
}


def _matches(origin, target):
    return bool(origin) and origin.replace(os.sep, "/").endswith("/" + target)


class _PatchOnExec(importlib.abc.MetaPathFinder):
    """Patches the target file's module right after it executes, whatever name it is imported under."""

    def __init__(self, target, patch):
        self.target, self.patch = target, patch

    def find_spec(
        self,
        fullname,
        path,
        target = None,
    ):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        return _arm(spec, self.target, self.patch)


def _arm(spec, target, patch):
    if spec is None or not _matches(spec.origin, target) or spec.loader is None:
        return spec
    exec_module = spec.loader.exec_module

    def exec_and_patch(module):
        exec_module(module)
        patch(module)

    spec.loader.exec_module = exec_and_patch
    return spec


def install_on_import(name = None):
    """Arms the fault named by ``name`` (default: $UNSLOTH_TEST_GA_FAULT). Returns its name, or None."""
    name = os.environ.get(ENV, "") if name is None else name
    if not name:
        return None
    if name not in FAULTS:
        raise ValueError(f"{ENV}={name!r}: one of {sorted(FAULTS)}")
    target, patch = FAULTS[name]
    for module in list(sys.modules.values()):
        if _matches(getattr(module, "__file__", None), target):
            patch(module)
    sys.meta_path.insert(0, _PatchOnExec(target, patch))
    # Tests that load a source file by path (spec_from_file_location) bypass the import system's finders.
    import importlib.util

    by_path = importlib.util.spec_from_file_location

    @functools.wraps(by_path)
    def spec_from_file_location(*args, **kwargs):
        return _arm(by_path(*args, **kwargs), target, patch)

    importlib.util.spec_from_file_location = spec_from_file_location
    return name
