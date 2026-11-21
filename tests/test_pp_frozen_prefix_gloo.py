# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Frozen-prefix skip under pipeline parallel: gloo pp 2 vs one process, fp32, gradient checkpointing on.

A later stage's layer list starts with pass-through stand-ins (no parameters), so its first trainable layer is
its first own layer. The skip must not cut there: the stand-ins return the previous stage's activations, and
detaching them leaves every earlier stage's decoder adapters without a gradient (Gemma-4 PP=2 on 2x T4 trained only its last
stage: step-1 grad norm 2.78 against 5.12 on one GPU)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _tiny_mm as tiny  # noqa: E402
import test_pp_vlm_text_only_gloo as vlm  # noqa: E402
from test_pp_native_gloo import _load, _torch_pipelining_error  # noqa: E402

pytest.importorskip("transformers.models.gemma4")
pytest.importorskip("transformers.modeling_layers")

ppn = vlm.ppn
LAYERS = 6


def _checkpointed(model, fp):
    """What Unsloth's get_peft_model leaves: checkpointed layers and the frozen-prefix hooks installed."""
    model.get_base_model().gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    )
    fp.install_frozen_prefix_skip(model)
    return model


def _case_parity(
    rank,
    world,
    tmp,
    schedule = "gpipe",
):
    fp = _load("_frozen_prefix")
    build = dict(layers = LAYERS, causal_lm = True)
    ref = _checkpointed(tiny.build("gemma4_e2b", **build), fp)
    ids, labels = tiny.batch()
    ref_res = vlm.reference(ref, ids, labels)
    ref_cuts = dict(fp.FROZEN_PREFIX_STATS)
    model = _checkpointed(tiny.build("gemma4_e2b", **build), fp)
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2, schedule = schedule)
    layers = pp.parts.layers
    before = dict(fp.FROZEN_PREFIX_STATS)
    pp_res = vlm.pipeline(model, pp, ids, labels)
    return {
        "ref": ref_res,
        "pp": pp_res,
        "bounds": pp.bounds,
        "ref_cuts": ref_cuts["forwards"],
        "pp_cuts": fp.FROZEN_PREFIX_STATS["forwards"] - before["forwards"],
        "stand_ins": sum(type(layer).__name__ == "_PPPassThrough" for layer in layers),
        "checkpointed": sum(
            bool(getattr(layer, "gradient_checkpointing", False)) for layer in layers
        ),
    }


vlm.CASES["frozen_prefix_parity"] = _case_parity


def _worker(*args):
    vlm._worker(*args)


def spawn(
    case,
    world = 2,
    **kwargs,
):
    # Spawned ranks unpickle this module's _worker, so they import it and see the case registered above.
    worker = vlm._worker
    vlm._worker = _worker
    try:
        return vlm.spawn(case, world, **kwargs)
    finally:
        vlm._worker = worker


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_pp2_with_frozen_prefix_skip_trains_every_stage(schedule, monkeypatch):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    # The ranks inherit the environment: an exported kill switch would let the merge base pass.
    monkeypatch.setenv("UNSLOTH_FROZEN_PREFIX_SKIP", "1")
    results = spawn("frozen_prefix_parity", schedule = schedule)
    bounds = results[0]["bounds"]
    assert bounds[1][0] > 0, bounds
    # The setup the bug needs: rank 1's list starts with stand-ins, its own layers are checkpointed.
    assert results[1]["stand_ins"] == bounds[1][0], results[1]["stand_ins"]
    assert results[1]["checkpointed"] > 0
    # One process trains layer 0, so the skip never engages there; under pp it must stay off on every stage.
    assert results[0]["ref_cuts"] == 0
    assert [res["pp_cuts"] for res in results] == [0, 0]
    # Every stage's gradients (stage 0's included) and the losses match one process at every step.
    print("worst gradient error", vlm.check_parity(results))


def test_stand_in_marks_the_stack_for_the_skip():
    fp = _load("_frozen_prefix")
    assert getattr(ppn._PPPassThrough, "_unsloth_pipeline_stand_in", False)
    model = _checkpointed(tiny.build("gemma4_e2b", layers = LAYERS, causal_lm = True), fp)
    stacks = fp._decoder_stacks(model)
    stack = max(stacks, key = len)
    heads = fp._head_parameters(model)
    # Freeze everything below layer 3 (the per-layer projection too) as a later stage sees it: the cut sits at 3.
    upper = {id(p) for layer in stack[3:] for p in layer.parameters()}
    for p in model.parameters():
        if id(p) not in upper:
            p.requires_grad_(False)
    plan = fp._plan(model, stacks, heads)
    assert plan is not None and plan[1] == 3, plan
    # The same list once layers 0-2 are pipeline stand-ins: no cut.
    ctx = ppn._StageContext(3, LAYERS, False, True, False)
    originals = [stack[i] for i in range(3)]
    for i in range(3):
        stack[i] = ppn._PPPassThrough(i, ctx, originals[i])
    assert fp._plan(model, stacks, heads) is None
    # Stage 0's view (stand-ins only above the cut) and a sibling stack that could still cut: no cut either.
    for i in range(3):
        stack[i] = originals[i]
    tail = ppn._StageContext(0, 4, True, False, False)
    for i in range(4, LAYERS):
        stack[i] = ppn._PPPassThrough(i, tail, stack[i])
    assert fp._plan(model, stacks, heads) is None
    assert fp._plan(model, [list(originals) + list(stack[3:4])] + stacks, heads) is None
