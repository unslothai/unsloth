# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block CUDA graphs (UNSLOTH_BLOCK_CUDAGRAPHS=1) on a real GPU with real cudagraph trees.

Fixed shape without checkpointing: replays, no copies out of the pool, weights match eager over
gradient accumulation. Reentrant checkpointing: still copies, still matches. Checkpointed layers
between layers that keep their activations: the recompute in backward leaves the kept layers' saved
tensors alive. Variable lengths: graphs go off past the cap and the graph pool is released (it stayed
reserved: +3 to 4.75 GB on a T4)."""

import importlib.util
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU")

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models"
)
H = 512


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ublock_compile_cg_gpu", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load()


def _forward(self, hidden_states, **kwargs):
    x = self.norm(hidden_states)
    return hidden_states + self.down(torch.nn.functional.silu(self.up(x)))


def _init(self):
    torch.nn.Module.__init__(self)
    self.norm = torch.nn.LayerNorm(H)
    self.up = torch.nn.Linear(H, 4 * H)
    self.down = torch.nn.Linear(4 * H, H)


class EagerBlock(torch.nn.Module):
    __init__ = _init
    forward = _forward


class GpuCgDecoderLayer(torch.nn.Module):
    __init__ = _init
    forward = _forward


bc._wrap(GpuCgDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})


class Counters:
    def __init__(self, monkeypatch):
        from torch._inductor import cudagraph_trees as T

        self.record = self.replay = self.clones = 0
        manager = T.CUDAGraphTreeManager
        record, execute = manager.record_function, manager.execute_node

        def counted_record(this, *a, **k):
            self.record += 1
            return record(this, *a, **k)

        def counted_execute(this, *a, **k):
            self.replay += 1
            return execute(this, *a, **k)

        monkeypatch.setattr(manager, "record_function", counted_record)
        monkeypatch.setattr(manager, "execute_node", counted_execute)
        clone = bc._clone_outputs

        def counted_clone(out):
            self.clones += 1
            return clone(out)

        monkeypatch.setattr(bc, "_clone_outputs", counted_clone)


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)
    torch._dynamo.reset()  # also resets the cudagraph trees
    if hasattr(bc, "_CUDAGRAPH_STATE"):
        monkeypatch.setattr(
            bc,
            "_CUDAGRAPH_STATE",
            getattr(bc, "_new_cudagraph_state", dict)(),
        )
    yield
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def _models(n = 3):
    torch.manual_seed(0)
    graphed = torch.nn.ModuleList(GpuCgDecoderLayer() for _ in range(n)).cuda().train()
    eager = torch.nn.ModuleList(EagerBlock() for _ in range(n)).cuda().train()
    eager.load_state_dict(graphed.state_dict())
    return graphed, eager


def _step(layers, xs, opt, checkpoint):
    from torch.utils.checkpoint import checkpoint as ckpt

    # checkpoint: one flag for every layer, or one per layer.
    flags = (
        tuple(checkpoint) if isinstance(checkpoint, (tuple, list)) else (checkpoint,) * len(layers)
    )
    losses = []
    for x in xs:  # gradient accumulation: the first .grad of a micro step lives across generations
        y = x
        for layer, flag in zip(layers, flags):
            y = ckpt(layer, y, use_reentrant = True) if flag else layer(y)
        loss = y.float().square().mean()
        loss.backward()
        losses.append(loss.item())
    opt.step()
    opt.zero_grad(set_to_none = True)
    return losses


def _train(
    lengths,
    checkpoint,
    micro = 2,
    steps = None,
):
    graphed, eager = _models(len(checkpoint) if isinstance(checkpoint, (tuple, list)) else 3)
    opts = [torch.optim.SGD(m.parameters(), lr = 0.1) for m in (graphed, eager)]
    out = {"graphed": [], "eager": []}
    any_checkpoint = any(checkpoint) if isinstance(checkpoint, (tuple, list)) else checkpoint
    for step, L in enumerate(lengths):
        g = torch.Generator(device = "cuda").manual_seed(step)
        xs = [
            torch.randn(2, L, H, device = "cuda", generator = g).requires_grad_(any_checkpoint)
            for _ in range(micro)
        ]
        out["graphed"].append(_step(graphed, xs, opts[0], checkpoint))
        out["eager"].append(
            _step(
                eager,
                [x.detach().requires_grad_(any_checkpoint) for x in xs],
                opts[1],
                checkpoint,
            )
        )
        if steps is not None:
            steps(step)
    return graphed, eager, out


def _assert_same(graphed, eager, out):
    torch.testing.assert_close(out["graphed"], out["eager"], rtol = 2e-3, atol = 2e-4)
    for (name, p), q in zip(graphed.named_parameters(), eager.parameters()):
        torch.testing.assert_close(p, q, rtol = 2e-3, atol = 2e-4, msg = name)


def test_fixed_shape_replays_without_pool_copies(monkeypatch):
    c = Counters(monkeypatch)
    seen = []
    graphed, eager, out = _train(
        [64] * 6, checkpoint = False, steps = lambda s: seen.append((c.record, c.replay))
    )
    _assert_same(graphed, eager, out)
    assert seen[-1][1] > seen[2][1], seen  # replays keep coming
    assert seen[-1][0] == seen[2][0], seen  # and nothing re-records after warm-up
    assert c.clones == 0  # outputs stay in the pool inside one generation


def test_reentrant_checkpoint_still_copies_and_matches(monkeypatch):
    c = Counters(monkeypatch)
    graphed, eager, out = _train([64] * 5, checkpoint = True)
    _assert_same(graphed, eager, out)
    assert c.replay > 0 and c.clones > 0


def test_checkpointed_layers_between_kept_layers_keep_saved_tensors(monkeypatch):
    # Backward runs kept 3, recompute 2, kept 1, recompute 0: the recompute of layer 2 must not
    # start a new tree generation while layer 1 still waits for its backward (its saved tensors
    # would be freed: "accessing tensor output of CUDAGraphs that has been overwritten").
    c = Counters(monkeypatch)
    graphed, eager, out = _train([64] * 6, checkpoint = (True, False, True, False))
    _assert_same(graphed, eager, out)
    assert c.record > 0 and c.replay > 0
    assert not bc._CUDAGRAPH_STATE.get("kept")  # every kept layer ran its backward


def test_variable_lengths_turn_graphs_off_and_release_the_pool(monkeypatch):
    monkeypatch.setattr(bc, "_CUDAGRAPH_MAX_SHAPES", 2)
    c = Counters(monkeypatch)
    records = []

    def after(step):
        torch.cuda.synchronize()
        records.append(c.record)

    lengths = [256, 256, 384, 384, 512, 256, 640, 768]
    torch.cuda.empty_cache()
    base = torch.cuda.memory_reserved()
    graphed, eager, out = _train(lengths, checkpoint = False, steps = after)
    _assert_same(graphed, eager, out)
    assert records[3] > 0, records
    # The third distinct length (step 4) turns graphs off for good: nothing recorded afterwards.
    assert records[-1] == records[3], records
    del graphed, eager, out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    left = torch.cuda.memory_reserved() - base
    # The graph pool is gone: what stays reserved is at most allocator slack.
    assert left < 64 * 2**20, left / 2**20
    state = getattr(bc, "_CUDAGRAPH_STATE", None)
    if state is not None:
        assert state["off"] == "shapes" and state["release"] == "done", state
