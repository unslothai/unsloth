# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO training-step host syncs: deferred loss metrics, sync-free nanmin / nanmax, one read in the
PrefixGrouper layout build and host lengths carried on packed_seq_lengths. Values must match the old reads."""

import math

import pytest
import torch

import unsloth.models.rl_replacements as rr
import unsloth.utils.packing as packing
from unsloth.utils import prefix_grouper as pg


def _trl_nanmin(tensor):
    # trl.trainer.utils.nanmin
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype = tensor.dtype, device = tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def _trl_nanmax(tensor):
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype = tensor.dtype, device = tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def _same_float(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_nanmin_nanmax_match_trl(dtype):
    g = torch.Generator().manual_seed(0)
    for i in range(400):
        x = torch.randn(int(torch.randint(1, 9, (1,), generator = g)), generator = g).to(dtype)
        if i % 3 == 0:
            x[torch.rand(x.shape, generator = g) < 0.5] = float("nan")
        if i % 11 == 0:
            x[:] = float("nan")
        if i % 13 == 0:
            x[0] = float("inf") if i % 2 else float("-inf")
        for new, old in (
            (rr._unsloth_grpo_nanmin, _trl_nanmin),
            (rr._unsloth_grpo_nanmax, _trl_nanmax),
        ):
            a, b = new(x), old(x)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert _same_float(a.item(), b.item())


class _Trainer:
    def __init__(self):
        self.logged = []

    def log(
        self,
        logs,
        start_time = None,
    ):
        self.logged.append(dict(logs))


class _ItemCounter:
    def __init__(self, monkeypatch):
        self.n = 0
        orig_item, orig_tolist = torch.Tensor.item, torch.Tensor.tolist

        def item(t):
            self.n += 1
            return orig_item(t)

        def tolist(t):
            self.n += 1
            return orig_tolist(t)

        monkeypatch.setattr(torch.Tensor, "item", item)
        monkeypatch.setattr(torch.Tensor, "tolist", tolist)


def test_deferred_metrics_match_item_and_read_once_per_log(monkeypatch):
    g = torch.Generator().manual_seed(1)
    values = [
        torch.randn((), generator = g).to(dt)
        for dt in (torch.float32, torch.bfloat16, torch.float16) * 5
    ]
    values.append(torch.tensor(7))  # an integer tensor stays an int, like .item()
    expected = {"a": [], "b": []}
    for i, v in enumerate(values):
        expected["ab"[i % 2]].append(v.item())

    t = _Trainer()
    metrics = {"a": [], "b": []}
    counter = _ItemCounter(monkeypatch)
    for i, v in enumerate(values):
        rr._unsloth_grpo_defer_metric(t, metrics["ab"[i % 2]], v)
    assert counter.n == 0  # nothing read during the step
    assert metrics == {"a": [], "b": []}
    t.log({"loss": 1.0})
    assert (
        counter.n == 2
    )  # one batched read per (device, float / int) group: the floats, then the int
    assert metrics == expected
    assert [type(x) for x in metrics["a"] + metrics["b"]] == [
        type(x) for x in expected["a"] + expected["b"]
    ]
    assert t.logged == [{"loss": 1.0}]
    t.log({"loss": 2.0})  # nothing pending: no read, original log still runs
    assert counter.n == 2 and len(t.logged) == 2


def test_deferred_metrics_keep_order_and_bound(monkeypatch):
    t = _Trainer()
    target = []
    vals = [torch.tensor(float(i)) for i in range(4096 + 10)]
    for v in vals:
        rr._unsloth_grpo_defer_metric(t, target, v)
    assert target == [float(i) for i in range(4096)]  # bounded queue flushed once
    t.log({})
    assert target == [float(i) for i in range(4096 + 10)]
    rr._unsloth_grpo_defer_metric(t, target, 3.5)  # plain floats append directly
    assert target[-1] == 3.5


def test_compute_loss_has_no_item_calls():
    src = rr.grpo_trainer_compute_loss("compute_loss", "def compute_loss(self): pass")
    assert ".item()" not in src
    assert "nanmin(" not in src.replace("_unsloth_grpo_nanmin(", "")
    assert "nanmax(" not in src.replace("_unsloth_grpo_nanmax(", "")
    assert src.count("_unsloth_grpo_defer_metric(") >= 11


def test_helpers_are_emitted_into_generated_trainer():
    pre = "\n".join(rr.RL_PRE_ITEMS["grpo_trainer"])
    for name in (
        "_unsloth_grpo_nanmin",
        "_unsloth_grpo_nanmax",
        "_unsloth_grpo_flush_metrics",
        "_unsloth_grpo_defer_metric",
    ):
        assert f"def {name}(" in pre


def _old_layout_inputs(input_ids, logits_to_keep, pad_id, left_pad):
    total_rows, L = input_ids.shape
    keep = input_ids != pad_id
    cstart = ((L - logits_to_keep) - left_pad).to(torch.long)
    keep_cpu = keep.tolist()
    real_cols = [[c for c in range(L) if keep_cpu[r][c]] for r in range(total_rows)]
    return input_ids.tolist(), real_cols, cstart.tolist(), int(left_pad.max().item())


def test_build_group_layout_single_read_same_inputs(monkeypatch):
    g = torch.Generator().manual_seed(2)
    seen = []
    orig = pg._build_groups

    def spy(ids_cpu, real_cols_cpu, cstart_cpu, num_generations, total_rows):
        seen.append((ids_cpu, real_cols_cpu, cstart_cpu))
        return orig(ids_cpu, real_cols_cpu, cstart_cpu, num_generations, total_rows)

    monkeypatch.setattr(pg, "_build_groups", spy)
    for trial in range(200):
        G = 2 + trial % 3
        groups = 1 + trial % 3
        P = int(torch.randint(2, 7, (1,), generator = g))
        C = int(torch.randint(2, 7, (1,), generator = g))
        rows = []
        for _ in range(groups):
            prompt = torch.randint(5, 40, (P,), generator = g)
            lp = int(torch.randint(0, P - 1, (1,), generator = g))
            for _ in range(G):
                row = torch.cat([prompt, torch.randint(5, 40, (C,), generator = g)])
                row[:lp] = 0
                cl = int(torch.randint(1, C + 1, (1,), generator = g))
                row[P + cl :] = 0
                if trial % 4 == 1:
                    row[P + cl // 2] = 0  # non-contiguous row -> the per-column fallback
                rows.append(row)
        raw = torch.stack(rows)
        left_pad = (raw[:, :P] == 0).sum(dim = 1)
        seen.clear()
        counter = _ItemCounter(monkeypatch)
        layout = pg.build_group_layout(raw, C, 0, G, left_pad, apply_tokr_gate = False)
        reads = counter.n
        monkeypatch.undo()
        monkeypatch.setattr(pg, "_build_groups", spy)
        ids_cpu, real_cols, cstart, max_lp = _old_layout_inputs(raw, C, 0, left_pad)
        assert len(seen) == 1
        assert seen[0] == (ids_cpu, real_cols, cstart)
        assert reads == 1
        if layout is not None:
            assert layout.W == C + max_lp


def test_attached_host_lengths_skip_the_read(monkeypatch):
    packing.clear_packed_caches()
    lengths = torch.tensor([3, 5, 2], dtype = torch.int32)
    lengths._unsloth_lengths_tuple = (3, 5, 2)
    counter = _ItemCounter(monkeypatch)
    lens, cu, max_seqlen = packing.get_packed_info_from_kwargs(
        {"packed_seq_lengths": lengths}, torch.device("cpu")
    )
    assert counter.n == 0
    assert max_seqlen == 5 and cu.tolist() == [0, 3, 8, 10] and torch.equal(lens, lengths)
    assert packing._host_segment_lengths(lengths) == (3, 5, 2)
    packing.clear_packed_caches()
    # a tuple that does not match the tensor is ignored (read as before)
    bad = torch.tensor([4, 4], dtype = torch.int32)
    bad._unsloth_lengths_tuple = (4,)
    assert (
        packing.get_packed_info_from_kwargs({"packed_seq_lengths": bad}, torch.device("cpu"))[2]
        == 4
    )
    packing.clear_packed_caches()


def test_flush_never_converts_to_float64_on_device(monkeypatch):
    # No float64 on FP64-less XPUs.
    class _Trainer:
        pass

    trainer = _Trainer()
    targets = {dt: [] for dt in (torch.float32, torch.bfloat16, torch.float16, torch.int64)}
    expected = {dt: [] for dt in targets}
    g = torch.Generator().manual_seed(5)
    trainer._unsloth_pending_metrics = []
    for dt, target in targets.items():
        for _ in range(7):
            v = (torch.randn((), generator = g) * 100).to(dt)
            expected[dt].append(v.item())
            trainer._unsloth_pending_metrics.append((target, v))
    requested = []
    orig_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        dtypes = [a for a in args if isinstance(a, torch.dtype)] + [kwargs.get("dtype")]
        requested.extend(d for d in dtypes if d is not None)
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    rr._unsloth_grpo_flush_metrics(trainer)
    assert torch.float64 not in requested
    for dt in targets:
        assert targets[dt] == expected[dt]
        assert all(type(x) is type(y) for x, y in zip(targets[dt], expected[dt]))


def test_deferred_metrics_do_not_keep_the_trainer_alive():
    # `del trainer` must free it without gc.collect (no reference cycle).
    import gc
    import weakref

    class _Owner(_Trainer):
        pass

    gc.disable()
    try:
        t = _Owner()
        target = []
        rr._unsloth_grpo_defer_metric(t, target, torch.tensor(1.5))
        t.log({})
        assert target == [1.5]
        rr._unsloth_grpo_defer_metric(t, target, torch.tensor(2.5))  # pending again
        ref = weakref.ref(t)
        del t
        assert ref() is None
    finally:
        gc.enable()
