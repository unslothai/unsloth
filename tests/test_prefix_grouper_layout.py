# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PrefixGrouper layout: the single-copy builder matches the token-by-token reference, and its
host / device traffic does not grow with the number of groups or segments."""

import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest
import torch
from real_accelerator import has_real_cuda

_UTILS = Path(__file__).resolve().parents[1] / "unsloth" / "utils"


def _load():
    pkg_name = "_pg_layout_test_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_UTILS)]
        sys.modules[pkg_name] = pkg
    mods = {}
    for name in ("prefix_grouper_kernel", "prefix_grouper"):
        full = f"{pkg_name}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, _UTILS / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
        mods[name] = sys.modules[full]
    return mods["prefix_grouper"], mods["prefix_grouper_kernel"]


pg, pgk = _load()

_LAYOUT_TENSORS = ("flat_ids", "position_ids", "tgt_rows", "tgt_cols", "tgt_pred", "tgt_flat")
_SEG_TENSORS = ("group_of_kv", "is_prefix", "suffix_of_kv")


def _assert_same(ref, got):
    if ref is None or got is None:
        assert ref is None and got is None
        return
    for name in _LAYOUT_TENSORS:
        a, b = getattr(ref, name), getattr(got, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device, name
        assert torch.equal(a, b), name
        assert b.is_contiguous() and b.storage_offset() == 0, name
    for name in ("total_rows", "L", "W", "tok_r", "signature"):
        assert getattr(ref, name) == getattr(got, name), name
    for name in _SEG_TENSORS:
        a, b = getattr(ref.prefix_seg_info, name), getattr(got.prefix_seg_info, name)
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), name
        assert b.is_contiguous() and b.storage_offset() == 0, name
    for name in ("signature", "T", "T_pad"):
        assert getattr(ref.prefix_seg_info, name) == getattr(got.prefix_seg_info, name), name


def _batch(
    rng,
    num_groups,
    G,
    P,
    C,
    pad = 0,
    left_packed = True,
):
    """GRPO-shaped rows: G completions per prompt, left pad in the prompt, right pad after
    each completion; left_packed shifts every real run to column 0 like the de-padded input."""
    L = P + C
    ids = torch.randint(1, 50, (num_groups * G, L), generator = rng)
    ids[ids == pad] = pad + 1
    lp = torch.zeros(num_groups * G, dtype = torch.long)
    for g in range(num_groups):
        rows = range(g * G, (g + 1) * G)
        plen = int(torch.randint(1, P + 1, (1,), generator = rng))
        prompt = ids[rows[0], :P].clone()
        for r in rows:
            ids[r, :P] = prompt
            ids[r, : P - plen] = pad
            lp[r] = P - plen
            clen = int(torch.randint(0, C + 1, (1,), generator = rng))
            ids[r, P + clen :] = pad
            if left_packed:
                k = P - plen
                ids[r] = torch.cat([ids[r, k:], torch.full((k,), pad, dtype = ids.dtype)])
    return ids, lp


def _cases():
    rng = torch.Generator().manual_seed(3407)
    pick = random.Random(3407)
    for i in range(400):
        G = pick.choice([2, 3, 4, 8])
        num_groups = pick.choice([1, 2, 3])
        P = pick.choice([1, 4, 17, 64])
        C = pick.choice([1, 5, 33, 96])
        pad = pick.choice([0, 7])
        ids, lp = _batch(rng, num_groups, G, P, C, pad, left_packed = pick.random() < 0.7)
        kind = pick.choice(["plain"] * 6 + ["hole", "other_prompt", "lp_shift", "int32", "bad_G"])
        if kind == "hole":  # a pad token inside a real run: not one contiguous run
            ids[pick.randrange(ids.shape[0]), pick.randrange(ids.shape[1])] = pad
        elif kind == "other_prompt" and ids.shape[0] > 1:
            ids[1, 0] = (int(ids[1, 0]) % 40) + 3
        elif kind == "lp_shift":
            lp = lp + torch.randint(-2, 3, lp.shape, generator = rng)
        elif kind == "int32":
            ids, lp = ids.to(torch.int32), lp.to(torch.int32)
        num_gen = G
        if kind == "bad_G":
            num_gen = pick.choice([None, 1, G + 1])
        ltk = C + pick.choice([0, 0, 1])
        kwargs = dict(
            apply_tokr_gate = pick.random() < 0.3,
            max_segment_cap = pick.choice([None, None, P + C // 2]),
        )
        yield i, (ids, ltk, pad, num_gen, lp), kwargs


def test_layout_matches_token_by_token_reference():
    built = 0
    for i, args, kwargs in _cases():
        ref = pg._build_group_layout_reference(*args, **kwargs)
        got = pg.build_group_layout(*args, **kwargs)
        _assert_same(ref, got)
        built += got is not None
    assert built >= 100


def test_seg_info_matches_per_segment_build():
    specs = [(5, [3, 0, 7]), (1, [1, 1]), (130, [2, 9, 0, 4])]
    seg, meta = pgk.build_seg_info_multigroup(specs, torch.device("cpu"))
    group_of, is_prefix, suffix_of = [], [], []
    counter = 0
    for gid, (P, Rs) in enumerate(specs):
        group_of += [gid] * P
        is_prefix += [True] * P
        suffix_of += [-1] * P
        for r in Rs:
            group_of += [gid] * r
            is_prefix += [False] * r
            suffix_of += [counter] * r
            counter += 1
    T = len(group_of)
    T_pad = pgk._pad_len(T)
    pad = T_pad - T
    assert (seg.T, seg.T_pad) == (T, T_pad)
    assert torch.equal(seg.group_of_kv, torch.tensor(group_of + [pgk._PAD_GROUP] * pad))
    assert torch.equal(seg.is_prefix, torch.tensor(is_prefix + [False] * pad))
    assert torch.equal(seg.suffix_of_kv, torch.tensor(suffix_of + [pgk._PAD_GROUP] * pad))
    assert seg.signature == ("multi", T_pad, tuple((P, tuple(Rs)) for P, Rs in specs))
    assert meta[2]["prefix_last_index"] == meta[2]["base"] + 129
    assert meta[0]["suffix_slices"] == [(5, 8), (8, 8), (8, 15)]


class _CountFactories(torch.overrides.TorchFunctionMode):
    """Counts tensor creations that name a device (each one a fill kernel or a host to
    device copy when that device is an accelerator)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_function__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        kwargs = kwargs or {}
        if kwargs.get("device") is not None and func in (
            torch.tensor,
            torch.full,
            torch.ones,
            torch.zeros,
        ):
            self.calls += 1
        return func(*args, **kwargs)


def _factory_calls(num_groups, G):
    rng = torch.Generator().manual_seed(0)
    ids, lp = _batch(rng, num_groups, G, 64, 16)
    with _CountFactories() as mode:
        layout = pg.build_group_layout(ids, 16, 0, G, lp, apply_tokr_gate = False)
    assert layout is not None
    return mode.calls


def test_device_tensor_creations_do_not_grow_with_groups():
    # The per-segment build made three device tensors per prefix and per completion.
    assert _factory_calls(1, 8) == _factory_calls(6, 8) == _factory_calls(6, 2)


# Not torch.cuda.is_available(): the CUDA spoof makes it answer True on CPU runners.
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_cuda_layout_reads_the_device_once_and_copies_once():
    import warnings

    rng = torch.Generator().manual_seed(0)
    ids, lp = _batch(rng, 4, 8, 256, 128)
    ids, lp = ids.cuda(), lp.cuda()
    ref = pg._build_group_layout_reference(ids, 128, 0, 8, lp)
    got = pg.build_group_layout(ids, 128, 0, 8, lp)
    _assert_same(ref, got)

    def syncs(fn):
        torch.cuda.synchronize()
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                fn(ids, 128, 0, 8, lp)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        # Count the per-operation reports, not set_sync_debug_mode's "prototype feature" notice.
        return [str(w.message) for w in caught if "called a synchronizing" in str(w.message)]

    # The counter sees the token-by-token build's reads (cstart, ids, contiguity, ...).
    assert len(syncs(pg._build_group_layout_reference)) > 2
    # One device to host read of the ids + one host to device copy of the index tables.
    got_syncs = syncs(pg.build_group_layout)
    assert len(got_syncs) <= 2, got_syncs


def _outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:
        return ("err", type(exc), str(exc))


def test_unusual_arguments_take_the_reference_decision():
    # A float num_generations must raise the reference's error, a numpy one must keep a
    # plain int signature, and a pad id outside the ids dtype must compare like the reference.
    np = pytest.importorskip("numpy")
    rng = torch.Generator().manual_seed(0)
    ids, lp = _batch(rng, 2, 4, 32, 16)
    cases = [
        ((ids, 16, 0, 4.0, lp), {}),
        ((ids, 16, 0, np.int64(4), lp), dict(apply_tokr_gate = False)),
        ((ids.to(torch.int32), 16, 2**32, 4, lp), dict(apply_tokr_gate = False)),
    ]
    for args, kwargs in cases:
        ref = _outcome(pg._build_group_layout_reference, *args, **kwargs)
        got = _outcome(pg.build_group_layout, *args, **kwargs)
        assert ref[0] == got[0], (ref, got)
        if ref[0] == "err":
            assert ref[1:] == got[1:]
        else:
            _assert_same(ref[1], got[1])
            if ref[1] is not None:
                assert type(got[1].signature[0]) is type(ref[1].signature[0])


def test_host_tables_ignore_the_default_device():
    rng = torch.Generator().manual_seed(0)
    ids, lp = _batch(rng, 3, 4, 32, 16)
    ref = pg._build_group_layout_reference(ids, 16, 0, 4, lp, apply_tokr_gate = False)
    torch.set_default_device("meta")
    try:
        got = pg.build_group_layout(ids, 16, 0, 4, lp, apply_tokr_gate = False)
    finally:
        torch.set_default_device(None)
    _assert_same(ref, got)
