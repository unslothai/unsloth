# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU checks of the FlexAttention prefix-trie mask (TreeSegInfo) against TRL's tree attention definition:
each segment's queries attend to the K/V of its root-to-segment path, causally at the end."""

from __future__ import annotations

import importlib.util
import os
import random
import sys

import pytest
import torch

PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "unsloth", "utils", "prefix_grouper_kernel.py"
)


def _kernel():
    name = "_pg_kernel_tree_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        sys.modules.pop(name, None)
        pytest.skip(f"flex attention unavailable: {error}")
    return module


def _random_tree(rng, n_segments):
    # DFS pre-order trie as TRL's TreeRow.pack lays it out: offsets, parents, and the gathered K/V paths.
    parents, children = [], {}
    stack = []
    for s in range(n_segments):
        while stack and rng.random() < 0.4:
            stack.pop()
        parent = stack[-1] if stack else -1
        parents.append(parent)
        stack.append(s)
    lengths = [rng.randint(1, 9) for _ in parents]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    paths = []
    for s, parent in enumerate(parents):
        own = list(range(offsets[s], offsets[s + 1]))
        paths.append((paths[parent] if parent >= 0 else []) + own)
    return offsets, parents, paths


def _trl_reference(q, k, v, offsets, paths):
    # TRL tree_attention: per segment, varlen causal attention aligned bottom-right over its gathered path.
    out = torch.zeros_like(q)
    for s, path in enumerate(paths):
        index = torch.tensor(path)
        qs = q[:, :, offsets[s] : offsets[s + 1]]
        ks, vs = k[:, :, index], v[:, :, index]
        n_q, n_k = qs.shape[2], ks.shape[2]
        mask = torch.ones(n_q, n_k, dtype = torch.bool).tril(n_k - n_q)
        out[:, :, offsets[s] : offsets[s + 1]] = torch.nn.functional.scaled_dot_product_attention(
            qs, ks, vs, attn_mask = mask
        )
    return out


@pytest.mark.parametrize("seed", range(6))
def test_tree_mask_matches_trl_tree_attention(seed):
    kernel = _kernel()
    rng = random.Random(seed)
    offsets, parents, paths = _random_tree(rng, rng.randint(2, 12))
    seg = kernel.build_tree_seg_info(
        torch.tensor(offsets), torch.tensor(parents), torch.device("cpu")
    )
    T = offsets[-1]
    assert seg.T == T and seg.T_pad % 128 == 0
    torch.manual_seed(seed)
    q, k, v = (torch.randn(1, 4, T, 16, dtype = torch.float64, requires_grad = True) for _ in range(3))
    ref = _trl_reference(q, k, v, offsets, paths)
    # Forward only: flex has no CPU backward.
    block_mask = kernel.get_block_mask(seg, torch.device("cpu"), compile_mask = False)
    with torch.no_grad():
        got = kernel._run_flex(
            q.float(), k.float(), v.float(), block_mask, False, None, False, T, seg.T_pad
        )
    torch.testing.assert_close(got.double(), ref.detach(), atol = 1e-5, rtol = 1e-5)
    idx = torch.arange(T)
    dense = kernel._make_tree_mask_mod(seg.reach_end)(0, 0, idx[:, None], idx[None, :])
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = dense)
    torch.testing.assert_close(out, ref, atol = 1e-10, rtol = 1e-10)
    grad = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(ref, (q, k, v), grad)
    got_grads = torch.autograd.grad(out, (q, k, v), grad)
    for a, b in zip(got_grads, ref_grads):
        torch.testing.assert_close(a, b, atol = 1e-10, rtol = 1e-10)


def test_tree_mask_dense_form():
    # Root [0, 3); its first child [3, 5) with a child [5, 6); its second child [6, 8).
    kernel = _kernel()
    seg = kernel.build_tree_seg_info(
        torch.tensor([0, 3, 5, 6, 8]), torch.tensor([-1, 0, 1, 0]), torch.device("cpu")
    )
    assert seg.reach_end[:8].tolist() == [8, 8, 8, 6, 6, 6, 8, 8]
    assert (seg.reach_end[8:] == -1).all()
    mask_mod = kernel._make_tree_mask_mod(seg.reach_end)
    idx = torch.arange(seg.T_pad)
    dense = mask_mod(0, 0, idx[:, None], idx[None, :])
    # The second child (6, 7) never sees the first child's subtree (3-5); the grandchild (5) sees 0-5.
    assert dense[7, :8].tolist() == [True, True, True, False, False, False, True, True]
    assert dense[5, :8].tolist() == [True, True, True, True, True, True, False, False]
    assert not dense[8:].any() and not dense[:, 8:].any()


@pytest.mark.parametrize(
    "offsets, parents",
    [
        ([0, 2, 4, 6], [-1, 0, 2]),  # parent after child
        ([0, 2, 4, 6, 8], [-1, 0, -1, 1]),  # child of a closed subtree
        ([0, 2, 2], [-1, 0]),  # empty segment
        ([0, 2], [-1, -1]),  # offsets / parents mismatch
    ],
)
def test_tree_seg_info_rejects_non_preorder(offsets, parents):
    kernel = _kernel()
    with pytest.raises(ValueError):
        kernel.build_tree_seg_info(
            torch.tensor(offsets), torch.tensor(parents), torch.device("cpu")
        )
