# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""GPU-free tests for the Qwen3.5 compiled-RoPE seq_len==1 guard (#4801).

GRPO scoring runs a compiled Qwen3.5 forward one token at a time (seq_len == 1)
during per-token log-prob scoring. transformers' apply_rotary_pos_emb rebuilds
q/k by torch.cat-ing the rotated rotary slice back onto the untouched
partial_rotary_factor pass-through slice; that concat's shape guard can blow up
under torch.compile once seq_len collapses to 1. The guard in
unsloth/models/_utils.py wraps the compiled module's apply_rotary_pos_emb so a
"must match" RuntimeError at seq_len == 1 falls back to a shape-safe
implementation (indexed assignment instead of torch.cat) rather than crashing
the eval step.

`import unsloth` pulls in triton via unsloth/_gpu_init.py, which is unavailable
on this Windows dev box (and CI's GPU-free harness stubs the accelerator, not
triton). So -- matching tests/test_fast_generate_slow_guard.py's convention --
these tests extract the target functions straight out of _utils.py's source via
ast and exec them into an isolated namespace instead of importing the package.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
_SRC = open(UTILS).read()

_FUNCTION_NAMES = (
    "_unsloth_compile_cache_leaves",
    "_rope_seq_len_1_fallback",
    "_wrap_qwen3_5_rope",
    "_patch_qwen3_5_rope_seq_len_1",
)


def _load_namespace():
    ns = {"torch": torch, "functools": functools, "os": os, "sys": sys}
    found = set()
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCTION_NAMES:
            exec(ast.get_source_segment(_SRC, node), ns)
            found.add(node.name)
    missing = set(_FUNCTION_NAMES) - found
    if missing:
        raise AssertionError(f"not found in _utils.py: {sorted(missing)}")
    return ns


_ns = _load_namespace()
_unsloth_compile_cache_leaves = _ns["_unsloth_compile_cache_leaves"]
_rope_seq_len_1_fallback = _ns["_rope_seq_len_1_fallback"]
_wrap_qwen3_5_rope = _ns["_wrap_qwen3_5_rope"]
_patch_qwen3_5_rope_seq_len_1 = _ns["_patch_qwen3_5_rope_seq_len_1"]


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim = -1)


def _naive_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    # Mirrors transformers.models.qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb:
    # the naive torch.cat-based recombination this guard exists to work around.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim = -1)
    k_embed = torch.cat([k_embed, k_pass], dim = -1)
    return q_embed, k_embed


def _make_qkv(batch, heads, seq_len, head_dim, rotary_dim):
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    cos = torch.randn(batch, seq_len, rotary_dim)
    sin = torch.randn(batch, seq_len, rotary_dim)
    return q, k, cos, sin


# --------------------------------------------------------------------------
# Static checks
# --------------------------------------------------------------------------


def test_qwen3_5_rope_patch_registered():
    # The patch must actually run: _run_temporary_patches only fires functions
    # present in TEMPORARY_PATCHES.
    assert "TEMPORARY_PATCHES.append(_patch_qwen3_5_rope_seq_len_1)" in _SRC


def test_patch_uses_compile_cache_leaves():
    # Provenance-gating must reuse the same leaf-directory check as
    # _forward_is_unsloth_compiled, not a bespoke path heuristic, so the two stay
    # in sync as UNSLOTH_COMPILE_LOCATION overrides evolve.
    src = ast.get_source_segment(
        _SRC,
        next(
            node
            for node in ast.parse(_SRC).body
            if isinstance(node, ast.FunctionDef) and node.name == "_patch_qwen3_5_rope_seq_len_1"
        ),
    )
    assert "_unsloth_compile_cache_leaves" in src


# --------------------------------------------------------------------------
# Reproduction: the bug this guard exists to catch
# --------------------------------------------------------------------------


def test_naive_concat_raises_at_seq_len_1():
    # Simulates the compiled-graph shape mismatch: at seq_len == 1 the rotated
    # slice and the partial_rotary_factor pass-through slice can disagree outside
    # the concat dim (eg a stale cached buffer), which is exactly the "Sizes of
    # tensors must match except in dimension ..." RuntimeError the wrapper guards.
    batch, heads, head_dim = 2, 4, 8
    rotary_dim = 4  # partial_rotary_factor == 0.5
    seq_len = 1

    q_embed = torch.randn(batch, heads, seq_len, rotary_dim)
    q_pass = torch.randn(batch, heads, seq_len + 1, head_dim - rotary_dim)

    with pytest.raises(RuntimeError, match = "must match"):
        torch.cat([q_embed, q_pass], dim = -1)


# --------------------------------------------------------------------------
# Behavioral: the fallback itself
# --------------------------------------------------------------------------


def test_fallback_raises_on_empty_cos_sin():
    batch, heads, head_dim, rotary_dim = 2, 4, 8, 4
    q = torch.randn(batch, heads, 1, head_dim)
    k = torch.randn(batch, heads, 1, head_dim)
    cos = torch.randn(batch, 0, rotary_dim)
    sin = torch.randn(batch, 0, rotary_dim)

    with pytest.raises(RuntimeError, match = "empty cos/sin"):
        _rope_seq_len_1_fallback(q, k, cos, sin, unsqueeze_dim = 1)


def test_guarded_rope_handles_seq_len_1():
    batch, heads, head_dim, rotary_dim = 2, 4, 8, 4
    q, k, cos, sin = _make_qkv(batch, heads, 1, head_dim, rotary_dim)

    q_out, k_out = _rope_seq_len_1_fallback(q, k, cos, sin, unsqueeze_dim = 1)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_guarded_rope_matches_naive_for_seq_len_gt_1():
    batch, heads, head_dim, rotary_dim = 2, 4, 8, 4
    seq_len = 5
    q, k, cos, sin = _make_qkv(batch, heads, seq_len, head_dim, rotary_dim)

    naive_q, naive_k = _naive_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1)
    guarded_q, guarded_k = _rope_seq_len_1_fallback(q, k, cos, sin, unsqueeze_dim = 1)

    assert torch.allclose(naive_q, guarded_q)
    assert torch.allclose(naive_k, guarded_k)


def test_patch_noop_when_module_absent():
    # No qwen3_5 module loaded (eg a training run that never touches Qwen3.5)
    # must not raise or otherwise disturb sys.modules.
    stray = [name for name in sys.modules if "qwen3_5" in name]
    for name in stray:
        del sys.modules[name]

    before = dict(sys.modules)
    _patch_qwen3_5_rope_seq_len_1(phase = "post_compile")
    assert dict(sys.modules) == before


# --------------------------------------------------------------------------
# Extra coverage: provenance gating end-to-end, and the wrapper's dispatch
# --------------------------------------------------------------------------


def test_patch_wraps_only_compiled_cache_copy(monkeypatch):
    # A fake "qwen3_5" module living outside the compile cache is left alone; one
    # living inside it (matching _unsloth_compile_cache_leaves()) gets its
    # apply_rotary_pos_emb wrapped and marked.
    leaf = sorted(_unsloth_compile_cache_leaves())[0]

    stock = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    stock.__file__ = "/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
    stock.apply_rotary_pos_emb = _naive_apply_rotary_pos_emb

    compiled = types.ModuleType("unsloth_compiled_module_qwen3_5")
    compiled.__file__ = f"/tmp/{leaf}/modeling_qwen3_5.py"
    compiled.apply_rotary_pos_emb = _naive_apply_rotary_pos_emb

    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", stock)
    monkeypatch.setitem(sys.modules, "unsloth_compiled_module_qwen3_5", compiled)

    _patch_qwen3_5_rope_seq_len_1(phase = "post_compile")

    assert stock.apply_rotary_pos_emb is _naive_apply_rotary_pos_emb  # untouched
    assert getattr(compiled.apply_rotary_pos_emb, "_unsloth_seqlen1_patched", False)


def test_wrapped_rope_falls_back_only_on_seq_len_1_shape_error():
    wrapped = _wrap_qwen3_5_rope(_naive_apply_rotary_pos_emb)
    assert wrapped._unsloth_seqlen1_patched is True

    batch, heads, head_dim, rotary_dim = 2, 4, 8, 4
    q, k, cos, sin = _make_qkv(batch, heads, 5, head_dim, rotary_dim)
    q_out, k_out = wrapped(q, k, cos, sin, unsqueeze_dim = 1)
    naive_q, naive_k = _naive_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1)
    assert torch.allclose(q_out, naive_q)
    assert torch.allclose(k_out, naive_k)

    def _broken_apply(q, k, cos, sin, unsqueeze_dim = 1):
        raise RuntimeError("Sizes of tensors must match except in dimension 3")

    wrapped_broken = _wrap_qwen3_5_rope(_broken_apply)
    q1, k1, cos1, sin1 = _make_qkv(batch, heads, 1, head_dim, rotary_dim)
    q_out, k_out = wrapped_broken(q1, k1, cos1, sin1, unsqueeze_dim = 1)
    assert q_out.shape == q1.shape
    assert k_out.shape == k1.shape

    def _broken_apply_other_error(q, k, cos, sin, unsqueeze_dim = 1):
        raise RuntimeError("out of memory")

    wrapped_oom = _wrap_qwen3_5_rope(_broken_apply_other_error)
    with pytest.raises(RuntimeError, match = "out of memory"):
        wrapped_oom(q1, k1, cos1, sin1, unsqueeze_dim = 1)


# --------------------------------------------------------------------------
# Upstream source canary: detect when transformers fixes the vulnerable concat
# --------------------------------------------------------------------------


def test_upstream_source_canary():
    # Skip if transformers doesn't ship Qwen3.5 yet (too old).
    if importlib.util.find_spec("transformers.models.qwen3_5") is None:
        pytest.skip("transformers has no qwen3_5 module (too old)")

    try:
        mod = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
    except Exception as exc:
        pytest.skip(f"could not import qwen3_5 modeling module: {exc!r}")

    fn = getattr(mod, "apply_rotary_pos_emb", None)
    if fn is None:
        pytest.skip("apply_rotary_pos_emb not found in qwen3_5 module")

    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        pytest.skip(f"could not inspect apply_rotary_pos_emb source: {exc!r}")

    # The vulnerable pattern: torch.cat recombines the rotary-embedded slice
    # with the passthrough slice.  When upstream removes this concat in favor
    # of a shape-safe alternative, our temporary _wrap_qwen3_5_rope guard can
    # be retired.
    has_vulnerable_concat = (
        "torch.cat" in src
        and "embed" in src
        and "pass" in src
        and "dim" in src
    )

    if not has_vulnerable_concat:
        pytest.fail(
            "DRIFT DETECTED: upstream transformers' Qwen3.5 "
            "apply_rotary_pos_emb no longer uses the vulnerable "
            "torch.cat([...embed, ...pass], dim=-1) concat pattern. "
            "The temporary _wrap_qwen3_5_rope seq_len==1 guard and "
            "_patch_qwen3_5_rope_seq_len_1 can be removed."
        )
