# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A traced RoPE launch never passes a seq_len-dependent stride (dK as [b, h, d, s]) as a constexpr.

CPU only: fake tensors with a symbolic seq dim and a recording launcher."""

import pytest
import torch

pytest.importorskip("triton")

from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402
from torch.fx.experimental.symbolic_shapes import (  # noqa: E402
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
    is_concrete_int,
)

from unsloth.kernels import rope_embedding as RE  # noqa: E402

B, H, S, D = 2, 4, 13, 16


class _Recorder:
    def __init__(self):
        self.launches = []

    def __call__(self, kernel):
        def grid(_grid):
            def launch(*args, **kwargs):
                self.launches.append((kernel, args, kwargs))

            return launch

        return type("_Grid", (), {"__getitem__": lambda _self, g: grid(g)})()


def _bound(kernel, args, kwargs):
    names = [p.name for p in kernel.params]
    values = dict(zip(names, args))
    values.update(kwargs)
    return {p.name: values[p.name] for p in kernel.params if p.is_constexpr}


def _fake_seq_dynamic(mode, t, seq_dim):
    sizes = [DimDynamic.STATIC] * t.dim()
    sizes[seq_dim] = DimDynamic.DYNAMIC
    return mode.from_tensor(t, symbolic_context = StatelessSymbolicContext(dynamic_sizes = sizes))


def _launch(layout_of_k, backward):
    mode = FakeTensorMode(shape_env = ShapeEnv())
    Q = _fake_seq_dynamic(mode, torch.empty(B, S, H, D), 1).transpose(1, 2)
    if layout_of_k == "bhds":
        # [b, h, d, s] storage seen as [b, h, s, d]: dim stride = seq_len.
        K = _fake_seq_dynamic(mode, torch.empty(B, H, D, S), 3).transpose(2, 3)
    else:
        K = _fake_seq_dynamic(mode, torch.empty(B, S, H, D), 1).transpose(1, 2)
    with mode:
        Q_out = torch.empty(Q.transpose(1, 2).shape).transpose(1, 2)
        K_out = torch.empty(K.transpose(1, 2).shape).transpose(1, 2)
        cos = torch.empty(64, D)
        rec = _Recorder()
        RE._rope_launch(Q, Q_out, K, K_out, cos, cos, None, False, backward, False, rec)
    assert len(rec.launches) == 1
    return rec.launches[0], K


@pytest.fixture(autouse = True)
def _no_device_queries(monkeypatch):
    monkeypatch.setattr(RE, "_rope_num_warps", lambda device: 4)


@pytest.mark.parametrize("backward", [False, True])
def test_seq_major_dK_never_reaches_a_constexpr(backward):
    (kernel, args, kwargs), K = _launch("bhds", backward)
    assert not is_concrete_int(K.stride(3)), "the setup must give K a symbolic dim stride"
    symbolic = {
        name: value
        for name, value in _bound(kernel, args, kwargs).items()
        if isinstance(value, torch.SymInt) and not is_concrete_int(value)
    }
    assert not symbolic, f"{kernel.fn.__name__}: symbolic constexpr args {symbolic}"
    assert kernel is RE._rope_embedding_QK_strided


@pytest.mark.parametrize("backward", [False, True])
def test_projection_layouts_keep_the_constexpr_kernel(backward):
    (kernel, args, kwargs), _ = _launch("bshd", backward)
    assert kernel is RE._rope_embedding_QK
    constexprs = _bound(kernel, args, kwargs)
    for name in RE._LAYOUT_STRIDES:
        assert type(constexprs[name]) is int, (name, constexprs[name])


def test_strided_kernel_has_runtime_layout_strides():
    fixed = {p.name: p.is_constexpr for p in RE._rope_embedding_QK.params}
    strided = {p.name: p.is_constexpr for p in RE._rope_embedding_QK_strided.params}
    assert fixed.keys() == strided.keys()
    for name in fixed:
        if name in RE._LAYOUT_STRIDES:
            assert fixed[name] and not strided[name], name
            assert name in RE._rope_embedding_QK_strided.do_not_specialize
        else:
            assert fixed[name] == strided[name], name


def test_eager_seq_major_layout_takes_the_strided_kernel():
    X = torch.empty(B, H, D, S).transpose(2, 3)
    assert not RE._fixed_layout(X)
    assert RE._fixed_layout(torch.empty(B, S, H, D).transpose(1, 2))
    assert RE._fixed_layout(torch.empty(B, H, S, D))
    assert RE._fixed_layout(torch.empty(1, 1, 1, 1))
    assert RE._fixed_layout(torch.empty(1, 1, 1, D).expand(B, H, S, D))
