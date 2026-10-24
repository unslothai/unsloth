# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block CUDA graphs (opt-in) must copy block outputs and input gradients out of the graph pool.

Under Unsloth checkpointing every block call starts a new cudagraph tree generation, so a
tensor the checkpoint keeps (the saved block input, the gradient handed to the previous
block) would be overwritten by the next replay.
"""

import importlib

import torch

from unsloth.models import _block_compile as bc


def test_clone_grad_is_identity_with_copied_gradient():
    x = torch.randn(3, 4, requires_grad = True)
    seen = {}

    def keep(grad):
        seen["ptr"] = grad.data_ptr()
        return grad

    y = bc._CloneGrad.apply(x)
    y.register_hook(keep)
    assert torch.equal(y, x)
    (y * 2).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.0))
    assert x.grad.data_ptr() != seen["ptr"]


def test_clone_outputs_copies_every_tensor():
    a, b = torch.randn(2), torch.randn(3)
    out = bc._clone_outputs((a, None, b))
    assert out[1] is None
    assert torch.equal(out[0], a) and out[0].data_ptr() != a.data_ptr()
    assert torch.equal(out[2], b) and out[2].data_ptr() != b.data_ptr()
    single = bc._clone_outputs(a)
    assert torch.equal(single, a) and single.data_ptr() != a.data_ptr()


def test_cudagraphs_opt_in_flag(monkeypatch):
    try:
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "1")
        assert importlib.reload(bc)._COMPILE_OPTIONS["triton.cudagraphs"] is True
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
        assert importlib.reload(bc)._COMPILE_OPTIONS["triton.cudagraphs"] is False
    finally:
        monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS", raising = False)
        importlib.reload(bc)
