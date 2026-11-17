# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Eager flex_attention in a non-reentrant checkpoint must not restart the layer's recompute mid forward."""

import importlib.util
import os
import sys

import pytest
import torch
import torch.utils.checkpoint as checkpoint
from real_accelerator import has_real_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXES = os.path.join(_ROOT, "unsloth", "import_fixes.py")
_FIX = "fix_flex_attention_eager_saved_tensor_hooks"
_HOP = "torch._higher_order_ops.flex_attention"


def _apply_fix(monkeypatch):
    flex_hop = pytest.importorskip(_HOP)
    if not hasattr(flex_hop, "create_fw_bw_graph"):
        pytest.skip("this torch has no flex create_fw_bw_graph")
    monkeypatch.setattr(flex_hop, "create_fw_bw_graph", flex_hop.create_fw_bw_graph)
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_flex_hooks", _FIXES)
    fixes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixes)
    getattr(fixes, _FIX, lambda: None)()
    return sys.modules[_HOP]


def test_trace_runs_under_identity_hooks_only_when_hooks_are_active(monkeypatch):
    seen = []

    def fake_create_fw_bw_graph(*args, **kwargs):
        seen.append(torch._C._autograd._top_saved_tensors_default_hooks(False))
        return "graphs"

    flex_hop = pytest.importorskip(_HOP)
    monkeypatch.setattr(flex_hop, "create_fw_bw_graph", fake_create_fw_bw_graph)
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_flex_hooks_cpu", _FIXES)
    fixes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixes)
    getattr(fixes, _FIX, lambda: None)()
    getattr(fixes, _FIX, lambda: None)()
    wrapped = flex_hop.create_fw_bw_graph
    assert getattr(wrapped, "_unsloth_original", None) is fake_create_fw_bw_graph

    assert wrapped() == "graphs"
    assert seen[-1] is None

    def pack(t):
        return ("packed", t)

    def unpack(p):
        return p[1]

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        assert wrapped() == "graphs"
        assert torch._C._autograd._top_saved_tensors_default_hooks(False)[0] is pack
    inner_pack = seen[-1][0]
    assert inner_pack is not pack
    probe = torch.ones(1)
    assert inner_pack(probe) is probe


@pytest.mark.skipif(not has_real_cuda(), reason = "flex_attention has no CPU backward")
def test_eager_flex_in_nonreentrant_checkpoint_matches_plain(monkeypatch):
    _apply_fix(monkeypatch)
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    proj = torch.nn.Linear(8, 8, device = "cuda")
    calls = [0]

    def softcap(score, b, h, q, k):
        return 50.0 * torch.tanh(score / 50.0)

    def layer(x):
        calls[0] += 1
        return flex_attention(proj(x), proj(x) * 2, proj(x) + 1, score_mod = softcap)

    x = torch.randn(1, 2, 32, 8, device = "cuda", requires_grad = True)
    layer(x).sum().backward()
    want = x.grad.clone()
    for _ in range(2):
        x.grad = None
        calls[0] = 0
        out = checkpoint.checkpoint(layer, x, use_reentrant = False)
        assert calls[0] == 1, "the checkpoint recomputed the layer inside its own forward"
        out.sum().backward()
        assert calls[0] == 2
        torch.testing.assert_close(x.grad, want)
