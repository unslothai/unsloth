# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

"""The lean varlen path runs only where torch's varlen_attn would run flash, for each torch varlen API."""

import importlib
import sys
import types

import pytest
import torch

ad = pytest.importorskip("unsloth.utils.attention_dispatch")

CUDNN = torch.nn.attention.SDPBackend.CUDNN_ATTENTION.value
FLASH = torch.nn.attention.SDPBackend.FLASH_ATTENTION.value
Q = types.SimpleNamespace(is_cuda = True, device = types.SimpleNamespace(index = 0))


def _varlen_attn(
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    *,
    scale = None,
    window_size = None,
    enable_gqa = False,
):
    raise AssertionError("not called")


@pytest.fixture
def reload_with(monkeypatch):
    real = sys.modules.get("torch.nn.attention.varlen")

    def load(**attrs):
        fake = types.ModuleType("torch.nn.attention.varlen")
        fake.varlen_attn = _varlen_attn
        fake._should_use_cudnn = lambda device_index: True  # every SM90 / SM100 with cuDNN >= 9.18
        for k, v in attrs.items():
            setattr(fake, k, v)
        monkeypatch.setitem(sys.modules, "torch.nn.attention.varlen", fake)
        monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: True)
        monkeypatch.setenv("UNSLOTH_LEAN_VARLEN", "1")
        monkeypatch.setenv("UNSLOTH_TORCH_VARLEN", "1")
        return importlib.reload(ad)

    yield load
    monkeypatch.undo()
    if real is not None:
        sys.modules["torch.nn.attention.varlen"] = real
    importlib.reload(ad)


def test_torch_215_lean_where_torch_selects_flash(reload_with):
    seen = []

    def select(query, key, value, cu_seq_q, cu_seq_k, max_q, window_size, *rest):
        seen.append((cu_seq_q, cu_seq_k, max_q, window_size))
        return FLASH

    m = reload_with(_select_backend = select, _CUDNN_ATTENTION_BACKEND = CUDNN)
    assert m._LEAN_VARLEN
    cu = object()
    assert m._varlen_can_use_cudnn(Q, Q, Q, cu, 512, [-1, 0], False) is False
    assert seen == [(cu, cu, 512, [-1, 0])]


def test_torch_215_no_lean_where_torch_selects_cudnn(reload_with):
    m = reload_with(_select_backend = lambda *a: CUDNN, _CUDNN_ATTENTION_BACKEND = CUDNN)
    assert m._varlen_can_use_cudnn(Q, Q, Q, object(), 512, [-1, 0], False) is True


def test_torch_215_no_lean_when_no_backend_is_viable(reload_with):
    def select(*a):
        raise RuntimeError("No viable backend for varlen_attn")

    m = reload_with(_select_backend = select, _CUDNN_ATTENTION_BACKEND = CUDNN)
    assert m._varlen_can_use_cudnn(Q, Q, Q, object(), 512, [-1, 0], False) is True


def test_torch_214_keeps_can_use_cudnn(reload_with):
    seen = []

    def can_use(
        query,
        key,
        value,
        max_q,
        window_size,
        enable_gqa = False,
    ):
        seen.append((max_q, window_size, enable_gqa))
        return False

    m = reload_with(_can_use_cudnn = can_use)
    assert m._varlen_can_use_cudnn(Q, Q, Q, object(), 100, [-1, 0], True) is False
    assert seen == [(100, [-1, 0], True)]


def test_torch_213_falls_back_to_the_device_check(reload_with):
    m = reload_with()
    assert m._varlen_can_use_cudnn(Q, Q, Q, object(), 100, [-1, 0], False) is True
