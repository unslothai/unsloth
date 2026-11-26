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

"""A once-per-device kernel self check that hits an OOM must not cache "unavailable": the OOM is
transient, and caching it kept the slower route for the rest of the run. CPU only (stand-in kernels)."""

import contextlib
import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")

_ROOT = Path(__file__).resolve().parents[1] / "unsloth"


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        Path(module.__file__).resolve().is_relative_to(_ROOT)
    )  # this tree, not an installed unsloth
    return module


def _oom():
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")


def test_bf16_sr_triton_check_oom_is_not_cached(monkeypatch):
    sr = _load("optimizers/bf16_stochastic_rounding.py", "_sr_oom_test")
    calls = []

    class _Kernel:
        def __getitem__(self, grid):
            def run(p, *args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise _oom()
                p.fill_(1)

            return run

    monkeypatch.setattr(sr, "_adamw_kernel", _Kernel())
    monkeypatch.setattr(sr, "_round_kernel", _Kernel())
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: None)
    device = torch.device("cpu")  # the stand-in kernels run on CPU tensors
    assert sr._triton_ok(device) is False
    assert str(device) not in sr._TRITON_OK
    assert sr._triton_ok(device) is True
    assert sr._TRITON_OK[str(device)] is True
