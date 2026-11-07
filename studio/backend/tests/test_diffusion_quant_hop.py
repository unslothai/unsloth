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

"""_hop_each: host-placed Linears are quantised on the CUDA device and always come back home."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_transformer_quant as tq


class _Mod:
    def __init__(
        self,
        name,
        log,
        home = "cpu",
    ):
        self.name = name
        self.log = log
        self.weight = types.SimpleNamespace(device = torch.device(home))

    def to(self, device):
        self.log.append(("to", self.name, str(device)))
        self.weight = types.SimpleNamespace(device = torch.device(device))
        return self


def test_hop_device_only_takes_cuda():
    assert tq._hop_device(None) is None
    assert tq._hop_device("cpu") is None
    assert tq._hop_device("mps") is None
    assert tq._hop_device("not a device") is None
    assert tq._hop_device("cuda:1") == torch.device("cuda:1")


def test_no_work_device_quantises_in_place():
    log = []
    mods = [_Mod(i, log) for i in range(3)]
    tq._hop_each(mods, None, lambda m: log.append(("q", m.name, str(m.weight.device))))
    assert log == [("q", 0, "cpu"), ("q", 1, "cpu"), ("q", 2, "cpu")]


def test_each_module_quantises_on_work_and_returns_home():
    log = []
    mods = [_Mod(i, log) for i in range(4)] + [_Mod(9, log, home = "meta")]
    seen = []
    tq._hop_each(
        mods, torch.device("cuda:0"), lambda m: seen.append((m.name, str(m.weight.device)))
    )
    assert seen == [(0, "cuda:0"), (1, "cuda:0"), (2, "cuda:0"), (3, "cuda:0"), (9, "meta")]
    assert all(str(m.weight.device) in ("cpu", "meta") for m in mods)
    assert [m.weight.device.type for m in mods[:4]] == ["cpu"] * 4
    assert not any(entry[1] == 9 for entry in log if entry[0] == "to")


def test_a_raise_leaves_every_module_home():
    log = []
    mods = [_Mod(i, log) for i in range(4)]

    def fn(m):
        if m.name == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        tq._hop_each(mods, torch.device("cuda:0"), fn)
    assert [m.weight.device.type for m in mods] == ["cpu"] * 4


def test_staged_upload_refuses_what_it_cannot_copy_exactly():
    assert tq._upload_staged(object(), torch.device("cuda:0")) is False
    lin = torch.nn.Linear(4, 4)
    lin.weight.data = lin.weight.data.t()  # non-contiguous: left to module.to
    assert tq._upload_staged(lin, torch.device("cuda:0")) is False
    assert lin.weight.device.type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason = "checks the no-CUDA fallback")
def test_staged_upload_without_cuda_moves_nothing():
    lin = torch.nn.Linear(4, 4)
    assert tq._upload_staged(lin, torch.device("cuda:0")) is False
    assert lin.weight.device.type == "cpu"


def test_an_unindexed_cuda_device_resolves_in_the_calling_thread(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert tq._hop_device("cuda") == torch.device("cuda:3")
    assert tq._hop_device(torch.device("cuda")) == torch.device("cuda:3")
    assert tq._hop_device("cuda:1") == torch.device("cuda:1")
