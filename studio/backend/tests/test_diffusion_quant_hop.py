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
    # a module already off the host is never moved
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
