from __future__ import annotations
import subprocess, sys, types
import pytest
from core.inference.llama_cpp import LlamaCppBackend

def _fake_torch(names, *, hip=None, cuda_ok=True):
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip=hip)
    t.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_ok, device_count=lambda: len(names),
        get_device_properties=lambda i: types.SimpleNamespace(name=names[i]))
    return t

def _use_topo(monkeypatch, text, returncode=0):
    monkeypatch.setattr(subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=returncode, stdout=text, stderr=""))
    LlamaCppBackend._NVLINK_TOPO_CACHE = None

# 4x A100 PCIe, NVLink bridges on (0,1) and (2,3): the standard bridged config.
TOPO_BRIDGED_4X = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\x1b[0m\n"
    "GPU0\t X \tNV12\tSYS\tSYS\t0-23\t0\n"
    "GPU1\tNV12\t X \tSYS\tSYS\t0-23\t0\n"
    "GPU2\tSYS\tSYS\t X \tNV12\t24-47\t1\n"
    "GPU3\tSYS\tSYS\tNV12\t X \t24-47\t1\n"
    "\nLegend:\n\n  X    = Self\n")

@pytest.fixture(autouse=True)
def _iso(monkeypatch):
    for v in ("CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER","UNSLOTH_DISABLE_DC_TUNING",
              "UNSLOTH_DISABLE_DC_P2P","UNSLOTH_FORCE_DC_P2P"):
        monkeypatch.delenv(v, raising=False)
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    monkeypatch.setattr(LlamaCppBackend, "_iommu_is_translating", staticmethod(lambda *a: False))
    monkeypatch.setattr(LlamaCppBackend, "_running_virtualized", staticmethod(lambda: False))
    yield
    LlamaCppBackend._NVLINK_TOPO_CACHE = None

def test_default_order(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"]*4))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    r = LlamaCppBackend._p2p_veto_reason([0,1])
    with capsys.disabled():
        print("\n  [default CUDA_DEVICE_ORDER] selection [0,1] genuinely NV12 ->", repr(r))

def test_pci_bus_id(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_DEVICE_ORDER","PCI_BUS_ID")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"]*4))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    r = LlamaCppBackend._p2p_veto_reason([0,1])
    with capsys.disabled():
        print("  [CUDA_DEVICE_ORDER=PCI_BUS_ID] selection [0,1] ->", repr(r))
