"""Scratch: _p2p_veto_reason across index spaces. Not a test file."""
import os, sys, types, subprocess
sys.path.insert(0, "studio/backend")
from core.inference.llama_cpp import LlamaCppBackend as B

TWO_ISLANDS = (
    "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\n"
    "GPU0\t X \tNV12\tPHB\tPHB\t0-23\n"
    "GPU1\tNV12\t X \tPHB\tPHB\t0-23\n"
    "GPU2\tPHB\tPHB\t X \tNV12\t24-47\n"
    "GPU3\tPHB\tPHB\tNV12\t X \t24-47\n"
)
ALL_NV = (
    "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\n"
    "GPU0\t X \tNV12\tNV12\tNV12\t0-23\n"
    "GPU1\tNV12\t X \tNV12\tNV12\t0-23\n"
    "GPU2\tNV12\tNV12\t X \tNV12\t24-47\n"
    "GPU3\tNV12\tNV12\tNV12\t X \t24-47\n"
)

class _Props:
    def __init__(self, name): self.name = name

def fake_torch(names):
    cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(names),
        get_device_properties = lambda i: _Props(names[i]),
    )
    return types.SimpleNamespace(cuda = cuda, version = types.SimpleNamespace(hip = None))

def veto(topo, names, gpu_indices, order=None, mask=None):
    B._NVLINK_TOPO_CACHE = None
    real_run, real_torch = subprocess.run, sys.modules.get("torch")
    subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=topo, stderr="")
    sys.modules["torch"] = fake_torch(names)
    for k, v in (("CUDA_DEVICE_ORDER", order), ("CUDA_VISIBLE_DEVICES", mask)):
        os.environ.pop(k, None)
        if v is not None: os.environ[k] = v
    for k in ("UNSLOTH_DISABLE_DC_P2P", "UNSLOTH_FORCE_DC_P2P"): os.environ.pop(k, None)
    try:
        return B._p2p_veto_reason(gpu_indices)
    finally:
        subprocess.run = real_run
        if real_torch is not None: sys.modules["torch"] = real_torch
        else: sys.modules.pop("torch", None)
        B._NVLINK_TOPO_CACHE = None

A4 = ["NVIDIA A100-SXM4-40GB"] * 4

print("--- two NVLink islands, selecting the genuinely bridged pair [0,1] ---")
print("  default order (FASTEST_FIRST):", veto(TWO_ISLANDS, A4, [0, 1]))
print("  CUDA_DEVICE_ORDER=PCI_BUS_ID :", veto(TWO_ISLANDS, A4, [0, 1], order="PCI_BUS_ID"))
print("  lowercase pci_bus_id         :", veto(TWO_ISLANDS, A4, [0, 1], order="pci_bus_id"))

print("--- two islands, selecting an unbridged pair [0,2] (must veto both ways) ---")
print("  default order:", veto(TWO_ISLANDS, A4, [0, 2]))
print("  PCI_BUS_ID   :", veto(TWO_ISLANDS, A4, [0, 2], order="PCI_BUS_ID"))

print("--- fully NVLinked 4x A100 ---")
print("  default order:", veto(ALL_NV, A4, [0, 1]))
print("  PCI_BUS_ID   :", veto(ALL_NV, A4, [0, 1], order="PCI_BUS_ID"))
print("  gpu_indices=None:", veto(ALL_NV, A4, None))

print("--- masked host, PCI_BUS_ID, mask=2,3 selection [2,3] on two-island box ---")
print("  ", veto(TWO_ISLANDS, A4, [2, 3], order="PCI_BUS_ID", mask="2,3"))

print("--- UUID mask (issue #8873), gpu_indices=None ---")
print("  all-NV box:", veto(ALL_NV, A4, None, order="PCI_BUS_ID",
                            mask="GPU-aaaa-bbbb,GPU-cccc-dddd"))
print("  two-island:", veto(TWO_ISLANDS, A4, None, order="PCI_BUS_ID",
                            mask="GPU-aaaa-bbbb,GPU-cccc-dddd"))

print("--- selection index beyond the matrix (topo shows 4, selection names 5) ---")
print("  ", veto(ALL_NV, A4, [0, 5], order="PCI_BUS_ID"))

print("--- non-NVLink-capable part with a lying all-NV matrix ---")
print("  ", veto(ALL_NV, ["NVIDIA RTX 6000 Ada Generation"] * 4, [0, 1]))
