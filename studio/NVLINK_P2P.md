# GPU peer-to-peer (`GGML_CUDA_P2P`) and why it is gated on NVLink

Recorded so the gate in `_apply_datacenter_env` is not "fixed" back into a product-name
allowlist by someone who reasonably assumes a data-center GPU implies a working peer
fabric. It does not. Background: issue #10613.

## Summary

| | verdict |
|---|---|
| NVLink-connected multi-GPU (`NV#` in the topology matrix) | **P2P enabled** — the configuration PR #6098 benchmarked, +33-51% tensor-split |
| PCIe multi-GPU (`NODE` / `PHB` / `PXB` / `PIX` / `SYS`) | **P2P not enabled** — copies can be silently discarded |
| topology unreadable | **P2P not enabled** — unknown means no |
| single GPU | not applicable, no peer traffic |

`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` is unaffected by all of this and still applies to
every data-center part. It is a property of the silicon, not of the interconnect.

## The failure it prevents

On bare-metal Linux with the IOMMU in translating mode, a PCIe peer-to-peer copy between
two NVIDIA GPUs can be dropped by the chipset while CUDA reports `cudaSuccess`. The write
faults (`DMAR: [DMA Write NO_PASID] ... [fault reason 0x71]` in `dmesg`), the IOMMU
handles it as an Unsupported Request and discards it, and `cudaMemcpyPeerAsync` has no
return code for "the platform ate your DMA".

Inside llama.cpp with `GGML_CUDA_P2P` set, that means tensors that never arrive. The model
emits `!!!!!`, `/////`, a repeated Cyrillic token in reply to English, or fluent word
salad. Nothing is logged and no error is raised, so it looks exactly like a broken quant
or a wrong chat template. The reporter of #10613 spent a day on it.

It is also load-dependent: whether a given layer placement pushes enough traffic across
the bus varies, so the same configuration can look healthy on a short prompt and collapse
on a long one. A clean short test proves nothing here.

NVIDIA documents the configuration as unsupported ([CUDA C++ Programming Guide, IOMMU on
Linux](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#iommu-on-linux)):
bare-metal PCIe peer-to-peer copies require the IOMMU disabled. Most distributions enable
it by default, so "bare-metal Linux, no NVLink" is the common case, not a corner case.

## Why a name allowlist could not work

Four families on the data-center allowlist have no NVLink connector at all: RTX 6000 Ada,
RTX PRO 6000, L40 / L40S and L4. They ship in 2-, 4- and 8-way workstations and servers.
A product name cannot tell you whether the box has a bridge.

## Why the driver is not asked either

`torch.cuda.can_device_access_peer()` returns `True` and `nvidia-smi topo -p2p w` reports
`OK` on hosts where every peer copy drops. The driver's answer is the thing that is wrong;
asking it again is not verification. vLLM reached the same conclusion and performs a real
data-integrity check (`can_actually_p2p`, from vllm#2728).

Studio reads `nvidia-smi topo -m` instead and requires `NV#` between every selected pair.
NVLink traffic does not traverse the PCIe root complex, so the IOMMU fault class above
cannot apply to it. The probe fails closed: a missing `nvidia-smi`, a non-zero exit, a
timeout, an unparsable matrix, or a device mask that cannot be mapped to PCI indices all
mean no P2P.

### Partially bridged boxes, and which pairs get checked

Only the pairs actually selected are checked, so on a 4-way or 8-way box with NVLink
bridges over pairs (0-1 and 2-3, say) and PCIe between the islands, running on a bridged
pair keeps P2P while a selection spanning the islands does not.

Both index spaces here come from nvidia-smi: the GPU selection is sourced from
`nvidia-smi --query-gpu=index` and the matrix from `nvidia-smi topo -m`, one enumeration,
so the selection indexes the matrix directly. Do not "translate" it into CUDA ordinals.
CUDA enumerates in `FASTEST_FIRST` order by default, so under a permutation that remapping
would turn a PCIe-crossing selection into an NVLinked-looking one and enable the flag this
gate exists to withhold.

When no explicit selection is given, every pair on the visible box must be `NV#`.

The log line names the pair that vetoed it.

## Checking a host

```
nvidia-smi topo -m
```

`NV#` between the GPU pair means NVLink and you are unaffected. `NODE`, `PHB`, `PXB`,
`PIX` or `SYS` means the copy goes over PCIe.

To test whether peer copies on this host actually move data:

```
python scripts/p2p_integrity_probe.py
```

It fills the destination with a sentinel first, so a copy that transfers nothing is
distinguishable from one that legitimately writes zeros, and it sweeps every ordered pair
at several sizes. Exit 0 means intact, 1 means data was lost, 2 means it could not run.

## Environment variables

| variable | effect |
|---|---|
| `UNSLOTH_DISABLE_DC_TUNING=1` | disables all data-center tuning, including FP32 accumulate |
| `UNSLOTH_DISABLE_DC_P2P=1` | disables `GGML_CUDA_P2P` only, keeping FP32 accumulate and `CUDA_SCALE_LAUNCH_QUEUES`. Also removes an inherited `GGML_CUDA_P2P`, so it holds even if the variable is already set elsewhere in your environment |
| `UNSLOTH_FORCE_DC_P2P=1` | enables P2P on an unverified fabric (use after the probe passes). It cannot override `UNSLOTH_DISABLE_DC_P2P=1`, an off-meaning `GGML_CUDA_P2P` in your environment, or the data-center gate itself |

`CUDA_SCALE_LAUNCH_QUEUES` is deliberately not gated on the fabric: it sizes a CUDA command
buffer, moves no data between devices, and the reporter of #10613 measured it clean in
isolation on the affected host. The set of hosts that receive it is unchanged by this gate.

### `GGML_CUDA_P2P=0` does not disable P2P

llama.cpp tests the variable for presence, not value:

```c
// ggml/src/ggml-cuda/ggml-cuda.cu
if (getenv("GGML_CUDA_P2P") != nullptr) { ...
bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
```

So `GGML_CUDA_P2P=0` turns peer access **on**. The variable must be unset entirely.
Studio deletes it from the llama-server environment when its inherited value reads as off
(`0`, `false`, `no`, `off`, empty), on every backend, so the intuitive spelling of the
opt-out does what the user meant. Set it to `1` and it is honoured as a deliberate request.
