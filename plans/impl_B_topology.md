# #10613 fix, angle B: static topology signals, no runtime probe

## The defect

`_apply_datacenter_env` infers "this box has a working NVLink fabric" from the
GPU's marketing name. `_DATACENTER_GPU_RE` includes RTX 6000 Ada, RTX PRO 6000,
L40, L40S and L4, none of which have an NVLink connector. On a multi-GPU host
built from those parts the only peer path is PCIe, and on bare-metal Linux behind
a translating IOMMU a PCIe peer copy is *silently discarded*: `cudaSuccess`, zero
bytes moved, no log line, every model emits `!!!!!` / `/////` / word salad.

Two aggravating facts from the report:

- `cudaDeviceCanAccessPeer` returns **True** on the failing host, and
  `nvidia-smi topo -p2p w` reports `OK`. Neither can be used as evidence.
- llama.cpp tests `GGML_CUDA_P2P` for **presence**, not value
  (`getenv(...) != nullptr`), so the documented `GGML_CUDA_P2P=0` opt-out
  *enables* the thing it looks like it disables.

## Approach

Cheap static signals only. No CUDA context, no subprocess fork of a probe
process, no allocation on any device: everything here is a regex over a name we
already have, one cached `nvidia-smi` call, and a couple of `/sys` reads.

### 1. Split the gate in two

`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` is about FP32 accumulation. It is safe on
every part in the current allowlist and the reporter's own isolation table
confirms it is not implicated. It keeps `_DATACENTER_GPU_RE` unchanged.

`GGML_CUDA_P2P` + `CUDA_SCALE_LAUNCH_QUEUES` move behind a new, narrower
`_NVLINK_FABRIC_GPU_RE`: A100, A30, H100, H200, H800, GH200, B100, B200, B300,
GB200, GB300. RTX 6000 Ada / RTX PRO 6000 / L40 / L40S / L4 are deliberately
absent.

### 2. Name membership is necessary, never sufficient

The name gate alone would fix the reporter's host, but it is still an inference
about hardware from a string. `_p2p_veto_reason` demands *positive* evidence that
every selected pair is NVLink-bonded, from `nvidia-smi topo -m`: `NV#` means a
bonded set of # NVLinks; `NODE`, `PHB`, `PXB`, `PIX` and `SYS` all traverse PCIe
and all veto.

### 3. Fail closed

Every unknown withholds P2P: no `nvidia-smi` on PATH, non-zero exit, timeout,
unparsable table, an index space we cannot join, a pair missing from the matrix.
The old code failed *open* into silent corruption, which is the worst possible
direction for this particular flag.

### 4. Index spaces (the subtle part)

Three spaces are in play and two of them are easy to confuse:

| space | who speaks it |
| --- | --- |
| CUDA ordinal | `torch.cuda.get_device_properties(i)` |
| physical id | `gpu_indices`, `CUDA_VISIBLE_DEVICES` entries |
| nvidia-smi index | `topo -m` row/column labels, `--query-gpu=index` |

`_resolve_visible_physical_ids()` already bridges ordinal -> physical.
Physical -> nvidia-smi index is only an identity map when
`CUDA_DEVICE_ORDER=PCI_BUS_ID`; otherwise CUDA enumerates FASTEST_FIRST and
"GPU 5" means different cards to the two tools. `_cuda_compute_caps` handles the
same problem by bailing out, and this follows that convention rather than
inventing a second one.

Rather than lose the tuning on every host that has not set `CUDA_DEVICE_ORDER`,
there is an escape hatch that is safe by construction: when the mapping is
inexact, require the **entire** GPU matrix to be uniformly `NV#`. If every pair
on the box is NVLink-bonded then any mis-mapping selects a different NVLink pair,
so the conclusion holds regardless of which permutation is real. A non-uniform
matrix plus an inexact mapping vetoes. (Verified against this host: 8x B200, all
`NV18`, so the uniform path is the common DGX/HGX case, not a corner.)

### 5. IOMMU

`/sys/kernel/iommu_groups/*/type` distinguishes `identity` (passthrough) from
`DMA` / `DMA-FQ` (translating). Per the CUDA C++ Programming Guide ("IOMMU on
Linux"), bare-metal PCIe P2P is unsupported while the IOMMU translates, and VM
pass-through inverts the guidance, so the check is skipped under a hypervisor
(`hypervisor` flag in `/proc/cpuinfo`).

**Deliberate carve-out, and a deviation from a literal reading of "IOMMU vetoes
regardless of card names":** the veto is not applied to a selection whose pairs
are all `NV#`-confirmed. NVLink traffic does not traverse PCIe, so the IOMMU
cannot drop it, and a literal veto would disable this tuning on essentially every
bare-metal DGX/HGX box (VT-d / AMD-Vi is on by default on the distributions those
ship with) - regressing exactly the 6x B200 configuration PR #6098 measured. What
survives is the part that matters: *names never rescue a translating-IOMMU host*,
only positive NVLink evidence does. On any PCIe pair the IOMMU state is decisive
and is quoted in the veto reason, because it is the actionable diagnosis.

### 6. pynvml: checked, deliberately not used

The directive asked me to check first. `pynvml` is **not** a declared dependency
of unsloth (`pyproject.toml` lists typer/rich/pydantic/pyyaml/nest-asyncio/
huggingface-hub/structlog/click); it appears only as a stub in
`tests/vllm_compat/*` and via `unsloth_zoo`. It happens to be importable in this
venv, which is exactly the kind of accident that makes a feature work on the
developer's box and vanish in the field.

It also offers *less* information than the tool already in use.
`nvmlDeviceGetNvLinkState(handle, link)` says a link is up, not who is on the
other end; reconstructing the peer graph needs
`nvmlDeviceGetNvLinkRemotePciInfo`, whose `_v2` signature churn across pynvml
releases is a maintenance liability. `topo -m` states the bonded pairwise result
directly, through the `subprocess` conventions this file already uses. Adding an
undeclared optional dependency to obtain weaker evidence is the wrong trade.

### 7. Make the opt-out real, and say what was set

- A falsy user-supplied `GGML_CUDA_P2P` (`""`, `0`, `false`, `off`, `no`) is
  **removed** from the child env, mirroring the `GGML_CUDA_ENABLE_UNIFIED_MEMORY`
  precedent at the call site (#8651). Unconditional: passing "0" through is just
  as wrong on a box that never qualified for the tuning.
- The log names every variable and value it set, and on withholding P2P names the
  signal that vetoed. The old message named none of them, which is what cost the
  reporter most of a day.

## What this cannot catch

A host whose pairs really are `NV#` but whose fabric is degraded at runtime. Only
a real data-integrity probe (vLLM's `can_actually_p2p`) closes that, and it costs
a CUDA context per device. This angle is deliberately the cheap half; it removes
every configuration in the report's blast radius without spending a subprocess at
launch.
