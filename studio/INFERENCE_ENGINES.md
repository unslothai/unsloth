# Optional inference engines

Studio can install and run vLLM or SGLang for local text and image chat. Installation
is opt-in. Select a model, open its run settings, and choose **Inference engine**.
Choose **Install engine** to install and load the selected model. Selecting **Default** returns
to Studio's normal backend. Remembered model settings include the engine choice.

**Settings > System > Inference engines** provides installation, cancellation,
repair, removal, and restoration of the previous installation. Unload the model
before changing an installed engine. Installation continues if the panel closes;
if it closes before installation finishes, use **Load model** when you return.
Removing an engine keeps downloaded models and the shared package cache.
Installation progress appears in the shared download panel and returns after
refreshing the page. It shows live package-manager output and expandable details.
Model settings show a brief installation status and a cancel action, without
duplicating the progress bar or live log.
The progress bar stays indeterminate because the installer does not report reliable
byte totals; Studio does not invent a percentage, transfer speed or ETA.

## Initial support

This experimental profile requires Linux x86_64, glibc 2.34 or newer, an NVIDIA
GPU with compute capability 8.0 or newer, and driver 580 or newer; vLLM also runs on
AMD GPUs (see [AMD GPUs](#amd-gpus)). It uses
the model's standard chat template and each engine's native model loaders.
Select one or more GPUs in the model settings. With multiple GPUs, choose
**Multi-GPU mode**:

- **Tensor parallel** splits computations within model layers. This is the default
  for existing settings. Studio checks head and layer dimensions before unloading
  the current model.
- **Pipeline parallel** places different layers on each GPU. It can reduce
  communication overhead on systems without NVLink. Layer allocation follows the
  engine's default partition; Studio does not automatically balance unequal cards.
- **Replicas (data parallel)** distributes requests across GPU workers. Dense
  models need a full model copy and cache on each GPU. Native vLLM data parallelism
  can shard MoE experts across workers, so these are not independent full replicas
  for every architecture.

All selected devices must satisfy the hardware requirements. The engine validates
model and quantization support for the selected mode. These modes use one machine;
combined tensor/pipeline/data configurations are not exposed.
Text and vision architectures are validated by the selected engine. Individual
models can still exceed available memory or require unsupported kernels.

OpenAI chat requests support client-owned tool calls, streamed arguments and tool-result
history. Studio can also execute its existing built-in and MCP tools through the
shared approval and execution flow. Native output parsers are selected from the
model's Jinja template syntax; Studio does not replace the template. Models with
unrecognized tool syntax remain available for ordinary chat. The pinned SGLang
version can return multiple calls even with `parallel_tool_calls: false`.
Legacy completions and Anthropic endpoints retain their existing GGUF requirement;
exact token counting for tool-enabled prompts is not available.

LoRA adapters, audio/video input, structured output, reasoning
controls and continuation are outside this profile. Custom model code uses the
existing trust-remote-code consent setting. Multi-node serving is not configured by Studio.
Engine servers honor Studio's VRAM budget, capped by the most constrained
selected GPU's measured free-memory fraction, with at least 512 MiB left per GPU for driver allocations.
GPU memory is not treated as a single pool: each device must fit its own shard
and runtime allocations. Training evicts them before allocating
GPU memory. Startup includes model downloads and kernel compilation and can take
several minutes. The UI reports phases without inventing progress percentages.

The load API accepts `engine_parallelism`: `tensor` (default), `pipeline` or `data`.
The selected mode spans `gpu_ids`; the other parallel dimensions stay at one.
For example, `{"engine": "vllm", "gpu_ids": [0, 1], "engine_parallelism": "pipeline"}`
uses two pipeline stages with tensor parallel size one. With one selected GPU, all
parallel sizes are one and the saved mode is retained for later multi-GPU loads.

The existing `tensor_parallel` request switch remains a GGUF setting. Optional
engine load and status responses report `engine_parallelism`, selected GPU IDs,
and whether tensor parallelism is actually active. Remembered model settings and
API auto-loads preserve mode and GPU order. Changing either requires reloading.

## Precision and images

The existing model settings include a **Precision** selector for optional engines:

- **Model default** lets the engine detect the checkpoint's dtype or stored
  quantization, including compatible AWQ, GPTQ, FP8 and BitsAndBytes checkpoints.
- **BF16** and **FP16** choose a 16-bit dtype for an unquantized checkpoint.
- **4-bit** converts an unquantized checkpoint with BitsAndBytes in vLLM
  tensor/data modes, or TorchAO in vLLM pipeline mode and SGLang.
- **INT8** uses the engines' native TorchAO weight-only conversion.
- **FP8** uses vLLM's TorchAO weight-only conversion. SGLang uses native FP8
  conversion on Ada and newer GPUs, or TorchAO weight-only conversion on Ampere.

Available kernels depend on the model and GPUs. Prequantized checkpoints use
**Model default** and are not requantized. Prequantized BitsAndBytes models cannot use
tensor parallelism in these pinned engines. SGLang also cannot split these
prequantized weights across pipeline stages; use replicas or an unquantized
checkpoint with on-load 4-bit conversion instead. Other combinations use the
native loader and retain its model-specific constraints. FP8 conversion on Ampere GPUs and SGLang prequantized BitsAndBytes INT8
use eager execution because their compiled paths are incompatible with these
pinned dependencies. Conversion can need more memory during loading
than the final model occupies. A dependency profile change shows **Update engine**
before loading from the picker.

The load API accepts `engine_precision`: `auto`, `bf16`, `fp16`, `int4`, `int8` or
`fp8`. Remembered settings and status responses preserve the value. Existing API
clients that explicitly send `load_in_4bit: true` without `engine_precision` select
4-bit conversion. Changing precision requires a reload.

Vision models accept text-only chat and OpenAI `image_url` content parts, including
multiple images and images on earlier turns. Studio forwards the original image
bytes to the engine, fetching remote URLs itself with the same limits as other backends. The selected model determines its image and context limits.

## Environment and cache design

Engines run in their own processes from environments under the Studio home's
`engines` directory. Studio never imports engine packages into its own Python
process. Compiler caches, including Triton, are scoped to each engine profile, model,
precision, context and GPU selection to avoid reusing incompatible kernels.
The committed requirements profiles lock versions and wheel hashes for Python 3.13 (the ROCm
lock: Python 3.12).
Each engine has one locked release per PyTorch build it can share, newest first:

| Engine | Release | PyTorch | Lock |
|---|---|---|---|
| vLLM | 0.30.0 | 2.13.0 (CUDA 13.0) | `vllm-linux-cu130-torch213.txt` |
| vLLM | 0.26.0 | 2.11.0 (CUDA 13.0) | `vllm-linux-cu130.txt` |
| SGLang | 0.5.20 | 2.13.0 (CUDA 13.0) | `sglang-linux-cu130-torch213.txt` |
| SGLang | 0.5.17 | 2.11.0 (CUDA 13.0) | `sglang-linux-cu130.txt` |
| vLLM on AMD | 0.30.0 | 2.12.0 (ROCm 7.2) | `vllm-linux-rocm723.txt` |

Studio installs the release built on its own PyTorch, so a new install (torch 2.13) and an
existing default install (torch 2.11) both share. Any other PyTorch (2.10, 2.12, a CUDA 12 build)
gets the newest release in a complete isolated environment, since no vLLM or SGLang release is
built on it. Windows hosts install the newest release inside WSL.

The engines cannot share Studio's site-packages: they pin Transformers and other
packages at versions Studio does not use, and they conflict with each other. They
can share its interpreter, PyTorch and CUDA libraries. When Studio runs Python 3.13
and its torch, together with the Triton and NVIDIA packages torch loads, is exactly
the version an engine is locked to, the engine environment uses Studio's
interpreter and holds only the locked packages Studio lacks or has at another
version. A `.pth` file adds Studio's site-packages after the engine's own, so the
engine's pins win and multiprocessing workers see the same packages. A CUDA library torch accepts as a range (nvjitlink)
only has to satisfy every locked package's requirement. Otherwise, for example with
another torch version or CUDA build, the engine gets a complete isolated environment
from the newest lock.

A shared environment records the Studio packages it was checked against. If a
Studio update changes any of them, or the Python version, the engine reports an
available update, refuses to load until it is repaired, and cannot be rolled back
to an environment built on the old packages. Repair reuses cached downloads.

Installation uses the shared uv cache, honoring `UV_CACHE_DIR` when configured.
Otherwise it reuses the installer's recorded cache path, with Studio's own cache
as the fallback.
Copy-on-write clones allow identical cached files to share storage on supporting
filesystems; uv falls back to copies elsewhere. Environments do not hardlink
mutable files into Studio's Python environment. Existing installations may use
different Torch or CUDA builds, or may not retain cached wheels, so no fixed disk
savings are promised. Model downloads use Studio's configured Hugging Face cache.

An exclusive file lock protects installation and removal across Studio processes.
Running engines hold shared leases. A staged environment must pass dependency,
CUDA import and server entry-point checks before an atomic active marker changes.
The dependency check runs in the engine's interpreter, so it covers both layers of a
shared environment: every locked package must resolve to its locked version, with
its requirements met.
Failed or cancelled installations preserve the prior marker. A successful repair
keeps one previous environment for restoration. Explicitly restoring an older
profile allows model loading while still offering the pinned update. That choice
survives restart and failed repairs; a successful update replaces it. Interrupted
jobs are reported after restarting Studio.

Servers bind to loopback with a random port and authentication key. Studio owns
their process trees, startup cancellation, health checks, streaming cancellation
and GPU handoff. These engines participate in the local model lifecycle. They
are not entries in the external provider registry.

Engine adapters own command lines and startup phases; the orchestrator remains
the source of resident-model state. HTTP response cleanup is shared with the
external OpenAI transport. A bounded async bridge provides synchronous streaming
to the orchestrator without letting a slow consumer buffer unlimited output.
Children are spawned through Studio's process-lifetime thread so they survive
the short-lived thread that requested their start.

Controls use the shared Select, Switch and Settings components. New copy lives in the
translation catalog, with explicit English fallback for untranslated locales.

SGLang uses Triton attention and PyTorch sampling in this profile so its default
FlashInfer JIT path does not require a matching system CUDA toolkit. Runtime
executables such as Ninja are resolved from the engine environment first, then from
Studio's environment for a shared engine.
For tensor parallelism, SGLang probes GPU peer access and falls back to its
standard collective when peer access is unavailable. Its equal-free-memory
guard is disabled for these managed groups: Studio checks every GPU, and SGLang
sizes its cache from the minimum available memory across ranks. This allows
cards with different memory capacities without treating their VRAM as a pool.

## AMD GPUs

vLLM runs on AMD GPUs from vLLM's own ROCm build: vLLM 0.30.0 with the PyTorch 2.12 it
publishes for ROCm 7.2 (`vllm-linux-rocm723.txt`). SGLang stays NVIDIA only. Studio picks the
ROCm build when its own PyTorch is a ROCm build, or on Linux when the kernel lists an AMD GPU and
no NVIDIA driver is loaded.

On Linux the host needs:

- x86_64 and glibc 2.39 or newer: the wheels are `manylinux_2_39`, as on Ubuntu 24.04.
- ROCm 7.2 or a newer 7.x release in `/opt/rocm`, with its math libraries. vLLM's PyTorch links
  the host's ROCm (MIOpen, rocBLAS, hipBLASLt, RCCL and the rest) instead of bundling it.
- Open MPI 4 and libnuma, which that PyTorch also links (`libopenmpi3t64 libnuma1` on Ubuntu 24.04).
- A C compiler: Triton builds its HIP driver module when the engine first compiles a kernel.
- Access to `/dev/kfd` (the `render` and `video` groups).
- A GPU whose gfx target vLLM's ROCm kernels are built for: gfx90a, gfx942, gfx950, gfx1100,
  gfx1101, gfx1150, gfx1151, gfx1200 or gfx1201. Studio reads the target the GPU presents to
  PyTorch, so an `HSA_OVERRIDE_GFX_VERSION` in Studio's environment applies to the engine too.

The engine always gets an isolated environment on a uv-managed CPython 3.12: vLLM publishes its
ROCm build for CPython 3.12 only, and Triton compiles against the interpreter's `Python.h`, which a
distribution's Python lacks without its `-dev` package. Its device mask sets `HIP_VISIBLE_DEVICES`
and `CUDA_VISIBLE_DEVICES` to the same ids (vLLM refuses a mismatch), translated through an
inherited `ROCR_VISIBLE_DEVICES`. The memory budget comes from the engine's own PyTorch instead of
nvidia-smi. Model default, BF16, FP16, INT8 and FP8 work. 4-bit conversion and prequantized
BitsAndBytes checkpoints are refused before the resident model unloads, because vLLM's ROCm build
has no BitsAndBytes.

On Windows the engine runs in Studio's WSL distro, as on NVIDIA, on one AMD GPU. Installing it
also installs ROCm 7.2.1 from AMD's apt repository into the distro, plus AMD's prebuilt WSL
packages from `ROCm/librocdxg` v1.2.2: `rocdxg-roct`, the HSA runtime's bridge to the Windows
driver over `/dev/dxg`, and `rocdxg-amd-smi-lib`, through which vLLM detects the GPU. The apt key
and both packages are pinned by sha256. The AMD Windows driver must expose the GPU to WSL. The
engine runs with `HSA_ENABLE_DXG_DETECTION=1`, `LD_LIBRARY_PATH=/opt/rocm-wsl/lib` and
`ROCPROFILER_REGISTER_ENABLED=0`: rocprofiler-sdk, which vLLM's PyTorch links, aborts HIP start-up
without the KFD sysfs topology that WSL lacks. The pool HIP reports under WSL is the dedicated
memory plus a share of the host's RAM, and allocations past what the host can back stall rather
than fail, so Studio budgets a discrete card against its dedicated memory and an APU against its
dedicated memory plus 80% of the RAM Windows reports available. When Windows reports neither, the
budget falls back to the WSL VM's available memory, and the load is refused when that cannot be
read either. This was measured on one Strix Halo machine.

## Maintaining profiles

Run from the repository root using uv. Seed each output file with `uv pip freeze`
from a default Studio install first: uv prefers versions already in the output file,
which keeps each engine's own layer small.

```sh
cd studio/backend/requirements/engines
for lock in vllm-linux-cu130-torch213 vllm-linux-cu130 sglang-linux-cu130-torch213 sglang-linux-cu130; do
  uv pip compile $lock.in --python-version 3.13 --python-platform x86_64-manylinux_2_34 --index-url https://pypi.org/simple --refresh --generate-hashes --excludes $lock.excludes -o $lock.txt
  python engine_compat.py $lock
done
```

The ROCm lock resolves vLLM's own index beside PyPI; its hashes pin every wheel, whichever index
serves it:

```sh
uv pip compile vllm-linux-rocm723.in --python-version 3.12 --python-platform x86_64-manylinux_2_39 --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723 --generate-hashes -o vllm-linux-rocm723.txt
python engine_compat.py vllm-linux-rocm723
```

Keep each lock's torch, torchvision, torchaudio and CUDA pins equal to the Studio
install it serves, or no installation can share them. `engine_compat.py` also records
each wheel's download size, which the install prompt shows. SGLang's excludes file drops `outlines`,
whose `outlines-core` pin has no Python 3.13 wheel; SGLang only imports it for
`--grammar-backend outlines`. Every lock drops `nvidia-cutlass-dsl-libs-cu12`: CUTLASS DSL loads the newest
CUDA flavour the driver runs, and the engines require driver 580 (CUDA 13).

Add a release to `PROFILES` (newest first) and `PYTHON` in `engine_install.py` when
changing the runtime baseline. SGLang's requirements explicitly pin its required Flash
Attention beta and a Transformers-compatible `kernels` release. Do not enable
prereleases globally. Regeneration must refresh hashes when changing indexes.

Validate installation, failed repair, restoration, streaming usage, cancellation,
process cleanup and training handoff. Test server startup and real generation on
supported hardware: importing an engine alone does not validate its CUDA runtime.
Exact SGLang prompt counting is currently unavailable; generated usage is reported
from the server. Performance and storage savings require separate measurements.
