# Optional inference engines

Studio can install and run vLLM or SGLang for local text inference. Installation
is opt-in. Select a model, open its run settings, and choose **Inference engine**.
The installation panel offers **Install and load**. Selecting **Default** returns
to Studio's normal backend. Remembered model settings include the engine choice.

**Settings > System > Inference engines** provides installation, cancellation,
repair, removal, and restoration of the previous installation. Unload the model
before changing an installed engine. Installation continues if the panel closes;
automatic model loading only occurs while its installation panel stays open.
Removing an engine keeps downloaded models and the shared package cache.

## Initial support

This experimental profile requires Linux x86_64, glibc 2.34 or newer, an NVIDIA
GPU with compute capability 8.0 or newer, and driver 580 or newer. It uses
full precision safetensors weights and the model's standard chat template.
Select one or more GPUs in the model settings. Multiple selected GPUs split the
model with tensor parallelism on this machine; all selected devices must satisfy
the hardware requirements. The model must support partitioning across that GPU
count. Studio checks attention heads, KV heads and layer dimensions before
unloading the current model. The engine also validates its runtime constraints.
Llama, Mistral, Qwen2 and Qwen3 text model families are admitted. Individual models
can still exceed available memory or require features outside this profile.

Quantized checkpoints, adapters, tool calling, structured output, reasoning
controls, continuation, custom model code and multimodal input are outside this
profile. Multi-node, pipeline and data parallelism are not configured by Studio.
Engine servers honor Studio's VRAM budget, capped by the most constrained
selected GPU's measured free-memory fraction, with at least 512 MiB left per GPU for driver allocations.
GPU memory is not treated as a single pool: each device must fit its own shard
and runtime allocations. Training evicts them before allocating
GPU memory. Startup includes model downloads and kernel compilation and can take
several minutes. The UI reports phases without inventing progress percentages.

The API derives tensor-parallel size from `gpu_ids`. For example,
`{"engine": "vllm", "gpu_ids": [0, 1]}` selects two physical GPUs for a load.
The existing `tensor_parallel` load switch remains a GGUF setting; optional
engines always use tensor parallelism when more than one GPU is selected. Load
and status responses report the selected IDs and effective `tensor_parallel`.
Remembered model settings preserve the GPU list. Changing it requires reloading.

## Environment and cache design

Each engine has a complete isolated Python 3.12 environment under the Studio
home's `engines` directory. Studio never imports engine packages into its own
Python process. Compiler caches, including Triton, are scoped to each engine.
The committed requirements profiles lock versions and wheel hashes.
Both profiles use PyTorch 2.11.0 with CUDA 13.0 and Transformers 5.6.0.

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
Failed or cancelled installations preserve the prior marker. A successful repair
keeps one previous environment for restoration. Interrupted jobs are reported
after restarting Studio.

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
executables such as Ninja are resolved from the engine environment first.
For tensor parallelism, SGLang probes GPU peer access and falls back to its
standard collective when peer access is unavailable. Its equal-free-memory
guard is disabled for these managed groups: Studio checks every GPU, and SGLang
sizes its cache from the minimum available memory across ranks. This allows
cards with different memory capacities without treating their VRAM as a pool.

## Maintaining profiles

Run from the repository root using uv:

```sh
uv pip compile studio/backend/requirements/engines/vllm-linux-cu130.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 --index-url https://pypi.org/simple --refresh --generate-hashes --upgrade -o studio/backend/requirements/engines/vllm-linux-cu130.txt
uv pip compile studio/backend/requirements/engines/sglang-linux-cu130.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 --index-url https://pypi.org/simple --refresh --generate-hashes --upgrade -o studio/backend/requirements/engines/sglang-linux-cu130.txt
```

Update `PROFILES` in `engine_install.py` and the CUDA import check when changing
the runtime baseline. SGLang's requirements explicitly pin its required Flash
Attention beta and a Transformers-compatible `kernels` release. Do not enable
prereleases globally. Regeneration must refresh hashes when changing indexes.

Validate installation, failed repair, restoration, streaming usage, cancellation,
process cleanup and training handoff. Test server startup and real generation on
supported hardware: importing an engine alone does not validate its CUDA runtime.
Exact SGLang prompt counting is currently unavailable; generated usage is reported
from the server. Performance and storage savings require separate measurements.
