# Unsloth on AMD (ROCm)

Fine-tune and run LLMs on AMD GPUs with no setup. This image carries the full training stack built against ROCm: PyTorch with ROCm 7.2, Unsloth, unsloth-zoo, a bitsandbytes build with the ROCm 4-bit fix, triton-rocm, TRL, PEFT and diffusers.

This is the AMD counterpart to [`unsloth/unsloth`](https://hub.docker.com/r/unsloth/unsloth), which is CUDA-only. Source: [`docker/Dockerfile.rocm`](https://github.com/unslothai/unsloth/blob/main/docker/Dockerfile.rocm). Guide: [docs.unsloth.ai](https://docs.unsloth.ai/get-started/install/docker).

## Quick start

Needs Docker and, on Linux, a working amdgpu driver. Windows goes through WSL2 and is covered below. Without Docker, start here:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
```

AMD GPUs are reached through the kernel driver's device nodes, not through a container toolkit, so the run command differs from the NVIDIA one. On Linux:

```bash
GPU_FLAGS="--device /dev/kfd"
[ -e /dev/dri ] && GPU_FLAGS="$GPU_FLAGS --device /dev/dri"
for g in video render; do
  gid=$(getent group "$g" | cut -d: -f3)
  [ -n "$gid" ] && GPU_FLAGS="$GPU_FLAGS --group-add $gid"
done

docker run --rm -it \
  $GPU_FLAGS \
  --ipc=host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  unsloth/unsloth-rocm
```

Those few lines build the flags the way `docker/run.sh` does, and each part earns its place. `--group-add` needs the numeric group ids, because a name is resolved inside the container, where the host's `video` and `render` groups do not exist, so passing the names can add the wrong groups and leave `/dev/kfd` unreadable. A minimal host may have no `render` group at all, and a plain `--group-add "$(getent group render | cut -d: -f3)"` would then expand to an empty argument. And Docker refuses to start over a device that does not exist, so `/dev/dri` is named only when it is there, leaving the entrypoint to explain an incomplete driver rather than failing in the daemon.

On Windows there is no `/dev/kfd`. WSL2 reaches the card over its DXG bridge, so the flags are the `/dev/dxg` node, the runtime's opt-in, and the host's `librocdxg` together with the `libdxcore` it loads from WSL's own lib directory (the image cannot ship `librocdxg`: its build needs Windows SDK headers). This needs ROCm for WSL on the host (`scripts/install_rocm_wsl_strixhalo.sh` installs it), a docker engine running inside the WSL distribution (Docker Desktop's own engine exposes neither node), and a per-architecture image: the generic image's torch bundles `librocprofiler-sdk`, which aborts on the bridge, and the entrypoint refuses before it gets that far. Until the per-architecture tags are published, build one with `ROCM_GFX=<your gfx> bash docker/build.sh --rocm`, which tags it `unsloth-rocm:latest` locally, then:

```bash
GPU_FLAGS="--device /dev/dxg -e HSA_ENABLE_DXG_DETECTION=1 \
  -v /opt/rocm/lib/librocdxg.so.1:/usr/lib/x86_64-linux-gnu/librocdxg.so:ro \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib"

docker run --rm -it \
  $GPU_FLAGS \
  --ipc=host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  unsloth-rocm:latest
```

No `--group-add` here: WSL exposes `/dev/dxg` to everyone and has no `render` group.

The Hugging Face mount is not optional if you care about your downloads: `HF_HOME` inside the container is `/workspace/.cache/huggingface`, which lives in the container's writable layer, so without it every model is fetched again after `docker rm`.

Or let the launcher work out the device nodes, group ids and mounts for you, on Linux and on WSL alike. It defaults to the published image, so on WSL name the build from above:

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/run.sh -o run.sh
bash run.sh --rocm                                   # Linux
UNSLOTH_IMAGE=unsloth-rocm:latest bash run.sh --rocm # WSL
```

Check the GPU is visible before anything else, with `GPU_FLAGS` set as above (on WSL, `unsloth-rocm:latest` in place of `unsloth/unsloth-rocm`):

```bash
docker run --rm $GPU_FLAGS \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  unsloth/unsloth-rocm python /workspace/smoke_test_rocm.py
```

That runs a real 5-step LoRA on a 1B model and fails loudly if the GPU is not usable. The cache mount is there so a second attempt reuses the model rather than downloading it again: the container is `--rm`, so without it every run starts from nothing.

## Tags

| tag | what it is |
|---|---|
| `latest` | the current build from `main`, linux/amd64 |
| `sha-<commit>` | the same image, pinned to the commit it was built from |
| `nightly` | the scheduled weekly build |
| `gfx1150`, `gfx1151`, `gfx1152`, `gfx1200`, `gfx1201` | builds using AMD's per-architecture wheels, when published |

Pin a digest for anything reproducible. `latest` moves.

## Supported hardware

RDNA2 and newer, and CDNA, except `gfx1033`. The image is built against the generic ROCm 7.2 PyTorch index, which covers most cards.

Three cases need care:

- **Strix Halo / Strix Point APUs (`gfx1150`, `gfx1151`, `gfx1152`) and RDNA4 (`gfx1200`, `gfx1201`)** run on the generic wheels, and the entrypoint says so on start, but AMD's per-architecture wheels carry fixes the generic index lacks, not only tuning: the `_grouped_mm` segfault on `gfx1151` is one of them. The generic image may well work for what you run, and everything measured below was measured on it, but if training crashes on one of these arches, rebuilding with `ROCM_GFX=gfx1151` (or your arch) is the fix rather than a speedup.
- **Van Gogh (`gfx1033`, Steam Deck)** is refused outright. It is RDNA2, but training diverges to NaN under ROCm while forward passes look valid, so the entrypoint exits rather than train on it, and `HSA_OVERRIDE_GFX_VERSION` does not help because it hides the silicon, not the arithmetic.
- **Vega 20 (`gfx906`)** lost its kernels after ROCm 6.3. Build with `ROCM_GFX=gfx906` and `ROCM_VERSION=6.3.4`; that variant ships without bitsandbytes, since no prebuilt wheel carries gfx906 kernels.

The container prints what it found and refuses to start rather than silently falling back to the CPU. Set `UNSLOTH_SKIP_GPU_CHECK=1` to bypass the diagnostics.

## What is measured

On a Radeon 8060S (`gfx1151`, Strix Halo), against the published `latest` digest:

| workload | result |
|---|---|
| 4-bit QLoRA, 5 steps, Llama 3.2 1B | loss 2.8146 to 1.2242 |
| text to image, sd-turbo, 4 steps, 512px | 118 s, 4.21 GB peak VRAM |
| text to video, text-to-video-ms-1.7b, 8 frames, 256px | 279 s, 4.77 GB peak VRAM |

The diffusion outputs match the same seed on an NVIDIA B200 to three decimal places of pixel spread, so this is numerical agreement rather than only "it ran". Speed is another matter: an integrated APU sharing system memory is roughly 25x to 50x slower than a datacentre card on those two generations.

Discrete RDNA2, RDNA4 and CDNA cards are not covered by that testing.

## What is in the image, and what is not

Included: PyTorch with ROCm, Unsloth, unsloth-zoo, transformers, TRL, PEFT, accelerate, bitsandbytes, triton-rocm, diffusers, timm.

Not included, unlike `unsloth/unsloth`: Unsloth Studio and its web UI, JupyterLab, prebuilt llama.cpp and whisper.cpp, vLLM and xformers. This is a training image; GGUF tooling and the UI are CUDA-only for now.

## Environment

| variable | effect |
|---|---|
| `UNSLOTH_SKIP_GPU_CHECK=1` | skip the startup diagnostics |
| `HSA_OVERRIDE_GFX_VERSION` | present an unsupported card as a supported one. Ignored on images with native kernels for the card |
| `HF_TOKEN` | forwarded for gated models |

Model downloads land in `/workspace/.cache/huggingface`, which is in the container's writable layer unless you mount it. Mount it to keep them, and to reuse what the host has already downloaded.

The build's own record is in the image: `cat /etc/unsloth-rocm-build` reports the ROCm version, the wheel index and the architecture it was built for, and `/opt/unsloth-venv/requirements.lock.txt` is the resolved package set.

## Links

- Source and issues: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- Docker files: [`docker/`](https://github.com/unslothai/unsloth/tree/main/docker)
- Documentation: [docs.unsloth.ai](https://docs.unsloth.ai)
- Licences: the image is labelled `Apache-2.0 AND AGPL-3.0-only`. [Apache-2.0](https://github.com/unslothai/unsloth/blob/main/LICENSE) covers the repository and the training stack. The [AGPL-3.0](https://github.com/unslothai/unsloth/blob/main/studio/LICENSE.AGPL-3.0) half is there for the one AGPL file this image ships, `/workspace/smoke_test_rocm.py`, which carries an `AGPL-3.0-only` header; Unsloth Studio itself is not in this image.
