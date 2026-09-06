# Unsloth Docker Image

Pre-built images for [Unsloth](https://github.com/unslothai/unsloth): fine-tune and run LLMs, vision, audio and diffusion models with no setup. Every image carries the full training stack (PyTorch 2.11 with CUDA 12.8, Unsloth, unsloth-zoo, bitsandbytes, xformers, TRL, PEFT), JupyterLab with the [Unsloth notebooks](https://github.com/unslothai/notebooks) pre-synced, and prebuilt llama.cpp and whisper.cpp for GGUF work.

Source: [`docker/`](https://github.com/unslothai/unsloth/tree/main/docker) in the main repository. Guide: [docs.unsloth.ai](https://docs.unsloth.ai/get-started/install-and-update/docker).

## Tags

| Tag | Contents | Use it for |
|---|---|---|
| `latest`, `studio` | Unsloth Studio web UI + JupyterLab + notebooks + key-only SSH | Most users. Train and chat in the browser. |
| `core` | Training stack + JupyterLab + notebooks, no Studio | Notebooks, scripts, CI, slimmer pulls. |
| `nightly-<YYYY.MM.DD>`, `core-nightly-<YYYY.MM.DD>` | The same two images, one immutable pin per daily rebuild, kept 60 days | Reproducible runs. |
| `<version>`, `core-<version>` | Release builds | Pin a release. |

`latest` and `core` move with every push to `main` and on a daily rebuild. Both images are multi-arch: `linux/amd64` and `linux/arm64` (GH200, DGX Spark).

## Quick start

Needs an NVIDIA driver of 570.26 or newer and, on Linux, the NVIDIA Container Toolkit. One command installs the toolkit (Ubuntu, Debian, RHEL, Fedora, Rocky, Amazon Linux, SUSE) and checks a container can see the GPU:

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh
```

On Windows use Docker Desktop with the WSL 2 backend and a current NVIDIA Windows driver; nothing else to install. Then:

```bash
docker run -d --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8888:8888 \
  -e UNSLOTH_STUDIO_PASSWORD="choose-a-password" \
  -e JUPYTER_PASSWORD="choose-a-password" \
  -v "$PWD":/workspace/host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  unsloth/unsloth
```

`docker run -d` returns at once; follow the startup with `docker logs -f <container>`, which ends with a ready block once both services answer (Studio takes about a minute). Then open Studio at `http://localhost:8000` (user `unsloth`) and JupyterLab at `http://localhost:8888`. Leave either password variable unset and a random one is generated and printed in that log.

The `docker/run.sh` helper in the repository sets these flags for you:

```bash
git clone https://github.com/unslothai/unsloth && cd unsloth
UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash docker/run.sh
```

### Notebooks only (`core`)

The `core` image has no service manager. Start JupyterLab on the command line:

```bash
docker run -d --gpus all --ipc=host -p 8888:8888 \
  -v "$PWD":/workspace/host \
  unsloth/unsloth:core \
  jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
```

The login token is printed in `docker logs`. With no command the image runs `python`, so a bare `docker run unsloth/unsloth:core` exits immediately.

### Scripts

```bash
docker run --rm --gpus all --ipc=host -v "$PWD":/workspace/host \
  unsloth/unsloth:core python /workspace/host/train.py
```

### CPU-only hosts

Without a GPU the container refuses to start unless you opt in. Studio chat with GGUF models, JupyterLab and the GGUF tooling work; training does not.

```bash
docker run -d -e UNSLOTH_ALLOW_CPU=1 -p 8000:8000 -p 8888:8888 unsloth/unsloth
```

## Supported GPUs

Compiled for `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`: Turing (T4, RTX 20), Ampere (A100, A10, RTX 30), Ada (L4, L40, RTX 40), Hopper (H100, H200, GH200), Blackwell (B200, GB200, RTX 50, RTX PRO 6000) and GB10 (DGX Spark). The container prints the detected GPU on start and explains what to do when the driver is too old.

Driver requirements:

- 570.26 or newer for CUDA 12.8 on every GPU.
- 580 or newer for B300, GB300 and GB10.
- On `linux/arm64` the bundled llama.cpp is a CUDA 13 build because upstream ships no CUDA 12 build for that architecture. Training works from driver 570, but GGUF export and Studio chat need 580 or newer.

Turing has no bfloat16; Unsloth falls back to float16 there. AMD GPUs are not supported by these images.

## Ports

| Port | Service | Image |
|---|---|---|
| 8000 | Unsloth Studio | `latest` |
| 8888 | JupyterLab | both |
| 22 | SSH, key only, off unless `SSH_KEY` or `PUBLIC_KEY` is set | `latest` |

## Environment variables

| Variable | Effect |
|---|---|
| `UNSLOTH_STUDIO_PASSWORD` | Initial Studio admin password for user `unsloth`; ignored once a password is stored. Unset: generated once and printed in the logs, and Studio stops after an hour unless it is changed (`UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0` disables). |
| `JUPYTER_PASSWORD` | JupyterLab password. Unset: generated once and printed in the logs. |
| `JUPYTER_PORT` | JupyterLab port inside the container. Default `8888`. |
| `SSH_KEY` or `PUBLIC_KEY` | OpenSSH public key for root login. Enables sshd on port 22. Password login is never enabled. |
| `UNSLOTH_ALLOW_CPU=1` | Allow starting without a GPU. |
| `UNSLOTH_JUPYTER_CLOUDFLARE=1` | Publish JupyterLab through a Cloudflare quick tunnel and print the URL. |
| `UNSLOTH_SKIP_NOTEBOOK_SYNC=1` | Do not refresh the notebooks from GitHub on start. |
| `HF_TOKEN`, `WANDB_API_KEY` | Forwarded to Hugging Face and Weights and Biases. |

## Volumes

The working directory is `/workspace`. Mount what you want to keep:

| Container path | What it holds |
|---|---|
| `/workspace/host` | Your files. Mount your project directory here. |
| `/workspace/.cache/huggingface` | Model downloads. Mount your host HF cache to reuse it. |
| `/workspace/.cache/triton` | Compiled kernels. Optional, speeds up restarts. |
| `/workspace/unsloth-notebooks` | The synced notebooks. Your edits are kept across refreshes. |
| `/workspace/Unsloth Notebooks` | The same notebooks grouped by topic, rebuilt on each start. |

The container runs as root by default. `--user <uid>:<gid>` is supported and keeps files on your mounts owned by you.

## Updating inside a running container

On the `latest` image:

- `unsloth-studio-update` upgrades Studio and Unsloth in place.
- `unsloth-llama-update` fetches the newest prebuilt llama.cpp.
- `unsloth-jupyter-tunnel` opens a Cloudflare quick tunnel to JupyterLab.

On both images the notebooks refresh from GitHub on each start unless `UNSLOTH_SKIP_NOTEBOOK_SYNC=1`. Pull a new image tag to update everything else.

## Help

- [Documentation](https://docs.unsloth.ai)
- [r/unsloth](https://reddit.com/r/unsloth)
- [Issues](https://github.com/unslothai/unsloth/issues)

## License

AGPL-3.0, following the main repository. See [LICENSE](https://github.com/unslothai/unsloth/blob/main/LICENSE).
