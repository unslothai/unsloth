# Unsloth Docker Image

Pre-built images for [Unsloth](https://github.com/unslothai/unsloth): fine-tune and run LLMs, vision, audio and diffusion models with no setup. Every image carries the full training stack (PyTorch 2.11 with CUDA 12.8, Unsloth, unsloth-zoo, bitsandbytes, TRL, PEFT, plus xformers on `linux/amd64`), JupyterLab with the [Unsloth notebooks](https://github.com/unslothai/notebooks) pre-synced, and prebuilt llama.cpp for GGUF work. The `latest` image adds whisper.cpp for Studio's speech-to-text.

Source: [`docker/`](https://github.com/unslothai/unsloth/tree/main/docker) in the main repository. Guide: [docs.unsloth.ai](https://docs.unsloth.ai/get-started/install/docker).

## Tags

| Tag | Contents | Use it for |
|---|---|---|
| `latest`, `studio` | Unsloth Studio web UI + JupyterLab + notebooks + key-only SSH | Most users. Train and chat in the browser. |
| `core` | Training stack + JupyterLab + notebooks, no Studio | Notebooks, scripts, CI, slimmer pulls. |
| `nightly-<YYYY.MM.DD>`, `core-nightly-<YYYY.MM.DD>` | The same two images, one immutable pin per daily rebuild, kept 60 days | Reproducible runs. |
| `<version>`, `core-<version>` | Release builds | Pin a release. |

`latest` and `core` are rebuilt daily and on every release tag, not on every merge to `main`. Both images are multi-arch: `linux/amd64` and `linux/arm64` (GH200, DGX Spark).

## Quick start

Needs an NVIDIA driver of 570.26 or newer and, on Linux, Docker plus the NVIDIA Container Toolkit. Without Docker, start here:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
```

Then one command installs the toolkit (Ubuntu, Debian, RHEL, Fedora, Rocky, Amazon Linux, SUSE) and checks a container can see the GPU:

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh
```

On Windows use Docker Desktop with the WSL 2 backend and a current NVIDIA Windows driver; nothing else to install. Then:

```bash
docker run -d --name unsloth --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8888:8888 \
  -v "$PWD":/workspace/host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  -v unsloth-studio:/opt/unsloth-studio \
  unsloth/unsloth && docker logs -f unsloth
```

That starts the container and follows its startup, which ends after about a minute with your links and the passwords generated for this container:

```
  Unsloth container ready
  Studio      http://localhost:8000   username: unsloth   password: HumpedSneerDislikeRetiring   (change it on first sign-in: Studio stops after 60 minutes with the default password)
  JupyterLab  http://localhost:8888   generated password: Wja7F9OPH00S6GHS
```

Sign in with those, and change the Studio one when you first sign in. Ctrl-C stops following the log, not the container. Add `-e UNSLOTH_STUDIO_PASSWORD=...` and `-e JUPYTER_PASSWORD=...` to choose your own, and pick a real one: these ports publish on every interface, so on a cloud host use `-p 127.0.0.1:8000:8000 -p 127.0.0.1:8888:8888` and reach it with `ssh -L 8000:localhost:8000 user@your-host`. Studio reports on start whether the port answered from the public internet.

Or let `run.sh` set these flags, including the `unsloth-studio` volume, for you. It offers to install the NVIDIA Container Toolkit if the daemon has no `nvidia` runtime:

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/run.sh -o run.sh
UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash run.sh
```

### Stopping and removing it

```bash
docker stop unsloth      # shut it down, keep it
docker start unsloth     # bring it back, same state
docker rm -f unsloth     # stop and delete the container
docker ps -a             # find it again, running or not
```

`docker rm -f` deletes the container, not your work. Models stay in the Hugging Face cache, your files stay in the directory you mounted, and Studio's accounts, chats, outputs and runs stay on the `unsloth-studio` volume, so the next container with the same `-v unsloth-studio:/opt/unsloth-studio` picks up where this one left off, password included. Only what was written inside the container is lost, such as an in-container `unsloth-studio-update`.

To discard the Studio data too, which cannot be undone:

```bash
docker volume rm unsloth-studio
```

These take a container, not an image, so `docker stop unsloth/unsloth` fails with "No such container": that is the image the container was made from. Use the `NAMES` or `CONTAINER ID` column of `docker ps -a`. Without `--name unsloth`, Docker assigns a random name like `adoring_albattani`, and the id from `docker run -d` works anywhere a name does.

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

`latest` starts without a GPU on its own. Studio chat with GGUF models, JupyterLab and the GGUF tooling work; training does not. `core` refuses to start without a GPU unless you opt in:

```bash
docker run -d -p 8000:8000 -p 8888:8888 unsloth/unsloth
docker run --rm -e UNSLOTH_ALLOW_CPU=1 unsloth/unsloth:core python -c "import unsloth"
```

## Supported GPUs

Turing (T4, RTX 20), Ampere (A100, A10, RTX 30), Ada (L4, L40, RTX 40), Hopper (H100, H200, GH200) and Blackwell (B200, GB200, RTX 50, RTX PRO 6000) all run precompiled SASS. The cu128 wheels the image ships carry native SASS for `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` on `linux/amd64` and `sm_80 sm_90 sm_90a sm_100 sm_100a sm_120 sm_120a` on `linux/arm64`; SASS is forward-compatible within a major version, so Ada runs the `sm_86` binaries and B300 and GB300 run the `sm_100` ones. A source build inside the image is a separate matter: it compiles for `7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX` on either architecture. GB10 (DGX Spark, `sm_121`) runs the `sm_120` binaries through Blackwell forward compatibility; only kernels compiled at run time, such as Triton, use the CUDA 13 compiler the container switches to on that GPU. The container prints the detected GPU on start and explains what to do when the driver is too old.

Driver requirements:

- 570.26 or newer for CUDA 12.8 on every GPU.
- 580 or newer for B300, GB300 and GB10.
- On `linux/arm64` the bundled llama.cpp is a CUDA 13 build because upstream ships no CUDA 12 build for that architecture. Training works from driver 570, but GGUF export and Studio chat need 580 or newer.

Turing has no bfloat16; Unsloth falls back to float16 there. These images are CUDA only; for AMD use [`unsloth/unsloth-rocm`](https://hub.docker.com/r/unsloth/unsloth-rocm), which carries a ROCm build of the training stack.

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
| `JUPYTER_PASSWORD` | JupyterLab password, read by the `latest` service launcher. Unset: generated once and printed in the logs. On `core` you start JupyterLab yourself, so pass its own flags instead. |
| `JUPYTER_PORT` | JupyterLab port inside the container, read by the `latest` service launcher. Default `8888`. `8000` is refused: that is Studio's port inside the container. |
| `SSH_KEY` or `PUBLIC_KEY` | OpenSSH public key for root login. Enables sshd on port 22. Password login is never enabled. |
| `UNSLOTH_ALLOW_CPU=1` | Allow starting without a GPU (`latest` already does). Dropped when a GPU is visible, where it would turn off Unsloth's training patches. |
| `UNSLOTH_JUPYTER_CLOUDFLARE=1` | Publish JupyterLab through a Cloudflare quick tunnel and print the URL. |
| `UNSLOTH_SKIP_NOTEBOOK_REFRESH=1` | Do not refresh the notebooks from GitHub on start; the copy baked into the image is still used. |
| `UNSLOTH_SKIP_NOTEBOOK_SYNC=1` | Do not set up the notebooks at all: nothing is created at `/workspace/unsloth-notebooks` or `/workspace/Unsloth Notebooks` (copies left there by an earlier start on a mounted `/workspace` stay as they are). |
| `HF_TOKEN`, `WANDB_API_KEY` | Forwarded to Hugging Face and Weights and Biases. |

On a host with no GPU, Studio, JupyterLab and its kernels, and login shells all run in
CPU mode. A non-login `docker exec` is built from the image, not the container's first
process, so `docker exec <c> python -c "import unsloth"` still reports that it needs a
GPU: use `docker exec <c> bash -lc '...'` or `docker exec -e UNSLOTH_ALLOW_CPU=1 <c> ...`.

## Volumes

The working directory is `/workspace`. Mount what you want to keep:

| Container path | What it holds |
|---|---|
| `/workspace/host` | Your files. Mount your project directory here. |
| `/workspace/.cache/huggingface` | Model downloads. Mount your host HF cache to reuse it. |
| `/opt/unsloth-studio` | Studio's accounts, chats, outputs, exports and runs (`latest`). Use a named volume: without one, `docker rm` loses them. A new image still brings new Studio code (see below). |
| `/workspace/.cache/triton` | Compiled kernels. Optional, speeds up restarts. |
| `/workspace/unsloth-notebooks` | The synced notebooks. Your edits are kept across refreshes. |
| `/workspace/Unsloth Notebooks` | The same notebooks grouped by topic, rebuilt on each start. |

The container runs as root by default. On `core`, `--user <uid>:<gid>` is supported and keeps files on your mounts owned by you. `latest` runs its services as root and does not start under `--user`.

### Studio data and Studio code

Studio's code (its venv, source tree, Node, prebuilt tools) ships in the image under `/opt/unsloth-studio-app` and is linked into `/opt/unsloth-studio` at every start, so the volume holds only your data and every image runs its own code. What that means in practice:

- A volume created by an image from before this split holds that image's code as real directories. The first start of a newer image moves each of them aside to `/opt/unsloth-studio/.unsloth-studio-legacy/<name>` on the volume and links the new code in; nothing is deleted, and the log lists what moved. Delete the legacy directory once the new image works (`docker exec <c> rm -rf /opt/unsloth-studio/.unsloth-studio-legacy`), or start with `-e UNSLOTH_STUDIO_KEEP_LEGACY=0` to skip keeping it. Such a volume also holds that image's uv download cache at `/opt/unsloth-studio/cache/uv` (about 9 GB, hardlinked with the legacy venv); once the legacy directory is gone, `rm -rf /opt/unsloth-studio/cache/uv` frees it.
- To go back to an image from before the split on the same volume, move the legacy entries back first: `docker run --rm -v unsloth-studio:/h alpine sh -c 'cd /h && for e in .unsloth-studio-legacy/* .unsloth-studio-legacy/.[!.]*; do [ -e "$e" ] || [ -L "$e" ] || continue; rm -rf "${e##*/}"; mv "$e" .; done'`. Images from after the split need nothing. A volume that never held old code (first used after the split, or its legacy directory deleted) gets a copy of the current image's code instead: `docker run --rm -v unsloth-studio:/opt/unsloth-studio --entrypoint unsloth-studio-home <current image> --restore` (about 5 GB, half a minute), after which the older image runs it.
- `docker rm` still discards anything written into the image's copy: an in-container `unsloth-studio-update` and the `cloudflared` binary `unsloth-jupyter-tunnel` downloads. Studio's caches under `/opt/unsloth-studio/cache` (download resume state, dataset caches) are in the home, so they stay with the volume. Models stay in the Hugging Face cache mount.
- Use a named volume, not a bind mount of a Windows or macOS host directory. The Studio home needs symlinks, and bind mounts through Docker Desktop's file sharing (a Windows drive under WSL 2 in particular) may refuse to create them; the container then stops at start with the linker's error instead of running a half-linked Studio. A bind mount of a Linux directory (including a directory inside the WSL 2 distribution) works.

## Updating inside a running container

On the `latest` image:

- `unsloth-studio-update` upgrades Studio and Unsloth in place, leaving the torch and CUDA stack pinned (in the image's copy: survives `docker restart`, not `docker rm`).
- `unsloth-llama-update` fetches the newest prebuilt llama.cpp.
- `unsloth-jupyter-tunnel --force` opens a Cloudflare quick tunnel to JupyterLab. Without `--force`, and without `UNSLOTH_JUPYTER_CLOUDFLARE=1` in the environment, it prints that it is disabled and exits.

On both images the notebooks refresh from GitHub on each start unless `UNSLOTH_SKIP_NOTEBOOK_REFRESH=1`. Pull a new image tag to update everything else.

Setting them up costs roughly 10 to 20 seconds of every start, depending on the disk, so a one-shot `docker run --rm ... python script.py` is worth running with `UNSLOTH_SKIP_NOTEBOOK_SYNC=1`.

## Help

- [Documentation](https://docs.unsloth.ai)
- [r/unsloth](https://reddit.com/r/unsloth)
- [Issues](https://github.com/unslothai/unsloth/issues)

## License

Unsloth is Apache-2.0 ([LICENSE](https://github.com/unslothai/unsloth/blob/main/LICENSE)). Both images also include Unsloth Studio's code (`studio/`), which is AGPL-3.0 ([studio/LICENSE.AGPL-3.0](https://github.com/unslothai/unsloth/blob/main/studio/LICENSE.AGPL-3.0)).
