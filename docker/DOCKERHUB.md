# Unsloth Docker Image

Fine-tune and run LLMs, vision, audio and diffusion models with no setup. Every image carries the training stack (PyTorch 2.11 with CUDA 12.8, [Unsloth](https://github.com/unslothai/unsloth), unsloth-zoo, bitsandbytes, TRL, PEFT, plus xformers on `linux/amd64`), JupyterLab with the [Unsloth notebooks](https://github.com/unslothai/notebooks), and prebuilt llama.cpp. The `latest` image adds whisper.cpp for Unsloth Studio's speech-to-text.

Source: [`docker/`](https://github.com/unslothai/unsloth/tree/main/docker). Guide: [docs.unsloth.ai](https://docs.unsloth.ai/get-started/install/docker).

## Quick start

```bash
# use  -e UNSLOTH_STUDIO_SECURE=1  instead of -p 8000:8000 for a public Cloudflare HTTPS link
docker run -d --name unsloth --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8888:8888 \
  -v "$PWD":/workspace/host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  -v unsloth-studio:/opt/unsloth-studio \
  unsloth/unsloth && docker logs -f unsloth
```

That starts the container and follows its startup, which ends after about a minute with your links and a generated JupyterLab password. On the first run against a new `unsloth-studio` volume it generates an Unsloth Studio password too; change that one on first sign-in or Unsloth Studio stops after an hour. Reusing an existing volume keeps the Unsloth Studio password already stored on it, which the log says rather than printing one. Ctrl-C stops following the log, not the container.

Those ports publish on every interface, which is fine on a laptop and not on a cloud host. Three ways to close that, in order of least work:

- `-e UNSLOTH_STUDIO_SECURE=1`, with `-p 8000:8000` dropped and 8888 changed to `-p 127.0.0.1:8888:8888`: Unsloth Studio is served only over a Cloudflare HTTPS link, printed in the log, and binds to loopback inside the container so no raw port exists. It fails closed, so no tunnel means no link rather than serving in the clear. Move JupyterLab too, or it stays published on every interface over plain HTTP, which is a shell on your GPU box with its password in the clear. On a new volume the public page will not hand out the generated first-boot password, on purpose; read it from the log, where it is printed for exactly this case, or set one yourself with `UNSLOTH_STUDIO_PASSWORD`. `UNSLOTH_STUDIO_CLOUDFLARE=1` keeps the local port as well.
- Publish to `127.0.0.1` and tunnel: `-p 127.0.0.1:8000:8000 -p 127.0.0.1:8888:8888`, then `ssh -L 8000:localhost:8000 -L 8888:localhost:8888 user@your-host`.
- Keep the ports and set real passwords with `-e UNSLOTH_STUDIO_PASSWORD=...` and `-e JUPYTER_PASSWORD=...`.

Unsloth Studio reports on start whether the port answered from the public internet, so you are told either way.

### Before that, on a new machine

Needs an NVIDIA driver of 570.26 or newer. On Linux you also need Docker and the NVIDIA Container Toolkit:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh
```

The second command covers Ubuntu, Debian, RHEL, Fedora, Rocky, Amazon Linux and SUSE, and checks that a container can see the GPU. On Windows, Docker Desktop with the WSL 2 backend and a current NVIDIA Windows driver is all you need; run the quick start above in a WSL 2 shell.

Or let `run.sh` set the quick start's flags, including the `unsloth-studio` volume, for you. It offers to install the toolkit if the daemon has no `nvidia` runtime:

```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/run.sh -o run.sh
UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash run.sh
```

### Resetting the password

```bash
docker exec unsloth unsloth studio reset-password --username unsloth
```

It mints a new password and prints it, revokes sessions and API keys, and rotates in place, so nothing needs restarting. `--username` is required once more than one account exists and harmless before that. Shared `/p` preview links are not revoked; rotate those in Settings if the old password leaked.

### Stopping and removing it

```bash
docker stop unsloth      # shut it down, keep it
docker start unsloth     # bring it back, same state
docker rm -f unsloth     # stop and delete the container
docker ps -a             # find it again, running or not
```

`docker rm -f` deletes the container, not your work: models stay in the Hugging Face cache, your files in the directory you mounted, and Unsloth Studio's accounts and chats on the `unsloth-studio` volume, so the next container with the same `-v unsloth-studio:/opt/unsloth-studio` resumes where this one left off, password included. Only what was written inside the container is lost. `docker volume rm unsloth-studio` discards the Unsloth Studio data too, and cannot be undone.

These take a container, not an image, so `docker stop unsloth/unsloth` fails with "No such container". Use the `NAMES` or `CONTAINER ID` column of `docker ps -a`; without `--name`, Docker assigns a random one.

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

`latest` starts without a GPU on its own. Unsloth Studio chat with GGUF models, JupyterLab and the GGUF tooling work; training does not. `core` refuses to start without a GPU unless you opt in:

```bash
docker run -d -p 8000:8000 -p 8888:8888 unsloth/unsloth
docker run --rm -e UNSLOTH_ALLOW_CPU=1 unsloth/unsloth:core python -c "import unsloth"
```

## Tags

| Tag | Contents | Use it for |
|---|---|---|
| `latest`, `studio` | Unsloth Studio web UI + JupyterLab + notebooks + key-only SSH | Most users. Train and chat in the browser. |
| `core` | Training stack + JupyterLab + notebooks, no Unsloth Studio | Notebooks, scripts, CI, slimmer pulls. |
| `nightly-<YYYY.MM.DD>`, `core-nightly-<YYYY.MM.DD>` | The same two images, one immutable pin per daily rebuild, kept 60 days | Reproducible runs. |

`latest` and `core` are rebuilt daily, not on every merge to `main`. Both are multi-arch: `linux/amd64` and `linux/arm64` (GH200, DGX Spark).

## Supported GPUs

Turing (T4, RTX 20) through Blackwell (B200, GB200, RTX 50, RTX PRO 6000) run precompiled SASS: `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` on `linux/amd64`, `sm_80 sm_90 sm_90a sm_100 sm_100a sm_120 sm_120a` on `linux/arm64`. SASS is forward-compatible within a major version, so Ada runs the `sm_86` binaries, B300 and GB300 the `sm_100` ones, and GB10 (DGX Spark) the `sm_120` ones. A source build inside the image compiles for `7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX`.

Drivers: 570.26 or newer for CUDA 12.8 on every GPU, 580 or newer for B300, GB300 and GB10. On `linux/arm64` the bundled llama.cpp is a CUDA 13 build, because upstream ships no CUDA 12 one there, so training works from 570 but GGUF export and Unsloth Studio chat need 580.

Turing has no bfloat16; Unsloth falls back to float16 there. These images are CUDA only; for AMD use [`unsloth/unsloth-rocm`](https://hub.docker.com/r/unsloth/unsloth-rocm), a ROCm build of the training stack.

## Ports

| Port | Service | Image |
|---|---|---|
| 8000 | Unsloth Studio (`UNSLOTH_STUDIO_PORT`) | `latest` |
| 8888 | JupyterLab | both |
| 22 | SSH, key only, off unless `SSH_KEY` or `PUBLIC_KEY` is set | `latest` |

## Environment variables

| Variable | Effect |
|---|---|
| `UNSLOTH_STUDIO_PASSWORD` | Initial Unsloth Studio password for user `unsloth`; ignored once one is stored. Unset: generated, printed in the logs, and Unsloth Studio stops after an hour unless changed (`UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0` disables). |
| `JUPYTER_PASSWORD` | JupyterLab password, read by the `latest` launcher. Unset: generated and printed in the logs. On `core` you start JupyterLab yourself. |
| `JUPYTER_PORT` | JupyterLab's port inside the container. Default `8888`. Unsloth Studio's port (`UNSLOTH_STUDIO_PORT`) is refused. |
| `UNSLOTH_STUDIO_PORT` | Unsloth Studio's port inside the container, for `--network host` when 8000 is taken. Default `8000`. |
| `SSH_KEY` or `PUBLIC_KEY` | OpenSSH public key for root login; enables sshd on port 22. Password login is never enabled. |
| `UNSLOTH_ALLOW_CPU=1` | Allow starting without a GPU (`latest` already does). Dropped when a GPU is visible, where it would turn off Unsloth's training patches. |
| `UNSLOTH_STUDIO_SECURE=1` | Serve Unsloth Studio over a Cloudflare HTTPS link only, bound to loopback inside the container so no raw port is published. Fails closed if the tunnel does not come up. |
| `UNSLOTH_STUDIO_CLOUDFLARE=1` | The same HTTPS link, with the local port still served. Not valid together with `UNSLOTH_STUDIO_SECURE=1`. |
| `UNSLOTH_JUPYTER_CLOUDFLARE=1` | Publish JupyterLab through a Cloudflare quick tunnel and print the URL. |
| `UNSLOTH_SKIP_NOTEBOOK_REFRESH=1` | Do not refresh the notebooks from GitHub on start; the copy baked into the image is still used. |
| `UNSLOTH_SKIP_NOTEBOOK_SYNC=1` | Do not set up the notebooks at all: nothing is created at `/workspace/unsloth-notebooks` or `/workspace/Unsloth Notebooks`. |
| `HF_TOKEN`, `WANDB_API_KEY` | Forwarded to Hugging Face and Weights and Biases. |
| `UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S` | How long a training run gets to save a checkpoint when the container stops. Default `120`, and capped by `UNSLOTH_STUDIO_TRAINING_STOP_TIMEOUT_S` (default `600`). |

On a CPU-only host a non-login `docker exec` is built from the image, not the container's first process, so `docker exec <c> python -c "import unsloth"` still asks for a GPU. Use `bash -lc '...'` or pass `-e UNSLOTH_ALLOW_CPU=1`.

## Volumes

The working directory is `/workspace`. Mount what you want to keep:

| Container path | What it holds |
|---|---|
| `/workspace/host` | Your files. Mount your project directory here. |
| `/workspace/.cache/huggingface` | Model downloads. Mount your host HF cache to reuse it. |
| `/opt/unsloth-studio` | Unsloth Studio's accounts, chats, outputs, exports and runs (`latest`). Use a named volume: without one, `docker rm` loses them. |
| `/workspace/.cache/triton` | Compiled kernels. Optional, speeds up restarts. |
| `/workspace/unsloth-notebooks` | The synced notebooks. Your edits are kept across refreshes. |
| `/workspace/Unsloth Notebooks` | The same notebooks grouped by topic, rebuilt on each start. |

The container runs as root by default. On `core`, `--user <uid>:<gid>` is supported and keeps files on your mounts owned by you. `latest` runs its services as root and does not start under `--user`.

<a id="studio-data-and-studio-code"></a>

### Unsloth Studio data and code

Unsloth Studio's code ships in the image under `/opt/unsloth-studio-app` and is linked into `/opt/unsloth-studio` at every start, so the volume holds only your data and every image runs its own code.

- A volume from before that split holds an old image's code. The first start of a newer image moves it aside to `/opt/unsloth-studio/.unsloth-studio-legacy/` and links the new code in; nothing is deleted. Delete it once the new image works (`docker exec <c> rm -rf /opt/unsloth-studio/.unsloth-studio-legacy`), along with the 9 GB uv cache at `/opt/unsloth-studio/cache/uv`, or start with `-e UNSLOTH_STUDIO_KEEP_LEGACY=0` to skip keeping it. Going back to a pre-split image needs those entries moved back, or `docker run --rm -v unsloth-studio:/opt/unsloth-studio --entrypoint unsloth-studio-home <current image> --restore`.
- Use a named volume, not a bind mount of a Windows or macOS host directory: the Unsloth Studio home needs symlinks, and Docker Desktop's file sharing may refuse to create them, stopping the container at start. A Linux directory, including one inside a WSL 2 distribution, works.

## Stopping while training

On `docker stop` and `docker restart`, Unsloth Studio stops a running training job at the next step and saves a checkpoint before it exits, so the run can be resumed from the Training page. Docker only waits 10 seconds by default, which is not enough for a large model, so give it the budget:

```bash
docker stop -t 150 <container>
```

or `stop_grace_period: 150s` in Compose. `docker/run.sh` sets this for you. `UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S` (default 120) is the time Unsloth Studio itself waits for the save; the container's own limit is 30 seconds above it.

A save is also bounded by the training stop watchdog, which force-terminates a worker that has not finished after `UNSLOTH_STUDIO_TRAINING_STOP_TIMEOUT_S` (default 600). Raising the shutdown budget past that does nothing on its own; raise both, and `docker/run.sh` forwards both.

A host shutdown is not covered: the daemon stops every container under its own `--shutdown-timeout` (15 seconds by default), so a save that takes longer is cut short. Stop the container yourself before shutting the host down.

## Updating inside a running container

On the `latest` image:

- `unsloth-studio-update` upgrades Unsloth Studio and Unsloth in place, leaving the torch and CUDA stack pinned. It survives `docker restart`, not `docker rm`.
- `unsloth-llama-update` fetches the newest prebuilt llama.cpp.
- `unsloth-jupyter-tunnel --force` opens a Cloudflare quick tunnel to JupyterLab. Without `--force` or `UNSLOTH_JUPYTER_CLOUDFLARE=1` it prints that it is disabled and exits.

Both images refresh the notebooks from GitHub on each start, which costs 10 to 20 seconds, so a one-shot `docker run --rm ... python script.py` is worth running with `UNSLOTH_SKIP_NOTEBOOK_SYNC=1`. Pull a new image tag to update everything else.

## Help

- [Documentation](https://docs.unsloth.ai)
- [r/unsloth](https://reddit.com/r/unsloth)
- [Issues](https://github.com/unslothai/unsloth/issues)

## License

Unsloth is Apache-2.0 ([LICENSE](https://github.com/unslothai/unsloth/blob/main/LICENSE)). Both images also include Unsloth Studio's code (`studio/`), which is AGPL-3.0 ([studio/LICENSE.AGPL-3.0](https://github.com/unslothai/unsloth/blob/main/studio/LICENSE.AGPL-3.0)).
