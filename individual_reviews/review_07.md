# Review 7/12 (rc=0)

Operating as regression persona.

**Summary**

This PR adds a Blackwell-oriented Docker image build/publish path, helper scripts, a runtime GPU preflight entrypoint, and two Python runtime changes: `_gpu_init.py` forces single-worker Inductor compilation for selected Docker GPU launches, and `vision.py` stops pre-injecting `logits_to_keep` on Transformers 5+. The Docker workflow and helper scripts are the largest behavioral surface; the Python changes are narrow but affect import-time environment policy.

Cross-block check: no asymmetric-fix patterns detected.

**Findings**

**[P1] `docker/run.sh:59`** -- The wrapper leaks user secrets to stderr because it builds `-e HF_TOKEN=${HF_TOKEN}`, `-e WANDB_API_KEY=${WANDB_API_KEY}`, and `-e UNSLOTH_LICENSE=${UNSLOTH_LICENSE}` in `ENV_FORWARD`, then enables `set -x` immediately before `docker run`. This triggers whenever a user has any of those tokens in their shell and uses `bash docker/run.sh`; the full token values are printed into terminal logs, CI logs, or support transcripts.

Suggested fix:

```bash
# Forward common secrets only if they're set in the host environment.
# Use Docker's "read value from the current environment" form so tokens are not
# expanded into the traced command line.
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"        ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"   ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)

exec docker run --rm -it \
    --gpus "$GPUS_REQUEST" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HF_CACHE":/workspace/.cache/huggingface \
    -v "$TRITON_CACHE":/workspace/.cache/triton \
    -v "$WORK_DIR":/workspace/host \
    "${ENV_FORWARD[@]}" \
    "$IMAGE" "$@"
```

If you still want debuggability, print a redacted command instead of using `set -x`.

**[P2] `docker/run.sh:28`** -- The documented `UNSLOTH_GPUS=0` / `UNSLOTH_GPUS=0,1` examples are passed straight through as `--gpus "$GPUS"` at line 61. Docker treats numeric `--gpus` values as a GPU count request, not a device filter, so `UNSLOTH_GPUS=0` does not mean “GPU 0” and can fail or attach the wrong set. This breaks the wrapper path users need for the same single-device Docker mode that the PR is trying to support.

Suggested fix:

```bash
GPUS="${UNSLOTH_GPUS:-all}"
GPUS_REQUEST="$GPUS"
if [[ "$GPUS" != "all" && "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    GPUS_REQUEST="device=${GPUS}"
fi

exec docker run --rm -it \
    --gpus "$GPUS_REQUEST" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HF_CACHE":/workspace/.cache/huggingface \
    -v "$TRITON_CACHE":/workspace/.cache/triton \
    -v "$WORK_DIR":/workspace/host \
    "${ENV_FORWARD[@]}" \
    "$IMAGE" "$@"
```

Also update the comment to say `UNSLOTH_GPUS=0` maps to Docker’s `device=0` request.

**[P2] `unsloth/_gpu_init.py:88`** -- The new single-worker gate is broader than the scenario described in the comment. NVIDIA CUDA containers commonly have `NVIDIA_VISIBLE_DEVICES=all` while `CUDA_VISIBLE_DEVICES` is absent; the NVIDIA docs describe `all` as the default visible-device value for base CUDA images. In that normal `docker run --gpus all` case, this condition still sets `TORCHINDUCTOR_COMPILE_THREADS=1`, even though the PR metadata says `--gpus all` should be untouched. The trigger is any Docker CUDA image import with `NVIDIA_VISIBLE_DEVICES=all` and no `CUDA_VISIBLE_DEVICES`, which includes the default command path for this new image.

Suggested fix:

```python
_nvidia_visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
_is_explicit_nvidia_device_filter = (
    _nvidia_visible_devices not in (None, "", "all", "none", "void")
)

if (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and _is_explicit_nvidia_device_filter
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") in (None, "1"):
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"

del _nvidia_visible_devices, _is_explicit_nvidia_device_filter
```

That keeps the fix on explicit Docker device filters such as `device=0` / `device=0,1`, while leaving `--gpus all` and non-GPU/offline modes alone.

**Test Results**

I ran these checks inside the provided cwd:

```text
bash -n unsloth/docker/*.sh unsloth/docker/entrypoint.sh
PASS

.venv/bin/python -m py_compile unsloth/docker/smoke_test.py
PASS

PyYAML parse of unsloth/.github/workflows/docker-publish.yml
PASS as YAML syntax, with the usual PyYAML 1.1 caveat that "on" parses as True locally

Simulated _gpu_init guard:
NVIDIA_VISIBLE_DEVICES=all => TORCHINDUCTOR_COMPILE_THREADS=1
NVIDIA_VISIBLE_DEVICES=0   => TORCHINDUCTOR_COMPILE_THREADS=1
NVIDIA_VISIBLE_DEVICES=0 plus TORCHINDUCTOR_COMPILE_THREADS=1 => sentinel not set
```

I also checked live package metadata for `numpy`, `torch`, `torchvision`, and `torchaudio`; the pinned `numpy>=2.4` exists, and `torchvision==0.25.0` declares `torch==2.10.0`. I could not run the full Docker build or image smoke test because this environment cannot access the Docker daemon socket (`permission denied`), and there is no usable GPU path for the container smoke test here.

External references used: NVIDIA Container Toolkit’s Docker environment variable docs for `NVIDIA_VISIBLE_DEVICES=all` behavior, Docker’s GPU CLI docs for `--gpus`, and Docker build-push-action/action-toolkit source showing list inputs are passed as list items unless a comment option is explicitly used:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html  
https://docs.docker.com/engine/containers/gpu/  
https://raw.githubusercontent.com/docker/build-push-action/v6/src/context.ts  
https://raw.githubusercontent.com/docker/actions-toolkit/v0.63.0/src/util.ts

**Verdict**

REQUEST_CHANGES. The Docker image/build direction is plausible, and the syntax checks passed, but the wrapper currently leaks credentials with `set -x`. The GPU-selection wrapper and `_gpu_init.py` gate also need tightening so the new Docker paths behave as documented.
