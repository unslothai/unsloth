# Review 8/12 (rc=0)

Operating as simulation persona.

**Summary**

PR #5748 adds a new Docker publishing pipeline and Docker image layout for CUDA 12.8 / Blackwell-era NVIDIA GPUs, plus two runtime compatibility patches in `unsloth/_gpu_init.py` and `unsloth/models/vision.py`. The Docker work is broad: multi-arch CI, build/run/freeze/HF helper scripts, a GPU-checking entrypoint, a smoke test, and a Studio image variant.

**Findings**

**[P2] `unsloth/_gpu_init.py:84`** -- The Docker GPU fingerprint is too broad and forces single-thread Inductor compilation for normal `--gpus all` containers. The code keys only on `NVIDIA_VISIBLE_DEVICES` being present and `CUDA_VISIBLE_DEVICES` being absent, but NVIDIA CUDA base images commonly set `NVIDIA_VISIBLE_DEVICES=all` by default, so ordinary all-GPU Docker runs are treated like the broken cgroup-pinned `device=N` case. I reproduced the branch behavior with the exact condition from the diff:

```text
docker_all_default -> {'NVIDIA_VISIBLE_DEVICES': 'all', 'CUDA_VISIBLE_DEVICES': None, 'TORCHINDUCTOR_COMPILE_THREADS': '1', 'UNSLOTH_FORCE_SINGLE_COMPILE_WORKER': '1'}
docker_device_0 -> {'NVIDIA_VISIBLE_DEVICES': '0', 'CUDA_VISIBLE_DEVICES': None, 'TORCHINDUCTOR_COMPILE_THREADS': '1', 'UNSLOTH_FORCE_SINGLE_COMPILE_WORKER': '1'}
```

This contradicts the PR compatibility note that `--gpus all` is untouched, and it slows compile-heavy runs unnecessarily. NVIDIA’s CUDA image sources also show the base images setting `ENV NVIDIA_VISIBLE_DEVICES all` in CUDA Ubuntu images: https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/12.6.3/ubuntu2404/base/Dockerfile

Suggested fix:

```python
_visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
_is_cgroup_pinned = (
    _visible_devices is not None
    and _visible_devices.strip().lower() not in {"", "all", "none", "void"}
)
if (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and _is_cgroup_pinned
    and "CUDA_VISIBLE_DEVICES" not in os.environ
    and "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ
):
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

**[P2] `unsloth/_gpu_init.py:84`** -- A user who already set `TORCHINDUCTOR_COMPILE_THREADS=1` does not get the sentinel, so the later re-assertion never runs. This is the exact “explicitly forced single-worker” case the PR is trying to preserve against older `unsloth_zoo.patch_torch_compile`, but the new guard skips the block when `TORCHINDUCTOR_COMPILE_THREADS` is already present. Reproduction from the same condition:

```text
user_already_forced -> {'NVIDIA_VISIBLE_DEVICES': '0', 'CUDA_VISIBLE_DEVICES': None, 'TORCHINDUCTOR_COMPILE_THREADS': '1', 'UNSLOTH_FORCE_SINGLE_COMPILE_WORKER': None}
```

Because `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER` remains unset, the later block at lines 144-154 does not re-populate the env var after zoo pops it.

Suggested fix:

```python
_visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
_is_cgroup_pinned = (
    _visible_devices is not None
    and _visible_devices.strip().lower() not in {"", "all", "none", "void"}
)
if (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and _is_cgroup_pinned
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") in {None, "1"}:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

**[P2] `docker/Dockerfile:199`** -- The vLLM install pass says `--no-deps` protects the pinned Unsloth stack, but the command does not pass `--no-deps`. With `--pre` enabled, the resolver is allowed to bring prerelease transitive dependencies into the published amd64 image. I ran the vLLM resolver path with the same indexes and saw prerelease packages selected, including `pydantic==2.14.0a1`, `safetensors==0.8.0rc0`, `tokenizers==0.23.0rc0`, `grpcio==1.81.0rc1`, and `sentry-sdk==3.0.0a7`. This reintroduces the dependency drift the Dockerfile comments say the split install is meant to avoid.

Suggested fix:

```dockerfile
        ${VENV}/bin/uv pip install \
            --python ${VENV}/bin/python \
            --pre \
            --index-strategy unsafe-best-match \
            --extra-index-url https://wheels.vllm.ai/nightly \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            --no-deps \
            "torch==2.10.0" \
            vllm; \
        ${VENV}/bin/uv pip check; \
```

If vLLM truly needs additional runtime deps beyond the Unsloth stack, install those explicitly with stable bounds instead of letting a global `--pre vllm` solve the whole environment.

**[P2] `docker/smoke_test.py:42`** -- The smoke test rejects Turing GPUs even though the image and entrypoint advertise sm_75 support. The Dockerfile compiles with `TORCH_CUDA_ARCH_LIST` including `7.5`, and `entrypoint.sh` allows sm_75 with only an fp16 note, but `smoke_test.py` exits for every `cap[0] < 8`. A T4 / RTX 20-series host therefore passes container startup and then fails the bundled validation script before testing imports or training.

Suggested fix:

```python
    if cap[0] < 7 or (cap[0] == 7 and cap[1] < 5):
        sys.exit(f"FAIL: GPU {name} sm_{cap[0]}{cap[1]} is not supported by this image")
    if cap[0] < 8:
        print(f"NOTE: {name} is Turing (sm_{cap[0]}{cap[1]}) -- bfloat16 is not supported.")
        print("      Unsloth will fall back to fp16.")
```

Alternatively, if sm_80+ is the real support boundary, remove sm_75 from the Dockerfile arch list and make `entrypoint.sh` reject it consistently.

**Test Results**

I ran these checks from the provided cwd only:

```text
python pr metadata/revert/lint summaries
```

Result: `revert_report.json` initially listed high-severity reverts, but `auto_fix.applied` is `true` and `post_fix_report` is clean. The checked-out PR branch contains the merge commit and the reported `install.sh`, `install.ps1`, Studio XML strip tests, and `__version__` lines are present in the working tree.

```text
for f in unsloth/docker/*.sh unsloth/docker/entrypoint.sh; do bash -n "$f" || exit 1; done
```

Result: `bash -n ok`.

```text
.venv/bin/python -m py_compile unsloth/docker/smoke_test.py
```

Result: `smoke py_compile ok`.

```text
uv pip install --dry-run --python .venv/bin/python --target ./tmp_uv_probe --pre \
  --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.10.0' vllm
```

Result: resolver selected 173 packages, including prerelease transitive dependencies such as `pydantic==2.14.0a1`, `safetensors==0.8.0rc0`, `tokenizers==0.23.0rc0`, and `grpcio==1.81.0rc1`.

```text
Python reproduction of the new _gpu_init environment guard
```

Result: `NVIDIA_VISIBLE_DEVICES=all` incorrectly sets `TORCHINDUCTOR_COMPILE_THREADS=1`, and `NVIDIA_VISIBLE_DEVICES=0` plus preexisting `TORCHINDUCTOR_COMPILE_THREADS=1` fails to set `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=1`.

```text
docker buildx build --call=outline ...
```

Result: could not complete Docker-level validation because this environment cannot access the Docker daemon socket: `permission denied while trying to connect to the Docker daemon socket`.

Cross-block check: no asymmetric-fix patterns detected.

**Verdict**

REQUEST_CHANGES. The shell/Python syntax is clean and the accidental reverts were auto-fixed in the checked-out tree, but the Docker GPU env guard currently affects normal `--gpus all` containers, misses an explicit user-forced single-worker case, and the vLLM install pass allows prerelease dependency drift in the published image. Those should be tightened before this starts publishing `unsloth/unsloth:latest` from `main`.
