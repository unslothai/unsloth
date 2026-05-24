# Review 3/12 (rc=0)

Operating as regression persona.

**Summary**

This PR adds a Docker publishing pipeline and a new `docker/` image build/test toolchain for a CUDA 12.8 Unsloth image, plus two runtime compatibility changes: a container-specific Inductor compile-thread workaround in `unsloth/_gpu_init.py` and a Transformers 5.x `logits_to_keep` behavior change in VLM generation. The Docker image path is mostly coherent, but I found one cross-block asymmetric fix in the Inductor workaround and a few CI/container correctness issues that should be addressed before relying on the workflow.

**Findings**

**[P1] `unsloth/_gpu_init.py:88`** -- Cross-block asymmetric fix: the new single-worker guard skips the exact case where the user already set `TORCHINDUCTOR_COMPILE_THREADS=1`, so `unsloth_zoo.patch_torch_compile(debug=False)` can still remove it at `unsloth-zoo/unsloth_zoo/patching_utils.py:113`. Triggers when a Docker user applies the known workaround manually, for example `docker run --gpus '"device=0"' -e TORCHINDUCTOR_COMPILE_THREADS=1 ...`; the new block does not set `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=1`, Zoo pops the env var, and the Inductor worker pool can still hit the original `Could not find an active GPU backend` failure.

Suggested fix:
```python
visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
force_single_compile_worker = (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and visible_devices not in (None, "", "void", "none", "all")
    and "CUDA_VISIBLE_DEVICES" not in os.environ
)

if force_single_compile_worker:
    compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if compile_threads in (None, "", "1"):
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

**[P2] `unsloth/_gpu_init.py:90`** -- The Docker fingerprint is too broad and forces single-threaded Inductor compilation for `--gpus all`, not just the single-device cgroup case described in the comment. NVIDIA’s container runtime uses `NVIDIA_VISIBLE_DEVICES=all` as a valid/default “all GPUs” value, so the new condition applies to the image’s normal `docker/run.sh` default path and slows every compile-heavy workload even though the subprocess enumeration bug is specific to selected-device containers.

Suggested fix:
```python
visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
is_single_selected_device = (
    visible_devices not in (None, "", "void", "none", "all")
    and "," not in visible_devices
)

if (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and is_single_selected_device
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if compile_threads in (None, "", "1"):
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

**[P2] `.github/workflows/docker-publish.yml:84`** -- The disk-reclaim step tries to delete `$AGENT_TOOLSDIRECTORY`, but GitHub-hosted runners expose the hosted tool cache as `RUNNER_TOOL_CACHE`; `AGENT_TOOLSDIRECTORY` is not the documented default variable. Triggers on the hosted build jobs where the CUDA base image plus cu128 PyTorch wheels need the reclaimed space: this line silently expands to an empty string and leaves the tool cache in place, making first-run image builds more likely to fail on disk.

Suggested fix:
```yaml
      - name: Reclaim disk
        run: |
          for path in \
              /usr/share/dotnet \
              /usr/local/lib/android \
              /opt/ghc \
              /opt/hostedtoolcache/CodeQL \
              "${RUNNER_TOOL_CACHE:-}"; do
            if [ -n "$path" ]; then
              sudo rm -rf "$path" || true
            fi
          done
          df -h /
```

**[P2] `docker/Dockerfile:76`** -- `TORCH_CUDA_ARCH_LIST` includes `12.1+PTX` while the builder is CUDA 12.8. PyTorch turns that into `compute_121`/`sm_121` flags, but NVIDIA’s CUDA 12.8 release notes list compiler support for `SM_100`, `SM_101`, and `SM_120`, not `SM_121`. Triggers when any dependency or user-installed CUDA extension actually source-builds under the CUDA 12.8 builder/runtime path; nvcc will receive an unsupported Blackwell arch even though the comment says source builds are covered.

Suggested fix:
```dockerfile
# CUDA 12.8 supports up through sm_120. Keep sm_121 out of the common
# arch list; GB10 can run sm_120/PTX through the runtime cu13 workaround.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX" \
    MAX_JOBS=4 \
    CUDA_HOME=/usr/local/cuda \
    UNSLOTH_COMPILE_DISABLE=1 \
    UNSLOTH_COMPILE_OVERWRITE=0 \
    UNSLOTH_DISABLE_GPU_PROBE=1 \
    CUDA_VISIBLE_DEVICES=""
```

Apply the same `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0+PTX"` change to the runtime-stage `ENV` at `docker/Dockerfile:314`.

**Test Results**

I read the full PR diff and inspected the post-merge tree directly. `revert_report.json` originally listed stale-base reverts, but `auto_fix.applied` is true and the post-fix report is clean; `lint_delta.json` reports no new Ruff errors.

I ran shell syntax validation for the added scripts:
```bash
bash -n unsloth/docker/*.sh
```
Result: passed.

I simulated the new `_gpu_init.py` environment transitions against Zoo’s existing non-debug pop behavior. The important result:
```text
auto device=N => TORCHINDUCTOR_COMPILE_THREADS=1, sentinel=1
user already set threads=1 => TORCHINDUCTOR_COMPILE_THREADS=None, sentinel=None
all gpus => TORCHINDUCTOR_COMPILE_THREADS=1, sentinel=1
```
That confirms both the asymmetric manual-workaround hole and the over-broad `all` case.

I ran a `uv pip install --dry-run` against the live PyTorch cu128 index for the pinned torch/vision/audio set. It resolves `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, and `torchaudio==2.11.0+cu128`; I did not flag that as a resolver bug.

I checked PyTorch’s generated CUDA arch flags locally:
```text
TORCH_CUDA_ARCH_LIST=12.1+PTX -> -gencode=arch=compute_121,... -gencode=...,code=sm_121
```
Combined with NVIDIA CUDA 12.8 release notes, this confirms the Dockerfile’s common CUDA 12.8 source-build arch list is too new.

I could not run the full Docker build or smoke test in this worker because the local Docker daemon socket is not accessible to the current user, and this environment does not expose a GPU. Attempting a minimal Docker build check failed with Docker socket permission denied before parsing/build execution.

Live references checked:
- NVIDIA Container Toolkit docs for `NVIDIA_VISIBLE_DEVICES=all`, selected device lists, `none`, and `void`: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.6/docker-specialized.html
- GitHub Actions variable docs for `RUNNER_TOOL_CACHE`: https://docs.github.com/en/actions/reference/workflows-and-actions/variables
- NVIDIA CUDA 12.8 release notes listing compiler support for `SM_100`, `SM_101`, and `SM_120`: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html

**Verdict**

REQUEST_CHANGES. The Docker build machinery is close, but the Inductor workaround has a real asymmetric-fix regression that leaves a documented/manual workaround path broken, and the Docker/CI defaults include correctness issues that will either slow normal container runs or make hosted builds/source-builds fail in realistic scenarios.
