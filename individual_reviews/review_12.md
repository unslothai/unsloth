# Review 12/12 (rc=0)

**Summary**

This PR adds a new Docker publishing pipeline and a multi-stage CUDA 12.8 Blackwell image, plus helper scripts for local build/run/freeze/HF tarball transfer. It also changes Unsloth runtime behavior in two places: `_gpu_init.py` now tries to force a single Inductor compile worker for Docker `--gpus "device=N"` containers, and `vision.py` stops injecting `logits_to_keep` on transformers 5.x VLM generation.

**Findings**

**[P1] `.github/workflows/docker-publish.yml:118`** -- The `build-args` block includes comment lines that are passed to `docker buildx` as build arguments. `docker/build-push-action@v6` treats `build-args` as a raw list and appends each item as `--build-arg`; it does not enable comment parsing for this input. On every CI build, the five `# ...` lines at 122-126 become invalid build arg keys, so the publish workflow can fail before the Dockerfile starts building.

Suggested fix:

```yaml
          # Workflow-dispatch: honour the explicit input. Tag pushes bake the
          # tag's source ref (for example v1.2.3) so the published tag image
          # contains that release. Branch pushes and scheduled runs bake the
          # triggering commit SHA.
          build-args: |
            CUDA_VERSION=12.8.1
            UBUNTU_VERSION=24.04
            PYTHON_VERSION=3.12
            UNSLOTH_REF=${{ github.event.inputs.unsloth_ref || (startsWith(github.ref, 'refs/tags/') && github.ref_name) || github.sha || 'main' }}
            UNSLOTH_ZOO_REF=${{ github.event.inputs.unsloth_zoo_ref || 'main' }}
```

**[P1] `unsloth/_gpu_init.py:88`** -- The single-worker Docker fix is asymmetric for users who already set `TORCHINDUCTOR_COMPILE_THREADS=1`. When the container has `NVIDIA_VISIBLE_DEVICES` but no `CUDA_VISIBLE_DEVICES`, the new guard only creates `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=1` if `TORCHINDUCTOR_COMPILE_THREADS` is absent. If the user already set the correct value, older `unsloth_zoo.patch_torch_compile` can still pop `TORCHINDUCTOR_COMPILE_THREADS`, and the reassertion block at line 147 will not restore it because the sentinel was never set. This reintroduces the exact Docker pinned-GPU Inductor worker failure for the “explicit env var already set” case.

Suggested fix:

```python
_force_single_compile_worker = (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and "NVIDIA_VISIBLE_DEVICES" in os.environ
    and "CUDA_VISIBLE_DEVICES" not in os.environ
)

if _force_single_compile_worker:
    compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if compile_threads in (None, "", "1"):
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

This keeps the existing opt-out behavior, does not override users who intentionally set another thread count, and makes the later reassertion path cover both auto-forced and user-pre-forced `1`.

**[P2] `docker/run.sh:28`** -- The wrapper documents `UNSLOTH_GPUS=0` and `UNSLOTH_GPUS=0,1`, but passes the value directly as `--gpus "$GPUS"`. Docker’s documented syntax for selecting GPU IDs is `--gpus device=0` or `--gpus '"device=0,2"'`; a bare `0`/`0,1` is not the advertised device-selection form. Users following the comment to pin one GPU can fail to attach the intended GPU, which then trips the entrypoint’s “No GPU visible” path.

Suggested fix:

```bash
IMAGE="${UNSLOTH_IMAGE:-unsloth/unsloth:latest}"
GPUS="${UNSLOTH_GPUS:-all}"
if [[ "$GPUS" != "all" && "$GPUS" != device=* && "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    GPUS="device=${GPUS}"
fi
```

Optionally update the comment to say `UNSLOTH_GPUS=all | device=0 | device=0,1`, while still accepting the convenient short form above.

**Test Results**

I reviewed the full PR diff and inspected the checked-out post-PR files directly.

Ran:

```bash
python - <<'PY'
import json
# inspected pr_metadata.json, pr_diff.json, revert_report.json, lint_delta.json
PY
```

Result: parsed successfully. `lint_delta.json` reports `pre_count=0`, `post_count=0`, `new_count=0`. `revert_report.json` initially listed stale-main reverts, but `auto_fix.applied` is true and `post_fix_report` is clean.

Ran:

```bash
bash -n unsloth/docker/*.sh
python -m py_compile unsloth/docker/smoke_test.py
```

Result: passed.

Ran:

```bash
uv pip install --dry-run --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.10.0' 'torchaudio==2.11.0'
```

Result: resolver accepted the torch/torchaudio pins and selected `torch==2.10.0+cu128`, `torchaudio==2.11.0+cu128`, `triton==3.6.0`.

Ran a targeted simulation of the `_gpu_init.py` env logic with an older zoo-style `TORCHINDUCTOR_COMPILE_THREADS` pop:

```text
auto unset -> {'NVIDIA_VISIBLE_DEVICES': '0', 'UNSLOTH_FORCE_SINGLE_COMPILE_WORKER': '1', 'TORCHINDUCTOR_COMPILE_THREADS': '1'}
explicit threads=1 -> {'NVIDIA_VISIBLE_DEVICES': '0'}
opt out -> {'NVIDIA_VISIBLE_DEVICES': '0', 'UNSLOTH_FORCE_SINGLE_COMPILE_WORKER': '0'}
```

This reproduces the asymmetric env-var case in the second finding.

I could not run the full Docker build or GPU smoke test in this environment because the local Docker daemon socket is not accessible to this user (`permission denied` on `/var/run/docker.sock`), and there is no attached GPU validation path available here.

Cross-block check: asymmetric-fix pattern detected in `_gpu_init.py` for the new Docker pinned-GPU single-worker guard versus the later reassertion path when `TORCHINDUCTOR_COMPILE_THREADS=1` was already present.

**Verdict**

REQUEST_CHANGES.

The Docker image/publish work is directionally coherent, but the workflow currently risks failing before build due to comments inside `build-args`, and the `_gpu_init.py` compatibility guard misses a realistic explicit-env case for the exact Inductor worker issue it is trying to harden. The `docker/run.sh` GPU selector issue is smaller, but it should be fixed because it contradicts the wrapper’s documented interface.

Sources checked during review: GitHub hosted runner labels confirm `ubuntu-24.04-arm` exists in current GitHub-hosted runner docs, Docker docs show device selection syntax as `--gpus device=0`, and `docker/build-push-action@v6` source shows `build-args` are read with `Util.getInputList(..., {ignoreComma: true})` and then passed directly as `--build-arg`.  
Links: https://docs.github.com/actions/reference/runners/github-hosted-runners, https://docs.docker.com/engine/containers/gpu/, https://raw.githubusercontent.com/docker/build-push-action/v6/src/context.ts, https://raw.githubusercontent.com/docker/actions-toolkit/master/src/util.ts
