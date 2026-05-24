# Review 9/12 (rc=0)

**Summary**

PR #5748 adds a Docker-based Blackwell/Ampere image build and publish pipeline, including multi-arch GitHub Actions publishing, local Docker helper scripts, runtime GPU preflight checks, and smoke tests. It also patches Unsloth import-time Inductor compile-thread handling for Docker GPU device pinning and changes VLM `generate()` handling so Transformers 5+ owns `logits_to_keep`.

**Findings**

**[P1] `docker/Dockerfile:161`** -- The Docker build pins a mismatched PyTorch audio wheel. `torch==2.10.0` is installed together with `torchaudio==2.11.0`; TorchAudio wheels are built against a specific matching Torch version, so this will either fail the resolver or produce an incompatible stack during the single unified `uv pip install`. This triggers on every Docker build path because the pin is in the base dependency install. PyTorch’s own docs state TorchAudio packages must be paired with the correct PyTorch version, and the published compatibility pattern keeps `torchaudio` aligned to the Torch version.

Suggested fix:
```dockerfile
        "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
```

**[P1] `unsloth/_gpu_init.py:88`** -- The single-worker guard is asymmetric when the user already set `TORCHINDUCTOR_COMPILE_THREADS=1`. The first guard only sets `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=1` when `TORCHINDUCTOR_COMPILE_THREADS` is absent, but the later repair block at lines 147-154 only reasserts the env var when that sentinel exists. With `NVIDIA_VISIBLE_DEVICES` set, `CUDA_VISIBLE_DEVICES` absent, and `TORCHINDUCTOR_COMPILE_THREADS=1` already present, an older `unsloth_zoo.patch_torch_compile` can still pop the env var and this PR will not restore it, reintroducing the exact Docker `--gpus '"device=N"'` Inductor subprocess failure the patch is meant to prevent.

Suggested fix:
```python
_force_single_compile_worker = (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and "NVIDIA_VISIBLE_DEVICES" in os.environ
    and "CUDA_VISIBLE_DEVICES" not in os.environ
)

if _force_single_compile_worker:
    existing_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if existing_threads in (None, "", "1"):
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

**[P1] `docker/run.sh:55`** -- The local run wrapper leaks forwarded secrets into shell traces. Lines 55-57 append `HF_TOKEN`, `WANDB_API_KEY`, and `UNSLOTH_LICENSE` as literal `-e NAME=value` arguments, then line 59 enables `set -x`; running the wrapper with any of those variables set prints the secrets directly in terminal logs before `docker run` executes. This triggers for normal authenticated Hugging Face or W&B runs.

Suggested fix:
```bash
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"        ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"   ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)

exec docker run --rm -it \
    --gpus "$GPUS" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HF_CACHE":/workspace/.cache/huggingface \
    -v "$TRITON_CACHE":/workspace/.cache/triton \
    -v "$WORK_DIR":/workspace/host \
    "${ENV_FORWARD[@]}" \
    "$IMAGE" "$@"
```

**[P2] `docker/entrypoint.sh:117`** -- The runtime preflight contradicts the image’s own support gate and lets Turing GPUs proceed. The header says Unsloth requires `sm_80+`, the Dockerfile’s entrypoint comments say it catches `compute capability >= sm_80`, and `smoke_test.py` exits for `cap[0] < 8`, but the entrypoint only rejects below `sm_75` and then allows T4 / RTX 20-series to run into later failures. This triggers when a user starts the image on a T4 or RTX 20-series host.

Suggested fix:
```python
if major < 8:
    print()
    print(f"ERROR: Unsloth image requires Ampere or newer (sm_80+). Got {name} sm_{major}{minor}.")
    print()
    print("Supported architectures in this image:")
    for arch, fam, ex in SUPPORTED:
        if arch != "sm_75":
            print(f"  {arch:7s} {fam:13s} ({ex})")
    sys.exit(1)
```

**Test Results**

I inspected `pr_changes.diff`, `integration_diff.diff`, `revert_report.json`, `lint_delta.json`, `pr_metadata.json`, and the checked-out post-PR tree. `revert_report.json` showed high-severity accidental reverts in the raw PR integration diff, but `auto_fix.applied` is true and the current review tree includes merge commit `d450c06a`; the post-fix revert report is clean.

I ran:
```bash
for f in unsloth/docker/*.sh; do bash -n "$f" || exit 1; done
.venv/bin/python -m py_compile unsloth/docker/smoke_test.py
git -C unsloth diff --check origin/main...HEAD -- .github/workflows/docker-publish.yml docker unsloth/_gpu_init.py unsloth/models/vision.py
```
All passed.

I simulated the `_gpu_init.py` environment logic and confirmed the asymmetric case: when `NVIDIA_VISIBLE_DEVICES=0` and `TORCHINDUCTOR_COMPILE_THREADS=1` are already set, the PR does not set `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER`, so a zoo-side pop leaves `TORCHINDUCTOR_COMPILE_THREADS` unset.

I simulated `docker/run.sh` with fake secrets and a stubbed `docker` function; the script printed:
```text
-e HF_TOKEN=hf_test_secret -e WANDB_API_KEY=wandb_secret
```
because of `set -x`.

I could not run a real Docker build or container smoke test: Docker is installed, but this environment cannot access the Docker daemon socket (`permission denied ... /var/run/docker.sock`). I also did not have `actionlint` or `shellcheck` available.

External checks used: GitHub’s hosted runner docs confirm the `ubuntu-24.04-arm` label exists, and PyTorch/TorchAudio docs confirm TorchAudio wheels must match the corresponding PyTorch version.

Cross-block check: asymmetric-fix pattern detected in the new Inductor single-worker guard and reported above.

**Verdict**

REQUEST_CHANGES. The Docker image is likely to fail dependency resolution because of the `torchaudio` pin, the run wrapper leaks credentials in a normal authenticated workflow, and the Inductor guard has a real asymmetric case that preserves the old failure when the user has already set the documented workaround env var.
