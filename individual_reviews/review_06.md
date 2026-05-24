# Review 6/12 (rc=0)

Operating as dataflow persona.

**Summary**

This PR adds a multi-arch Blackwell-oriented Docker image, publishing workflow, local Docker helper scripts, a container smoke test, and two runtime fixes in Unsloth: a Docker GPU visibility workaround in `_gpu_init.py` and a Transformers v5 `logits_to_keep` change for VLM generation. The merged workspace has the auto-fix for the stale-branch accidental reverts applied, so I reviewed the post-fix tree; the raw branch did contain high-severity reverts, but `post_fix_report` is clean.

**Findings**

**[P1] `docker/Dockerfile:161`** -- The image pins `torch==2.10.0` with `torchaudio==2.11.0`, which is a version-pair mismatch. PyTorch’s published install matrix pairs torch 2.10.0 with torchvision 0.25.0 and torchaudio 2.10.0 for cu128, while torchaudio 2.11.0 is the matching package for torch 2.11.0. This triggers when any audio/TTS path imports or loads torchaudio native extensions in the built image; the Dockerfile’s build-time verification does not import torchaudio, so the image can publish with the incompatible pair. PyTorch’s previous-version instructions show the correct 2.10.0 cu128 triplet.

Suggested fix:
```dockerfile
        "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
```

**[P1] `docker/entrypoint.sh:117`** -- The entrypoint GPU validation accepts Turing `sm_75`, but the same PR’s smoke test rejects every GPU below Ampere at `docker/smoke_test.py:45`. This is an asymmetric validation bug in the new GPU support guard: a T4/RTX 20-series host passes container startup, then the official smoke test and the PR’s own “pre-Ampere unsupported” contract fail later. The entrypoint comment says the check catches “pre-Ampere GPUs”, but the code only rejects pre-Turing.

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

Also remove `sm_75` from `SUPPORTED` in the same block unless Turing is intentionally supported end-to-end.

**[P1] `unsloth/_gpu_init.py:88`** -- The single-worker sentinel is not set when the user already exported `TORCHINDUCTOR_COMPILE_THREADS=1`. In the Docker `--gpus '"device=N"'` scenario this PR is trying to fix, the new first block skips because the env var exists, then current `unsloth_zoo.patch_torch_compile(debug=False)` pops `TORCHINDUCTOR_COMPILE_THREADS`, and the later reassertion at `unsloth/_gpu_init.py:147` does not run because `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER` was never set. So the explicit user workaround is erased and the original Inductor subprocess-pool path can still hit `Could not find an active GPU backend`.

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

This preserves the opt-out, honors an explicit `TORCHINDUCTOR_COMPILE_THREADS=1`, and avoids silently overriding a user who deliberately set another thread count.

**Cross-block check**

Cross-block check found one asymmetric-fix pattern: the new GPU capability validation in `docker/entrypoint.sh:117` accepts `sm_75`, while the new runtime validation in `docker/smoke_test.py:45` rejects `sm_75`. Both blocks validate the same logical operation, “is this GPU supported by the image?”, but they use different thresholds.

I also checked the new destructive operations and guards: workflow disk cleanup `rm -rf`, Dockerfile cache cleanup `rm -rf`, arm64-only CUDA 13 install/NVRTC swap, `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER` env gate, and `logits_to_keep` stripping. I did not find another same-operation block missing the same protection in the merged tree.

**Test Results**

Ran:
```bash
for f in unsloth/docker/*.sh; do bash -n "$f" || exit 1; done
```
Result: passed.

Ran:
```bash
./.venv/bin/python -m py_compile unsloth/docker/smoke_test.py
```
Result: passed.

Ran YAML parse on `.github/workflows/docker-publish.yml`.
Result: parsed successfully and found jobs `build`, `merge`, `smoke-test`.

Ran an env-state simulation of the `_gpu_init.py` guard plus current `unsloth_zoo.patch_torch_compile` behavior.
Result: `auto_absent` sets the sentinel and reasserts correctly; `explicit_threads_1` does not set the sentinel, so after zoo pops `TORCHINDUCTOR_COMPILE_THREADS`, reassertion is false.

Ran a capability-threshold simulation for entrypoint vs smoke test.
Result: `(7, 5)` is accepted by entrypoint and rejected by smoke test; `(8, 0)` and `(12, 0)` are accepted by both.

Ran:
```bash
uv pip install --dry-run --python ./.venv/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.11.0'
```
Result: uv resolves the set, but this does not prove ABI compatibility; PyTorch’s own published cu128 install command for torch 2.10 uses `torchaudio==2.10.0`, not 2.11.0.

Attempted:
```bash
docker buildx build --check unsloth/docker
```
Result: blocked by local Docker socket permissions (`permission denied` connecting to `/var/run/docker.sock`), so I could not run Dockerfile check/build or the GPU smoke test in this environment.

Live references used:
- GitHub-hosted runner docs confirm `ubuntu-24.04-arm` exists for standard GitHub-hosted runners: https://docs.github.com/actions/reference/runners/github-hosted-runners
- PyTorch previous-version install commands show the correct `torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0` cu128 triplet: https://pytorch.org/get-started/previous-versions/
- Docker metadata-action docs confirm `enable=` tag expressions are supported: https://github.com/docker/metadata-action

**Verdict**

REQUEST_CHANGES. The PR is close structurally, but the merged Docker image still has a package-version mismatch, a contradictory GPU support gate, and a dataflow bug that drops an explicit single-worker Inductor override in the exact Docker GPU visibility scenario this PR is meant to fix.
