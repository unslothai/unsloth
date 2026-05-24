# Review 4/12 (rc=0)

Operating as simulation persona.

**Summary**
This PR adds a multi-arch Blackwell CUDA Docker image, publishing workflow, helper scripts, runtime GPU preflight checks, and two Unsloth runtime patches: one for Docker GPU visibility/Inductor compile workers and one for Transformers 5 VLM generation kwargs. The local checkout has already been auto-merged with `origin/main`; the stale-rebase deletions reported in `revert_report.json` are resolved in the reviewed tree (`post_fix_report.severity=none`).

**Findings**

**[P1] [docker/test_locally.sh](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_cgf4ei6e/unsloth/docker/test_locally.sh:370)** -- The HF token forwarding is an asymmetric fix and still passes secrets as command-line values. `docker/run.sh` adds a non-empty guard for forwarded secrets at [docker/run.sh](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_cgf4ei6e/unsloth/docker/run.sh:55), but the analogous notebook validation path unconditionally passes `-e HF_TOKEN="${HF_TOKEN:-}"`. This triggers in two concrete cases: when `HF_TOKEN` is unset, the script injects an empty `HF_TOKEN=` and can shadow a token already configured in the container environment; when it is set, the value is exposed in the host process arguments. `docker/run.sh` also exposes all three secret values in `set -x` output. Repro with fake secrets:
```text
HF_TOKEN=hf_fake_token WANDB_API_KEY=wandb_fake_key UNSLOTH_LICENSE=lic_fake UNSLOTH_IMAGE=example/image:latest bash unsloth/docker/run.sh true
+ exec docker run ... -e HF_TOKEN=hf_fake_token -e WANDB_API_KEY=wandb_fake_key -e UNSLOTH_LICENSE=lic_fake ...
```
Suggested fix:
```bash
# docker/run.sh
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"        ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"   ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)

# Do not enable xtrace around secrets.
exec docker run --rm "${TTY_ARGS[@]}" \
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
```bash
# docker/test_locally.sh, before the notebook docker run
declare -a NOTEBOOK_ENV=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}" ]] && NOTEBOOK_ENV+=(-e HF_TOKEN)

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HOST_RUN_DIR:/workspace/host" \
    -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
    "${NOTEBOOK_ENV[@]}" \
    "$TAG" \
    bash /workspace/host/run_notebook.sh 2>&1 | tee "$GPT_LOG"
```

**[P2] [docker/run.sh](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_cgf4ei6e/unsloth/docker/run.sh:60)** -- The wrapper always uses `-it`, so the documented non-interactive usage fails before the command runs when invoked from CI, logs, or any non-TTY context. I reproduced this with `bash unsloth/docker/run.sh true`; Docker exits with `the input device is not a TTY`. This affects the script’s own examples such as `bash docker/run.sh python /workspace/smoke_test.py` when run from automation.
Suggested fix:
```bash
TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_ARGS=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" \
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

**[P2] [docker/entrypoint.sh](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_cgf4ei6e/unsloth/docker/entrypoint.sh:117)** -- The preflight check allows Turing `sm_75` even though the entrypoint header says it catches GPUs older than Ampere and `smoke_test.py` rejects anything below Ampere at [docker/smoke_test.py](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_cgf4ei6e/unsloth/docker/smoke_test.py:45). A T4 host therefore passes container startup and only fails later in the smoke test or user workload. I executed the same condition with mocked capabilities and got `Fake sm_75: allowed with NOTE path`, `Fake sm_70: rejected`, `Fake sm_80: allowed`.
Suggested fix:
```python
# entrypoint.py heredoc replacement logic inside docker/entrypoint.sh
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
Also remove `sm_75` from the `SUPPORTED` tuple and from `TORCH_CUDA_ARCH_LIST` unless Turing is intentionally supported end-to-end, in which case `smoke_test.py` should be relaxed instead.

**[P3] [docker/Dockerfile.studio](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_cgf4ei6e/unsloth/docker/Dockerfile.studio:42)** -- The Studio image always clones `unsloth` from default `main`, even when the base image was built from a tag, SHA, or PR ref. This makes `unsloth-blackwell:<tag>` plus `Dockerfile.studio` non-reproducible and can install Studio code that does not match the Python package baked into the base layer.
Suggested fix:
```dockerfile
ARG UNSLOTH_REF=main

RUN mkdir -p "${UNSLOTH_STUDIO_HOME}/src" \
 && git init "${UNSLOTH_STUDIO_HOME}/src" \
 && cd "${UNSLOTH_STUDIO_HOME}/src" \
 && git remote add origin https://github.com/unslothai/unsloth \
 && git fetch --depth 1 origin "${UNSLOTH_REF}" \
 && git checkout --detach FETCH_HEAD \
 && UNSLOTH_STUDIO_HOME="${UNSLOTH_STUDIO_HOME}" bash install.sh --local \
 && rm -rf "${UNSLOTH_STUDIO_HOME}/src/.git" /root/.cache
```

**Cross-Block Check**
I enumerated the modified guards/destructive operations/env gates in `pr_changes.diff`: `rm -rf` cleanup blocks, `TARGETARCH`/`INSTALL_VLLM` gates, `NVIDIA_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`/`TORCHINDUCTOR_COMPILE_THREADS` guards, `UNSLOTH_SKIP_GPU_CHECK`, `HAS_GPU_RUNNER`, and `logits_to_keep`/`num_logits_to_keep` validation. The asymmetric-fix pattern I found is the guarded secret forwarding in `docker/run.sh` versus the unguarded `HF_TOKEN` forwarding in `docker/test_locally.sh`, reported above as [P1]. I did not find another same-operation block missing the newly introduced GPU/compile/logits guards.

**Test Results**
Ran:
```bash
bash -n unsloth/docker/*.sh
.venv/bin/python -m py_compile unsloth/docker/smoke_test.py
.venv/bin/python - <<'PY'
import yaml
yaml.safe_load(open("unsloth/.github/workflows/docker-publish.yml"))
print("yaml ok")
PY
```
Result: shell syntax, Python compile, and YAML parse passed.

Ran resolver simulations:
```bash
uv pip compile temp/docker-amd64-local.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
uv pip compile temp/docker-arm64-local.in --python-version 3.12 --python-platform aarch64-manylinux_2_28 --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
```
Result: both resolved successfully with `torch==2.10.0+cu128`, `triton==3.6.0`, `transformers==5.5.0`; AMD64 included xformers, arm64 omitted it as intended.

Ran `actionlint` v1.7.12 downloaded from the upstream GitHub release page (`https://github.com/rhysd/actionlint/releases`):
```bash
temp/actionlint/actionlint unsloth/.github/workflows/docker-publish.yml
```
Result: only reported `runs-on: [self-hosted, gpu]` as an unknown custom self-hosted label. That is not a runtime bug if the repo actually registers a runner with label `gpu`; add an actionlint config if this workflow will be linted in CI.

Ran Hugging Face model metadata check:
```python
from huggingface_hub import model_info
model_info("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
```
Result: `private=False`, `gated=False`, so the smoke model itself does not require a token.

Could not run a real Docker build or `docker buildx --check` because the current user cannot access `/var/run/docker.sock`:
```text
permission denied while trying to connect to the Docker daemon socket
```
Monkey patching was not applicable: the reproduced failures are shell/CI wrapper defects, not Unsloth model-runtime failures.

**Verdict**
REQUEST_CHANGES. The core Docker dependency resolution looks plausible from resolver simulation, but the helper scripts need fixes before merge: one asymmetric env-forwarding bug leaks/shadows tokens, the wrapper fails in non-TTY contexts, and the GPU support gate contradicts the stated Ampere+ requirement and the smoke test.
