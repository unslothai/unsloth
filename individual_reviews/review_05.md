# Review 5/12 (rc=0)

Operating as security persona.

**Summary**

This PR adds a new multi-arch Docker image build/publish path for Unsloth, plus helper scripts and smoke tests, and it patches two runtime behaviors: Docker-specific `TORCHINDUCTOR_COMPILE_THREADS=1` handling in `_gpu_init.py`, and Transformers 5 VLM generation kwargs in `vision.py`. The Docker packaging is broad, but the review found two should-fix-before-merge issues in the new runtime helpers and one cross-block guard mismatch.

**Findings**

**[P1] `docker/run.sh:59`** -- Secret values are printed by shell tracing. The script builds `ENV_FORWARD` with `HF_TOKEN`, `WANDB_API_KEY`, and `UNSLOTH_LICENSE`, then enables `set -x` immediately before `docker run`, so every forwarded secret is emitted into the terminal/logs as `-e HF_TOKEN=... -e WANDB_API_KEY=...`. This triggers whenever a user runs `HF_TOKEN=... WANDB_API_KEY=... bash docker/run.sh ...`; the token leak is directly reproducible from the xtrace output. This is a security issue because users often paste these wrapper logs into support tickets or CI logs.

Suggested fix:
```bash
# Forward common secrets only if they're set in the host environment.
# Empty strings would shadow whatever is already inside the image.
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"        ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"   ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)

printf "Running %s with GPUs=%s\n" "$IMAGE" "$GPUS" >&2
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

**[P1] `docker/entrypoint.sh:117`** -- Cross-block check: asymmetric GPU capability guard. The entrypoint comments say the image requires Ampere or newer (`sm_80+`) and `docker/smoke_test.py:45` exits on `cap[0] < 8`, but the entrypoint only rejects `< sm_75` and then allows Turing through with a note. A T4 / RTX 20-series host will pass container startup, then the smoke test and real Unsloth path reject it as pre-Ampere. This is the exact asymmetric-fix pattern: two blocks perform the same support-floor validation with different guards.

Suggested fix:
```python
SUPPORTED = (
    ("sm_80",  "Ampere DC",    "A100, A30"),
    ("sm_86",  "Ampere",       "A40, RTX A6000, RTX 30-series"),
    ("sm_89",  "Ada",          "L4, L40, L40S, RTX 40-series"),
    ("sm_90",  "Hopper",       "H100, H200, GH200"),
    ("sm_100", "Blackwell DC", "B100, B200, GB200"),
    ("sm_103", "Blackwell DC", "B300, GB300"),
    ("sm_120", "Blackwell",    "RTX 50-series, RTX PRO 6000 Blackwell"),
    ("sm_121", "Blackwell",    "GB10 (DGX Spark)"),
)
if major < 8:
    print()
    print(f"ERROR: Unsloth image requires Ampere or newer (sm_80+). Got {name} sm_{major}{minor}.")
    print()
    print("Supported architectures in this image:")
    for arch, fam, ex in SUPPORTED:
        print(f"  {arch:7s} {fam:13s} ({ex})")
    sys.exit(1)
```

**[P2] `docker/run.sh:60`** -- The wrapper documents `UNSLOTH_GPUS=0` and `UNSLOTH_GPUS=0,1`, but passes the value straight to Docker as `--gpus "$GPUS"`. Docker’s GPU selection syntax for specific GPU indices is `--gpus '"device=0,2"'`, while a bare numeric value is a GPU count, not an index list. This breaks the PR’s own targeted `docker --gpus '"device=N"'` scenario when users follow the new wrapper docs; `UNSLOTH_GPUS=0,1 bash docker/run.sh ...` emits `--gpus 0,1`, which Docker does not interpret as “devices 0 and 1”. Docker’s current docs show the `device=` form for specific GPUs.

Suggested fix:
```bash
IMAGE="${UNSLOTH_IMAGE:-unsloth/unsloth:latest}"
GPUS="${UNSLOTH_GPUS:-all}"

case "$GPUS" in
    all|device=*|count=*)
        DOCKER_GPUS="$GPUS"
        ;;
    ''|*[!0-9,]*)
        printf "ERROR: UNSLOTH_GPUS must be 'all', 'device=...', or a comma-separated GPU index list; got '%s'\n" "$GPUS" >&2
        exit 2
        ;;
    *)
        DOCKER_GPUS="device=${GPUS}"
        ;;
esac

exec docker run --rm -it \
    --gpus "$DOCKER_GPUS" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HF_CACHE":/workspace/.cache/huggingface \
    -v "$TRITON_CACHE":/workspace/.cache/triton \
    -v "$WORK_DIR":/workspace/host \
    "${ENV_FORWARD[@]}" \
    "$IMAGE" "$@"
```

**Test Results**

I ran lightweight checks only; I did not run the full Docker build or GPU smoke test because that would require pulling/building a large CUDA image and an attached NVIDIA GPU.

Commands/checks run:

```text
bash -n unsloth/docker/*.sh
.venv/bin/python -m py_compile unsloth/docker/smoke_test.py
```

Both syntax checks passed.

I simulated the `docker/run.sh` secret path with `HF_TOKEN=hf_secret WANDB_API_KEY=wandb_secret UNSLOTH_LICENSE=lic_secret UNSLOTH_GPUS=0,1 ... bash unsloth/docker/run.sh python -V`. The xtrace output printed:

```text
-e HF_TOKEN=hf_secret -e WANDB_API_KEY=wandb_secret -e UNSLOTH_LICENSE=lic_secret
--gpus 0,1
```

That confirms both the secret leak and the malformed specific-GPU selector.

I also simulated the two capability guards with `sm_75`:

```text
entrypoint_allows_sm75= True
smoke_allows_sm75= False
asymmetric= True
```

`revert_report.json` reports `severity=high`, `5` files, `217` reverted lines, but `auto_fix.applied=true`; the local reviewed tree is already merged with `origin/main` and contains the restored `unsloth>=2026.5.7` installer pins and tool XML stripping tests. I did not raise those as findings against the local merged state, but the raw integration diff did contain those stale-branch deletions, so the branch should be updated/rebased before final merge if the hosted PR is not using this merged state.

External check: I used live web search for the current Docker GPU selector syntax; Docker’s GPU access docs show specific GPU indices with `--gpus '"device=0,2"'`, matching the wrapper fix above: https://docs.docker.com/engine/containers/gpu/

**Verdict**

REQUEST_CHANGES.

The Docker work is directionally coherent, but the new helper leaks user secrets, and the GPU capability validation is inconsistent across the entrypoint and smoke test. Those two should be fixed before merge; the wrapper GPU selector bug should be fixed at the same time because it affects the exact single-device Docker workflow this PR is trying to support.
