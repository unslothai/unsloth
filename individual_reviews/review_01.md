# Review 1/12 (rc=0)

Operating as security persona.

**Summary**

This PR adds a Blackwell-oriented Docker image, publication workflow, helper scripts, a Docker/Inductor compile-thread workaround in `unsloth/_gpu_init.py`, and a Transformers v5 generation compatibility tweak for VLMs. The main Docker path is structurally reasonable, but I found one mandatory asymmetric-fix bug in the compile-thread workaround, plus a concrete secret leak in the local Docker wrapper.

**Findings**

**[P1] [unsloth/_gpu_init.py:147](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_yug4g249/unsloth/unsloth/_gpu_init.py:147)** -- The single-worker Docker workaround is undone later in the same import path. The PR sets and reasserts `TORCHINDUCTOR_COMPILE_THREADS=1` before `_gpu_init.py` imports `.models`, but `.models.__init__` imports `._utils`, and `_utils.py:1513` calls `patch_torch_compile(debug=False)`. In the current/older zoo implementation, `unsloth_zoo/patching_utils.py:113` still does `os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)`. That means the exact `docker --gpus '"device=N"'` case this PR is trying to fix can still finish `import unsloth` with the env var removed and Inductor compile workers enabled. This is the required cross-block asymmetric-fix pattern: the new guard/reassert exists in one block, but the analogous env-removal block still runs afterward without honoring the same sentinel.

Suggested fix:
```python
# after importing torch in unsloth/_gpu_init.py
def _reassert_single_compile_worker_if_forced():
    if os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "0") != "1":
        return
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    try:
        torch._inductor.config.compile_threads = 1
    except Exception:
        pass

_reassert_single_compile_worker_if_forced()
```

Then call it again after the model imports that trigger `patch_torch_compile`:
```python
from .models import *
_reassert_single_compile_worker_if_forced()
from .models import __version__
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *
```

The paired zoo-side fix should also guard the pop:
```python
if os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "0") != "1":
    os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
```

**[P1] [docker/run.sh:55](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_yug4g249/unsloth/docker/run.sh:55)** -- `set -x` prints forwarded secrets in full. When `HF_TOKEN`, `WANDB_API_KEY`, or `UNSLOTH_LICENSE` is set, the wrapper builds `-e "HF_TOKEN=${HF_TOKEN}"` style arguments and then enables shell tracing before `exec docker run`. I reproduced this with a stubbed `docker`; stderr contained the full token values. This leaks credentials into terminal scrollback and CI logs whenever users run the documented helper with tracing enabled by default.

Suggested fix:
```bash
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"        ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"   ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)

if [[ "${UNSLOTH_DOCKER_TRACE:-0}" == "1" ]]; then
    set -x
fi
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

**[P2] [docker/Dockerfile.studio:42](/mnt/disks/unslothai/ubuntu/workspace_0/unsloth_src/temp/temporary_yug4g249/unsloth/docker/Dockerfile.studio:42)** -- The Studio image is not reproducible and can install code from a different Unsloth revision than the base image. `Dockerfile.studio` accepts only `BASE_TAG`, then clones the default branch of `https://github.com/unslothai/unsloth`. Trigger: build `unsloth-blackwell:studio` on top of a base image pinned to a release tag, PR SHA, or historical digest after `main` has moved. The Studio venv will contain current `main`, while the base image contains the pinned Python package stack, so the container can run a CLI/backend revision that was never validated with that base.

Suggested fix:
```dockerfile
ARG BASE_TAG=test
ARG UNSLOTH_REF=main
FROM unsloth-blackwell:${BASE_TAG}

USER root
ENV UNSLOTH_STUDIO_HOME=/opt/unsloth-studio \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${UNSLOTH_STUDIO_HOME}/src" \
 && git init "${UNSLOTH_STUDIO_HOME}/src" \
 && cd "${UNSLOTH_STUDIO_HOME}/src" \
 && git remote add origin https://github.com/unslothai/unsloth \
 && git fetch --depth 1 origin "${UNSLOTH_REF}" \
 && git checkout --detach FETCH_HEAD \
 && UNSLOTH_STUDIO_HOME="${UNSLOTH_STUDIO_HOME}" bash install.sh --local \
 && rm -rf "${UNSLOTH_STUDIO_HOME}/src/.git" /root/.cache
```

**Test Results**

I ran:

```text
bash -n unsloth/docker/*.sh
python -m py_compile unsloth/docker/smoke_test.py
PyYAML parse of .github/workflows/docker-publish.yml
uv dry-run for the torch/torchvision/torchaudio/triton/bitsandbytes pins
uv dry-run for the vLLM nightly install path
stubbed execution of docker/run.sh with fake HF_TOKEN/WANDB_API_KEY/UNSLOTH_LICENSE
rg-based cross-block check for guards, destructive operations, env gates, and analogous unguarded paths
```

Results:

```text
Shell syntax: passed
smoke_test.py syntax: passed
workflow YAML parse: passed
lint_delta.json: 0 new ruff errors
uv dry-runs: resolver completed for the checked package subsets
docker/run.sh secret simulation: failed as expected; all three fake secrets appeared in stderr
Docker buildx --check: not run successfully because this environment cannot access the Docker daemon socket
GPU/runtime smoke test: not run; no usable Docker daemon/GPU access from this review environment
```

`revert_report.json` reports high reverts, but `auto_fix.applied` is `true` and the checked-out tree includes the merge commit, so I reviewed the merged local state rather than flagging the pre-auto-fix revert report as blocking.

Cross-block check: asymmetric-fix pattern detected in the `TORCHINDUCTOR_COMPILE_THREADS` handling described in the first finding.

External reference checked: the official `docker/build-push-action` README documents that `build-args` is a newline-delimited list input, matching the workflow’s intended action input shape: https://github.com/docker/build-push-action.

**Verdict**

REQUEST_CHANGES. The Docker publication scaffolding is close, but the compile-thread fix is currently undone by a later import-time patch, so the PR does not reliably fix the Docker `device=N` Inductor failure it claims to address. The helper script also leaks user secrets by default, which should be fixed before merge.
