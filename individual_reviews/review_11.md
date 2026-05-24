# Review 11/12 (rc=0)

**Summary**

This PR adds a Docker-based Blackwell image build/publish pipeline, helper scripts, an entrypoint GPU preflight, a smoke test, and two runtime compatibility tweaks in `_gpu_init.py` and `models/vision.py`. The Docker image path is the main behavioral change: it pins a CUDA 12.8 / torch 2.10 stack, builds multi-arch images, and publishes them through GitHub Actions.

**Findings**

**[P1] `docker/Dockerfile:161`** -- The Docker image pins an incompatible PyTorch audio stack. The Dockerfile installs `torch==2.10.0` with `torchaudio==2.11.0`, but the PyTorch release matrix pairs torch 2.10.0 with torchaudio 2.10.0, and PyTorch’s previous-version install command for 2.10 uses `torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0`. This image will carry a torchaudio binary from the wrong release line; any runtime path that imports torchaudio or uses audio preprocessing inside the container is exposed to ABI/import failures that the current smoke test does not cover.

Suggested fix:
```dockerfile
        "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
```

**[P2] `docker/run.sh:61`** -- The wrapper documents `UNSLOTH_GPUS=0` and `UNSLOTH_GPUS=0,1`, but passes those values directly as `--gpus "$GPUS"`. Docker’s device-selection syntax is `--gpus '"device=0,2"'` / `--gpus device=...`, while bare numeric values are interpreted as a GPU count, not device IDs. This means the documented `UNSLOTH_GPUS=0` path does not select GPU 0 and can fail or expose the wrong set of GPUs.

Suggested fix:
```bash
GPU_ARG="$GPUS"
if [[ "$GPUS" != "all" && "$GPUS" != device=* ]]; then
    GPU_ARG="device=${GPUS}"
fi

exec docker run --rm -it \
    --gpus "$GPU_ARG" \
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

I ran:

```bash
bash -n docker/entrypoint.sh
bash -n docker/build.sh
bash -n docker/freeze.sh
bash -n docker/hf_pull.sh
bash -n docker/hf_push.sh
bash -n docker/run.sh
bash -n docker/setup_qemu.sh
bash -n docker/test_locally.sh
python -m py_compile docker/smoke_test.py unsloth/_gpu_init.py unsloth/models/vision.py
python - <<'PY'
import yaml
yaml.safe_load(open(".github/workflows/docker-publish.yml"))
PY
```

All local syntax checks passed.

I could not run a real Docker build or smoke test because the local user cannot access the Docker daemon socket:

```text
permission denied while trying to connect to the Docker daemon socket
```

I also checked the supplied review artifacts:

```text
lint_delta.json: no new ruff errors
revert_report.json: initial high-severity accidental reverts detected, auto_fix.applied=true, post_fix_report severity=none
```

The initial integration diff showed stale-branch reverts in `install.sh`, `install.ps1`, and Studio tool XML stripping tests, but the provided reviewed tree has already been locally merged with `origin/main` and `post_fix_report` is clean. I did not count those as current findings against the auto-fixed tree.

Cross-block check: no asymmetric-fix patterns detected.

Sources used for version/syntax confirmation:
- PyTorch version matrix shows `torch 2.10.0` pairs with `torchvision 0.25.0` and `torchaudio 2.10.0`: https://github.com/pytorch/pytorch/wiki/PyTorch-Versions
- PyTorch previous-version install command for 2.10 uses `torchaudio==2.10.0`: https://pytorch.org/get-started/previous-versions/
- Docker GPU device selection examples use `--gpus '"device=0,2"'`: https://docs.docker.com/engine/containers/gpu/
- GitHub arm64 runner label `ubuntu-24.04-arm` is valid for public repos: https://docs.github.com/actions/reference/runners/github-hosted-runners

**Verdict**

REQUEST_CHANGES. The Docker image should not ship with a mismatched torch/torchaudio release pair, especially because the smoke test does not import torchaudio and therefore would publish a broken image without detecting it. The `docker/run.sh` GPU selector issue is smaller but should be fixed while touching the Docker support scripts.
