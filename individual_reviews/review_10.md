# Review 10/12 (rc=0)

**Summary**

This PR adds a new Docker publishing pipeline and Docker image assets for a CUDA 12.8 / PyTorch 2.10 Unsloth image, plus two runtime compatibility changes: a Docker GPU visibility workaround in `unsloth/_gpu_init.py` and a Transformers v5 VLM `logits_to_keep` adjustment in `unsloth/models/vision.py`. The Docker workflow builds per-arch images, merges them into a multi-arch manifest, and optionally smoke-tests the published image on a self-hosted GPU runner.

**Findings**

**[P1] `.github/workflows/docker-publish.yml:127`** -- Manual dispatch can publish arbitrary baked refs as `latest`. The workflow allows `workflow_dispatch` callers to override `unsloth_ref`, but the merge job still enables the `latest` tag whenever the workflow runs on the default branch. A maintainer testing `workflow_dispatch` with `unsloth_ref=<feature-branch-or-sha>` from `main` will push that non-main source as `docker.io/unsloth/unsloth:latest`, and the smoke test will then validate the same incorrect tag. That makes a test dispatch capable of replacing the public default image.

Suggested fix:

```yaml
      - name: Resolve tags
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest,enable=${{ github.event_name != 'workflow_dispatch' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch) }}
            type=ref,event=tag
            type=schedule,pattern=nightly
            type=sha,prefix=sha-,format=short
```

Apply the same tag policy in the `smoke-test` job’s `Resolve published tag` step so it pulls the same non-`latest` tag set:

```yaml
      - name: Resolve published tag
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest,enable=${{ github.event_name != 'workflow_dispatch' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch) }}
            type=ref,event=tag
            type=schedule,pattern=nightly
            type=sha,prefix=sha-,format=short
```

A stricter alternative is to remove the ref override inputs from the publishing workflow and keep custom-ref image tests in a separate non-publishing workflow.

**[P1] `unsloth/_gpu_init.py:88`** -- The single-worker workaround drops an explicit user-set `TORCHINDUCTOR_COMPILE_THREADS=1`. The new Docker GPU fingerprint sets `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=1` only when `TORCHINDUCTOR_COMPILE_THREADS` is absent. If a user already sets the documented env var to `1` under `docker --gpus '"device=N"'`, this branch does not set the sentinel; then current `unsloth_zoo.patch_torch_compile(debug=False)` still runs `os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)`, and the reassert block at line 147 does not run. The original Inductor subprocess-pool failure therefore remains for the explicit-env path.

Suggested fix:

```python
if (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and "NVIDIA_VISIBLE_DEVICES" in os.environ
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
    if compile_threads in (None, "", "1"):
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

This preserves the opt-out (`UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=0`), keeps explicit `TORCHINDUCTOR_COMPILE_THREADS=1` protected from the zoo pop, and avoids overriding a user who intentionally set a different thread count.

Cross-block check: found an asymmetric-fix pattern. The PR adds a Docker environment-mode gate and reassertion in `unsloth/_gpu_init.py:88` and `unsloth/_gpu_init.py:147`, but the analogous explicit `TORCHINDUCTOR_COMPILE_THREADS=1` path is not given the sentinel needed to survive the existing removal in `unsloth-zoo/unsloth_zoo/patching_utils.py:113`.

**Test Results**

I reviewed the changed files directly from the checked-out PR tree and compared them with `pr_changes.diff`, `integration_diff.diff`, `revert_report.json`, and sibling `unsloth-zoo` sources.

Commands run:

```text
bash -n docker/*.sh docker/entrypoint.sh
python -m py_compile docker/smoke_test.py unsloth/_gpu_init.py unsloth/models/vision.py
uv pip compile --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 torch/torchvision/torchaudio constraints
Python simulation of the new _gpu_init env logic vs current unsloth_zoo.patch_torch_compile env pop
rg cross-block scan for guards, env gates, destructive operations, cleanup, and logits_to_keep paths
```

Results:

```text
Shell syntax checks passed.
Python compile checks passed.
lint_delta.json reports 0 new ruff errors.
Resolver check for the pinned torch/torchvision/torchaudio subset completed successfully.
The env simulation reproduced the asymmetric case:
  auto absent      -> TORCHINDUCTOR_COMPILE_THREADS restored to 1
  explicit threads -> TORCHINDUCTOR_COMPILE_THREADS removed and not restored
```

I could not run the full Docker build or `docker buildx --check` in this environment because the current user cannot access `/var/run/docker.sock`:

```text
permission denied while trying to connect to the Docker daemon socket
```

`revert_report.json` initially listed high-severity accidental reverts, but `auto_fix.applied` is true and the checked-out tree is already merged with `origin/main`; the post-fix report has zero remaining reverts, so I did not raise those as findings.

I also used live web references to sanity-check external assumptions: GitHub’s hosted runner docs list `ubuntu-24.04-arm`, Docker’s metadata-action docs show the `enable={{is_default_branch}}` raw-tag pattern, and NVIDIA’s CUDA 12.8 release notes document the CUDA driver floor. Sources: GitHub hosted runners docs `https://docs.github.com/actions/reference/runners/github-hosted-runners`, Docker metadata-action `https://github.com/docker/metadata-action`, NVIDIA CUDA 12.8 release notes `https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html`.

**Verdict**

REQUEST_CHANGES. The Docker image work is broadly coherent, but the workflow can publish a manually selected source ref as `latest`, and the Inductor single-worker fix has an asymmetric environment path that leaves the exact workaround disabled when the user already set `TORCHINDUCTOR_COMPILE_THREADS=1`. Both are concrete, reproducible issues that should be fixed before merge.
