# Review 2/12 (rc=0)

Operating as dataflow persona.

**Summary**

This PR adds a Docker publishing pipeline and a multi-stage CUDA 12.8 image intended to build without a GPU, plus runtime helper scripts, a smoke test, and two small Unsloth runtime patches around Inductor compile workers and VLM generation kwargs. The Docker/image work is the main surface; the Python changes are compatibility shims for containerized Blackwell validation.

**Findings**

**[P1] `.github/workflows/docker-publish.yml:66`** -- The publish workflow targets standard GitHub-hosted runners, but the Docker build cannot realistically fit on their documented 14 GB disks. This triggers on every `push` to `main`, tag, scheduled run, or manual dispatch: the build starts from `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, whose compressed layers alone are about 5.46 GiB on amd64 and 5.06 GiB on arm64, before Docker unpacks layers, installs Python, PyTorch/cu128 wheels, vLLM, Unsloth, cache metadata, and the runtime stage. GitHub’s current hosted-runner reference lists `ubuntu-latest` and `ubuntu-24.04-arm` with 14 GB SSD storage, so this workflow is set up to fail with disk exhaustion despite the `Reclaim disk` step. Source checked: GitHub runner specs at https://docs.github.com/en/actions/reference/runners/github-hosted-runners.

Suggested fix:

```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - platform: linux/amd64
        runner: [self-hosted, linux, x64, docker-large]
      - platform: linux/arm64
        runner: [self-hosted, linux, arm64, docker-large]
runs-on: ${{ matrix.runner }}
```

If the intent is to keep this on GitHub-hosted runners, the Dockerfile needs to be redesigned around a much smaller builder base and no CUDA devel image, but the current `nvidia/cuda:*cudnn-devel*` approach is not compatible with the documented 14 GB standard runners.

**[P1] `unsloth/_gpu_init.py:88`** -- The single-worker Docker fix drops user-provided `TORCHINDUCTOR_COMPILE_THREADS=1` because the sentinel is only set when that env var is absent. Trigger: run a container with `NVIDIA_VISIBLE_DEVICES` set, no `CUDA_VISIBLE_DEVICES`, and `TORCHINDUCTOR_COMPILE_THREADS=1` already provided by the user or wrapper. The new guard skips setting `UNSLOTH_FORCE_SINGLE_COMPILE_WORKER`, then the current `unsloth_zoo.patch_torch_compile(debug=False)` path pops `TORCHINDUCTOR_COMPILE_THREADS`, and the reassertion block at line 147 never runs. I simulated that dataflow; the final env loses the compile-thread override.

Suggested fix:

```python
_force_single_compile_worker = (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and "NVIDIA_VISIBLE_DEVICES" in os.environ
    and "CUDA_VISIBLE_DEVICES" not in os.environ
)
if _force_single_compile_worker:
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
```

To avoid the open unsloth-zoo dependency still overriding this through compile options, also patch the already-imported zoo helper before any later `get_torch_compile_options()` calls:

```python
if os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "0") == "1":
    try:
        torch._inductor.config.compile_threads = 1
    except Exception:
        pass
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    try:
        import unsloth_zoo.temporary_patches.common as _uz_common
        _uz_common.determine_compile_threads.cache_clear()
        _uz_common.determine_compile_threads = lambda: 1
        if hasattr(_uz_common, "torch_compile_options"):
            _uz_common.torch_compile_options["compile_threads"] = 1
    except Exception:
        pass
```

**[P1] `docker/entrypoint.sh:117`** -- Cross-block check found an asymmetric compute-capability validation: the entrypoint accepts Turing `sm_75`, while the smoke test rejects every pre-Ampere GPU at `docker/smoke_test.py:45`. Trigger: a self-hosted GPU runner or user host with a T4/RTX 20-series GPU. The container preflight passes, but the PR’s own smoke test fails with `FAIL: pre-Ampere GPU ... is not supported by this image`. The PR metadata says the image supports Ampere through Blackwell (`sm_80` through `sm_120`), so the entrypoint and arch-list comments should enforce the same boundary as the smoke test.

Suggested fix:

```bash
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

And align `docker/Dockerfile:76`:

```dockerfile
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX" \
```

**Test Results**

I ran static parsing and focused simulations from the provided workspace:

```text
python ast parse: docker/smoke_test.py OK
bash -n: docker/*.sh OK
ruff delta: no new ruff errors per lint_delta.json
uv resolver dry-run: torch==2.10.0, torchvision==0.25.0, torchaudio==2.11.0 resolve against cu128
uv resolver dry-run: vLLM nightly resolves with torch pinned to 2.10.0+cu128
docker manifest inspect: nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 is ~5.46 GiB compressed on amd64, ~5.06 GiB compressed on arm64
env simulation: user-provided TORCHINDUCTOR_COMPILE_THREADS=1 is lost under the new sentinel logic plus current unsloth-zoo pop behavior
```

I could not run a full Docker build or `docker buildx --check` because this environment cannot connect to the Docker daemon socket. I also could not run the actual GPU smoke test because no accessible Docker GPU runtime was available here. The runner-storage finding was verified against live GitHub-hosted runner documentation and the NVIDIA CUDA image manifest.

Cross-block check: detected one asymmetric-fix pattern, the entrypoint/smoke-test compute capability mismatch above. I also checked analogous env/compile-thread guards and destructive cleanup blocks; the compile-thread path has the sentinel propagation bug described above, and the other cleanup blocks did not show an additional asymmetric ownership/path guard issue.

**Verdict**

REQUEST_CHANGES. The workflow is likely to fail before publishing on the documented standard runners, and the container/runtime fixes have two concrete dataflow mismatches: the compile-thread sentinel can be lost, and the compute-capability gates disagree across entrypoint and smoke test.
