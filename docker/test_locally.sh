#!/usr/bin/env bash
# End-to-end Docker validation for the unsloth-blackwell image.
#
# Runs three blocks:
#   1. Host pre-flight (docker, nvidia-smi, nvidia runtime registered)
#   2. Build the image (no GPU required at build time)
#   3a. Smoke test: 5-step LoRA on Llama-3.2-1B (~1-2 min)
#   3b. Real workload: gpt-oss-20B fine-tuning notebook with max_steps=10
#       (~10 min, needs ~30GB free for the model cache)
#
# Usage:
#   bash docker/test_locally.sh                   # all blocks (native arch)
#   bash docker/test_locally.sh --skip-notebook   # blocks 1-3a only (fast)
#   bash docker/test_locally.sh --skip-build      # assume $TAG already built
#   bash docker/test_locally.sh --platform arm64  # cross-build for DGX Spark
#                                                 # (auto-skips smoke/notebook)
#   TAG=my-image:latest bash docker/test_locally.sh
#   HF_TOKEN=hf_xxx bash docker/test_locally.sh   # for gated models (optional)
#
# All output is teed to $LOG_DIR (default /tmp/unsloth-docker-test/).
# Paste the listed log snippets back if anything fails.
set -uo pipefail

TAG="${TAG:-unsloth-blackwell:test}"
LOG_DIR="${LOG_DIR:-/tmp/unsloth-docker-test}"
SKIP_BUILD=0
SKIP_NOTEBOOK=0
# Platform selector. Empty = let buildx default to the host arch (no
# --platform passed). "amd64" / "arm64" = single-arch cross-build via QEMU
# (requires `bash docker/setup_qemu.sh` to have been run once).
PLATFORM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)    SKIP_BUILD=1; shift ;;
        --skip-notebook) SKIP_NOTEBOOK=1; shift ;;
        --tag)           TAG="$2"; shift 2 ;;
        --log-dir)       LOG_DIR="$2"; shift 2 ;;
        --platform)
            case "$2" in
                amd64|arm64|linux/amd64|linux/arm64) PLATFORM="${2#linux/}" ;;
                *) echo "ERROR: --platform must be amd64 or arm64 (got '$2')" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --help|-h)       sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

# When cross-building, the resulting image cannot be exercised on this host
# (CUDA does not work under QEMU runtime emulation). Auto-skip the GPU blocks
# and warn the user. They can paste back the build log either way to prove
# the wheels resolve + the build-time torch._C._cuda_getArchFlags() assertion
# passes on the foreign arch.
HOST_ARCH="$(uname -m)"
case "${HOST_ARCH}" in
    x86_64|amd64) HOST_DOCKER_ARCH="amd64" ;;
    aarch64|arm64) HOST_DOCKER_ARCH="arm64" ;;
    *)            HOST_DOCKER_ARCH="${HOST_ARCH}" ;;
esac
CROSS_ARCH=0
if [[ -n "${PLATFORM}" && "${PLATFORM}" != "${HOST_DOCKER_ARCH}" ]]; then
    CROSS_ARCH=1
fi

mkdir -p "$LOG_DIR"

GREEN='\033[1;32m'; RED='\033[1;31m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'
banner() { printf "\n${BLUE}==== %s ====${NC}\n" "$*"; }
ok()     { printf "${GREEN}OK${NC}    %s\n" "$*"; }
warn()   { printf "${YELLOW}WARN${NC}  %s\n" "$*"; }
err()    { printf "${RED}ERROR${NC} %s\n" "$*" >&2; }
fail()   { err "$*"; exit 1; }

# ============================================================================
# Block 1: pre-flight
# ============================================================================
banner "Block 1: host pre-flight"

command -v docker >/dev/null 2>&1 || fail "docker not found on PATH"
echo "  docker:       $(docker --version)"

# Verify we can actually talk to the docker daemon as the current user.
# This catches the "user not in docker group" case up front, instead of
# letting docker buildx blow up with a "permission denied on /var/run/docker.sock"
# error that looks like a build failure but is really a host permissions issue.
DOCKER_INFO_OUT=$(docker info 2>&1)
DOCKER_INFO_RC=$?
if [[ $DOCKER_INFO_RC -ne 0 ]]; then
    err "Cannot talk to the docker daemon as user '$USER'."
    cat >&2 <<MSG

docker info exited $DOCKER_INFO_RC. The most common cause is that your user
is not in the 'docker' group. Fix:

  sudo usermod -aG docker \$USER
  newgrp docker            # activate the new group in this shell
  docker info | head -3    # verify

Then re-run this script in the same shell (or any new login session).

Alternative: run the script with sudo, but be aware it will use root's
home directory for HF cache (~/root/.cache/huggingface) which is probably
not what you want.

Raw docker info output:
$DOCKER_INFO_OUT
MSG
    fail "docker daemon unreachable"
fi
echo "  daemon:       reachable as '$USER'"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  host gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  host driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not on the host -- you may not be able to run --gpus all"
fi

# This grep only makes sense once we know `docker info` succeeded above.
if echo "$DOCKER_INFO_OUT" | grep -qiE 'Runtimes:.*nvidia'; then
    echo "  nvidia runtime: registered with docker"
else
    warn "docker info does not list 'nvidia' as a runtime"
    warn "(on Docker 28+ with CDI this is often a false positive; the real"
    warn " test is whether --gpus all works in Block 3a below)"
    warn "if --gpus all fails, install nvidia-container-toolkit:"
    warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    warn "  then: sudo systemctl restart docker"
fi
ok "pre-flight done"

# ============================================================================
# Block 2: build
# ============================================================================
if [[ $SKIP_BUILD -eq 1 ]]; then
    warn "skipping build (--skip-build); expecting $TAG to exist"
else
    banner "Block 2: build $TAG"

    # Find the build context: current dir, docker/ subdir, or clone the PR branch
    if [[ -f "Dockerfile" && -f "smoke_test.py" ]]; then
        BUILD_CTX="$PWD"
    elif [[ -f "docker/Dockerfile" ]]; then
        BUILD_CTX="$PWD/docker"
    else
        BUILD_CTX="/tmp/unsloth-pr/docker"
        if [[ ! -d /tmp/unsloth-pr/.git ]]; then
            echo "  cloning docker-blackwell-build branch..."
            if ! git clone --depth 1 -b docker-blackwell-build \
                    https://github.com/unslothai/unsloth.git /tmp/unsloth-pr 2>&1 | tail -3; then
                fail "could not clone docker-blackwell-build into /tmp/unsloth-pr; refusing to build from stale context"
            fi
        else
            # `set -e` is not active in this script, so a failing pull would
            # otherwise be silently masked and we'd build from a stale clone.
            # Explicitly fail loudly when the fast-forward refresh cannot run.
            if ! git -C /tmp/unsloth-pr pull --ff-only 2>&1 | tail -2; then
                fail "git pull --ff-only failed in /tmp/unsloth-pr; refusing to build from stale context (delete /tmp/unsloth-pr to reclone)"
            fi
        fi
    fi
    echo "  build context: $BUILD_CTX"

    BUILD_LOG="$LOG_DIR/build.log"
    echo "  log:           $BUILD_LOG"

    # The Dockerfile uses BuildKit-only features ('# syntax=docker/dockerfile:1.7'
    # and 'RUN ... <<\'PY\'' heredocs). Docker 28 removed the legacy builder
    # entirely -- DOCKER_BUILDKIT=1 now delegates to buildx, so without the
    # buildx component installed there is no fallback that works. Fail fast
    # with install instructions before attempting the build.
    if ! docker buildx version >/dev/null 2>&1; then
        cat >&2 <<'MSG'

ERROR: docker buildx is not installed.

The Dockerfile requires BuildKit (syntax=docker/dockerfile:1.7 + RUN heredocs).
Docker 28 removed the legacy builder, so buildx is required for any build.

Install buildx, then re-run this script:

  Ubuntu / Debian (apt):
    sudo apt-get update && sudo apt-get install -y docker-buildx

  Ubuntu / Debian (Docker's official repo, recommended):
    # Follow https://docs.docker.com/engine/install/ubuntu/ -- the docker-ce
    # package bundles docker-buildx-plugin and is what most production guides
    # assume. The Ubuntu-shipped docker.io package omits buildx.

  RHEL / Fedora (dnf):
    sudo dnf install -y docker-buildx-plugin

  Manual install (any distro):
    https://github.com/docker/buildx/releases  (download into ~/.docker/cli-plugins/)

Verify with:  docker buildx version
MSG
        fail "docker buildx required -- install per the message above"
    fi
    echo "  builder:       docker buildx ($(docker buildx version | head -1))"

    BUILD_ARGS=( --progress=plain )
    if [[ -n "${PLATFORM}" ]]; then
        echo "  platform:      linux/${PLATFORM}"
        BUILD_ARGS+=( --platform "linux/${PLATFORM}" )
        if [[ ${CROSS_ARCH} -eq 1 ]]; then
            echo "  cross-build:   yes (host=${HOST_DOCKER_ARCH}); verifying QEMU binfmt..."
            if ! docker run --rm --privileged tonistiigi/binfmt 2>/dev/null \
                  | grep -q "\"linux/${PLATFORM}\""; then
                cat >&2 <<MSG

ERROR: QEMU binfmt handler for linux/${PLATFORM} is not registered.

Run the one-time host setup first:

  bash docker/setup_qemu.sh

Then re-run this script.
MSG
                fail "QEMU binfmt missing for linux/${PLATFORM}"
            fi
        fi
    else
        echo "  platform:      (native, no --platform)"
    fi
    # --load works only for single-platform builds; we never multi-platform here.
    BUILD_ARGS+=( --load )

    docker buildx build "${BUILD_ARGS[@]}" -t "$TAG" "$BUILD_CTX" 2>&1 | tee "$BUILD_LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        fail "docker build exited $rc -- see $BUILD_LOG"
    fi

    # Sanity check the build's own self-test ran and passed
    if grep -q "FAIL: missing wheels\|sm_100 (B200/GB200) missing\|sm_120 (RTX 5090) missing on amd64\|no Blackwell consumer SASS" "$BUILD_LOG"; then
        fail "build-time sanity check failed -- see $BUILD_LOG"
    fi
    grep -E "OK: torch 2.10.0|OK: all required wheels|import cleanly on no-GPU host" "$BUILD_LOG" || \
        warn "could not find 'OK:' lines in build log -- did the verification step run?"
    ok "built $TAG"
fi

# When the image we just built (or were told to use) does not match the host
# architecture, the smoke test and notebook blocks would attempt to launch
# foreign-arch user-space under QEMU plus --gpus all -- which is broken by
# design: nvidia-container-toolkit cannot expose a GPU to a QEMU-emulated
# guest, and even if it could, CUDA kernels do not run under user-space CPU
# emulation. Skip those blocks with a loud warning so the user doesn't think
# they're seeing a real validation pass.
if [[ ${CROSS_ARCH} -eq 1 ]]; then
    warn "cross-arch build (host=${HOST_DOCKER_ARCH}, image=${PLATFORM})."
    warn "skipping smoke test + notebook -- CUDA does not work under QEMU runtime."
    warn "to validate end-to-end on linux/${PLATFORM}, transfer the image to an"
    warn "actual ${PLATFORM} host (e.g. DGX Spark for arm64) and re-run with --skip-build."
    banner "summary"
    echo "  image:    $TAG"
    echo "  platform: linux/${PLATFORM}  (cross-built on ${HOST_DOCKER_ARCH})"
    echo "  log dir:  $LOG_DIR"
    echo
    [[ $SKIP_BUILD -eq 0 ]] && echo "  to paste back for PR validation:"
    [[ $SKIP_BUILD -eq 0 ]] && echo "    tail -80 $LOG_DIR/build.log"
    ok "cross-arch build verified (wheels + arch-flags assertion passed)"
    exit 0
fi

# ============================================================================
# Block 3a: smoke test
# ============================================================================
banner "Block 3a: smoke test (5-step LoRA on Llama-3.2-1B)"
SMOKE_LOG="$LOG_DIR/smoke.log"
echo "  log: $SMOKE_LOG"
docker run --rm --gpus all "$TAG" python /workspace/smoke_test.py 2>&1 | tee "$SMOKE_LOG"
rc=${PIPESTATUS[0]}
if [[ $rc -ne 0 ]]; then
    fail "smoke test exited $rc -- see $SMOKE_LOG"
fi
if ! grep -q "all checks passed" "$SMOKE_LOG"; then
    fail "smoke test did not print 'all checks passed' -- see $SMOKE_LOG"
fi
ok "smoke test passed"

# ============================================================================
# Block 3b: gpt-oss-20B fine-tuning notebook
# ============================================================================
if [[ $SKIP_NOTEBOOK -eq 1 ]]; then
    warn "skipping gpt-oss-20B notebook (--skip-notebook)"
else
    banner "Block 3b: gpt-oss-20B fine-tuning notebook (10 LoRA steps)"
    GPT_LOG="$LOG_DIR/gpt_oss.log"
    HOST_RUN_DIR="$LOG_DIR/host"
    mkdir -p "$HOST_RUN_DIR"
    echo "  log:        $GPT_LOG"
    echo "  host dir:   $HOST_RUN_DIR"

    cat > "$HOST_RUN_DIR/run_notebook.sh" <<'INNER'
#!/bin/bash
set -e
cd /workspace/host

echo "=== install triton_kernels (MXFP4 support for unsloth/gpt-oss-20b) ==="
pip install -q 'git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels' 2>&1 | tail -5

echo
echo "=== fetch + convert notebook ==="
# Use nbformat directly. We then post-process to:
#   1. Skip install cells -- the container already has unsloth + deps baked in;
#      the notebook's install cell uses Jupyter !shell magic (raw `!pip install
#      ...` lines) that nbformat dumps verbatim and Python cannot parse.
#   2. Comment out any stray !cmd / %magic lines in non-install cells.
pip install -q nbformat
# Pin to an immutable commit so this validation script doesn't silently
# change semantics when notebooks/main rolls forward. Bump deliberately
# when the upstream notebook gets a fix you want to verify against.
NB_REPO_REF="${NB_REPO_REF:-efe20c97a5bba3088b25fe068a4b1c98c0cf3a3a}"
curl -fsSL "https://raw.githubusercontent.com/unslothai/notebooks/${NB_REPO_REF}/nb/gpt-oss-(20B)-Fine-tuning.ipynb" -o nb.ipynb
test -s nb.ipynb || { echo "FAIL: nb.ipynb was not downloaded"; exit 1; }
python - <<'PY'
import nbformat, re
nb = nbformat.read('nb.ipynb', as_version=4)
out, skipped = [], 0
INSTALL_MARKERS = (
    "pip install", "uv pip install", "apt-get install",
    "_original_packages", "COLAB_", "importlib.util.find_spec",
)
for c in nb.cells:
    if c.cell_type != "code":
        continue
    src = c.source or ""
    if any(m in src for m in INSTALL_MARKERS):
        skipped += 1
        first = next((ln for ln in src.splitlines() if ln.strip()), "")[:80]
        out.append(f"# (skipped install/setup cell: {first!r})")
        out.append("")
        continue
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out.append(f"# (jupyter magic stripped) {line}")
        else:
            out.append(line)
    out.append("")
with open("nb.py", "w") as f:
    f.write("\n".join(out) + "\n")
print(f"  converted nb.py: {sum(1 for _ in open('nb.py'))} lines, {skipped} install cell(s) skipped")
PY
test -s nb.py || { echo "FAIL: nb.py was not produced"; exit 1; }
# Sanity-check: nb.py must parse as valid Python before we try to run it.
python -c "import ast; ast.parse(open('nb.py').read()); print('  nb.py is valid Python')"

echo
echo "=== patch nb.py: max_steps 30 -> 10, drop pre-train demo generations ==="
python - <<'PY'
import re
src = open('nb.py').read()
src = src.replace('max_steps = 30', 'max_steps = 10')
src = re.sub(
    r'messages = \[\s*\{[\"\']role[\"\']: [\"\']user[\"\'], [\"\']content[\"\']: [\"\']Solve x\^5.*?\n_ = model\.generate.*?streamer = TextStreamer\(tokenizer\)\)\n',
    '# (pre-train inference skipped)\n',
    src, flags=re.DOTALL, count=3,
)
open('nb.py', 'w').write(src)
print('  patched. max_steps now:', re.search(r'max_steps = (\d+)', src).group(1))
PY

echo
echo "=== run gpt-oss-20B fine-tuning ==="
python -u nb.py
INNER
    chmod +x "$HOST_RUN_DIR/run_notebook.sh"

    # Only forward HF_TOKEN if the host has one set, so an empty
    # `-e HF_TOKEN=` does not shadow whatever is already inside the image.
    # Use the dash-only form `-e HF_TOKEN` so the secret value never
    # lands in argv (visible via /proc/<pid>/cmdline to any user on
    # the host for the lifetime of the docker CLI process).
    HF_ARGS=()
    [[ -n "${HF_TOKEN:-}" ]] && HF_ARGS+=(-e HF_TOKEN)
    docker run --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$HOST_RUN_DIR:/workspace/host" \
        -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
        "${HF_ARGS[@]}" \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        "$TAG" \
        bash /workspace/host/run_notebook.sh 2>&1 | tee "$GPT_LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        fail "gpt-oss-20B notebook exited $rc -- see $GPT_LOG"
    fi
    ok "gpt-oss-20B notebook completed"
fi

# ============================================================================
# Summary
# ============================================================================
banner "summary"
echo "  image:    $TAG"
echo "  log dir:  $LOG_DIR"
echo
echo "  to paste back for PR validation:"
[[ $SKIP_BUILD    -eq 0 ]] && echo "    tail -40 $LOG_DIR/build.log"
echo "    cat      $LOG_DIR/smoke.log"
[[ $SKIP_NOTEBOOK -eq 0 ]] && echo "    tail -100 $LOG_DIR/gpt_oss.log"
echo
ok "all blocks completed"
