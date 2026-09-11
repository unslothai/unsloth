#!/usr/bin/env bash
# End-to-end Docker validation for the unsloth-rocm image.
#
# Runs three blocks:
#   1. Host pre-flight (docker, rocm-smi, /dev/kfd accessible)
#   2. Build the image (no GPU required at build time)
#   3a. Smoke test: 5-step LoRA on Llama-3.2-1B (~1-2 min)
#   3b. Real workload: gpt-oss-20B fine-tuning notebook with max_steps=10
#       (~10 min, needs ~30GB free for the model cache)
#
# Usage:
#   bash docker/test_locally-rocm.sh                   # all blocks
#   bash docker/test_locally-rocm.sh --skip-notebook   # blocks 1-3a only (fast)
#   bash docker/test_locally-rocm.sh --skip-build      # assume $TAG already built
#   TAG=my-image:latest bash docker/test_locally-rocm.sh
#   HF_TOKEN=hf_xxx bash docker/test_locally-rocm.sh
#
# For RDNA4 / Strix Halo (gfx1150/1151/1200/1201), build with:
#   ROCM_VERSION=7.x.x TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 \
#     TAG=unsloth-rocm-rdna4:test bash docker/test_locally-rocm.sh
#
# All output is teed to $LOG_DIR (default /tmp/unsloth-rocm-test/).
set -uo pipefail

TAG="${TAG:-unsloth-rocm:test}"
LOG_DIR="${LOG_DIR:-/tmp/unsloth-rocm-test}"
SKIP_BUILD=0
SKIP_NOTEBOOK=0
ROCM_VERSION="${ROCM_VERSION:-6.2.4}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm6.2}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)    SKIP_BUILD=1; shift ;;
        --skip-notebook) SKIP_NOTEBOOK=1; shift ;;
        --tag)           TAG="$2"; shift 2 ;;
        --log-dir)       LOG_DIR="$2"; shift 2 ;;
        --help|-h)       sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR"

GREEN='\033[1;32m'; RED='\033[1;31m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'
banner() { printf "\n${BLUE}==== %s ====${NC}\n" "$*"; }
ok()     { printf "${GREEN}OK${NC}    %s\n" "$*"; }
warn()   { printf "${YELLOW}WARN${NC}  %s\n" "$*"; }
err()    { printf "${RED}ERROR${NC} %s\n" "$*" >&2; }
fail()   { err "$*"; exit 1; }

# AMD GPU device flags used for all `docker run` invocations.
AMD_DEVICE_FLAGS=(--device /dev/kfd --device /dev/dri --group-add video)

# ============================================================================
# Block 1: pre-flight
# ============================================================================
banner "Block 1: host pre-flight"

command -v docker >/dev/null 2>&1 || fail "docker not found on PATH"
echo "  docker:       $(docker --version)"

DOCKER_INFO_OUT=$(docker info 2>&1)
DOCKER_INFO_RC=$?
if [[ $DOCKER_INFO_RC -ne 0 ]]; then
    err "Cannot talk to the docker daemon as user '$USER'."
    cat >&2 <<MSG

docker info exited $DOCKER_INFO_RC. The most common cause is that your user
is not in the 'docker' group. Fix:

  sudo usermod -aG docker \$USER
  newgrp docker
  docker info | head -3

MSG
    fail "docker daemon unreachable"
fi
echo "  daemon:       reachable as '$USER'"

if [[ -e /dev/kfd ]]; then
    echo "  /dev/kfd:     present"
    if command -v rocm-smi >/dev/null 2>&1; then
        echo "  host gpu:     $(rocm-smi --showproductname 2>/dev/null | grep -m1 'GPU\[' | sed 's/.*: //' || echo '(rocm-smi present)')"
    else
        warn "rocm-smi not on the host PATH (ROCm may still work inside the container)"
    fi
else
    warn "/dev/kfd not found -- ROCm drivers may not be installed or user lacks access"
    warn "Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    warn "Then: sudo usermod -aG video,render \$USER  (log out and back in)"
fi

# Check the user is in the video group -- required for /dev/dri access.
if id -nG 2>/dev/null | grep -qw video; then
    echo "  video group:  ok (current user is a member)"
else
    warn "current user is not in the 'video' group"
    warn "Fix: sudo usermod -aG video,render \$USER  (log out/in after)"
fi

ok "pre-flight done"

# ============================================================================
# Block 2: build
# ============================================================================
if [[ $SKIP_BUILD -eq 1 ]]; then
    warn "skipping build (--skip-build); expecting $TAG to exist"
else
    banner "Block 2: build $TAG"

    if [[ -f "Dockerfile.rocm" && -f "smoke_test_rocm.py" ]]; then
        BUILD_CTX="$PWD"
    elif [[ -f "docker/Dockerfile.rocm" ]]; then
        BUILD_CTX="$PWD/docker"
    else
        fail "Cannot find Dockerfile.rocm. Run from the repo root or docker/ directory."
    fi
    echo "  build context: $BUILD_CTX"
    echo "  ROCm version:  $ROCM_VERSION"
    echo "  torch index:   $TORCH_INDEX_URL"

    BUILD_LOG="$LOG_DIR/build.log"
    echo "  log:           $BUILD_LOG"

    if ! docker buildx version >/dev/null 2>&1; then
        fail "docker buildx is not installed. Install: sudo apt-get install -y docker-buildx"
    fi
    echo "  builder:       docker buildx ($(docker buildx version | head -1))"

    docker buildx build \
        --progress=plain \
        --load \
        --build-arg ROCM_VERSION="${ROCM_VERSION}" \
        --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
        -f "${BUILD_CTX}/Dockerfile.rocm" \
        -t "$TAG" \
        "$BUILD_CTX" 2>&1 | tee "$BUILD_LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        fail "docker build exited $rc -- see $BUILD_LOG"
    fi

    # Sanity-check the build's own self-test ran and passed.
    if grep -q "FAIL: missing packages\|Expected a ROCm torch wheel" "$BUILD_LOG"; then
        fail "build-time sanity check failed -- see $BUILD_LOG"
    fi
    grep -E "OK: torch .* HIP|OK: all required packages" "$BUILD_LOG" || \
        warn "could not find 'OK:' lines in build log -- did the verification step run?"
    ok "built $TAG"
fi

# ============================================================================
# Block 3a: smoke test
# ============================================================================
banner "Block 3a: smoke test (5-step LoRA on Llama-3.2-1B)"
SMOKE_LOG="$LOG_DIR/smoke.log"
echo "  log: $SMOKE_LOG"
docker run --rm \
    "${AMD_DEVICE_FLAGS[@]}" \
    "$TAG" python /workspace/smoke_test_rocm.py 2>&1 | tee "$SMOKE_LOG"
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

echo "=== fetch + convert notebook ==="
pip install -q nbformat
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
python -c "import ast; ast.parse(open('nb.py').read()); print('  nb.py is valid Python')"

echo
echo "=== patch nb.py: max_steps 30 -> 10 ==="
python - <<'PY'
import re
src = open('nb.py').read()
src = src.replace('max_steps = 30', 'max_steps = 10')
open('nb.py', 'w').write(src)
print('  patched. max_steps now:', re.search(r'max_steps = (\d+)', src).group(1))
PY

echo
echo "=== run gpt-oss-20B fine-tuning ==="
python -u nb.py
INNER
    chmod +x "$HOST_RUN_DIR/run_notebook.sh"

    HF_ARGS=()
    [[ -n "${HF_TOKEN:-}" ]] && HF_ARGS+=(-e HF_TOKEN)
    docker run --rm \
        "${AMD_DEVICE_FLAGS[@]}" \
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
