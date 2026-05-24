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
#   bash docker/test_locally.sh                   # all blocks
#   bash docker/test_locally.sh --skip-notebook   # blocks 1-3a only (fast)
#   bash docker/test_locally.sh --skip-build      # assume $TAG already built
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)    SKIP_BUILD=1; shift ;;
        --skip-notebook) SKIP_NOTEBOOK=1; shift ;;
        --tag)           TAG="$2"; shift 2 ;;
        --log-dir)       LOG_DIR="$2"; shift 2 ;;
        --help|-h)       sed -n '2,20p' "$0"; exit 0 ;;
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

# ============================================================================
# Block 1: pre-flight
# ============================================================================
banner "Block 1: host pre-flight"

command -v docker >/dev/null 2>&1 || fail "docker not found on PATH"
echo "  docker:       $(docker --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  host gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  host driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not on the host -- you may not be able to run --gpus all"
fi

if docker info 2>&1 | grep -qiE 'Runtimes:.*nvidia'; then
    echo "  nvidia runtime: registered with docker"
else
    warn "docker info does not list 'nvidia' as a runtime"
    warn "if --gpus all fails below, install nvidia-container-toolkit:"
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
            git clone --depth 1 -b docker-blackwell-build \
                https://github.com/unslothai/unsloth.git /tmp/unsloth-pr 2>&1 | tail -3
        else
            git -C /tmp/unsloth-pr pull --ff-only 2>&1 | tail -2
        fi
    fi
    echo "  build context: $BUILD_CTX"

    BUILD_LOG="$LOG_DIR/build.log"
    echo "  log:           $BUILD_LOG"
    docker build --progress=plain -t "$TAG" "$BUILD_CTX" 2>&1 | tee "$BUILD_LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        fail "docker build exited $rc -- see $BUILD_LOG"
    fi

    # Sanity check the build's own self-test ran and passed
    if grep -q "FAIL: missing wheels\|sm_100 (B200) missing\|sm_120 (RTX 5090) missing" "$BUILD_LOG"; then
        fail "build-time sanity check failed -- see $BUILD_LOG"
    fi
    grep -E "OK: torch 2.10.0|OK: all required wheels|OK: xformers \+ bitsandbytes" "$BUILD_LOG" || \
        warn "could not find 'OK:' lines in build log -- did the verification step run?"
    ok "built $TAG"
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
pip install -q nbconvert
curl -fsSL 'https://raw.githubusercontent.com/unslothai/notebooks/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb' -o nb.ipynb
jupyter nbconvert --to script nb.ipynb --output nb 2>/dev/null
echo "  nb.py: $(wc -l < nb.py) lines"

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

    docker run --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$HOST_RUN_DIR:/workspace/host" \
        -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
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
