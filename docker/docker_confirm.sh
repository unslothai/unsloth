#!/usr/bin/env bash
#
# docker_confirm.sh  (Unsloth Docker image confirmation - Linux / WSL2 / macOS)
# Confirms the published Unsloth Docker images actually work on this machine:
# pulls them, checks GPU passthrough (or CPU fallback), runs a real 5-step
# LoRA training smoke, checks the baked llama.cpp GGUF tooling, boots the
# full image and probes Studio + JupyterLab, then prints a PASS/FAIL report.
#
# Nothing is installed on the host beyond the Docker images themselves; the
# containers it starts are removed afterwards (KEEP=1 keeps them running).
#
# One-liner:
#   curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/docker_confirm.sh | bash
#
# What to expect per machine class:
#   Linux + NVIDIA (B200 / RTX 6000 / RTX 50-series). GPU mode, all phases.
#   Windows + NVIDIA via Docker Desktop (WSL2 backend): run inside the WSL2
#     distro or Git Bash. GPU mode if Docker Desktop has WSL2 GPU enabled.
#   DGX Spark / GB10 (Linux arm64): GPU mode, the arm64 image child is pulled
#     automatically.
#   macOS (M-series) and Windows + AMD (Strix Halo): CPU mode is auto-detected
#     (no NVIDIA passthrough exists for these); training phases are skipped,
#     Studio chat / Jupyter / GGUF tooling still validate.
#
# Env overrides: IMAGE (default unsloth/unsloth:latest)
#                BASE_IMAGE (default unsloth/unsloth:base)
#                GPUS=all|none|0|0,1   (default: auto-detect)
#                PORT_STUDIO=18000  PORT_JUPYTER=18888
#                WORK=~/unsloth_docker_test   (logs)
#                HF_CACHE=~/.cache/huggingface (mounted to speed model pulls)
#                SKIP_PULL=1 (use local images)  SKIP_TRAIN=1  KEEP=1
#
set -uo pipefail

IMAGE="${IMAGE:-unsloth/unsloth:latest}"
BASE_IMAGE="${BASE_IMAGE:-unsloth/unsloth:base}"
GPUS="${GPUS:-auto}"
PORT_STUDIO="${PORT_STUDIO:-18000}"
PORT_JUPYTER="${PORT_JUPYTER:-18888}"
WORK="${WORK:-$HOME/unsloth_docker_test}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SKIP_PULL="${SKIP_PULL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
KEEP="${KEEP:-0}"
ARCH="$(uname -m)"
OS="$(uname -s)"

PASS_N=0; FAIL_N=0; WARN_N=0; STUDIO_CID=""
bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
ok(){   printf '  [PASS] %s\n' "$*"; PASS_N=$((PASS_N+1)); }
bad(){  printf '  [FAIL] %s\n' "$*"; FAIL_N=$((FAIL_N+1)); }
warn(){ printf '  [WARN] %s\n' "$*"; WARN_N=$((WARN_N+1)); }
info(){ printf '         %s\n' "$*"; }
hr(){   printf -- '---------------------------------------------------------------\n'; }

cleanup(){
  if [ "$KEEP" != "1" ] && [ -n "$STUDIO_CID" ]; then
    docker rm -f "$STUDIO_CID" >/dev/null 2>&1
  fi
}
trap cleanup EXIT

mkdir -p "$WORK" "$HF_CACHE"
echo; bold "=== Unsloth Docker image confirmation ==="
echo "scratch dir : $WORK"; hr

# --------------------------------------------------------------------------- #
# 1. Host detection
# --------------------------------------------------------------------------- #
bold "1) Host detection"
info "uname     : $OS $ARCH ($(uname -r 2>/dev/null))"
IS_WSL=0
grep -qiE "microsoft|wsl" /proc/version 2>/dev/null && { IS_WSL=1; info "WSL       : yes"; }
if ! command -v docker >/dev/null 2>&1; then
  bad "docker not found on PATH - install Docker Engine / Docker Desktop first"
  echo; bold "RESULT: cannot continue without docker."; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  bad "docker daemon not reachable (permission denied or not running)"
  info "try: sudo usermod -aG docker \$USER && re-login, or start Docker Desktop"
  echo; bold "RESULT: cannot continue without a reachable docker daemon."; exit 1
fi
ok "docker daemon reachable ($(docker --version 2>/dev/null))"

GPU_MODE=0
NVRT_LISTED=0
if [ "$GPUS" = "none" ]; then
  info "GPU mode  : disabled by GPUS=none"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
  info "GPU(s)    :"
  nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/           - /'
  # `docker info | grep Runtimes:.*nvidia` misses CDI setups (docker 25+
  # with nvidia-ctk cdi) and Docker Desktop's WSL2 backend, both of which
  # expose GPUs without a host-visible runtime entry. Treat the listing as
  # a hint only; phase 3 probes --gpus for real and demotes to CPU mode if
  # the probe fails.
  if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    ok "NVIDIA GPU visible and docker lists the nvidia runtime"
    NVRT_LISTED=1
  else
    warn "nvidia runtime not listed by docker info (normal under CDI or Docker Desktop WSL2) - probing --gpus directly in phase 3"
  fi
  GPU_MODE=1
else
  info "no NVIDIA GPU on the host (or nvidia-smi missing)"
fi
if [ "$GPU_MODE" = "0" ]; then
  warn "CPU mode: training phases are skipped; Studio chat / Jupyter / GGUF tooling still validate"
fi
GPU_FLAG=(--gpus all)
case "$GPUS" in
  auto|all|none) ;;
  *) GPU_FLAG=(--gpus "\"device=${GPUS}\"") ;;
esac
hr

# --------------------------------------------------------------------------- #
# 2. Pull images
# --------------------------------------------------------------------------- #
bold "2) Pull images"
for img in "$BASE_IMAGE" "$IMAGE"; do
  if [ "$SKIP_PULL" = "1" ]; then
    docker image inspect "$img" >/dev/null 2>&1 && ok "local image present: $img" || bad "SKIP_PULL=1 but image missing locally: $img"
  elif docker pull "$img" >"$WORK/pull_$(echo "$img" | tr '/:' '__').log" 2>&1; then
    ok "pulled $img"
  elif docker image inspect "$img" >/dev/null 2>&1; then
    # Locally built tags (test_locally.sh / docker build) are not on a
    # registry; that is fine as long as the image is present.
    warn "not pullable but present locally: $img"
  else
    bad "could not pull $img (see $WORK/pull_*.log)"
  fi
done
hr

# --------------------------------------------------------------------------- #
# 3. GPU passthrough / CPU fallback inside the container
# --------------------------------------------------------------------------- #
bold "3) Container runtime check"
if [ "$GPU_MODE" = "1" ]; then
  if docker run --rm "${GPU_FLAG[@]}" "$BASE_IMAGE" python -c \
      "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '-', torch.cuda.get_device_name(0))" \
      >"$WORK/gpu_check.log" 2>&1; then
    ok "torch.cuda available in-container: $(tail -1 "$WORK/gpu_check.log")"
  else
    if [ "$NVRT_LISTED" = "1" ]; then
      bad "GPU passthrough failed despite a listed nvidia runtime (see $WORK/gpu_check.log) - falling back to CPU mode"
    else
      warn "--gpus probe failed - docker has no nvidia runtime or CDI spec (install nvidia-container-toolkit); falling back to CPU mode"
    fi
    tail -5 "$WORK/gpu_check.log" | sed 's/^/         /'
    GPU_MODE=0
  fi
fi
if [ "$GPU_MODE" = "0" ]; then
  if docker run --rm -e UNSLOTH_ALLOW_CPU=1 "$BASE_IMAGE" python -c \
      "import torch; print('torch', torch.__version__, 'cpu-mode ok')" \
      >"$WORK/cpu_check.log" 2>&1; then
    ok "CPU mode boots: $(tail -1 "$WORK/cpu_check.log")"
  else
    bad "container failed to start even in CPU mode (see $WORK/cpu_check.log)"
    tail -5 "$WORK/cpu_check.log" | sed 's/^/         /'
  fi
fi
hr

# --------------------------------------------------------------------------- #
# 4. Training smoke (GPU only): 5 LoRA steps on Llama-3.2-1B 4-bit
# --------------------------------------------------------------------------- #
bold "4) Training smoke"
if [ "$GPU_MODE" = "1" ] && [ "$SKIP_TRAIN" != "1" ]; then
  if docker run --rm "${GPU_FLAG[@]}" --ipc=host \
      -v "$HF_CACHE":/workspace/.cache/huggingface \
      ${HF_TOKEN:+-e HF_TOKEN} \
      "$BASE_IMAGE" python /workspace/smoke_test.py >"$WORK/train_smoke.log" 2>&1; then
    ok "smoke_test.py: 5 LoRA steps completed"
    grep -E '^step|loss' "$WORK/train_smoke.log" | tail -5 | sed 's/^/         /'
  else
    bad "training smoke failed (see $WORK/train_smoke.log)"
    tail -10 "$WORK/train_smoke.log" | sed 's/^/         /'
  fi
else
  warn "skipped (CPU mode or SKIP_TRAIN=1)"
fi
hr

# --------------------------------------------------------------------------- #
# 5. GGUF tooling: baked llama.cpp prebuilt
# --------------------------------------------------------------------------- #
bold "5) GGUF tooling (baked llama.cpp)"
if docker run --rm -e UNSLOTH_SKIP_GPU_CHECK=1 "$BASE_IMAGE" bash -c '
    set -e
    test -x "$UNSLOTH_LLAMA_CPP_PATH/llama-quantize"
    test -f "$UNSLOTH_LLAMA_CPP_PATH/convert_hf_to_gguf.py"
    "$UNSLOTH_LLAMA_CPP_PATH/llama-server" --version 2>&1 | head -2
    cat "$UNSLOTH_LLAMA_CPP_PATH/UNSLOTH_PREBUILT_INFO.json" 2>/dev/null | head -5
  ' >"$WORK/gguf_check.log" 2>&1; then
  ok "llama-quantize + llama-server + convert_hf_to_gguf.py present and runnable"
  grep -E 'version|asset' "$WORK/gguf_check.log" | head -3 | sed 's/^/         /'
else
  bad "baked llama.cpp check failed (see $WORK/gguf_check.log)"
  tail -5 "$WORK/gguf_check.log" | sed 's/^/         /'
fi
hr

# --------------------------------------------------------------------------- #
# 5b. vLLM (GRPO fast_inference=True)
# --------------------------------------------------------------------------- #
bold "5b) vLLM (GRPO fast_inference=True)"
if docker run --rm -e UNSLOTH_SKIP_GPU_CHECK=1 "$BASE_IMAGE" \
     python -c 'import vllm; print("vllm", vllm.__version__)' \
     >"$WORK/vllm_check.log" 2>&1; then
  ok "vllm importable: $(grep -oE 'vllm [0-9][^ ]*' "$WORK/vllm_check.log" | head -1)"
else
  IMG_ARCH="$(docker run --rm -e UNSLOTH_SKIP_GPU_CHECK=1 "$BASE_IMAGE" uname -m 2>/dev/null || echo unknown)"
  if [ "$IMG_ARCH" = "x86_64" ]; then
    bad "vllm missing or broken on x86_64 image (see $WORK/vllm_check.log)"
    tail -3 "$WORK/vllm_check.log" | sed 's/^/         /'
  else
    warn "vllm not available on $IMG_ARCH image; GRPO fast_inference=True unavailable (arm64 wheels are newer, fail-soft at image build)"
  fi
fi
hr

# --------------------------------------------------------------------------- #
# 6. Full image: Studio + JupyterLab boot
# --------------------------------------------------------------------------- #
bold "6) Studio + JupyterLab (full image)"
RUN_ARGS=(-d -p "$PORT_STUDIO":8000 -p "$PORT_JUPYTER":8888)
if [ "$GPU_MODE" = "1" ]; then RUN_ARGS+=("${GPU_FLAG[@]}"); else RUN_ARGS+=(-e UNSLOTH_ALLOW_CPU=1); fi
STUDIO_CID="$(docker run "${RUN_ARGS[@]}" "$IMAGE" 2>"$WORK/studio_run.err")" || STUDIO_CID=""
if [ -z "$STUDIO_CID" ]; then
  bad "full image failed to start (see $WORK/studio_run.err)"
else
  info "container : ${STUDIO_CID:0:12} (studio http://localhost:$PORT_STUDIO, jupyter http://localhost:$PORT_JUPYTER)"
  ok_studio=0; ok_jupyter=0
  for _ in $(seq 1 60); do
    if [ "$ok_studio" = 0 ] && curl -fsS "http://localhost:$PORT_STUDIO/api/health" >/dev/null 2>&1; then ok_studio=1; fi
    # /login, not /api: a password hash is always configured so /api returns 403.
    if [ "$ok_jupyter" = 0 ] && curl -fsS "http://localhost:$PORT_JUPYTER/login" >/dev/null 2>&1; then ok_jupyter=1; fi
    [ "$ok_studio" = 1 ] && [ "$ok_jupyter" = 1 ] && break
    sleep 5
  done
  [ "$ok_studio" = 1 ]  && ok "Studio /api/health healthy" || { bad "Studio /api/health never went healthy (docker logs ${STUDIO_CID:0:12})"; docker logs --tail 15 "$STUDIO_CID" 2>&1 | sed 's/^/         /'; }
  [ "$ok_jupyter" = 1 ] && ok "JupyterLab /login responding"  || bad "JupyterLab /login never responded"
fi
hr

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
bold "=== SUMMARY ==="
echo "host      : $OS $ARCH  wsl=$IS_WSL  gpu_mode=$GPU_MODE"
echo "images    : $IMAGE / $BASE_IMAGE"
echo "logs      : $WORK"
echo "PASS: $PASS_N   WARN: $WARN_N   FAIL: $FAIL_N"
if [ "$KEEP" = "1" ] && [ -n "$STUDIO_CID" ]; then
  echo "container ${STUDIO_CID:0:12} left running (KEEP=1): studio :$PORT_STUDIO jupyter :$PORT_JUPYTER"
fi
if [ "$FAIL_N" -eq 0 ]; then
  bold "RESULT: CONFIRMED - the Unsloth Docker images work on this machine."
  exit 0
else
  bold "RESULT: $FAIL_N hard failure(s) - paste this whole output back."
  exit 1
fi
