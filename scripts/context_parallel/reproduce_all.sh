#!/bin/bash
# Reproduce the context-parallel validation end to end on a 2-GPU box.
#
#   MODELS_DIR=/root/autodl-tmp/models \
#   PYTHON=/root/autodl-tmp/BIV/train/.venv/bin/python \
#   TORCHRUN=/root/autodl-tmp/BIV/train/.venv/bin/torchrun \
#   bash scripts/context_parallel/reproduce_all.sh [download] [cpu] [align] [bench]
#
# Steps (default: cpu align bench; "download" is opt-in, ~290 GB):
#   download  ModelScope checkpoints into $MODELS_DIR (download_models.py)
#   cpu       2-rank gloo vs 1-process tests on tiny random models, incl. chunked V4 / V3.2 paths
#   align     one-step LoRA gradient alignment, seq 256, every model; Gemma-4-E4B also in fp32
#   bench     16K / 32K peak memory and step time: vanilla, fsdp, cp, fsdp_cp
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHON=${PYTHON:-python}
export TORCHRUN=${TORCHRUN:-torchrun}
export MODELS_DIR=${MODELS_DIR:-/root/autodl-tmp/models}
M=$MODELS_DIR
STEPS=${*:-cpu align bench}
V2_OVERLAY=${V2_OVERLAY:-$(dirname "$M")/align_models/DeepSeek-V2-Lite}

MODELS=(
  k3-0.40B=$M/Kimi-K3-0.40B
  gemma4-E4B=$M/gemma-4-E4B-it
  dsv2-lite=$V2_OVERLAY
  moonlight=$M/Moonlight-16B-A3B
  gemma4-26B-A4B=$M/gemma-4-26B-A4B-it
  gemma4-31B=$M/gemma-4-31B-it
  kimi-linear-48B=$M/Kimi-Linear-48B-A3B-Instruct
)

for step in $STEPS; do
  case $step in
    download)
      $PYTHON "$HERE/download_models.py"
      ;;
    cpu)
      CUDA_VISIBLE_DEVICES= $PYTHON -W ignore "$HERE/run_cpu_tests.py"
      UNSLOTH_CP_ATTN_CHUNK=4 CUDA_VISIBLE_DEVICES= $PYTHON -W ignore "$HERE/run_cpu_tests.py" _tiny_deepseek_v4 _tiny_deepseek_v32
      ;;
    align)
      $PYTHON "$HERE/make_pad_overlay.py" "$M/DeepSeek-V2-Lite" "$V2_OVERLAY"
      bash "$HERE/run_align.sh" 256 "${MODELS[@]}"
      ALIGN_EXTRA="--dtype fp32" bash "$HERE/run_align.sh" 256 gemma4-E4B-fp32=$M/gemma-4-E4B-it
      ;;
    bench)
      $PYTHON "$HERE/make_pad_overlay.py" "$M/DeepSeek-V2-Lite" "$V2_OVERLAY"
      bash "$HERE/run_bench.sh" "${MODELS[@]}"
      ;;
    *)
      echo "unknown step: $step"; exit 1
      ;;
  esac
done
