#!/bin/bash
# One-step gradient alignment: vanilla (1 GPU) vs fsdp / cp / fsdp_cp (2 GPUs), then compare.
# LoRA r=16, fixed seed and batch; reports loss diff and cosine of the flattened LoRA gradients.
#
# usage: run_align.sh <seq_len> <name>=<model_dir> [<name>=<model_dir> ...]
#   ALIGN_EXTRA="--dtype fp32"   extra align_sft args (fp32 separates CP bugs from bf16 rounding;
#                                fsdp modes need bf16 because apply_fsdp uses a bf16 MixedPrecisionPolicy)
#   PYTHON / TORCHRUN            interpreter and launcher (default: python / torchrun on PATH)
#   OUT_DIR                      results, default /root/autodl-tmp/align_out  (<OUT_DIR>/<name>/<mode>.{pt,log})
#   MASTER_PORT                  default 29555
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/align_out}
MASTER_PORT=${MASTER_PORT:-29555}
export PYTHONPATH=$REPO
cd "$REPO"
SEQ=$1; shift
for spec in "$@"; do
  name=${spec%%=*}; dir=${spec#*=}; out=$OUT_DIR/$name; mkdir -p "$out"
  echo "=== $name vanilla $(date +%T)"
  $PYTHON -m unsloth.distributed.align_sft --mode vanilla --model "$dir" --seq-len $SEQ --out-dir "$out" $ALIGN_EXTRA > "$out/vanilla.log" 2>&1 || echo "FAILED $name vanilla"
  for m in fsdp cp fsdp_cp; do
    echo "=== $name $m $(date +%T)"
    $TORCHRUN --nproc_per_node=2 --master_port $MASTER_PORT -m unsloth.distributed.align_sft --mode $m --model "$dir" --seq-len $SEQ --out-dir "$out" $ALIGN_EXTRA > "$out/$m.log" 2>&1 || echo "FAILED $name $m"
  done
  echo "=== $name compare"
  $PYTHON -m unsloth.distributed.align_sft --mode compare --out-dir "$out" 2>&1 | grep -E "^(vanilla|fsdp|cp|fsdp_cp) +:"
done
echo "ALL DONE $(date +%T)"
