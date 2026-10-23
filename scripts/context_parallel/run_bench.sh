#!/bin/bash
# Long-sequence memory / step time: 2 LoRA training steps per run (step 2 is the steady state).
# OOM is recorded as status "oom" in the jsonl (the shell line also prints FAILED).
#
# usage: run_bench.sh <name>=<model_dir> [<name>=<model_dir> ...]
#   MODES                 default "vanilla fsdp cp fsdp_cp"  (vanilla = 1 GPU, the rest = 2 GPUs)
#   SEQS                  default "16384 32768"
#   UNSLOTH_CP_ATTN_CHUNK query chunk of the DeepSeek-V4 / V3.2 CP attention (default 1024;
#                         a value >= seq reproduces the old dense [S, S] behaviour)
#   PYTHON / TORCHRUN     interpreter and launcher (default: python / torchrun on PATH)
#   OUT_DIR               default /root/autodl-tmp/bench_out/full  (<OUT_DIR>/<name>.jsonl + logs)
#   MASTER_PORT           default 29555
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/bench_out/full}
MASTER_PORT=${MASTER_PORT:-29555}
MODES=${MODES:-vanilla fsdp cp fsdp_cp}
SEQS=${SEQS:-16384 32768}
export PYTHONPATH=$REPO
cd "$REPO"
mkdir -p "$OUT_DIR"
for spec in "$@"; do
  name=${spec%%=*}; dir=${spec#*=}
  for seq in $SEQS; do
    for mode in $MODES; do
      echo "=== $name seq=$seq $mode $(date +%T)"
      log="$OUT_DIR/${name}_${seq}_${mode}.log"
      if [ "$mode" = vanilla ]; then
        timeout 1800 $PYTHON -m unsloth.distributed.bench_sft --mode $mode --model "$dir" --seq-len $seq --samples 2 --out "$OUT_DIR/$name.jsonl" > "$log" 2>&1 || echo "FAILED $name $seq $mode"
      else
        timeout 1800 $TORCHRUN --nproc_per_node=2 --master_port $MASTER_PORT -m unsloth.distributed.bench_sft --mode $mode --model "$dir" --seq-len $seq --samples 2 --out "$OUT_DIR/$name.jsonl" > "$log" 2>&1 || echo "FAILED $name $seq $mode"
      fi
    done
  done
  echo "--- $name results"; cut -c1-260 "$OUT_DIR/$name.jsonl" 2>/dev/null
done
echo "BENCH DONE $(date +%T)"
