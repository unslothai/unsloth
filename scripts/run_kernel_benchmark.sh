#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [attention|moe] [options...]"
  echo ""
  echo "Options:"
  echo "  attention: Run Attention Kernel benchmark (scripts/attention_impl_benchmark.py)"
  echo "  moe:       Run MoE Kernel benchmark (scripts/moe_impl_benchmark.py)"
  echo ""
  echo "Attention options:"
  echo "  --batch-sizes --seq-lens --num-heads --head-dim --dtypes --num-iters --warmup --device --seed --no-causal --allow-tf32"
  echo ""
  echo "MoE options:"
  echo "  --batch-sizes --seq-lens --dtypes --device --num-iters --warmup --seed --hidden-size --moe-intermediate-size --num-experts --top-k --norm-topk-prob --hidden-act"
  echo ""
  echo "Examples:"
  echo "  $0 attention --batch-sizes 1 4 --seq-lens 128 512 --dtypes fp16 --num-heads 16"
  echo "  $0 moe --batch-sizes 1 4 --seq-lens 128 512 --dtypes fp16 --num-iters 3 --num-experts 32"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

choice="$1"
shift || true 

export PYTHONPATH="/workspace/unsloth:${PYTHONPATH:-}"

case "$choice" in
  attention)
    # Attention kernel测试
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    #export HIP_VISIBLE_DEVICES=2
    script_path="scripts/attention_impl_benchmark.py"
    
    if [[ $# -eq 0 ]]; then
      set -- --batch-sizes 1 4 --seq-lens 128 2048 --dtypes fp16
    fi
    ;;
  moe)
    # MoE kernel测试
    #export HIP_VISIBLE_DEVICES=2
    script_path="scripts/moe_impl_benchmark.py"
    
    # 如果没有提供额外参数，使用默认参数
    if [[ $# -eq 0 ]]; then
      set -- --batch-sizes 1 4 --seq-lens 128 512 --dtypes fp16 --num-iters 10
    fi
    ;;
  *)
    usage
    ;;
esac

# 检查脚本是否存在
if [[ ! -f "$script_path" ]]; then
  echo "Error: cannot find $script_path" >&2
  echo "请确保在项目根目录运行, 并且scripts目录包含相应的测试脚本" >&2
  exit 1
fi

echo "========================================"
echo "Running $script_path"
echo "Choice: $choice"
echo "Device: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "Python path: $PYTHONPATH"
echo "Arguments: $*"
echo "========================================"
echo ""

# 运行Python脚本
python "$script_path" "$@"