#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [llama|qwen-moe]"
  echo "  llama: run tests/qlora/test_llama_qlora_train_and_merge.py with its required env"
  echo "  qwen-moe: run tests/qlora/test_qwenmoe_qlora_train_and_merge.py with its required env and flash-attn blocked"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

choice="$1"
shift || true  # allow passing extra args to the test script

case "$choice" in
  llama)
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export PYTHONPATH="/workspace/unsloth:${PYTHONPATH:-}"
    export HIP_VISIBLE_DEVICES=2
    script_path="tests/qlora/test_llama_qlora_train_and_merge.py"
    ;;
  qwen-moe)
    # Prepend blocker dir; sitecustomize inside will hide flash-attn only
    STUB_DIR="$(cd -- "$(dirname -- "$0")" && pwd)/disable_flash_attn"
    export PYTHONPATH="${STUB_DIR}:/workspace/unsloth:${PYTHONPATH:-}"
    export HIP_VISIBLE_DEVICES=1,2,3,0
    export TORCH_COMPILE_DISABLE=1
    export TORCHDYNAMO_DISABLE=1
    script_path="tests/qlora/test_qwenmoe_qlora_train_and_merge.py"
    ;;
  *)
    usage
    ;;
esac

if [[ ! -f "$script_path" ]]; then
  echo "Error: cannot find $script_path (run from repo root where tests/ exists)." >&2
  exit 1
fi

echo "Running $script_path with choice '$choice'..."
python "$script_path" "$@"