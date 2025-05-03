#!/bin/bash

set -euo pipefail

SEQLENS=(1024)  
DTYPES=(bfloat16)
PERMUTE_X=(false true)
PERMUTE_Y=(false true)
AUTOTUNE=(false true)

for SEQLEN in "${SEQLENS[@]}"; do
    for DTYPE in "${DTYPES[@]}"; do
        for PX in "${PERMUTE_X[@]}"; do
            for PY in "${PERMUTE_Y[@]}"; do
                for AT in "${AUTOTUNE[@]}"; do

                    ARGS=()
                    [[ "$PX" == "true" ]] && ARGS+=("--permute_x")
                    [[ "$PY" == "true" ]] && ARGS+=("--permute_y")
                    [[ "$AT" == "true" ]] && ARGS+=("--autotune")

                    ARGS+=(--seqlen "$SEQLEN" --dtype "$DTYPE")

                    echo "Running with args: ${ARGS[*]}"
                    if ! python -m tests.test_qwen3_moe "${ARGS[@]}"; then
                        echo "❌ Test failed with args: --permute_x=$PX --permute_y=$PY --autotune=$AT --seqlen=$SEQLEN --dtype=$DTYPE" >&2
                    else
                        echo "✅ Test passed with args: --permute_x=$PX --permute_y=$PY --autotune=$AT --seqlen=$SEQLEN --dtype=$DTYPE"
                    fi

                done
            done
        done
    done
done
