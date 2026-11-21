"""CPU-only check that a converted SenseNova-Loop checkpoint builds the int8 table.

Usage: python sensenova-loop-verify-build.py <checkpoint-dir-or-repo-id>

No weights are read and no GPU is touched: the model is built on the meta device after the
converted-checkpoint prepare, and the script prints the embed_tokens class name. A plain
nn.Embedding here means the load would materialize the bf16 table (or worse, hold the int8
values cast to bf16), so a failure is a wiring bug, not a memory warning.
"""

import sys

from unsloth.models.sensenova_loop import verify_sensenova_loop_int8_build

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    print("embed_tokens:", verify_sensenova_loop_int8_build(sys.argv[1]))
