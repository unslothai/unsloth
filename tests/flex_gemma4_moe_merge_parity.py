# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Numerical parity check for ``refresh_moe_lora_merge_from_pristine``
at Gemma 4 26B-A4B expert shapes.

Reuses ``merge_under_test`` and ``reference_merge`` from
``tests/flex_moe_merge_parity.py`` (both are standalone helpers — they
import nothing from the flex engine, only exercise the bitwise kernel
logic lifted from ``unsloth/inference/flex_moe.py:682-800``).

Shapes covered (Gemma 4 26B-A4B):
- gate_up_proj: (E=128, 2I=1408, H=2816)  standard ``(E, out, in)``
- down_proj:    (E=128, H=2816, I=704)    standard ``(E, out, in)``
Rank-16 LoRA, single + dual adapter, bf16 + fp32.

Gemma 4 MoE uses the same ``F.linear``-oriented expert layout as Qwen3,
so only the ``transposed=False`` branch of
``refresh_moe_lora_merge_from_pristine`` is exercised here. (The
transposed branch is already covered for gpt-oss in
``flex_moe_merge_parity.py``.)

Usage::
    CUDA_VISIBLE_DEVICES=2 python -u tests/flex_gemma4_moe_merge_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.flex_moe_merge_parity import (  # noqa: E402
    merge_under_test,
    reference_merge,
    test_correctness,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3407)
    print(f"[merge-parity-gemma4] device={device} dtype=bf16 + fp32")
    print(f"[merge-parity-gemma4] torch={torch.__version__}")
    print()

    print("== Correctness (fp32 golden + bf16 realistic Gemma 4 shapes) ==")
    # gate_up_proj: (E=128, 2I=1408, H=2816) standard
    # down_proj:    (E=128, H=2816,  I=704 ) standard
    cases = [
        # (E, in_dim, out_dim, R, dtype, transposed, n_adapters)
        ( 8,  64, 128, 4,  torch.float32, False, 1),
        ( 8,  64, 128, 4,  torch.float32, False, 2),
        (16, 128, 256, 8,  torch.float32, False, 2),
        # bf16 at Gemma 4 26B-A4B MoE shapes.
        (128, 2816, 1408, 16, torch.bfloat16, False, 1),  # gate_up_proj
        (128,  704, 2816, 16, torch.bfloat16, False, 1),  # down_proj
        (128, 2816, 1408, 16, torch.bfloat16, False, 2),  # two adapters
        (128, 2816, 1408, 64, torch.bfloat16, False, 1),  # higher rank
    ]
    all_ok = True
    for E, in_dim, out_dim, R, dtype, tr, na in cases:
        ok = test_correctness(E, in_dim, out_dim, R, dtype, device,
                              transposed=tr, n_adapters=na)
        all_ok = all_ok and ok
    print(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
