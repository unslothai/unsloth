# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Drift detector for unsloth#2068.

The fused-CE path never materializes ``logits``, so its ``not return_dict``
return must use ``EMPTY_LOGITS`` (it once used ``logits`` -> UnboundLocalError).
Pure text inspection, so it runs GPU-free (the fused path is GPU/triton only)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent


def _fused_ce_not_return_dict_return(source: str) -> str:
    """Return the `output = (...) + outputs[1:]` assignment from the first
    `if not return_dict:` after the `unsloth_fused_ce_loss(` call. Whitespace-
    tolerant regexes survive reformatting; the anchor targets the fused branch."""
    fused = re.search(r"unsloth_fused_ce_loss\s*\(", source)
    if fused is None:
        raise ValueError("unsloth_fused_ce_loss( call not found")
    guard = re.search(r"if\s+not\s+return_dict\s*:", source[fused.end() :])
    if guard is None:
        raise ValueError("`if not return_dict:` guard not found after fused-CE call")
    guard_start = fused.end() + guard.start()
    out = re.search(
        r"output\s*=\s*\(.*?outputs\s*\[\s*1\s*:\s*\]", source[guard_start:], re.DOTALL
    )
    if out is None:
        raise ValueError("`output = (...) + outputs[1:]` assignment not found")
    return out.group(0)


@pytest.mark.parametrize(
    "rel", ["unsloth/models/llama.py", "unsloth/models/mistral.py"]
)
def test_fused_ce_not_return_dict_uses_empty_logits(rel):
    path = _REPO / rel
    source = path.read_text(encoding = "utf-8")
    assert "unsloth_fused_ce_loss(" in source, f"{rel}: fused-CE call vanished"

    ret = _fused_ce_not_return_dict_return(source)
    assert "EMPTY_LOGITS" in ret, (
        f"DRIFT (#2068): {rel} fused-CE `not return_dict` returns {ret!r}; it must "
        f"use EMPTY_LOGITS since `logits` is never assigned on the fused-CE path."
    )
    assert "(logits,)" not in ret, (
        f"DRIFT (#2068): {rel} fused-CE `not return_dict` references the unassigned "
        f"`logits` ({ret!r}) -> UnboundLocalError at runtime."
    )
