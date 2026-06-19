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

The fused cross-entropy path in ``CausalLM_fast_forward`` computes the loss
straight from ``hidden_states`` and never materializes ``logits``. The
``return_dict=True`` branch correctly returns ``EMPTY_LOGITS``; the
``not return_dict`` branch used to return ``(logits,) + outputs[1:]``, which
raised ``UnboundLocalError: ... 'logits'`` whenever it ran (e.g. training with
``return_dict=False``).

This guards the source so the fused-CE ``not return_dict`` return keeps using
``EMPTY_LOGITS``. Pure text inspection, so it runs under the GPU-free
``tests/conftest.py`` (the fused path itself is GPU/triton only)."""

from __future__ import annotations

from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent


def _fused_ce_not_return_dict_return(source: str) -> str:
    """Return the `output = (...) + outputs[1:]` line from the FIRST
    `if not return_dict:` that follows an `unsloth_fused_ce_loss(` call."""
    fused = source.index("unsloth_fused_ce_loss(")
    guard = source.index("if not return_dict:", fused)
    after = source.index("output = (", guard)
    line_end = source.index("\n", after)
    return source[after:line_end]


@pytest.mark.parametrize("rel", ["unsloth/models/llama.py", "unsloth/models/mistral.py"])
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
