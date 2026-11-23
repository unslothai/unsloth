# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO no-grad PrefixGrouper must stay within one padded mini-batch's tokens, like the packed path."""

import ast
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
SOURCE = SOURCE_PATH.read_text(encoding = "utf-8")
ROWS, L, KEEP = 8, 16, 8


def _pg_block():
    outer = next(
        n
        for n in ast.parse(SOURCE).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "grpo_trainer__get_per_token_logps_and_entropies"
    )
    inner = next(
        n
        for n in ast.walk(outer)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_per_token_logps_and_entropies"
    )
    src = ast.get_source_segment(SOURCE, inner)
    start = src.index("_pg_pad = self.processing_class.pad_token_id")
    start = src.rindex("if _pg_engage:", 0, start)
    start = src.rindex("\n", 0, start) + 1
    end = src.index("# Sequence packing", start)
    end = src.rindex("\n", 0, end) + 1
    return textwrap.dedent(src[start:end])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        self.calls.append(input_ids.numel())
        return SimpleNamespace(logits = torch.zeros(*input_ids.shape, 4))


def _run(
    tokens,
    batch_size,
    verify = False,
):
    model = _Model()
    layout = SimpleNamespace(
        flat_ids = torch.zeros(1, tokens, dtype = torch.long),
        position_ids = torch.arange(tokens).unsqueeze(0),
        prefix_seg_info = None,
        signature = ("layout",),
        extract_logps = lambda *args: torch.zeros(ROWS, KEEP),
    )
    ns = {
        "os": os,
        "torch": torch,
        "self": SimpleNamespace(
            processing_class = SimpleNamespace(pad_token_id = 0),
            _autocast_dtype = torch.bfloat16,
            _autocast_enabled = False,
        ),
        "unwrapped_model": model,
        "_pg_engage": True,
        "_pg_build_layout": lambda *args, **kwargs: layout,
        "_pg_verify_on": lambda: verify,
        "_pg_num_gen": 4,
        "_pg_result": None,
        "_pg_use": False,
        "_pg_skip_pk": False,
        "_pg_forward_fn": None,
        "_unsloth_head_hooked": False,
        "input_ids": torch.ones(ROWS, L, dtype = torch.long),
        "logits_to_keep": KEEP,
        "left_pad_tokens_per_prompt": torch.zeros(ROWS, dtype = torch.long),
        "max_left_pad": 0,
        "total_rows": ROWS,
        "seq_len": L,
        "batch_size": batch_size,
        "multiplier": 1,
        "lm_head": torch.zeros(16, 4),
        "chunked_hidden_states_selective_log_softmax": None,
        "logit_scale_multiply": 0.0,
        "logit_scale_divide": 0.0,
        "logit_softcapping": 0.0,
        "temperature": 1.0,
        "DEVICE_TYPE_TORCH": "cpu",
        "device_synchronize": lambda: None,
        "UNSLOTH_ENABLE_LOGGING": False,
    }
    exec(_pg_block(), ns)
    return ns, model.calls


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_prefix_grouper_over_the_mini_batch_tokens_is_skipped(batch_size):
    # A grouped stream longer than batch_size padded rows would run in one forward.
    ns, calls = _run(tokens = batch_size * L + 1, batch_size = batch_size)
    assert not ns["_pg_use"] and calls == []
    # Released at once, not kept alive through the fallback forwards.
    assert ns["_pg_layout"] is None


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_unverified_prefix_grouper_over_the_mini_batch_tokens_defers_nothing(batch_size):
    ns, calls = _run(tokens = batch_size * L + 1, batch_size = batch_size, verify = True)
    assert ns["_pg_forward_fn"] is None and ns["_pg_layout"] is None and calls == []


@pytest.mark.parametrize("batch_size", [1, 2, ROWS])
def test_prefix_grouper_within_the_mini_batch_tokens_runs(batch_size):
    ns, calls = _run(tokens = batch_size * L, batch_size = batch_size)
    assert ns["_pg_use"] and calls == [batch_size * L]
    # First use defers the forward to the verify site.
    ns, calls = _run(tokens = batch_size * L, batch_size = batch_size, verify = True)
    assert ns["_pg_forward_fn"] is not None and calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
