# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO old / ref scorer sets UNSLOTH_RETURN_HIDDEN_STATES for its forwards. When the section raises
(an OOM the caller retries, auto_find_batch_size), the variable must not stay "1": every later
generate / evaluate forward of an Unsloth model would return hidden states as `.logits`."""

from __future__ import annotations

import ast
import os
import textwrap
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
_NAME = "UNSLOTH_RETURN_HIDDEN_STATES"


def _named_function(tree, name):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _is_request(stmt):
    """`os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0" if _unsloth_head_hooked else "1"`."""
    return (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.targets[0], ast.Subscript)
        and isinstance(stmt.targets[0].slice, ast.Constant)
        and stmt.targets[0].slice.value == _NAME
        and isinstance(stmt.value, ast.IfExp)
    )


def _scorer_section():
    """The statements from the hidden-states request to the return, as a function body, taken from the
    source Unsloth generates for the trainer (what actually runs), not the template."""
    from unsloth.models import rl_replacements

    assert Path(rl_replacements.__file__).resolve() == _SOURCE.resolve(), rl_replacements.__file__
    text = textwrap.dedent(
        rl_replacements.grpo_trainer__get_per_token_logps_and_entropies(
            "_get_per_token_logps_and_entropies", ""
        )
    )
    inner = _named_function(ast.parse(text), "_get_per_token_logps_and_entropies")
    found = []
    for node in ast.walk(inner):
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(node, field, None)
            if not isinstance(seq, list):
                continue
            for i, stmt in enumerate(seq):
                # The section's own request ends in the scorer's return (the packed and PrefixGrouper
                # fallbacks re-request inside their except handlers).
                end = next((j for j in range(i, len(seq)) if isinstance(seq[j], ast.Return)), None)
                if _is_request(stmt) and end is not None:
                    # From just after the chunk zip, so whatever the section saves before the request runs.
                    start = max(
                        k + 1
                        for k in range(i)
                        if isinstance(seq[k], ast.Assign)
                        and getattr(seq[k].targets[0], "id", None) == "zipped_inputs"
                    )
                    found.append(seq[start : end + 1])
    assert len(found) == 1, f"expected one hidden-states request in the scorer, found {len(found)}"
    lines = text.splitlines(keepends = True)
    source = "".join(lines[found[0][0].lineno - 1 : found[0][-1].end_lineno])
    return "def _section():\n" + textwrap.indent(textwrap.dedent(source), "    ")


class _Trainer:
    """The first trainer attribute the section reads raises, as a CUDA OOM in its first forward would."""

    def __getattr__(self, name):
        raise torch.OutOfMemoryError(f"CUDA out of memory (injected at self.{name})")


@pytest.mark.parametrize("prior", [None, "0"])
def test_hidden_states_request_is_undone_when_the_scorer_raises(prior, monkeypatch):
    if prior is None:
        monkeypatch.delenv(_NAME, raising = False)
    else:
        monkeypatch.setenv(_NAME, prior)
    namespace = {
        "os": os,
        "torch": torch,
        "self": _Trainer(),
        "_unsloth_head_hooked": False,
        "UNSLOTH_GRPO_PREFIX_GROUPER_ON": True,
        "UNSLOTH_GRPO_SEQ_PACKING_ON": True,
    }
    exec(_scorer_section(), namespace)
    with pytest.raises(torch.OutOfMemoryError):
        namespace["_section"]()
    # The trainer caught it and moved on: the next generate() must get logits.
    assert os.environ.get(_NAME) == prior
    assert os.environ.get(_NAME, "0") != "1"
