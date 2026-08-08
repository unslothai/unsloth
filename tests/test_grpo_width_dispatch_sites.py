# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every lm_head matmul in the no-grad GRPO logprob path dispatches on width.

`_get_per_token_logps_and_entropies` sets UNSLOTH_RETURN_HIDDEN_STATES=1, but
`.logits` carries hidden states only when the model's forward is the Unsloth
generated one. When it is not, `.logits` is a real [.., vocab] tensor, and
handing that to `chunked_hidden_states_selective_log_softmax` runs it into the
lm_head matmul:

    a and b must have same reduction dim, but got
    [((s47*s87 + 255)//256), s33] X [1536, 151936]

`s33` there is a backed symbol that specialises to the hidden size; the message
only appears when the tensor genuinely is the wrong width, and 151936 is the
vocab. The VLM branch of the padded loop already dispatched on
`logits_chunk.shape[-1] == lm_head.shape[1]`; the text branch of the same loop
and both sequence-packing call sites did not.

These checks are structural (AST), not textual, so that neither a comment
mentioning the guard nor a reformat can satisfy them.
"""

from __future__ import annotations

import ast
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")

HIDDEN_STATES_HELPER = "chunked_hidden_states_selective_log_softmax"
RAW_LOGITS_HELPER = "chunked_selective_log_softmax"

# One shared parse: nodes from separate parses never compare equal, which would
# silently make every containment check below vacuously true.
TREE = ast.parse(open(SOURCE_PATH, "r", encoding = "utf-8").read())


def _logprob_function():
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "_get_per_token_logps_and_entropies":
            return node
    return None


def _matmul_calls(scope):
    """Calls to the hidden-states helper, i.e. the ones that hit the matmul.

    The PrefixGrouper site passes the helper to `extract_logps` as a bare Name
    rather than calling it, so it is deliberately not one of these.
    """
    return [
        node for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == HIDDEN_STATES_HELPER
    ]


def _is_width_test(test):
    """`<anything>.shape[-1] == lm_head.shape[1]`, in either operand order."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False

    def _shape_index(node):
        # node is expected to be `X.shape[i]`; return i, else None.
        if not isinstance(node, ast.Subscript):
            return None
        value = node.value
        if not (isinstance(value, ast.Attribute) and value.attr == "shape"):
            return None
        return ast.unparse(node.slice), ast.unparse(value.value)

    left = _shape_index(test.left)
    right = _shape_index(test.comparators[0])
    if left is None or right is None:
        return False
    sides = {left, right}
    # one side is the candidate tensor's last dim, the other is lm_head's
    # hidden dim, which is exactly the `hidden_dim = lm_head.shape[1]` this
    # function already computes for the chunk autotuner.
    lm_head_side = {s for s in sides if s[1] == "lm_head"}
    if len(lm_head_side) != 1:
        return False
    (index, _), = lm_head_side
    if index != "1":
        return False
    other, = sides - lm_head_side
    return other[0] == "-1"


def _guard_for(call):
    """The nearest enclosing `if` that width-tests and holds `call` in its body."""
    best = None
    for node in ast.walk(TREE):
        if not isinstance(node, ast.If) or not _is_width_test(node.test):
            continue
        if not any(call is inner for stmt in node.body for inner in ast.walk(stmt)):
            continue
        if best is None or node.lineno > best.lineno:
            best = node
    return best


def _called_names(statements):
    names = set()
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return names


def test_the_logprob_function_is_present():
    assert _logprob_function() is not None, (
        "_get_per_token_logps_and_entropies not found; this file's other "
        "checks would pass vacuously"
    )


def test_the_matmul_call_sites_are_all_accounted_for():
    """Four sites: packed, packed verifier, padded text, padded VLM.

    Pinned so that a new unguarded call site added later fails here rather than
    slipping past the per-site checks below.
    """
    calls = _matmul_calls(_logprob_function())
    assert len(calls) == 4, [call.lineno for call in calls]


def test_every_matmul_call_site_dispatches_on_width():
    calls = _matmul_calls(_logprob_function())
    unguarded = [call.lineno for call in calls if _guard_for(call) is None]
    assert not unguarded, (
        f"lines {unguarded} call {HIDDEN_STATES_HELPER} without first comparing "
        "the tensor's last dim against lm_head's hidden dim, so a forward that "
        "returns real logits reaches the lm_head matmul"
    )


def test_every_width_guard_falls_back_to_the_raw_logits_helper():
    calls = _matmul_calls(_logprob_function())
    for call in calls:
        guard = _guard_for(call)
        assert guard is not None, call.lineno
        assert RAW_LOGITS_HELPER in _called_names(guard.orelse), (
            f"the guard at line {guard.lineno} has no {RAW_LOGITS_HELPER} "
            "fallback, so the raw-logits case is unhandled"
        )


def test_the_raw_logits_fallback_skips_scaling_and_softcapping():
    """The forward already applied them, so re-applying would double them."""
    forbidden = {"logit_scale_multiply", "logit_scale_divide", "logit_softcapping"}
    calls = _matmul_calls(_logprob_function())
    for call in calls:
        guard = _guard_for(call)
        assert guard is not None, call.lineno
        for node in ast.walk(ast.Module(body = guard.orelse, type_ignores = [])):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != RAW_LOGITS_HELPER:
                continue
            passed = {ast.unparse(arg) for arg in node.args}
            passed |= {kw.arg for kw in node.keywords if kw.arg is not None}
            passed |= {ast.unparse(kw.value) for kw in node.keywords}
            leaked = forbidden & passed
            assert not leaked, (
                f"the raw-logits fallback at line {node.lineno} passes {leaked}; "
                "the model forward already applied them"
            )


def test_the_padded_text_branch_is_guarded():
    """The crash site: pixel_values is None, so no enclosing try catches it.

    Located by structure rather than by line number: the `if pixel_values is
    None` inside the padded loop.
    """
    function = _logprob_function()
    branches = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ast.unparse(node.test) == "pixel_values is None"
        and _matmul_calls(ast.Module(body = node.body, type_ignores = []))
    ]
    assert len(branches) == 1, [node.lineno for node in branches]
    text_branch, = branches
    calls = _matmul_calls(ast.Module(body = text_branch.body, type_ignores = []))
    assert len(calls) == 1, [call.lineno for call in calls]
    assert _guard_for(calls[0]) is not None, (
        "the text branch of the padded loop reaches the lm_head matmul "
        "unguarded, and unlike the packing sites it is not inside a try, so "
        "this is what surfaces as a TorchRuntimeError during training"
    )


def test_the_packing_sites_are_guarded():
    """Both `_pk_` sites: the packed forward and its first-use verifier.

    They sit inside `except Exception`, so a failure here is swallowed into a
    permanent packing-disable rather than a crash.
    """
    function = _logprob_function()
    packed = [
        call for call in _matmul_calls(function)
        if any(
            isinstance(node, ast.Name) and node.id.startswith("_pk_")
            for node in ast.walk(call)
        )
    ]
    assert len(packed) == 2, [call.lineno for call in packed]
    unguarded = [call.lineno for call in packed if _guard_for(call) is None]
    assert not unguarded, unguarded
