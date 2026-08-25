# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context compaction, and why one half of the check is worthless alone.

Studio reports the fit on the completion itself, as `context_truncated` with
`dropped_messages`, so a conversation past the window can be shown to have been
SHORTENED rather than refused, without a browser.

The rule is a pair, deliberately:

* an over-length conversation returns 200 with `dropped_messages > 0`;
* a two-message conversation drops nothing.

The first alone passes on a server that reports truncation unconditionally,
which is indistinguishable from working. The second alone passes on a server
that never compacts and returns a context-length error instead. Only together
do they say the field tracks length.

It also has to run where the window is KNOWN. `assert_server_flags` reloads the
model pinned to `--studio-ctx`; a compaction check before that is aimed at a
context length nobody set.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _func(name: str) -> ast.FunctionDef:
    for cls in ast.walk(ast.parse(SRC)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f"no method named {name!r}")


def _body(name: str = "assert_compaction") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def test_the_assertion_exists_and_is_driven_from_the_run():
    assert _body()
    assert "self.assert_compaction()" in _body("execute")


def test_it_runs_after_the_window_is_pinned():
    body = _body("execute")
    flags_at = body.index("self.assert_server_flags()")
    ours_at = body.index("self.assert_compaction()")
    assert flags_at < ours_at, (
        "the reload in assert_server_flags is what pins the context length; "
        "before it, this check overflows a window nobody set"
    )


def test_the_over_length_case_must_report_dropped_messages():
    func = _func("assert_compaction")
    tests = [ast.unparse(n.test) for n in ast.walk(func) if isinstance(n, ast.If)]
    assert any(
        "long_dropped" in t for t in tests
    ), "nothing depends on the long conversation having been shortened"


def test_the_short_control_is_present_and_is_the_opposite_claim():
    """Without it, a server that always claims truncation passes."""
    func = _func("assert_compaction")
    short = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.If) and "short_dropped" in ast.unparse(n.test)
    ]
    assert short, "there is no negative control"
    for node in short:
        # It must fail when the short chat DID drop, which is the opposite
        # polarity to the long one.
        assert not (
            isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
        ), "the control must fire when a SHORT conversation reports a drop"


def test_a_refusal_is_a_failure_rather_than_a_pass():
    """A context-length error is not compaction. The distinction is the whole
    feature: one shortens the prompt and answers, the other gives up.

    Asserted structurally. The first version of this test matched the failure
    MESSAGE, with an `or` that was true of any body at all -- a guard satisfied
    by its own surrounding text, which is the exact shape this directory has
    been caught by five times.
    """
    func = _func("assert_compaction")
    refusals = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "code != 200"
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "append"
            for c in ast.walk(n)
        )
    ]
    assert len(refusals) >= 2, (
        "both the long conversation and the short control must fail on a "
        "non-200, or a server that refuses everything reads as a pass"
    )


def test_the_overflow_is_built_past_the_budget_and_not_merely_to_it():
    """The budget subtracts the reply reserve and the template's own framing,
    so a prompt equal to the context length is not reliably over it."""
    body = _body()
    assert "for i in range(40):" in body
    assert "self.args.studio_ctx" in body


def test_the_cpu_fallback_records_the_assertion_rather_than_omitting_it():
    assert '"compaction",' in _body("execute")


def test_every_status_read_comes_from_a_real_call():
    """Found by mutation, not by reasoning: assigning `code = 200` after the
    request survived every other rule here, because they all read the shape of
    the branches and none of them asked where the value came from.

    A hardcoded status is unlikely as drift and trivial as a "fix" for a red,
    which is the same thing.
    """
    func = _func("assert_compaction")
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not ({"code"} & targets or {"code", "body"} <= _tuple_names(node)):
            continue
        assert isinstance(node.value, ast.Call), (
            f"`code` is assigned from {ast.unparse(node.value)!r} rather than "
            f"from a request, so the status this checks is not the server's"
        )


def _tuple_names(node: ast.Assign) -> set:
    out = set()
    for target in node.targets:
        if isinstance(target, ast.Tuple):
            out.update(e.id for e in target.elts if isinstance(e, ast.Name))
    return out


def test_compaction_is_REQUESTED_rather_than_expected_by_default():
    """Measured on kernel unsloth-probe-studio-full2-815a0c, where this
    assertion failed a documented default.

    `context_overflow` defaults to "error": an over-length conversation comes
    back 400 with code=context_length_exceeded, so a client's own trim loop can
    see it. Compaction is a policy you ASK for. "truncate_oldest" is the one
    that applies to a plain chat; "truncate_middle" is limited to client-tool
    and response_format passthrough
    (studio/backend/models/inference.py:context_overflow).
    """
    body = _body()
    assert 'context_overflow = "truncate_oldest"' in body
    assert 'context_overflow = "truncate_middle"' not in body, (
        "truncate_middle does not apply to a plain chat, so it would report a "
        "policy that never ran"
    )


def test_the_default_policy_control_is_present_and_expects_a_refusal():
    """The half that stops the request field from being decorative.

    Without it, the check above passes on a server that compacts everything
    regardless of what was asked for, and naming the policy proves nothing.
    """
    func = _func("assert_compaction")
    refusals = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Compare) and "400" in ast.unparse(node)
    ]
    assert refusals, "nothing asserts the default policy still refuses"
    assert "long_status_default_policy" in _body()


def test_the_over_length_conversation_is_sent_twice_under_two_policies():
    """One request cannot compare two policies. Both calls have to exist, and
    they have to send the SAME messages, or the pair compares a policy against
    a different conversation."""
    func = _func("assert_compaction")
    calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "chat"
        and "long_messages" in ast.unparse(node)
    ]
    assert len(calls) == 2, f"expected two long-conversation calls, found {len(calls)}"
    overflow = [call for call in calls if any(kw.arg == "context_overflow" for kw in call.keywords)]
    assert len(overflow) == 1, (
        "exactly one of the two must name the policy: both naming it removes "
        "the control, neither naming it removes the feature under test"
    )
