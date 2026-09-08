# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""llama.cpp server flags: a quantized KV cache, a two-card split, a pinned ctx.

The vacuity here is specific and easy to write by accident: **asserting that
the load succeeded**. llama-server starts happily when a flag it does not like
is dropped, so a load that ignored `cache_type_kv` entirely, fell back to one
card, and ran at the model's default context is indistinguishable from a
correct one by return code alone.

Studio's status separates the request from what is in force, which is what
makes a real check possible. So the rules are about the APPLIED values, and a
downgrade is allowed only when Studio says why -- a model whose cache layout
cannot be quantized is entitled to refuse, and silence is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _func(name: str = "assert_server_flags") -> ast.FunctionDef:
    tree = ast.parse(SRC)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _body() -> str:
    tree = ast.parse(SRC)
    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "assert_server_flags"
    )
    return ast.get_source_segment(SRC, func) or ""


def test_the_load_requests_all_three_flags():
    body = _body()
    assert '"cache_type_kv": "q8_0"' in body
    # Sized to the VISIBLE cards rather than hardcoded to two.
    # --studio-concurrent pins this half to one card so it can share with a
    # training leg, and [1.0, 1.0] against a one-card server asks llama.cpp to
    # split across a device that is not there: what comes back is a failure
    # about the load rather than about the flag.
    assert '"tensor_split": [1.0] * max(1, len(cards))' in body
    assert "cards = gpu_inventory()" in body
    assert '"max_seq_length": self.args.studio_ctx' in body


def test_a_one_card_run_says_the_two_card_split_was_not_exercised():
    """The coverage this shares away must be STATED, not silently dropped.

    Under --studio-concurrent the split is over one device, which is not the
    flag the brief asks about. A check that keeps its name while testing less
    is the failure this directory keeps being caught by, so the report carries
    `tensor_split_over_two_cards` and a note naming what was and was not
    covered.
    """
    body = _body()
    assert 'detail["tensor_split_over_two_cards"] = len(cards) >= 2' in body
    assert "was NOT exercised" in body
    # And it is a RECORD, not a failure: sharing is a scheduling decision, so a
    # one-card run is not a defect and must not go red for being one.
    func = _func("assert_server_flags")
    for node in ast.walk(func):
        if isinstance(node, ast.If) and "cards" in ast.unparse(node.test):
            appended = [
                inner
                for inner in ast.walk(node)
                if isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "append"
            ]
            assert not appended, (
                "a one-card run fails rather than records, so every "
                "--studio-concurrent run would go red by design"
            )


def test_the_check_reads_the_applied_values_not_the_request():
    """The whole point. A check on the request is a check on a dict this file
    just built."""
    body = _body()
    assert '"/api/inference/status"' in body
    assert 'applied.get("cache_type_kv")' in body
    assert 'applied.get("context_length")' in body


def test_no_branch_in_the_check_is_wired_to_a_constant():
    """The guard that caught five vacuous guards, including four of my own.

    Every rule in this file was first written as "the failure message appears
    in the source". That is satisfied by `if False:` above an untouched
    message, so disabling a rule outright left the test green -- the exact
    "assertion satisfied by its own surrounding text" failure this repo has
    recorded before.

    A constant test means a branch that can never be taken (or always is), and
    no rule here has any business being either.
    """
    tree = ast.parse(_body())
    constants = [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant)
    ]
    assert constants == [], f"branches wired to a constant: {constants}"


def test_the_cache_rules_append_a_failure_on_both_paths():
    """Both the missing-value path and the silent-downgrade path must be able
    to fail, and each is reached through its own branch."""
    tree = ast.parse(_body())
    messages = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "append"
    ]
    joined = " ".join(messages)
    assert "no cache_type_kv at all" in joined
    assert "silent downgrade" in joined.lower()


def test_the_context_pin_is_checked_against_the_requested_value():
    """llama-server admits a prompt on n_ctx alone, so a server at the model
    default behaves differently from one at 2048 and the difference does not
    show in a chat response."""
    body = _body()
    assert "int(ctx) > self.args.studio_ctx" in body


def test_gpu_residency_is_confirmed_after_the_split():
    """A tensor_split that fell back to CPU reports a healthy server and proves
    nothing about either card. Checked structurally: the threshold comparison
    must exist, not merely the message about it."""
    tree = ast.parse(_body())
    compares = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    assert any(
        "used" in c and "200" in c for c in compares
    ), f"no residency threshold comparison: {compares}"
    assert "nvidia_used_mib()" in _body()


def test_it_is_skipped_rather_than_passed_when_the_model_is_not_on_the_gpu():
    """Structural: the skip branch must record FALSE. Recording True would turn
    "we could not test this" into "this passed"."""
    tree = ast.parse(SRC)
    records = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "record"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "server_flags"
    ]
    skips = [n for n in records if len(n.args) > 1 and "skipped" in ast.unparse(n)]
    assert skips, "no skip path records server_flags"
    for node in skips:
        assert (
            node.args[1].value is False
        ), "the skip path records a PASS, so an untested flag reads as a working one"


def test_it_runs_between_gpu_inference_and_training():
    """It reloads the chat model with different flags. Earlier would change the
    model the inference checks measured; later would put a reload between the
    adapter and the export."""
    infer_at = SRC.index("gpu_ok = self.assert_gpu_inference()")
    flags_at = SRC.index("self.assert_server_flags()")
    train_at = SRC.index("trained = self.assert_training()")
    assert infer_at < flags_at < train_at

    # And it is guarded by gpu_ok rather than by a constant, or the call sits
    # in the right place and never runs.
    tree = ast.parse(SRC)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "assert_server_flags" in ast.unparse(node)
        and not isinstance(node, ast.Module)
    ]
    assert calls, "the call is not inside a branch at all"
    assert any(ast.unparse(n.test) == "gpu_ok" for n in calls), (
        f"the call is gated on something other than gpu_ok: "
        f"{[ast.unparse(n.test) for n in calls]}"
    )


def test_the_context_default_is_the_one_the_brief_asks_for():
    assert '"--studio-ctx",' in SRC
    assert "default = 2048," in SRC


def _payload_module():
    """The real payload, imported by path so the rules DRIVE it.

    Every rule above this point reads the source with `ast`, which is the right
    instrument for "does the branch exist" and the wrong one for "does it answer
    correctly". The bug below was invisible to all of them: the code was
    exactly as written and the value it produced was false.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_studio_payload_flags", PAYLOAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_smi(module, monkeypatch, rows: str, visible: str | None):
    class _Proc:
        returncode = 0
        stdout = rows

    monkeypatch.setattr(module, "run", lambda *a, **k: _Proc())
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)


TWO_ROWS = "Tesla T4, 15360 MiB, 7.5\nTesla T4, 15360 MiB, 7.5\n"


def test_a_pinned_payload_does_not_see_both_cards(monkeypatch):
    """The exact reading from unsloth-probe-full-concurrent-417238.

    nvidia-smi lists two T4s. build_kernel.py:835 pinned this payload to card 0
    with CUDA_VISIBLE_DEVICES. The report recorded `cards_visible: 2` and
    `tensor_split_over_two_cards: True` and sent `tensor_split: [1.0, 1.0]` to a
    one-card server, which loaded anyway -- so the assertion passed green while
    asking llama.cpp to split across a device that was not there.
    """
    module = _payload_module()
    _with_smi(module, monkeypatch, TWO_ROWS, "0")
    assert len(module.gpu_inventory()) == 1


def test_an_unpinned_payload_still_sees_every_card(monkeypatch):
    """The other direction, and it is the one the two-card claim depends on: a
    leg that is deliberately NOT pinned must still report both cards, or real
    multi-GPU coverage would be reported as single-card."""
    module = _payload_module()
    _with_smi(module, monkeypatch, TWO_ROWS, None)
    assert len(module.gpu_inventory()) == 2
    _with_smi(module, monkeypatch, TWO_ROWS, "0,1")
    assert len(module.gpu_inventory()) == 2


def test_an_empty_setting_means_no_cards_rather_than_all_of_them(monkeypatch):
    """CUDA_VISIBLE_DEVICES="" is a deliberate "no GPU" and is not the same as
    unset. Reading it as unset would report a full inventory on a session that
    has no card at all, which preflight exists to catch."""
    module = _payload_module()
    _with_smi(module, monkeypatch, TWO_ROWS, "")
    assert module.gpu_inventory() == []


def test_a_uuid_selection_still_counts_the_cards(monkeypatch):
    """CUDA_VISIBLE_DEVICES may name GPU-<uuid> rather than an index. The
    description cannot be recovered from an nvidia-smi row, but the COUNT is
    what tensor_split is sized from, so it must stay right."""
    module = _payload_module()
    _with_smi(module, monkeypatch, TWO_ROWS, "GPU-abcdef12")
    assert len(module.gpu_inventory()) == 1


def test_the_split_is_sized_from_the_visible_cards_and_not_from_nvidia_smi():
    """The link between the fix and the flag. `tensor_split` is built from
    `gpu_inventory()`, so a rule that only tested the helper would leave the
    request free to be sized from anything."""
    body = _body()
    assert "cards = gpu_inventory()" in body
    assert "[1.0] * max(1, len(cards))" in body
    assert 'detail["tensor_split_over_two_cards"] = len(cards) >= 2' in body
