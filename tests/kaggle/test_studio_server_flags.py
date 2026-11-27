# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Check the applied values, since llama-server starts even when a flag it dislikes is dropped."""

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
    # Sized to visible cards: --studio-concurrent pins this half to one card.
    assert '"tensor_split": [1.0] * max(1, len(cards))' in body
    assert "cards = gpu_inventory()" in body
    assert '"max_seq_length": self.args.studio_ctx' in body


def test_a_one_card_run_says_the_two_card_split_was_not_exercised():
    """A one-card run records that the two-card split was not exercised, rather than keeping the name."""
    body = _body()
    assert 'detail["tensor_split_over_two_cards"] = len(cards) >= 2' in body
    assert "was NOT exercised" in body
    # A record, not a failure: a one-card run is a scheduling decision.
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
    """A check on the request only checks a dict this file built, so read the applied status."""
    body = _body()
    assert '"/api/inference/status"' in body
    assert 'applied.get("cache_type_kv")' in body
    assert 'applied.get("context_length")' in body


def test_no_branch_in_the_check_is_wired_to_a_constant():
    """A branch whose test is a constant is never taken or always taken, so no rule here may be one."""
    tree = ast.parse(_body())
    constants = [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant)
    ]
    assert constants == [], f"branches wired to a constant: {constants}"


def test_the_cache_rules_append_a_failure_on_both_paths():
    """Missing values and silent downgrades must each append a failure via their own branch."""
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
    """Unpinned, llama-server admits prompts on n_ctx alone and differs silently from 2048."""
    body = _body()
    assert "int(ctx) > self.args.studio_ctx" in body


def test_gpu_residency_is_confirmed_after_the_split():
    """A tensor_split on CPU still reports healthy; the threshold comparison must exist."""
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
    """Server flags reload the model, so they must sit between GPU inference and training."""
    infer_at = SRC.index("gpu_ok = self.assert_gpu_inference()")
    flags_at = SRC.index("self.assert_server_flags()")
    train_at = SRC.index("trained = self.assert_training()")
    assert infer_at < flags_at < train_at

    # Guarded by gpu_ok, not a constant, or the call never runs.
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
    """Imports the real payload by path: AST rules see that a branch exists, not that it answers right."""
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
    """A payload pinned to card 0 must see one GPU, or tensor_split reaches a device that is not there."""
    module = _payload_module()
    _with_smi(module, monkeypatch, TWO_ROWS, "0")
    assert len(module.gpu_inventory()) == 1


def test_an_unpinned_payload_still_sees_every_card(monkeypatch):
    """Unpinned legs must still see every GPU, or multi-GPU coverage reads as single-card."""
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
