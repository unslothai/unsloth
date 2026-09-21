# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The '-bf16' notice must not claim 16bit when a quantization_config survives.

A `-bf16` name drops the plain load_in_4bit / 8bit / fp8 flags, but a user
`quantization_config` stays in `**kwargs` and still quantizes, so there the
notice would say the opposite of what happens.

Source-level, because reaching the branch needs a real checkpoint download.
"""

import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "unsloth" / "models" / "loader.py"
SRC = LOADER.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)


def _bf16_notice_guards():
    """Every `if` whose body is only the '-bf16' notice `print`."""
    guards = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.If) or len(node.body) != 1:
            continue
        stmt = node.body[0]
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        func = stmt.value.func
        if not (isinstance(func, ast.Name) and func.id == "print"):
            continue
        text = ast.get_source_segment(SRC, stmt) or ""
        if "load in 16bit" in text and "-bf16" in text:
            guards.append(node)
    return guards


def test_all_four_bf16_branches_carry_the_notice():
    """Two in FastLanguageModel.from_pretrained, two in FastModel.from_pretrained."""
    assert len(_bf16_notice_guards()) == 4


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_the_notice_is_gated_on_no_user_quantization_config(index):
    guards = _bf16_notice_guards()
    assert len(guards) == 4, "the notice moved; update this test"
    test_src = ast.get_source_segment(SRC, guards[index].test) or ""
    assert "quantization_config" in test_src, (
        "the '-bf16' notice claims a 16bit load, but a user-supplied "
        "quantization_config is still forwarded to Transformers and still "
        "quantizes, so the notice must be suppressed when one is present"
    )
    assert "kwargs" in test_src, (
        "read kwargs['quantization_config']: the local `quantization_config` is "
        "not cleared when the bitsandbytes-unavailable branch pops it from kwargs"
    )


def test_the_flags_are_still_cleared():
    """The notice is a message; it must not change what the branch does."""
    assert SRC.count("load_in_16bit = True") >= 4


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_the_notice_never_calls_the_load_requested(index):
    """`load_in_4bit` defaults to True, so a bare `from_pretrained("org/model-bf16")`
    reaches this branch with the flag set without the caller requesting anything."""
    guards = _bf16_notice_guards()
    assert len(guards) == 4, "the notice moved; update this test"
    text = ast.get_source_segment(SRC, guards[index].body[0]) or ""
    assert "request" not in text.lower(), (
        "the '-bf16' notice must describe what happens, not claim the caller "
        f"asked for it: load_in_4bit defaults to True. Got: {text}"
    )


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_an_explicit_16bit_request_is_not_told_its_quant_was_dropped(index):
    """`load_in_16bit` defaults to False and nothing sets it True before this
    branch, so True here is the caller's own word: they asked for this load."""
    guards = _bf16_notice_guards()
    assert len(guards) == 4, "the notice moved; update this test"
    test_src = ast.get_source_segment(SRC, guards[index].test) or ""
    assert "load_in_16bit" in test_src, (
        "gate the notice on `not load_in_16bit`: with `load_in_16bit = True` the "
        "caller explicitly asked for the 16bit load this branch performs"
    )


BEHAVIOUR_PROBE = r"""
import contextlib, io, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"  # the notice prints before any download
from unsloth import FastLanguageModel, FastModel
# Without bitsandbytes both loaders clear `load_in_4bit` before the branch, so
# there is no quant left to disable and the notice correctly stays silent.
from unsloth.models.loader import ALLOW_BITSANDBYTES

NAME = "unslothtestorg/Definitely-Not-A-Real-Repo-bf16"

def notice(cls, **kwargs):
    buf = io.StringIO()
    err = ""
    try:
        with contextlib.redirect_stdout(buf):
            cls.from_pretrained(NAME, **kwargs)
    except BaseException as e:
        err = f"{type(e).__name__}: {e}"  # the fake repo cannot resolve, and the
        # notice prints long before that -- but the mode check does not.
    # FastModel rejects mutually exclusive modes before the '-bf16' branch, so an
    # empty result from that raise means the call never reached the code under
    # test and an assertion on it would pass without exercising anything.
    assert "Can only load in" not in err, f"{cls.__name__} {kwargs}: never reached the branch: {err}"
    lines = [l for l in buf.getvalue().splitlines() if "(-bf16) checkpoint" in l]
    return lines[0] if lines else ""

for cls in (FastLanguageModel, FastModel):
    name = cls.__name__
    bare = notice(cls)
    if ALLOW_BITSANDBYTES:
        assert bare, f"{name}: bare call printed no notice"
        assert "request" not in bare.lower(), f"{name}: bare call was told it requested 4bit: {bare}"
    else:
        assert not bare, f"{name}: 4bit was already off, but the notice still ran: {bare}"
    assert not notice(cls, load_in_4bit = False), f"{name}: 4bit-off call got a notice"

# `load_in_16bit = True` on its own leaves the default `load_in_4bit = True` set:
# the one combination where the notice is suppressed by the caller's 16bit word
# rather than by there being no quant to drop. FastLanguageModel takes that pair,
# so it is the half that pins the gate; FastModel raises on it, and there the
# explicit 16bit call can only be spelled with 4bit off.
assert not notice(FastLanguageModel, load_in_16bit = True), "FastLanguageModel: explicit 16bit call got a notice"
assert not notice(FastModel, load_in_4bit = False, load_in_16bit = True), "FastModel: explicit 16bit call got a notice"
print("PROBE_OK")
"""


def test_the_notice_on_a_real_bare_call():
    """Out of process: importing unsloth patches the interpreter, and the probe
    needs an offline environment that must not leak into the rest of the suite."""
    import subprocess

    env = dict(os.environ, PYTHONPATH = str(ROOT), HF_HUB_OFFLINE = "1")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", BEHAVIOUR_PROBE],
            capture_output = True,
            text = True,
            timeout = 1200,
            env = env,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("unsloth import timed out")
    if "PROBE_OK" not in proc.stdout and "AssertionError" not in proc.stderr:
        pytest.skip(f"unsloth could not be imported here:\n{proc.stderr[-2000:]}")
    assert "PROBE_OK" in proc.stdout, proc.stderr[-3000:]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
