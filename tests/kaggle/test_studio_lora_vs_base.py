# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The exported model against the base it came from.

`assert_gguf_export` proves a file was produced, carries the GGUF magic, loads
on the GPU and generates. Every one of those is ALSO true of an export that
silently merged nothing and shipped the base weights, which is the regression
worth catching and is invisible to file size, to the magic, and to "it
generated text".

Greedy decoding at temperature 0 makes it visible: identical weights answer
identically, so a difference is the adapter.

The determinism control is the part that makes that argument valid, and it runs
FIRST. If the same weights loaded twice do not reproduce their own answer, a
difference between two models is noise, and the assertion has to say it could
not compare rather than pass on it.

What is deliberately NOT asserted is the canary. Studio's training run is a
handful of steps, and whether that is enough to learn a specific string is a
property of the run length rather than of the export path: asserting it would
be a red about training tuning wearing an export label.
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


def _body(name: str = "assert_lora_vs_base") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def test_the_assertion_exists_and_is_driven_from_the_run():
    assert _body()
    assert "self.assert_lora_vs_base(" in _body("execute")


def test_it_compares_against_the_file_the_export_actually_made():
    """A path recomputed here could name a stale GGUF from an earlier run,
    which would compare the base against something this session never made."""
    body = _body("execute")
    assert 'entry["name"] == "gguf_export"' in body
    assert 'exported_gguf = entry.get("gguf")' in body


def test_the_claim_is_that_the_two_differ():
    func = _func("assert_lora_vs_base")
    guarded = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.If) and "tuned == base_one" in ast.unparse(n.test)
    ]
    assert guarded, (
        "nothing fails when the export answers exactly as the base does, "
        "which is what a no-op merge looks like"
    )


def test_the_determinism_control_runs_before_the_claim():
    """Ordering is the argument. A difference only means "adapter" once the
    same weights have been shown to reproduce themselves."""
    body = _body()
    control_at = body.index("base_one != base_two")
    claim_at = body.index("tuned == base_one")
    assert control_at < claim_at


def test_a_non_reproducing_base_is_reported_as_a_failure_not_a_pass():
    """ "I could not compare" and "they matched" are opposite outcomes. Letting
    the first through as a pass is the exact shape this directory keeps being
    caught by."""
    func = _func("assert_lora_vs_base")
    for node in ast.walk(func):
        if isinstance(node, ast.If) and "base_one != base_two" in ast.unparse(node.test):
            appends = [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append"
            ]
            assert appends, "a base that will not reproduce itself must fail"
            break
    else:
        raise AssertionError("there is no determinism control at all")


def test_the_canary_is_recorded_and_never_asserted():
    """It is a property of how many steps the training ran, not of the export
    path. Failing on it would put a training-tuning red under an export name."""
    func = _func("assert_lora_vs_base")
    for node in ast.walk(func):
        if isinstance(node, ast.If) and "CANARY" in ast.unparse(node.test):
            raise AssertionError(
                "the verdict branches on the canary, which makes this red "
                "whenever the training run is too short to learn it"
            )
    assert "canary_in_exported" in _body()


def test_a_missing_export_is_a_failure_rather_than_a_skip():
    body = _body()
    assert "if not gguf:" in body
    assert 'return self.record("lora_vs_base", False, detail)' in body


def test_both_models_are_driven_with_the_same_prompt():
    """Two different questions would differ for reasons that have nothing to
    do with the adapter."""
    func = _func("assert_lora_vs_base")
    prompts = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets)
    ]
    assert len(prompts) == 1, "the prompt must be built once and reused"
    assert (
        _body().count("self.chat(prompt)") == 1
    ), "one call site, reached by every arm, or the arms can drift apart"


def test_the_exported_gguf_picked_is_a_model_and_not_an_mmproj_sidecar():
    """Driven through the REAL `newest_gguf`, because this is a selection bug
    and a rule fed a path proves nothing about the selector.

    A vision export writes two files, and the projector is often the newer.
    Handing `Qwen3.5-2B.F16-mmproj.gguf` to llama.cpp as a model is not an
    error: the server starts, reports gpu_layers=-1, offloads nothing and still
    returns text. On kernel unsloth-probe-studio-full2-815a0c that failed both
    the export assertion and this one, as a GPU fallback that never happened.
    """
    import sys as _sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import time as _time  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    _sys.path.insert(0, str(ROOT / "tests" / "kaggle" / "studio_gpu"))
    from studio_client import newest_gguf  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = _Path(tmp)
        model = root / "Qwen3.5-2B.Q8_0.gguf"
        model.write_bytes(b"GGUF")
        _time.sleep(0.01)
        # Written LAST, so "newest" alone would pick it.
        sidecar = root / "Qwen3.5-2B.F16-mmproj.gguf"
        sidecar.write_bytes(b"GGUF")

        assert newest_gguf(root) == model

        # And with only a sidecar present, the answer is None rather than the
        # sidecar: no model was exported, and saying so is the point.
        sidecar_only = root / "sub"
        sidecar_only.mkdir()
        (sidecar_only / "x.F16-mmproj.gguf").write_bytes(b"GGUF")
        assert newest_gguf(sidecar_only) is None
