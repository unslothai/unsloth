# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""No test may gate hardware on a call this repo's own harnesses spoof.

`tests/_zoo_aggressive_cuda_spoof.py` sets `torch.cuda.is_available` to return
True and leaves it there. It is applied at module import by everything under
`tests/version_compat` and `tests/vllm_compat`. pytest imports every selected
module before running anything and evaluates `@pytest.mark.skipif` at import, so
one spoofing module in a session decides the guard for every module collected
after it, whatever directory it lives in.

A guard written as `skipif(not torch.cuda.is_available())` therefore un-skips on
a CPU-only box as soon as it shares a session with one of those files, and the
test dies inside torch with `RuntimeError: Cannot access accelerator device when
none is available` -- a message about torch, from a cause that is neither torch
nor the test.

CI is only clear of this because the three jobs covering those directories each
`--ignore` the others. Nothing enforced that, and dropping one `--ignore` would
have reopened it silently, so enforce the guard side instead:
`tests/_shared/real_accelerator.py` records the answer before any spoof runs.

Scanned with ast rather than grep so a reformatted decorator, a multi-line
`skipif(...)` or a `# noqa` cannot walk past it.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_TESTS_ROOT = Path(__file__).resolve().parents[1]

# The spoof itself, the recorded-answer helper and conftest are what implement the
# real probe. They are the only places allowed to call the raw torch API for it.
_ALLOWED = {
    _TESTS_ROOT / "_zoo_aggressive_cuda_spoof.py",
    _TESTS_ROOT / "_shared" / "real_accelerator.py",
    _TESTS_ROOT / "conftest.py",
}

_SPOOFED_CALLS = {
    ("torch", "cuda", "is_available"),
    ("torch", "xpu", "is_available"),
    ("torch", "accelerator", "is_available"),
}

# The other half of the hazard above: a skip guard decides whether a test RUNS, these
# decide where its tensors LAND, because the library reads the same spoofed probe itself.
# An allow-list, not a general lint: catch another instance of this, not every loader call.
_DEVICE_INFERRING_CALLS = {
    ("PeftModel", "from_pretrained"): "torch_device",
    ("PeftMixedModel", "from_pretrained"): "torch_device",
    ("load_peft_weights",): "device",
}


def _dotted(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return ()


def _skipif_calls(tree: ast.AST):
    """Every `pytest.mark.skipif(...)` decorator call in the module."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if _dotted(decorator.func)[-1:] == ("skipif",):
                yield decorator


def _python_test_files():
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if path in _ALLOWED:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_skip_guard_reads_a_spoofable_accelerator_probe():
    offenders = []
    for path in _python_test_files():
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except SyntaxError:
            # tests/python/ holds fixtures that are deliberately unparseable on this
            # interpreter (the 3.9 floor checks). Not our business.
            continue
        for decorator in _skipif_calls(tree):
            for call in ast.walk(decorator):
                if isinstance(call, ast.Call) and _dotted(call.func) in _SPOOFED_CALLS:
                    offenders.append(
                        f"{path.relative_to(_TESTS_ROOT)}:{call.lineno}: "
                        f"{'.'.join(_dotted(call.func))}()"
                    )
    assert not offenders, (
        "these skip guards read a probe tests/_zoo_aggressive_cuda_spoof.py patches to "
        "True process-wide, so they un-skip on a CPU-only box whenever they share a "
        "pytest session with tests/version_compat or tests/vllm_compat. Use "
        "`from real_accelerator import has_real_accelerator` instead:\n  " + "\n  ".join(offenders)
    )


def _device_inferring_calls(tree: ast.AST):
    """Every call in the module that lets a library resolve the device from the spoof."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        for suffix, keyword in _DEVICE_INFERRING_CALLS.items():
            if name[-len(suffix) :] == suffix:
                given = {kw.arg for kw in node.keywords if kw.arg}
                if keyword not in given:
                    yield node, ".".join(name), keyword


def test_no_test_lets_a_loader_infer_the_device_from_a_spoofed_probe():
    offenders = []
    for path in _python_test_files():
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except SyntaxError:
            continue
        for node, name, keyword in _device_inferring_calls(tree):
            offenders.append(
                f"{path.relative_to(_TESTS_ROOT)}:{node.lineno}: {name}() with no {keyword}="
            )
    assert not offenders, (
        "these calls let the library pick the device, and it picks it off a probe "
        "tests/_zoo_aggressive_cuda_spoof.py patches to True process-wide. On a CPU-only "
        "runner the load then dispatches to a CUDA backend that is not there. Pass the "
        "device: `torch_device = 'cuda' if has_real_accelerator() else 'cpu'`, or plain "
        "'cpu' when the test does not care:\n  " + "\n  ".join(offenders)
    )


def test_the_device_scanner_would_catch_a_regression(tmp_path):
    """Not vacuous: the forbidden shape must trip it and the fixed shape must not."""
    bad = ast.parse(
        "from peft import PeftModel\n\n\n"
        "def test_x(base, path):\n    PeftModel.from_pretrained(base, path)\n"
    )
    assert len(list(_device_inferring_calls(bad))) == 1

    good = ast.parse(
        "from peft import PeftModel\n\n\n"
        "def test_x(base, path):\n"
        "    PeftModel.from_pretrained(base, path, torch_device = 'cpu')\n"
    )
    assert list(_device_inferring_calls(good)) == []


_SURVIVES_THE_SPOOF_PROBE = textwrap.dedent(
    """
    import json, sys

    sys.path.insert(0, {tests_root!r})
    sys.path.insert(0, {shared_dir!r})

    from real_accelerator import has_real_accelerator

    before = has_real_accelerator()

    import _zoo_aggressive_cuda_spoof as spoof

    spoof.apply()

    import torch

    print("SPOOF_PROBE " + json.dumps({{
        "before": before,
        "after": has_real_accelerator(),
        "spoof_patched_is_available": bool(torch.cuda.is_available()),
    }}))
    """
)


def test_the_recorded_answer_survives_the_spoof():
    """The property the helper exists for, asserted rather than assumed.

    In a SUBPROCESS, and that is not incidental. `spoof.apply()` sets
    `torch.cuda.is_available` to return True and never restores it, so calling it
    in-process poisons the rest of the xdist worker: every later test on that
    worker sees a machine with a CUDA card that is not there. peft's
    `infer_device()` is one of the things that reads it, so
    tests/test_save_lora_without_vllm.py then asks safetensors to load onto CUDA
    and dies with

        NotImplementedError: Could not run 'aten::empty_strided' with arguments
        from the 'CUDA' backend

    which is the exact class of cross-test damage this file exists to prevent. An
    earlier version of this test did apply the spoof in-process and caused that
    failure; it reproduces deterministically with just this file and that one, in
    that order.

    Same subprocess pattern as tests/vllm_compat/test_unsloth_zoo_imports.py
    (#10855), for the same reason: a question about global state has to be asked
    somewhere the answer cannot leak back.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _SURVIVES_THE_SPOOF_PROBE.format(
                tests_root = str(_TESTS_ROOT),
                shared_dir = str(_TESTS_ROOT / "_shared"),
            ),
        ],
        capture_output = True,
        text = True,
        timeout = 300,
        cwd = str(_TESTS_ROOT.parent),
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"the spoof probe failed to run:\n{combined[-4000:]}"

    marker = "SPOOF_PROBE "
    line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
    assert line is not None, f"probe produced no verdict:\n{combined[-4000:]}"
    verdict = json.loads(line[len(marker) :])

    assert verdict["spoof_patched_is_available"] is True, (
        "the spoof no longer patches torch.cuda.is_available, so this test is not "
        "checking anything; re-point it at whatever it patches now"
    )
    assert verdict["after"] == verdict["before"], (
        "has_real_accelerator() moved after the spoof was applied, which is the whole "
        f"thing it is supposed to be immune to: {verdict}"
    )


def test_this_file_never_applies_the_spoof_in_process():
    """The regression guard for the bug the test above used to be.

    Applying the spoof in-process is invisible here and fails somewhere else
    entirely, on whichever test the xdist scheduler happens to put next on the
    same worker. So pin it structurally rather than trusting it to stay fixed.
    """
    tree = ast.parse(Path(__file__).read_text(encoding = "utf-8"))
    offenders = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted(node.func)[-2:] == ("spoof", "apply")
    ]
    assert not offenders, (
        "this file calls spoof.apply() in-process. That leaves torch.cuda.is_available "
        "patched for every later test on this xdist worker. Ask it in a subprocess "
        f"instead, as _SURVIVES_THE_SPOOF_PROBE does: {offenders}"
    )


@pytest.mark.parametrize(
    "probe", ["torch.cuda.is_available", "torch.xpu.is_available", "torch.accelerator.is_available"]
)
def test_the_scanner_would_catch_a_regression(probe, tmp_path):
    """The scan is not vacuous: hand it the shape it forbids and it must object."""
    offending = tmp_path / "test_offending.py"
    offending.write_text(
        "import pytest\nimport torch\n\n\n"
        f"@pytest.mark.skipif(not {probe}(), reason = 'needs a GPU')\n"
        "def test_x():\n    pass\n",
        encoding = "utf-8",
    )
    tree = ast.parse(offending.read_text(encoding = "utf-8"))
    found = [
        call
        for decorator in _skipif_calls(tree)
        for call in ast.walk(decorator)
        if isinstance(call, ast.Call) and _dotted(call.func) in _SPOOFED_CALLS
    ]
    assert len(found) == 1
