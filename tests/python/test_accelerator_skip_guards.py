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
_SPOOF = _TESTS_ROOT / "_zoo_aggressive_cuda_spoof.py"

_ALLOWED = {
    _SPOOF,
    _TESTS_ROOT / "_shared" / "real_accelerator.py",
    _TESTS_ROOT / "conftest.py",
}

# Every probe the spoof answers that a skip guard could plausibly ask. Listing only the
# is_available three left `skipif(torch.cuda.device_count() == 0)` reading as safe and
# evading this test; test_the_spoofed_probe_list_keeps_up_with_the_spoof keeps it in step.
_SPOOFED_CALLS = {
    ("torch", "cuda", "is_available"),
    ("torch", "xpu", "is_available"),
    ("torch", "accelerator", "is_available"),
    ("torch", "cuda", "device_count"),
    ("torch", "cuda", "is_initialized"),
    ("torch", "cuda", "is_bf16_supported"),
    ("torch", "cuda", "get_device_capability"),
    ("torch", "cuda", "get_device_name"),
    ("torch", "cuda", "get_device_properties"),
}

# The rest of what the spoof patches: things a test DOES, not things it asks. Listed so the
# meta-test below can insist every newly spoofed name lands in one bucket or the other.
_SPOOFED_NON_PREDICATES = {
    ("torch", "Tensor", "is_pinned"),
    ("torch", "Tensor", "pin_memory"),
    ("torch", "cuda", "Event"),
    ("torch", "cuda", "Stream"),
    ("torch", "cuda", "_is_in_bad_fork"),
    ("torch", "cuda", "_unsloth_consolidated_spoof"),
    ("torch", "cuda", "amp"),
    ("torch", "cuda", "cudart"),
    ("torch", "cuda", "current_device"),
    ("torch", "cuda", "current_stream"),
    ("torch", "cuda", "default_stream"),
    ("torch", "cuda", "empty_cache"),
    ("torch", "cuda", "get_rng_state"),
    ("torch", "cuda", "get_rng_state_all"),
    ("torch", "cuda", "initial_seed"),
    ("torch", "cuda", "manual_seed"),
    ("torch", "cuda", "manual_seed_all"),
    ("torch", "cuda", "nvtx"),
    ("torch", "cuda", "seed"),
    ("torch", "cuda", "seed_all"),
    ("torch", "cuda", "set_device"),
    ("torch", "cuda", "set_rng_state"),
    ("torch", "cuda", "set_rng_state_all"),
    ("torch", "cuda", "stream"),
    ("torch", "cuda", "synchronize"),
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


# has_real_accelerator() for a test that just needs somewhere to put a tensor, has_real_cuda()
# for one that names cuda. Which of the two suffices is per-namespace; see _satisfying_gates.
_REAL_PROBES = ("has_real_accelerator", "has_real_cuda")


def _is_real_accelerator_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _dotted(node.func)[-1:] in {(p,) for p in _REAL_PROBES}


def _is_negated_real_accelerator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and _is_real_accelerator_call(node.operand)
    )


def _gate_named(node: ast.AST, negated: bool) -> str:
    """The recorded-answer probe this conjunct gates on, or "" if it is not one."""
    if negated:
        if not (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
            return ""
        node = node.operand
    if not isinstance(node, ast.Call):
        return ""
    name = _dotted(node.func)[-1:]
    return name[0] if name and name[0] in _REAL_PROBES else ""


def _satisfying_gates(call: ast.Call) -> frozenset[str]:
    """Which recorded-answer probes are strong enough to dominate this probe.

    A torch.cuda.* call needs has_real_cuda(). has_real_accelerator() is true on an XPU-only
    or Ascend NPU-only host, so it lets a CUDA-only API through to be evaluated on a machine
    with no CUDA at all, where it raises or hands back the spoof's answer. That is the same
    defect as gating a CUDA-only test on the broad probe, one level up in the decorator.
    """
    if _dotted(call.func)[:2] == ("torch", "cuda"):
        return frozenset({"has_real_cuda"})
    return frozenset(_REAL_PROBES)


def _unguarded_spoofed_calls(node: ast.AST, gates: frozenset[str] = frozenset()):
    """Spoofable probes in `node` whose value the spoof is still free to decide.

    Python's `and` / `or` short-circuit, so a probe is safe once an earlier conjunct in the
    same chain has settled the case it would otherwise be asked about:

        skipif(not has_real_cuda() or torch.cuda.device_count() < 2)
        skipif(has_real_cuda() and torch.cuda.get_device_capability()[0] >= 12)

    `gates` carries the probes already established at this point in the chain. Which of them
    suffices depends on the namespace being called into; see `_satisfying_gates`.
    """
    if isinstance(node, ast.BoolOp):
        negated = not isinstance(node.op, ast.And)
        seen = gates
        for value in node.values:
            yield from _unguarded_spoofed_calls(value, seen)
            named = _gate_named(value, negated)
            if named:
                seen = seen | {named}
        return
    if (
        isinstance(node, ast.Call)
        and _dotted(node.func) in _SPOOFED_CALLS
        and not (gates & _satisfying_gates(node))
    ):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _unguarded_spoofed_calls(child, gates)


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
            for call in _unguarded_spoofed_calls(decorator):
                offenders.append(
                    f"{path.relative_to(_TESTS_ROOT)}:{call.lineno}: "
                    f"{'.'.join(_dotted(call.func))}()"
                )
    assert not offenders, (
        "these skip guards read a probe tests/_zoo_aggressive_cuda_spoof.py patches to "
        "True process-wide, so they un-skip on a CPU-only box whenever they share a "
        "pytest session with tests/version_compat or tests/vllm_compat. Gate on "
        "`from real_accelerator import has_real_accelerator` first, either alone or as the "
        "conjunct that short-circuits ahead of the torch call:\n  " + "\n  ".join(offenders)
    )


def test_the_spoofed_probe_list_keeps_up_with_the_spoof():
    """Every name the spoof patches has to be classified, or this guard rots quietly.

    _SPOOFED_CALLS started as the three is_available probes while the spoof was already
    answering device_count, is_initialized, is_bf16_supported and get_device_capability. A
    skip guard reading any of those was un-skipped by the spoof and invisible to the test
    above. Adding a patch to the spoof now fails here until it is called a predicate a skip
    guard could read, or plumbing no guard would."""
    spoof = ast.parse(_SPOOF.read_text(encoding = "utf-8"))
    patched = set()
    for node in ast.walk(spoof):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = _dotted(target)
            if name[:1] == ("torch",) and len(name) >= 3:
                patched.add(name)

    classified = _SPOOFED_CALLS | _SPOOFED_NON_PREDICATES
    unclassified = sorted(".".join(n) for n in patched - classified)
    assert not unclassified, (
        "the spoof patches these and this file does not say what they are. Put each in "
        "_SPOOFED_CALLS if a skip guard could read it, or _SPOOFED_NON_PREDICATES if no "
        "guard would:\n  " + "\n  ".join(unclassified)
    )

    # Nothing listed may have quietly stopped being spoofed, or the guard above now rejects
    # an idiom that has become safe.
    stale = sorted(".".join(n) for n in (classified - patched) if n[1] == "cuda")
    assert not stale, (
        "these are listed as spoofed but the spoof no longer patches them:\n  " + "\n  ".join(stale)
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

    from real_accelerator import has_real_accelerator, has_real_cuda

    before = has_real_accelerator()
    before_cuda = has_real_cuda()

    import _zoo_aggressive_cuda_spoof as spoof

    spoof.apply()

    import torch

    print("SPOOF_PROBE " + json.dumps({{
        "before": before,
        "after": has_real_accelerator(),
        "before_cuda": before_cuda,
        "after_cuda": has_real_cuda(),
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
    assert verdict["after_cuda"] == verdict["before_cuda"], (
        "has_real_cuda() moved after the spoof was applied. It is the gate for the tests that "
        f"name cuda, so it is the one the spoof most directly targets: {verdict}"
    )
    assert not (verdict["before_cuda"] and not verdict["before"]), (
        "has_real_cuda() is true where has_real_accelerator() is false, so the narrow probe is "
        f"no longer a subset of the broad one: {verdict}"
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
        call for decorator in _skipif_calls(tree) for call in _unguarded_spoofed_calls(decorator)
    ]
    assert len(found) == 1


def _offenders_in(condition: str) -> list[str]:
    tree = ast.parse(
        "import pytest\nimport torch\n"
        "from real_accelerator import has_real_accelerator, has_real_cuda\n\n\n"
        f"@pytest.mark.skipif({condition}, reason = 'x')\n"
        "def test_x():\n    pass\n"
    )
    return [
        ".".join(_dotted(call.func))
        for decorator in _skipif_calls(tree)
        for call in _unguarded_spoofed_calls(decorator)
    ]


@pytest.mark.parametrize(
    "condition",
    [
        # Both polarities, and nested one level down.
        "not has_real_cuda() or torch.cuda.device_count() < 2",
        "has_real_cuda() and torch.cuda.get_device_capability()[0] >= 12",
        "not has_real_cuda() or (torch.cuda.device_count() < 2 or x)",
        "has_real_cuda() and not torch.cuda.is_bf16_supported()",
        # Outside the torch.cuda namespace the broad probe is the matching one.
        "not has_real_accelerator() or not torch.xpu.is_available()",
        "has_real_accelerator() and torch.accelerator.is_available()",
    ],
)
def test_a_guard_short_circuited_on_the_real_probe_is_accepted(condition):
    assert _offenders_in(condition) == []


@pytest.mark.parametrize(
    "condition",
    [
        # The gate is present but does not dominate the call, so the spoof still decides.
        ("torch.cuda.device_count() < 2 or not has_real_accelerator()", "gate comes after"),
        ("has_real_accelerator() or torch.cuda.device_count() < 2", "and-gate under or"),
        ("not has_real_accelerator() and torch.cuda.device_count() < 2", "or-gate under and"),
        ("torch.cuda.device_count() < 2", "no gate at all"),
        ("x if has_real_cuda() else torch.cuda.device_count() < 2", "not a bool chain"),
        # The broad probe is true on an XPU-only host, so it does not settle whether
        # torch.cuda is there to answer.
        ("not has_real_accelerator() or torch.cuda.device_count() < 2", "broad gate on cuda"),
        (
            "has_real_accelerator() and torch.cuda.get_device_capability()[0] >= 12",
            "broad gate on cuda",
        ),
    ],
)
def test_a_gate_that_does_not_short_circuit_is_still_an_offender(condition):
    expression, why = condition
    assert _offenders_in(expression), f"missed: {why}"


_OTHER_ACCELERATOR_DEVICES = ("xpu", "npu", "mps", "hpu")


def _names_a_cuda_device(node: ast.AST) -> list[int]:
    """Lines where the body asks for a cuda device by name.

    Only the device string itself, so a test that merely mentions cuda in a message is not
    swept up: `device = "cuda"`, `.to("cuda:1")`, `torch.device("cuda")`.

    A body that also names xpu or npu as a device picks its device at run time and is not
    CUDA-only, whatever it calls the CUDA branch. tests/utils/test_packing.py is the case:
    it falls through to torch.device("xpu"), and _build_packed_training_setup has an xpu
    dtype arm, so narrowing its gate would have dropped real XPU coverage.
    """
    strings = [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]
    if any(
        s == other or s.startswith(other + ":")
        for s in strings
        for other in _OTHER_ACCELERATOR_DEVICES
    ):
        return []
    return [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and (child.value == "cuda" or child.value.startswith("cuda:"))
    ]


def _gated_only_on_the_broad_probe(func: ast.AST) -> bool:
    probes = {
        _dotted(call.func)[-1]
        for decorator in func.decorator_list
        if isinstance(decorator, ast.Call) and _dotted(decorator.func)[-1:] == ("skipif",)
        for call in ast.walk(decorator)
        if isinstance(call, ast.Call) and _dotted(call.func)[-1:] in {(p,) for p in _REAL_PROBES}
    }
    return probes == {"has_real_accelerator"}


def test_no_cuda_only_test_is_gated_on_the_broad_accelerator_probe():
    """has_real_accelerator() is true on an XPU-only or Ascend NPU-only host.

    A test that allocates on "cuda" and gates on it therefore un-skips on those machines and
    dies in torch, or, when the decorator itself calls into torch.cuda, raises during
    collection. Unsloth supports both backends, so this is a host somebody runs. Caught on
    tests/test_stopping_criteria_device.py, which named cuda three times behind the broad gate.
    """
    offenders = []
    for path in _python_test_files():
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _gated_only_on_the_broad_probe(node):
                continue
            for lineno in _names_a_cuda_device(node):
                offenders.append(f"{path.relative_to(_TESTS_ROOT)}:{lineno}: in {node.name}")
    assert not offenders, (
        "these tests ask for a cuda device but gate on has_real_accelerator(), which is also "
        "true on an XPU-only or Ascend NPU-only host. Gate them on has_real_cuda() instead:"
        "\n  " + "\n  ".join(offenders)
    )


def test_the_cuda_gate_scanner_would_catch_a_regression():
    """Not vacuous, and not over-broad: the narrow gate and a non-device string both pass."""

    def offenders(source):
        tree = ast.parse(source)
        return [
            lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and _gated_only_on_the_broad_probe(node)
            for lineno in _names_a_cuda_device(node)
        ]

    head = (
        "import pytest\n"
        "import torch\n"
        "from real_accelerator import has_real_accelerator, has_real_cuda\n\n\n"
    )
    bad = "@pytest.mark.skipif(not has_real_accelerator(), reason = 'x')\ndef test_x():\n    torch.tensor([1], device = 'cuda')\n"
    assert len(offenders(head + bad)) == 1

    fixed = bad.replace("has_real_accelerator", "has_real_cuda")
    assert offenders(head + fixed) == []

    # A cuda mention that is not a device string must not be swept up.
    prose = "@pytest.mark.skipif(not has_real_accelerator(), reason = 'x')\ndef test_x():\n    assert True, 'cuda is not required here'\n"
    assert offenders(head + prose) == []

    # Nor may a test that picks its own device: naming cuda in one branch and xpu in another
    # makes it device-generic, and narrowing its gate would drop the XPU coverage it has.
    generic = (
        "@pytest.mark.skipif(not has_real_accelerator(), reason = 'x')\n"
        "def test_x():\n"
        "    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('xpu')\n"
    )
    assert offenders(head + generic) == []
