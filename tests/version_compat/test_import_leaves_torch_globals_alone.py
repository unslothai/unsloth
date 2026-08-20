# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The CPU-only fake-train modules must not spoof torch for the whole session.

`test_trl_fake_train_cpu.py` and `test_trl_padding_free_max_length.py` pretend
this process has no GPU: eager `torch.compile`, `torch.accelerator.is_available`
answering False, and every `device = "cuda"` allocation rewritten to CPU. Done at
module scope that outlives the module, and pytest imports every selected file
during collection, so a GPU test running later in the same process gets CPU
tensors out of `torch.randn(..., device = "cuda")`, its CUDA autocast never
applies, and it passes without having tested anything. CI runners have no GPU, so
they skip such tests and never notice; local GPU verification is what breaks.

Both checks below fail against the import-time version of those patches.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
_GUARDED = ("test_trl_fake_train_cpu.py", "test_trl_padding_free_max_length.py")

# Imported in a subprocess: the whole point is that importing these modules must
# not touch torch, and doing it in this process would be the damage itself.
_PROBE = r"""
import importlib.util, sys
from pathlib import Path

import torch

before = {
    "compile": torch.compile,
    "Tensor.to": torch.Tensor.to,
    "Tensor.cuda": torch.Tensor.cuda,
}
if hasattr(torch, "accelerator"):
    before["accelerator.is_available"] = torch.accelerator.is_available
has_cuda = torch.cuda.is_available()

for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as e:  # a module-level pytest.skip is not an error here
        print("skipped %s: %s" % (path, type(e).__name__))

failures = []
for name, original in before.items():
    obj = torch
    for part in name.split(".")[:-1]:
        obj = getattr(obj, part)
    if getattr(obj, name.split(".")[-1]) is not original:
        failures.append("torch.%s was replaced" % name)

if has_cuda:
    # The allocator rewrite is the one that makes a GPU test pass vacuously.
    if not torch.randn(4, 4, device = "cuda").is_cuda:
        failures.append('torch.randn(device = "cuda") returned a CPU tensor')
    if not torch.zeros(4).to("cuda").is_cuda:
        failures.append('Tensor.to("cuda") returned a CPU tensor')

print("FAILURES:" + "|".join(failures))
sys.exit(1 if failures else 0)
"""


def test_importing_the_cpu_modules_leaves_torch_globals_alone():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed; there is no global state to protect")

    proc = subprocess.run(
        [sys.executable, "-B", "-c", _PROBE, *[str(_HERE / name) for name in _GUARDED]],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


def _module_level_nodes(node):
    """Everything that runs on import: loops and try blocks included, function
    and class bodies excluded, since those only run when something calls them."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from _module_level_nodes(child)


def _rooted_at_torch(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == "torch"


def test_no_version_compat_module_patches_torch_at_import_time():
    """The static half, so a new module cannot reintroduce the leak.

    Cheap and GPU-free, unlike the subprocess above, and it covers every file in
    the directory rather than the two known offenders.
    """
    offenders = []
    for path in sorted(_HERE.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        for node in _module_level_nodes(ast.parse(path.read_text(encoding = "utf-8"))):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and node.args
            ):
                targets = [node.args[0]]
            for target in targets:
                if isinstance(target, (ast.Attribute, ast.Name)) and _rooted_at_torch(target):
                    offenders.append(f"{path.name}:{node.lineno} {ast.unparse(target)}")

    assert not offenders, (
        "patch torch from a fixture instead, so it is undone for the rest of the "
        f"session: {offenders}"
    )
