# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CPU training fallback: `import unsloth` with no accelerator loads the CPU backend; a GPU host never does."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_INIT = _ROOT / "unsloth" / "__init__.py"
_GPU_INIT = _ROOT / "unsloth" / "_gpu_init.py"

# Modules the CPU backend must never load: GPU kernels, compiled model patches, the GPU bootstrap.
_GPU_ONLY = (
    "unsloth._gpu_init",
    "unsloth.models",
    "unsloth.kernels",
    "unsloth_zoo.temporary_patches",
    "unsloth_zoo.compiler",
    "unsloth_zoo.fused_losses",
    "unsloth_zoo.gradient_checkpointing",
)
_CPU_ONLY = ("unsloth._cpu_init", "unsloth_zoo.cpu")

_PROBE = r"""
import json, sys
import unsloth
if unsloth.DEVICE_TYPE == "cpu":
    from unsloth.trainer import UnslothTrainer
mods = sorted(sys.modules)
print("PROBE " + json.dumps({
    "device_type": getattr(unsloth, "DEVICE_TYPE", None),
    "fast": [c.__name__ for c in unsloth.FastLanguageModel.__mro__],
    "modules": mods,
    "trainer_file": getattr(sys.modules.get("unsloth.trainer"), "__file__", None),
}))
"""


def _import_unsloth(env_update):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_FORCE_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")
    }
    env.update(env_update)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output = True,
        text = True,
        env = env,
        timeout = 1200,
    )
    for line in out.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:]), out
    return None, out


def _loaded(modules, prefixes):
    return [m for m in modules if any(m == p or m.startswith(p + ".") for p in prefixes)]


def _zoo_has_cpu_backend():
    import importlib.util
    spec = importlib.util.find_spec(
        "unsloth_zoo"
    )  # locates only; the package __init__ does not run
    return bool(spec and spec.submodule_search_locations) and any(
        (Path(p) / "cpu" / "loader.py").is_file() for p in spec.submodule_search_locations
    )


@pytest.mark.skipif(not _zoo_has_cpu_backend(), reason = "unsloth_zoo without the CPU backend")
def test_no_accelerator_loads_the_cpu_backend():
    torch = pytest.importorskip("torch")
    for name in ("xpu", "npu"):
        try:
            if getattr(torch, name).is_available():
                pytest.skip(f"{name} host: hiding CUDA leaves that accelerator selected")
        except Exception:
            pass
    # UNSLOTH_FORCE_GPU_PATH keeps an Apple Silicon host off the MLX backend.
    probe, out = _import_unsloth({"CUDA_VISIBLE_DEVICES": "", "UNSLOTH_FORCE_GPU_PATH": "1"})
    if probe is None and "ROCm" in out.stderr:
        pytest.skip("ROCm runtime host without a ROCm torch keeps its install hint")
    assert probe is not None, out.stderr[-3000:]
    assert probe["device_type"] == "cpu"
    assert "FastCPUModel" in probe["fast"]
    assert _loaded(probe["modules"], _GPU_ONLY) == []
    assert _loaded(probe["modules"], _CPU_ONLY)
    assert (
        probe["trainer_file"] is None
    ), "unsloth.trainer must be the CPU shim, not unsloth/trainer.py"


def test_gpu_host_never_loads_the_cpu_backend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    probe, out = _import_unsloth({})
    assert probe is not None, out.stderr[-3000:]
    assert probe["device_type"] in ("cuda", "hip")
    assert _loaded(probe["modules"], _CPU_ONLY) == []


def test_cpu_handoff_sits_before_any_gpu_setup():
    """_gpu_init raises the handoff right after `import unsloth_zoo` and before `import torch`."""
    tree = ast.parse(_GPU_INIT.read_text(encoding = "utf-8"))
    raises = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and getattr(n.exc.func, "id", None) == "_CPUBackendSelected"
    ]
    assert len(raises) == 1
    zoo_import = min(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Import) and any(a.name == "unsloth_zoo" for a in n.names)
    )
    first_torch = min(
        n.lineno
        for n in tree.body
        if isinstance(n, ast.Try)
        and any(
            isinstance(s, ast.Import) and any(a.name == "torch" for a in s.names) for s in n.body
        )
    )
    assert zoo_import < raises[0].lineno < first_torch


def test_cpu_init_only_reached_from_the_handoff():
    tree = ast.parse(_INIT.read_text(encoding = "utf-8"))
    handlers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ExceptHandler) and getattr(n.type, "id", None) == "_CPUBackendSelected"
    ]
    assert len(handlers) == 1, "unsloth/__init__.py must catch _CPUBackendSelected exactly once"
    inside = [
        n
        for n in ast.walk(handlers[0])
        if isinstance(n, ast.ImportFrom) and n.module == "_cpu_init" and n.level == 1
    ]
    everywhere = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "_cpu_init" and n.level == 1
    ]
    assert inside and len(inside) == len(everywhere), "_cpu_init must only load from the handoff"


def test_cpu_backend_query_only_reads_an_already_imported_zoo(monkeypatch):
    """unsloth.device_type asks zoo about the CPU backend without importing it: an import from there
    reorders zoo's own initialisation under harnesses that preload device_type (CUDA spoof in the
    test suite), which sent Qwen3.5's linear attention down the Triton path on CPU runners."""
    import builtins
    import types

    src = (_ROOT / "unsloth" / "device_type.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_zoo_cpu_backend"
    )
    namespace = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), "device_type.py", "exec"), namespace)
    query = namespace["_zoo_cpu_backend"]
    imports = []
    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *a, **k: imports.append(name) or real_import(name, *a, **k),
    )
    monkeypatch.delitem(sys.modules, "unsloth_zoo.device_type", raising = False)
    assert query("cpu_backend_forced") is False
    assert not any(name.startswith("unsloth_zoo") for name in imports)
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.device_type",
        types.SimpleNamespace(cpu_backend_forced = lambda: True),
    )
    assert query("cpu_backend_forced") is True
    assert query("cpu_backend_allowed") is False  # an older zoo without the function


@pytest.mark.parametrize("entry", ["studio/backend/main.py", "studio/backend/run.py"])
def test_studio_keeps_training_off_on_cpu_only_hosts(entry):
    """Studio / Desktop stay chat-only without a GPU: both entry points opt out of the CPU training backend
    at module level, before anything can import unsloth, and spawned training workers inherit it."""
    tree = ast.parse((_ROOT / entry).read_text(encoding = "utf-8"))
    opt_out = first_unsloth_import = None
    for node in tree.body:
        line = ast.unparse(node)
        if opt_out is None and "UNSLOTH_DISABLE_CPU_FALLBACK" in line and "'1'" in line:
            opt_out = node.lineno
        if first_unsloth_import is None and isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(n == "unsloth" or n.startswith("unsloth.") for n in names):
                first_unsloth_import = node.lineno
    assert opt_out is not None, f"{entry} no longer sets UNSLOTH_DISABLE_CPU_FALLBACK=1"
    assert first_unsloth_import is None or opt_out < first_unsloth_import
