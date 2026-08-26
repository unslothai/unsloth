# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: ``import main`` must not import torch or pandas, and the warm that
replaces torch must be safe.

uvicorn binds the socket only after ``import main`` and the lifespan both finish, so
anything they import is time the login screen does not exist. torch and what it drags
in was about 5s of that on a GPU host. Four eager edges caused it:

  utils/models/model_config.py  _build_detection_sets() at module scope
  routes/models.py              from core.inference import get_inference_backend
  core/inference/orchestrator.py  from utils.hf_xet_fallback import DownloadStallError
  utils/datasets/raw_text.py    from datasets import Dataset  (annotation only)

pandas arrived by a fifth, through the data-recipe seed route:

  routes/data_recipe/seed.py    from data_designer_unstructured_seed.chunking import ...
  ...chunking.py                import pandas as pd  at module scope
  ...__init__.py                re-exports .config and .impl, which import the data
                                designer engine, which imports pandas and pyarrow

Importing the submodule runs the package first, so dropping only the chunking-level
import left the cost in place. The route resolves the plugin on first use now. The
Startup profile workflow measured this edge at 2.247s of a 7.284s ``import main`` on
windows-latest, 901ms self on macos-15.

All of them are lazy. A fresh interpreter is used for the import invariant, since
importing in-process would measure an already-warm sys.modules. CPU-only, no network,
no GPU, no weights.

The runtime guards only bite where the optional plugin is installed, so a source-level
guard covers the environments that do not have it.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend

_HEAVY = ("torch", "transformers", "unsloth_zoo", "scipy", "sklearn", "sympy", "pandas", "pyarrow")

_IMPORT_MAIN_SNIPPET = r"""
import sys

# Never start the warm here: it would import the very modules under test, and
# the assertion below would race it.
import os
os.environ["UNSLOTH_STUDIO_DISABLE_TORCH_WARM"] = "1"

import main  # noqa: F401

HEAVY = %(heavy)r
leaked = sorted(
    m for m in sys.modules
    if any(m == h or m.startswith(h + ".") for h in HEAVY)
)
assert not leaked, "import main pulled heavy ML modules: %%s" %% (leaked,)
print("IMPORT_MAIN_CLEAN")
""" % {"heavy": list(_HEAVY)}


def _run(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd = str(_BACKEND_DIR),
        capture_output = True,
        text = True,
        timeout = 900,
    )


def test_import_main_does_not_import_torch():
    """`import main` must leave torch and its scientific stack unimported."""
    proc = _run(_IMPORT_MAIN_SNIPPET)
    assert (
        proc.returncode == 0
    ), f"import main was not clean\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    assert "IMPORT_MAIN_CLEAN" in proc.stdout


@pytest.mark.parametrize(
    "module_path",
    [
        "utils.models.model_config",
        "utils.datasets.raw_text",
        "core.inference.orchestrator",
        "routes.models",
        "utils.hf_xet_fallback",
        "core.rag.embeddings",
        "routes.data_recipe.seed",
    ],
)
def test_module_import_does_not_pull_torch(module_path: str):
    """Each module that used to force torch or pandas must import clean on its own."""
    snippet = (
        "import sys, importlib\n"
        f"importlib.import_module({module_path!r})\n"
        f"HEAVY = {list(_HEAVY)!r}\n"
        "leaked = sorted(m for m in sys.modules"
        " if any(m == h or m.startswith(h + '.') for h in HEAVY))\n"
        "assert not leaked, leaked\n"
        "print('CLEAN')\n"
    )
    proc = _run(snippet)
    assert proc.returncode == 0, f"{module_path}: {proc.stderr[-3000:]}"
    assert "CLEAN" in proc.stdout


def _module_scope_imports(tree: ast.Module) -> list[ast.stmt]:
    """Imports that run on `import <module>`: module body and nesting like try/if,
    but nothing inside a function or class."""
    found: list[ast.stmt] = []
    stack: list[ast.stmt] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        else:
            stack.extend(c for c in ast.iter_child_nodes(node) if isinstance(c, ast.stmt))
    return found


@pytest.mark.parametrize(
    ("rel_path", "banned"),
    [("routes/data_recipe/seed.py", "data_designer_unstructured_seed")],
)
def test_module_scope_does_not_import_the_seed_plugin(rel_path: str, banned: str):
    """The runtime guards above pass vacuously wherever the plugin is not installed,
    which is most CI jobs. This one reads the source, so it holds either way."""
    tree = ast.parse((_BACKEND_DIR / rel_path).read_text(encoding = "utf-8"))
    offenders = [
        node.lineno
        for node in _module_scope_imports(tree)
        if (getattr(node, "module", None) or "").startswith(banned)
        or any(alias.name.startswith(banned) for alias in node.names)
    ]
    assert not offenders, (
        f"{rel_path} imports {banned} at module scope (line(s) {offenders}). The package "
        "re-exports the data designer engine, so this puts pandas and pyarrow back into "
        "main's startup graph. Resolve it on first use instead."
    )


def test_detection_sets_still_resolve_under_their_old_names():
    """The PEP 562 shim must keep `from ... import _VLM_MODEL_TYPES` working."""
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")
    from utils.models.model_config import (
        _AUDIO_ONLY_MODEL_TYPES,
        _VISION_CHECK_INLINE_HELPERS,
        _VISION_CHECK_SCRIPT,
        _VLM_CLASS_NAMES,
        _VLM_MODEL_TYPES,
    )

    assert "llava" in _VLM_MODEL_TYPES
    assert {"csm", "whisper"} <= _AUDIO_ONLY_MODEL_TYPES
    assert _VLM_CLASS_NAMES
    assert "_VLM_MODEL_TYPES = " in _VISION_CHECK_INLINE_HELPERS
    assert _VISION_CHECK_INLINE_HELPERS in _VISION_CHECK_SCRIPT


def test_detection_sets_are_built_once_under_concurrency():
    """Two requests racing the warm must not each read the registry."""
    from utils.models import model_config as mc

    calls = []
    original = mc._build_detection_sets
    saved = mc._DETECTION_SETS
    mc._DETECTION_SETS = None
    try:

        def counted():
            calls.append(1)
            return (frozenset({"x"}), frozenset(), frozenset())

        mc._build_detection_sets = counted
        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(mc._detection_sets())

        threads = [threading.Thread(target = worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(calls) == 1, f"built {len(calls)} times"
        assert all(r is results[0] for r in results)
    finally:
        mc._build_detection_sets = original
        mc._DETECTION_SETS = saved


def test_hardware_is_detected_once_under_concurrency():
    """ensure_hardware_detected() collapses the warm thread and any request arriving
    mid-detection into a single run."""
    from utils.hardware import hardware as hw

    calls = []
    saved_device, saved_impl = hw.DEVICE, hw._detect_hardware_locked
    hw.DEVICE = None
    try:

        def counted():
            calls.append(1)
            hw.DEVICE = hw.DeviceType.CPU
            return hw.DEVICE

        hw._detect_hardware_locked = counted
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            hw.ensure_hardware_detected()

        threads = [threading.Thread(target = worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert len(calls) == 1, f"detected {len(calls)} times"
    finally:
        hw._detect_hardware_locked = saved_impl
        hw.DEVICE = saved_device


def test_warm_starts_once_and_honours_the_kill_switch(monkeypatch):
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_thread", None, raising = False)
    monkeypatch.setenv(torch_warmup.DISABLE_ENV_VAR, "1")
    assert torch_warmup.start_background_warm() is False
    assert torch_warmup.warm_status()["started"] is False

    monkeypatch.delenv(torch_warmup.DISABLE_ENV_VAR)
    monkeypatch.setattr(torch_warmup, "_STAGES", (("noop", lambda: None),))
    assert torch_warmup.start_background_warm() is True
    # Second call is a no-op: one warm thread per process.
    assert torch_warmup.start_background_warm() is False
    assert torch_warmup.join_background_warm(60) is True
    status = torch_warmup.warm_status()
    assert status["finished"] is True
    assert status["stages"]["noop"]["ok"] is True


def test_the_warm_keeps_optional_gpu_consumers_cold():
    """Startup prepares hardware and metadata, but first-use integrations stay lazy."""
    from utils import torch_warmup

    stage_names = [name for name, _ in torch_warmup._STAGES]
    assert stage_names == [
        "hardware",  # torch, via utils.hardware
        # Builds the metadata-only orchestrator after hardware detection.
        "inference_backend",
        "transformers",  # model_config registry read
        "datasets",  # raw-text dataset helpers
    ]
    assert "unsloth_zoo" not in stage_names
    assert not hasattr(
        torch_warmup, "_warm_unsloth_zoo"
    ), "the optional Hub/Xet integration must be loaded by a real download, not startup"


def test_a_failing_warm_stage_is_reported_not_swallowed(monkeypatch, capsys, caplog):
    """A broken stage must be visible in the log and must not kill the warm."""
    from utils import torch_warmup

    def boom():
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(torch_warmup, "_thread", None, raising = False)
    monkeypatch.delenv(torch_warmup.DISABLE_ENV_VAR, raising = False)
    monkeypatch.setattr(torch_warmup, "_STAGES", (("boom", boom), ("after", lambda: None)))
    assert torch_warmup.start_background_warm() is True
    assert torch_warmup.join_background_warm(60) is True

    status = torch_warmup.warm_status()
    assert status["stages"]["boom"]["ok"] is False
    assert "stage exploded" in status["stages"]["boom"]["error"]
    # The stage after the failure still ran, and the process is still alive.
    assert status["stages"]["after"]["ok"] is True
    assert status["finished"] is True
    # The operator has to be able to grep it. Which sink structlog is bound to depends
    # on what configured logging earlier, so accept stdout or the stdlib records.
    logged = capsys.readouterr().out + "\n".join(r.getMessage() for r in caplog.records)
    assert "stage exploded" in logged


# ---------------------------------------------------------------------------
# The warm window: routes must not block uvicorn's loop while torch loads
# ---------------------------------------------------------------------------
#
# get_device() used to be free by the time any request arrived. For the length of the
# warm it now blocks on _DETECT_LOCK and the torch import, stalling every other request
# on the loop (1547ms measured on a /api/liveness that touches nothing).
#
# First-paint and polled routes, certain to land inside the warm window. Each must
# reach its blocking helper only through asyncio.to_thread.

_OFFLOAD_REQUIRED = [
    ("main.py", "get_gpu_visibility", "get_backend_visible_gpu_info"),
    ("routes/training.py", "get_hardware_utilization", "get_gpu_utilization"),
    # Not first-paint, but lands in the warm window when a start is submitted early,
    # and its MLX streaming guard forces detection itself.
    ("routes/training.py", "start_training", "ensure_hardware_detected"),
    ("routes/export.py", "_ensure_export_supported", "export_capability"),
    ("routes/models.py", "list_models", "get_inference_backend"),
    # get_status is deliberately absent. It used to offload get_inference_backend; it
    # now peeks, which constructs nothing and so needs no offload at all. The stronger
    # invariant lives in test_async_singleton_access.py::
    # test_read_only_endpoints_never_construct_the_singleton, which fails if it goes
    # back to building. Listing it here would require the offload it no longer needs.
    ("routes/inference.py", "get_api_monitor", "_monitor_active_model"),
    ("routes/inference.py", "get_api_monitor", "_monitor_context_length"),
    # Also not first-paint, but a load or validate carrying gpu_ids lands in the warm
    # window and this probes the device before any teardown.
    ("routes/inference.py", "_resolve_gguf_gpu_ids_for_request", "get_device"),
]


def _find_function(tree: ast.AST, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no function {name!r} in the parsed module")


def _is_to_thread(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
    )


def _offloaded_nodes(func: ast.AST) -> set[int]:
    """Every node lexically inside an asyncio.to_thread(...) argument."""
    offloaded: set[int] = set()
    for node in ast.walk(func):
        if not _is_to_thread(node):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                offloaded.add(id(sub))
    return offloaded


@pytest.mark.parametrize("rel_path, func_name, callee", _OFFLOAD_REQUIRED)
def test_first_paint_routes_do_not_block_the_event_loop(rel_path, func_name, callee):
    source = (_BACKEND_DIR / rel_path).read_text(encoding = "utf-8")
    func = _find_function(ast.parse(source), func_name)
    assert isinstance(func, ast.AsyncFunctionDef), (
        f"{rel_path}:{func_name} is no longer `async def`; this guard assumes it runs on "
        "the loop (a plain `def` handler is already safe -- drop it from the list)"
    )
    offloaded = _offloaded_nodes(func)

    # The bare-name form: asyncio.to_thread(callee).
    handed_off = any(
        isinstance(n, ast.Name) and n.id == callee and id(n) in offloaded for n in ast.walk(func)
    )
    direct = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == callee
        and id(n) not in offloaded
    ]
    assert not direct, (
        f"{rel_path}:{func_name} calls {callee}() directly on the event loop at line(s) "
        f"{[n.lineno for n in direct]}; it blocks on hardware detection until the "
        "startup warm has imported torch. Wrap it in await asyncio.to_thread(...)."
    )
    called_inside = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == callee
        and id(n) in offloaded
        for n in ast.walk(func)
    )
    assert (
        handed_off or called_inside
    ), f"{rel_path}:{func_name} no longer references {callee}; update this guard"


def test_a_failed_detection_degrades_instead_of_raising():
    """A torch that raises must not leave DEVICE unset. The warm swallows stage failures, so a
    raising detection would leave DEVICE None, and then every get_device() retries the same
    broken import while /api/health, which waits on this, answers 500."""
    from utils.hardware import hardware as hw

    saved_device, saved_impl = hw.DEVICE, hw._detect_hardware_locked
    saved_chat, saved_reason = hw.CHAT_ONLY, hw.CHAT_ONLY_REASON
    hw.DEVICE = None
    calls = []
    try:

        def boom():
            calls.append(1)
            raise OSError("libcudart.so.12: cannot open shared object file")

        hw._detect_hardware_locked = boom
        assert hw.ensure_hardware_detected() == hw.DeviceType.CPU
        assert hw.CHAT_ONLY is True
        assert hw.CHAT_ONLY_REASON == "detection_failed"
        # Cached: the second call must not re-enter the failing import.
        assert hw.ensure_hardware_detected() == hw.DeviceType.CPU
        assert len(calls) == 1, f"retried the broken import {len(calls)} times"
    finally:
        hw._detect_hardware_locked = saved_impl
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = saved_device, saved_chat, saved_reason


def test_a_broken_torch_counts_as_no_torch(monkeypatch):
    """_has_torch() must not let a non-ImportError escape into detection."""
    import builtins

    from utils.hardware import hardware as hw

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("undefined symbol: cudaGetDeviceCount")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert hw._has_torch() is False


def test_purge_partial_import_clears_the_zombie_and_leaves_live_ones():
    """A half-imported package must not survive into the retry."""
    import sys

    from utils.torch_warmup import purge_partial_import

    sys.modules["zzz_fake_pkg.sub"] = object()
    sys.modules["zzz_fake_pkg.sub.deep"] = object()
    try:
        # Parent absent + submodules present: the zombie signature.
        assert sorted(purge_partial_import("zzz_fake_pkg")) == [
            "zzz_fake_pkg.sub",
            "zzz_fake_pkg.sub.deep",
        ]
        assert not [m for m in sys.modules if m.startswith("zzz_fake_pkg")]

        # Parent present: a healthy (or still-importing) package is left alone.
        sys.modules["zzz_fake_pkg"] = object()
        sys.modules["zzz_fake_pkg.sub"] = object()
        assert purge_partial_import("zzz_fake_pkg") == []
        assert "zzz_fake_pkg.sub" in sys.modules
    finally:
        for name in [m for m in list(sys.modules) if m.startswith("zzz_fake_pkg")]:
            sys.modules.pop(name, None)


def test_purge_declines_when_a_compiled_submodule_is_loaded():
    """Evicting a loaded C extension is worse than the zombie it would fix. Re-importing one
    re-runs its module init, and pybind11 calls std::terminate on a second registration of
    the same type."""
    import sys
    from importlib.machinery import EXTENSION_SUFFIXES
    from types import ModuleType

    from utils.torch_warmup import purge_partial_import

    compiled = ModuleType("zzz_ext_pkg.binding")
    compiled.__file__ = "/nowhere/zzz_ext_pkg/binding" + EXTENSION_SUFFIXES[0]
    sys.modules["zzz_ext_pkg.binding"] = compiled
    sys.modules["zzz_ext_pkg.pure"] = ModuleType("zzz_ext_pkg.pure")
    try:
        assert purge_partial_import("zzz_ext_pkg") == []
        # Both survive: a half-purged package is not a state worth creating.
        assert "zzz_ext_pkg.binding" in sys.modules
        assert "zzz_ext_pkg.pure" in sys.modules
    finally:
        for name in [m for m in list(sys.modules) if m.startswith("zzz_ext_pkg")]:
            sys.modules.pop(name, None)


def test_a_broken_torch_purges_its_own_zombie(monkeypatch):
    """_has_torch() must clean up after the import it just watched fail. Restores the real
    torch entries exactly: leaving `torch` in sys.modules with its submodules evicted aborts
    the pytest process the next time anything imports one."""
    import builtins
    import sys
    from types import ModuleType

    from utils.hardware import hardware as hw

    real_import = builtins.__import__
    saved = {name: mod for name, mod in sys.modules.items() if name.split(".")[0] == "torch"}

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            # Zombie signature: `torch/__init__` raised after executing a pure-Python
            # submodule, leaving it behind with the parent evicted.
            sys.modules["torch._early"] = ModuleType("torch._early")
            sys.modules.pop("torch", None)
            raise OSError("undefined symbol: cudaGetDeviceCount")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for name in saved:
        sys.modules.pop(name, None)
    try:
        assert hw._has_torch() is False
        assert "torch._early" not in sys.modules
    finally:
        for name in [m for m in list(sys.modules) if m.split(".")[0] == "torch"]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_one_detection_pass_probes_torch_once(monkeypatch):
    """A detection pass must import torch at most once.

    _has_torch() is the expensive part: a broken wheel takes seconds to fail, and the purge
    then declines because a compiled submodule is loaded, so the partial tree stays and a
    second probe re-runs torch/__init__ against those cache hits -- same cost again, with no
    guarantee it fails the same way. The CUDA branch and the XPU fallback share one probe."""
    import builtins
    from types import ModuleType

    from utils.hardware import hardware as hw

    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] == "torch"}
    saved_device = hw.DEVICE
    saved_chat, saved_reason = hw.CHAT_ONLY, hw.CHAT_ONLY_REASON
    real_import = builtins.__import__
    attempts = []

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            attempts.append(name)
            # A loaded compiled submodule makes the purge decline, so a retry re-runs
            # torch/__init__ in full.
            ext = ModuleType("torch._C")
            ext.__file__ = "/nonexistent/torch/_C.cpython-313-x86_64-linux-gnu.so"
            sys.modules["torch._C"] = ext
            sys.modules.pop("torch", None)
            raise OSError("libcudart.so.12: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    for name in saved:
        sys.modules.pop(name, None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        hw.DEVICE = None
        assert hw._detect_hardware_locked() == hw.DeviceType.CPU
        assert len(attempts) == 1, (
            f"detection imported torch {len(attempts)} times in one pass; the "
            "CUDA branch and the XPU fallback must share a single probe"
        )
    finally:
        monkeypatch.undo()
        for name in [m for m in list(sys.modules) if m.split(".")[0] == "torch"]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = saved_device, saved_chat, saved_reason


def test_every_importing_warm_stage_purges_on_failure():
    """A failed stage must not leave a half-imported package behind. An import that dies
    partway leaves its submodules cached under an evicted parent, so the retry returns a
    package that imports but is missing attributes: broken until restart, while
    warm_status() reports nothing worse than a cold stage."""
    from utils import torch_warmup

    importing = {name for name, _ in torch_warmup._STAGES} - {"inference_backend"}
    assert importing <= set(torch_warmup._STAGE_PACKAGE), (
        "a warm stage imports a package with no purge mapping: "
        f"{sorted(importing - set(torch_warmup._STAGE_PACKAGE))}"
    )


def test_a_failed_stage_actually_purges(monkeypatch):
    purged = []
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "purge_partial_import", lambda pkg: purged.append(pkg))

    def _boom():
        raise RuntimeError("datasets exploded partway")

    torch_warmup._run_stage("datasets", _boom)

    assert purged == ["datasets"], f"expected the datasets purge, got {purged}"
    assert torch_warmup._status["stages"]["datasets"]["ok"] is False


def test_a_stage_that_imports_nothing_is_not_purged(monkeypatch):
    """inference_backend builds an object; there is no package to clean up."""
    purged = []
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "purge_partial_import", lambda pkg: purged.append(pkg))

    def _boom():
        raise RuntimeError("constructor exploded")

    torch_warmup._run_stage("inference_backend", _boom)

    assert purged == [], f"nothing to purge for a non-importing stage, got {purged}"
