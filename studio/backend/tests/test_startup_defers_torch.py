# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: ``import main`` must not import torch, and the warm that replaces
it must be safe.

uvicorn binds the listening socket only after ``import main`` and the lifespan
have both finished, so anything either of them imports is time the login screen
does not exist. torch (plus the sympy/scipy/pandas/sklearn it drags in through
transformers) was about 5s of that on a GPU host, for data no request needs
before it is asked for. Four eager edges caused it:

  utils/models/model_config.py  _build_detection_sets() at module scope
  routes/models.py              from core.inference import get_inference_backend
  core/inference/orchestrator.py  from utils.hf_xet_fallback import DownloadStallError
  utils/datasets/raw_text.py    from datasets import Dataset  (annotation only)

All four are lazy now and hardware detection moved off the lifespan onto
utils/torch_warmup.py. These tests lock that in: a fresh interpreter for the
import invariant (importing in-process would measure an already-warm
sys.modules), plus unit coverage of the warm's idempotency and of the
single-detection guarantee the endpoints rely on.

CPU-only, no network, no GPU, no weights -- runs in the standard
studio-backend-ci matrix.
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

_HEAVY = ("torch", "transformers", "unsloth_zoo", "scipy", "sklearn", "sympy")

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
    assert proc.returncode == 0, (
        f"import main was not clean\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    )
    assert "IMPORT_MAIN_CLEAN" in proc.stdout


@pytest.mark.parametrize(
    "module_path",
    [
        "utils.models.model_config",
        "utils.datasets.raw_text",
        "core.inference.orchestrator",
        "routes.models",
    ],
)
def test_module_import_does_not_pull_torch(module_path: str):
    """Each module that used to force torch must import clean on its own."""
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
    """ensure_hardware_detected() collapses the warm thread and any request
    that arrives mid-detection into a single run."""
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


def test_the_warm_covers_every_package_import_main_used_to_pull():
    """Deferring an import only moves its cost if something still pays it.

    One stage per package `import main` used to leave in sys.modules. Dropping
    one does not fail any other test -- it just moves that import onto the first
    request that needs it, silently.
    """
    from utils import torch_warmup
    assert [name for name, _ in torch_warmup._STAGES] == [
        "hardware",  # torch, via utils.hardware
        "transformers",  # via model_config's registry read
        "datasets",  # via utils/datasets/raw_text.py
        "unsloth_zoo",  # via orchestrator's utils.hf_xet_fallback import
    ]


def test_the_unsloth_zoo_stage_goes_through_the_shim(monkeypatch):
    """The warm must reproduce the eager import, not just import the package.

    The edge it replaces was orchestrator.py's ``from utils.hf_xet_fallback
    import DownloadStallError``, and the shim retries under
    UNSLOTH_ZOO_DISABLE_GPU_INIT=1 when unsloth_zoo's GPU init raises. A bare
    ``import unsloth_zoo`` skips that retry: on a host with a bitsandbytes wheel
    that cannot find libcudart it raises where startup used to succeed.
    """
    import builtins

    from utils import hf_xet_fallback, torch_warmup

    monkeypatch.setattr(torch_warmup, "_torch_installed", lambda: True)
    calls = []
    monkeypatch.setattr(hf_xet_fallback, "_load_shared", lambda: (calls.append(1), True)[1])

    real_import = builtins.__import__

    def no_bare_zoo(name, *args, **kwargs):
        assert name != "unsloth_zoo", (
            "the warm imported unsloth_zoo directly; it must go through "
            "utils.hf_xet_fallback so the UNSLOTH_ZOO_DISABLE_GPU_INIT retry runs"
        )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_bare_zoo)
    torch_warmup._warm_unsloth_zoo()
    assert calls == [1]

    # A shim that could not load leaves the watchdog degraded, and that has to
    # surface as a failed stage rather than a silent success.
    monkeypatch.setattr(hf_xet_fallback, "_load_shared", lambda: False)
    with pytest.raises(RuntimeError, match = "unsloth_zoo unavailable"):
        torch_warmup._warm_unsloth_zoo()


def test_a_failing_warm_stage_is_reported_not_swallowed(monkeypatch, capsys):
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
    # structlog renders to stdout, not through the stdlib root handler caplog
    # attaches to, so assert on what the operator would actually see.
    assert "stage exploded" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The warm window: routes must not block uvicorn's loop while torch loads
# ---------------------------------------------------------------------------
#
# Detection used to finish before the socket bound, so get_device() was free by
# the time any request arrived and an `async def` could call it inline. It is
# not free any more: for the length of the warm it blocks on _DETECT_LOCK and
# the torch import. Called on the event loop that stalls every other request --
# measured at 1547ms on a /api/liveness that touches nothing.
#
# These are the first-paint and polled routes, i.e. the ones certain to land
# inside the warm window. Each must reach its blocking helper only through
# asyncio.to_thread.

_OFFLOAD_REQUIRED = [
    ("main.py", "get_gpu_visibility", "get_backend_visible_gpu_info"),
    ("routes/training.py", "get_hardware_utilization", "get_gpu_utilization"),
    ("routes/models.py", "list_models", "get_inference_backend"),
    ("routes/inference.py", "get_status", "get_inference_backend"),
    ("routes/inference.py", "get_api_monitor", "_monitor_active_model"),
    ("routes/inference.py", "get_api_monitor", "_monitor_context_length"),
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
    assert handed_off or called_inside, (
        f"{rel_path}:{func_name} no longer references {callee}; update this guard"
    )


def test_a_failed_detection_degrades_instead_of_raising():
    """A torch that raises must not leave DEVICE unset.

    The warm thread swallows stage failures, so a raising detection would leave
    DEVICE None -- and then every get_device() retries the same broken import
    (re-running torch/__init__ against the submodules a partial import left in
    sys.modules) and /api/health, which waits on this, answers 500.
    """
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


def test_a_broken_torch_purges_its_own_zombie(monkeypatch):
    """_has_torch() must clean up after the import it just watched fail."""
    import builtins
    import sys

    from utils.hardware import hardware as hw

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            sys.modules["torch._C"] = object()
            sys.modules.pop("torch", None)
            raise OSError("undefined symbol: cudaGetDeviceCount")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    saved = sys.modules.pop("torch", None)
    try:
        assert hw._has_torch() is False
        assert "torch._C" not in sys.modules
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        sys.modules.pop("torch._C", None)
