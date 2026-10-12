# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A missing `triton` must not stop `unsloth/_gpu_init.py`.

The PyPI `triton` project publishes no Windows wheel (Windows is served by the separate
`triton-windows` package), so a CPU only Windows install has no `triton` at all. The
module level `import triton` in `unsloth/_gpu_init.py` was unconditional, so `import
unsloth` there died with a bare `ModuleNotFoundError: No module named 'triton'` from a
line that only exists to resolve `libcuda_dirs` on CUDA hosts.

The absent module is simulated with `sys.modules["triton"] = None`, which is what the
import system already uses to mean "blocked": `import triton` then raises `ImportError`
without touching the filesystem, and it blocks `triton.*` submodules too.
"""

from __future__ import annotations

import ast
import functools
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"
GPU_INIT_NAME = os.path.join("unsloth", "_gpu_init.py")


def _run(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter so no module state leaks between cases."""
    path = [str(REPO_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output = True,
        text = True,
        env = dict(os.environ, PYTHONPATH = os.pathsep.join(path)),
        timeout = 1800,
    )


_IMPORT_WITHOUT_TRITON = """
    import sys, traceback
    for name in [key for key in sys.modules if key == "triton" or key.startswith("triton.")]:
        del sys.modules[name]
    sys.modules["triton"] = None          # the import system's own "blocked" marker
    frames = []
    try:
        import unsloth
    except BaseException as error:
        traceback.print_exc()
        frames = traceback.extract_tb(error.__traceback__)
        # Every frame that is itself an `import triton` statement, i.e. every place that
        # still treats Triton as mandatory on the `import unsloth` path.
        for frame in frames:
            if frame.line and "import triton" in frame.line:
                print("TRITON_FRAME", frame.filename, frame.lineno)
        print("INNERMOST", frames[-1].filename, frames[-1].lineno)
        print("UNSLOTH_IMPORT_FAILED")
    else:
        print("UNSLOTH_IMPORT_OK")
    # Did unsloth/_gpu_init.py run at all? Apple Silicon takes the MLX path and never
    # executes it, so the message this file checks for is not emitted there. A failed
    # import drops the module from sys.modules again, so the traceback counts too.
    ran = ("unsloth._gpu_init" in sys.modules) or any(
        "_gpu_init.py" in frame.filename for frame in frames
    )
    print("GPU_INIT_RAN", ran)
"""


@functools.cache
def _import_without_triton() -> subprocess.CompletedProcess:
    return _run(_IMPORT_WITHOUT_TRITON)


@functools.cache
def _unsloth_is_importable() -> bool:
    return _run("import unsloth").returncode == 0


def _needs_unsloth():
    if not _unsloth_is_importable():
        pytest.skip("unsloth is not importable in this environment")


def _module_level_triton_imports(tree: ast.Module) -> list[ast.Import]:
    """Every module level `import triton` that is not inside a `try`."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "triton" for alias in node.names):
            found.append(node)
    return found


def test_the_module_level_triton_import_is_inside_a_try():
    """Structure, so the guard cannot be removed silently.

    The bare statement at module scope is the defect: nothing below it can recover,
    because the exception escapes `unsloth/__init__.py` itself.
    """
    tree = ast.parse(GPU_INIT.read_text(encoding = "utf-8"))
    bare = _module_level_triton_imports(tree)
    assert not bare, (
        "unsloth/_gpu_init.py imports triton unguarded at module scope on line(s) "
        f"{[node.lineno for node in bare]}; PyPI triton has no Windows wheel, so this "
        "raises ModuleNotFoundError on a CPU only Windows install"
    )
    guarded = [
        node
        for handler in ast.walk(tree)
        if isinstance(handler, ast.Try)
        for node in handler.body
        if isinstance(node, ast.Import) and any(alias.name == "triton" for alias in node.names)
    ]
    assert guarded, "the guarded `import triton` is gone"


def test_the_guard_binds_triton_to_none_and_records_why():
    """`triton` and `TRITON_IMPORT_ERROR` are the contract the rest of the file reads."""
    source = GPU_INIT.read_text(encoding = "utf-8")
    assert "TRITON_IMPORT_ERROR" in source
    # Every later use of `triton` in this module has to tolerate None.
    tree = ast.parse(source)
    version_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "__version__"
        and isinstance(node.value, ast.Name)
        and node.value.id == "triton"
    ]
    assert version_calls, "triton.__version__ is no longer read, update this test"


def test_a_missing_triton_does_not_fail_in_gpu_init():
    """The behaviour: with triton blocked, nothing raises from `unsloth/_gpu_init.py`.

    This does not assert that `import unsloth` completes. The Triton kernel modules
    under `unsloth/kernels` and the zoo's compiler import Triton for their own reasons,
    so the import can still stop further down; what must no longer happen is stopping
    here, on a line whose only job is CUDA library resolution.
    """
    _needs_unsloth()
    result = _import_without_triton()
    combined = result.stdout + result.stderr
    triton_frames = [line for line in combined.splitlines() if line.startswith("TRITON_FRAME ")]
    offending = [line for line in triton_frames if GPU_INIT_NAME in line]
    assert not offending, (
        "importing unsloth without triton still raises at the `import triton` in "
        "_gpu_init.py:\n" + "\n".join(offending) + "\n\nfull output:\n" + combined[-4000:]
    )
    innermost = [line for line in combined.splitlines() if line.startswith("INNERMOST ")]
    assert GPU_INIT_NAME not in "".join(innermost), (
        "the import still dies inside _gpu_init.py:\n" + "\n".join(innermost)
    )


def test_a_missing_triton_is_reported_in_words():
    """A user on Windows has to be told which package to install."""
    _needs_unsloth()
    result = _import_without_triton()
    combined = (result.stdout + result.stderr).lower()
    if "gpu_init_ran true" not in combined:
        pytest.skip(
            "unsloth/_gpu_init.py does not run on this platform (Apple Silicon takes the MLX "
            "path), so the guard that emits this message never executes"
        )
    assert "triton" in combined
    assert "triton-windows" in combined, (
        "the message must name the package that provides Triton on Windows:\n" + combined[-4000:]
    )


def test_an_installed_triton_is_left_alone():
    """No behaviour change where triton imports: the module stays bound, no error recorded."""
    _needs_unsloth()
    probe = _run(
        """
        try:
            import triton  # noqa: F401
        except Exception:
            print("NO_TRITON")
        else:
            import unsloth  # noqa: F401
            from unsloth import _gpu_init
            print("BOUND", _gpu_init.triton is not None, _gpu_init.TRITON_IMPORT_ERROR)
        """
    )
    combined = probe.stdout + probe.stderr
    if "NO_TRITON" in combined:
        pytest.skip("triton is not installed in this environment")
    assert probe.returncode == 0, combined[-4000:]
    assert "BOUND True None" in combined, combined[-4000:]
