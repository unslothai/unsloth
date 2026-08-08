"""A broken xformers must be LOUD on the default path (NVIDIA P0-1).

Before this, ``unsloth/models/_utils.py`` silenced the xformers logger to ERROR right
before the import and then reported the failure only under ``UNSLOTH_ENABLE_LOGGING`` --
so a wheel built for a different torch dropped the user to SDPA attention with no output
at all, which is how a cu128 wheel shipped beside a cu130 runtime unnoticed.

Runs the real ``import unsloth`` in a subprocess against a stub xformers package that is
put first on ``sys.path``, so nothing here depends on the xformers actually installed --
the stub ships its own ``.dist-info`` as well as its own ``cpp_lib.json``, because
``import unsloth`` reads both.

The child also loads ``tests/conftest.py``, the same GPU-free harness pytest applies to
this directory, so the subprocess can finish the import on a host with no accelerator
exactly like the parent process can. See ``_import_unsloth_with``.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch


# A stub that imports fine and reports a version, then fails exactly where the real one
# does: torch.ops.load_library inside _register_extensions.
_STUB_VERSION = "0.0.34"
_STUB_INIT = f"__version__ = {_STUB_VERSION!r}\n"
_STUB_CPP_LIB = textwrap.dedent(
    """\
    def _register_extensions():
        raise OSError("[WinError 126] The specified module could not be found")
    """
)


def _write_stub_xformers(root: Path, built_torch: str, built_cuda: int) -> None:
    package = root / "xformers"
    package.mkdir()
    (package / "__init__.py").write_text(_STUB_INIT, encoding = "utf-8")
    (package / "_cpp_lib.py").write_text(_STUB_CPP_LIB, encoding = "utf-8")
    (package / "cpp_lib.json").write_text(
        json.dumps(
            {
                "version": {
                    "cuda": built_cuda,
                    "hip": None,
                    "torch": built_torch,
                    "python": "3.10.11",
                },
                "env": {"XFORMERS_PACKAGE_FROM": f"wheel-v{_STUB_VERSION}"},
            }
        ),
        encoding = "utf-8",
    )
    # A .dist-info beside the package, so the stub is a complete install rather than just
    # an importable module. `import unsloth` asks importlib.metadata for the xformers
    # version twice - unsloth/import_fixes.py:fix_xformers_performance_issue() and
    # unsloth/models/_utils.py - and a module without metadata makes both of those read
    # the HOST's xformers instead, which is precisely what this file's docstring promises
    # not to do. On a host with no xformers at all there is nothing to read and
    # importlib.metadata.version() raises PackageNotFoundError straight out of
    # fix_xformers_performance_issue(), killing the child before it reaches the subject
    # matter. Writing the metadata makes the stub self-sufficient in both cases.
    dist_info = root / f"xformers-{_STUB_VERSION}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: xformers\nVersion: {_STUB_VERSION}\n",
        encoding = "utf-8",
    )


def _import_unsloth_with(stub_root, repo_root: Path, enable_logging: str) -> str:
    """Import unsloth in a child, optionally with a stub xformers shadowing the real one."""
    inject = (
        [f"sys.path.insert(0, {str(stub_root)!r})"] if stub_root is not None else []
    )
    # Only meaningful with the stub: without one this asks the host for a version it may
    # legitimately not have.
    dist_probe = (
        [
            "from importlib.metadata import version as _dist_version",
            'print("XFORMERS_DIST_VERSION:", _dist_version("xformers"))',
        ]
        if stub_root is not None
        else []
    )
    # tests/conftest.py is the GPU-free harness pytest already applies to everything under
    # tests/: on a host with no accelerator it forces DEVICE_TYPE to "cuda" and stubs the
    # torch.cuda probes unsloth fires at import time (get_device_capability,
    # is_bf16_supported, mem_get_info). The parent pytest process gets it for free, a bare
    # `python -c` child does not, and that asymmetry is the whole reason these two tests
    # were the only red ones on the CPU-only CI runners: unsloth/_gpu_init.py calls
    # torch.cuda.get_device_capability() at module scope and the child died on
    # "Found no NVIDIA driver" before reaching any xformers code. Loaded by path under a
    # private name so it neither shadows nor is shadowed by the injected stub, and on a
    # real accelerator it is a no-op.
    harness = repo_root / "tests" / "conftest.py"
    assert harness.is_file(), f"GPU-free test harness missing at {harness}"
    # Assembled line by line rather than through a dedented f-string: an interpolated
    # multi-line fragment lands at column 0 and makes textwrap.dedent a no-op, which turns
    # the whole child into an IndentationError.
    code = "\n".join(
        ["import importlib.util, sys"]
        + inject
        + [
            "_spec = importlib.util.spec_from_file_location(",
            f"    '_unsloth_gpu_free_harness', {str(harness)!r}",
            ")",
            "_harness = importlib.util.module_from_spec(_spec)",
            "_spec.loader.exec_module(_harness)",
            "import unsloth  # noqa: F401",
            "from unsloth.models._utils import XFORMERS_BROKEN_REASON, xformers",
            'print("REASON:", XFORMERS_BROKEN_REASON)',
            'print("XFORMERS_IS_NONE:", xformers is None)',
        ]
        + dist_probe
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd = str(repo_root),
        capture_output = True,
        text = True,
        timeout = 900,
        env = {
            **os.environ,
            "UNSLOTH_ENABLE_LOGGING": enable_logging,
            # Pin one device rather than inheriting a list, so the child cannot pick a
            # different GPU than the one this process was given (it never allocates on it -
            # the stub xformers fails first). The harness above covers the case where the
            # named device does not exist.
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        },
    )
    assert result.returncode == 0, f"import unsloth failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout + result.stderr


@pytest.fixture(scope = "module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mismatched_build():
    """A (torch, cuda-int) pair one CUDA major away from whatever this host runs.

    The host's torch is whatever the test env has, so pick the build relative to it
    instead of hardcoding cu128/cu130 and hoping.
    """
    running_major = int((torch.version.cuda or "12").split(".", 1)[0])
    built_cuda = 1300 if running_major != 13 else 1208
    built_torch = (
        f"{torch.__version__.split('+')[0]}+cu{built_cuda // 100}{built_cuda % 100:d}"
    )
    return built_torch, built_cuda


@pytest.mark.skipif(
    torch.version.cuda is None,
    reason = "the mismatch is synthesized as a CUDA-major difference, which needs a CUDA "
    "torch to compare against; on ROCm/XPU/CPU there is no running CUDA major",
)
def test_broken_xformers_warns_without_enable_logging(tmp_path, repo_root):
    built_torch, built_cuda = _mismatched_build()
    _write_stub_xformers(tmp_path, built_torch, built_cuda)

    output = _import_unsloth_with(tmp_path, repo_root, enable_logging = "0")

    # The stub really is what unsloth inspected, metadata included - not the host's
    # xformers leaking in through importlib.metadata.
    assert f"XFORMERS_DIST_VERSION: {_STUB_VERSION}" in output
    # Loud on the default path: this is the whole point.
    assert "Xformers is installed but its optimized kernels cannot load" in output
    # Actionable: names what it was built for and what is actually running.
    assert built_torch in output
    assert "3.10.11" in output
    assert torch.__version__ in output
    # And it must actually fall back rather than half-work.
    assert "XFORMERS_IS_NONE: True" in output
    assert "REASON: None" not in output


def test_warning_is_printed_once(capsys):
    # In-process, because a fresh subprocess reaches the announcement once no matter what:
    # the subprocess version of this test passed with the once-guard deleted.
    from unsloth.models import _utils

    _utils._XFORMERS_BREAKAGE_ANNOUNCED = False
    try:
        _utils._announce_xformers_breakage("first", build_mismatch = True)
        _utils._announce_xformers_breakage("second", build_mismatch = True)
        printed = capsys.readouterr().out
    finally:
        _utils._XFORMERS_BREAKAGE_ANNOUNCED = False
    assert printed.count("Xformers is installed but its optimized kernels cannot load") == 1
    assert "second" not in printed


def test_a_non_mismatch_failure_is_not_relabelled_as_a_build_mismatch(capsys):
    # This arm also catches the sm_100/110/120 FA3 guard and the old-torch guards, whose
    # messages are multi-line, fenced and already actionable. Reflowing one of those into
    # "its optimized kernels cannot load ... install the matching build" states the wrong
    # cause and mangles the text.
    from unsloth.models import _utils

    original = (
        "Unsloth: Xformers 0.0.32.post2 has a broken FA3 dispatch on SM 12.0 GPUs.\n"
        "```\npip install ninja\n```\n"
    )
    _utils._XFORMERS_BREAKAGE_ANNOUNCED = False
    try:
        _utils._announce_xformers_breakage(original, build_mismatch = False)
        printed = capsys.readouterr().out
    finally:
        _utils._XFORMERS_BREAKAGE_ANNOUNCED = False
    assert "its optimized kernels cannot load" not in printed
    assert "--force-reinstall" not in printed
    # Verbatim: the fences and the newlines are part of a copy-pasteable instruction.
    assert original in printed


@pytest.mark.skipif(
    torch.version.cuda is None, reason = "needs a CUDA torch to have a matching xformers"
)
@pytest.mark.skipif(
    importlib.util.find_spec("xformers") is None,
    reason = "no xformers is installed on this host (importlib.util.find_spec('xformers') "
    "is None), so there is no healthy install for `import unsloth` to stay quiet about; "
    "the not-installed path is a different branch of unsloth/models/_utils.py and is "
    "covered by test_xformers_compat.py",
)
def test_a_healthy_install_says_nothing(repo_root):
    # The other half of "never cry wolf": on a machine where xformers works, importing
    # unsloth must print none of this. Uses the real installed xformers, and skips when
    # that one is itself broken -- then the warning is correct, not a false positive.
    # The skipif above already established xformers IS installed, so xformers is None here
    # can only mean the import of it failed, never that there was nothing to import.
    output = _import_unsloth_with(None, repo_root, enable_logging = "0")
    if "XFORMERS_IS_NONE: True" in output:
        pytest.skip("this host's xformers is genuinely broken, so a warning is correct")
    assert "Xformers is installed but" not in output
    assert "REASON: None" in output
