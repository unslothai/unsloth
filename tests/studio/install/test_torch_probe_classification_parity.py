# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared torch probe must classify exactly as the five probes it replaced.

Consolidating those probes moved their classification out of a subprocess `-c` string
and into ordinary Python in the repair paths. That is meant to be a translation and
nothing more, but a translation is precisely the kind of change that can be subtly
wrong while every existing test still passes, because the existing tests feed the
repair paths a *mocked* probe answer and therefore exercise the new derivation only,
never the old one.

So this compares the two directly. The old expressions are reproduced verbatim from
the merge base as reference implementations, cited by line. The new derivations are
pulled out of the live module with `ast` rather than copied, so they cannot drift from
what actually ships: if someone edits the derivation, this test reads the edit. If
someone renames the locals it asserts on, extraction fails loudly, which is the right
outcome, because a rename means the equivalence needs re-checking rather than assuming.

Both sides then run over the same matrix of torch states and must agree on every one.

Scope, stated honestly. This proves the classification is a faithful translation. It
does not prove the memoisation is safe, which is a separate property resting on
`pip_install` / `pip_install_try` being the only things that change the installed
torch, and it does not exercise real AMD, Intel or Windows hosts.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"

_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_parity_probe", _STACK_PATH
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

_SOURCE = _STACK_PATH.read_text(encoding = "utf-8")
_TREE = ast.parse(_SOURCE, str(_STACK_PATH))


# The torch states the classification has to agree on.
# Each is (torch.__version__, torch.version.hip, torch.version.cuda) as the probe reports them.
_TORCH_STATES = [
    ("2.9.1+cu128", "", "12.8"),
    ("2.7.1+cu118", "", "11.8"),
    ("2.11.0+cu130", "", "13.0"),
    ("2.10.0+cu126", "", "12.6"),
    ("2.11.0", "", "13.0"),
    ("2.9.1", "", "12.8"),
    ("2.10.0+rocm7.1", "7.1.12345", ""),
    ("2.9.1+rocm6.3", "6.3.42134", ""),
    ("2.11.0+rocm7.2", "7.14.60850", ""),
    ("2.9.1+rocm6.4", "", ""),
    ("2.10.0+rocmsdk20250901", "", ""),
    ("2.6.0+xpu", "", ""),
    ("2.9.1+xpu", "", ""),
    ("2.10.0+xpu", "", ""),
    ("2.5.1+xpu", "", ""),
    ("2.11.0+xpu", "", ""),
    ("3.0.0+xpu", "", ""),
    ("2.9.1+cpu", "", ""),
    ("2.10.0", "", ""),
    ("2.9.1", "", ""),
    ("", "", ""),
    ("2.9.1", "7.1", "12.8"),  # both set: hip must win Case.
    ("2.9.1+ROCM6.4", "", ""),
    ("2.10.0+XPU", "", ""),
    ("2.9.1+CU128", "", ""),
    ("2.10.0+ROCMSDK20250901", "", ""),
]


def _fn(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_STACK_PATH.name}")


def _run_assignments(fn_name, wanted, env):
    """Execute the live assignments for `wanted`, in source order, against `env`.

    Straight-line derivations over the probe's outputs, so running them outside their
    guards is faithful as long as the guard variables are bound in env.
    """
    found = set()
    for node in ast.walk(_fn(fn_name)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not any(n in wanted for n in names):
            continue
        if node.value is None:
            continue
        exec(compile(ast.Module([node], []), "<live>", "exec"), env)  # noqa: S102
        found.update(n for n in names if n in wanted)
    missing = set(wanted) - found
    assert not missing, (
        f"{fn_name}: could not extract {sorted(missing)} from the live source. "
        f"If these were renamed, the equivalence needs re-checking rather than assuming."
    )
    return env


def _if_test_containing(fn_name, needle, env):
    """Evaluate the live `if` condition that contains `needle`."""
    for node in ast.walk(_fn(fn_name)):
        if isinstance(node, ast.If) and needle in ast.unparse(node.test):
            return eval(compile(ast.Expression(node.test), "<live>", "eval"), env)  # noqa: S307
    raise AssertionError(f"{fn_name}: no `if` test containing {needle!r}")


# Reference implementations:
def _old_cuda_fields(ver, hip, cuda):
    """merge base studio/install_python_stack.py:2339-2346 (_ensure_cuda_torch)."""
    ver = ver.lower()
    m = re.search(r"\+(cu\d+)", ver)
    marker = "hip" if (hip or "rocm" in ver) else ("cuda" if cuda else "cpu")
    return (
        marker,
        m.group(1) if m else "",
        ver.split("+", 1)[0],
        ("cu" + cuda.replace(".", "")) if cuda else "",
    )


def _old_cpu_is_gpu(ver, hip, cuda):
    """merge base :2773-2780 (_ensure_cpu_torch)."""
    ver = ver.lower()
    return (
        bool(hip)
        or "rocm" in ver
        or bool(cuda)
        or bool(re.search(r"\+cu\d+", ver))
        or "+xpu" in ver
    )


def _old_xpu_ok(ver, hip, cuda):
    """merge base :2442-2447 (_ensure_xpu_torch)."""
    ver = ver.lower()
    rel = ver.split("+")[0].split(".")
    n = tuple(int(x) for x in rel[:2] if x.isdigit())
    return "+xpu" in ver and len(n) == 2 and (2, 6) <= n < (2, 11)


def _old_rocm_marker(ver, hip, cuda):
    """merge base :3023-3027 (_ensure_rocm_torch)."""
    ver = ver.lower()
    return hip if hip else ("rocm" if "rocm" in ver else "")


def _old_windows_rocm_yes(ver, hip, cuda):
    """merge base :382-385 (_installed_torch_is_windows_rocm)."""
    ver = ver.lower()
    return bool(hip or "rocm" in ver or "rocmsdk" in ver)


@pytest.mark.parametrize(
    "ver,hip,cuda", _TORCH_STATES, ids = [s[0] or "empty" for s in _TORCH_STATES]
)
class TestClassificationIsAFaithfulTranslation:
    def test_cuda_marker_tag_release_and_runtime_family(self, ver, hip, cuda):
        env = {"re": re, "_version": ver, "_hip": hip, "_cuda": cuda}
        _run_assignments(
            "_ensure_cuda_torch",
            {"_ver", "_cu_match", "_marker", "_installed_cu", "_installed_release", "_runtime_cu"},
            env,
        )
        new = (env["_marker"], env["_installed_cu"], env["_installed_release"], env["_runtime_cu"])
        assert new == _old_cuda_fields(ver, hip, cuda)

    def test_cpu_gpu_predicate(self, ver, hip, cuda):
        # _TORCH_RUNTIME_XPU is a fourth input the merge-base predicate did not have, so
        # the equivalence is claimed with it EMPTY: over everything the old one could
        # see, the two still agree. Its own effect is pinned separately below, because a
        # reference that cannot model it cannot be asked about it.
        env = {"re": re, "_version": ver, "_hip": hip, "_cuda": cuda, "_TORCH_RUNTIME_XPU": ""}
        _run_assignments("_ensure_cpu_torch", {"_ver", "_is_gpu_build"}, env)
        assert env["_is_gpu_build"] == _old_cpu_is_gpu(ver, hip, cuda)

    def test_the_xpu_runtime_marker_is_the_one_deliberate_divergence(self, ver, hip, cuda):
        """An untagged source, conda or private-index XPU wheel carries its runtime only
        in torch.version.xpu. The old predicate read it as CPU and declined to reinstall,
        which is the whole reason the marker was added."""
        env = {
            "re": re,
            "_version": ver,
            "_hip": hip,
            "_cuda": cuda,
            "_TORCH_RUNTIME_XPU": "20250101",
        }
        _run_assignments("_ensure_cpu_torch", {"_ver", "_is_gpu_build"}, env)
        assert (
            env["_is_gpu_build"] is True
        ), "with the marker set every state is a GPU build, however the version is tagged"

    def test_xpu_supported_range(self, ver, hip, cuda):
        env = {"re": re, "_version": ver, "_hip": hip, "_cuda": cuda}
        _run_assignments("_ensure_xpu_torch", {"_ver", "_rel", "_n"}, env)
        assert _if_test_containing("_ensure_xpu_torch", "+xpu", env) == _old_xpu_ok(ver, hip, cuda)

    def test_rocm_hip_marker(self, ver, hip, cuda):
        env = {
            "re": re,
            "_version": ver,
            "_hip": hip,
            "_cuda": cuda,
            "_ran": True,
            "_importable": True,
        }
        _run_assignments("_ensure_rocm_torch", {"_installed_torch_ver", "_hip_marker"}, env)
        assert env["_hip_marker"] == _old_rocm_marker(ver, hip, cuda)
        assert env["_installed_torch_ver"] == ver.lower()

    def test_windows_rocm_verdict(self, ver, hip, cuda):
        probe = (True, True, ver, hip, cuda)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(stack_mod, "IS_WINDOWS", True)
            mp.setattr(stack_mod, "_probe_torch_runtime", lambda: probe)
            got = stack_mod._installed_torch_is_windows_rocm()
        assert got == _old_windows_rocm_yes(ver, hip, cuda)


def test_the_extraction_actually_reads_the_live_source():
    """If extraction silently found nothing, every parity test above would be vacuous."""
    env = {"re": re, "_version": "2.9.1+cu128", "_hip": "", "_cuda": "12.8"}
    _run_assignments(
        "_ensure_cuda_torch",
        {"_ver", "_cu_match", "_marker", "_installed_cu", "_installed_release", "_runtime_cu"},
        env,
    )
    assert env["_marker"] == "cuda"
    assert env["_installed_cu"] == "cu128"
    assert env["_runtime_cu"] == "cu128"


def test_probe_survives_undecodable_import_chatter():
    """errors="replace" is invisible to a mock, so this runs a real subprocess.

    text=True alone decodes strictly and UnicodeDecodeError is a ValueError, so it
    escapes the except below the call and takes the installer down instead of falling
    back to the on-disk classifier.
    """
    emit = (
        "import sys\n"
        "sys.stdout.buffer.write(b'chatter \\xff\\xfe\\n')\n"
        f"print('{stack_mod._TORCH_PROBE_MARKER}' + '|'.join(('2.9.1+cu128', '', '12.8')))\n"
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stack_mod.sys, "executable", sys.executable)
        mp.setattr(stack_mod, "_TORCH_RUNTIME_PROBE", None)
        real_run = stack_mod.subprocess.run

        def _run(cmd, **kwargs):
            return real_run([sys.executable, "-c", emit], **kwargs)

        mp.setattr(stack_mod.subprocess, "run", _run)
        ran, importable, version, hip, cuda = stack_mod._probe_torch_runtime()

    assert (ran, importable) == (True, True)
    assert (version, hip, cuda) == ("2.9.1+cu128", "", "12.8")


def test_no_unreachable_code_in_the_shared_probe():
    """The hardening pass replaced the parser in place; the old one must not linger."""
    body = _fn("_probe_torch_runtime").body
    returns = [i for i, node in enumerate(body) if isinstance(node, ast.Return)]
    assert not returns or returns[0] == len(body) - 1, (
        "statements follow the first top-level return in _probe_torch_runtime, "
        "so a previous implementation was left behind"
    )
