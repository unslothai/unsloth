# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Whether `install_llm_compressor()` reaches for pip, per install state.

The guard exists because the in-process `import llmcompressor` can fail even when the
distribution is fine -- Unsloth's transformers patches interfere in that process, while the
real quantization runs in a clean subprocess that imports it without trouble. Falling through
to pip there made it re-resolve the version-capped, torch/transformers-pinned spec and
backtrack destructively (it drags in numpy<2 built from source and the export dies).

Presence alone is not the whole question though. A distribution outside
``_LLM_COMPRESSOR_SPEC`` is exactly what that pin exists to correct, so it must still be
reinstalled rather than silently quantized against -- and neither is metadata health: an
install missing a dependency reads as present and in range, and pip is what repairs it.

No GPU and no network. What is substituted is the two boundaries the decision reads
(``importlib.metadata.version``, and ``subprocess.run`` for the clean-subprocess import
probe) and the one it acts on (``subprocess.check_call``).
"""

from __future__ import annotations

import importlib.metadata as md
import subprocess
import sys

import pytest

from unsloth.save import _LLM_COMPRESSOR_SPEC, install_llm_compressor


def _pip_invoked_when(
    monkeypatch,
    reported: str | None,
    *,
    subprocess_import: object = 0,
    probes_out: list | None = None,
) -> bool:
    """Run the guard with llmcompressor reported as *reported*, answering "did it pip?".

    ``reported=None`` presents a genuinely absent distribution. ``subprocess_import`` is what
    the clean-subprocess probe finds: an exit status, or an exception instance to raise from
    ``subprocess.run`` (a timeout, a python that will not spawn).
    """
    calls: list[list[str]] = []
    probes: list[list[str]] = probes_out if probes_out is not None else []
    real_version = md.version

    def fake_run(cmd, *a, **k):
        probes.append(list(cmd))
        if isinstance(subprocess_import, BaseException):
            raise subprocess_import
        return subprocess.CompletedProcess(cmd, subprocess_import)

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_version(dist: str) -> str:
        if dist == "llmcompressor":
            if reported is None:
                raise md.PackageNotFoundError(dist)
            return reported
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: calls.append(list(cmd)))

    # The real in-process import must fail for the guard to be reached at all. It does when
    # llmcompressor is absent from this interpreter; when it is genuinely installed the first
    # branch returns early and there is nothing here to measure.
    try:
        import llmcompressor  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("llmcompressor imports in this interpreter, so the guard is unreachable")

    # After a "successful" install the function re-imports and raises, which is the outcome
    # for the absent case rather than a harness failure.
    try:
        install_llm_compressor()
    except Exception:
        pass
    return bool(calls)


def test_an_absent_distribution_is_still_installed(monkeypatch):
    assert _pip_invoked_when(monkeypatch, None) is True


def test_a_supported_version_is_not_reinstalled(monkeypatch):
    # The reported case: installed, in range, and importable in a clean subprocess, so the
    # in-process failure was Unsloth's own patches and pip has nothing to fix.
    probes: list[list[str]] = []
    assert _pip_invoked_when(monkeypatch, "0.12.0", probes_out = probes) is False
    # And it was decided by asking, not assumed: one probe, this interpreter, -c so unsloth
    # is never imported into it.
    assert len(probes) == 1, f"expected one clean-subprocess probe, got {probes}"
    assert probes[0][:2] == [sys.executable, "-c"], probes[0]


def test_an_installed_version_that_cannot_import_anywhere_is_repaired(monkeypatch):
    """Metadata is not health. An incomplete distribution -- compressed-tensors missing, a
    truncated install -- also presents as present and in range, and pip is what fixes it.
    Skipping there defers the failure past the whole model merge to _compressed_quantize.py,
    which imports the same two symbols this probe does."""
    assert _pip_invoked_when(monkeypatch, "0.12.0", subprocess_import = 1) is True


@pytest.mark.parametrize(
    "unknown",
    [
        subprocess.TimeoutExpired(cmd = ["python"], timeout = 600),
        OSError("no such interpreter"),
    ],
    ids = ["timeout", "will-not-spawn"],
)
def test_an_unknowable_probe_answer_keeps_the_install_skipped(monkeypatch, unknown):
    """The destructive re-resolve is the worse outcome, so an unanswered probe is not
    evidence of a broken install and must not be treated as one."""
    assert _pip_invoked_when(monkeypatch, "0.12.0", subprocess_import = unknown) is False


def test_the_probe_asks_the_question_the_export_runner_will_ask():
    """The probe is only evidence if it imports what _compressed_quantize.py imports, in the
    way it is launched: `sys.executable <runner>` with the ambient environment, never `-m`
    and never with unsloth imported. A probe of some other symbol could pass while the
    export's own import fails, which is the failure this is supposed to catch."""
    import inspect
    from pathlib import Path

    from unsloth.save import _llm_compressor_imports_cleanly

    probe_src = inspect.getsource(_llm_compressor_imports_cleanly)
    runner_src = (
        Path(inspect.getfile(install_llm_compressor)).parent / "_compressed_quantize.py"
    ).read_text()
    for line in (
        "from llmcompressor import oneshot",
        "from llmcompressor.modifiers.quantization import QuantizationModifier",
    ):
        assert line in runner_src, f"the export runner no longer imports: {line}"
        assert line in probe_src, f"the probe no longer imports: {line}"
    assert "sys.executable" in probe_src, "the probe must use this interpreter"
    assert "timeout" in probe_src, "an unbounded probe would hang the export"
    assert sys.executable


@pytest.mark.parametrize("version", ["0.5.0", "1.0.0", "0.13.0"])
def test_a_version_outside_the_pin_is_reinstalled(monkeypatch, version):
    """Presence is not support. Skipping these would quantize against an unsupported
    release and leave the pin decorative."""
    from packaging.requirements import Requirement

    assert not Requirement(_LLM_COMPRESSOR_SPEC).specifier.contains(
        version, prereleases = True
    ), f"{version} must be outside {_LLM_COMPRESSOR_SPEC} for this case to mean anything"
    assert _pip_invoked_when(monkeypatch, version) is True


def test_the_pin_the_guard_compares_against_is_a_bounded_range():
    """A spec that stopped bounding either end would make the checks above vacuous."""
    from packaging.requirements import Requirement

    spec = Requirement(_LLM_COMPRESSOR_SPEC).specifier
    ops = {s.operator for s in spec}
    assert ops & {">=", ">", "=="}, f"no floor in {_LLM_COMPRESSOR_SPEC}"
    assert ops & {"<=", "<", "=="}, f"no ceiling in {_LLM_COMPRESSOR_SPEC}"


def _install_outcome(monkeypatch, *, subprocess_import: int):
    """Run the guard through the INSTALL path, answering "what did it do afterwards?".

    Returns the function's result, or the exception it raised.
    """
    real_version = md.version

    def fake_version(dist: str) -> str:
        if dist == "llmcompressor":
            raise md.PackageNotFoundError(dist)
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)
    # The install "succeeds" without installing anything, so the re-import after it fails --
    # which is exactly what it does in the environment this guard is about.
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, subprocess_import),
    )
    try:
        import llmcompressor  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("llmcompressor imports in this interpreter, so the guard is unreachable")
    try:
        return install_llm_compressor()
    except Exception as exc:
        return exc


def test_a_repaired_install_is_validated_where_the_export_will_use_it(monkeypatch):
    """The import after the install runs in THIS process, with Unsloth's transformers patches
    in it, while the quantizing happens in a subprocess that has none. Raising on that killed
    an export whose environment was by then perfectly good."""
    assert _install_outcome(monkeypatch, subprocess_import = 0) == (None, None)


def test_an_install_that_fixed_nothing_still_fails_loudly(monkeypatch):
    """The other half: if a clean subprocess cannot import it either, the install genuinely
    did not work and proceeding would waste the whole model merge before saying so."""
    outcome = _install_outcome(monkeypatch, subprocess_import = 1)
    assert isinstance(outcome, RuntimeError), outcome
    assert "could not be imported" in str(outcome)


def _with_importable_llmcompressor(monkeypatch, reported: str):
    """Present an llmcompressor that IMPORTS, at version *reported*.

    The other cases here rely on the interpreter not having it, which is what made them
    blind to this branch: an out-of-range release that imports fine returns from the very
    first try block, before any version check. Substituting the module is the only way to
    reach that path without installing an unsupported release.
    """
    import types

    real_version = md.version

    def fake_version(dist: str) -> str:
        if dist == "llmcompressor":
            return reported
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)

    pkg = types.ModuleType("llmcompressor")
    pkg.oneshot = lambda *a, **k: None
    quant = types.ModuleType("llmcompressor.modifiers.quantization")
    quant.QuantizationModifier = type("QuantizationModifier", (), {})
    mods = types.ModuleType("llmcompressor.modifiers")
    mods.quantization = quant
    pkg.modifiers = mods
    for name, module in (
        ("llmcompressor", pkg),
        ("llmcompressor.modifiers", mods),
        ("llmcompressor.modifiers.quantization", quant),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: calls.append(list(cmd)))
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0)
    )
    return calls


def test_a_supported_version_that_imports_is_used_directly(monkeypatch):
    """The ordinary fast path: in range and importable, so the real symbols come back and
    nothing is installed."""
    calls = _with_importable_llmcompressor(monkeypatch, "0.12.0")
    oneshot, modifier = install_llm_compressor()
    assert oneshot is not None and modifier is not None
    assert calls == [], "a supported install must not be touched"


@pytest.mark.parametrize("version", ["0.5.0", "1.0.0", "0.13.0"])
def test_an_out_of_range_version_is_not_accepted_just_because_it_imports(
    monkeypatch, version
):
    """The gap: importability is not support. Returning the symbols here would quantize
    against a release _LLM_COMPRESSOR_SPEC explicitly excludes, and the pin would be
    decorative for exactly the installs it was written for."""
    from packaging.requirements import Requirement

    assert not Requirement(_LLM_COMPRESSOR_SPEC).specifier.contains(
        version, prereleases = True
    ), f"{version} must be outside {_LLM_COMPRESSOR_SPEC} for this case to mean anything"
    calls = _with_importable_llmcompressor(monkeypatch, version)
    try:
        install_llm_compressor()
    except Exception:
        pass
    assert calls, f"{version} imported and was accepted without being corrected"
    assert any(_LLM_COMPRESSOR_SPEC in " ".join(cmd) for cmd in calls), calls
