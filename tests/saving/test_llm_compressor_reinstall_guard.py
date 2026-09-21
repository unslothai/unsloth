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
