"""Whether `install_llm_compressor()` reaches for pip, per install state.

The guard exists because the in-process `import llmcompressor` can fail even when the
distribution is fine -- Unsloth's transformers patches interfere in that process, while the
real quantization runs in a clean subprocess that imports it without trouble. Falling through
to pip there made it re-resolve the version-capped, torch/transformers-pinned spec and
backtrack destructively (it drags in numpy<2 built from source and the export dies).

Presence alone is not the whole question though. A distribution outside
``_LLM_COMPRESSOR_SPEC`` is exactly what that pin exists to correct, so it must still be
reinstalled rather than silently quantized against.

No GPU and no network: the only thing substituted is the boundary the decision reads
(``importlib.metadata.version``) and the boundary it acts on (``subprocess.check_call``).
"""

from __future__ import annotations

import importlib.metadata as md
import subprocess

import pytest

from unsloth.save import _LLM_COMPRESSOR_SPEC, install_llm_compressor


def _pip_invoked_when(monkeypatch, reported: str | None) -> bool:
    """Run the guard with llmcompressor reported as *reported*, answering "did it pip?".

    ``reported=None`` presents a genuinely absent distribution.
    """
    calls: list[list[str]] = []
    real_version = md.version

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
    # The reported case: installed and in range, import failing in-process only.
    assert _pip_invoked_when(monkeypatch, "0.12.0") is False


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
