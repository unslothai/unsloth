# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch / torchcodec ABI guardrails (unslothai/unsloth#7225)."""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
IMPORT_FIXES_PATH = REPO_ROOT / "unsloth" / "import_fixes.py"


def _tomllib():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    return pytest.importorskip("tomli")


def _load_import_fixes_module():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test",
        IMPORT_FIXES_PATH,
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pyproject_declares_torch210_audio_extra_with_python_gate():
    text = PYPROJECT.read_text(encoding = "utf-8")
    assert "audio-torch210 = [" in text
    assert "torchcodec>=0.10.0,<0.11.0" in text
    assert "python_version >= '3.10'" in text
    assert "audio-torch290 = [" in text
    assert "audio-torch280 = [" in text
    assert "\naudio = [" not in text


def _stub_torch(monkeypatch, version: str):
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = version
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


def test_torch210_extras_bundle_audio_torch210():
    text = PYPROJECT.read_text(encoding = "utf-8")
    for extra in (
        "cu128-torch2100",
        "cu126-ampere-torch2100",
        "rocm72-torch2100",
    ):
        match = re.search(rf"^{extra} = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
        assert match is not None, extra
        assert "unsloth[audio-torch210]" in match.group(1)


def test_torchcodec_matrix_matches_notebook_validator():
    from scripts import notebook_validator as nv
    fixes = _load_import_fixes_module()
    assert fixes._TORCH_TORCHCODEC_MINORS == nv.TORCH_TORCHCODEC


def test_torchcodec_exclusive_upper_bound():
    fixes = _load_import_fixes_module()
    assert fixes._torchcodec_exclusive_upper("0.10") == "<0.11.0"
    assert fixes._torchcodec_exclusive_upper("0.9") == "<0.10.0"


def test_torch290_rejects_torchcodec_07(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.9.0+cu128")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.7.0")

    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None
    assert "audio-torch210" not in hint


def test_torch280_accepts_torchcodec_07(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.8.0+cu128")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.7.0")

    assert fixes._torchcodec_version_mismatch_hint() is None


def test_torch210_rejects_torchcodec_011(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.10.0+cu128")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.11.0",
    )

    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None
    assert "torchcodec 0.11.0" in hint
    assert "audio-torch210" in hint
    assert "<0.11.0" in hint
    assert "<11.0" not in hint


def test_torch210_accepts_torchcodec_010(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.10.0+cu128")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.10.0+cu128",
    )

    assert fixes._torchcodec_version_mismatch_hint() is None


def test_import_fixes_loads_on_python39_syntax():
    """Regression: module must import on 3.9 (postponed annotations for str | None)."""
    fixes = _load_import_fixes_module()
    assert callable(fixes._torchcodec_version_mismatch_hint)


def test_audio_extras_are_gated_to_platforms_with_a_torchcodec_wheel():
    """torchcodec publishes no sdist and no wheel for Linux aarch64, Windows ARM64 or
    Intel Mac, so an ungated pin makes pip fail the whole install on those hosts instead
    of just skipping audio -- and the cu*/rocm*/intel torch 2.10 extras pull it in.
    The marker must match PLATFORM_LACKS_TORCHCODEC_WHEEL in install_python_stack.py.
    """
    markers = pytest.importorskip("packaging.markers")
    tomllib = _tomllib()
    extras = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))["project"][
        "optional-dependencies"
    ]
    audio = {n: d for n, d in extras.items() if n.startswith("audio-torch")}
    assert audio, "expected audio-torch* extras"

    supported = [
        {"sys_platform": "linux", "platform_machine": "x86_64"},
        {"sys_platform": "win32", "platform_machine": "AMD64"},
        {"sys_platform": "darwin", "platform_machine": "arm64"},
    ]
    unsupported = [
        {"sys_platform": "linux", "platform_machine": "aarch64"},
        {"sys_platform": "win32", "platform_machine": "ARM64"},
        {"sys_platform": "darwin", "platform_machine": "x86_64"},
    ]
    for name, deps in audio.items():
        for dep in deps:
            _, _, marker_text = dep.partition(";")
            assert marker_text.strip(), f"{name}: {dep!r} has no marker"
            marker = markers.Marker(marker_text.strip())
            env = {"python_version": "3.12"}
            for case in supported:
                assert marker.evaluate({**env, **case}), f"{name} must install on {case}"
            for case in unsupported:
                assert not marker.evaluate(
                    {**env, **case}
                ), f"{name} has no wheel for {case} and must not be resolved there"
