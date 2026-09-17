# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""``fix_broken_hf_xet_wheel`` -- hf_xet installed but unimportable.

huggingface_hub decides whether to use Xet with ``is_xet_available()``, which only asks
importlib.metadata whether the distribution is installed, so an hf_xet built for another CPU
architecture is still routed to and every download dies with transformers' misleading "you need to
install the hf_xet package". Seen for real on Windows on ARM with a ``win_amd64`` hf_xet wheel in an
ARM64 interpreter.

Runs on every OS: the machine and the wheel metadata are faked, so Linux and macOS CI exercise the
same paths Windows on ARM would.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import platform
from pathlib import Path

import pytest

from unsloth import import_fixes as IF


def _platform_tag_cases():
    return (
        ("win_amd64", "x86_64"),
        ("win_arm64", "arm64"),
        ("win32", "x86"),
        ("manylinux2014_x86_64", "x86_64"),
        ("manylinux_2_28_aarch64", "arm64"),
        ("musllinux_1_2_x86_64", "x86_64"),
        ("linux_armv7l", "armv7l"),
        ("macosx_11_0_arm64", "arm64"),
        ("macosx_10_12_x86_64", "x86_64"),
        # Unreadable on purpose: these must never be reported as a mismatch.
        ("macosx_10_9_universal2", None),
        ("any", None),
        ("something_else", None),
    )


@pytest.mark.parametrize("platform_tag,expected", _platform_tag_cases())
def test_cpu_family_from_platform_tag(platform_tag, expected):
    assert IF._cpu_family_from_platform_tag(platform_tag) == expected


@pytest.mark.parametrize(
    "machine,tags,expected",
    (
        ("ARM64", ("win_amd64",), True),
        ("ARM64", ("win_arm64",), False),
        ("AMD64", ("win_amd64",), False),
        ("aarch64", ("manylinux2014_x86_64",), True),
        ("x86_64", ("manylinux2014_x86_64", "manylinux_2_17_x86_64"), False),
        # Anything unreadable on either side is "cannot tell", never a mismatch.
        ("arm64", ("macosx_10_9_universal2",), None),
        ("riscv64", ("manylinux_2_28_aarch64",), None),
        ("ARM64", (), None),
    ),
)
def test_architecture_mismatch_verdicts(monkeypatch, machine, tags, expected):
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: tags)
    assert IF._hf_xet_architecture_mismatch() is expected


def _install_fake_environment(monkeypatch, hf_xet_present, import_error):
    """Pretend huggingface_hub is installed, and hf_xet is installed or not, without touching disk."""
    monkeypatch.delitem(IF.sys.modules, "hf_xet", raising = False)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package = None):
        if name == "huggingface_hub":
            return importlib.machinery.ModuleSpec("huggingface_hub", None)
        if name == "hf_xet":
            if not hf_xet_present:
                return None
            spec = importlib.machinery.ModuleSpec("hf_xet", None, is_package = True)
            spec.submodule_search_locations = []
            return spec
        return real_find_spec(name, package)

    def fake_import_module(name, package = None):
        if name == "hf_xet":
            if import_error is not None:
                raise import_error
            return object()
        return importlib.import_module(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(IF.importlib, "import_module", fake_import_module)


_WRONG_ARCHITECTURE = ImportError(
    "DLL load failed while importing hf_xet: %1 is not a valid Win32 application."
)


def test_fires_on_a_wrong_architecture_wheel(monkeypatch, caplog):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE)

    with caplog.at_level("WARNING", logger = IF.logger.name):
        IF.fix_broken_hf_xet_wheel()

    assert IF.os.environ.get("HF_HUB_DISABLE_XET") == "1"
    # One line, and it must name the real cause rather than repeat "pip install hf_xet".
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "architecture" in message and "win_amd64" in message

    # Idempotent: a second call neither re-logs nor changes anything.
    caplog.clear()
    IF.fix_broken_hf_xet_wheel()
    assert IF.os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert not caplog.records


def test_does_not_fire_when_hf_xet_imports(monkeypatch):
    """The metadata only raises a suspicion; a successful import overrules it."""
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = None)

    IF.fix_broken_hf_xet_wheel()
    assert "HF_HUB_DISABLE_XET" not in IF.os.environ


def test_does_not_fire_on_a_healthy_wheel(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_arm64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE)

    IF.fix_broken_hf_xet_wheel()
    assert "HF_HUB_DISABLE_XET" not in IF.os.environ


def test_does_not_fire_when_hf_xet_is_absent(monkeypatch):
    """huggingface_hub already downgrades to HTTP by itself when hf_xet is not installed."""
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    _install_fake_environment(monkeypatch, hf_xet_present = False, import_error = None)

    IF.fix_broken_hf_xet_wheel()
    assert "HF_HUB_DISABLE_XET" not in IF.os.environ


@pytest.mark.parametrize("user_value", ("0", "1", "false"))
def test_never_overrides_an_explicit_setting(monkeypatch, user_value):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", user_value)
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE)

    IF.fix_broken_hf_xet_wheel()
    assert IF.os.environ["HF_HUB_DISABLE_XET"] == user_value


def test_runs_before_anything_imports_huggingface_hub():
    """huggingface_hub freezes HF_HUB_DISABLE_XET into constants.py at import, so the fix is
    worthless unless it runs before the first import of the Hub anywhere in _gpu_init.py."""
    source = (Path(__file__).resolve().parent.parent / "unsloth" / "_gpu_init.py").read_text(
        encoding = "utf-8"
    )
    assert "fix_broken_hf_xet_wheel()" in source, (
        "DRIFT DETECTED: fix_broken_hf_xet_wheel is defined but never called in _gpu_init.py, "
        "so real imports never apply it."
    )
    assert source.index("fix_broken_hf_xet_wheel()") < source.index("fix_huggingface_hub()"), (
        "fix_broken_hf_xet_wheel() must run before fix_huggingface_hub(), which imports "
        "huggingface_hub and freezes HF_HUB_DISABLE_XET."
    )
