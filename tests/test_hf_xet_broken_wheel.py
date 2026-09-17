# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""``fix_broken_hf_xet_wheel`` -- hf_xet installed but unimportable.

huggingface_hub decides whether to use Xet with ``is_xet_available()``, which only asks
importlib.metadata whether the distribution is installed, so an hf_xet built for another CPU
architecture is still routed to and every download dies with transformers' misleading "you need to
install the hf_xet package". Seen for real on Windows on ARM with a ``win_amd64`` hf_xet wheel in an
ARM64 interpreter.

Runs on every OS: the host platform, the wheel metadata and the compiled extension headers are all
faked, so Linux and macOS CI exercise the same paths Windows on ARM would.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import platform
import struct
import sysconfig
from pathlib import Path

import pytest

from unsloth import import_fixes as IF


def _fake_host(
    monkeypatch,
    sysconfig_platform,
    machine = "unknown-cpu",
):
    """Pretend to be an interpreter built for `sysconfig_platform`.

    Both sources are faked, and `machine` defaults to something unmappable, so any test that
    passes is passing on `sysconfig.get_platform()` alone. That is the point: on Windows
    `platform.machine()` reports the PHYSICAL cpu, not the interpreter's, and must not be what
    decides the verdict.
    """
    monkeypatch.setattr(sysconfig, "get_platform", lambda: sysconfig_platform)
    monkeypatch.setattr(platform, "machine", lambda: machine)


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
    "host_platform,tags,expected",
    (
        ("win-arm64", ("win_amd64",), True),
        ("win-arm64", ("win_arm64",), False),
        ("win-amd64", ("win_amd64",), False),
        ("win-amd64", ("win_arm64",), True),
        ("linux-aarch64", ("manylinux2014_x86_64",), True),
        ("linux-x86_64", ("manylinux2014_x86_64", "manylinux_2_17_x86_64"), False),
        ("macosx-11.0-arm64", ("macosx_11_0_arm64",), False),
        ("macosx-11.0-arm64", ("macosx_10_12_x86_64",), True),
        # Anything unreadable on either side is "cannot tell", never a mismatch.
        ("macosx-11.0-arm64", ("macosx_10_9_universal2",), None),
        ("linux-riscv64", ("manylinux_2_28_aarch64",), None),
        ("win-arm64", (), None),
    ),
)
def test_architecture_mismatch_verdicts(monkeypatch, host_platform, tags, expected):
    _fake_host(monkeypatch, host_platform)
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: tags)
    assert IF._hf_xet_architecture_mismatch() is expected


@pytest.mark.parametrize(
    "host_platform,machine,expected",
    (
        ("win-amd64", "AMD64", "x86_64"),
        ("win-arm64", "ARM64", "arm64"),
        ("win32", "x86", "x86"),
        ("linux-x86_64", "x86_64", "x86_64"),
        ("linux-aarch64", "aarch64", "arm64"),
        ("macosx-11.0-arm64", "arm64", "arm64"),
        ("macosx-10.12-x86_64", "x86_64", "x86_64"),
        # sysconfig cannot name a single CPU for a universal2 build, so platform.machine()
        # is the right answer there and only there.
        ("macosx-10.9-universal2", "arm64", "arm64"),
        ("something-unparseable", "x86_64", "x86_64"),
        ("something-unparseable", "sparc64", None),
    ),
)
def test_host_cpu_family_prefers_sysconfig(monkeypatch, host_platform, machine, expected):
    monkeypatch.setattr(sysconfig, "get_platform", lambda: host_platform)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert IF._host_cpu_family() == expected


def test_host_cpu_family_ignores_the_physical_cpu_on_windows(monkeypatch):
    """REGRESSION. CPython's platform.machine() on Windows asks WMI for the PHYSICAL processor
    (platform._get_machine_win32), so an emulated x86-64 interpreter on Windows on ARM answers
    "ARM64" while being win-amd64. Measured on such a box: every x86-64 venv there reports
    platform.machine() == "ARM64" beside sysconfig.get_platform() == "win-amd64".

    Keying the verdict off platform.machine() got it wrong in BOTH directions: a correct
    win_amd64 wheel read as a mismatch, and a genuinely unloadable win_arm64 wheel read as
    healthy, which is the one that matters because nothing downstream would have caught it.
    """
    monkeypatch.setattr(sysconfig, "get_platform", lambda: "win-amd64")
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")

    assert IF._host_cpu_family() == "x86_64"

    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    assert IF._hf_xet_architecture_mismatch() is False, "healthy win_amd64 wheel called broken"

    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_arm64",))
    assert IF._hf_xet_architecture_mismatch() is True, "unloadable win_arm64 wheel called healthy"


def _install_fake_environment(
    monkeypatch,
    hf_xet_present,
    import_error,
    distribution_installed = None,
):
    """Pretend huggingface_hub is installed, and hf_xet is installed or not, without touching disk.

    `distribution_installed` is what importlib.metadata would say, which is a SEPARATE question
    from what the import system says: huggingface_hub's is_xet_available() asks only the former.
    It defaults to following `hf_xet_present`.
    """
    monkeypatch.delitem(IF.sys.modules, "hf_xet", raising = False)
    real_find_spec = importlib.util.find_spec
    if distribution_installed is None:
        distribution_installed = hf_xet_present
    monkeypatch.setattr(IF, "_hf_xet_distribution_is_installed", lambda: distribution_installed)

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
    _fake_host(monkeypatch, "win-arm64")
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
    _fake_host(monkeypatch, "win-arm64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = None)

    IF.fix_broken_hf_xet_wheel()
    assert "HF_HUB_DISABLE_XET" not in IF.os.environ


def test_does_not_fire_on_a_healthy_wheel(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    _fake_host(monkeypatch, "win-arm64")
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


def test_fires_when_only_the_distribution_metadata_survives(monkeypatch, caplog):
    """REGRESSION. A leftover hf_xet-*.dist-info with no package directory beside it.

    find_spec() says "not installed", but huggingface_hub never asks find_spec:
    is_xet_available() -> is_package_available("hf_xet") -> importlib.metadata.version(), which
    still succeeds. Verified against huggingface_hub 1.31.0 with the package directory deleted and
    the dist-info left in place: is_xet_available() returned True, so every download still went
    down the Xet branch and died in `from hf_xet import XetFileInfo`. Returning early on
    `spec is None` left exactly that install broken.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    _install_fake_environment(
        monkeypatch,
        hf_xet_present = False,  # find_spec -> None
        import_error = ModuleNotFoundError("No module named 'hf_xet'"),
        distribution_installed = True,  # ...but importlib.metadata still sees it
    )

    with caplog.at_level("WARNING", logger = IF.logger.name):
        IF.fix_broken_hf_xet_wheel()

    assert IF.os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert "not importable" in caplog.records[0].getMessage()


def _compiled_object(kind, machine):
    """Smallest ELF / PE / Mach-O header that carries a machine id."""
    if kind == "elf":
        return b"\x7fELF\x02\x01" + b"\0" * 12 + struct.pack("<H", machine) + b"\0" * 40
    if kind == "pe":
        head = bytearray(b"\0" * 0x100)
        head[:2] = b"MZ"
        struct.pack_into("<I", head, 0x3C, 0x80)
        head[0x80:0x84] = b"PE\0\0"
        struct.pack_into("<H", head, 0x84, machine)
        return bytes(head)
    return struct.pack("<II", 0xFEEDFACF, machine) + b"\0" * 40


@pytest.mark.parametrize(
    "kind,machine,expected",
    (
        ("elf", 0x3E, "x86_64"),
        ("elf", 0xB7, "arm64"),
        ("elf", 0x28, "armv7l"),
        ("elf", 0x1234, None),
        ("pe", 0x8664, "x86_64"),
        ("pe", 0xAA64, "arm64"),
        ("pe", 0x014C, "x86"),
        ("pe", 0x1234, None),
        ("macho", 0x01000007, "x86_64"),
        ("macho", 0x0100000C, "arm64"),
        ("macho", 0x1234, None),
    ),
)
def test_cpu_family_from_compiled_object(tmp_path, kind, machine, expected):
    binary = tmp_path / "hf_xet.bin"
    binary.write_bytes(_compiled_object(kind, machine))
    assert IF._cpu_family_from_compiled_object(str(binary)) == expected


def test_cpu_family_from_compiled_object_survives_garbage(tmp_path):
    for name, payload in (("empty.bin", b""), ("short.bin", b"MZ"), ("text.bin", b"not a binary")):
        binary = tmp_path / name
        binary.write_bytes(payload)
        assert IF._cpu_family_from_compiled_object(str(binary)) is None
    assert IF._cpu_family_from_compiled_object(str(tmp_path / "missing.bin")) is None


def test_extension_header_decides_when_wheel_metadata_is_unreadable(monkeypatch, tmp_path):
    """REGRESSION. WHEEL metadata gone (vendored, repackaged, trimmed dist-info) but the wrong
    architecture extension still sitting there. huggingface_hub still routes to Xet, so giving up
    at "no readable tags" left the install broken. The .pyd's own header is the better source."""
    package = tmp_path / "hf_xet"
    package.mkdir()
    extension = package / ("hf_xet" + importlib.machinery.EXTENSION_SUFFIXES[0])
    extension.write_bytes(_compiled_object("pe", 0x8664))  # x86-64 binary

    spec = importlib.machinery.ModuleSpec("hf_xet", None, is_package = True)
    spec.submodule_search_locations = [str(package)]

    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ())
    _fake_host(monkeypatch, "win-arm64")
    assert IF._hf_xet_architecture_mismatch(spec) is True

    _fake_host(monkeypatch, "win-amd64")
    assert IF._hf_xet_architecture_mismatch(spec) is False


def test_extension_header_is_not_consulted_when_the_wheel_tag_is_readable(monkeypatch, tmp_path):
    """The WHEEL tag stays authoritative; the header is only a fallback."""
    package = tmp_path / "hf_xet"
    package.mkdir()
    (package / ("hf_xet" + importlib.machinery.EXTENSION_SUFFIXES[0])).write_bytes(
        _compiled_object("pe", 0x8664)
    )
    spec = importlib.machinery.ModuleSpec("hf_xet", None, is_package = True)
    spec.submodule_search_locations = [str(package)]

    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("macosx_10_9_universal2",))
    _fake_host(monkeypatch, "win-arm64")
    assert IF._hf_xet_architecture_mismatch(spec) is None


@pytest.mark.parametrize("user_value", ("0", "1", "false"))
def test_never_overrides_an_explicit_setting(monkeypatch, user_value):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", user_value)
    _fake_host(monkeypatch, "win-arm64")
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


def test_is_the_very_first_import_fix_called():
    """Stronger than the pairwise check above, and the one that survives refactoring.

    Naming the fixes known to import the Hub today (fix_huggingface_hub, check_fbgemm_gpu_version
    via transformers, disable_broken_vllm via vllm) only guards the call graph as it stands. Any
    fix added later, or any existing one that grows a transformers import, would import
    huggingface_hub first and freeze HF_HUB_DISABLE_XET before we ever set it, with no failure
    anywhere to say so. So require this one to be first, full stop.
    """
    gpu_init = Path(__file__).resolve().parent.parent / "unsloth" / "_gpu_init.py"
    source = gpu_init.read_text(encoding = "utf-8")

    imported = source.split("from .import_fixes import (", 1)[1].split(")", 1)[0]
    names = [line.strip().rstrip(",") for line in imported.splitlines() if line.strip()]
    assert "fix_broken_hf_xet_wheel" in names

    calls = sorted((source.index(f"\n{name}("), name) for name in names if f"\n{name}(" in source)
    assert calls[0][1] == "fix_broken_hf_xet_wheel", (
        f"fix_broken_hf_xet_wheel() must be the first import fix invoked in _gpu_init.py, but "
        f"{calls[0][1]}() runs before it. huggingface_hub freezes HF_HUB_DISABLE_XET into "
        f"constants.py at import time, so anything that reaches the Hub first makes this fix a "
        f"no-op."
    )
