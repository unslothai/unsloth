# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""``fix_broken_hf_xet_wheel``: the host platform, wheel metadata and extension headers are all
faked, so this runs on every OS and Linux CI exercises the Windows on ARM paths."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import platform
import struct
import sys
import sysconfig
import types
from pathlib import Path

import pytest


def _load_import_fixes():
    """`import unsloth` needs torch and an accelerator, so importing the package would make every
    test here an ImportError on a stock runner. This module needs only stdlib and packaging."""
    path = Path(__file__).resolve().parents[1] / "unsloth" / "import_fixes.py"
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


IF = _load_import_fixes()


def _fake_host(
    monkeypatch,
    sysconfig_platform,
    machine = "unknown-cpu",
):
    """`machine` defaults to something unmappable, so a passing test is passing on
    sysconfig.get_platform() alone, which is the point."""
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
    """REGRESSION. platform.machine() on Windows asks WMI for the PHYSICAL processor, so an
    emulated x86-64 interpreter on Windows on ARM answers "ARM64" while being win-amd64. The case
    that matters is the unloadable win_arm64 wheel read as healthy."""
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
    """`distribution_installed` is what importlib.metadata would say, a SEPARATE question from
    what the import system says, and the only one is_xet_available() asks."""
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
            # A real module, not a bare object: __file__ is what separates an installed package
            # from the empty namespace shell the confirm step has to reject.
            loaded = types.ModuleType("hf_xet")
            loaded.__file__ = "/site-packages/hf_xet/__init__.py"
            return loaded
        return importlib.import_module(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(IF.importlib, "import_module", fake_import_module)


def _raise_package_not_found(name):
    raise importlib.metadata.PackageNotFoundError(name)


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
    # one line, naming the real cause rather than repeating "pip install hf_xet"
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "architecture" in message and "win_amd64" in message

    # idempotent
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


def test_absence_is_decided_by_real_metadata_not_by_a_stub(monkeypatch):
    """REGRESSION. Every other no-fire test stubs `_hf_xet_distribution_is_installed`, so one that
    answered True for an uninstalled package would keep them green while a user with no hf_xet got
    the warning and HF_HUB_DISABLE_XET=1. Here the real helper is used."""
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.delitem(IF.sys.modules, "hf_xet", raising = False)
    monkeypatch.setattr(IF, "importlib_version", _raise_package_not_found, raising = False)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package = None):
        if name == "huggingface_hub":
            return importlib.machinery.ModuleSpec("huggingface_hub", None)
        if name == "hf_xet":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert IF._hf_xet_distribution_is_installed() is False
    IF.fix_broken_hf_xet_wheel()
    assert "HF_HUB_DISABLE_XET" not in IF.os.environ


def test_an_imported_hf_xet_short_circuits_before_any_lookup(monkeypatch):
    """REGRESSION. Removing that early return changes no other assertion, and raising from
    find_spec would not show it either since the lookup sits inside an `except Exception: return`.
    So the lookups are recorded and the assertion is that there were none."""
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    working = types.ModuleType("hf_xet")
    working.__file__ = "/site-packages/hf_xet/__init__.py"  # a bare ModuleType is a shell
    monkeypatch.setitem(IF.sys.modules, "hf_xet", working)

    lookups = []

    def recording_find_spec(name, package = None):
        lookups.append(name)
        return None

    monkeypatch.setattr(importlib.util, "find_spec", recording_find_spec)
    monkeypatch.setattr(
        IF, "_hf_xet_distribution_is_installed", lambda: lookups.append("metadata") or False
    )

    IF.fix_broken_hf_xet_wheel()
    assert lookups == []
    assert "HF_HUB_DISABLE_XET" not in IF.os.environ


def test_fires_when_hf_xet_is_only_an_empty_namespace_package(monkeypatch, caplog, tmp_path):
    """REGRESSION. An hf_xet/ directory with no __init__.py and no extension is a NAMESPACE
    package: it imports cleanly and defines nothing, so confirming with a bare import cleared a
    suspicion that was correct. huggingface_hub does `from hf_xet import PyXetDownloadInfo,
    download_files`, which still fails. Reproduced against huggingface_hub 0.36.2 with the package
    contents deleted and the directory and dist-info left: find_spec returned a namespace spec,
    `import hf_xet` succeeded, and every download still went to Xet and died with
    "cannot import name 'PyXetDownloadInfo' from 'hf_xet' (unknown location)".
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.delitem(IF.sys.modules, "hf_xet", raising = False)
    monkeypatch.setattr(IF, "_hf_xet_distribution_is_installed", lambda: True)
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ())

    package = tmp_path / "hf_xet"
    package.mkdir()  # no __init__.py, no extension: exactly what makes it a namespace package
    spec = importlib.machinery.ModuleSpec("hf_xet", None, is_package = True)
    spec.submodule_search_locations = [str(package)]

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package = None):
        if name == "huggingface_hub":
            return importlib.machinery.ModuleSpec("huggingface_hub", None)
        if name == "hf_xet":
            return spec
        return real_find_spec(name, package)

    namespace_module = types.ModuleType("hf_xet")
    namespace_module.__file__ = None  # what CPython gives a namespace package

    def fake_import_module(name, package = None):
        if name == "hf_xet":
            IF.sys.modules["hf_xet"] = namespace_module
            return namespace_module
        return importlib.import_module(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(IF.importlib, "import_module", fake_import_module)

    assert (
        IF._hf_xet_extension_is_missing(spec) is True
    ), "the suspicion itself must still be raised"

    with caplog.at_level("WARNING", logger = IF.logger.name):
        IF.fix_broken_hf_xet_wheel()

    assert IF.os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert "namespace package" in caplog.records[0].getMessage()
    # The shell must not be left behind for the next importer to trip over.
    assert "hf_xet" not in IF.sys.modules


def test_fires_when_the_hub_already_cached_the_namespace_shell(monkeypatch, caplog, tmp_path):
    """REGRESSION. huggingface_hub's own `from hf_xet import PyXetDownloadInfo, download_files`
    leaves hf_xet in sys.modules even when those symbols are missing, because the PACKAGE imported
    fine and only the attribute lookup failed. A cached module is therefore not proof of a working
    one, and short-circuiting on its mere presence left the user broken. Reproduced against
    huggingface_hub 0.36.2, Hub used before unsloth: the import raised "cannot import name
    'PyXetDownloadInfo' from 'hf_xet' (unknown location)", the shell stayed cached with
    __file__ None, and routes_to_xet was still True afterwards.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.setattr(IF, "_hf_xet_distribution_is_installed", lambda: True)
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ())

    package = tmp_path / "hf_xet"
    package.mkdir()
    spec = importlib.machinery.ModuleSpec("hf_xet", None, is_package = True)
    spec.submodule_search_locations = [str(package)]

    shell = types.ModuleType("hf_xet")
    shell.__file__ = None  # what the Hub left behind
    monkeypatch.setitem(IF.sys.modules, "hf_xet", shell)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package = None):
        if name == "huggingface_hub":
            return importlib.machinery.ModuleSpec("huggingface_hub", None)
        if name == "hf_xet":
            return spec
        return real_find_spec(name, package)

    def fake_import_module(name, package = None):
        if name == "hf_xet":
            IF.sys.modules["hf_xet"] = shell
            return shell
        return importlib.import_module(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(IF.importlib, "import_module", fake_import_module)

    with caplog.at_level("WARNING", logger = IF.logger.name):
        IF.fix_broken_hf_xet_wheel()

    assert IF.os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert "hf_xet" not in IF.sys.modules, "the shell must not be left cached"


def test_fires_when_only_the_distribution_metadata_survives(monkeypatch, caplog):
    """REGRESSION. find_spec says "not installed", but is_xet_available() asks
    importlib.metadata, which still succeeds, so every download still goes down the Xet branch."""
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
    """REGRESSION. WHEEL metadata gone but the wrong-architecture extension still there.
    huggingface_hub still routes to Xet, so giving up at "no readable tags" left it broken."""
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


@pytest.mark.parametrize("user_value", ("1", "true", "YES", "on"))
def test_an_explicit_disable_set_after_the_hub_was_imported_is_carried_through(
    monkeypatch, user_value
):
    """REGRESSION. Setting the variable late is read by nobody, so honour it on the constant.

    The user who imports transformers first and only then sets HF_HUB_DISABLE_XET=1 asked for Xet
    off and, without this, still gets every download routed to Xet. The truthy set is
    huggingface_hub's own ENV_VARS_TRUE_VALUES, matched case-insensitively as constants.py does.
    """
    monkeypatch.setenv("HF_HUB_DISABLE_XET", user_value)
    modules = _fake_hub_modules(monkeypatch, {"": _UNBOUND, "constants": False})

    IF.fix_broken_hf_xet_wheel()

    assert modules["constants"].HF_HUB_DISABLE_XET is True
    assert IF.os.environ["HF_HUB_DISABLE_XET"] == user_value


@pytest.mark.parametrize("user_value", ("0", "false", "no", "off"))
def test_an_explicit_enable_is_never_carried_through(monkeypatch, user_value):
    """The opposite direction must still win: a user who asked for Xet keeps it."""
    monkeypatch.setenv("HF_HUB_DISABLE_XET", user_value)
    _fake_host(monkeypatch, "win-arm64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE)
    modules = _fake_hub_modules(monkeypatch, {"": _UNBOUND, "constants": False})

    IF.fix_broken_hf_xet_wheel()

    assert modules["constants"].HF_HUB_DISABLE_XET is False
    assert IF.os.environ["HF_HUB_DISABLE_XET"] == user_value


def _fake_hub_modules(monkeypatch, bindings):
    """`bindings` maps a module suffix ("" is the package) to its starting HF_HUB_DISABLE_XET, or
    to `_UNBOUND` for a module that does not define the flag (every module before 0.34)."""
    modules = {}
    for suffix, value in bindings.items():
        name = "huggingface_hub" + (f".{suffix}" if suffix else "")
        module = types.ModuleType(name)
        if value is not _UNBOUND:
            module.HF_HUB_DISABLE_XET = value
        monkeypatch.setitem(sys.modules, name, module)
        modules[suffix] = module
    return modules


_UNBOUND = object()


def test_patches_the_frozen_constant_when_the_hub_is_already_imported(monkeypatch):
    """REGRESSION. huggingface_hub >= 0.34 evaluates HF_HUB_DISABLE_XET once, at import time, so
    for a user whose script starts with `import transformers` the variable alone fixes nothing."""
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    _fake_host(monkeypatch, "win-arm64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE)
    modules = _fake_hub_modules(monkeypatch, {"": _UNBOUND, "constants": False})

    IF.fix_broken_hf_xet_wheel()

    assert IF.os.environ.get("HF_HUB_DISABLE_XET") == "1", "the variable must still be set too"
    assert (
        modules["constants"].HF_HUB_DISABLE_XET is True
    ), "the already frozen constant was left False, so this process still routes to Xet"


def test_patches_every_module_that_binds_the_flag(monkeypatch):
    """Patch by attribute, not by hardcoded module name: a module that ever does
    `from .constants import HF_HUB_DISABLE_XET` would hold its own copy that patching constants.py
    could not reach. Today constants.py is the only binding that exists.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    _fake_host(monkeypatch, "win-arm64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE)
    modules = _fake_hub_modules(
        monkeypatch,
        {
            "": _UNBOUND,
            "constants": False,
            "file_download": False,  # a hypothetical `from .constants import ...` copy
            "utils._xet": False,  # ditto
            "utils._runtime": _UNBOUND,  # reads constants.X, holds no copy: must stay unbound
        },
    )
    # a look-alike package must not be caught by the prefix match
    decoy = types.ModuleType("huggingface_hubby")
    decoy.HF_HUB_DISABLE_XET = False
    monkeypatch.setitem(sys.modules, "huggingface_hubby", decoy)

    IF.fix_broken_hf_xet_wheel()

    for suffix in ("constants", "file_download", "utils._xet"):
        assert modules[suffix].HF_HUB_DISABLE_XET is True, f"{suffix} kept its stale copy"
    assert not hasattr(modules["utils._runtime"], "HF_HUB_DISABLE_XET")
    assert not hasattr(modules[""], "HF_HUB_DISABLE_XET")
    assert decoy.HF_HUB_DISABLE_XET is False, "patched an unrelated package by prefix"


def test_does_not_invent_the_flag_on_hub_versions_that_never_had_it(monkeypatch):
    """Before 0.34 the Hub read os.environ per call, so the variable alone is the whole fix and
    creating the attribute would put a value into a namespace upstream does not own.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    _fake_host(monkeypatch, "win-arm64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))
    _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE)
    modules = _fake_hub_modules(monkeypatch, {"": _UNBOUND, "constants": _UNBOUND})

    IF.fix_broken_hf_xet_wheel()

    assert IF.os.environ.get("HF_HUB_DISABLE_XET") == "1"
    assert not hasattr(modules["constants"], "HF_HUB_DISABLE_XET")


def test_does_not_import_huggingface_hub_just_to_patch_it(monkeypatch):
    """The normal ordering. Nothing is frozen yet, and importing the Hub here would move its cost
    into every unsloth import."""
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising = False)
    for name in [n for n in sys.modules if n.startswith("huggingface_hub.")]:
        monkeypatch.delitem(sys.modules, name, raising = False)

    assert IF._disable_xet_on_already_imported_huggingface_hub() == ()
    assert "huggingface_hub" not in sys.modules


@pytest.mark.parametrize("scenario", ("healthy", "absent", "explicit"))
def test_leaves_the_frozen_constant_alone_when_the_fix_does_not_fire(monkeypatch, scenario):
    modules = _fake_hub_modules(monkeypatch, {"": _UNBOUND, "constants": False})
    _fake_host(monkeypatch, "win-arm64")
    monkeypatch.setattr(IF, "_hf_xet_wheel_platform_tags", lambda: ("win_amd64",))

    if scenario == "healthy":
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
        _install_fake_environment(monkeypatch, hf_xet_present = True, import_error = None)
    elif scenario == "absent":
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
        _install_fake_environment(monkeypatch, hf_xet_present = False, import_error = None)
    else:
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
        _install_fake_environment(
            monkeypatch, hf_xet_present = True, import_error = _WRONG_ARCHITECTURE
        )

    IF.fix_broken_hf_xet_wheel()

    assert modules["constants"].HF_HUB_DISABLE_XET is False


def test_patching_is_idempotent_and_reports_only_real_changes(monkeypatch):
    modules = _fake_hub_modules(monkeypatch, {"": _UNBOUND, "constants": False})

    assert IF._disable_xet_on_already_imported_huggingface_hub() == ("huggingface_hub.constants",)
    assert modules["constants"].HF_HUB_DISABLE_XET is True
    assert IF._disable_xet_on_already_imported_huggingface_hub() == ()


def test_patching_survives_hostile_modules(monkeypatch):
    """Lazy packages raise from __getattr__, None entries linger, modules refuse setattr. A bad
    neighbour must not stop the module that matters from being patched."""
    modules = _fake_hub_modules(monkeypatch, {"": _UNBOUND, "constants": False})

    class Exploding(types.ModuleType):
        def __getattr__(self, item):
            raise RuntimeError("lazy import blew up")

    class ReadOnly(types.ModuleType):
        HF_HUB_DISABLE_XET = False

        def __setattr__(self, item, value):
            raise AttributeError("read only module")

    monkeypatch.setitem(sys.modules, "huggingface_hub.lazy", Exploding("huggingface_hub.lazy"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.frozen", ReadOnly("huggingface_hub.frozen"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.gone", None)

    patched = IF._disable_xet_on_already_imported_huggingface_hub()

    assert patched == ("huggingface_hub.constants",)
    assert modules["constants"].HF_HUB_DISABLE_XET is True


def test_runs_before_anything_imports_huggingface_hub():
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
    """Survives refactoring, unlike the pairwise check above: a fix added later could freeze
    HF_HUB_DISABLE_XET before we set it with no failure anywhere to say so."""
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
