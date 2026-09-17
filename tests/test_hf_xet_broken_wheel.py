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
    """Load `unsloth/import_fixes.py` directly, without importing the `unsloth` package.

    `import unsloth` needs torch and an accelerator, which is exactly what the runners this
    file is meant to cover do not have: a bare `from unsloth import import_fixes` turns every
    test here into `ImportError: Unsloth: torch not found` on a stock Linux, macOS or Windows
    runner, so the cross-platform coverage the module docstring claims was not actually being
    collected anywhere. The module's own top-level imports are stdlib plus `packaging`, so
    loading it by path costs nothing and runs everywhere.
    """
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


def _raise_package_not_found(name):
    """What importlib.metadata.version() does for a distribution that is not installed."""
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


def test_absence_is_decided_by_real_metadata_not_by_a_stub(monkeypatch):
    """REGRESSION. The absent case must survive the REAL `_hf_xet_distribution_is_installed`.

    Every other no-fire test replaces that helper with a lambda, so a helper that answered True
    for a package nobody installed would leave them all green while a user with no hf_xet at all
    got the "installed but cannot be imported" warning and HF_HUB_DISABLE_XET=1 exported to every
    child process. Here the helper is left alone and importlib.metadata is asked for real.
    """
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
    """REGRESSION. hf_xet already in sys.modules must return before touching the import system.

    Nothing can be fixed once the module is loaded, so the early return is what keeps repeat calls
    free. Without it every call would walk find_spec and the distribution metadata again, which no
    behavioural assertion elsewhere would notice, and raising from find_spec would not show it
    either since the lookup sits inside a `except Exception: return`. So the calls are recorded
    and the assertion is that there were none.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.setitem(IF.sys.modules, "hf_xet", types.ModuleType("hf_xet"))

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


def _fake_hub_modules(monkeypatch, bindings):
    """Put a fake huggingface_hub package into sys.modules.

    `bindings` maps a module suffix ("" for the package itself, "constants", ...) to the value its
    HF_HUB_DISABLE_XET attribute should start at, or to `_UNBOUND` for a module that does not
    define the flag at all (which is every module on huggingface_hub < 0.34).
    """
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
    """REGRESSION. The environment variable alone fixes nothing once the Hub has been imported.

    huggingface_hub >= 0.34 evaluates HF_HUB_DISABLE_XET exactly once, at import time, in
    constants.py. A user whose script starts with `import transformers` has already frozen it to
    False before unsloth runs, so setting the variable afterwards is read by nobody: every download
    still routes to Xet and still dies on the unimportable hf_xet. Measured on huggingface_hub
    1.31.0 with transformers 5.17.0 imported first and a win_arm64 hf_xet in a win-amd64
    interpreter: env HF_HUB_DISABLE_XET=1, constants.HF_HUB_DISABLE_XET still False,
    is_xet_available() still True, the transfer went through xet_get and raised.
    """
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
    """Readers today all go through `constants.HF_HUB_DISABLE_XET`, so rebinding constants.py is
    enough. A module that ever does `from .constants import HF_HUB_DISABLE_XET` would hold its own
    copy, which no amount of patching constants.py would reach, so patch by attribute rather than
    by hardcoded module name. Scanned across huggingface_hub 0.24.7, 0.30.2, 0.33.5, 0.34.4, 0.36.2
    and 1.31.0 (with transformers loaded): constants.py is the only binding that exists so far.
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
    # A look-alike top level package must not be touched by the prefix match.
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
    """HF_HUB_DISABLE_XET arrived in huggingface_hub 0.31.0, and 0.31 to 0.33 read os.environ on
    every call rather than freezing it, so for all of those the variable alone is the whole fix.
    Creating the attribute there would put a value into a namespace upstream does not own.
    Executed against 0.24.7, 0.30.2 and 0.33.5: attribute still absent afterwards, downloads still
    took plain HTTPS.
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
    """The normal ordering, where unsloth runs first. Nothing is frozen yet, the variable is read
    in time, and importing the Hub here would move its cost into every unsloth import."""
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising = False)
    for name in [n for n in sys.modules if n.startswith("huggingface_hub.")]:
        monkeypatch.delitem(sys.modules, name, raising = False)

    assert IF._disable_xet_on_already_imported_huggingface_hub() == ()
    assert "huggingface_hub" not in sys.modules


@pytest.mark.parametrize("scenario", ("healthy", "absent", "explicit"))
def test_leaves_the_frozen_constant_alone_when_the_fix_does_not_fire(monkeypatch, scenario):
    """The in-memory patch rides on exactly the same verdict as the environment variable: a healthy
    hf_xet, no hf_xet, or a user who chose for themselves must all come out untouched."""
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
    # Already True, so a second pass reports nothing changed.
    assert IF._disable_xet_on_already_imported_huggingface_hub() == ()


def test_patching_survives_hostile_modules(monkeypatch):
    """sys.modules is not a tidy place: lazy packages raise from __getattr__, None entries linger
    from failed imports, and a module can refuse setattr. None of that may take the fix down, and
    a bad neighbour must not stop the module that actually matters from being patched."""
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
