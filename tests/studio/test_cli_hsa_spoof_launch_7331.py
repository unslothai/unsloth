# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the launch half of issue #7331.

Routing the wheels is only half the fix. libhsakmt (topology.c) writes
HSA_OVERRIDE_GFX_VERSION's major.minor.stepping straight into the KFD node's EngineId
and ROCr names the agent from that, so the variable decides the ISA every later
process reports. AMD's per-gfx index ships single-arch wheels (the distribution beside
torch is named rocm_sdk_libraries_gfx1151), so an override naming a different arch
leaves the runtime asking for kernels the install does not contain.

install.sh clears the variable for the one launch it performs itself, but that unset
dies with the installer: `unsloth studio update` runs install_python_stack.py as a
child (studio/setup.sh:1444), so the repair path, the generated launch-studio.sh and a
hand-typed `unsloth studio` (issue #7331's own repro) all still start with the spoof
in place. The CLI is the chokepoint all of them pass through.

There is no AMD hardware and no ROCm CI in this repo; the venv layouts below are
fixtures and nothing here was validated on real silicon.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import unsloth_cli.commands.studio as studio_cli


def _make_venv(
    tmp_path: Path,
    dist: "str | None",
    layout: str = "posix",
    orphans: "tuple[str, ...]" = (),
    torch_needs_rocm: bool = True,
) -> Path:
    """A venv tree shaped like a real AMD per-gfx install.

    ``dist`` is the ACTIVE runtime, so it gets both its own distribution and the
    ``rocm`` meta-package whose Requires-Dist names it -- the real layout, verified
    against repo.amd.com/rocm/whl/gfx1151, where rocm 7.13.0 carries
    ``Requires-Dist: rocm-sdk-libraries-gfx1151==7.13.0; extra == "libraries"``.
    ``None`` is a generic multi-arch index, which installs neither.

    ``orphans`` are superseded runtimes left behind by a family switch: pip has no
    autoremove and the old distribution keeps its own name, so they accumulate.

    ``torch_needs_rocm`` is the dependency edge that makes the meta-package
    authoritative. AMD's per-gfx torch resolves through ``rocm``; the generic
    pytorch.org ROCm wheels vendor their runtime and require nothing, so False with a
    ``dist`` present is the "switched to generic, meta-package orphaned" shape.
    """
    venv = tmp_path / "unsloth_studio"
    sp = (
        venv / "lib" / "python3.12" / "site-packages"
        if layout == "posix"
        else venv / "Lib" / "site-packages"
    )
    sp.mkdir(parents = True)
    _torch = sp / "torch-2.11.0.dist-info"
    _torch.mkdir()
    (_torch / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch\nVersion: 2.11.0\nRequires-Dist: filelock\n"
        + ("Requires-Dist: rocm[libraries]==7.13.0\n" if torch_needs_rocm else ""),
        encoding = "utf-8",
    )
    for _orphan in orphans:
        (sp / f"rocm_sdk_libraries_{_orphan}-7.12.0.dist-info").mkdir()
    if dist is not None:
        (sp / f"{dist}-7.13.0.dist-info").mkdir()
        _family = dist.replace("rocm_sdk_libraries_", "")
        _meta = sp / "rocm-7.13.0.dist-info"
        _meta.mkdir()
        (_meta / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: rocm\nVersion: 7.13.0\n"
            "Provides-Extra: libraries\n"
            f'Requires-Dist: rocm-sdk-libraries-{_family}==7.13.0; extra == "libraries"\n'
            "Requires-Dist: rocm-sdk-core==7.13.0\n",
            encoding = "utf-8",
        )
    return venv


@pytest.fixture(autouse = True)
def _no_inherited_override(monkeypatch):
    """The developer box may export the variable; every assertion is about it."""
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising = False)


class TestOverrideParsing:
    """The third copy of the parser (install.sh and install_python_stack.py hold the
    others) has to agree with libhsakmt, which reads the value with
    sscanf("%u.%u.%u%c") != 3 and concatenates the stepping in HEX."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("11.0.0", "gfx1100"),  # the circulated Strix workaround
            ("11.5.1", "gfx1151"),  # Strix Halo, naming its own arch
            ("10.3.0", "gfx1030"),  # the documented RX 6800 override
            ("9.0.10", "gfx90a"),  # stepping 10 renders as 'a', not "gfx9010"
            ("  11.0.0  ", "gfx1100"),
            ("", None),
            ("garbage", None),
            ("11.0", None),
            ("11.0.0.0", None),
            ("-1.0.0", None),
            ("11.0.16", None),  # stepping wider than one hex nibble
            ("11.10.0", None),
        ],
    )
    def test_reading(self, value, expected):
        assert studio_cli._hsa_override_gfx_arch(value) == expected

    def test_matches_the_installer_copy(self):
        """A drifted parser would clear the variable on hosts the installer left alone."""
        import importlib.util
        import sys

        path = Path(studio_cli.__file__).resolve().parents[2] / "studio" / "install_python_stack.py"
        # setup.sh invokes it by path, so its own directory is sys.path[0] there.
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("_stack_for_7331_cli", path)
        stack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stack)
        for value in ("11.0.0", "11.5.1", "10.3.0", "9.0.10", "garbage", "", "11.0.16", "11.10.0"):
            assert studio_cli._hsa_override_gfx_arch(value) == stack._hsa_override_gfx_arch(
                value
            ), value


class TestInstalledArchReading:
    """The arch comes from the install, not from a hardware probe."""

    @pytest.mark.parametrize("layout", ["posix", "windows"])
    def test_reads_the_per_gfx_distribution(self, tmp_path, layout):
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151", layout)
        assert studio_cli._installed_rocm_single_arch(venv) == "gfx1151"

    def test_generic_index_installs_no_such_distribution(self, tmp_path):
        """Generic pytorch.org wheels are multi-arch and bring no rocm_sdk_libraries_*
        dist, so there is nothing to contradict."""
        venv = _make_venv(tmp_path, None)
        assert studio_cli._installed_rocm_single_arch(venv) is None

    def test_missing_venv_is_not_an_error(self, tmp_path):
        assert studio_cli._installed_rocm_single_arch(tmp_path / "absent") is None


class TestClearingTheContradictingOverride:
    def test_the_reported_host(self, tmp_path, monkeypatch):
        """gfx1151 wheels installed, HSA_OVERRIDE_GFX_VERSION=11.0.0 still exported.
        Every kernel image is gfx1151, so the agent must stop answering gfx1100 or the
        first allocation fails as before."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        assert studio_cli._clear_hsa_override_contradicting_install(venv) == "gfx1151"
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_override_naming_the_installed_arch_is_left_alone(self, tmp_path, monkeypatch):
        """11.5.1 on a gfx1151 install is a no-op remap, not a contradiction."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        assert studio_cli._clear_hsa_override_contradicting_install(venv) is None
        assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"

    def test_generic_wheels_keep_the_override(self, tmp_path, monkeypatch):
        """The override is often the ONLY thing making an unsupported card usable against
        a generic multi-arch index. Clearing it there breaks a working machine, which is
        worse than #7331."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, None)
        assert studio_cli._clear_hsa_override_contradicting_install(venv) is None
        assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"

    def test_unreadable_value_is_not_ours_to_remove(self, tmp_path, monkeypatch):
        """libhsakmt rejects it too (sscanf != 3), so it is not this spoof."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "garbage")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        assert studio_cli._clear_hsa_override_contradicting_install(venv) is None
        assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "garbage"

    def test_unset_variable_does_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        assert studio_cli._clear_hsa_override_contradicting_install(venv) is None

    def test_windows_is_untouched(self, tmp_path, monkeypatch):
        """ROCm on Windows ignores HSA_OVERRIDE_GFX_VERSION, so there is no spoof to undo."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Windows")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151", "windows")
        assert studio_cli._clear_hsa_override_contradicting_install(venv) is None
        assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


class TestTheLaunchPathActuallyCallsIt:
    """The call has to sit ABOVE all three launch paths (os.execvp, the Windows Popen,
    the in-process run_server), or the child inherits the spoof anyway."""

    def test_called_before_every_launch_path(self):
        source = Path(studio_cli.__file__).resolve().read_text(encoding = "utf-8")
        call = source.find("_clear_hsa_override_contradicting_install(")
        # The definition comes first;
        # find the CALL, which follows it.
        call = source.find("_clear_hsa_override_contradicting_install(", call + 1)
        assert call != -1, "the CLI must clear a contradicting override before launching"
        for marker in (
            "os.execvp(str(studio_python), args)",
            "proc = _sp.Popen(",
            "run_server = run_mod.run_server",
        ):
            at = source.find(marker)
            assert at != -1, marker
            assert call < at, f"the clear must precede {marker}"

    def test_it_mutates_the_environment_the_child_inherits(self, tmp_path, monkeypatch):
        """os.execvp and Popen pass os.environ through, so popping the key there is what
        reaches the launched process. A local variable would not."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        studio_cli._clear_hsa_override_contradicting_install(venv)
        assert "HSA_OVERRIDE_GFX_VERSION" not in dict(os.environ)


def test_the_installer_and_the_cli_agree_on_the_spoofable_arches():
    """install.sh, install_python_stack.py and this module all key on the RDNA 3.5 APU
    arches; a per-gfx index for any of them ships the matching rocm_sdk_libraries_*
    distribution the CLI keys on instead."""
    install_sh = (Path(studio_cli.__file__).resolve().parents[2] / "install.sh").read_text(
        encoding = "utf-8"
    )
    assert re.search(r"gfx1151\|gfx1150\|gfx1152", install_sh)


class TestOrphanedRuntimesFromAFamilySwitch:
    """pip never uninstalls the superseded arch-specific runtime across a family switch:
    `rocm` is upgraded in place, but rocm-sdk-libraries-<old> keeps its own name and
    stays on disk, so globbing for that directory reads whichever the filesystem hands
    back and clearing on a stale reading breaks a working machine. Same hazard as
    install_python_stack.py's _installed_rocm_wheel_family."""

    def test_the_active_family_wins_over_an_orphan(self, tmp_path):
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151", orphans = ("gfx1100",))
        assert studio_cli._installed_rocm_single_arch(venv) == "gfx1151"

    def test_an_orphan_alone_arbitrates_nothing(self, tmp_path):
        """Switched to generic wheels: `rocm` is gone, the old runtime is not, so there
        is no active single-arch install and nothing may be cleared."""
        venv = _make_venv(tmp_path, None, orphans = ("gfx1151",))
        assert studio_cli._installed_rocm_single_arch(venv) is None

    def test_switching_to_generic_wheels_keeps_the_override(self, tmp_path, monkeypatch):
        """The failure the orphan causes end to end: a host that once had gfx1151 wheels
        and now runs generic ones would lose its override to a directory nothing uses."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, None, orphans = ("gfx1151",))
        assert studio_cli._clear_hsa_override_contradicting_install(venv) is None
        assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"

    def test_a_multi_arch_family_contradicts_nothing(self, tmp_path):
        """gfx120x-all carries kernels for several ISAs, so an override naming one is
        not asking for code the install lacks."""
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx120X-all")
        assert studio_cli._installed_rocm_single_arch(venv) is None

    def test_two_families_named_by_the_metadata_decline(self, tmp_path):
        """Nothing to arbitrate with; guess and you break one of the two."""
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        _meta = next(venv.rglob("rocm-7.13.0.dist-info")) / "METADATA"
        _meta.write_text(
            _meta.read_text(encoding = "utf-8")
            + 'Requires-Dist: rocm-sdk-libraries-gfx1100==7.13.0; extra == "other"\n',
            encoding = "utf-8",
        )
        assert studio_cli._installed_rocm_single_arch(venv) is None


class TestEveryLaunchEntryPointClearsIt:
    """`unsloth studio` is not the only way in: the group callback returns as soon as a
    subcommand is named and `unsloth run` is bound straight to studio_run, so the two
    commands people actually use would keep the spoof."""

    def test_run_clears_it_itself(self):
        source = Path(studio_cli.__file__).resolve().read_text(encoding = "utf-8")
        _run_at = source.find("\ndef run(\n")
        assert _run_at != -1, "the run command moved"
        assert "_clear_hsa_override_before_launch(" in source[_run_at:], (
            "`unsloth studio run` and the `unsloth run` alias bypass the group "
            "callback's clear, so run() has to perform it itself"
        )

    def test_the_group_callback_uses_the_same_helper(self):
        source = Path(studio_cli.__file__).resolve().read_text(encoding = "utf-8")
        assert (
            source.count("_clear_hsa_override_before_launch(") >= 3
        ), "one definition plus a call from each entry point"

    def test_the_helper_is_idempotent(self, tmp_path, monkeypatch):
        """studio_default and run can both run in one process; the second call must not
        raise on an already-removed key."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        monkeypatch.setattr(studio_cli, "STUDIO_HOME", venv.parent)
        assert studio_cli._clear_hsa_override_before_launch(silent = True) == "gfx1151"
        assert studio_cli._clear_hsa_override_before_launch(silent = True) is None
        assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ

    def test_the_top_level_run_alias_is_the_same_function(self):
        """unsloth_cli/__init__.py binds `unsloth run` to studio_run directly, so the
        group callback is skipped there."""
        import unsloth_cli
        assert unsloth_cli.studio_run is studio_cli.run


def test_a_rocm_metapackage_orphaned_by_a_switch_to_generic_wheels_arbitrates_nothing(tmp_path):
    """A `rocm` meta-package orphaned by a switch to generic wheels must decide nothing.

    The generic pytorch.org ROCm wheels vendor their own runtime and depend on no
    meta-package, and pip has no autoremove, so `rocm` survives the switch describing the
    family the OLD torch resolved. Trusting it would clear an override the generic wheels
    may be the only reason the GPU works at all. Torch's own requirements are the
    discriminator, so the identical tree with an AMD torch still arbitrates.
    """
    from unsloth_cli.commands.studio import _installed_rocm_single_arch

    live = _make_venv(tmp_path / "live", "rocm_sdk_libraries_gfx1151")
    assert _installed_rocm_single_arch(live) == "gfx1151"

    orphaned = _make_venv(
        tmp_path / "orphaned", "rocm_sdk_libraries_gfx1151", torch_needs_rocm = False
    )
    assert _installed_rocm_single_arch(orphaned) is None
