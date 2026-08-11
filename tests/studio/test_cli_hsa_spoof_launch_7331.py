# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the launch half of issue #7331.

Routing the wheels is only half the fix. libhsakmt writes HSA_OVERRIDE_GFX_VERSION's
major.minor.stepping straight into the KFD node's EngineId and ROCr names the agent
from that, so the variable decides the ISA every later process reports. AMD's
per-gfx index ships single-arch wheels (the distribution beside torch is literally
named rocm_sdk_libraries_gfx1151), so an override naming a different arch leaves the
runtime asking for kernels the install does not contain.

install.sh clears the variable for the one launch it performs itself, but that unset
dies with the installer: `unsloth studio update` runs install_python_stack.py as a
child (studio/setup.sh:1444), so the repair path, the generated launch-studio.sh and
a hand-typed `unsloth studio` (issue #7331's own repro) all still start with the
spoof in place. The CLI is the chokepoint all of them pass through.

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
) -> Path:
    """A venv tree carrying at most one rocm_sdk_libraries_* dist-info."""
    venv = tmp_path / "unsloth_studio"
    sp = (
        venv / "lib" / "python3.12" / "site-packages"
        if layout == "posix"
        else venv / "Lib" / "site-packages"
    )
    sp.mkdir(parents = True)
    if dist is not None:
        (sp / f"{dist}-7.13.0.dist-info").mkdir()
    return venv


@pytest.fixture(autouse = True)
def _no_inherited_override(monkeypatch):
    """The developer box running these tests may export the variable itself, and
    every assertion below is about its presence."""
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising = False)


class TestOverrideParsing:
    """The third copy of the parser (install.sh and install_python_stack.py hold
    the others) has to agree with libhsakmt, which reads the value with
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
        """A parser that drifted from install_python_stack.py's would clear the
        variable on hosts the installer left alone, and vice versa."""
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
        """download.pytorch.org/whl/rocm6.3 wheels are multi-arch and bring no
        rocm_sdk_libraries_* dist, so there is nothing to contradict."""
        venv = _make_venv(tmp_path, None)
        assert studio_cli._installed_rocm_single_arch(venv) is None

    def test_missing_venv_is_not_an_error(self, tmp_path):
        assert studio_cli._installed_rocm_single_arch(tmp_path / "absent") is None


class TestClearingTheContradictingOverride:
    def test_the_reported_host(self, tmp_path, monkeypatch):
        """gfx1151 wheels installed, HSA_OVERRIDE_GFX_VERSION=11.0.0 still exported.
        Every kernel image in that install is gfx1151, so the agent must stop
        answering gfx1100 or the first allocation fails as it did before."""
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
        """The override is frequently the ONLY thing making an unsupported card
        usable against a generic multi-arch index. Clearing it there would break
        a working machine, which is worse than #7331."""
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
        """ROCm on Windows does not honour HSA_OVERRIDE_GFX_VERSION at all, so
        there is no spoof to undo and no reason to touch the user's environment."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Windows")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151", "windows")
        assert studio_cli._clear_hsa_override_contradicting_install(venv) is None
        assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


class TestTheLaunchPathActuallyCallsIt:
    """A helper nothing calls fixes nothing, and the call has to sit ABOVE all
    three launch paths: os.execvp, the Windows Popen, and the in-process
    run_server. Below any of them and the child inherits the spoof anyway."""

    def test_called_before_every_launch_path(self):
        source = Path(studio_cli.__file__).resolve().read_text(encoding = "utf-8")
        call = source.find("_clear_hsa_override_contradicting_install(")
        # The definition comes first; find the CALL, which follows it.
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
        """os.execvp and Popen both pass os.environ through, so popping the key
        there is what reaches the launched process. A local variable would not."""
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        monkeypatch.setattr(studio_cli.platform, "system", lambda: "Linux")
        venv = _make_venv(tmp_path, "rocm_sdk_libraries_gfx1151")
        studio_cli._clear_hsa_override_contradicting_install(venv)
        assert "HSA_OVERRIDE_GFX_VERSION" not in dict(os.environ)


def test_the_installer_and_the_cli_agree_on_the_spoofable_arches():
    """install.sh, install_python_stack.py and this module all key on the RDNA 3.5
    APU arches. A per-gfx index for any of them ships a matching
    rocm_sdk_libraries_* distribution, which is what the CLI keys on instead."""
    install_sh = (Path(studio_cli.__file__).resolve().parents[2] / "install.sh").read_text(
        encoding = "utf-8"
    )
    assert re.search(r"gfx1151\|gfx1150\|gfx1152", install_sh)
