# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows on ARM: a wheelhouse wheel only counts for the interpreter it was built for.

install.ps1 stages every win_arm64 wheel the wheelhouse publishes, cp311 through cp314,
so a filename alone proves nothing: a cp311 tiktoken is invisible to a cp313 resolver.
Counting one as available drops its skip and its requirement override, and the resolve
then falls to an sdist that needs the toolchain this whole path exists to avoid.

Also pins the blocker map's keys to their canonical form. WINDOWS_ARM64_SKIP_UNBLOCKED_BY
is read with _canonical_dist_name, which maps "-" to "_", so an "openai-whisper" key is
never found and the entry silently does nothing.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import sysconfig
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


@pytest.fixture(scope = "module")
def ips():
    spec = importlib.util.spec_from_file_location(
        "_ips_wheelhouse_tags",
        REPO_ROOT / "studio" / "install_python_stack.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _this_platform() -> str:
    return (sysconfig.get_platform() or "").replace("-", "_").replace(".", "_").lower()


def _wheel(
    dist: str,
    py: str,
    abi: str,
    plat: str | None = None,
    version: str = "1.0.0",
) -> str:
    return f"{dist}-{version}-{py}-{abi}-{plat or _this_platform()}.whl"


class TestWheelMatchesInterpreter:
    def test_own_tag_matches(self, ips):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        assert ips._wheel_matches_interpreter(_wheel("tiktoken", tag, tag))

    @pytest.mark.parametrize("offset", [-2, -1, 1, 2])
    def test_other_minors_do_not_match(self, ips, offset):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor + offset}"
        assert not ips._wheel_matches_interpreter(_wheel("tiktoken", tag, tag))

    def test_pure_python_any_matches(self, ips):
        assert ips._wheel_matches_interpreter("six-1.17.0-py2.py3-none-any.whl")

    def test_abi3_is_forward_compatible(self, ips):
        major, minor = sys.version_info[:2]
        free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        matched = ips._wheel_matches_interpreter(_wheel("cffi", f"cp{major}2", "abi3"))
        # The stable ABI is not implemented on free-threaded builds.
        assert matched is not free_threaded
        assert not ips._wheel_matches_interpreter(_wheel("cffi", f"cp{major}{minor + 1}", "abi3"))

    @pytest.mark.parametrize("gil_disabled", [0, 1])
    def test_an_exact_minor_abi3_wheel_follows_the_build(self, ips, monkeypatch, gil_disabled):
        """
        The exact-minor branch used to accept "abi3" outright, shadowing the guarded
        branch below it, so cp313-abi3 was installable on 3.13t. Free-threaded builds do
        not implement the stable ABI (CPython #111506, PEP 703) -- and uv excludes abi3
        wheels there for the same reason -- so accepting one marked a blocker available,
        dropped its skip, and sent the resolver at a wheel it cannot use.

        Simulated in both directions rather than read off this interpreter, which is
        whichever build happens to be running the suite.
        """
        major, minor = sys.version_info[:2]
        real = ips.sysconfig.get_config_var
        monkeypatch.setattr(
            ips.sysconfig,
            "get_config_var",
            lambda name: gil_disabled if name == "Py_GIL_DISABLED" else real(name),
        )
        exact = f"cp{major}{minor}"
        abi3_wheel = _wheel("cffi", exact, "abi3")
        assert ips._wheel_matches_interpreter(abi3_wheel) is (not gil_disabled)
        # The tag a free-threaded build CAN install, and the one a GIL build cannot.
        ft_wheel = _wheel("cffi", exact, f"{exact}t")
        assert ips._wheel_matches_interpreter(ft_wheel) is bool(gil_disabled)
        # And the forward-compatible spelling stays gated the same way.
        assert ips._wheel_matches_interpreter(_wheel("cffi", f"cp{major}2", "abi3")) is (
            not gil_disabled
        )

    def test_foreign_platform_does_not_match(self, ips):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        assert not ips._wheel_matches_interpreter(
            _wheel("brotli", tag, tag, plat = "some_other_platform")
        )

    def test_unparseable_name_is_not_installable(self, ips):
        assert not ips._wheel_matches_interpreter("garbage.whl")


class TestWheelhouseSkipList:
    def test_a_foreign_tagged_wheel_does_not_clear_the_skip(self, ips, tmp_path, monkeypatch):
        major, minor = sys.version_info[:2]
        other = f"cp{major}{minor + 1}"
        (tmp_path / _wheel("tiktoken", other, other)).write_bytes(b"")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        ips._find_links_wheel_names.cache_clear()
        assert "tiktoken" not in ips._find_links_wheel_names()
        assert "tiktoken" in ips._windows_arm64_skip_packages()
        ips._find_links_wheel_names.cache_clear()

    def test_a_matching_wheel_clears_the_skip(self, ips, tmp_path, monkeypatch):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        (tmp_path / _wheel("tiktoken", tag, tag)).write_bytes(b"")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        ips._find_links_wheel_names.cache_clear()
        assert "tiktoken" not in ips._windows_arm64_skip_packages()
        ips._find_links_wheel_names.cache_clear()


class TestBlockerMap:
    def test_every_key_is_canonical(self, ips):
        for key in ips.WINDOWS_ARM64_SKIP_UNBLOCKED_BY:
            assert key == ips._canonical_dist_name(key), f"{key} is looked up canonically"

    def test_every_key_is_a_package_that_is_actually_skipped(self, ips):
        skipped = {ips._canonical_dist_name(p) for p in ips.WINDOWS_ARM64_SKIP_PACKAGES}
        assert set(ips.WINDOWS_ARM64_SKIP_UNBLOCKED_BY) <= skipped

    def test_whisper_needs_tiktoken_as_well_as_the_numba_chain(self, ips):
        # Whisper's metadata requires tiktoken unconditionally, so hosting llvmlite alone
        # re-enables it straight into the tiktoken sdist.
        blockers = ips.WINDOWS_ARM64_SKIP_UNBLOCKED_BY[ips._canonical_dist_name("openai-whisper")]
        assert "tiktoken" in blockers
        assert "numba" in blockers

    def test_one_hosted_blocker_is_not_enough(self, ips, tmp_path, monkeypatch):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        (tmp_path / _wheel("llvmlite", tag, tag)).write_bytes(b"")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        ips._find_links_wheel_names.cache_clear()
        skipped = ips._windows_arm64_skip_packages()
        assert "librosa" in skipped, "librosa still needs numba"
        assert "openai-whisper" in skipped, "whisper still needs numba and tiktoken"
        ips._find_links_wheel_names.cache_clear()


class TestInstallPs1Mirror:
    """install.ps1 builds the same availability set for its requirement overrides."""

    def test_wheel_names_are_filtered_by_interpreter_tag(self):
        source = INSTALL_PS1.read_text(encoding = "utf-8")
        block = source[source.index("$WoaWheelNames = @{}") :]
        block = block[: block.index("$WoaDropCandidates")]
        assert "$WoaWheelTag" in block, "the distribution name alone is not proof of availability"
        assert re.search(r"if \(\$parts\.Count -lt 5\) \{ continue \}", block)
        assert "abi3" in block and "^py3" in block

    def test_uv_override_is_space_safe(self):
        source = INSTALL_PS1.read_text(encoding = "utf-8")
        # uv reads UV_OVERRIDE as a space-separated list, and the default StudioHome sits
        # under %USERPROFILE%, which routinely contains a space.
        assert re.search(r"\$env:UV_OVERRIDE\s*=\s*Get-UvSafePath\s+\$WoaOverrides", source)
        assert not re.search(r"\$env:UV_OVERRIDE\s*=\s*\$WoaOverrides\s*$", source, flags = re.M)

    def test_the_selected_torch_index_is_redacted(self):
        source = INSTALL_PS1.read_text(encoding = "utf-8")
        for line in source.splitlines():
            if "torch index:" in line:
                assert "Remove-IndexUrlCredentials" in line, line.strip()


class TestBlockersDecideEvenWhenThePackageItselfIsHosted:
    """A package skipped for its DEPENDENCIES is not re-enabled by its own wheel.

    tensorboard and librosa publish py3-none-any wheels, and the staging step copies
    ``*-any.whl`` as well as win_arm64, so a wheelhouse can hold the package while still
    lacking grpcio or numba. Unskipping it there sends the resolver to the blocker's
    sdist, which is the unbuildable thing the skip existed to avoid.
    """

    def _wheelhouse(self, tmp_path, *specs):
        for name, py, abi, plat in specs:
            (tmp_path / f"{name}-1.0.0-{py}-{abi}-{plat}.whl").write_bytes(b"")
        return tmp_path

    def _skips(self, ips, tmp_path, monkeypatch, *specs):
        self._wheelhouse(tmp_path, *specs)
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        ips._find_links_wheel_names.cache_clear()
        try:
            return ips._windows_arm64_skip_packages()
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_hosting_tensorboard_without_grpcio_keeps_the_skip(self, ips, tmp_path, monkeypatch):
        skips = self._skips(
            ips,
            tmp_path,
            monkeypatch,
            ("tensorboard", "py3", "none", "any"),
        )
        assert "tensorboard" in skips

    def test_hosting_both_lifts_it(self, ips, tmp_path, monkeypatch):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        skips = self._skips(
            ips,
            tmp_path,
            monkeypatch,
            ("tensorboard", "py3", "none", "any"),
            ("grpcio", tag, tag, _this_platform()),
        )
        assert "tensorboard" not in skips

    def test_librosa_needs_numba_as_well_as_llvmlite(self, ips, tmp_path, monkeypatch):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        skips = self._skips(
            ips,
            tmp_path,
            monkeypatch,
            ("librosa", "py3", "none", "any"),
            ("llvmlite", tag, tag, _this_platform()),
        )
        assert "librosa" in skips

    def test_a_package_with_no_blockers_still_lifts_on_its_own_wheel(
        self, ips, tmp_path, monkeypatch
    ):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        skips = self._skips(
            ips,
            tmp_path,
            monkeypatch,
            ("tiktoken", tag, tag, _this_platform()),
        )
        assert "tiktoken" not in skips


class TestFreeThreadedWheelsAreNotOfferedToTheRegularInterpreter:
    """cp313-cp313t is built for the free-threaded build; uv rejects it on cp313.

    Matching on the python tag alone counted it as available, which dropped the package's
    requirement override and sent the resolve to the sdist the override existed to avoid.
    """

    def test_python_side_rejects_a_free_threaded_abi(self, ips):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        matched = ips._wheel_matches_interpreter(_wheel("tiktoken", tag, f"{tag}t"))
        assert matched is free_threaded

    def test_install_ps1_checks_the_abi_not_just_the_python_tag(self):
        source = INSTALL_PS1.read_text(encoding = "utf-8")
        block = source[source.index("$WoaWheelNames = @{}") :]
        block = block[: block.index("$WoaDropCandidates")]
        first = block[block.index("foreach ($pyTag in") :]
        first = first[: first.index("$compatible = $true; break") + 30]
        # $WoaWheelAbi, not $WoaWheelTag: the ABI a free-threaded venv can install is
        # cp313t while its python tag is still cp313, so the two are only the same string
        # on a GIL build. Requiring the python tag here would keep exactly the wheels such
        # a venv cannot use.
        assert (
            "$abiTags -contains $WoaWheelAbi" in first
        ), "the exact-python-tag branch must also require a usable ABI"
        assert (
            "$WoaWheelStable -and ($abiTags -contains 'abi3')" in first
        ), "and abi3 is not installable on a free-threaded build"


class TestAHostedWheelMustAlsoSatisfyThePin:
    """Name-only was not enough. Every requirement these entries gate is ``==``-pinned, so
    a wheelhouse holding tiktoken 0.12.0 against a ``tiktoken==0.13.0`` line clears the
    skip and hands the resolver a version it cannot use -- it goes to PyPI, finds no
    win_arm64 wheel at 0.13.0, and falls to the sdist this list exists to avoid.
    """

    @staticmethod
    def _req(tmp_path, text):
        req = tmp_path / "extras.txt"
        req.write_text(text, encoding = "utf-8")
        return req

    @pytest.fixture
    def wheelhouse(self, tmp_path, monkeypatch):
        d = tmp_path / "wheels"
        d.mkdir()
        monkeypatch.setenv("UV_FIND_LINKS", str(d))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        return d

    @pytest.mark.parametrize(
        "have, pin, still_skipped, why",
        [
            ("0.12.0", "tiktoken==0.13.0", True, "the pinned version is not the one hosted"),
            ("0.13.0", "tiktoken==0.13.0", False, "an exact match re-enables it"),
            ("0.13", "tiktoken==0.13.0", False, "0.13 and 0.13.0 are the same release"),
            ("0.12.0", "tiktoken>=0.10", False, "a range the hosted wheel satisfies"),
            ("0.9.0", "tiktoken>=0.10", True, "a range it does not"),
            ("0.12.0", "tiktoken", False, "no specifier: nothing to fail"),
            ("0.12.0", "tiktoken===0.12.0", False,
             "arbitrary equality is beyond the comparison, so the name-only answer stands"),
        ],
    )
    def test_the_pin_decides(self, ips, wheelhouse, have, pin, still_skipped, why):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        (wheelhouse / _wheel("tiktoken", tag, tag, version = have)).write_bytes(b"")
        req = self._req(wheelhouse.parent, f"{pin}\n")
        ips._find_links_wheel_names.cache_clear()
        try:
            assert ("tiktoken" in ips._windows_arm64_skip_packages(req)) is still_skipped, why
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_any_hosted_version_that_satisfies_is_enough(self, ips, wheelhouse):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        for version in ("0.12.0", "0.13.0"):
            (wheelhouse / _wheel("tiktoken", tag, tag, version = version)).write_bytes(b"")
        req = self._req(wheelhouse.parent, "tiktoken==0.13.0\n")
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "tiktoken" not in ips._windows_arm64_skip_packages(req)
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_a_blocker_with_no_line_of_its_own_is_still_name_only(self, ips, wheelhouse):
        """grpcio arrives transitively -- extras.txt has no grpcio line to satisfy."""
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        (wheelhouse / _wheel("grpcio", tag, tag, version = "1.60.0")).write_bytes(b"")
        req = self._req(wheelhouse.parent, "tensorboard==2.21.0\n")
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "tensorboard" not in ips._windows_arm64_skip_packages(req), (
                "its only blocker is hosted, and tensorboard itself is pure Python"
            )
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_no_requirements_file_keeps_the_old_answer(self, ips, wheelhouse):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        (wheelhouse / _wheel("tiktoken", tag, tag, version = "0.12.0")).write_bytes(b"")
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "tiktoken" not in ips._windows_arm64_skip_packages()
        finally:
            ips._find_links_wheel_names.cache_clear()

    @pytest.mark.parametrize(
        "version, specifier, expected",
        [
            ("2.11.0", "==2.11.0", True),
            ("2.11.0", "==2.11", True),
            ("2.11.1", "==2.11.*", True),
            ("2.12.0", "==2.11.*", False),
            ("1.4.1", "<=1.4.1", True),
            ("1.4.2", "<=1.4.1", False),
            ("2.3.3", ">=2.0,<3", True),
            ("3.0.0", ">=2.0,<3", False),
            ("1.2.5", "~=1.2", True),
            ("2.0.0", "~=1.2", False),
            ("1.0.0", "!=1.0.0", False),
            ("1.0.1", "!=1.0.0", True),
            ("1.0", "", True),
            ("1!2.0", "==2.0", None),
            ("1.0", "===1.0", None),
            ("not-a-version", "==1.0", None),
        ],
    )
    def test_the_comparison_itself(self, ips, version, specifier, expected):
        assert ips._version_satisfies(version, specifier) is expected

    def test_pins_are_read_canonically_and_markers_dropped(self, ips, tmp_path):
        req = tmp_path / "r.txt"
        req.write_text(
            "# comment\n"
            "-r other.txt\n"
            "Hf_Transfer == 0.1.9 ; sys_platform != 'win32'\n"
            "httpx[brotli]>=0.27\n"
            "local @ file:///x\n",
            encoding = "utf-8",
        )
        pins = ips._requirement_pins(req)
        assert pins["hf_transfer"] == "== 0.1.9"
        assert pins["httpx"] == ">=0.27"
        assert "local" not in pins, "a direct URL has no version to compare"
        assert "-r" not in pins
