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
        # under %USERPROFILE%, which routinely contains a space. The value is now built
        # from more than one path -- ours plus any caller file kept in its own directory
        # -- so what matters is that EVERY entry goes through the 8.3 helper.
        assert re.search(r"\$_woaOverrideValue\s*=\s*@\(Get-UvSafePath\s+\$WoaOverrides\)", source)
        assert "$_woaOverrideValue += (Get-UvSafePath $_woaKeepFile)" in source
        assert re.search(r'\$env:UV_OVERRIDE\s*=\s*\(\$_woaOverrideValue -join " "\)', source)
        assert not re.search(r"\$env:UV_OVERRIDE\s*=\s*\$WoaOverrides\s*$", source, flags = re.M)

    def test_the_selected_torch_index_is_redacted(self):
        source = INSTALL_PS1.read_text(encoding = "utf-8")
        for line in source.splitlines():
            if "torch index:" in line:
                assert "Remove-IndexUrlCredentials" in line, line.strip()


class TestBlockersDecideEvenWhenThePackageItselfIsHosted:
    """A package skipped for its DEPENDENCIES is not re-enabled by its own wheel."""

    def _wheelhouse(self, tmp_path, *specs):
        # A spec may carry its own version, because a blocker with a stated floor is only
        # hosted-and-usable at or above it.
        for spec in specs:
            name, py, abi, plat = spec[:4]
            version = spec[4] if len(spec) > 4 else "1.0.0"
            (tmp_path / f"{name}-{version}-{py}-{abi}-{plat}.whl").write_bytes(b"")
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
            # At tensorboard's own floor: it requires grpcio>=1.74.0, so an older hosted
            # grpcio would be rejected by its metadata after the skip had been dropped.
            ("grpcio", tag, tag, _this_platform(), "1.74.0"),
        )
        assert "tensorboard" not in skips

    def test_a_blocker_below_the_floor_keeps_the_skip(self, ips, tmp_path, monkeypatch):
        """tensorboard 2.21.0 requires grpcio>=1.74.0, and nothing else can serve it here.

        Hosting 1.60.0 used to lift the skip on the name alone; the extras pass then failed
        on tensorboard's own metadata instead of leaving one optional feature disabled.
        """
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        skips = self._skips(
            ips,
            tmp_path,
            monkeypatch,
            ("tensorboard", "py3", "none", "any"),
            ("grpcio", tag, tag, _this_platform(), "1.60.0"),
        )
        assert "tensorboard" in skips

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
    """cp313-cp313t is built for the free-threaded build; uv rejects it on cp313."""

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
    """Name-only was not enough."""

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
            (
                "0.12.0",
                "tiktoken===0.12.0",
                False,
                "arbitrary equality is beyond the comparison, so the name-only answer stands",
            ),
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

    def test_a_blocker_with_no_line_of_its_own_is_checked_against_its_floor(self, ips, wheelhouse):
        """grpcio arrives transitively, so extras.txt has no grpcio line to satisfy.

        That absence used to mean any hosted version counted. It does not: the floor comes
        from the optional package's own metadata, which is what rejects a too-old blocker
        after the skip has been dropped.
        """
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        (wheelhouse / _wheel("grpcio", tag, tag, version = "1.60.0")).write_bytes(b"")
        req = self._req(wheelhouse.parent, "tensorboard==2.21.0\n")
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "tensorboard" in ips._windows_arm64_skip_packages(
                req
            ), "grpcio 1.60.0 is below tensorboard's grpcio>=1.74.0"
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_a_blocker_at_its_floor_lifts_the_skip(self, ips, wheelhouse):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        (wheelhouse / _wheel("grpcio", tag, tag, version = "1.74.0")).write_bytes(b"")
        req = self._req(wheelhouse.parent, "tensorboard==2.21.0\n")
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "tensorboard" not in ips._windows_arm64_skip_packages(req)
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_a_blocker_with_no_floor_keeps_the_name_only_answer(self, ips, wheelhouse):
        """llvmlite has no entry: nothing states a floor for it, so a guess is not made."""
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        for dist, version in (("llvmlite", "0.1.0"), ("numba", "0.62.0")):
            (wheelhouse / _wheel(dist, tag, tag, version = version)).write_bytes(b"")
        req = self._req(wheelhouse.parent, "librosa==0.11.0\n")
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "librosa" not in ips._windows_arm64_skip_packages(req)
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_the_floors_name_the_pins_they_were_read_from(self, ips):
        """A bump to extras.txt has to be a prompt to re-read the metadata.

        The floors come from the optional packages' own requirements, which only that
        version states. Recording the provenance turns a silent drift into a failure here.
        """
        extras = (REPO_ROOT / "studio" / "backend" / "requirements" / "extras.txt").read_text(
            encoding = "utf-8"
        )
        for blocker, (specifier, package, version) in ips.WINDOWS_ARM64_BLOCKER_FLOORS.items():
            assert re.search(rf"(?m)^{re.escape(package)}=={re.escape(version)}\b", extras), (
                f"{blocker}'s floor {specifier} was read from {package}=={version}, which "
                f"extras.txt no longer pins -- re-read that release's metadata"
            )
            assert ips._canonical_dist_name(blocker) == blocker, "keys are canonical"

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
            # packaging answers these now, and correctly: an epoch does not vanish, arbitrary
            # equality matches, and a version it cannot parse is not one a resolver would take.
            # None is still the contract when packaging is absent AND the numeric path applies.
            ("1!2.0", "==2.0", False),
            ("1.0", "===1.0", True),
            ("not-a-version", "==1.0", False),
        ],
    )
    def test_the_comparison_itself(self, ips, version, specifier, expected):
        assert ips._version_satisfies(version, specifier) is expected

    def test_pins_are_read_canonically_and_markers_evaluated(self, ips, tmp_path):
        req = tmp_path / "r.txt"
        # One marker that holds on every host and one that holds on none, so the answer
        # does not depend on the box running the tests: `sys_platform != 'win32'` here
        # would pass on Linux CI and fail on every Windows machine this file is about.
        req.write_text(
            "# comment\n"
            "-r other.txt\n"
            "Hf_Transfer == 0.1.9 ; python_version >= '3'\n"
            "elsewhere == 1.0 ; sys_platform == 'nonesuch'\n"
            "httpx[brotli]>=0.27\n"
            "local @ file:///x\n",
            encoding = "utf-8",
        )
        pins = ips._requirement_pins(req)
        assert pins["hf_transfer"] == ["== 0.1.9"], "its marker holds on every host"
        assert "elsewhere" not in pins, "an inactive marker drops the row"
        assert pins["httpx"] == [">=0.27"]
        assert "local" not in pins, "a direct URL has no version to compare"
        assert "-r" not in pins


class TestDuplicateRequirementRowsAreSplitByMarker:
    """extras.txt states MeCab twice, once per marker."""

    # One marker that holds on every host and one that holds on none. The real pair is
    # `sys_platform != "darwin" or python_version < "3.14"` against its complement, which
    # would make these tests answer differently on a macOS 3.14 box than on Linux CI --
    # the same host dependence that had to be taken out of the pin-reading test above.
    ACTIVE = 'python_version >= "3"'
    INACTIVE = 'sys_platform == "nonesuch"'

    @classmethod
    def _rows(cls) -> str:
        return f"MeCab==0.996.13; {cls.ACTIVE}\n" f"MeCab==0.996.5; {cls.INACTIVE}\n"

    def test_the_shipped_file_really_has_the_duplicate(self):
        text = (REPO_ROOT / "studio" / "backend" / "requirements" / "extras.txt").read_text(
            encoding = "utf-8",
        )
        rows = [line for line in text.splitlines() if line.lower().startswith("mecab")]
        assert len(rows) == 2, "the case this fix exists for"

    def test_only_the_active_row_is_kept(self, ips, tmp_path):
        req = tmp_path / "extras.txt"
        req.write_text(self._rows(), encoding = "utf-8")
        pins = ips._requirement_pins(req)
        assert pins["mecab"] == ["==0.996.13"], "the inactive row must not overwrite it"

    def test_an_inactive_row_cannot_unskip(self, ips, tmp_path, monkeypatch):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        wheels = tmp_path / "wheels"
        wheels.mkdir()
        (wheels / _wheel("mecab", tag, tag, version = "0.996.5")).write_bytes(b"")
        req = tmp_path / "extras.txt"
        req.write_text(self._rows(), encoding = "utf-8")
        monkeypatch.setenv("UV_FIND_LINKS", str(wheels))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "mecab" in ips._windows_arm64_skip_packages(
                req
            ), "the hosted 0.996.5 satisfies only the row that does not apply here"
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_the_active_row_still_unskips(self, ips, tmp_path, monkeypatch):
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        wheels = tmp_path / "wheels"
        wheels.mkdir()
        (wheels / _wheel("mecab", tag, tag, version = "0.996.13")).write_bytes(b"")
        req = tmp_path / "extras.txt"
        req.write_text(self._rows(), encoding = "utf-8")
        monkeypatch.setenv("UV_FIND_LINKS", str(wheels))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        ips._find_links_wheel_names.cache_clear()
        try:
            assert "mecab" not in ips._windows_arm64_skip_packages(req)
        finally:
            ips._find_links_wheel_names.cache_clear()

    def test_markers_are_evaluated_with_packaging(self, ips):
        assert ips._marker_is_active("") is True
        assert ips._marker_is_active('sys_platform == "%s"' % sys.platform) is True
        assert ips._marker_is_active('sys_platform == "nonesuch"') is False
        assert ips._marker_is_active("this is not a marker") is None

    def test_without_packaging_every_clause_is_kept(self, ips, tmp_path, monkeypatch):
        """The fallback is no stricter than the name-only check that came before."""
        monkeypatch.setattr(ips, "_marker_is_active", lambda marker: None)
        req = tmp_path / "extras.txt"
        req.write_text(self._rows(), encoding = "utf-8")
        assert ips._requirement_pins(req)["mecab"] == [
            "==0.996.13",
            "==0.996.5",
        ], "including the one whose marker would otherwise have excluded it"


class TestAPrereleaseWheelDoesNotSatisfyAFinalPin:
    """A wheelhouse holding 0.13.0rc1 must not unskip a package pinned to ==0.13.0.

    The numeric-release comparison reduces both to (0, 13, 0), so the wheel read as
    satisfying the pin, the skip was dropped, and uv -- which applies PEP 440 properly --
    rejected the wheel and fell to the ARM64 sdist the skip list exists to avoid.
    """

    @pytest.mark.parametrize(
        "version, specifier, expected, why",
        [
            ("0.13.0rc1", "==0.13.0", False, "a release candidate is not the release"),
            ("0.13.0", "==0.13.0", True, "the final version still satisfies it"),
            ("0.13.0.dev1", "==0.13.0", False, "nor is a dev build"),
            ("0.13.0rc1", ">=0.12", False, "a prerelease is excluded unless asked for"),
            ("0.13.0", ">=0.12", True, "an ordinary version is unaffected"),
            ("2.11", "==2.11.0", True, "trailing zeros still compare equal"),
        ],
    )
    def test_the_comparison_is_pep_440(self, ips, version, specifier, expected, why):
        assert ips._version_satisfies(version, specifier) is expected, why

    def test_the_fallback_refuses_what_it_cannot_model(self, ips, monkeypatch):
        """With packaging unavailable the numeric path runs, and it must not guess.

        Returning "satisfied" for a version it cannot parse is the failure this fixes, so
        the fallback answers False rather than falling through to the release compare.
        """
        import importlib

        real = importlib.import_module

        def no_packaging(name, *args, **kwargs):
            if "packaging" in name:
                raise ImportError(name)
            return real(name, *args, **kwargs)

        monkeypatch.setattr(ips.importlib, "import_module", no_packaging)
        assert ips._version_satisfies("0.13.0rc1", "==0.13.0") is False
        assert ips._version_satisfies("0.13.0", "==0.13.0") is True

    def test_a_prerelease_wheel_leaves_the_package_skipped(self, ips, tmp_path, monkeypatch):
        """End to end: the wheel is in the wheelhouse, and the skip survives anyway."""
        tag = _this_platform()
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if "win_arm64" not in tag:
            monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "tiktoken" in name)
        (tmp_path / f"tiktoken-0.13.0rc1-{py}-{py}-win_arm64.whl").write_bytes(b"PK\x03\x04")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        ips._find_links_wheel_versions.cache_clear()
        req = tmp_path / "extras.txt"
        req.write_text("tiktoken==0.13.0\n", encoding = "utf-8")
        skipped = ips._windows_arm64_skip_packages(req = req)
        ips._find_links_wheel_versions.cache_clear()
        assert "tiktoken" in skipped, (
            "an rc wheel satisfied an exact pin, so tiktoken was unskipped and the "
            "resolve fell to the sdist"
        )


class TestAnExplicitPinIsNotOverriddenByThePreservationShortcut:
    """The ARM64 CUDA-preservation shortcut distrusts the INFERRED expectation, not a pin.

    A native win_arm64 venv holding cu134 has a family tag download.pytorch.org does not
    publish, so the driver-derived expectation can only disagree and "repairing" it would
    resolve a cu130 with no wheel. But a user who asks for cu129 by URL or family has stated
    where they want to be, and exempting only a /cpu pin left them silently on the old build.
    setup.ps1 already exempts every explicit pin; this is the same rule.
    """

    class _Reached(Exception):
        """Raised where the shortcut used to return, so "got past it" is observable."""

    @pytest.fixture
    def native_arm64_cuda_venv(self, ips, monkeypatch):
        monkeypatch.setattr(ips, "NO_TORCH", False, raising = False)
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_probe_installed_torch_version", lambda: "2.14.0+cu134")

        def reached():
            raise TestAnExplicitPinIsNotOverriddenByThePreservationShortcut._Reached()

        monkeypatch.setattr(ips, "_expected_torch_flavor_tag", reached)
        for name in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
            monkeypatch.delenv(name, raising = False)
        return ips

    def test_an_unpinned_run_still_keeps_the_cuda_build(self, native_arm64_cuda_venv):
        assert native_arm64_cuda_venv._ensure_expected_torch_flavor() is True, (
            "without a pin the shortcut has to hold, or a working native CUDA venv is "
            "repaired into a cu130 with no win_arm64 wheel"
        )

    @pytest.mark.parametrize(
        "name, value",
        [
            ("UNSLOTH_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu129"),
            ("UNSLOTH_TORCH_INDEX_FAMILY", "cu129"),
            ("UNSLOTH_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu"),
            ("UNSLOTH_TORCH_INDEX_URL", "https://mirror.test/simple"),
        ],
    )
    def test_a_pinned_run_is_evaluated(self, native_arm64_cuda_venv, monkeypatch, name, value):
        monkeypatch.setenv(name, value)
        with pytest.raises(TestAnExplicitPinIsNotOverriddenByThePreservationShortcut._Reached):
            native_arm64_cuda_venv._ensure_expected_torch_flavor()

    def test_the_two_installers_agree(self):
        """setup.ps1 exempts every pin; the Python half must not narrow that to /cpu."""
        source = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
        assert "if _is_win_arm64_interpreter() and _explicit_torch_index_url() is None:" in source
        setup = (REPO_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
        assert "-not $_pinnedIdx) {" in setup, "setup.ps1's own preservation guard moved"


class TestOnlyTheResolversOwnLocationsCount:
    """uv does not read PIP_FIND_LINKS, so a wheel hosted only there is not available.

    Counting it dropped the package off the skip list and the uv resolve that followed
    could not see the wheel at all, reaching the sdist the skip exists to avoid. pip cannot
    be the resolver on this path either: pip_install refuses the fallback once the win_arm64
    overrides are in force, because it has nothing to translate them into.
    """

    @staticmethod
    def _skip_with(ips, tmp_path, monkeypatch, uv_value, pip_value):
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        (tmp_path / _wheel("tiktoken", tag, tag, _this_platform(), "0.13.0")).write_bytes(b"")
        for name, value in (("UV_FIND_LINKS", uv_value), ("PIP_FIND_LINKS", pip_value)):
            if value is None:
                monkeypatch.delenv(name, raising = False)
            else:
                monkeypatch.setenv(name, value)
        req = tmp_path / "extras.txt"
        req.write_text("tiktoken==0.13.0\n", encoding = "utf-8")
        ips._find_links_wheel_versions.cache_clear()
        try:
            return ips._windows_arm64_skip_packages(req = req)
        finally:
            ips._find_links_wheel_versions.cache_clear()

    def test_a_pip_only_location_does_not_unskip(self, ips, tmp_path, monkeypatch):
        skips = self._skip_with(ips, tmp_path, monkeypatch, None, str(tmp_path))
        assert "tiktoken" in skips, (
            "the wheel is only where pip would look, and uv is what resolves here, so "
            "counting it sends the resolve to an sdist that cannot build"
        )

    def test_a_uv_location_still_unskips(self, ips, tmp_path, monkeypatch):
        skips = self._skip_with(ips, tmp_path, monkeypatch, str(tmp_path), None)
        assert "tiktoken" not in skips

    def test_install_ps1_sets_both_so_the_managed_wheelhouse_is_unaffected(self):
        """The narrowing must not cost the path it was written for."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert "UV_FIND_LINKS" in text and "PIP_FIND_LINKS" in text
        assert '"UV_FIND_LINKS" = ","' in text, "and each keeps its own separator"


class TestAHostedOptionalIsActuallyInstalled:
    """Omitting the removal override only helps a package something still requires.

    install.ps1 reports "keeping X (the wheelhouse provides a win_arm64 wheel)" and then
    just declines to emit X's AMD64-only override line. For hf_transfer and xformers the
    RELEASED metadata already excludes win_arm64 by marker, so no requirement survives for
    a hosted wheel to satisfy; for torchcodec the only line that asks for it was being
    filtered out here unconditionally. In all three cases hosting a wheel changed nothing.
    """

    def test_the_optionals_are_the_ones_metadata_excludes(self, ips):
        """Named here only because pyproject.toml puts them out of reach on ARM64."""
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
        for name in ips.WINDOWS_ARM64_WHEELHOUSE_OPTIONALS:
            stem = name.replace("-", "[-_]")
            rows = [
                line
                for line in pyproject.splitlines()
                if re.search(rf'"\s*{stem}\b', line) and "ARM64" in line
            ]
            assert rows, f"{name} is no longer excluded on ARM64; the explicit install is stale"

    def test_a_hosted_optional_is_installed(self, ips, tmp_path, monkeypatch):
        (tmp_path / _wheel("hf_transfer", "cp313", "cp313", "win_arm64")).write_text("")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheelhouse_hosts", lambda name: name == "hf-transfer")
        calls = []
        monkeypatch.setattr(ips, "pip_install_try", lambda label, *a, **kw: calls.append(a) or True)
        monkeypatch.setattr(ips, "_note", lambda *a, **kw: None)
        ips._install_wheelhouse_optionals()
        assert len(calls) == 1, calls
        assert "hf-transfer" in calls[0]
        assert "--no-deps" in calls[0], "resolving here could walk torch off the CUDA build"

    def test_nothing_is_installed_without_a_hosted_wheel(self, ips, monkeypatch):
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheelhouse_hosts", lambda name: False)
        monkeypatch.setattr(
            ips, "pip_install_try", lambda *a, **kw: pytest.fail("installed with no wheel")
        )
        ips._install_wheelhouse_optionals()

    def test_no_other_platform_is_touched(self, ips, monkeypatch):
        """Every non-win_arm64 host must install exactly what it installed before."""
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: False)
        monkeypatch.setattr(ips, "_wheelhouse_hosts", lambda name: True)
        monkeypatch.setattr(
            ips, "pip_install_try", lambda *a, **kw: pytest.fail("installed off win_arm64")
        )
        ips._install_wheelhouse_optionals()

    def test_a_failed_optional_does_not_fail_the_install(self, ips, monkeypatch):
        """It is an optional feature: off is where it already was."""
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheelhouse_hosts", lambda name: True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **kw: False)
        monkeypatch.setattr(ips, "_note", lambda *a, **kw: None)
        ips._install_wheelhouse_optionals()

    def test_the_step_runs_in_the_install(self, ips):
        """A helper nothing calls re-enables nothing."""
        source = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
        assert "    _install_wheelhouse_optionals()" in source

    def test_a_hosted_torchcodec_keeps_its_requirement(self, ips):
        """The one line that asks for torchcodec was filtered out before the resolver."""
        source = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
        guard = source.index("and PLATFORM_LACKS_TORCHCODEC_WHEEL")
        block = source[guard : source.index("_filter_requirements", guard)]
        assert 'not _wheelhouse_hosts("torchcodec")' in block

    def test_the_hosted_check_reads_the_resolvers_own_wheels(self, ips, tmp_path, monkeypatch):
        """And only wheels THIS interpreter could install: the staging copies cp311
        through cp314, and a wheel tagged for another minor is invisible to the resolver."""
        major, minor = sys.version_info[:2]
        tag = f"cp{major}{minor}"
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        # The listing is memoized for the process, so each state needs its own read.
        ips._find_links_wheel_versions.cache_clear()
        assert not ips._wheelhouse_hosts("torchcodec")
        (
            tmp_path / _wheel("torchcodec", f"cp{major}{minor + 1}", f"cp{major}{minor + 1}")
        ).write_text("")
        ips._find_links_wheel_versions.cache_clear()
        assert not ips._wheelhouse_hosts("torchcodec"), "a foreign-tagged wheel is not hosted"
        (tmp_path / _wheel("torchcodec", tag, tag)).write_text("")
        ips._find_links_wheel_versions.cache_clear()
        assert ips._wheelhouse_hosts("torchcodec")
