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
STACK_PY = REPO_ROOT / "studio" / "install_python_stack.py"

# Read once: the source-level tests below want these whole files and none of them mutate
# what they read.
INSTALL_SRC = INSTALL_PS1.read_text(encoding = "utf-8")
STACK_SRC = STACK_PY.read_text(encoding = "utf-8")
EXTRAS_SRC = (REPO_ROOT / "studio" / "backend" / "requirements" / "extras.txt").read_text(
    encoding = "utf-8"
)

# Constants of the running interpreter, restated in nearly every test below.
MAJOR, MINOR = sys.version_info[:2]
TAG = f"cp{MAJOR}{MINOR}"


@pytest.fixture(scope = "module")
def ips():
    spec = importlib.util.spec_from_file_location("_ips_wheelhouse_tags", STACK_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse = True)
def _fresh_find_links(ips):
    """Empty the find-links listing before and after every test in this file.

    It is memoized for the process and `ips` is module scoped, so a listing read under one
    test's UV_FIND_LINKS would otherwise be the answer the next test got. Both names share
    one cache: install_python_stack.py assigns _find_links_wheel_names.cache_clear from
    _find_links_wheel_versions. Tests that change the wheelhouse mid-test still clear it
    themselves, which this cannot do for them.
    """
    ips._find_links_wheel_versions.cache_clear()
    yield
    ips._find_links_wheel_versions.cache_clear()


#: What `_pip_config_index_policy` reports when pip's files set no index key.
PIP_FILES_SILENT = {
    "no_index": None,
    "index_url": None,
    "extra_index_urls": [],
    "unreadable": False,
}


@pytest.fixture(autouse = True)
def _pip_files_silent(ips, monkeypatch):
    """Keep the host's pip configuration files out of every test but the reader's own.

    The pip path now consults `pip config list`, so a site pip.conf on the machine running
    the suite would otherwise decide rows about the environment. Yields the real reader for
    the class that tests it.
    """
    real = ips._pip_config_index_policy
    monkeypatch.setattr(ips, "_pip_config_index_policy", lambda: dict(PIP_FILES_SILENT))
    yield real


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
        assert ips._wheel_matches_interpreter(_wheel("tiktoken", TAG, TAG))

    @pytest.mark.parametrize("offset", [-2, -1, 1, 2])
    def test_other_minors_do_not_match(self, ips, offset):
        TAG = f"cp{MAJOR}{MINOR + offset}"
        assert not ips._wheel_matches_interpreter(_wheel("tiktoken", TAG, TAG))

    def test_pure_python_any_matches(self, ips):
        assert ips._wheel_matches_interpreter("six-1.17.0-py2.py3-none-any.whl")

    def test_abi3_is_forward_compatible(self, ips):
        free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        matched = ips._wheel_matches_interpreter(_wheel("cffi", f"cp{MAJOR}2", "abi3"))
        # The stable ABI is not implemented on free-threaded builds.
        assert matched is not free_threaded
        assert not ips._wheel_matches_interpreter(_wheel("cffi", f"cp{MAJOR}{MINOR + 1}", "abi3"))

    @pytest.mark.parametrize("gil_disabled", [0, 1])
    def test_an_exact_minor_abi3_wheel_follows_the_build(self, ips, monkeypatch, gil_disabled):
        """
        The exact-MINOR branch used to accept "abi3" outright, shadowing the guarded
        branch below it, so cp313-abi3 was installable on 3.13t. Free-threaded builds do
        not implement the stable ABI (CPython #111506, PEP 703) -- and uv excludes abi3
        wheels there for the same reason -- so accepting one marked a blocker available,
        dropped its skip, and sent the resolver at a wheel it cannot use.

        Simulated in both directions rather than read off this interpreter, which is
        whichever build happens to be running the suite.
        """
        real = ips.sysconfig.get_config_var
        monkeypatch.setattr(
            ips.sysconfig,
            "get_config_var",
            lambda name: gil_disabled if name == "Py_GIL_DISABLED" else real(name),
        )
        exact = f"cp{MAJOR}{MINOR}"
        abi3_wheel = _wheel("cffi", exact, "abi3")
        assert ips._wheel_matches_interpreter(abi3_wheel) is (not gil_disabled)
        # The TAG a free-threaded build CAN install, and the one a GIL build cannot.
        ft_wheel = _wheel("cffi", exact, f"{exact}t")
        assert ips._wheel_matches_interpreter(ft_wheel) is bool(gil_disabled)
        # And the forward-compatible spelling stays gated the same way.
        assert ips._wheel_matches_interpreter(_wheel("cffi", f"cp{MAJOR}2", "abi3")) is (
            not gil_disabled
        )

    def test_foreign_platform_does_not_match(self, ips):
        assert not ips._wheel_matches_interpreter(
            _wheel("brotli", TAG, TAG, plat = "some_other_platform")
        )

    def test_unparseable_name_is_not_installable(self, ips):
        assert not ips._wheel_matches_interpreter("garbage.whl")


class TestWheelhouseSkipList:
    def test_a_foreign_tagged_wheel_does_not_clear_the_skip(self, ips, tmp_path, monkeypatch):
        other = f"cp{MAJOR}{MINOR + 1}"
        (tmp_path / _wheel("tiktoken", other, other)).write_bytes(b"")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        assert "tiktoken" not in ips._find_links_wheel_names()
        assert "tiktoken" in ips._windows_arm64_skip_packages()

    def test_a_matching_wheel_clears_the_skip(self, ips, tmp_path, monkeypatch):
        (tmp_path / _wheel("tiktoken", TAG, TAG)).write_bytes(b"")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        assert "tiktoken" not in ips._windows_arm64_skip_packages()


class TestBlockerMap:
    def test_every_key_is_canonical(self, ips):
        for key in ips.WINDOWS_ARM64_SKIP_UNBLOCKED_BY:
            assert key == ips._canonical_dist_name(key), f"{key} is looked up canonically"

    def test_every_key_is_a_package_that_is_actually_skipped(self, ips):
        skipped = {ips._canonical_dist_name(p) for p in ips.WINDOWS_ARM64_SKIP_PACKAGES}
        assert set(ips.WINDOWS_ARM64_SKIP_UNBLOCKED_BY) <= skipped

    def test_whisper_needs_tiktoken_as_well_as_the_numba_chain(self, ips):
        # Whisper's metadata requires tiktoken, so hosting llvmlite alone re-enables the sdist.
        blockers = ips.WINDOWS_ARM64_SKIP_UNBLOCKED_BY[ips._canonical_dist_name("openai-whisper")]
        assert "tiktoken" in blockers
        assert "numba" in blockers

    def test_one_hosted_blocker_is_not_enough(self, ips, tmp_path, monkeypatch):
        (tmp_path / _wheel("llvmlite", TAG, TAG)).write_bytes(b"")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        skipped = ips._windows_arm64_skip_packages()
        assert "librosa" in skipped, "librosa still needs numba"
        assert "openai-whisper" in skipped, "whisper still needs numba and tiktoken"


class TestInstallPs1Mirror:
    """install.ps1 builds the same availability set for its requirement overrides."""

    def test_wheel_names_are_filtered_by_interpreter_tag(self):
        block = INSTALL_SRC[INSTALL_SRC.index("$WoaWheelNames = @{}") :]
        block = block[: block.index("$WoaDropCandidates")]
        assert "$WoaWheelTag" in block, "the distribution name alone is not proof of availability"
        assert re.search(r"if \(\$parts\.Count -lt 5\) \{ continue \}", block)
        assert "abi3" in block and "^py3" in block

    def test_uv_override_is_space_safe(self):
        # uv reads UV_OVERRIDE as a space-separated list, so EVERY entry needs the 8.3 helper.
        assert re.search(
            r"\$_woaOverrideValue\s*=\s*@\(Get-UvSafePath\s+\$WoaOverrides\)", INSTALL_SRC
        )
        assert "$_woaOverrideValue += (Get-UvSafePath $_woaKeepFile)" in INSTALL_SRC
        assert re.search(r'\$env:UV_OVERRIDE\s*=\s*\(\$_woaOverrideValue -join " "\)', INSTALL_SRC)
        assert not re.search(r"\$env:UV_OVERRIDE\s*=\s*\$WoaOverrides\s*$", INSTALL_SRC, flags = re.M)

    def test_the_selected_torch_index_is_redacted(self):
        for line in INSTALL_SRC.splitlines():
            if "torch index:" in line:
                assert "Remove-IndexUrlCredentials" in line, line.strip()


class TestBlockersDecideEvenWhenThePackageItselfIsHosted:
    """A package skipped for its DEPENDENCIES is not re-enabled by its own wheel."""

    @pytest.fixture
    def skips(self, ips, tmp_path, monkeypatch):
        """The skip list, given a wheelhouse holding exactly the wheels named."""

        def build(*specs):
            for spec in specs:
                # A spec may carry its own version: a blocker with a floor is only usable
                # at or above it.
                name, py, abi, plat = spec[:4]
                version = spec[4] if len(spec) > 4 else "1.0.0"
                (tmp_path / f"{name}-{version}-{py}-{abi}-{plat}.whl").write_bytes(b"")
            monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
            monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
            return ips._windows_arm64_skip_packages()

        return build

    def test_hosting_tensorboard_without_grpcio_keeps_the_skip(self, skips):
        assert "tensorboard" in skips(("tensorboard", "py3", "none", "any"))

    def test_hosting_both_lifts_it(self, skips):
        # grpcio at tensorboard's own floor: it requires grpcio>=1.74.0.
        assert "tensorboard" not in skips(
            ("tensorboard", "py3", "none", "any"),
            ("grpcio", TAG, TAG, _this_platform(), "1.74.0"),
        )

    def test_a_blocker_below_the_floor_keeps_the_skip(self, skips):
        """tensorboard 2.21.0 requires grpcio>=1.74.0, and nothing else can serve it here.

        Hosting 1.60.0 used to lift the skip on the name alone; the extras pass then failed
        on tensorboard's own metadata instead of leaving one optional feature disabled.
        """
        assert "tensorboard" in skips(
            ("tensorboard", "py3", "none", "any"),
            ("grpcio", TAG, TAG, _this_platform(), "1.60.0"),
        )

    def test_librosa_needs_numba_as_well_as_llvmlite(self, skips):
        assert "librosa" in skips(
            ("librosa", "py3", "none", "any"),
            ("llvmlite", TAG, TAG, _this_platform()),
        )

    def test_a_package_with_no_blockers_still_lifts_on_its_own_wheel(self, skips):
        assert "tiktoken" not in skips(("tiktoken", TAG, TAG, _this_platform()))


class TestFreeThreadedWheelsAreNotOfferedToTheRegularInterpreter:
    """cp313-cp313t is built for the free-threaded build; uv rejects it on cp313."""

    def test_python_side_rejects_a_free_threaded_abi(self, ips):
        free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        matched = ips._wheel_matches_interpreter(_wheel("tiktoken", TAG, f"{TAG}t"))
        assert matched is free_threaded

    def test_install_ps1_checks_the_abi_not_just_the_python_tag(self):
        block = INSTALL_SRC[INSTALL_SRC.index("$WoaWheelNames = @{}") :]
        block = block[: block.index("$WoaDropCandidates")]
        first = block[block.index("foreach ($pyTag in") :]
        first = first[: first.index("$compatible = $true; break") + 30]
        # $WoaWheelAbi, not $WoaWheelTag: a free-threaded venv installs cp313t but is tagged cp313.
        assert (
            "$abiTags -contains $WoaWheelAbi" in first
        ), "the exact-python-TAG branch must also require a usable ABI"
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
        (wheelhouse / _wheel("tiktoken", TAG, TAG, version = have)).write_bytes(b"")
        req = self._req(wheelhouse.parent, f"{pin}\n")
        assert ("tiktoken" in ips._windows_arm64_skip_packages(req)) is still_skipped, why

    def test_any_hosted_version_that_satisfies_is_enough(self, ips, wheelhouse):
        for version in ("0.12.0", "0.13.0"):
            (wheelhouse / _wheel("tiktoken", TAG, TAG, version = version)).write_bytes(b"")
        req = self._req(wheelhouse.parent, "tiktoken==0.13.0\n")
        assert "tiktoken" not in ips._windows_arm64_skip_packages(req)

    def test_a_blocker_with_no_line_of_its_own_is_checked_against_its_floor(self, ips, wheelhouse):
        """grpcio arrives transitively, so extras.txt has no grpcio line to satisfy.

        That absence used to mean any hosted version counted. It does not: the floor comes
        from the optional package's own metadata, which is what rejects a too-old blocker
        after the skip has been dropped.
        """
        (wheelhouse / _wheel("grpcio", TAG, TAG, version = "1.60.0")).write_bytes(b"")
        req = self._req(wheelhouse.parent, "tensorboard==2.21.0\n")
        assert "tensorboard" in ips._windows_arm64_skip_packages(
            req
        ), "grpcio 1.60.0 is below tensorboard's grpcio>=1.74.0"

    def test_a_blocker_at_its_floor_lifts_the_skip(self, ips, wheelhouse):
        (wheelhouse / _wheel("grpcio", TAG, TAG, version = "1.74.0")).write_bytes(b"")
        req = self._req(wheelhouse.parent, "tensorboard==2.21.0\n")
        assert "tensorboard" not in ips._windows_arm64_skip_packages(req)

    def test_a_blocker_with_no_floor_keeps_the_name_only_answer(self, ips, wheelhouse):
        """llvmlite has no entry: nothing states a floor for it, so a guess is not made."""
        for dist, version in (("llvmlite", "0.1.0"), ("numba", "0.62.0"), ("soxr", "1.0.0")):
            (wheelhouse / _wheel(dist, TAG, TAG, version = version)).write_bytes(b"")
        req = self._req(wheelhouse.parent, "librosa==0.11.0\n")
        assert "librosa" not in ips._windows_arm64_skip_packages(req)

    def test_the_floors_name_the_pins_they_were_read_from(self, ips):
        """A bump to extras.txt has to be a prompt to re-read the metadata.

        The floors come from the optional packages' own requirements, which only that
        version states. Recording the provenance turns a silent drift into a failure here.
        """
        for blocker, (specifier, package, version) in ips.WINDOWS_ARM64_BLOCKER_FLOORS.items():
            assert re.search(rf"(?m)^{re.escape(package)}=={re.escape(version)}\b", EXTRAS_SRC), (
                f"{blocker}'s floor {specifier} was read from {package}=={version}, which "
                f"extras.txt no longer pins -- re-read that release's metadata"
            )
            assert ips._canonical_dist_name(blocker) == blocker, "keys are canonical"

    def test_no_requirements_file_keeps_the_old_answer(self, ips, wheelhouse):
        (wheelhouse / _wheel("tiktoken", TAG, TAG, version = "0.12.0")).write_bytes(b"")
        assert "tiktoken" not in ips._windows_arm64_skip_packages()

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
            # packaging answers these now; None is still the contract without packaging.
            ("1!2.0", "==2.0", False),
            ("1.0", "===1.0", True),
            ("not-a-version", "==1.0", False),
        ],
    )
    def test_the_comparison_itself(self, ips, version, specifier, expected):
        assert ips._version_satisfies(version, specifier) is expected

    def test_pins_are_read_canonically_and_markers_evaluated(self, ips, tmp_path):
        req = tmp_path / "r.txt"
        # One marker true on every host and one true on none, so the answer cannot depend on the box.
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

    # One marker true on every host and one true on none; the real pair is host-dependent.
    ACTIVE = 'python_version >= "3"'
    INACTIVE = 'sys_platform == "nonesuch"'

    @classmethod
    def _rows(cls) -> str:
        return f"MeCab==0.996.13; {cls.ACTIVE}\nMeCab==0.996.5; {cls.INACTIVE}\n"

    def test_the_shipped_file_really_has_the_duplicate(self):
        rows = [line for line in EXTRAS_SRC.splitlines() if line.lower().startswith("mecab")]
        assert len(rows) == 2, "the case this fix exists for"

    def test_only_the_active_row_is_kept(self, ips, tmp_path):
        req = tmp_path / "extras.txt"
        req.write_text(self._rows(), encoding = "utf-8")
        pins = ips._requirement_pins(req)
        assert pins["mecab"] == ["==0.996.13"], "the inactive row must not overwrite it"

    def test_an_inactive_row_cannot_unskip(self, ips, tmp_path, monkeypatch):
        wheels = tmp_path / "wheels"
        wheels.mkdir()
        (wheels / _wheel("mecab", TAG, TAG, version = "0.996.5")).write_bytes(b"")
        req = tmp_path / "extras.txt"
        req.write_text(self._rows(), encoding = "utf-8")
        monkeypatch.setenv("UV_FIND_LINKS", str(wheels))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        assert "mecab" in ips._windows_arm64_skip_packages(
            req
        ), "the hosted 0.996.5 satisfies only the row that does not apply here"

    def test_the_active_row_still_unskips(self, ips, tmp_path, monkeypatch):
        wheels = tmp_path / "wheels"
        wheels.mkdir()
        (wheels / _wheel("mecab", TAG, TAG, version = "0.996.13")).write_bytes(b"")
        req = tmp_path / "extras.txt"
        req.write_text(self._rows(), encoding = "utf-8")
        monkeypatch.setenv("UV_FIND_LINKS", str(wheels))
        monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
        assert "mecab" not in ips._windows_arm64_skip_packages(req)

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
        TAG = _this_platform()
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if "win_arm64" not in TAG:
            monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "tiktoken" in name)
        (tmp_path / f"tiktoken-0.13.0rc1-{py}-{py}-win_arm64.whl").write_bytes(b"PK\x03\x04")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        req = tmp_path / "extras.txt"
        req.write_text("tiktoken==0.13.0\n", encoding = "utf-8")
        skipped = ips._windows_arm64_skip_packages(req = req)
        assert "tiktoken" in skipped, (
            "an rc wheel satisfied an exact pin, so tiktoken was unskipped and the "
            "resolve fell to the sdist"
        )


class TestAnExplicitPinIsNotOverriddenByThePreservationShortcut:
    """The ARM64 CUDA-preservation shortcut distrusts the INFERRED expectation, not a pin.

    A native win_arm64 venv holding cu134 has a family TAG download.pytorch.org does not
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
        assert (
            "if _is_win_arm64_interpreter() and _explicit_torch_index_url() is None:" in STACK_SRC
        )
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
        TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
        (tmp_path / _wheel("tiktoken", TAG, TAG, _this_platform(), "0.13.0")).write_bytes(b"")
        for name, value in (("UV_FIND_LINKS", uv_value), ("PIP_FIND_LINKS", pip_value)):
            if value is None:
                monkeypatch.delenv(name, raising = False)
            else:
                monkeypatch.setenv(name, value)
        req = tmp_path / "extras.txt"
        req.write_text("tiktoken==0.13.0\n", encoding = "utf-8")
        return ips._windows_arm64_skip_packages(req = req)

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
        assert "UV_FIND_LINKS" in INSTALL_SRC and "PIP_FIND_LINKS" in INSTALL_SRC
        assert '"UV_FIND_LINKS" = ","' in INSTALL_SRC, "and each keeps its own separator"


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

    @staticmethod
    def _calls(
        ips,
        monkeypatch,
        hosted,
        install_ok = True,
        **stubs,
    ):
        """Run the step with `hosted` standing in for the wheelhouse listing."""
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheelhouse_best_version", lambda name, floor: hosted.get(name))
        monkeypatch.setattr(ips, "_wheelhouse_hosts", lambda name: name in hosted)
        monkeypatch.setattr(ips, "_note", lambda *a, **kw: None)
        calls = []
        monkeypatch.setattr(
            ips, "pip_install_try", lambda label, *a, **kw: calls.append(a) or install_ok
        )
        for name, value in stubs.items():
            monkeypatch.setattr(ips, name, value)
        ips._install_wheelhouse_optionals()
        return calls

    def test_a_hosted_optional_is_installed(self, ips, monkeypatch):
        calls = self._calls(ips, monkeypatch, {"hf-transfer": "0.1.9"})
        assert len(calls) == 1, calls
        assert "hf-transfer==0.1.9" in calls[0], "a bare name installs whatever is hosted"
        assert "--no-deps" in calls[0], "resolving here could walk torch off the CUDA build"

    def test_nothing_is_installed_without_a_hosted_wheel(self, ips, monkeypatch):
        assert self._calls(ips, monkeypatch, {}) == []

    def test_no_other_platform_is_touched(self, ips, monkeypatch):
        """Every non-win_arm64 host must install exactly what it installed before."""
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: False)
        monkeypatch.setattr(ips, "_wheelhouse_best_version", lambda name, floor: "9.9.9")
        monkeypatch.setattr(
            ips, "pip_install_try", lambda *a, **kw: pytest.fail("installed off win_arm64")
        )
        ips._install_wheelhouse_optionals()

    def test_a_failed_optional_does_not_fail_the_install(self, ips, monkeypatch):
        """It is an optional feature: off is where it already was."""
        self._calls(ips, monkeypatch, {"hf-transfer": "0.1.9"}, install_ok = False)

    def test_a_wheel_below_the_declared_floor_is_not_installed(self, ips, monkeypatch, tmp_path):
        """xformers>=0.0.22.post7 is what pyproject.toml asks for; 0.0.20 satisfies nobody."""
        (tmp_path / _wheel("xformers", TAG, TAG, version = "0.0.20")).write_text("")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        assert ips._wheelhouse_best_version("xformers", ">=0.0.22.post7") is None
        (tmp_path / _wheel("xformers", TAG, TAG, version = "0.0.31")).write_text("")
        ips._find_links_wheel_versions.cache_clear()
        assert ips._wheelhouse_best_version("xformers", ">=0.0.22.post7") == "0.0.31"

    def test_the_newest_clearing_wheel_wins(self, ips, monkeypatch, tmp_path):
        # Both clear the floor: 0.0.100 is the newer release and the SMALLER of the two as text.
        for version in ("0.0.23", "0.0.100"):
            (tmp_path / _wheel("xformers", TAG, TAG, version = version)).write_text("")
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        assert (
            ips._wheelhouse_best_version("xformers", ">=0.0.22.post7") == "0.0.100"
        ), "sorted as text 0.0.23 would win"

    def test_an_xformers_built_for_another_torch_is_removed(self, ips, monkeypatch):
        """Its extension links against one exact pair; beside any other the ops vanish
        behind a log line, which would otherwise be reported here as installed."""
        removed = []
        self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"},
            _resident_xformers_build_torch = lambda: "2.9.0+cu128",
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert removed == ["xformers"]

    def test_a_failed_refresh_still_evicts_an_xformers_built_for_another_torch(
        self, ips, monkeypatch
    ):
        """The update path: torch moved, the wheelhouse refresh failed, and the copy the old torch
        left behind stayed resident, losing its ops only at import time."""
        removed = []
        self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"},
            install_ok = False,
            _resident_xformers_build_torch = lambda: "2.9.0+cu128",
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert removed == ["xformers"]

    def test_a_failed_refresh_keeps_a_matching_resident_copy(self, ips, monkeypatch):
        removed = []
        self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"},
            install_ok = False,
            _resident_xformers_build_torch = lambda: "2.15.0.dev20260101+cu134",
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert removed == []

    def test_an_evicted_xformers_is_not_reported_installed(self, ips, monkeypatch):
        notes = []
        self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"},
            _resident_xformers_build_torch = lambda: "2.9.0+cu128",
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: True,
            _note = lambda *a, **kw: notes.append(a[0]),
        )
        assert any("removed" in n for n in notes), notes
        assert not any("installed xformers" in n for n in notes), notes

    def test_an_xformers_nothing_hosts_is_still_evicted_beside_another_torch(
        self, ips, monkeypatch
    ):
        """The wheelhouse can stop offering a usable xformers (a refresh dropped it, or the
        floor moved) while the copy an earlier run installed stays resident. The check ran only
        after a hosted attempt, so that copy kept losing its ops at import time."""
        removed = []
        calls = self._calls(
            ips,
            monkeypatch,
            {},
            _resident_xformers_build_torch = lambda: "2.9.0+cu128",
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert calls == [], "nothing hosted, nothing installed"
        assert removed == ["xformers"]

    def test_an_xformers_nothing_hosts_is_kept_beside_its_own_torch(self, ips, monkeypatch):
        removed = []
        self._calls(
            ips,
            monkeypatch,
            {},
            _resident_xformers_build_torch = lambda: "2.15.0.dev20260101+cu134",
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert removed == []

    def test_a_matching_xformers_is_kept(self, ips, monkeypatch):
        removed = []
        self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"},
            _resident_xformers_build_torch = lambda: "2.15.0.dev20260101+cu134",
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert removed == []

    def test_a_wheel_with_no_build_metadata_is_left_alone(self, ips, monkeypatch):
        """No recorded pair is not evidence of a wrong one, and hf_transfer has none."""
        removed = []
        self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"},
            _resident_xformers_build_torch = lambda: None,
            _probe_installed_torch_version = lambda: "2.15.0.dev20260101+cu134",
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert removed == []

    def test_the_step_runs_in_the_install(self, ips):
        """A helper nothing calls re-enables nothing."""
        assert "    _install_wheelhouse_optionals()" in STACK_SRC

    def test_a_hosted_torchcodec_keeps_its_requirement(self, ips):
        """The one line that asks for torchcodec was filtered out before the resolver."""
        guard = STACK_SRC.index("and PLATFORM_LACKS_TORCHCODEC_WHEEL")
        block = STACK_SRC[guard : STACK_SRC.index("_filter_requirements", guard)]
        assert 'not _wheelhouse_hosts("torchcodec")' in block

    def test_the_hosted_check_reads_the_resolvers_own_wheels(self, ips, tmp_path, monkeypatch):
        """And only wheels THIS interpreter could install: the staging copies cp311
        through cp314, and a wheel tagged for another MINOR is invisible to the resolver."""
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        # The listing is memoized for the process, so each state needs its own read.
        ips._find_links_wheel_versions.cache_clear()
        assert not ips._wheelhouse_hosts("torchcodec")
        (
            tmp_path / _wheel("torchcodec", f"cp{MAJOR}{MINOR + 1}", f"cp{MAJOR}{MINOR + 1}")
        ).write_text("")
        ips._find_links_wheel_versions.cache_clear()
        assert not ips._wheelhouse_hosts("torchcodec"), "a foreign-tagged wheel is not hosted"
        (tmp_path / _wheel("torchcodec", TAG, TAG)).write_text("")
        ips._find_links_wheel_versions.cache_clear()
        assert ips._wheelhouse_hosts("torchcodec")


class TestThePublicIndexUnblocksWhatItAlreadyPublishes:
    """The skip list was decided from the local wheelhouse alone.

    llvmlite and numba publish win_arm64 wheels, and cp314 is the only TAG either publishes
    one for. So a native CPython 3.14 ARM64 host has librosa's whole chain resolvable from
    the public index, and the filter dropped librosa anyway.
    """

    def test_the_recorded_versions_clear_the_blocker_floors(self, ips):
        """A recorded version below the floor would unblock nothing but this table."""
        for name, tags in ips.WINDOWS_ARM64_PUBLIC_INDEX_WHEELS.items():
            floor = ips.WINDOWS_ARM64_BLOCKER_FLOORS.get(name)
            if floor is None:
                continue
            for version in tags.values():
                assert (
                    ips._version_satisfies(version, floor[0]) is not False
                ), f"{name} {version} does not satisfy {floor[0]}"

    def test_nothing_is_claimed_off_win_arm64(self, ips, monkeypatch):
        """Every other platform must see exactly the availability it saw before."""
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: False)
        for name in ips.WINDOWS_ARM64_PUBLIC_INDEX_WHEELS:
            assert ips._public_index_win_arm64_versions(name) == set()

    def test_the_tag_has_to_match_this_interpreter(self, ips, monkeypatch):
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "cp314" in name)
        assert ips._public_index_win_arm64_versions("numba") == {"0.67.0"}
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: False)
        assert ips._public_index_win_arm64_versions("numba") == set()

    def test_a_matching_interpreter_unblocks_librosa(self, ips, tmp_path, monkeypatch):
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "cp314" in name)
        # soxr publishes no win_arm64 wheel upstream, so answer for it: this test is about
        # the interpreter tag deciding availability, not about soxr. test_librosa_still_needs_soxr covers that on its own.
        _published = ips._public_index_win_arm64_versions
        monkeypatch.setattr(
            ips,
            "_public_index_win_arm64_versions",
            lambda name: {"1.0.0"} if name == "soxr" else _published(name),
        )
        assert "librosa" not in ips._windows_arm64_skip_packages()
        # And still dropped where the wheels are not published for this build.
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: False)
        ips._find_links_wheel_versions.cache_clear()
        assert "librosa" in ips._windows_arm64_skip_packages()

    def test_librosa_still_needs_soxr(self, ips, tmp_path, monkeypatch):
        """The numba pair is not enough on its own.

        librosa 0.11.0 requires soxr>=0.3.2, and soxr has published no win_arm64 wheel in any
        release, latest included. Unblocking on llvmlite and numba alone put librosa back in
        the extras pass, where soxr would then be built from an sdist: exactly what the skip
        list exists to avoid. Nothing is hosted here, so the index answers, and it has no soxr.
        """
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "cp314" in name)
        assert "librosa" in ips._windows_arm64_skip_packages()

    def test_openai_whisper_still_needs_tiktoken(self, ips, tmp_path, monkeypatch):
        """Its third blocker publishes no win_arm64 wheel, so unblocking two is not enough."""
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "cp314" in name)
        assert "openai-whisper" in ips._windows_arm64_skip_packages()

    def test_an_empty_wheelhouse_no_longer_short_circuits(self, ips, monkeypatch):
        """The early return read "nothing hosted" as "skip everything", which threw the
        public-index answer away before it was asked for."""
        monkeypatch.delenv("UV_FIND_LINKS", raising = False)
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "cp314" in name)
        # With nothing hosted, every blocker has to come off the index, and soxr publishes no
        # win_arm64 wheel at all. Answering for it is what leaves this test about the early
        # return rather than about soxr.
        published = ips._public_index_win_arm64_versions
        monkeypatch.setattr(
            ips,
            "_public_index_win_arm64_versions",
            lambda name: {"1.0.0"} if name == "soxr" else published(name),
        )
        assert "librosa" not in ips._windows_arm64_skip_packages()
        assert "mecab" in ips._windows_arm64_skip_packages(), "the rest still drop"


class TestThePublicIndexClaimNeedsTheIndex:
    """What PyPI publishes is only availability if the resolve will look at PyPI.

    Offline, or pointed at an exclusive corporate index, those wheels are neither cached nor
    served. Unblocking librosa there drops the skip and then fails the whole extras pass on
    an unavailable numba chain, which is the outcome the skip list exists to prevent.
    """

    @pytest.fixture(autouse = True)
    def _native(self, ips, monkeypatch):
        monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "cp314" in name)
        # uv runs the pass; the pip rows below switch it off explicitly.
        monkeypatch.setattr(ips, "USE_UV", True)
        for var in (
            "UV_OFFLINE",
            "UV_NO_INDEX",
            "UV_DEFAULT_INDEX",
            "UV_INDEX_URL",
            "UV_INDEX",
            "UV_EXTRA_INDEX_URL",
            "PIP_INDEX_URL",
            "PIP_NO_INDEX",
            "PIP_EXTRA_INDEX_URL",
        ):
            monkeypatch.delenv(var, raising = False)

    def test_the_default_case_still_claims_them(self, ips):
        assert ips._public_index_win_arm64_versions("numba") == {"0.67.0"}

    @pytest.mark.parametrize(
        "use_uv, var, value",
        [
            (True, "UV_OFFLINE", "1"),
            (True, "UV_NO_INDEX", "1"),
            (True, "UV_DEFAULT_INDEX", "https://pypi.corp.test/simple"),
            (True, "UV_INDEX_URL", "https://pypi.corp.test/simple"),
            (False, "PIP_INDEX_URL", "https://pypi.corp.test/simple"),
            (False, "PIP_NO_INDEX", "1"),
        ],
    )
    def test_an_unreachable_pypi_claims_nothing(self, ips, monkeypatch, use_uv, var, value):
        monkeypatch.setattr(ips, "USE_UV", use_uv)
        monkeypatch.setenv(var, value)
        assert ips._public_index_win_arm64_versions("numba") == set()

    @pytest.mark.parametrize(
        "use_uv, var, value",
        [
            (True, "PIP_INDEX_URL", "https://pypi.corp.test/simple"),
            (True, "PIP_NO_INDEX", "1"),
            (False, "UV_INDEX_URL", "https://pypi.corp.test/simple"),
            (False, "UV_NO_INDEX", "1"),
            (False, "UV_OFFLINE", "1"),
        ],
    )
    def test_the_other_resolvers_variables_do_not_move_it(
        self, ips, monkeypatch, use_uv, var, value
    ):
        """uv never reads PIP_*, pip never reads UV_*: the pass is judged for the one that runs."""
        monkeypatch.setattr(ips, "USE_UV", use_uv)
        monkeypatch.setenv(var, value)
        assert ips._public_index_win_arm64_versions("numba") == {"0.67.0"}

    def test_a_pip_extra_does_not_put_pypi_back_for_uv(self, ips, monkeypatch):
        """The cross-resolver case: an exclusive uv index with PyPI named only for pip."""
        monkeypatch.setenv("UV_INDEX_URL", "https://pypi.corp.test/simple")
        monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://pypi.org/simple")
        assert ips._public_index_win_arm64_versions("numba") == set()
        monkeypatch.setattr(ips, "USE_UV", False)
        assert ips._public_index_win_arm64_versions("numba") == {
            "0.67.0"
        }, "pip: its own default is PyPI"

    def test_a_pypi_mirror_url_still_counts(self, ips, monkeypatch):
        """Replacing the default index with PyPI itself changes nothing about availability."""
        monkeypatch.setenv("UV_DEFAULT_INDEX", "https://pypi.org/simple")
        assert ips._public_index_win_arm64_versions("numba") == {"0.67.0"}

    def test_an_extra_index_is_additive_and_not_consulted(self, ips, monkeypatch):
        """--extra-index-url ADDS to the default, so PyPI is still in play."""
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://pypi.corp.test/simple")
        assert ips._public_index_win_arm64_versions("numba") == {"0.67.0"}

    def test_librosa_goes_back_to_the_skip_list_offline(self, ips, monkeypatch, tmp_path):
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
        # soxr publishes no win_arm64 wheel upstream, so answer for it: this test is about
        # losing the index offline, not about soxr. test_librosa_still_needs_soxr covers that on its own.
        _published = ips._public_index_win_arm64_versions
        monkeypatch.setattr(
            ips,
            "_public_index_win_arm64_versions",
            lambda name: {"1.0.0"} if name == "soxr" else _published(name),
        )
        assert "librosa" not in ips._windows_arm64_skip_packages()
        monkeypatch.setenv("UV_OFFLINE", "1")
        ips._find_links_wheel_versions.cache_clear()
        assert "librosa" in ips._windows_arm64_skip_packages()


class TestUvConfigurationFilesDecideWherePyPIIs:
    """Only environment variables were read, and uv also discovers uv.toml, pyproject [tool.uv],
    and the user and system files. A no-index or exclusive default-index set there still
    unblocked librosa and then failed the extras resolve on a numba the configured source
    does not carry."""

    @pytest.fixture(autouse = True)
    def _clean(self, ips, monkeypatch, tmp_path):
        # uv's configuration files only matter to uv, the resolver that runs the pass here.
        monkeypatch.setattr(ips, "USE_UV", True)
        for var in (
            "UV_OFFLINE",
            "UV_NO_INDEX",
            "PIP_NO_INDEX",
            "UV_DEFAULT_INDEX",
            "UV_INDEX_URL",
            "PIP_INDEX_URL",
            "UV_INDEX",
            "UV_EXTRA_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "UV_NO_CONFIG",
            "UV_CONFIG_FILE",
            "APPDATA",
            "PROGRAMDATA",
        ):
            monkeypatch.delenv(var, raising = False)
        (tmp_path / "proj").mkdir()
        # uv reads the user file from %APPDATA% on Windows and $XDG_CONFIG_HOME elsewhere, so both
        # names point at one directory and the case is real on every platform.
        (tmp_path / "user").mkdir()
        monkeypatch.setenv("APPDATA", str(tmp_path / "user"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user"))
        monkeypatch.chdir(tmp_path / "proj")
        self.tmp = tmp_path

    #: Where _write puts the "user file" row, spelled the same on every platform.
    user_config = "user/uv/uv.toml"

    def _write(self, rel, body):
        p = self.tmp / rel
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_text(body, encoding = "utf-8")

    def test_nothing_configured_is_pypi(self, ips):
        assert ips._public_pypi_is_reachable() is True

    @pytest.mark.parametrize(
        "rel, body, why",
        [
            ("proj/uv.toml", "no-index = true\n", "no-index"),
            (
                "proj/uv.toml",
                'default-index = "https://pypi.corp.test/simple"\n',
                "exclusive default-index",
            ),
            ("proj/uv.toml", 'index-url = "https://pypi.corp.test/simple"\n', "the older spelling"),
            ("proj/uv.toml", "[pip]\nno-index = true\n", "under [pip]"),
            (
                "proj/uv.toml",
                '[[index]]\nurl = "https://pypi.corp.test/simple"\ndefault = true\n',
                "[[index]] default = true",
            ),
            (
                "proj/uv.toml",
                'index = [{ url = "https://pypi.corp.test/simple", default = true }]\n',
                "inline table",
            ),
            ("proj/pyproject.toml", "[tool.uv]\nno-index = true\n", "pyproject [tool.uv]"),
            (
                "proj/pyproject.toml",
                '[tool.uv.pip]\nindex-url = "https://pypi.corp.test/simple"\n',
                "pyproject [tool.uv.pip]",
            ),
            ("uv.toml", "no-index = true\n", "a parent directory"),
            ("user/uv/uv.toml", "no-index = true\n", "the user file"),
        ],
    )
    def test_a_configured_exclusive_source_is_not_pypi(self, ips, rel, body, why):
        self._write(rel, body)
        assert ips._public_pypi_is_reachable() is False, why

    def test_an_extra_index_leaves_pypi_in_play(self, ips):
        self._write("proj/uv.toml", '[[index]]\nurl = "https://pypi.corp.test/simple"\n')
        assert ips._public_pypi_is_reachable() is True

    def test_a_pyproject_without_tool_uv_is_ignored(self, ips):
        self._write("proj/pyproject.toml", '[project]\nname = "x"\n')
        assert ips._public_pypi_is_reachable() is True

    def test_project_outranks_user_for_a_scalar(self, ips):
        self._write("proj/uv.toml", "no-index = false\n")
        self._write(self.user_config, "no-index = true\n")
        # Vacuous unless the user file is somewhere uv would look.
        assert (self.tmp / self.user_config) in [p for p, _ in ips._uv_config_files()]
        assert ips._public_pypi_is_reachable() is True

    def test_uv_toml_beats_pyproject_in_the_same_directory(self, ips):
        self._write("proj/uv.toml", "no-index = false\n")
        self._write("proj/pyproject.toml", "[tool.uv]\nno-index = true\n")
        assert ips._public_pypi_is_reachable() is True

    def test_uv_no_config_discovers_nothing(self, ips, monkeypatch):
        self._write("proj/uv.toml", "no-index = true\n")
        monkeypatch.setenv("UV_NO_CONFIG", "1")
        assert ips._public_pypi_is_reachable() is True

    def test_uv_config_file_names_the_one_file_read(self, ips, monkeypatch):
        self._write("proj/uv.toml", "no-index = false\n")
        self._write("other.toml", "no-index = true\n")
        monkeypatch.setenv("UV_CONFIG_FILE", str(self.tmp / "other.toml"))
        assert ips._public_pypi_is_reachable() is False

    def test_the_environment_outranks_every_file(self, ips, monkeypatch):
        self._write("proj/uv.toml", "no-index = true\n")
        monkeypatch.setenv("UV_DEFAULT_INDEX", "https://pypi.org/simple")
        assert ips._public_pypi_is_reachable() is True

    @pytest.mark.parametrize(
        "body, reachable, why",
        [
            (
                "no-index = false\n[pip]\nno-index = true\n",
                False,
                "[pip].no-index = true beats the top-level false",
            ),
            (
                "no-index = true\n[pip]\nno-index = false\n",
                True,
                "[pip].no-index = false beats the top-level true",
            ),
            (
                'index-url = "https://pypi.org/simple"\n[pip]\nindex-url = "https://pypi.corp.test/simple"\n',
                False,
                "[pip].index-url beats the top-level index-url",
            ),
            (
                'index-url = "https://pypi.corp.test/simple"\n[pip]\nindex-url = "https://pypi.org/simple"\n',
                True,
                "the other way round",
            ),
            (
                '[[index]]\nurl = "https://pypi.corp.test/simple"\ndefault = true\n[pip]\nindex-url = "https://pypi.org/simple"\n',
                False,
                "[[index]] default = true beats [pip].index-url",
            ),
            (
                '[pip]\nindex-url = "https://pypi.corp.test/simple"\n[[index]]\nurl = "https://pypi.org/simple"\ndefault = true\n',
                True,
                "and still does when it comes later in the file",
            ),
            (
                'no-index = true\n[pip]\nindex-url = "https://pypi.org/simple"\n',
                False,
                "no-index disables every registry",
            ),
        ],
    )
    def test_pip_scalars_outrank_top_level_and_index_default_outranks_both(
        self, ips, body, reachable, why
    ):
        """uv pip's own precedence, verified on uv 0.10.7 with a dry-run resolve."""
        self._write("proj/uv.toml", body)
        assert ips._public_pypi_is_reachable() is reachable, why

    @pytest.mark.parametrize(
        "env, reachable, why",
        [
            (
                {
                    "UV_INDEX_URL": "https://pypi.corp.test/simple",
                    "UV_EXTRA_INDEX_URL": "https://pypi.org/simple",
                },
                True,
                "an extra that is PyPI beside a corporate default",
            ),
            (
                {
                    "UV_DEFAULT_INDEX": "https://pypi.corp.test/simple",
                    "UV_INDEX": "https://mirror.test/simple https://pypi.org/simple",
                },
                True,
                "UV_INDEX, space-separated",
            ),
            (
                {
                    "PIP_INDEX_URL": "https://pypi.corp.test/simple",
                    "PIP_EXTRA_INDEX_URL": "https://pypi.org/simple",
                },
                True,
                "pip's variables: uv never reads them, so its default PyPI stands",
            ),
            (
                {
                    "UV_INDEX_URL": "https://pypi.corp.test/simple",
                    "PIP_EXTRA_INDEX_URL": "https://pypi.org/simple",
                },
                False,
                "a pip extra does not put PyPI back for uv",
            ),
            (
                {
                    "UV_INDEX_URL": "https://pypi.corp.test/simple",
                    "UV_EXTRA_INDEX_URL": "https://pypi.org.corp.example/simple",
                },
                False,
                "an extra that is not PyPI changes nothing",
            ),
        ],
    )
    def test_an_extra_index_that_is_pypi_keeps_pypi_in_play(
        self, ips, monkeypatch, env, reachable, why
    ):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert ips._public_pypi_is_reachable() is reachable, why

    @pytest.mark.parametrize(
        "body, reachable, why",
        [
            (
                'index-url = "https://pypi.corp.test/simple"\nextra-index-url = ["https://pypi.org/simple"]\n',
                True,
                "extra-index-url in the file",
            ),
            (
                'index-url = "https://pypi.corp.test/simple"\n[pip]\nextra-index-url = ["https://pypi.org/simple"]\n',
                True,
                "under [pip]",
            ),
            (
                '[[index]]\nurl = "https://pypi.corp.test/simple"\ndefault = true\n\n[[index]]\nurl = "https://pypi.org/simple"\n',
                True,
                "a second [[index]] without default = true",
            ),
            (
                'no-index = true\nextra-index-url = ["https://pypi.org/simple"]\n',
                False,
                "no-index disables extras too",
            ),
            (
                'index-url = "https://pypi.corp.test/simple"\nextra-index-url = ["https://mirror.test/simple"]\n',
                False,
                "an extra that is not PyPI",
            ),
        ],
    )
    def test_a_configured_extra_index_that_is_pypi(self, ips, body, reachable, why):
        self._write("proj/uv.toml", body)
        assert ips._public_pypi_is_reachable() is reachable, why

    def test_the_same_under_tool_uv_pip(self, ips):
        self._write(
            "proj/pyproject.toml", "[tool.uv]\nno-index = false\n[tool.uv.pip]\nno-index = true\n"
        )
        assert ips._public_pypi_is_reachable() is False

    def test_an_unreadable_file_is_not_guessed_at(self, ips):
        self._write("proj/uv.toml", "this is = not [ toml\n")
        assert ips._public_pypi_is_reachable() is False

    @pytest.mark.parametrize(
        "url, is_pypi, why",
        [
            ("https://pypi.org/simple", True, "the default index"),
            ("HTTPS://PYPI.ORG/simple/", True, "case does not matter for a host"),
            ("https://user:token@pypi.org/simple", True, "credentials do not hide the host"),
            ("https://pypi.org.corp.example/simple", False, "a subdomain lookalike"),
            ("https://packages.example/api/pypi/pypi.org/simple", False, "the name in the path"),
            ("https://test.pypi.org/simple", False, "TestPyPI does not carry these packages"),
            ("pypi.org/simple", False, "no scheme, no host"),
            ("", False, "empty"),
        ],
    )
    def test_the_host_decides_not_a_substring(self, ips, monkeypatch, url, is_pypi, why):
        assert ips._url_is_public_pypi(url) is is_pypi, why
        if url:
            monkeypatch.setenv("UV_DEFAULT_INDEX", url)
            assert ips._public_pypi_is_reachable() is is_pypi, "the environment path: " + why
            monkeypatch.delenv("UV_DEFAULT_INDEX")
            self._write("proj/uv.toml", f'default-index = "{url}"\n')
            assert ips._public_pypi_is_reachable() is is_pypi, "the config path: " + why


class TestPipConfigurationFilesDecideWherePyPIIs:
    """The pip path read PIP_* only. pip also reads its site, user and global files, where
    `[global] index-url` or `no-index` replaces PyPI exactly as the variables do; a host with
    such a file had librosa unblocked and the extras pass then failed on numba."""

    CORP = "https://pypi.corp.test/simple"
    PYPI = "https://pypi.org/simple"

    @pytest.fixture(autouse = True)
    def _pip_runs(self, ips, monkeypatch, _pip_files_silent):
        monkeypatch.setattr(ips, "USE_UV", False)
        for var in ("PIP_NO_INDEX", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_CONFIG_FILE"):
            monkeypatch.delenv(var, raising = False)
        monkeypatch.setattr(ips, "_pip_config_index_policy", _pip_files_silent)
        self.ips = ips
        self.monkeypatch = monkeypatch

    def _listing(
        self,
        text,
        returncode = 0,
    ):
        """Stand in for `pip config list` with this output."""

        class Done:
            stdout = text
            stderr = ""

        Done.returncode = returncode
        self.monkeypatch.setattr(self.ips.subprocess, "run", lambda *a, **kw: Done())

    def test_no_index_keys_means_pypi(self):
        self._listing(":env:.cache-dir='/tmp/pip'\nglobal.timeout='60'\n")
        assert self.ips._public_pypi_is_reachable() is True

    @pytest.mark.parametrize(
        "text, reachable, why",
        [
            (f"global.index-url='{CORP}'\n", False, "an exclusive index-url"),
            ("global.no-index='true'\n", False, "no-index"),
            ("global.no-index='false'\n", True, "no-index switched off"),
            (f"install.index-url='{CORP}'\n", False, "under [install]"),
            (f"global.index-url='{PYPI}'\n", True, "PyPI named explicitly"),
            (
                f"global.index-url='{CORP}'\nglobal.extra-index-url='{PYPI}'\n",
                True,
                "an extra that is PyPI beside a corporate default",
            ),
            (
                f"global.index-url='{CORP}'\nglobal.extra-index-url='https://mirror.test/simple\\n{PYPI}'\n",
                True,
                "one of several extras, newline-separated as pip prints them",
            ),
            (
                f"global.index-url='{CORP}'\nglobal.extra-index-url='https://mirror.test/simple'\n",
                False,
                "an extra that is not PyPI",
            ),
            (
                f"global.no-index='true'\nglobal.extra-index-url='{PYPI}'\n",
                False,
                "no-index disables extras too",
            ),
            (
                f"global.index-url='{CORP}'\ninstall.index-url='{PYPI}'\n",
                True,
                "[install] outranks [global]",
            ),
            (
                f"install.index-url='{CORP}'\nglobal.index-url='{PYPI}'\n",
                False,
                "and still does when printed first",
            ),
            (
                "global.no-index='true'\ninstall.no-index='false'\n",
                True,
                "[install].no-index = false beats the global true",
            ),
            (
                f":env:.index-url='{CORP}'\n",
                True,
                ":env: rows mirror PIP_* the caller already read, and the variable is not set",
            ),
            (
                f'global.index-url="{CORP}"\n',
                False,
                "double quotes, should pip ever print them",
            ),
        ],
    )
    def test_what_the_files_set(self, text, reachable, why):
        self._listing(text)
        assert self.ips._public_pypi_is_reachable() is reachable, why

    def test_the_environment_outranks_the_files(self):
        self._listing("global.no-index='true'\n")
        self.monkeypatch.setenv("PIP_INDEX_URL", self.PYPI)
        assert self.ips._public_pypi_is_reachable() is True

    def test_a_pip_extra_from_the_environment_counts_beside_a_file_index(self):
        self._listing(f"global.index-url='{self.CORP}'\n")
        self.monkeypatch.setenv("PIP_EXTRA_INDEX_URL", self.PYPI)
        assert self.ips._public_pypi_is_reachable() is True

    def test_a_pip_config_that_fails_is_not_guessed_at(self):
        self._listing("", returncode = 1)
        assert self.ips._public_pypi_is_reachable() is False

    def test_a_pip_that_cannot_run_is_not_guessed_at(self):
        def boom(*a, **kw):
            raise OSError("no pip")

        self.monkeypatch.setattr(self.ips.subprocess, "run", boom)
        assert self.ips._public_pypi_is_reachable() is False

    def test_uv_never_reads_pip_files(self):
        self._listing("global.no-index='true'\n")
        self.monkeypatch.setattr(self.ips, "USE_UV", True)
        for var in ("UV_OFFLINE", "UV_NO_INDEX", "UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX"):
            self.monkeypatch.delenv(var, raising = False)
        self.monkeypatch.setenv("UV_NO_CONFIG", "1")
        assert self.ips._public_pypi_is_reachable() is True

    @pytest.mark.parametrize(
        "body, reachable",
        [
            ("[global]\nindex-url = https://pypi.corp.test/simple\n", False),
            ("[global]\nno-index = true\n", False),
            (
                "[global]\nindex-url = https://pypi.corp.test/simple\n[install]\nindex-url = https://pypi.org/simple\n",
                True,
            ),
            ("[global]\ntimeout = 60\n", True),
        ],
    )
    def test_the_real_pip_reads_a_real_file(self, tmp_path, body, reachable):
        """End to end through this interpreter's pip: PIP_CONFIG_FILE names the one file read."""
        pytest.importorskip("pip")
        conf = tmp_path / "pip.conf"
        conf.write_text(body, encoding = "utf-8")
        self.monkeypatch.setenv("PIP_CONFIG_FILE", str(conf))
        assert self.ips._public_pypi_is_reachable() is reachable, body

    def test_the_reader_is_the_one_the_pip_path_calls(self):
        body = STACK_SRC[STACK_SRC.index("def _pip_reaches_public_pypi") :]
        body = body[: body.index("\ndef ")]
        assert "_pip_config_index_policy()" in body


class TestSqliteVecIsAnExplicitOptionalToo:
    """Marker-excluded on ARM64 in pyproject.toml and studio.txt alike, and the one
    unconditional line lives in no-torch-runtime.txt, which the torch install never applies:
    a staged wheel was ignored and RAG stayed unavailable."""

    def test_it_is_in_the_explicit_install_map(self, ips):
        assert "sqlite-vec" in ips.WINDOWS_ARM64_WHEELHOUSE_OPTIONALS

    def test_no_torch_requirement_reaches_it_on_the_torch_path(self):
        """The premise: without an explicit install nothing asks for it."""
        for name in ("pyproject.toml", "studio/backend/requirements/studio.txt"):
            text = (REPO_ROOT / name).read_text(encoding = "utf-8")
            rows = [
                l for l in text.splitlines() if "sqlite-vec" in l and not l.strip().startswith("#")
            ]
            assert rows and all("ARM64" in r for r in rows), name
