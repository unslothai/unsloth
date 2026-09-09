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

#: The two index URLs the tables below distinguish: PyPI itself, and an exclusive mirror.
PYPI = "https://pypi.org/simple"
CORP = "https://pypi.corp.test/simple"

#: Every variable either resolver reads for its index configuration, cleared so the box
#: running the suite cannot decide a row.
RESOLVER_INDEX_VARS = tuple(
    "UV_OFFLINE UV_NO_INDEX UV_DEFAULT_INDEX UV_INDEX_URL UV_INDEX UV_EXTRA_INDEX_URL "
    "PIP_INDEX_URL PIP_NO_INDEX PIP_EXTRA_INDEX_URL".split()
)


@pytest.fixture(scope = "module")
def ips():
    spec = importlib.util.spec_from_file_location("_ips_wheelhouse_tags", STACK_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse = True)
def _fresh_find_links(ips):
    """Empty the memoized find-links listing around every test in this file.

    `ips` is module scoped, so one test's UV_FIND_LINKS would otherwise be the answer the
    next test got. Both names share one cache, and a test that changes the wheelhouse
    mid-test still has to clear it itself.
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
    """Keep the host's own pip.conf from deciding rows about the environment.

    The pip path consults `pip config list`. Yields the real reader for the class that
    tests it.
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


def _stage(
    directory: Path,
    dist: str,
    version: str = "1.0.0",
    py: str | None = None,
    **kw,
) -> None:
    """Put one wheel in `directory`, tagged for this interpreter unless told otherwise."""
    py = py or TAG
    (directory / _wheel(dist, py, kw.pop("abi", py), version = version, **kw)).write_bytes(b"")


def _req(directory: Path, text: str) -> Path:
    """A requirements file the skip list can read its pins from."""
    path = directory / "extras.txt"
    path.write_text(text, encoding = "utf-8")
    return path


@pytest.fixture
def wheelhouse(tmp_path, monkeypatch):
    """An empty wheelhouse that is the only find-links location the uv resolver reads."""
    d = tmp_path / "wheels"
    d.mkdir()
    monkeypatch.setenv("UV_FIND_LINKS", str(d))
    monkeypatch.delenv("PIP_FIND_LINKS", raising = False)
    return d


@pytest.fixture
def native_cp314(ips, monkeypatch):
    """A native win_arm64 interpreter that installs exactly the cp314 wheels: the only tag
    llvmlite and numba publish a win_arm64 wheel for, so the index can unblock librosa."""
    monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: True)
    monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "cp314" in name)


@pytest.fixture
def soxr_published(ips, monkeypatch):
    """Answer the index for soxr, which publishes no win_arm64 wheel in any release, so it
    alone would decide tests that are about something else. test_librosa_still_needs_soxr
    covers soxr itself."""
    real = ips._public_index_win_arm64_versions
    monkeypatch.setattr(
        ips,
        "_public_index_win_arm64_versions",
        lambda name: {"1.0.0"} if name == "soxr" else real(name),
    )


def _woa_wheel_names_block() -> str:
    """install.ps1's own availability scan, from $WoaWheelNames to $WoaDropCandidates."""
    block = INSTALL_SRC[INSTALL_SRC.index("$WoaWheelNames = @{}") :]
    return block[: block.index("$WoaDropCandidates")]


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
        """The exact-MINOR branch accepted "abi3" outright, so cp313-abi3 was installable on
        3.13t, which implements no stable ABI (CPython #111506, PEP 703): the skip was dropped
        and the resolver sent at a wheel it cannot use. Simulated in both directions rather
        than read off whichever build is running the suite.
        """
        real = ips.sysconfig.get_config_var
        monkeypatch.setattr(
            ips.sysconfig,
            "get_config_var",
            lambda name: gil_disabled if name == "Py_GIL_DISABLED" else real(name),
        )
        exact = f"cp{MAJOR}{MINOR}"
        assert ips._wheel_matches_interpreter(_wheel("cffi", exact, "abi3")) is (not gil_disabled)
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
    def test_a_foreign_tagged_wheel_does_not_clear_the_skip(self, ips, wheelhouse):
        _stage(wheelhouse, "tiktoken", py = f"cp{MAJOR}{MINOR + 1}")
        assert "tiktoken" not in ips._find_links_wheel_names()
        assert "tiktoken" in ips._windows_arm64_skip_packages()

    def test_a_matching_wheel_clears_the_skip(self, ips, wheelhouse):
        _stage(wheelhouse, "tiktoken")
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

    def test_one_hosted_blocker_is_not_enough(self, ips, wheelhouse):
        _stage(wheelhouse, "llvmlite")
        skipped = ips._windows_arm64_skip_packages()
        assert "librosa" in skipped, "librosa still needs numba"
        assert "openai-whisper" in skipped, "whisper still needs numba and tiktoken"


class TestInstallPs1Mirror:
    """install.ps1 builds the same availability set for its requirement overrides."""

    def test_wheel_names_are_filtered_by_interpreter_tag(self):
        block = _woa_wheel_names_block()
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


#: Tag triples for the wheels staged below: a pure-python wheel, and one built right here.
PURE = ("py3", "none", "any")
HERE = (TAG, TAG, _this_platform())


class TestBlockersDecideEvenWhenThePackageItselfIsHosted:
    """A package skipped for its DEPENDENCIES is not re-enabled by its own wheel."""

    @pytest.fixture
    def skips(self, ips, wheelhouse):
        """The skip list, given a wheelhouse holding exactly the wheels named."""

        def build(*specs):
            for name, version, py, abi, plat in specs:
                (wheelhouse / f"{name}-{version}-{py}-{abi}-{plat}.whl").write_bytes(b"")
            return ips._windows_arm64_skip_packages()

        return build

    TENSORBOARD = ("tensorboard", "1.0.0") + PURE
    LIBROSA = ("librosa", "1.0.0") + PURE

    @pytest.mark.parametrize(
        "specs, package, still_skipped, why",
        [
            ((TENSORBOARD,), "tensorboard", True, "its own wheel does not answer for grpcio"),
            (
                (TENSORBOARD, ("grpcio", "1.74.0") + HERE),
                "tensorboard",
                False,
                "grpcio at tensorboard's own floor of >=1.74.0 lifts it",
            ),
            (
                (TENSORBOARD, ("grpcio", "1.60.0") + HERE),
                "tensorboard",
                True,
                "a blocker below that floor used to lift the skip on the name alone, and the "
                "extras pass then failed on tensorboard's own metadata",
            ),
            (
                (LIBROSA, ("llvmlite", "1.0.0") + HERE),
                "librosa",
                True,
                "librosa needs numba as well as llvmlite",
            ),
            (
                (("tiktoken", "1.0.0") + HERE,),
                "tiktoken",
                False,
                "a package with no blockers still lifts on its own wheel",
            ),
        ],
        ids = [
            "tensorboard-alone",
            "tensorboard-and-grpcio",
            "grpcio-below-the-floor",
            "librosa-llvmlite-only",
            "tiktoken-no-blockers",
        ],
    )
    def test_the_blockers_decide(self, skips, specs, package, still_skipped, why):
        assert (package in skips(*specs)) is still_skipped, why


class TestFreeThreadedWheelsAreNotOfferedToTheRegularInterpreter:
    """cp313-cp313t is built for the free-threaded build; uv rejects it on cp313."""

    def test_python_side_rejects_a_free_threaded_abi(self, ips):
        free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        assert ips._wheel_matches_interpreter(_wheel("tiktoken", TAG, f"{TAG}t")) is free_threaded

    def test_install_ps1_checks_the_abi_not_just_the_python_tag(self):
        block = _woa_wheel_names_block()
        first = block[block.index("foreach ($pyTag in") :]
        first = first[: first.index("$compatible = $true; break") + 30]
        # $WoaWheelAbi, not $WoaWheelTag: a free-threaded venv installs cp313t but is tagged cp313.
        assert (
            "$abiTags -contains $WoaWheelAbi" in first
        ), "the exact-python-TAG branch must also require a usable ABI"
        assert (
            "$WoaWheelStable -and ($abiTags -contains 'abi3')" in first
        ), "and abi3 is not installable on a free-threaded build"


class TestAHostedOptionalIsActuallyInstalled:
    """Omitting the removal override only helps a package something still requires.

    install.ps1 reported "keeping X (the wheelhouse provides a win_arm64 wheel)" and then
    just declined to emit X's AMD64-only override line. hf_transfer and xformers are
    marker-excluded on win_arm64 by their released metadata and torchcodec's only line was
    filtered out here, so in all three cases hosting a wheel changed nothing.
    """

    #: The torch an xformers was built against, and the one now resident beside it.
    OTHER_TORCH = "2.9.0+cu128"
    THIS_TORCH = "2.15.0.dev20260101+cu134"

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

    def test_a_wheel_below_the_declared_floor_is_not_installed(self, ips, wheelhouse):
        """xformers>=0.0.22.post7 is what pyproject.toml asks for; 0.0.20 satisfies nobody."""
        _stage(wheelhouse, "xformers", "0.0.20")
        assert ips._wheelhouse_best_version("xformers", ">=0.0.22.post7") is None
        _stage(wheelhouse, "xformers", "0.0.31")
        ips._find_links_wheel_versions.cache_clear()
        assert ips._wheelhouse_best_version("xformers", ">=0.0.22.post7") == "0.0.31"

    def test_the_newest_clearing_wheel_wins(self, ips, wheelhouse):
        # Both clear the floor: 0.0.100 is the newer release and the SMALLER of the two as text.
        for version in ("0.0.23", "0.0.100"):
            _stage(wheelhouse, "xformers", version)
        assert (
            ips._wheelhouse_best_version("xformers", ">=0.0.22.post7") == "0.0.100"
        ), "sorted as text 0.0.23 would win"

    @pytest.mark.parametrize(
        "hosted, install_ok, built_for, evicted, why",
        [
            (True, True, OTHER_TORCH, True, "an xformers built for another torch is removed"),
            (True, False, OTHER_TORCH, True, "torch moved and the refresh failed: still evict"),
            (True, False, THIS_TORCH, False, "a failed refresh keeps a matching resident copy"),
            (True, True, THIS_TORCH, False, "a matching xformers is kept"),
            (True, True, None, False, "no recorded pair is not evidence of a wrong one"),
            # The check ran only after a hosted attempt, so a wheelhouse that stopped offering
            # xformers left the copy an earlier run installed losing its ops at import time.
            (False, True, OTHER_TORCH, True, "nothing hosted, and the resident copy is wrong"),
            (False, True, THIS_TORCH, False, "nothing hosted, and the resident copy matches"),
        ],
        ids = [
            "hosted-other-torch",
            "failed-refresh-other-torch",
            "failed-refresh-same-torch",
            "hosted-same-torch",
            "hosted-no-build-metadata",
            "unhosted-other-torch",
            "unhosted-same-torch",
        ],
    )
    def test_an_xformers_built_for_another_torch_is_removed(
        self, ips, monkeypatch, hosted, install_ok, built_for, evicted, why
    ):
        """Its extension links against one exact torch pair; beside any other the ops vanish
        behind a log line, which would otherwise be reported here as installed."""
        removed = []
        calls = self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"} if hosted else {},
            install_ok = install_ok,
            _resident_xformers_build_torch = lambda: built_for,
            _probe_installed_torch_version = lambda: self.THIS_TORCH,
            _uninstall_distribution = lambda name: removed.append(name) or True,
        )
        assert removed == (["xformers"] if evicted else []), why
        assert len(calls) == (1 if hosted else 0), "nothing hosted, nothing installed"

    def test_an_evicted_xformers_is_not_reported_installed(self, ips, monkeypatch):
        notes = []
        self._calls(
            ips,
            monkeypatch,
            {"xformers": "0.0.31"},
            _resident_xformers_build_torch = lambda: self.OTHER_TORCH,
            _probe_installed_torch_version = lambda: self.THIS_TORCH,
            _uninstall_distribution = lambda name: True,
            _note = lambda *a, **kw: notes.append(a[0]),
        )
        assert any("removed" in n for n in notes), notes
        assert not any("installed xformers" in n for n in notes), notes

    def test_the_step_runs_in_the_install(self, ips):
        """A helper nothing calls re-enables nothing."""
        assert "    _install_wheelhouse_optionals()" in STACK_SRC

    def test_a_hosted_torchcodec_keeps_its_requirement(self, ips):
        """The one line that asks for torchcodec was filtered out before the resolver."""
        guard = STACK_SRC.index("and PLATFORM_LACKS_TORCHCODEC_WHEEL")
        block = STACK_SRC[guard : STACK_SRC.index("_filter_requirements", guard)]
        assert 'not _wheelhouse_hosts("torchcodec")' in block

    def test_the_hosted_check_reads_the_resolvers_own_wheels(self, ips, wheelhouse):
        """And only wheels THIS interpreter could install: the staging copies cp311
        through cp314, and a wheel tagged for another MINOR is invisible to the resolver."""
        assert not ips._wheelhouse_hosts("torchcodec")
        _stage(wheelhouse, "torchcodec", py = f"cp{MAJOR}{MINOR + 1}")
        # The listing is memoized for the process, so each state needs its own read.
        ips._find_links_wheel_versions.cache_clear()
        assert not ips._wheelhouse_hosts("torchcodec"), "a foreign-tagged wheel is not hosted"
        _stage(wheelhouse, "torchcodec")
        ips._find_links_wheel_versions.cache_clear()
        assert ips._wheelhouse_hosts("torchcodec")


class TestAHostedWheelMustAlsoSatisfyThePin:
    """Name-only was not enough."""

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
        _stage(wheelhouse, "tiktoken", have)
        req = _req(wheelhouse.parent, f"{pin}\n")
        assert ("tiktoken" in ips._windows_arm64_skip_packages(req)) is still_skipped, why

    def test_any_hosted_version_that_satisfies_is_enough(self, ips, wheelhouse):
        for version in ("0.12.0", "0.13.0"):
            _stage(wheelhouse, "tiktoken", version)
        req = _req(wheelhouse.parent, "tiktoken==0.13.0\n")
        assert "tiktoken" not in ips._windows_arm64_skip_packages(req)

    @pytest.mark.parametrize(
        "grpcio, still_skipped",
        [("1.60.0", True), ("1.74.0", False)],
        ids = ["below-the-floor", "at-the-floor"],
    )
    def test_a_blocker_with_no_line_of_its_own_is_checked_against_its_floor(
        self, ips, wheelhouse, grpcio, still_skipped
    ):
        """grpcio arrives transitively, so extras.txt has no grpcio line to satisfy.

        That absence used to mean any hosted version counted. It does not: the floor comes
        from tensorboard's own metadata, which is what rejects a too-old blocker after the
        skip has been dropped.
        """
        _stage(wheelhouse, "grpcio", grpcio)
        req = _req(wheelhouse.parent, "tensorboard==2.21.0\n")
        assert (
            "tensorboard" in ips._windows_arm64_skip_packages(req)
        ) is still_skipped, f"grpcio {grpcio} against tensorboard's grpcio>=1.74.0"

    def test_a_blocker_with_no_floor_keeps_the_name_only_answer(self, ips, wheelhouse):
        """llvmlite has no entry: nothing states a floor for it, so a guess is not made."""
        for dist, version in (("llvmlite", "0.1.0"), ("numba", "0.62.0"), ("soxr", "1.0.0")):
            _stage(wheelhouse, dist, version)
        req = _req(wheelhouse.parent, "librosa==0.11.0\n")
        assert "librosa" not in ips._windows_arm64_skip_packages(req)

    def test_the_floors_name_the_pins_they_were_read_from(self, ips):
        """A bump to extras.txt has to be a prompt to re-read the metadata: the floors come
        from a version only that release states, so recorded provenance turns silent drift
        into a failure here.
        """
        for blocker, (specifier, package, version) in ips.WINDOWS_ARM64_BLOCKER_FLOORS.items():
            assert re.search(rf"(?m)^{re.escape(package)}=={re.escape(version)}\b", EXTRAS_SRC), (
                f"{blocker}'s floor {specifier} was read from {package}=={version}, which "
                f"extras.txt no longer pins -- re-read that release's metadata"
            )
            assert ips._canonical_dist_name(blocker) == blocker, "keys are canonical"

    def test_no_requirements_file_keeps_the_old_answer(self, ips, wheelhouse):
        _stage(wheelhouse, "tiktoken", "0.12.0")
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
        req = _req(tmp_path, self._rows())
        pins = ips._requirement_pins(req)
        assert pins["mecab"] == ["==0.996.13"], "the inactive row must not overwrite it"

    @pytest.mark.parametrize(
        "hosted, still_skipped, why",
        [
            ("0.996.5", True, "the hosted 0.996.5 satisfies only the row that does not apply here"),
            ("0.996.13", False, "the version the active row asks for still unskips"),
        ],
        ids = ["inactive-row", "active-row"],
    )
    def test_only_the_active_row_can_unskip(self, ips, wheelhouse, hosted, still_skipped, why):
        _stage(wheelhouse, "mecab", hosted)
        req = _req(wheelhouse.parent, self._rows())
        assert ("mecab" in ips._windows_arm64_skip_packages(req)) is still_skipped, why

    def test_markers_are_evaluated_with_packaging(self, ips):
        assert ips._marker_is_active("") is True
        assert ips._marker_is_active('sys_platform == "%s"' % sys.platform) is True
        assert ips._marker_is_active('sys_platform == "nonesuch"') is False
        assert ips._marker_is_active("this is not a marker") is None

    def test_without_packaging_every_clause_is_kept(self, ips, tmp_path, monkeypatch):
        """The fallback is no stricter than the name-only check that came before."""
        monkeypatch.setattr(ips, "_marker_is_active", lambda marker: None)
        req = _req(tmp_path, self._rows())
        assert ips._requirement_pins(req)["mecab"] == [
            "==0.996.13",
            "==0.996.5",
        ], "including the one whose marker would otherwise have excluded it"


class TestAPrereleaseWheelDoesNotSatisfyAFinalPin:
    """A wheelhouse holding 0.13.0rc1 must not unskip a package pinned to ==0.13.0: the
    numeric comparison reduced both to (0, 13, 0), so the skip was dropped and uv -- which
    applies PEP 440 properly -- fell to the ARM64 sdist the skip list exists to avoid."""

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
        """With packaging unavailable the numeric path runs, and answering "satisfied" for a
        version it cannot parse is the failure this fixes, so it answers False instead."""
        import importlib

        real = importlib.import_module

        def no_packaging(name, *args, **kwargs):
            if "packaging" in name:
                raise ImportError(name)
            return real(name, *args, **kwargs)

        monkeypatch.setattr(ips.importlib, "import_module", no_packaging)
        assert ips._version_satisfies("0.13.0rc1", "==0.13.0") is False
        assert ips._version_satisfies("0.13.0", "==0.13.0") is True

    def test_a_prerelease_wheel_leaves_the_package_skipped(self, ips, wheelhouse, monkeypatch):
        """End to end: the wheel is in the wheelhouse, and the skip survives anyway."""
        if "win_arm64" not in _this_platform():
            monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: "tiktoken" in name)
        _stage(wheelhouse, "tiktoken", "0.13.0rc1", plat = "win_arm64")
        req = _req(wheelhouse.parent, "tiktoken==0.13.0\n")
        assert "tiktoken" in ips._windows_arm64_skip_packages(req = req), (
            "an rc wheel satisfied an exact pin, so tiktoken was unskipped and the "
            "resolve fell to the sdist"
        )


class TestOnlyTheResolversOwnLocationsCount:
    """uv does not read PIP_FIND_LINKS, so a wheel hosted only there is not available, and
    counting it sent the resolve at the sdist the skip exists to avoid. pip cannot be the
    resolver here either: pip_install refuses the fallback under the win_arm64 overrides."""

    @pytest.mark.parametrize(
        "uv, pip, still_skipped, why",
        [
            (
                False,
                True,
                True,
                "the wheel is only where pip would look, and uv is what resolves here, so "
                "counting it sends the resolve to an sdist that cannot build",
            ),
            (True, False, False, "a uv location still unskips"),
        ],
        ids = ["pip-only", "uv"],
    )
    def test_only_a_uv_location_unskips(
        self, ips, tmp_path, monkeypatch, uv, pip, still_skipped, why
    ):
        _stage(tmp_path, "tiktoken", "0.13.0")
        for name, wanted in (("UV_FIND_LINKS", uv), ("PIP_FIND_LINKS", pip)):
            if wanted:
                monkeypatch.setenv(name, str(tmp_path))
            else:
                monkeypatch.delenv(name, raising = False)
        req = _req(tmp_path, "tiktoken==0.13.0\n")
        assert ("tiktoken" in ips._windows_arm64_skip_packages(req = req)) is still_skipped, why

    def test_install_ps1_sets_both_so_the_managed_wheelhouse_is_unaffected(self):
        """The narrowing must not cost the path it was written for."""
        assert "UV_FIND_LINKS" in INSTALL_SRC and "PIP_FIND_LINKS" in INSTALL_SRC
        assert '"UV_FIND_LINKS" = ","' in INSTALL_SRC, "and each keeps its own separator"


class TestAnExplicitPinIsNotOverriddenByThePreservationShortcut:
    """The ARM64 CUDA-preservation shortcut distrusts the INFERRED expectation, not a pin.

    A native cu134 venv has a family tag download.pytorch.org does not publish, so
    "repairing" it would resolve a cu130 with no wheel. But a user who names cu129 by URL or
    family has stated where they want to be, and exempting only a /cpu pin left them on the
    old build. setup.ps1 exempts every explicit pin; this is the same rule.
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


class TestThePublicIndexUnblocksWhatItAlreadyPublishes:
    """The skip list was decided from the local wheelhouse alone, so a native CPython 3.14
    ARM64 host -- where llvmlite and numba both publish a win_arm64 wheel, cp314 being the
    only tag either publishes one for -- had librosa dropped with its chain resolvable."""

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

    def test_the_tag_has_to_match_this_interpreter(self, ips, monkeypatch, native_cp314):
        assert ips._public_index_win_arm64_versions("numba") == {"0.67.0"}
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: False)
        assert ips._public_index_win_arm64_versions("numba") == set()

    def test_a_matching_interpreter_unblocks_librosa(
        self, ips, monkeypatch, wheelhouse, native_cp314, soxr_published
    ):
        assert "librosa" not in ips._windows_arm64_skip_packages()
        # And still dropped where the wheels are not published for this build.
        monkeypatch.setattr(ips, "_wheel_matches_interpreter", lambda name: False)
        ips._find_links_wheel_versions.cache_clear()
        assert "librosa" in ips._windows_arm64_skip_packages()

    def test_librosa_still_needs_soxr(self, ips, wheelhouse, native_cp314):
        """librosa 0.11.0 requires soxr>=0.3.2 and soxr has published no win_arm64 wheel in
        any release, so unblocking on llvmlite and numba alone put librosa back in the extras
        pass, where soxr is then built from the sdist the skip list exists to avoid.
        """
        assert "librosa" in ips._windows_arm64_skip_packages()

    def test_openai_whisper_still_needs_tiktoken(self, ips, wheelhouse, native_cp314):
        """Its third blocker publishes no win_arm64 wheel, so unblocking two is not enough."""
        assert "openai-whisper" in ips._windows_arm64_skip_packages()

    def test_an_empty_wheelhouse_no_longer_short_circuits(
        self, ips, monkeypatch, native_cp314, soxr_published
    ):
        """The early return read "nothing hosted" as "skip everything", which threw the
        public-index answer away before it was asked for."""
        monkeypatch.delenv("UV_FIND_LINKS", raising = False)
        assert "librosa" not in ips._windows_arm64_skip_packages()
        assert "mecab" in ips._windows_arm64_skip_packages(), "the rest still drop"


class TestThePublicIndexClaimNeedsTheIndex:
    """What PyPI publishes is only availability if the resolve will look at PyPI: offline, or
    pointed at an exclusive corporate index, unblocking librosa drops the skip and then fails
    the whole extras pass on a numba chain nothing serves."""

    @pytest.fixture(autouse = True)
    def _native(self, ips, monkeypatch, native_cp314):
        # uv runs the pass; the pip rows below switch it off explicitly.
        monkeypatch.setattr(ips, "USE_UV", True)
        for var in RESOLVER_INDEX_VARS:
            monkeypatch.delenv(var, raising = False)

    @pytest.mark.parametrize(
        "env, why",
        [
            ({}, "nothing configured is PyPI"),
            ({"UV_DEFAULT_INDEX": PYPI}, "a default index that is PyPI itself is still PyPI"),
            ({"UV_EXTRA_INDEX_URL": CORP}, "--extra-index-url ADDS to the default"),
        ],
        ids = ["unset", "pypi-as-default", "extra-index"],
    )
    def test_the_default_case_still_claims_them(self, ips, monkeypatch, env, why):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert ips._public_index_win_arm64_versions("numba") == {"0.67.0"}, why

    @pytest.mark.parametrize(
        "use_uv, var, value",
        [
            (True, "UV_OFFLINE", "1"),
            (True, "UV_NO_INDEX", "1"),
            (True, "UV_DEFAULT_INDEX", CORP),
            (True, "UV_INDEX_URL", CORP),
            (False, "PIP_INDEX_URL", CORP),
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
            (True, "PIP_INDEX_URL", CORP),
            (True, "PIP_NO_INDEX", "1"),
            (False, "UV_INDEX_URL", CORP),
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
        monkeypatch.setenv("UV_INDEX_URL", CORP)
        monkeypatch.setenv("PIP_EXTRA_INDEX_URL", PYPI)
        assert ips._public_index_win_arm64_versions("numba") == set()
        monkeypatch.setattr(ips, "USE_UV", False)
        assert ips._public_index_win_arm64_versions("numba") == {
            "0.67.0"
        }, "pip: its own default is PyPI"

    def test_librosa_goes_back_to_the_skip_list_offline(
        self, ips, monkeypatch, wheelhouse, soxr_published
    ):
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
        extra = ("UV_NO_CONFIG", "UV_CONFIG_FILE", "APPDATA", "PROGRAMDATA")
        for var in RESOLVER_INDEX_VARS + extra:
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

    #: The two files uv discovers in the project directory, and the fragments the rows
    #: below compose out of, so a case that is one statement is one line.
    UV = "proj/uv.toml"
    PJ = "proj/pyproject.toml"
    NO_IDX = "no-index = true\n"
    NO_IDX_OFF = "no-index = false\n"
    CORP_IDX = f'index-url = "{CORP}"\n'
    PYPI_IDX = f'index-url = "{PYPI}"\n'
    PIP_NO_IDX = "[pip]\nno-index = true\n"
    PIP_NO_IDX_OFF = "[pip]\nno-index = false\n"
    PIP_CORP = f'[pip]\nindex-url = "{CORP}"\n'
    PIP_PYPI = f'[pip]\nindex-url = "{PYPI}"\n'
    CORP_DEFAULT = f'[[index]]\nurl = "{CORP}"\ndefault = true\n'
    PYPI_DEFAULT = f'[[index]]\nurl = "{PYPI}"\ndefault = true\n'
    CORP_ADDED = f'[[index]]\nurl = "{CORP}"\n'
    PYPI_ADDED = f'[[index]]\nurl = "{PYPI}"\n'
    PYPI_EXTRA = f'extra-index-url = ["{PYPI}"]\n'
    MIRROR_EXTRA = 'extra-index-url = ["https://mirror.test/simple"]\n'
    CORP_EXPLICIT = f'[[index]]\nurl = "{CORP}"\nexplicit = true\n'
    PYPI_EXPLICIT = f'[[index]]\nurl = "{PYPI}"\nexplicit = true\n'
    CORP_EXPLICIT_DEFAULT = f'[[index]]\nurl = "{CORP}"\nexplicit = true\ndefault = true\n'

    @pytest.mark.parametrize(
        "rel, body, reachable, why",
        [
            # An exclusive source, however and wherever it is spelled.
            (UV, NO_IDX, False, "no-index"),
            (UV, f'default-index = "{CORP}"\n', False, "an exclusive default-index"),
            (UV, CORP_IDX, False, "the older spelling of the same key"),
            (UV, PIP_NO_IDX, False, "no-index under [pip]"),
            (UV, CORP_DEFAULT, False, "[[index]] default = true"),
            (UV, f'index = [{{ url = "{CORP}", default = true }}]\n', False, "an inline table"),
            (PJ, "[tool.uv]\n" + NO_IDX, False, "pyproject [tool.uv]"),
            (PJ, f'[tool.uv.pip]\nindex-url = "{CORP}"\n', False, "pyproject [tool.uv.pip]"),
            ("uv.toml", NO_IDX, False, "a parent directory"),
            ("user/uv/uv.toml", NO_IDX, False, "the user file"),
            (UV, "this is = not [ toml\n", False, "an unreadable file is not guessed at"),
            # Sources that leave PyPI in play, so the reader is not a blanket "not reachable".
            (UV, CORP_ADDED, True, "an [[index]] without default = true is additive"),
            (PJ, '[project]\nname = "x"\n', True, "a pyproject with no [tool.uv] at all"),
            # uv pip's own precedence, verified on uv 0.10.7 with a dry-run resolve.
            (UV, NO_IDX_OFF + PIP_NO_IDX, False, "[pip].no-index beats the top-level false"),
            (UV, NO_IDX + PIP_NO_IDX_OFF, True, "[pip].no-index = false beats the top-level true"),
            (UV, PYPI_IDX + PIP_CORP, False, "[pip].index-url beats the top-level index-url"),
            (UV, CORP_IDX + PIP_PYPI, True, "the other way round"),
            (UV, CORP_DEFAULT + PIP_PYPI, False, "[[index]] default beats [pip].index-url"),
            (UV, PIP_CORP + PYPI_DEFAULT, True, "and still does when it comes later in the file"),
            (UV, NO_IDX + PIP_PYPI, False, "no-index disables every registry"),
            (
                PJ,
                "[tool.uv]\n" + NO_IDX_OFF + "[tool.uv.pip]\n" + NO_IDX,
                False,
                "[tool.uv.pip] outranks [tool.uv], as [pip] does the top level",
            ),
            # An extra index that is PyPI keeps PyPI in play, wherever it is written.
            (UV, CORP_IDX + PYPI_EXTRA, True, "extra-index-url in the file"),
            (UV, CORP_IDX + "[pip]\n" + PYPI_EXTRA, True, "extra-index-url under [pip]"),
            (UV, CORP_DEFAULT + "\n" + PYPI_ADDED, True, "a second [[index]], not the default"),
            (UV, NO_IDX + PYPI_EXTRA, False, "no-index disables extras too"),
            (UV, CORP_IDX + MIRROR_EXTRA, False, "an extra that is not PyPI"),
            # uv: explicit = true serves only packages pinned via [tool.uv.sources].
            (UV, CORP_EXPLICIT, True, "an explicit corporate index leaves PyPI the default"),
            (UV, CORP_IDX + PYPI_EXPLICIT, False, "an explicit PyPI does not put PyPI back"),
            (UV, CORP_EXPLICIT_DEFAULT, False, "explicit and default: PyPI removed, not modelled"),
        ],
    )
    def test_what_the_uv_files_set(self, ips, rel, body, reachable, why):
        self._write(rel, body)
        assert ips._public_pypi_is_reachable() is reachable, why

    def test_an_explicit_index_is_not_an_extra(self, ips):
        self._write(self.UV, self.CORP_EXPLICIT + self.PYPI_ADDED)
        policy = ips._uv_config_index_policy()
        assert policy["extra_indexes"] == [PYPI], policy
        assert policy["unreadable"] is False

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
        "env, reachable, why",
        [
            (
                {"UV_INDEX_URL": CORP, "UV_EXTRA_INDEX_URL": PYPI},
                True,
                "an extra that is PyPI beside a corporate default",
            ),
            (
                {"UV_DEFAULT_INDEX": CORP, "UV_INDEX": f"https://mirror.test/simple {PYPI}"},
                True,
                "UV_INDEX, space-separated",
            ),
            (
                {"PIP_INDEX_URL": CORP, "PIP_EXTRA_INDEX_URL": PYPI},
                True,
                "pip's variables: uv never reads them, so its default PyPI stands",
            ),
            (
                {"UV_INDEX_URL": CORP, "PIP_EXTRA_INDEX_URL": PYPI},
                False,
                "a pip extra does not put PyPI back for uv",
            ),
            (
                {
                    "UV_INDEX_URL": CORP,
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

    #: Rows as `pip config list` prints them, composed so a case stays one line.
    G_CORP = f"global.index-url='{CORP}'\n"
    G_PYPI = f"global.index-url='{PYPI}'\n"
    G_NO_IDX = "global.no-index='true'\n"
    I_CORP = f"install.index-url='{CORP}'\n"
    I_PYPI = f"install.index-url='{PYPI}'\n"
    G_EXTRA_PYPI = f"global.extra-index-url='{PYPI}'\n"
    MIRROR = "https://mirror.test/simple"

    @pytest.mark.parametrize(
        "text, reachable, why",
        [
            (G_CORP, False, "an exclusive index-url"),
            (G_NO_IDX, False, "no-index"),
            ("global.no-index='false'\n", True, "no-index switched off"),
            (I_CORP, False, "under [install]"),
            (G_PYPI, True, "PyPI named explicitly"),
            (G_CORP + G_EXTRA_PYPI, True, "an extra that is PyPI beside a corporate default"),
            (
                G_CORP + f"global.extra-index-url='{MIRROR}\\n{PYPI}'\n",
                True,
                "one of several extras, newline-separated as pip prints them",
            ),
            (G_CORP + f"global.extra-index-url='{MIRROR}'\n", False, "an extra that is not PyPI"),
            (G_NO_IDX + G_EXTRA_PYPI, False, "no-index disables extras too"),
            (G_CORP + I_PYPI, True, "[install] outranks [global]"),
            (I_CORP + G_PYPI, False, "and still does when printed first"),
            (
                G_NO_IDX + "install.no-index='false'\n",
                True,
                "[install].no-index = false beats the global true",
            ),
            (
                f":env:.index-url='{CORP}'\n",
                True,
                ":env: rows mirror PIP_* the caller already read, and the variable is not set",
            ),
            (f'global.index-url="{CORP}"\n', False, "double quotes, should pip ever print them"),
        ],
    )
    def test_what_the_files_set(self, text, reachable, why):
        self._listing(text)
        assert self.ips._public_pypi_is_reachable() is reachable, why

    def test_the_environment_outranks_the_files(self):
        self._listing("global.no-index='true'\n")
        self.monkeypatch.setenv("PIP_INDEX_URL", PYPI)
        assert self.ips._public_pypi_is_reachable() is True

    def test_a_pip_extra_from_the_environment_counts_beside_a_file_index(self):
        self._listing(f"global.index-url='{CORP}'\n")
        self.monkeypatch.setenv("PIP_EXTRA_INDEX_URL", PYPI)
        assert self.ips._public_pypi_is_reachable() is True

    @pytest.mark.parametrize("cannot_run", [False, True], ids = ["exit-1", "no-pip"])
    def test_a_pip_config_that_fails_is_not_guessed_at(self, cannot_run):
        if cannot_run:

            def boom(*a, **kw):
                raise OSError("no pip")

            self.monkeypatch.setattr(self.ips.subprocess, "run", boom)
        else:
            self._listing("", returncode = 1)
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
            (f"[global]\nindex-url = {CORP}\n", False),
            ("[global]\nno-index = true\n", False),
            (f"[global]\nindex-url = {CORP}\n[install]\nindex-url = {PYPI}\n", True),
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
