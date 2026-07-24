# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards that the installer test suites actually run on a PR.

Two ways coverage went missing without anyone noticing:

1. Backend CI ran a hardcoded list of tests/sh/*.sh files. New tests were added
   to the directory and never to the list, so by the time this was written the
   list was seven files behind -- including test_strixhalo_wsl_reroute.sh, the
   only shell coverage of the ROCm WSL reroute, which had never run on a PR.
   tests/run_all.sh, the local entrypoint, had drifted the other way.

2. Backend CI's path filter did not include install.sh / install.ps1, while a
   large share of the suites it runs (tests/sh/*, tests/studio/install/*) assert
   against exactly those two files. An install-only change -- the shape most
   AMD/ROCm routing fixes take, e.g. #7277 / #7293 / #7300 -- skipped the
   workflow that tests it.

Both are now discovery-based. These tests fail if either reverts to a list, if a
shell test lands somewhere the discovery cannot see it, or if a skip is added
without a reason next to it.
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

_WORKFLOWS = REPO_ROOT / ".github" / "workflows"
_BACKEND_CI = _WORKFLOWS / "studio-backend-ci.yml"
_PARITY_CI = _WORKFLOWS / "cross-platform-parity-ci.yml"
_RUN_ALL = REPO_ROOT / "tests" / "run_all.sh"
_SH_DIR = REPO_ROOT / "tests" / "sh"

# Files deliberately not run by the auto-discovered Backend CI step. Each needs
# a reason here AND in the workflow; anything else in tests/sh must run.
_EXPECTED_CI_SKIPS = {
    "test_install_host_defaults.sh": "asserts an install.ps1 layout that has drifted",
    "test_install_rollback_lifecycle.sh": "runs on both platforms in cross-platform-parity-ci.yml",
}


def _shell_test_files():
    files = sorted(p.name for p in _SH_DIR.glob("test_*.sh"))
    assert files, "tests/sh has no test_*.sh files -- did the directory move?"
    return files


def _skip_list(source: str) -> set[str]:
    """The skip= / SH_SKIP= line from a discovery loop."""
    m = re.search(r"^\s*(?:skip|SH_SKIP)=\"([^\"]*)\"", source, re.MULTILINE)
    assert m, "no skip list found; the discovery loop must declare one (even if empty)"
    return {name for name in m.group(1).split() if name}


class TestBackendCiRunsEveryShellTest:
    def test_step_discovers_the_directory_instead_of_listing_files(self):
        source = _BACKEND_CI.read_text(encoding = "utf-8")
        assert "for s in tests/sh/test_*.sh; do" in source, (
            "Backend CI must glob tests/sh; a hardcoded list is how the ROCm WSL "
            "suite went unrun for months"
        )

    def test_step_fails_loudly_if_discovery_finds_nothing(self):
        """A moved directory must break the build, not pass vacuously."""
        source = _BACKEND_CI.read_text(encoding = "utf-8")
        assert 'no shell tests discovered under tests/sh' in source

    def test_every_shell_test_runs_or_is_a_known_skip(self):
        skips = _skip_list(_BACKEND_CI.read_text(encoding = "utf-8"))
        unexpected = skips - set(_EXPECTED_CI_SKIPS)
        assert not unexpected, (
            f"Backend CI skips {sorted(unexpected)} without a reason recorded in "
            "_EXPECTED_CI_SKIPS; add one or stop skipping it"
        )
        # Everything else in the directory is covered by the glob.
        for name in _shell_test_files():
            assert name not in skips or name in _EXPECTED_CI_SKIPS, name

    def test_skip_entries_are_not_stale(self):
        """A skip for a deleted file quietly widens next time a name is reused."""
        existing = set(_shell_test_files())
        for name in _skip_list(_BACKEND_CI.read_text(encoding = "utf-8")):
            assert name in existing, f"{name} is skipped but no longer exists in tests/sh"

    def test_each_skip_is_documented_in_the_workflow(self):
        source = _BACKEND_CI.read_text(encoding = "utf-8")
        for name in _EXPECTED_CI_SKIPS:
            assert source.count(name) >= 2, (
                f"{name} is skipped in Backend CI without a comment explaining why"
            )

    def test_rollback_lifecycle_really_does_run_elsewhere(self):
        """The one skip justified by 'another workflow covers it' must be true."""
        assert "tests/sh/test_install_rollback_lifecycle.sh" in _PARITY_CI.read_text(encoding = "utf-8")

    def test_rocm_shell_suite_is_in_scope(self):
        """The suite whose absence prompted this file: it must exist and be
        picked up (i.e. not skipped)."""
        assert "test_strixhalo_wsl_reroute.sh" in _shell_test_files()
        assert "test_strixhalo_wsl_reroute.sh" not in _skip_list(_BACKEND_CI.read_text(encoding = "utf-8"))


class TestRunAllMatchesCi:
    """tests/run_all.sh is what a contributor runs before pushing. If it and CI
    disagree, one of them is lying about the state of the tree."""

    def test_run_all_discovers_the_directory(self):
        source = _RUN_ALL.read_text(encoding = "utf-8")
        assert 'for _t in "$TESTS_DIR"/sh/test_*.sh; do' in source

    def test_run_all_skips_are_a_subset_of_ci_skips(self):
        local = _skip_list(_RUN_ALL.read_text(encoding = "utf-8"))
        unexpected = local - set(_EXPECTED_CI_SKIPS)
        assert not unexpected, (
            f"tests/run_all.sh skips {sorted(unexpected)} that CI still runs: a "
            "contributor would see green locally and red on the PR"
        )


class TestBackendCiPathFilters:
    """The workflow has to fire on the files its tests assert against."""

    def _paths(self) -> set[str]:
        source = _BACKEND_CI.read_text(encoding = "utf-8")
        start = source.find("    paths:")
        assert start != -1
        end = source.find("  push:", start)
        assert end != -1
        return set(re.findall(r"^\s+- '([^']+)'", source[start:end], re.MULTILINE))

    @pytest.mark.parametrize(
        "path,why",
        [
            ("install.sh", "tests/sh/* and tests/studio/install/* assert against it"),
            ("install.ps1", "the Windows/ROCm arch tables and pin allowlist live here"),
            ("studio/**", "covers studio/setup.sh, studio/setup.ps1, install_python_stack.py"),
            ("tests/**", "test-only changes must run the tests they touch"),
        ],
    )
    def test_trigger_covers(self, path, why):
        assert path in self._paths(), f"Backend CI does not run when {path} changes ({why})"

    def test_installer_change_would_trigger_the_workflow(self):
        """End to end: the exact filenames the ROCm fixes edit."""
        paths = self._paths()
        for changed in ("install.sh", "install.ps1"):
            assert changed in paths
        for changed in ("studio/setup.ps1", "studio/setup.sh", "studio/install_python_stack.py"):
            assert any(
                changed.startswith(pattern.rstrip("*").rstrip("/")) for pattern in paths if pattern.endswith("/**")
            ), f"nothing in the path filter matches {changed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
