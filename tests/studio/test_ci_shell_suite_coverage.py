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
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

_WORKFLOWS = REPO_ROOT / ".github" / "workflows"
_BACKEND_CI = _WORKFLOWS / "studio-backend-ci.yml"
_PARITY_CI = _WORKFLOWS / "cross-platform-parity-ci.yml"
_RUN_ALL = REPO_ROOT / "tests" / "run_all.sh"
_SH_DIR = REPO_ROOT / "tests" / "sh"

# Files deliberately not run by the auto-discovered Backend CI step.
_EXPECTED_CI_SKIPS = {
    "test_install_rollback_lifecycle.sh": "runs on both platforms in cross-platform-parity-ci.yml",
}


def _backend_ci() -> dict:
    return yaml.safe_load(_BACKEND_CI.read_text(encoding = "utf-8"))


def _shell_step_script() -> str:
    """The `run:` body of the shell-installer step, located by name through the
    parsed YAML rather than by slicing the raw file."""
    for job in _backend_ci()["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == "Shell installer tests":
                return step["run"]
    raise AssertionError("Backend CI has no 'Shell installer tests' step")


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
        """Matched against the parsed step script, and on the glob rather than a
        verbatim line, so reformatting the loop does not turn CI red -- only
        going back to a hardcoded list does."""
        script = _shell_step_script()
        assert re.search(r"for\s+\w+\s+in\s+tests/sh/test_\*\.sh", script), (
            "Backend CI must glob tests/sh; a hardcoded list is how the ROCm WSL "
            f"suite went unrun for months. Step script was:\n{script}"
        )
        listed = re.findall(r"tests/sh/test_[a-z0-9_]+\.sh", script)
        assert not listed, f"Backend CI still names individual shell tests: {sorted(set(listed))}"

    def test_step_fails_loudly_if_discovery_finds_nothing(self):
        """A moved directory must break the build, not pass vacuously."""
        assert "no shell tests discovered under tests/sh" in _shell_step_script()

    def test_every_shell_test_runs_or_is_a_known_skip(self):
        skips = _skip_list(_shell_step_script())
        unexpected = skips - set(_EXPECTED_CI_SKIPS)
        assert not unexpected, (
            f"Backend CI skips {sorted(unexpected)} without a reason recorded in "
            "_EXPECTED_CI_SKIPS; add one or stop skipping it"
        )
        for name in _shell_test_files():
            assert name not in skips or name in _EXPECTED_CI_SKIPS, name

    def test_skip_entries_are_not_stale(self):
        """A skip for a deleted file quietly widens next time a name is reused."""
        existing = set(_shell_test_files())
        for name in _skip_list(_shell_step_script()):
            assert name in existing, f"{name} is skipped but no longer exists in tests/sh"

    def test_each_skip_is_documented_in_the_workflow(self):
        source = _BACKEND_CI.read_text(encoding = "utf-8")
        for name in _EXPECTED_CI_SKIPS:
            assert (
                source.count(name) >= 2
            ), f"{name} is skipped in Backend CI without a comment explaining why"

    def test_rollback_lifecycle_really_does_run_elsewhere(self):
        """The one skip justified by 'another workflow covers it' must be true."""
        assert "tests/sh/test_install_rollback_lifecycle.sh" in _PARITY_CI.read_text(
            encoding = "utf-8"
        )

    def test_rocm_shell_suite_is_in_scope(self):
        """The suite whose absence prompted this file: it must exist and be
        picked up (i.e. not skipped)."""
        assert "test_strixhalo_wsl_reroute.sh" in _shell_test_files()
        assert "test_strixhalo_wsl_reroute.sh" not in _skip_list(_shell_step_script())


class TestRunAllMatchesCi:
    """tests/run_all.sh is what a contributor runs before pushing. If it and CI
    disagree, one of them is lying about the state of the tree."""

    def test_run_all_discovers_the_directory(self):
        source = _RUN_ALL.read_text(encoding = "utf-8")
        assert 'for _t in "$TESTS_DIR"/sh/test_*.sh; do' in source

    def test_run_all_invokes_the_tests_with_bash(self):
        """Both runners must use the interpreter the tests declare. Every file
        under tests/sh/ has a bash shebang, and on Debian/Ubuntu /bin/sh is
        dash, under which three of them fail on bashisms. Running them with sh
        would fail the suite locally for reasons CI never reproduces."""
        source = _RUN_ALL.read_text(encoding = "utf-8")
        assert 'bash "$_t"' in source, "tests/run_all.sh must run tests/sh/ with bash"
        assert 'sh "$_t"' not in source.replace(
            'bash "$_t"', ""
        ), "tests/run_all.sh still invokes a discovered test with sh"
        assert 'bash "$s"' in _shell_step_script(), "Backend CI must run tests/sh/ with bash"

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
        """Read the real trigger through the YAML parser. `on:` is a YAML 1.1
        boolean, so pyyaml keys it as True."""
        wf = _backend_ci()
        triggers = wf.get("on", wf.get(True))
        assert triggers, "Backend CI has no trigger block"
        paths = triggers["pull_request"]["paths"]
        assert paths, "Backend CI pull_request trigger has no paths filter"
        return set(paths)

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
                changed.startswith(pattern.rstrip("*").rstrip("/"))
                for pattern in paths
                if pattern.endswith("/**")
            ), f"nothing in the path filter matches {changed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _github_path_matcher(pattern: str) -> re.Pattern:
    """GitHub path filters: ** crosses directories, * and ? do not."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _workflows_running_powershell_tests():
    """Every workflow that invokes a tests/**.ps1 file, with its PR path filter."""
    found = {}
    for workflow in sorted(_WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding = "utf-8")
        invoked = sorted(set(re.findall(r"pwsh -NoProfile -File (tests/[^\s`\"']+\.ps1)", text)))
        if not invoked:
            continue
        parsed = yaml.safe_load(text)
        triggers = parsed.get(True, parsed.get("on", {})) or {}
        paths = (triggers.get("pull_request") or {}).get("paths")
        found[workflow.name] = (invoked, paths)
    return found


class TestGithubPathMatcher:
    """The guard below is only as good as this matcher; a wrong one would pass
    everything silently."""

    @pytest.mark.parametrize(
        "pattern,path,expected",
        [
            ("tests/studio/*.ps1", "tests/studio/test_x.ps1", True),
            ("tests/studio/*.ps1", "tests/studio/nested/test_x.ps1", False),
            ("tests/studio/*.ps1", "tests/studio/test_x.py", False),
            ("tests/studio/**", "tests/studio/nested/test_x.ps1", True),
            ("studio/**", "studio/setup.ps1", True),
            ("studio/**", "tests/studio/setup.ps1", False),
            (
                "tests/studio/test_uninstall_*.ps1",
                "tests/studio/test_uninstall_arg_guard.ps1",
                True,
            ),
            ("tests/studio/test_uninstall_*.ps1", "tests/studio/test_node_decision.ps1", False),
            ("install.ps1", "install.ps1", True),
            ("install.ps1", "studio/install.ps1", False),
        ],
    )
    def test_matcher_semantics(self, pattern, path, expected):
        assert bool(_github_path_matcher(pattern).match(path)) is expected


class TestPowerShellTestsRunOnAPr:
    """tests/sh had this exact hole (see the module docstring) and so did the
    Windows side: studio-windows-inference-smoke.yml ran six PowerShell tests
    while its path filter matched none of them, so a PR fixing one of those
    tests never ran it."""

    def test_some_workflow_runs_powershell_tests(self):
        assert (
            _workflows_running_powershell_tests()
        ), "no workflow invokes a tests/*.ps1 file; did the invocation form change?"

    def test_every_invoked_powershell_test_triggers_its_workflow(self):
        unguarded = []
        for name, (invoked, paths) in _workflows_running_powershell_tests().items():
            if paths is None:
                continue  # no filter at all means it always runs
            matchers = [_github_path_matcher(p) for p in paths]
            for test in invoked:
                if not any(m.match(test) for m in matchers):
                    unguarded.append(f"{name} runs {test} but its paths filter never matches it")
        assert not unguarded, (
            "these PowerShell tests can break without any PR running them; add the "
            f"path (or a scoped glob) to the workflow's paths filter: {unguarded}"
        )

    def test_multi_test_steps_propagate_each_exit_code(self):
        """A `shell: pwsh` step inherits only the LAST command's exit code, so a
        step running several tests must check $LASTEXITCODE after each one.
        Without it, test_resolve_cuda_toolkit.ps1 failed two checks on every
        Windows run for as long as anyone can tell, and CI stayed green."""
        offenders = []
        for workflow in sorted(_WORKFLOWS.glob("*.yml")):
            for block in re.findall(
                r"run: \|\n(.*?)(?=\n      [-a-zA-Z]|\Z)",
                workflow.read_text(encoding = "utf-8"),
                re.S,
            ):
                invocations = re.findall(
                    r"pwsh -NoProfile -File (tests/[^\s`\"']+\.ps1)[^\n]*\n(.*?)(?=pwsh -NoProfile -File|\Z)",
                    block,
                    re.S,
                )
                if len(invocations) < 2:
                    continue  # a single invocation's exit code is the step's
                for test, following in invocations:
                    if "$LASTEXITCODE" not in following:
                        offenders.append(f"{workflow.name}: {test} runs without an exit-code check")
        assert not offenders, (
            "these tests can fail without failing their step; add "
            f"`if ($LASTEXITCODE) {{ exit $LASTEXITCODE }}` after each: {offenders}"
        )

    def test_every_invoked_powershell_test_exists(self):
        missing = [
            f"{name} -> {test}"
            for name, (invoked, _) in _workflows_running_powershell_tests().items()
            for test in invoked
            if not (REPO_ROOT / test).is_file()
        ]
        assert not missing, f"workflows invoke PowerShell tests that do not exist: {missing}"
