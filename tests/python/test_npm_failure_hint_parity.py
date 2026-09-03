"""Parity tests for the npm-failure hint in setup.sh and setup.ps1 (issue #8725).

A Windows user installing Studio with Node 24 on PATH hit this while the OXC
validator runtime was being installed::

    npm error code EACCES
    npm error FetchError: request to https://registry.npmjs.org/oxlint/-/oxlint-1.65.0.tgz failed
    npm error The operation was rejected by your operating system.

That is a local failure -- a locked or unwritable npm cache -- but the installer
answered "registry.npmjs.org looks blocked (corporate firewall/proxy?)", so the
reporter went looking for a corporate proxy that did not exist. Re-running and
running as Administrator changed nothing; only downgrading Node fixed it.

Two separate defects produced that message:

1. ``Show-NpmRegistryHint`` in studio/setup.ps1 took no arguments and printed the
   registry hint on *every* non-zero npm exit -- it had no way to know why the
   install failed, because ``Invoke-SetupCommand`` captured npm's output into a
   local variable and dropped it.
2. ``_suggest_npm_registry`` in studio/setup.sh *was* log-aware, but its
   network regex includes the literal ``registry.npmjs.org``, and npm's
   FetchError line names that host even when the cause is a permission error --
   so the same log matched there too.

The behavioural coverage lives next to each implementation
(tests/sh/test_npm_failure_hint.sh, and the Pester suite for the PowerShell
side). These tests run everywhere, including the Linux backend CI job, and pin
the structure that keeps the two installers agreeing.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"

# Errno markers npm uses for failures the network cannot explain.
LOCAL_MARKERS = ["EACCES", "EPERM", "EBUSY", "ENOSPC"]


def _ps_function_body(text: str, name: str) -> str:
    """Return the body of a PowerShell function by brace counting.

    Naive on purpose: these two functions contain no unbalanced braces inside
    string literals. A regex with a fixed window silently truncates when the
    function grows, which is worse than being simple.
    """
    match = re.search(rf"(?im)^\s*function\s+{re.escape(name)}\b", text)
    assert match, f"{name} not found in the script under test"
    start = text.index("{", match.end())
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    raise AssertionError(f"unbalanced braces while reading {name}")


class TestPowerShellHintIsConditional:
    """setup.ps1 must decide from the failure text, not print the hint blindly."""

    def test_hint_accepts_the_failure_output(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        body = _ps_function_body(text, "Show-NpmRegistryHint")
        assert re.search(r"param\s*\(\s*\[string\]\s*\$FailureOutput", body), (
            "Show-NpmRegistryHint must take the captured npm output; without it "
            "every npm failure is reported as a blocked registry (issue #8725)."
        )

    def test_both_call_sites_pass_the_failure_output(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        invocations = [
            line.strip()
            for line in text.splitlines()
            if "Show-NpmRegistryHint" in line
            and not line.lstrip().startswith("#")
            and not re.match(r"\s*function\b", line)
        ]
        assert len(invocations) >= 2, f"expected the frontend and OXC call sites, got {invocations}"
        for call in invocations:
            assert "-FailureOutput" in call, f"call site passes no failure output: {call.strip()!r}"

    def test_setup_command_publishes_its_captured_output(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$script:LastSetupCommandOutput" in text, (
            "Invoke-SetupCommand must publish the captured output so the hint can "
            "classify the failure."
        )
        # The slot has to be cleared per call, or one command's failure text
        # would explain the next command's failure.
        body = _ps_function_body(text, "Invoke-SetupCommand")
        assert re.search(
            r'\$script:LastSetupCommandOutput\s*=\s*""', body
        ), "Invoke-SetupCommand must reset $script:LastSetupCommandOutput before running"

    def test_local_failures_are_classified_before_network_ones(self):
        """npm names registry.npmjs.org even in a permission error, so order matters."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        local_at = text.find("NpmLocalFailureRe")
        network_at = text.find("NpmNetworkFailureRe")
        assert local_at != -1, "no local-failure classifier in studio/setup.ps1"
        assert network_at != -1, "no network-failure classifier in studio/setup.ps1"

        body = _ps_function_body(text, "Show-NpmRegistryHint")
        local_use = body.find("NpmLocalFailureRe")
        network_use = body.find("NpmNetworkFailureRe")
        assert (
            local_use != -1 and network_use != -1
        ), "Show-NpmRegistryHint must consult both classifiers"
        assert local_use < network_use, (
            "the local-failure check must come first: npm's FetchError line names "
            "registry.npmjs.org even when the failure is a local permission error"
        )


class TestInstallersAgree:
    """Both installers must recognise the same local-failure markers."""

    def test_local_markers_present_in_both(self):
        sh = SETUP_SH.read_text(encoding = "utf-8")
        ps1 = SETUP_PS1.read_text(encoding = "utf-8")
        for marker in LOCAL_MARKERS:
            assert marker in sh, f"studio/setup.sh does not classify {marker}"
            assert marker in ps1, f"studio/setup.ps1 does not classify {marker}"

    def test_both_have_a_local_failure_hint(self):
        sh = SETUP_SH.read_text(encoding = "utf-8")
        ps1 = SETUP_PS1.read_text(encoding = "utf-8")
        assert "_suggest_npm_local_failure" in sh
        assert "Show-NpmLocalFailureHint" in ps1

    def test_local_hint_is_not_gated_on_the_registry_opt_out(self):
        """A mirror does not make a locked cache writable, so the hint must still print."""
        sh = SETUP_SH.read_text(encoding = "utf-8")
        match = re.search(r"_suggest_npm_registry\(\)\s*\{(.*?)\n\}", sh, re.DOTALL)
        assert match, "_suggest_npm_registry not found in studio/setup.sh"
        body = match.group(1)
        local_at = body.find("_suggest_npm_local_failure")
        optout_at = body.find('[ -n "${UNSLOTH_NPM_REGISTRY:-}" ]')
        assert local_at != -1, "the local-failure branch is missing"
        assert optout_at != -1, "the UNSLOTH_NPM_REGISTRY early-return is missing"
        assert local_at < optout_at, (
            "the local-failure branch must run before the UNSLOTH_NPM_REGISTRY "
            "early return, or users with a mirror configured never hear about it"
        )
