# Copyright 2025-present the Unsloth AI Inc. team. All rights reserved.

"""Truth-table tests for `resolve_tool_policy`: tools default on for every bind
(loopback, --secure tunnel, raw network), explicit on/off wins, and the resolver
never prompts (yes/silent/prompt kept for compatibility)."""

import pytest

from unsloth_cli._tool_policy import is_external_host, resolve_tool_policy


def _never_prompt(_msg: str) -> bool:
    raise AssertionError("resolve_tool_policy must not prompt")


class TestLocalhostHost:
    @pytest.mark.parametrize("flag", [None, True, False])
    def test_no_prompt(self, flag):
        # localhost never prompts regardless of flag
        result = resolve_tool_policy(
            host = "127.0.0.1",
            flag = flag,
            yes = False,
            silent = False,
            prompt = _never_prompt,
        )
        assert result is (True if flag in (None, True) else False)

    def test_default_is_on(self):
        assert (
            resolve_tool_policy(
                host = "127.0.0.1",
                flag = None,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is True
        )

    def test_explicit_off(self):
        assert (
            resolve_tool_policy(
                host = "127.0.0.1",
                flag = False,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is False
        )


class TestZeroHost:
    def test_default_is_on(self):
        # Network bind defaults ON now (operator owns network security).
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = None,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is True
        )

    def test_explicit_off_no_prompt(self):
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = False,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is False
        )

    def test_explicit_on_no_prompt(self):
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = True,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is True
        )

    def test_yes_and_silent_accepted_but_do_not_change_result(self):
        # Retained for backward compatibility; they no longer gate the result.
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = None,
                yes = True,
                silent = True,
                prompt = _never_prompt,
            )
            is True
        )


class TestIsExternalHost:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "localhost", "::1", "LOCALHOST", "Localhost"]
    )
    def test_loopback_aliases_are_local(self, host):
        assert is_external_host(host) is False

    @pytest.mark.parametrize(
        "host", ["0.0.0.0", "::", "127.0.0.2", "192.168.1.5", "10.0.0.1", "example.com"]
    )
    def test_non_loopback_is_external(self, host):
        assert is_external_host(host) is True


class TestSpecificNetworkIP:
    """Binding to a specific LAN IP follows the same default-on rules as 0.0.0.0."""

    def test_default_is_on(self):
        assert (
            resolve_tool_policy(
                host = "192.168.1.5",
                flag = None,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is True
        )

    def test_explicit_on_no_prompt(self):
        assert (
            resolve_tool_policy(
                host = "192.168.1.5",
                flag = True,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is True
        )

    def test_explicit_off(self):
        assert (
            resolve_tool_policy(
                host = "192.168.1.5",
                flag = False,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is False
        )

    def test_localhost_alias_does_not_prompt(self):
        assert (
            resolve_tool_policy(
                host = "localhost",
                flag = True,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is True
        )
