# Copyright 2025-present the Unsloth AI Inc. team. All rights reserved.

"""
Truth-table tests for `resolve_tool_policy` -- the pure resolver behind
`unsloth run --enable-tools/--disable-tools`.

Covers:
  - 127.0.0.1 default-on, explicit on, explicit off
  - 0.0.0.0 default-off, explicit off
  - 0.0.0.0 + explicit on: confirm prompt unless --silent or --yes,
    abort on negative answer.
"""

import pytest
import typer

from unsloth_cli._tool_policy import is_external_host, resolve_tool_policy


def _never_prompt(_msg: str) -> bool:
    raise AssertionError("prompt should not have been called")


def _prompt_yes(_msg: str) -> bool:
    return True


def _prompt_no(_msg: str) -> bool:
    return False


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
    def test_default_is_off(self):
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = None,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is False
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

    def test_explicit_on_silent_skips_prompt(self):
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = True,
                yes = False,
                silent = True,
                prompt = _never_prompt,
            )
            is True
        )

    def test_explicit_on_yes_skips_prompt(self):
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = True,
                yes = True,
                silent = False,
                prompt = _never_prompt,
            )
            is True
        )

    def test_explicit_on_prompt_yes(self):
        assert (
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = True,
                yes = False,
                silent = False,
                prompt = _prompt_yes,
            )
            is True
        )

    def test_explicit_on_prompt_no_aborts(self):
        with pytest.raises(typer.Exit) as exc_info:
            resolve_tool_policy(
                host = "0.0.0.0",
                flag = True,
                yes = False,
                silent = False,
                prompt = _prompt_no,
            )
        assert exc_info.value.exit_code == 1


class TestIsExternalHost:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "localhost", "::1", "LOCALHOST", "Localhost"]
    )
    def test_loopback_aliases_are_local(self, host):
        assert is_external_host(host) is False

    @pytest.mark.parametrize(
        "host", ["0.0.0.0", "::", "192.168.1.5", "10.0.0.1", "example.com"]
    )
    def test_non_loopback_is_external(self, host):
        assert is_external_host(host) is True


class TestSpecificNetworkIP:
    """Binding to a specific LAN IP must follow the same rules as 0.0.0.0."""

    def test_default_is_off(self):
        assert (
            resolve_tool_policy(
                host = "192.168.1.5",
                flag = None,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is False
        )

    def test_explicit_on_prompts(self):
        seen = []

        def _prompt(msg: str) -> bool:
            seen.append(msg)
            return True

        assert (
            resolve_tool_policy(
                host = "192.168.1.5",
                flag = True,
                yes = False,
                silent = False,
                prompt = _prompt,
            )
            is True
        )
        assert any("192.168.1.5" in m for m in seen)

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
