# Copyright 2025-present the Unsloth AI Inc. team. All rights reserved.

"""Truth-table tests for `resolve_tool_policy`: no flag installs no process-wide
OVERRIDE on any bind (loopback, --secure tunnel, raw network), so a request's own
`enable_tools: false` is honored -- tools still default on for a request that
omits the field, via the backend's separate tool-policy default. Explicit on/off
wins, and the resolver never prompts (yes/silent/prompt kept for compatibility)."""

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
        assert result is flag

    def test_default_is_unset(self):
        assert (
            resolve_tool_policy(
                host = "127.0.0.1",
                flag = None,
                yes = False,
                silent = False,
                prompt = _never_prompt,
            )
            is None
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
    @pytest.mark.parametrize(
        "flag, yes, silent, expected",
        [
            # A network bind installs no override, so the UI's tool pills (which send enable_tools: false when all
            # off) are honored rather than overridden.
            pytest.param(None, False, False, None, id = "default_is_unset"),
            pytest.param(False, False, False, False, id = "explicit_off_no_prompt"),
            pytest.param(True, False, False, True, id = "explicit_on_no_prompt"),
            # Retained for backward compatibility; they no longer gate the result.
            pytest.param(
                None, True, True, None, id = "yes_and_silent_accepted_but_do_not_change_result"
            ),
        ],
    )
    def test_zero_host_cases(self, flag, yes, silent, expected):
        assert (
            resolve_tool_policy(
                host = "0.0.0.0", flag = flag, yes = yes, silent = silent, prompt = _never_prompt
            )
            is expected
        )


class TestIsExternalHost:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "LOCALHOST", "Localhost"])
    def test_loopback_aliases_are_local(self, host):
        assert is_external_host(host) is False

    @pytest.mark.parametrize(
        "host", ["0.0.0.0", "::", "127.0.0.2", "192.168.1.5", "10.0.0.1", "example.com"]
    )
    def test_non_loopback_is_external(self, host):
        assert is_external_host(host) is True


class TestSpecificNetworkIP:
    """Binding to a specific LAN IP follows the same rules as 0.0.0.0."""

    @pytest.mark.parametrize(
        "host, flag, expected",
        [
            pytest.param("192.168.1.5", None, None, id = "default_is_unset"),
            pytest.param("192.168.1.5", True, True, id = "explicit_on_no_prompt"),
            pytest.param("192.168.1.5", False, False, id = "explicit_off"),
            pytest.param("localhost", True, True, id = "localhost_alias_does_not_prompt"),
        ],
    )
    def test_specific_network_i_p_cases(self, host, flag, expected):
        assert (
            resolve_tool_policy(host = host, flag = flag, yes = False, silent = False, prompt = _never_prompt)
            is expected
        )
