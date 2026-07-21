# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for the exposed-first-run auto-shutdown deadline.

Tests the env parsing, the pure arm/no-arm decision matrix, and the deadline
handler (shut down iff the seeded admin password is still unchanged). The
threading.Timer itself is not exercised; the handler is invoked directly.
"""

from types import SimpleNamespace

from auth.bootstrap_timeout import (
    DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS,
    _format_duration,
    bootstrap_timeout_seconds,
    enforce_bootstrap_password_deadline,
    should_arm_bootstrap_timeout,
)


# ── bootstrap_timeout_seconds ───────────────────────────────────────


def test_default_when_unset():
    assert bootstrap_timeout_seconds(env = {}) == DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS


def test_default_when_empty():
    assert bootstrap_timeout_seconds(
        env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": "  "}
    ) == (DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS)


def test_explicit_value_parsed():
    assert (
        bootstrap_timeout_seconds(env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": "1800"})
        == 1800
    )


def test_zero_disables():
    assert bootstrap_timeout_seconds(env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": "0"}) == 0


def test_negative_disables():
    assert (
        bootstrap_timeout_seconds(env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": "-5"}) == 0
    )


def test_invalid_falls_back_to_default():
    # A typo must keep the protection, not silently disable it.
    assert bootstrap_timeout_seconds(
        env = {"UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT": "abc"}
    ) == (DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS)


# ── should_arm_bootstrap_timeout matrix ─────────────────────────────


def _arm_kwargs(**overrides):
    kwargs = dict(
        host = "0.0.0.0",
        secure = False,
        api_only = False,
        frontend_served = True,
        is_colab = False,
        requires_change = True,
        timeout_seconds = 3600,
    )
    kwargs.update(overrides)
    return kwargs


def test_arm_exposed_wildcard_web_ui():
    assert should_arm_bootstrap_timeout(**_arm_kwargs()) is True


def test_arm_secure_loopback_bind():
    # --secure forces a loopback bind but exposes a public tunnel.
    assert (
        should_arm_bootstrap_timeout(**_arm_kwargs(host = "127.0.0.1", secure = True))
        is True
    )


def test_no_arm_loopback_bind():
    assert (
        should_arm_bootstrap_timeout(**_arm_kwargs(host = "127.0.0.1", secure = False))
        is False
    )


def test_no_arm_api_only():
    assert should_arm_bootstrap_timeout(**_arm_kwargs(api_only = True)) is False


def test_no_arm_no_frontend():
    assert should_arm_bootstrap_timeout(**_arm_kwargs(frontend_served = False)) is False


def test_no_arm_colab():
    assert should_arm_bootstrap_timeout(**_arm_kwargs(is_colab = True)) is False


def test_no_arm_password_already_changed():
    assert should_arm_bootstrap_timeout(**_arm_kwargs(requires_change = False)) is False


def test_no_arm_timeout_disabled():
    assert should_arm_bootstrap_timeout(**_arm_kwargs(timeout_seconds = 0)) is False


# ── enforce_bootstrap_password_deadline ─────────────────────────────


def _fake_storage(requires_change: bool):
    return SimpleNamespace(
        DEFAULT_ADMIN_USERNAME = "unsloth",
        requires_password_change = lambda _username: requires_change,
    )


def test_deadline_shuts_down_when_password_unchanged():
    calls = []
    result = enforce_bootstrap_password_deadline(
        _fake_storage(requires_change = True),
        lambda: calls.append("shutdown"),
        timeout_seconds = 3600,
    )
    assert result is True
    assert calls == ["shutdown"]


def test_deadline_keeps_running_when_password_changed():
    calls = []
    result = enforce_bootstrap_password_deadline(
        _fake_storage(requires_change = False),
        lambda: calls.append("shutdown"),
        timeout_seconds = 3600,
    )
    assert result is False
    assert calls == []


def test_deadline_swallows_shutdown_errors():
    def _boom():
        raise RuntimeError("shutdown failed")

    # A failing shutdown must not propagate out of the timer thread.
    result = enforce_bootstrap_password_deadline(
        _fake_storage(requires_change = True),
        _boom,
        timeout_seconds = 3600,
    )
    assert result is True


# ── _format_duration ────────────────────────────────────────────────


def test_format_duration_sub_minute_uses_seconds():
    assert _format_duration(30) == "30 seconds"


def test_format_duration_singular_second():
    assert _format_duration(1) == "1 second"


def test_format_duration_exact_minutes():
    assert _format_duration(60) == "1 minute"
    assert _format_duration(3600) == "60 minutes"


def test_format_duration_minutes_and_seconds():
    assert _format_duration(90) == "1 minute 30 seconds"


def test_shutdown_message_uses_formatted_duration():
    # The deadline message must reflect the real timeout, not a rounded
    # "minute(s)" placeholder. Capture the warning via a fake logger.
    logged = []

    class _Logger:
        def warning(self, msg, *args):
            logged.append(msg)

    enforce_bootstrap_password_deadline(
        _fake_storage(requires_change = True),
        lambda: None,
        timeout_seconds = 3600,
        logger = _Logger(),
    )
    assert any("60 minutes" in m for m in logged)
    assert not any("minute(s)" in m for m in logged)
