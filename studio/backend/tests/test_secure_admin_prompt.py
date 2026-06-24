# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit coverage for the pre-exposure admin password prompt.

Exercises the pure decision logic (``resolve_admin_password_source``), the
prompt/confirm/retry loop with an injected reader, and the exposed-bind check.
No real TTY or terminal I/O is touched here.
"""

import pytest

from auth.secure_admin_prompt import (
    MIN_PASSWORD_LENGTH,
    _is_exposed_bind,
    prompt_new_admin_password,
    resolve_admin_password_source,
)


# ── resolve_admin_password_source matrix ────────────────────────────


def _base_kwargs(**overrides):
    kwargs = dict(
        frontend_served = True,
        exposed = True,
        is_colab = False,
        requires_change = True,
        has_tty = True,
        env_password = None,
    )
    kwargs.update(overrides)
    return kwargs


def test_exposed_tty_prompts():
    assert resolve_admin_password_source(**_base_kwargs()) == "prompt"


def test_exposed_env_takes_env():
    assert resolve_admin_password_source(**_base_kwargs(env_password = "x" * 12)) == "env"


def test_env_wins_over_tty():
    # An explicit env var is honored even on an interactive terminal.
    assert (
        resolve_admin_password_source(**_base_kwargs(env_password = "x" * 12, has_tty = True)) == "env"
    )


def test_exposed_no_tty_no_env_is_backstop():
    assert resolve_admin_password_source(**_base_kwargs(has_tty = False)) == "backstop"


def test_empty_env_treated_as_env_not_backstop():
    # An explicitly empty UNSLOTH_STUDIO_ADMIN_PASSWORD must fail fast through the
    # min-length guard, not silently fall back to the bootstrap state.
    assert (
        resolve_admin_password_source(**_base_kwargs(env_password = "", has_tty = False))
        == "env"
    )


def test_unset_env_none_falls_through_to_tty():
    assert resolve_admin_password_source(**_base_kwargs(env_password = None)) == "prompt"


def test_not_exposed_is_skip():
    assert resolve_admin_password_source(**_base_kwargs(exposed = False)) == "skip"


def test_api_only_no_frontend_is_skip():
    assert resolve_admin_password_source(**_base_kwargs(frontend_served = False)) == "skip"


def test_colab_is_skip():
    assert resolve_admin_password_source(**_base_kwargs(is_colab = True)) == "skip"


def test_already_changed_is_skip():
    assert resolve_admin_password_source(**_base_kwargs(requires_change = False)) == "skip"


# ── _is_exposed_bind ────────────────────────────────────────────────


def test_secure_is_exposed_even_on_loopback():
    assert _is_exposed_bind("127.0.0.1", secure = True) is True


def test_wildcard_binds_are_exposed():
    assert _is_exposed_bind("0.0.0.0", secure = False) is True
    assert _is_exposed_bind("::", secure = False) is True


def test_loopback_bind_is_not_exposed():
    assert _is_exposed_bind("127.0.0.1", secure = False) is False


# ── prompt_new_admin_password (injected reader) ─────────────────────


def _scripted_reader(values):
    it = iter(values)

    def _reader(_prompt):
        return next(it)

    return _reader


def test_prompt_returns_matching_password():
    pw = prompt_new_admin_password(
        reader = _scripted_reader(["correct horse battery", "correct horse battery"])
    )
    assert pw == "correct horse battery"


def test_prompt_rejects_too_short_then_accepts():
    # A short entry is rejected before the confirm prompt, so only one read for it.
    reader = _scripted_reader(["short", "longenoughpw", "longenoughpw"])
    assert prompt_new_admin_password(reader = reader) == "longenoughpw"


def test_prompt_rejects_mismatch_then_accepts():
    reader = _scripted_reader(["longenoughpw", "different-typo", "longenoughpw", "longenoughpw"])
    assert prompt_new_admin_password(reader = reader) == "longenoughpw"


def test_prompt_exhausts_attempts_raises():
    reader = _scripted_reader(["longenoughpw", "nope-mismatch", "longenoughpw", "nope-mismatch"])
    with pytest.raises(SystemExit):
        prompt_new_admin_password(attempts = 2, reader = reader)


def test_min_length_matches_frontend_rule():
    assert MIN_PASSWORD_LENGTH == 8
