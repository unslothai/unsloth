# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused coverage for cached, rate-limited HF token validation."""

from __future__ import annotations

from pathlib import Path
import sys

import httpx
import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import utils.hf_token_validation as validation


@pytest.fixture(autouse = True)
def _reset_validation_state():
    validation.reset_hf_token_validation_state()
    yield
    validation.reset_hf_token_validation_state()


def test_cached_token_does_not_spend_another_attempt(monkeypatch):
    calls = []

    def _check(token):
        calls.append(token)
        return validation.TokenValidationResult(status = "valid")

    monkeypatch.setattr(validation, "_check_remote", _check)
    first = validation.validate_hf_token("hf_valid", rate_key = "user:ip")
    second = validation.validate_hf_token("hf_valid", rate_key = "user:ip")

    assert first.status == second.status == "valid"
    assert calls == ["hf_valid"]


def test_three_uncached_attempts_per_hour(monkeypatch):
    monkeypatch.setattr(
        validation,
        "_check_remote",
        lambda _token: validation.TokenValidationResult(status = "invalid"),
    )

    for index in range(3):
        result = validation.validate_hf_token(f"hf_bad_{index}", rate_key = "user:ip")
        assert result.status == "invalid"

    limited = validation.validate_hf_token("hf_bad_4", rate_key = "user:ip")
    assert limited.status == "rate_limited"
    assert limited.retry_after_seconds is not None
    assert limited.retry_after_seconds > 0

    other_user = validation.validate_hf_token("hf_other", rate_key = "other:ip")
    assert other_user.status == "invalid"


def test_window_rolls_forward(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(validation.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(validation, "_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(validation, "_WINDOW_SECONDS", 10.0)
    monkeypatch.setattr(
        validation,
        "_check_remote",
        lambda _token: validation.TokenValidationResult(status = "invalid"),
    )

    assert validation.validate_hf_token("hf_a", rate_key = "user:ip").status == "invalid"
    assert (
        validation.validate_hf_token("hf_b", rate_key = "user:ip").status
        == "rate_limited"
    )
    clock["now"] += 11.0
    assert validation.validate_hf_token("hf_b", rate_key = "user:ip").status == "invalid"


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(200, "valid"), (401, "invalid"), (429, "rate_limited"), (500, "unavailable")],
)
def test_remote_status_classification(monkeypatch, status_code, expected):
    response = httpx.Response(
        status_code,
        request = httpx.Request("GET", "https://huggingface.co/api/whoami-v2"),
        headers = {"Retry-After": "42"} if status_code == 429 else None,
    )

    class _Session:
        def get(self, url, *, headers, timeout):
            assert url == "https://huggingface.co/api/whoami-v2"
            assert headers["authorization"] == "Bearer hf_test"
            assert timeout == validation._REMOTE_TIMEOUT_SECONDS
            return response

    monkeypatch.setattr(validation, "get_session", lambda: _Session())
    result = validation._check_remote("hf_test")
    assert result.status == expected
    if status_code == 429:
        assert result.retry_after_seconds == 42


def test_wrapped_http_401_is_invalid(monkeypatch):
    response = httpx.Response(
        401,
        request = httpx.Request("GET", "https://huggingface.co/api/whoami-v2"),
    )

    class _Session:
        def get(self, _url, **_kwargs):
            error = RuntimeError("Invalid user token.")
            error.response = response
            raise error

    monkeypatch.setattr(validation, "get_session", lambda: _Session())
    assert validation._check_remote("hf_test").status == "invalid"


def test_remote_timeout_is_bounded_and_unavailable(monkeypatch):
    class _Session:
        def get(self, _url, *, headers, timeout):
            assert headers["authorization"] == "Bearer hf_test"
            assert timeout == validation._REMOTE_TIMEOUT_SECONDS
            raise TimeoutError("timed out")

    monkeypatch.setattr(validation, "get_session", lambda: _Session())
    assert validation._check_remote("hf_test").status == "unavailable"


def test_raw_token_is_not_retained(monkeypatch):
    monkeypatch.setattr(
        validation,
        "_check_remote",
        lambda _token: validation.TokenValidationResult(status = "valid"),
    )
    token = "hf_do_not_store_this_value"
    validation.validate_hf_token(token, rate_key = "user:ip")

    assert token not in repr(validation._cache)
    assert token not in repr(validation._attempts)


def test_unexpected_remote_exception_releases_singleflight(monkeypatch):
    calls = 0
    monkeypatch.setattr(validation, "_INFLIGHT_WAIT_SECONDS", 0.0)

    def _check(_token):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("unexpected failure")
        return validation.TokenValidationResult(status = "valid")

    monkeypatch.setattr(validation, "_check_remote", _check)

    with pytest.raises(RuntimeError, match = "unexpected failure"):
        validation.validate_hf_token("hf_test", rate_key = "user:ip")

    result = validation.validate_hf_token("hf_test", rate_key = "user:ip")
    assert result.status == "valid"
    assert calls == 2
    assert validation._inflight == {}
