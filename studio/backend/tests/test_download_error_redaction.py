# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hub download errors and recipe job logs are masked by the same redactor as the log viewer."""

import sys
import types

import pytest

from hub.utils.download_registry import scrub_secrets

SIGNED_URL = (
    "401 Client Error: Unauthorized for url: https://cas-bridge.xethub.hf.co/xet-bridge-us/abc/def"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA%2F20261124%2Fus-east-1"
    "&X-Amz-Signature=0123456789abcdef0123456789abcdef&X-Amz-SignedHeaders=host"
)


def test_scrub_secrets_masks_a_signed_cdn_url():
    cleaned = scrub_secrets(SIGNED_URL)
    assert "0123456789abcdef" not in cleaned
    assert "AKIA%2F20261124" not in cleaned
    assert cleaned.startswith(
        "401 Client Error: Unauthorized for url: https://cas-bridge.xethub.hf.co/"
    )
    assert "X-Amz-SignedHeaders=host" in cleaned


def test_scrub_secrets_masks_url_userinfo_and_authorization_header():
    cleaned = scrub_secrets("Connection to https://alice:hunter2pass@mirror.example.com/m failed")
    assert "hunter2pass" not in cleaned
    assert "mirror.example.com/m failed" in cleaned
    header = scrub_secrets("headers: {'Authorization': 'Basic YWxpY2U6aHVudGVyMg=='}")
    assert "YWxpY2U6aHVudGVyMg" not in header
    assert "Basic" in header


def test_scrub_secrets_masks_a_short_token_in_an_explicit_authorization_header():
    assert scrub_secrets("request headers: Authorization: Bearer letmein") == (
        "request headers: Authorization: Bearer <redacted>"
    )


def test_scrub_secrets_masks_the_given_token_and_hub_tokens():
    token = "plain-token-value-42"
    cleaned = scrub_secrets(f"rejected {token}; also hf_{'A' * 34}", hf_token = token)
    assert token not in cleaned
    assert "A" * 34 not in cleaned


@pytest.mark.parametrize(
    "text",
    [
        "Repository Not Found for url: https://huggingface.co/api/models/org/model/revision/main",
        "No space left on device: /tmp/hf/models--org--model/blobs",
        "401 Client Error: Unauthorized. Bearer credentials were not accepted",
        "",
    ],
)
def test_scrub_secrets_leaves_ordinary_errors_alone(text):
    assert scrub_secrets(text) == text


@pytest.fixture(name = "worker")
def fixture_worker(monkeypatch):
    try:
        import data_designer.config.utils.io_helpers  # noqa: F401
    except ImportError:
        module = types.ModuleType("data_designer.config.utils.io_helpers")
        module.read_parquet_dataset = lambda path: None
        for name in ("data_designer", "data_designer.config", "data_designer.config.utils"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    from core.data_recipe.jobs import worker
    return worker


def test_recipe_job_log_masks_tokens_and_keeps_cursor_redaction(worker):
    github = "ghp_" + "a" * 36
    unsloth_key = "sk-unsloth-" + "0" * 32
    hf = "hf_" + "B" * 34
    message = worker._sanitize_log_message(
        f"page cursor=Y3Vyc29yOnYyOpK0, {github} {unsloth_key} {hf} api_key=abcdef123456"
    )
    assert "cursor=<redacted>," in message
    for secret in ("Y3Vyc29y", github, unsloth_key, hf, "abcdef123456"):
        assert secret not in message
    assert worker._sanitize_log_message("Generated 12 rows in 3.4s") == "Generated 12 rows in 3.4s"


@pytest.mark.parametrize(
    "text",
    [
        "HfHubHTTPError('401 Client Error\\nhf_AbCdEfGhIjKlMnOpQrStUvWxYz012345 rejected')",
        "GET /api/models?auth=Bearer%20hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
    ],
)
def test_scrub_secrets_masks_an_hf_token_glued_to_a_word_character(text):
    assert "AbCdEfGhIjKlMnOpQrStUvWxYz012345" not in scrub_secrets(text)


def test_recipe_job_log_keeps_the_old_github_and_studio_key_coverage(worker):
    unsloth_key = "sk-unsloth-" + "0123456789abcdef" * 2
    message = worker._sanitize_log_message(f"wrote run-{unsloth_key}.json, token ghp_abc12 cut")
    assert unsloth_key not in message and "ghp_abc12" not in message
    assert message.startswith("wrote run-") and message.endswith(".json, token <redacted> cut")


@pytest.mark.parametrize(
    "text",
    [
        "Repository Not Found for url: https://huggingface.co/api/models/org/disk-unsloth-test",
        "loading task-unsloth-lora adapter",
    ],
)
def test_scrub_secrets_keeps_a_name_that_merely_contains_sk_unsloth(text):
    assert scrub_secrets(text) == text


def test_a_dotted_github_app_token_is_masked_whole(worker):
    token = "ghs_123456_eyJhbGciOiJIUzI1NiJ9.eyJpc3MiOiIxMjMifQ.c2lnbmF0dXJl"
    assert (
        worker._sanitize_log_message(f"push failed with {token}") == "push failed with <redacted>"
    )
