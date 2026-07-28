# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest

from core.user_assets_validation import (
    MAX_TIMESTAMP_MS,
    UserAssetValidationError,
    validate_id,
    validate_recipe_payload,
    validate_timestamp,
)


ROOT = Path(__file__).resolve().parents[3]


def test_persistence_policy_is_declared_as_studio_package_data():
    tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding = "utf-8"))
    studio_files = project["tool"]["setuptools"]["package-data"]["studio"]
    assert "user_assets_persistence_policy.json" in studio_files


def test_timestamp_rejects_values_outside_the_javascript_date_range():
    assert validate_timestamp(MAX_TIMESTAMP_MS, "createdAt") == MAX_TIMESTAMP_MS
    for value in (-1, MAX_TIMESTAMP_MS + 1, 2**63, True, 1.5):
        with pytest.raises(UserAssetValidationError, match = "valid Date range"):
            validate_timestamp(value, "createdAt")


@pytest.mark.parametrize("asset_id", [".", ".."])
def test_asset_ids_reject_url_dot_segments(asset_id):
    with pytest.raises(UserAssetValidationError, match = "URL path segment"):
        validate_id(asset_id)


def test_proxy_authorization_is_rejected_and_redacted_by_the_same_policy():
    payload = {"headers": {"Proxy-Authorization": "Basic secret"}}
    with pytest.raises(UserAssetValidationError, match = "secret fields"):
        validate_recipe_payload(payload)
    clean, paths = validate_recipe_payload(payload, legacy = True)
    assert clean == {"headers": {}}
    assert paths == ['$.headers["Proxy-Authorization"]']


@pytest.mark.parametrize(
    "key",
    ["private_key", "privateKey", "access_key", "access_key_id", "accessKeyId"],
)
def test_exact_private_and_access_key_names_are_rejected_and_redacted(key):
    payload = {"credentials": {key: "secret"}}
    with pytest.raises(UserAssetValidationError, match = "secret fields"):
        validate_recipe_payload(payload)
    clean, paths = validate_recipe_payload(payload, legacy = True)
    assert clean == {"credentials": {}}
    assert paths == [f"$.credentials.{key}"]


@pytest.mark.parametrize("key", ["secret_key", "secretKey"])
def test_normalized_secret_key_aliases_are_rejected_and_redacted(key):
    payload = {"credentials": {key: "secret"}}
    with pytest.raises(UserAssetValidationError, match = "secret fields"):
        validate_recipe_payload(payload)
    clean, paths = validate_recipe_payload(payload, legacy = True)
    assert clean == {"credentials": {}}
    assert paths == [f"$.credentials.{key}"]


def test_explicitly_safe_secret_looking_key_still_round_trips():
    payload = {"provider": {"api_key_env": "PROVIDER_API_KEY"}}
    assert validate_recipe_payload(payload) == payload
    assert validate_recipe_payload(payload, legacy = True) == (payload, [])


def test_structured_output_schema_preserves_secret_looking_property_names():
    payload = {
        "recipe": {
            "columns": [
                {
                    "column_type": "llm-structured",
                    "output_format": {
                        "type": "object",
                        "required": ["api_key", "password"],
                        "properties": {
                            "api_key": {"type": "string"},
                            "password": {
                                "type": "object",
                                "properties": {
                                    "secretKey": {"type": "string"},
                                },
                            },
                        },
                    },
                }
            ]
        }
    }

    assert validate_recipe_payload(payload) == payload
    assert validate_recipe_payload(payload, legacy = True) == (payload, [])


def test_schema_exception_does_not_allow_actual_or_malformed_credentials():
    actual_credential = {
        "recipe": {
            "columns": [
                {
                    "column_type": "llm-structured",
                    "api_key": "credential-value",
                    "output_format": {"type": "object", "properties": {}},
                }
            ]
        }
    }
    malformed_schema_credential = {
        "recipe": {
            "columns": [
                {
                    "column_type": "llm-structured",
                    "output_format": {
                        "type": "object",
                        "properties": {"password": "credential-value"},
                    },
                }
            ]
        }
    }

    for payload in (actual_credential, malformed_schema_credential):
        with pytest.raises(UserAssetValidationError, match = "secret fields"):
            validate_recipe_payload(payload)
        clean, paths = validate_recipe_payload(payload, legacy = True)
        assert "credential-value" not in str(clean)
        assert len(paths) == 1


@pytest.mark.parametrize(
    "credential_key",
    [
        "private_key",
        "access_key_id",
        "AWS_ACCESS_KEY_ID",
        "MY_SERVICE_KEY",
        "MY_SERVICE_TOKEN",
    ],
)
def test_mcp_stdio_env_credentials_are_rejected_and_redacted(credential_key):
    payload = {
        "recipe": {
            "mcp_providers": [
                {
                    "provider_type": "stdio",
                    "env": {
                        "NODE_ENV": "production",
                        credential_key: "credential-value",
                    },
                }
            ]
        }
    }

    with pytest.raises(UserAssetValidationError, match = "secret fields"):
        validate_recipe_payload(payload)

    clean, paths = validate_recipe_payload(payload, legacy = True)
    env = clean["recipe"]["mcp_providers"][0]["env"]
    assert env == {"NODE_ENV": "production"}
    assert paths == [f"$.recipe.mcp_providers[0].env.{credential_key}"]


def test_mcp_stdio_operational_env_values_round_trip():
    payload = {
        "recipe": {
            "mcp_providers": [
                {
                    "provider_type": "stdio",
                    "env": {
                        "NODE_ENV": "production",
                        "LOG_LEVEL": "debug",
                        "SERVICE_REGION": "eu-west-1",
                    },
                }
            ]
        }
    }

    assert validate_recipe_payload(payload) == payload
    clean, paths = validate_recipe_payload(payload, legacy = True)
    assert clean == payload
    assert paths == []
