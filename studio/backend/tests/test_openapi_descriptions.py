# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Public routes describe their contract in the generated OpenAPI document, not internal notes."""

import pytest


@pytest.fixture(scope = "module")
def openapi():
    import main
    return main.app.openapi()


@pytest.mark.parametrize(
    "path, method, phrase",
    [
        ("/api/auth/desktop-login", "post", "Exchange a local desktop secret"),
        ("/v1/images/generations", "post", "OpenAI-compatible text-to-image"),
        ("/api/inference/images/generations", "post", "OpenAI-compatible text-to-image"),
        ("/api/health", "get", "Liveness plus launcher capability bits"),
    ],
)
def test_route_description_states_the_contract(openapi, path, method, phrase):
    assert phrase in openapi["paths"][path][method]["description"]


def test_chat_completion_schema_description(openapi):
    schema = openapi["components"]["schemas"]["ChatCompletionRequest"]
    assert schema["description"].startswith("OpenAI-compatible chat completion request.")
