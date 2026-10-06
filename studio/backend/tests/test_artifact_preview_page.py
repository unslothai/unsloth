# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat HTML opened in the user's default browser is served at a capability URL. It must keep
the in-app frame's isolation (an opaque-origin sandbox, the network switch) as a top-level tab."""

import asyncio

import routes.inference as inf_mod


def _stage(html: str, allow_network: bool = False) -> str:
    request = inf_mod.ArtifactPreviewPageRequest(html = html, allow_network = allow_network)
    result = asyncio.run(inf_mod.create_artifact_preview_page(request, current_subject = "u"))
    return result["path"].rsplit("/", 1)[1]


def _serve(token: str):
    return asyncio.run(inf_mod.artifact_preview_page(token))


def test_a_staged_page_is_served_sandboxed_with_the_strict_policy():
    response = _serve(_stage("<!doctype html><html><head><title>Hi</title></head></html>"))
    csp = response.headers["content-security-policy"]
    assert response.status_code == 200
    assert "sandbox allow-scripts" in csp
    assert "allow-same-origin" not in csp
    assert "connect-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["cache-control"] == "no-store"


def test_the_network_switch_carries_over():
    csp = _serve(_stage("<p>x</p>", allow_network = True)).headers["content-security-policy"]
    assert "connect-src http: https:" in csp
    assert "allow-same-origin" not in csp


def test_the_storage_fallback_goes_inside_head_after_the_doctype():
    body = _serve(_stage("<!DOCTYPE html><html><head><title>Hi</title>")).body.decode()
    assert body.startswith("<!DOCTYPE html><html><head><script>")
    assert "localStorage" in body


def test_an_unknown_token_is_not_found():
    assert _serve("not-a-token").status_code == 404


def test_tokens_are_unguessable_and_distinct():
    first, second = _stage("<p>a</p>"), _stage("<p>a</p>")
    assert first != second
    assert len(first) >= 24
