# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep first-party event streams usable through Cloudflare Quick Tunnels.

Measured on three fresh quick tunnels, one generator on both verbs so only the method
differs: GET delivers its first byte when the stream closes (~12s), POST in under 300ms,
and no response header recovers GET. Checks registration rather than source text, since a
route that stopped resolving or lost a dependency would still spell "post" in the file.
"""

import importlib

import pytest


_CASES = [
    ("routes.training", "/progress", "stream_training_progress"),
    ("routes.export", "/logs/stream", "stream_export_logs"),
    ("routes.rag", "/jobs/{job_id}/events", "job_events"),
    ("routes.rag", "/linked-folder-jobs/{job_id}/events", "folder_job_events"),
    ("routes.data_recipe.jobs", "/jobs/{job_id}/events", "job_events"),
]


def _routes_at(module_name: str, route_path: str) -> list:
    router = importlib.import_module(module_name).router
    return [r for r in router.routes if getattr(r, "path", None) == route_path]


@pytest.mark.parametrize("module_name,route_path,function_name", _CASES)
def test_first_party_event_streams_accept_post_and_get(
    module_name: str, route_path: str, function_name: str
) -> None:
    routes = _routes_at(module_name, route_path)
    by_method = {m: r for r in routes for m in r.methods if m in ("GET", "POST")}

    assert set(by_method) == {"GET", "POST"}
    assert by_method["GET"].endpoint is by_method["POST"].endpoint
    assert by_method["GET"].endpoint.__name__ == function_name
    # Only POST is public, so a generated client picks the verb that survives a tunnel.
    assert by_method["POST"].include_in_schema is True
    assert by_method["GET"].include_in_schema is False


@pytest.mark.parametrize("module_name,route_path,function_name", _CASES)
def test_both_verbs_carry_the_same_dependencies(
    module_name: str, route_path: str, function_name: str
) -> None:
    """A second registration re-runs the decorator, so divergence here is divergent auth."""
    signatures = {
        tuple(sorted(d.call.__name__ for d in r.dependant.dependencies))
        for r in _routes_at(module_name, route_path)
    }
    assert len(signatures) == 1, signatures


def test_only_post_reaches_the_schema() -> None:
    """One api_route for both verbs would give them one operationId; two registrations
    with GET hidden keeps the ids unique and the public contract single."""
    import main

    schema = main.app.openapi()
    for full_path in (
        "/api/train/progress",
        "/api/export/logs/stream",
        "/api/rag/jobs/{job_id}/events",
        "/api/rag/linked-folder-jobs/{job_id}/events",
        "/api/data-recipe/jobs/{job_id}/events",
        "/api/chat/research-runs/{run_id}/events",
    ):
        assert full_path in schema["paths"], full_path
        assert {v for v in schema["paths"][full_path] if v in ("get", "post")} == {"post"}

    operation_ids = [
        operation["operationId"]
        for path_item in schema["paths"].values()
        for verb, operation in path_item.items()
        if verb in ("get", "post", "put", "patch", "delete") and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))
