# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep first-party event streams usable through Cloudflare Quick Tunnels."""

import ast
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "relative_path,function_name,route_path",
    [
        ("routes/training.py", "stream_training_progress", "/progress"),
        ("routes/export.py", "stream_export_logs", "/logs/stream"),
        ("routes/rag.py", "job_events", "/jobs/{job_id}/events"),
        (
            "routes/rag.py",
            "folder_job_events",
            "/linked-folder-jobs/{job_id}/events",
        ),
        ("routes/data_recipe/jobs.py", "job_events", "/jobs/{job_id}/events"),
    ],
)
def test_first_party_event_streams_accept_post_and_get(
    relative_path: str, function_name: str, route_path: str
) -> None:
    tree = ast.parse((_BACKEND_ROOT / relative_path).read_text(encoding = "utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    decorators = {
        decorator.func.attr: decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.args
        and isinstance(decorator.args[0], ast.Constant)
        and decorator.args[0].value == route_path
    }

    assert set(decorators) >= {"get", "post"}
    get_options = {keyword.arg: keyword.value for keyword in decorators["get"].keywords}
    assert isinstance(get_options.get("include_in_schema"), ast.Constant)
    assert get_options["include_in_schema"].value is False
    assert not decorators["post"].keywords
