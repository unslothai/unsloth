# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The local inventory answers an API key without handing it the host's filesystem.

An API key is authenticated, so it keeps the listing: repo ids, sizes, capabilities, partial
state, everything it needs to know what is downloaded. What it stops receiving is the absolute
cache directory, the models folder and the scan roots, which say where the operator keeps their
files and what their account is called, and which the caller never needed in order to enumerate.

The browser session is unchanged, deliberately: it is the operator looking at their own machine,
it renders the path and it hands it back to the delete endpoint.

Every assertion here is on the serialised response body, because the response model is where a
field survives or is dropped, and a unit test on the helper alone would pass with the helper
wired to nothing.
"""

import ast
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from hub.routes import inventory as inventory_routes
from hub.schemas.inventory import (
    HiddenModelsResponse,
    OrphanCompanionInfo,
    LocalModelListResponse,
    ModelsFolderResponse,
    OrphanCompanionsResponse,
    ScanFolderInfo,
    ScanFoldersResponse,
)
from hub.services.models import cache_inventory, companion_cleanup, local_inventory
from hub.utils import host_paths
from hub.utils.host_paths import (
    cache_reference,
    redact_host_paths,
    redact_inventory_host_paths,
    response_leaks_host_path,
    scrub_paths,
    short_path_for_log,
)
from routes import models as models_routes

# ``load_id`` is the one path-valued field an API key still receives, because it is the handle
# the load and train endpoints take for a copy its bare repo id cannot reach. Named here so every
# tolerant assertion below says which field it is tolerating, and why, at the call site.
LOAD_HANDLE = ("load_id",)

# A root that cannot exist by accident, so finding it in a body is proof and not a coincidence.
HOST_ROOT = "/home/operator-7f3c/.cache/huggingface/hub"
REPO_DIR = f"{HOST_ROOT}/models--unsloth--Llama-3.2-1B-Instruct"


def _client(router, prefix: str, *, via_api_key: bool) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app, raise_server_exceptions = False)


def _hub(via_api_key: bool) -> TestClient:
    return _client(inventory_routes.router, "/api/hub", via_api_key = via_api_key)


def _models(via_api_key: bool) -> TestClient:
    return _client(models_routes.router, "/api/models", via_api_key = via_api_key)


def _cached_row(**extra) -> dict:
    row = {
        "repo_id": "unsloth/Llama-3.2-1B-Instruct",
        "size_bytes": 2471234567,
        "cache_path": REPO_DIR,
        "partial": False,
        "model_format": "safetensors",
        "runtime": "transformers",
        "load_id": "unsloth/Llama-3.2-1B-Instruct",
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------------------


def test_a_reference_is_opaque_stable_and_not_the_path():
    first = cache_reference(REPO_DIR)
    assert first is not None
    assert first == cache_reference(REPO_DIR)
    assert first != cache_reference(REPO_DIR + "x")
    assert REPO_DIR not in first
    assert "operator-7f3c" not in first
    assert cache_reference("") is None
    assert cache_reference(None) is None


def test_a_ui_session_payload_is_returned_untouched():
    payload = {"cached": [_cached_row()]}
    assert redact_host_paths(payload, via_api_key = False) is payload
    assert redact_inventory_host_paths(payload, via_api_key = False) is payload


def test_redaction_keeps_every_non_path_field():
    payload = {"cached": [_cached_row()], "scan_confirmed": True}
    out = redact_host_paths(payload, via_api_key = True)
    row = out["cached"][0]
    assert row["repo_id"] == "unsloth/Llama-3.2-1B-Instruct"
    assert row["size_bytes"] == 2471234567
    assert row["model_format"] == "safetensors"
    assert out["scan_confirmed"] is True
    assert row["cache_path"] == ""
    assert row["cache_ref"] == cache_reference(REPO_DIR)


def test_a_null_path_stays_null_so_the_discriminator_survives():
    """``cache_path`` null means "no cache dir at all" to the download manager, and an empty
    string means "there is one, you may not see it". Redaction must not turn one into the other."""
    out = redact_host_paths({"cache_path": None}, via_api_key = True)
    assert out["cache_path"] is None
    assert "cache_ref" not in out


def test_the_ambiguous_path_field_is_only_redacted_where_it_is_a_path():
    payload = {"path": REPO_DIR}
    assert redact_host_paths(payload, via_api_key = True)["path"] == REPO_DIR
    assert redact_inventory_host_paths(payload, via_api_key = True)["path"] == ""


def test_scan_root_lists_are_emptied_not_referenced():
    payload = {"lmstudio_dirs": [f"{HOST_ROOT}/lm"], "exact_paths": [REPO_DIR]}
    out = redact_host_paths(payload, via_api_key = True)
    assert out["lmstudio_dirs"] == []
    assert out["exact_paths"] == []


def test_the_leak_detector_finds_what_it_is_for():
    assert response_leaks_host_path({"cached": [_cached_row()]}, [HOST_ROOT]) is not None
    assert (
        response_leaks_host_path(
            redact_host_paths({"cached": [_cached_row()]}, via_api_key = True), [HOST_ROOT]
        )
        is None
    )
    # A root anywhere in the body, under a field name this module has never heard of.
    assert response_leaks_host_path({"detail": f"could not read {REPO_DIR}"}, [HOST_ROOT])


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            f"Skipping {REPO_DIR}: Permission denied",
            "Skipping .../hub/models--unsloth--Llama-3.2-1B-Instruct: Permission denied",
        ),
        (
            "ratio 3/4 and https://huggingface.co/api/models",
            "ratio 3/4 and https://huggingface.co/api/models",
        ),
        ("nothing to see", "nothing to see"),
    ],
)
def test_log_messages_lose_the_layout_and_keep_the_meaning(message, expected):
    assert scrub_paths(message) == expected


def test_scrub_paths_takes_an_exception_not_only_a_string():
    assert scrub_paths(OSError(f"cannot open {REPO_DIR}/blobs/abc")) == (
        "cannot open .../blobs/abc"
    )
    assert scrub_paths(None) == ""
    assert short_path_for_log(Path(REPO_DIR)) == ".../hub/models--unsloth--Llama-3.2-1B-Instruct"


# --------------------------------------------------------------------------------------
# The routes, over HTTP, both caller classes
# --------------------------------------------------------------------------------------


@pytest.fixture
def _cached_inventory(monkeypatch):
    async def _models_response(hf_token = None):
        return {"cached": [_cached_row()], "scan_confirmed": True}

    async def _gguf_response(hf_token = None):
        return {
            "cached": [
                _cached_row(
                    repo_id = "unsloth/gemma-3-270m-it-GGUF",
                    model_format = "gguf",
                    cache_path = f"{HOST_ROOT}/models--unsloth--gemma-3-270m-it-GGUF",
                    load_id = f"{HOST_ROOT}/models--unsloth--gemma-3-270m-it-GGUF/snapshots/abc",
                )
            ],
            "scan_confirmed": True,
        }

    monkeypatch.setattr(cache_inventory, "list_cached_models_response", _models_response)
    monkeypatch.setattr(cache_inventory, "list_cached_gguf_response", _gguf_response)
    monkeypatch.setattr(models_routes, "cached_model_rows", lambda *a, **k: [_cached_row()])
    monkeypatch.setattr(models_routes, "cached_gguf_rows", lambda *a, **k: [_cached_row()])


@pytest.mark.parametrize(
    "route",
    ["/api/hub/cached-models", "/api/hub/cached-gguf"],
)
def test_an_api_key_enumerates_the_cache_without_its_paths(_cached_inventory, route):
    body = _hub(via_api_key = True).get(route)
    assert body.status_code == 200
    payload = body.json()
    assert [row["repo_id"] for row in payload["cached"]], "the listing must still answer"
    assert response_leaks_host_path(payload, [HOST_ROOT], ignore = LOAD_HANDLE) is None
    for row in payload["cached"]:
        assert row["cache_path"] == ""
        assert row["cache_ref"].startswith("ref:")
        assert HOST_ROOT not in json.dumps({k: v for k, v in row.items() if k != "load_id"})


@pytest.mark.parametrize(
    "route",
    ["/api/hub/cached-models", "/api/hub/cached-gguf"],
)
def test_a_ui_session_still_sees_its_own_machine(_cached_inventory, route):
    payload = _hub(via_api_key = False).get(route).json()
    assert payload["cached"]
    assert all(row["cache_path"].startswith(HOST_ROOT) for row in payload["cached"])


@pytest.mark.parametrize(
    "route",
    ["/api/models/cached-models", "/api/models/cached-gguf"],
)
def test_the_models_mirror_of_the_inventory_is_covered_too(_cached_inventory, route):
    payload = _models(via_api_key = True).get(route).json()
    assert payload["cached"]
    assert response_leaks_host_path(payload, [HOST_ROOT], ignore = LOAD_HANDLE) is None


def test_the_models_folder_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(
        local_inventory, "get_models_folder_response", lambda: ModelsFolderResponse(path = HOST_ROOT)
    )
    assert _hub(via_api_key = True).get("/api/hub/models-folder").json()["path"] == ""
    assert _hub(via_api_key = False).get("/api/hub/models-folder").json()["path"] == HOST_ROOT


def test_the_scan_folder_list_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(
        local_inventory,
        "get_scan_folders_response",
        lambda: ScanFoldersResponse(
            folders = [ScanFolderInfo(id = 1, path = f"{HOST_ROOT}/extra", created_at = "2026-09-01")]
        ),
    )
    payload = _hub(via_api_key = True).get("/api/hub/scan-folders").json()
    assert payload["folders"], "the folder is still listed, by id"
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None


def test_the_local_model_scan_is_not_disclosed(monkeypatch):
    async def _response(models_dir = "./models"):
        return LocalModelListResponse(
            models_dir = f"{HOST_ROOT}/../models",
            hf_cache_dir = HOST_ROOT,
            lmstudio_dirs = [f"{HOST_ROOT}/lmstudio"],
            ollama_dirs = [],
            hermes_dirs = [],
            models = [],
        )

    monkeypatch.setattr(local_inventory, "list_local_models_response", _response)
    payload = _hub(via_api_key = True).get("/api/hub/local").json()
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None
    assert payload["lmstudio_dirs"] == []
    ui = _hub(via_api_key = False).get("/api/hub/local").json()
    assert ui["hf_cache_dir"] == HOST_ROOT


def test_the_hidden_model_matchers_keep_their_ids_and_drop_their_paths(monkeypatch):
    monkeypatch.setattr(
        models_routes,
        "hidden_model_matchers",
        lambda: (["needle"], ["org/repo"], [REPO_DIR]),
    )
    payload = _hub(via_api_key = True).get("/api/hub/hidden-models").json()
    assert payload["needles"] == ["needle"]
    assert payload["exact_ids"] == ["org/repo"]
    assert payload["exact_paths"] == []
    ui = _hub(via_api_key = False).get("/api/hub/hidden-models").json()
    assert ui["exact_paths"] == [REPO_DIR]


def test_orphan_companions_keep_their_repo_ids(monkeypatch):
    async def _response():
        return OrphanCompanionsResponse(
            companions = [
                OrphanCompanionInfo(
                    repo_id = "org/vae",
                    size_bytes = 12,
                    cache_path = f"{HOST_ROOT}/models--org--vae",
                )
            ],
            total_bytes = 12,
        )

    monkeypatch.setattr(companion_cleanup, "orphan_companions_response", _response)
    payload = _hub(via_api_key = True).get("/api/hub/orphan-companions").json()
    assert payload["companions"][0]["repo_id"] == "org/vae"
    assert payload["companions"][0]["cache_ref"].startswith("ref:")
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None


def test_download_progress_hides_the_cache_dir_it_measured(monkeypatch):
    async def _progress(
        repo_id,
        expected_bytes = 0,
        hf_token = None,
    ):
        return {
            "repo_id": repo_id,
            "downloaded_bytes": 10,
            "expected_bytes": 100,
            "progress": 0.1,
            "complete": False,
            "cache_path": REPO_DIR,
        }

    from hub.services.models import downloads

    monkeypatch.setattr(downloads, "get_download_progress_response", _progress)
    payload = (
        _hub(via_api_key = True)
        .get("/api/hub/download-progress", params = {"repo_id": "org/repo"})
        .json()
    )
    assert payload["cache_path"] == ""
    assert payload["cache_ref"].startswith("ref:")
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None


def test_an_api_key_can_still_delete_by_repo_id_without_ever_seeing_a_path(monkeypatch):
    """The reason the inventory may keep answering: nothing it hands back is load bearing.

    ``cache_path`` on the delete request is optional, so a caller that was never given one
    still names the row by repo id and variant.
    """
    seen = {}

    async def _delete(repo_id, variant, hf_token, cache_path, only_if_orphan):
        seen.update(
            repo_id = repo_id,
            variant = variant,
            cache_path = cache_path,
            only_if_orphan = only_if_orphan,
        )
        return {"status": "deleted", "repo_id": repo_id, "variant": variant}

    from hub.services.models import deletion

    monkeypatch.setattr(deletion, "delete_cached_model_response", _delete)
    response = _hub(via_api_key = True).request(
        "DELETE",
        "/api/hub/delete-cached",
        json = {"repo_id": "unsloth/Llama-3.2-1B-Instruct", "variant": None},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert seen["cache_path"] is None


# --------------------------------------------------------------------------------------
# The drift gate
# --------------------------------------------------------------------------------------

# Routes on the hub inventory router that answer no host path, with the reason. Anything else
# must take the caller class, so a new route cannot quietly reopen this.
_ROUTES_WITHOUT_HOST_PATHS = {
    "remove_scan_folder_endpoint": "status and id only",
    "get_gguf_variants": "filenames and quant labels, no directory",
    "download_model": "job id and status",
    "cancel_download_model": "job id and status",
    "get_download_status": "job status",
    "get_active_downloads": "repo ids and byte counts",
    "delete_impact": "repo ids, byte counts and blockers",
    "delete_cached_model": "status, repo id and variant",
}


def test_every_inventory_route_that_could_answer_a_path_takes_the_caller_class():
    source = Path(inventory_routes.__file__).read_text()
    tree = ast.parse(source)
    missing = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorated = any(
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and isinstance(dec.func.value, ast.Name)
            and dec.func.value.id == "router"
            for dec in node.decorator_list
        )
        if not decorated:
            continue
        args = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        if "via_api_key" in args:
            continue
        if node.name in _ROUTES_WITHOUT_HOST_PATHS:
            continue
        missing.append(node.name)
    assert not missing, (
        "these inventory routes neither take via_api_key nor are recorded as path-free: "
        f"{missing}. Add the dependency and redact, or list the route with its reason."
    )


def test_the_path_field_lists_cover_the_inventory_schemas():
    """A field added to a cached row with a path-shaped name must join the redaction list."""
    from hub.schemas import inventory as inventory_schemas

    known = (
        host_paths.HOST_PATH_SCALAR_FIELDS
        | host_paths.HOST_PATH_LIST_FIELDS
        | {host_paths.HOST_PATH_AMBIGUOUS_FIELD}
    )
    suspicious = []
    for name in dir(inventory_schemas):
        model = getattr(inventory_schemas, name)
        fields = getattr(model, "model_fields", None)
        if not isinstance(fields, dict):
            continue
        for field in fields:
            if field in known:
                continue
            if field.endswith(("_path", "_paths", "_dir", "_dirs")) or field in {"path", "paths"}:
                suspicious.append(f"{name}.{field}")
    assert not suspicious, (
        "path-shaped response fields that redaction does not know about: " + repr(suspicious)
    )
