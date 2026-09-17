# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An API key gets the local inventory listing but not the host paths; a browser session still
gets them. Assertions are on the serialised response body, since a unit test on the helper alone
would pass with the helper wired to nothing.
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

# Empty: no field is exempt from redaction any more, including ``load_id``.
LOAD_HANDLE: "tuple[str, ...]" = ()

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
        assert HOST_ROOT not in json.dumps(row), row


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


def test_a_rejected_scan_folder_does_not_disclose_where_it_resolved(monkeypatch):
    from fastapi import HTTPException

    def _raises(path):
        raise HTTPException(
            status_code = 400,
            detail = f"Path is not readable: [Errno 36] File name too long: '{REPO_DIR}'",
        )

    monkeypatch.setattr(local_inventory, "add_scan_folder_response", _raises)

    response = _hub(via_api_key = True).post("/api/hub/scan-folders", json = {"path": "./models"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert HOST_ROOT not in detail, detail
    assert REPO_DIR not in detail, detail
    assert "File name too long" in detail, detail

    ui = _hub(via_api_key = False).post("/api/hub/scan-folders", json = {"path": "./models"})
    assert REPO_DIR in ui.json()["detail"]


def test_the_models_folder_error_is_not_disclosed_either(monkeypatch):
    from fastapi import HTTPException

    def _raises():
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to create models folder: {HOST_ROOT}/hub: Permission denied",
        )

    monkeypatch.setattr(local_inventory, "get_models_folder_response", _raises)

    response = _hub(via_api_key = True).get("/api/hub/models-folder")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert HOST_ROOT not in detail, detail
    assert "Permission denied" in detail, detail
    assert "Failed to create models folder" in detail, detail

    session = _hub(via_api_key = False).get("/api/hub/models-folder")
    assert session.status_code == 500
    assert HOST_ROOT in session.json()["detail"]


@pytest.mark.parametrize("root", ["/tmp", "/cache", "C:\\cache"])
def test_a_root_with_one_component_is_redacted_too(monkeypatch, root):
    from fastapi import HTTPException

    def _raises():
        raise HTTPException(
            status_code = 500,
            detail = f"Models folder path is not a directory: {root}",
        )

    monkeypatch.setattr(local_inventory, "get_models_folder_response", _raises)
    detail = _hub(via_api_key = True).get("/api/hub/models-folder").json()["detail"]
    assert root not in detail, detail
    assert "not a directory" in detail, detail


def test_ordinary_prose_is_not_read_as_a_path(monkeypatch):
    from hub.utils.host_paths import redact_paths_in_text
    for kept in ("3/4 of the shards", "and/or the projector", "no paths here"):
        assert redact_paths_in_text(kept) == kept, kept


def test_a_structured_error_detail_is_walked_too(monkeypatch):
    from fastapi import HTTPException

    def _raises():
        raise HTTPException(
            status_code = 500,
            detail = {"message": "bad folder", "path": f"{HOST_ROOT}/hub"},
        )

    monkeypatch.setattr(local_inventory, "get_models_folder_response", _raises)
    detail = _hub(via_api_key = True).get("/api/hub/models-folder").json()["detail"]
    assert HOST_ROOT not in str(detail), detail
    assert detail["message"] == "bad folder"


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


# Hub inventory routes that answer no host path. Anything else must take the caller class.
_ROUTES_WITHOUT_HOST_PATHS = {
    "remove_scan_folder_endpoint": "status and id only",
    "get_gguf_variants": "filenames and quant labels, no directory",
    "download_model": "job id and status",
    "cancel_download_model": "job id and status",
    "get_active_downloads": "repo ids and byte counts",
    "delete_impact": "repo ids, byte counts and blockers",
    "delete_cached_model": "status, repo id and variant",
}


def test_every_inventory_route_that_could_answer_a_path_takes_the_caller_class():
    source = Path(inventory_routes.__file__).read_text(encoding = "utf-8")
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


def test_a_windows_network_share_is_shortened_like_any_other_path():
    assert scrub_paths(r"Skipping \\fileserver\models\hub\models--acme--x: denied") == (
        "Skipping .../hub/models--acme--x: denied"
    )
    assert scrub_paths(r"open \\srv\share\acme\config.json failed") == (
        "open .../acme/config.json failed"
    )
    assert scrub_paths("ratio 3/4") == "ratio 3/4"
    assert scrub_paths("https://huggingface.co/api/models/a/b") == (
        "https://huggingface.co/api/models/a/b"
    )


def test_a_relative_path_in_a_log_line_is_left_as_written():
    for line in (
        "Skipping ./models/repo: denied",
        "Skipping ../models/repo: denied",
        "Skipping ../../a/b/c: denied",
        "Skipping models/team/repo: denied",
    ):
        assert scrub_paths(line) == line, line
    assert short_path_for_log("./models/repo") == "./models/repo"
    assert short_path_for_log("../models/repo") == "../models/repo"

    # A two-component absolute path keeps its root: rebuilding turned `/srv/cache` into
    # `srv/cache`, which reads relative.
    assert scrub_paths("cache root /srv/cache is unreadable") == (
        "cache root /srv/cache is unreadable"
    )
    assert scrub_paths("Failed /home/op/.cache/huggingface/hub/models--acme--x: EACCES") == (
        "Failed .../hub/models--acme--x: EACCES"
    )


def test_a_path_run_stops_at_the_end_of_the_path():
    line = "Scan folder rejected: /home/jane.doe/.cache/huggingface/hub (path=/home/jane.doe/x)"
    scrubbed = scrub_paths(line)
    assert scrubbed == "Scan folder rejected: .../huggingface/hub (path=.../jane.doe/x)"
    assert "(path=" in scrubbed, "the label between the two paths was swallowed"

    # The stop must not break on punctuation or a space inside a directory name.
    assert scrub_paths("Skipping /srv/client(acme)/models: denied") == (
        "Skipping .../client(acme)/models: denied"
    )
    assert scrub_paths("Skipping /srv/Program Files/models--x: denied") == (
        "Skipping .../Program Files/models--x: denied"
    )


def test_a_long_line_with_no_path_in_it_is_not_a_stall():
    import time

    line = ("word " * 20_000) + "x"
    started = time.monotonic()
    assert scrub_paths(line) == line
    assert time.monotonic() - started < 1.0


def test_a_named_tuple_in_a_payload_is_rebuilt_rather_than_raising():
    import collections

    Row = collections.namedtuple("Row", "repo_id size_bytes")
    payload = {"rows": [Row("acme/x", 5)]}
    out = redact_host_paths(payload, via_api_key = True)
    assert out["rows"][0] == Row("acme/x", 5)


# `cache_ref` is written exactly when the row HAD a path and omitted when it did not; it is the
# "is this cached" discriminator that a truthiness test on `cache_path` can no longer be.


def test_a_redacted_row_is_still_distinguishable_from_one_with_no_path_at_all():
    cached = {"repo_id": "acme/model", "cache_path": "/home/op/.cache/huggingface/hub/x"}
    uncached = {"repo_id": "acme/other", "cache_path": None}

    red_cached = host_paths.redact_host_paths(cached, via_api_key = True)
    red_uncached = host_paths.redact_host_paths(uncached, via_api_key = True)

    assert red_cached["cache_path"] == ""
    assert red_uncached["cache_path"] is None
    assert red_cached[host_paths.CACHE_REFERENCE_FIELD].startswith("ref:")
    assert (
        host_paths.CACHE_REFERENCE_FIELD not in red_uncached
    ), "a row with no path must not gain a reference, or the discriminator says nothing"


def test_the_reference_is_stable_within_the_process_and_differs_per_path():
    a = {"cache_path": "/home/op/.cache/huggingface/hub/a"}
    b = {"cache_path": "/home/op/.cache/huggingface/hub/b"}
    ref = lambda row: host_paths.redact_host_paths(row, via_api_key = True)[
        host_paths.CACHE_REFERENCE_FIELD
    ]
    assert ref(a) == ref(dict(a))
    assert ref(a) != ref(b)
    assert "/home/op" not in ref(a) and "hub" not in ref(a)


def test_a_browser_session_keeps_every_path_and_gains_nothing():
    row = {"repo_id": "acme/model", "cache_path": "/home/op/.cache/huggingface/hub/x"}
    same = host_paths.redact_host_paths(row, via_api_key = False)
    assert same is row
    assert host_paths.CACHE_REFERENCE_FIELD not in same


def test_a_local_adapter_base_model_is_redacted_but_a_repo_id_is_kept():
    """`base_model` is a path for one adapter and a repo id for the next; the `base_model_source`
    sibling decides which."""
    from hub.utils.host_paths import redact_inventory_host_paths

    payload = {
        "models": [
            {
                "id": "my-lora",
                "path": "/home/op/models/my-lora",
                "base_model": "/home/op/models/Llama-3.1-8B",
                "base_model_source": "local",
            },
            {
                "id": "other-lora",
                "path": "/home/op/models/other-lora",
                "base_model": "meta-llama/Llama-3.1-8B",
                "base_model_source": "huggingface",
            },
        ]
    }

    via_key = redact_inventory_host_paths(payload, via_api_key = True)["models"]
    assert via_key[0]["base_model"] == "", "a local base-model PATH must not reach an API key"
    assert via_key[0]["base_model_source"] == "local", "the source itself is not a path"
    assert (
        via_key[1]["base_model"] == "meta-llama/Llama-3.1-8B"
    ), "a Hub repo id is what the caller asked for and must survive redaction"

    session = redact_inventory_host_paths(payload, via_api_key = False)["models"]
    assert session[0]["base_model"] == "/home/op/models/Llama-3.1-8B"


def test_the_leak_finder_knows_a_local_base_model_is_a_path():
    from hub.utils.host_paths import response_leaks_host_path

    leaky = {
        "models": [
            {
                "id": "my-lora",
                "base_model": "/home/op/models/Llama-3.1-8B",
                "base_model_source": "local",
            },
        ]
    }
    assert response_leaks_host_path(leaky, ["/home/op"]) is not None

    clean = {
        "models": [
            {
                "id": "other-lora",
                "base_model": "meta-llama/Llama-3.1-8B",
                "base_model_source": "huggingface",
            },
        ]
    }
    assert response_leaks_host_path(clean, ["/home/op"]) is None


# /api/models is the OpenAI-compatible mirror of /api/hub, reachable with the same key, so the
# boundary has to be drawn on both routers.


def test_the_compat_scan_folder_list_is_not_disclosed(monkeypatch):
    monkeypatch.setattr(
        models_routes,
        "annotate_scan_folders",
        lambda folders: [{"id": 1, "path": f"{HOST_ROOT}/extra", "created_at": "2026-09-01"}],
    )
    monkeypatch.setattr(models_routes, "refresh_failed_scan_folders", lambda folders: None)
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"id": 1}])

    payload = _models(via_api_key = True).get("/api/models/scan-folders").json()
    assert payload["folders"], "the folder is still listed, by id"
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None
    ui = _models(via_api_key = False).get("/api/models/scan-folders").json()
    assert ui["folders"][0]["path"] == f"{HOST_ROOT}/extra"


def test_the_compat_download_progress_hides_the_cache_dir(monkeypatch):
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
        _models(via_api_key = True)
        .get("/api/models/download-progress", params = {"repo_id": "org/repo"})
        .json()
    )
    assert payload["cache_path"] == ""
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None


def test_the_compat_gguf_download_progress_hides_the_cache_dir(monkeypatch):
    async def _progress(
        repo_id,
        variant = "",
        expected_bytes = 0,
        hf_token = None,
    ):
        return {"repo_id": repo_id, "progress": 0.5, "cache_path": REPO_DIR}

    from hub.services.models import downloads

    monkeypatch.setattr(downloads, "get_gguf_download_progress_response", _progress)
    payload = (
        _models(via_api_key = True)
        .get("/api/models/gguf-download-progress", params = {"repo_id": "org/repo"})
        .json()
    )
    assert payload["cache_path"] == ""
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None


def test_the_compat_local_scan_is_not_disclosed(monkeypatch):
    async def _scan(models_root, sources):
        return []

    monkeypatch.setattr(models_routes, "_shared_compat_local_inventory_scan", _scan)
    payload = (
        _models(via_api_key = True).get("/api/models/local", params = {"models_dir": "./models"}).json()
    )
    assert payload["hf_cache_dir"] == ""
    assert payload["lmstudio_dirs"] == []
    assert payload["models_dir"] == ""


_COMPAT_INVENTORY_ROUTES = (
    "list_local_models",
    "get_scan_folders",
    "get_download_progress",
    "get_gguf_download_progress",
    "list_cached_gguf",
    "list_cached_models",
)


def test_every_compat_mirror_of_an_inventory_route_takes_the_caller_class():
    source = Path(models_routes.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in _COMPAT_INVENTORY_ROUTES:
            continue
        found[node.name] = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
    missing_routes = [name for name in _COMPAT_INVENTORY_ROUTES if name not in found]
    assert not missing_routes, f"these routes were renamed or removed: {missing_routes}"
    without = [name for name, args in found.items() if "via_api_key" not in args]
    assert not without, (
        "these compatibility routes answer inventory data without taking the caller class: "
        f"{without}"
    )


def _local_row():
    from hub.services.models.common import _local_model_info
    load_path = Path(HOST_ROOT) / "my models" / "Llama-3.2-1B"
    return _local_model_info(
        scan_path = load_path,
        load_path = load_path,
        source = "models_dir",
        model_format = "safetensors",
    )


def test_a_filesystem_backed_row_is_not_named_by_its_path(monkeypatch):
    async def _response(models_dir = "./models"):
        return LocalModelListResponse(
            models_dir = f"{HOST_ROOT}/models",
            hf_cache_dir = HOST_ROOT,
            lmstudio_dirs = [],
            ollama_dirs = [],
            hermes_dirs = [],
            models = [_local_row()],
        )

    monkeypatch.setattr(local_inventory, "list_local_models_response", _response)
    payload = _hub(via_api_key = True).get("/api/hub/local").json()
    row = payload["models"][0]
    assert row["path"] == ""
    assert row["id"].startswith("ref:"), row["id"]
    assert row["load_id"].startswith("ref:"), row["load_id"]
    assert row["inventory_id"].startswith("models_dir:safetensors:"), row["inventory_id"]
    assert HOST_ROOT not in json.dumps(payload), payload
    assert "my%20models" not in json.dumps(payload), payload
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None

    ui = _hub(via_api_key = False).get("/api/hub/local").json()
    assert ui["models"][0]["load_id"].startswith(HOST_ROOT)


def test_a_cached_row_keeps_the_repo_id_it_is_named_by(_cached_inventory):
    payload = _hub(via_api_key = True).get("/api/hub/cached-models").json()
    assert payload["cached"][0]["repo_id"] == "unsloth/Llama-3.2-1B-Instruct"
    assert not payload["cached"][0]["repo_id"].startswith("ref:")


def test_a_local_scan_failure_is_redacted_like_its_payload(monkeypatch):
    from fastapi import HTTPException

    async def _raises(models_dir = "./models"):
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to list local models: unable to open database file {HOST_ROOT}/studio.db",
        )

    monkeypatch.setattr(local_inventory, "list_local_models_response", _raises)
    detail = _hub(via_api_key = True).get("/api/hub/local").json()["detail"]
    assert HOST_ROOT not in str(detail), detail
    assert "unable to open database file" in str(detail), detail
    ui = _hub(via_api_key = False).get("/api/hub/local").json()["detail"]
    assert HOST_ROOT in str(ui), ui


def test_a_referenced_local_row_is_still_loadable(monkeypatch):
    from models.inference import LoadRequest

    async def _response(models_dir = "./models"):
        return LocalModelListResponse(
            models_dir = f"{HOST_ROOT}/models",
            hf_cache_dir = HOST_ROOT,
            lmstudio_dirs = [],
            ollama_dirs = [],
            hermes_dirs = [],
            models = [_local_row()],
        )

    monkeypatch.setattr(local_inventory, "list_local_models_response", _response)
    row = _hub(via_api_key = True).get("/api/hub/local").json()["models"][0]
    assert row["load_id"].startswith("ref:")

    loaded = LoadRequest(model_path = row["load_id"])
    assert loaded.model_path == str(Path(HOST_ROOT) / "my models" / "Llama-3.2-1B")

    # An unissued reference is left as-is, so it fails like an unknown model rather than
    # resolving to somebody else's row.
    stranger = "ref:" + "0" * 32
    assert LoadRequest(model_path = stranger).model_path == stranger
    assert LoadRequest(model_path = "unsloth/Llama-3.2-1B").model_path == "unsloth/Llama-3.2-1B"


def test_a_reference_table_that_fills_up_drops_the_oldest(monkeypatch):
    """Evicted by AGE, not by count, or a listing longer than the table breaks its own rows."""
    clock = {"now": 0.0}
    # The table is process-global; other tests' entries carry real timestamps this fake clock
    # cannot age out.
    host_paths._reference_paths.clear()
    monkeypatch.setattr(host_paths.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(host_paths, "_REFERENCE_LIMIT", 4)
    first = host_paths.cache_reference("/host/first")
    for index in range(6):
        host_paths.cache_reference(f"/host/filler-{index}")
    assert host_paths.resolve_host_path_reference(first) == "/host/first"
    clock["now"] = host_paths._REFERENCE_PIN_SECONDS + 1.0
    for index in range(6):
        host_paths.cache_reference(f"/host/later-{index}")
    assert host_paths.resolve_host_path_reference(first) is None
    newest = host_paths.cache_reference("/host/newest")
    assert host_paths.resolve_host_path_reference(newest) == "/host/newest"


def test_every_request_that_consumes_an_inventory_identity_resolves_the_handle():
    from models.inference import (
        DiffusionLoadRequest,
        LoadRequest,
        ValidateModelRequest,
        VideoLoadRequest,
    )
    from models.training import TrainingStartRequest

    reference = host_paths.cache_reference(f"{HOST_ROOT}/my models/Llama-3.2-1B")
    resolved = f"{HOST_ROOT}/my models/Llama-3.2-1B"

    for request_model in (
        LoadRequest,
        ValidateModelRequest,
        DiffusionLoadRequest,
        VideoLoadRequest,
    ):
        assert request_model(model_path = reference).model_path == resolved, request_model
        assert request_model(model_path = "unsloth/Llama-3.2-1B").model_path == (
            "unsloth/Llama-3.2-1B"
        ), request_model

    training = TrainingStartRequest(
        model_name = reference,
        training_type = "LoRA/QLoRA",
        format_type = "chat",
    )
    assert training.model_name == resolved


def test_a_cache_reference_can_delete_the_copy_it_names(monkeypatch):
    seen = {}

    async def _delete(repo_id, variant, hf_token, cache_path, only_if_orphan):
        seen["cache_path"] = cache_path
        return {"status": "deleted", "repo_id": repo_id}

    from hub.services.models import deletion

    monkeypatch.setattr(deletion, "delete_cached_model_response", _delete)
    reference = host_paths.cache_reference(REPO_DIR)
    body = {"repo_id": "unsloth/Llama-3.2-1B", "cache_path": reference}
    response = _hub(via_api_key = True).request(
        "DELETE",
        "/api/hub/delete-cached",
        json = body,
    )
    assert response.status_code == 200, response.text
    assert seen["cache_path"] == REPO_DIR

    _hub(via_api_key = False).request(
        "DELETE",
        "/api/hub/delete-cached",
        json = {"repo_id": "unsloth/Llama-3.2-1B", "cache_path": REPO_DIR},
    )
    assert seen["cache_path"] == REPO_DIR


def test_a_cache_row_pinned_to_a_snapshot_is_referenced_too(monkeypatch):
    pinned = f"{HOST_ROOT}/models--unsloth--Llama-3.2-1B-Instruct/snapshots/deadbeef"

    async def _models_response(hf_token = None):
        return {"cached": [_cached_row(load_id = pinned)], "scan_confirmed": True}

    monkeypatch.setattr(cache_inventory, "list_cached_models_response", _models_response)
    payload = _hub(via_api_key = True).get("/api/hub/cached-models").json()
    row = payload["cached"][0]
    assert row["repo_id"] == "unsloth/Llama-3.2-1B-Instruct"
    assert row["load_id"].startswith("ref:"), row["load_id"]
    assert HOST_ROOT not in json.dumps(payload), payload
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None
    assert host_paths.resolve_host_path_reference(row["load_id"]) == pinned

    ui = _hub(via_api_key = False).get("/api/hub/cached-models").json()
    assert ui["cached"][0]["load_id"] == pinned


def test_the_leak_finder_knows_a_path_valued_identity_is_a_path():
    assert response_leaks_host_path({"load_id": f"{HOST_ROOT}/snapshots/abc"}) is not None
    assert response_leaks_host_path({"load_id": "unsloth/Llama-3.2-1B-Instruct"}) is None
    assert response_leaks_host_path({"load_id": "ref:0123456789abcdef"}) is None


def test_the_compat_scan_folder_add_does_not_answer_with_the_path(monkeypatch):
    created = {"id": 7, "path": f"{HOST_ROOT}/extra", "created_at": "2026-09-01"}
    monkeypatch.setattr(
        "storage.studio_db.add_scan_folder_with_status", lambda path: (created, False)
    )

    payload = _models(via_api_key = True).post("/api/models/scan-folders", json = {"path": "."})
    assert payload.status_code == 201, payload.text
    body = payload.json()
    assert body["id"] == 7, "the folder is still identified"
    assert response_leaks_host_path(body, [HOST_ROOT]) is None

    ui = _models(via_api_key = False).post("/api/models/scan-folders", json = {"path": "."})
    assert ui.json()["path"] == f"{HOST_ROOT}/extra"


def test_the_handle_a_caller_sent_is_the_handle_it_gets_back():
    from models.inference import LoadRequest

    path = f"{HOST_ROOT}/my models/Llama-3.2-1B"
    reference = host_paths.cache_reference(path)
    assert LoadRequest(model_path = reference).model_path == path

    answer = {
        "status": "loaded",
        "model": path,
        "display_name": path,
        "inference": {"identifier": f"{path}/snapshots/abc"},
        "warnings": [f"could not read {path}/config.json"],
        "unrelated": "unsloth/Llama-3.2-1B",
    }
    restored = host_paths.restore_inventory_handles(answer)
    assert HOST_ROOT not in json.dumps(restored), restored
    assert restored["model"] == reference
    assert restored["display_name"] == reference
    assert restored["inference"]["identifier"] == f"{reference}/snapshots/abc"
    assert restored["warnings"] == [f"could not read {reference}/config.json"]
    assert restored["unrelated"] == "unsloth/Llama-3.2-1B"


def test_a_request_that_named_a_path_directly_is_answered_with_that_path():
    answer = {"model": f"{HOST_ROOT}/models/Llama-3.2-1B"}
    assert host_paths.restore_inventory_handles(answer) == answer


def test_the_load_and_validate_answers_go_through_the_restoration():
    import inspect
    import re

    from routes import inference as inference_routes

    # Whitespace-insensitive: a re-wrap by the formatting bot must not read as a removal.
    def _squeeze(text: str) -> str:
        return re.sub(r"\s+", "", text)

    def _calls(needle: str) -> bool:
        return _squeeze(needle) in _squeeze(source)

    source = inspect.getsource(inference_routes)
    assert _calls("restore_inventory_handles(task.result())")
    assert _calls("_handle_restored_http_exception(exc)")
    assert _calls("restore_inventory_handles(exc.detail)")
    assert _calls("restore_inventory_handles(ValidateModelResponse(")
    assert _calls("jsonable_encoder(restore_inventory_handles(payload))")
    assert _calls("restore_inventory_handles(redact_native_paths(str(e)))")
    assert _calls("raise _handle_restored_http_exception(http_error) from http_error")
    assert _calls("detail = restore_inventory_handles(str(e))")


def test_a_refusal_names_the_handle_the_caller_sent(monkeypatch):
    from fastapi import HTTPException
    from routes import inference as inference_routes
    from models.inference import LoadRequest

    path = f"{HOST_ROOT}/my models/Broken-1B"
    reference = host_paths.cache_reference(path)
    assert LoadRequest(model_path = reference).model_path == path

    raised = HTTPException(status_code = 400, detail = f"Invalid model identifier: {path}")
    restored = inference_routes._handle_restored_http_exception(raised)
    assert restored.status_code == 400
    assert restored.detail == f"Invalid model identifier: {reference}"
    assert HOST_ROOT not in restored.detail

    untouched = HTTPException(status_code = 409, detail = "A model is already loading")
    assert inference_routes._handle_restored_http_exception(untouched) is untouched


def test_the_reference_that_loaded_a_model_can_unload_it():
    from models.inference import UnloadRequest

    path = f"{HOST_ROOT}/my models/Llama-3.2-1B"
    reference = host_paths.cache_reference(path)
    assert UnloadRequest(model_path = reference).model_path == path
    assert UnloadRequest(model_path = "unsloth/Llama-3.2-1B").model_path == ("unsloth/Llama-3.2-1B")


def test_a_path_that_outlives_its_request_is_still_referenced():
    path = f"{HOST_ROOT}/my models/Llama-3.2-1B"
    reference = host_paths.cache_reference(path)

    status = {"loaded": True, "repo_id": path, "device": "cuda"}
    redacted = host_paths.redact_host_paths(status, via_api_key = True)
    assert redacted["repo_id"] == reference
    assert HOST_ROOT not in json.dumps(redacted), redacted
    assert host_paths.resolve_host_path_reference(redacted["repo_id"]) == path

    run = {"run_id": "abc", "model_name": path, "status": "completed"}
    assert host_paths.redact_host_paths(run, via_api_key = True)["model_name"] == reference

    hub_row = {"repo_id": "unsloth/Llama-3.2-1B", "model_name": "unsloth/Llama-3.2-1B"}
    assert host_paths.redact_host_paths(hub_row, via_api_key = True) == hub_row
    assert host_paths.redact_host_paths(status, via_api_key = False) == status


def test_a_local_row_keeps_a_repo_id_that_is_not_a_path(monkeypatch):
    row = {
        "source": "models_dir",
        "id": f"{HOST_ROOT}/models/Llama-3.2-1B",
        "load_id": f"{HOST_ROOT}/models/Llama-3.2-1B",
        "repo_id": "unsloth/Llama-3.2-1B",
    }
    redacted = host_paths.redact_inventory_host_paths(row, via_api_key = True)
    assert redacted["id"].startswith("ref:")
    assert redacted["load_id"].startswith("ref:")
    assert redacted["repo_id"] == "unsloth/Llama-3.2-1B"


def test_the_long_lived_routes_redact_what_they_persisted():
    import inspect
    from routes import inference as inference_routes
    from routes import training_history as training_routes
    from routes import video as video_routes

    for module, needle in (
        (inference_routes, "redact_host_paths(DiffusionStatusResponse("),
        (video_routes, "redact_host_paths(VideoStatusResponse("),
        (training_routes, 'TrainingRunListResponse(runs = runs, total = result["total"]),'),
    ):
        source = inspect.getsource(module)
        assert needle in source, (module.__name__, needle)
        assert "authenticated_via_api_key" in source, module.__name__


def test_a_second_load_does_not_hand_back_the_resident_path():
    import asyncio

    from models.inference import DiffusionStatusResponse
    from routes import inference as inference_routes
    from routes import video as video_routes

    resident = f"{HOST_ROOT}/my models/previous-image-model"
    resident_reference = host_paths.cache_reference(resident)

    async def _gated(request, subject, **kwargs):
        return DiffusionStatusResponse(
            **{"loaded": True, "repo_id": resident, "model_path": resident}
        )

    class _Request:
        model_path = "ref:whatever"
        base_repo = None

    original = inference_routes.load_diffusion_model_gated
    inference_routes.load_diffusion_model_gated = _gated
    try:
        answered = asyncio.run(
            inference_routes.load_diffusion_model(
                _Request(), current_subject = "api", via_api_key = True
            )
        )
    finally:
        inference_routes.load_diffusion_model_gated = original
    body = json.dumps(jsonable(answered))
    assert HOST_ROOT not in body, body
    assert resident_reference in body, body
    assert host_paths.resolve_host_path_reference(resident_reference) == resident

    inference_routes.load_diffusion_model_gated = _gated
    try:
        seen = asyncio.run(
            inference_routes.load_diffusion_model(
                _Request(), current_subject = "browser", via_api_key = False
            )
        )
    finally:
        inference_routes.load_diffusion_model_gated = original
    assert resident in json.dumps(jsonable(seen))
    del video_routes


def jsonable(payload):
    from fastapi.encoders import jsonable_encoder
    return jsonable_encoder(payload)


def test_every_route_that_answers_with_a_persisted_record_redacts():
    import inspect

    from routes import inference as inference_routes
    from routes import training_history as training_routes
    from routes import video as video_routes

    expected = {
        inference_routes: (
            "load_diffusion_model",
            "unload_diffusion_model",
            "diffusion_status",
        ),
        video_routes: ("load_video_model", "unload_video_model", "video_status"),
        training_routes: (
            "list_training_runs",
            "get_training_run_detail",
            "update_training_run",
        ),
    }
    for module, names in expected.items():
        tree = ast.parse(inspect.getsource(module))
        found = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in names:
            assert name in found, (module.__name__, name)
            body = ast.unparse(found[name])
            assert "redact_host_paths(" in body, (module.__name__, name)
            assert "authenticated_via_api_key" in body, (module.__name__, name)


def test_opening_or_renaming_a_run_does_not_hand_back_the_path():
    import asyncio

    from routes import training_history as training_routes

    path = f"{HOST_ROOT}/my models/trained-from"
    reference = host_paths.cache_reference(path)
    row = {
        "id": "run-1",
        "run_id": "run-1",
        "model_name": path,
        "dataset_name": "some/dataset",
        "started_at": "2026-01-01T00:00:00Z",
        "status": "completed",
        "config_json": json.dumps({"model_name": path}),
        "output_dir": None,
        "display_name": None,
    }

    async def _detail():
        return await training_routes.get_training_run_detail(
            "run-1", current_subject = "api", no_credential = False, via_api_key = True
        )

    saved = (
        training_routes.get_run,
        training_routes.get_run_metrics,
        training_routes.get_preview_sharing_enabled,
    )
    training_routes.get_run = lambda run_id: dict(row)
    training_routes.get_run_metrics = lambda run_id: {}
    training_routes.get_preview_sharing_enabled = lambda: False
    try:
        detail = json.dumps(jsonable(asyncio.run(_detail())))
        assert HOST_ROOT not in detail, detail
        assert reference in detail, detail

        async def _patch():
            return await training_routes.update_training_run(
                "run-1",
                training_routes.TrainingRunUpdateRequest(),
                current_subject = "api",
                no_credential = False,
                via_api_key = True,
            )

        renamed = json.dumps(jsonable(asyncio.run(_patch())))
        assert HOST_ROOT not in renamed, renamed
        assert reference in renamed, renamed
    finally:
        (
            training_routes.get_run,
            training_routes.get_run_metrics,
            training_routes.get_preview_sharing_enabled,
        ) = saved


def test_a_prepared_dataset_cache_still_counts_when_the_hub_copy_is_unusable():
    from hub.utils import hf_tokens

    import sys
    import types

    cache_state = types.ModuleType("hub.utils.hf_cache_state")
    cache_state.iter_repo_cache_dirs = lambda repo_type, repo_id: iter(["a-directory"])
    cache_state.repo_cache_has_usable_snapshot = lambda repo_type, repo_id: False
    dataset_cache = types.ModuleType("hub.utils.dataset_cache")
    dataset_cache.latest_processed_dataset_cache_path = lambda repo_id: "/prepared/here"

    saved = {
        name: sys.modules.get(name)
        for name in ("hub.utils.hf_cache_state", "hub.utils.dataset_cache")
    }
    sys.modules["hub.utils.hf_cache_state"] = cache_state
    sys.modules["hub.utils.dataset_cache"] = dataset_cache
    try:
        assert hf_tokens._repo_present_on_disk("owner/ds", "dataset") is True
        assert hf_tokens._repo_present_on_disk("owner/model", "model") is False
        dataset_cache.latest_processed_dataset_cache_path = lambda repo_id: None
        assert hf_tokens._repo_present_on_disk("owner/ds", "dataset") is False

        def _raise(repo_type, repo_id):
            raise OSError("unreadable cache root")

        cache_state.iter_repo_cache_dirs = _raise
        dataset_cache.latest_processed_dataset_cache_path = lambda repo_id: "/prepared/here"
        assert hf_tokens._repo_present_on_disk("owner/ds", "dataset") is True
        assert hf_tokens._repo_present_on_disk("owner/model", "model") is False
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_every_route_that_consumes_an_inventory_identity_resolves_the_handle():
    import inspect

    from routes import models as model_routes
    for name in ("get_model_config", "scan_model_remote_code"):
        body = inspect.getsource(getattr(model_routes, name))
        assert "resolve_inventory_handle(" in body, name
        assert "restore_inventory_handles(" in body, name


def test_the_metadata_route_resolves_and_answers_with_the_handle(monkeypatch):
    path = f"{HOST_ROOT}/my models/Local-Model"
    reference = host_paths.cache_reference(path)

    from models.inference import resolve_inventory_handle

    token = host_paths._request_handles.set(None)
    try:
        assert resolve_inventory_handle(reference) == path
        answered = {"model": path, "detail": f"could not read {path}/config.json"}
        restored = host_paths.restore_inventory_handles(answered)
        assert restored["model"] == reference
        assert HOST_ROOT not in json.dumps(restored), restored
        assert resolve_inventory_handle("ref:not-one-of-ours") == "ref:not-one-of-ours"
        assert resolve_inventory_handle("unsloth/Llama-3.2-1B") == "unsloth/Llama-3.2-1B"
    finally:
        host_paths._request_handles.reset(token)


def test_a_directory_name_with_punctuation_is_removed_whole():
    from hub.utils.host_paths import redact_paths_in_text

    for message, expected in (
        ("Failed /srv/client(acme)/models", "Failed <path>"),
        ("cannot read /home/o'connor/models", "cannot read <path>"),
        (r"C:\cache\Models (private)", "<path>"),
        # Trailing prose goes with it, since a directory name may contain spaces.
        ("open /mnt/data[1]/models denied", "open <path>"),
    ):
        cleaned = redact_paths_in_text(message)
        assert cleaned == expected, (message, cleaned)
        for fragment in ("acme", "connor", "private", "data"):
            assert fragment not in cleaned, (message, cleaned)

    assert redact_paths_in_text("Skipping /a/b: Permission denied") == (
        "Skipping <path>: Permission denied"
    )
    assert redact_paths_in_text('opened "/a/b" already') == 'opened "<path>" already'
    assert redact_paths_in_text("tried /a/b, /c/d") == "tried <path>, <path>"
    for kept in ("3/4 of the shards", "and/or the projector", "no paths here"):
        assert redact_paths_in_text(kept) == kept, kept


def test_a_comma_in_a_directory_name_does_not_leave_the_rest_of_the_path():
    from hub.utils.host_paths import redact_paths_in_text

    for message, expected in (
        ("load /home/operator/Acme, Inc/private/model.bin failed", "load <path>"),
        ("read /srv/Acme, Inc, Ltd/private/model.bin", "read <path>"),
        (r"C:\Users\operator\Acme, Inc\private\model.bin", "<path>"),
        ("read /srv/models; Acme; Inc/private/model.bin", "read <path>"),
    ):
        cleaned = redact_paths_in_text(message)
        assert cleaned == expected, (message, cleaned)
        for leaked in ("Inc", "private", "model.bin", "operator", "Acme", "Ltd"):
            assert leaked not in cleaned, (message, cleaned)

    assert redact_paths_in_text("Skipping /a/b: Permission denied") == (
        "Skipping <path>: Permission denied"
    )
    assert redact_paths_in_text("tried /a/b, /c/d") == "tried <path>, <path>"
    assert redact_paths_in_text('opened "/a/b" already') == 'opened "<path>" already'
    assert redact_paths_in_text("Skipping /a/b: denied, see /etc/fstab") == (
        "Skipping <path>: denied, see <path>"
    )


def test_the_redacted_tail_pass_stays_linear():
    import time

    from hub.utils.host_paths import redact_paths_in_text

    text = "/srv/models" + ", filler" * 4000
    started = time.monotonic()
    redact_paths_in_text(text)
    assert time.monotonic() - started < 1.0


def test_every_preflight_schema_resolves_an_inventory_handle():
    from models.inference import EstimateMemoryRequest, TransformersUpgradeCheckRequest

    path = f"{HOST_ROOT}/my models/Local-Model"
    reference = host_paths.cache_reference(path)

    assert EstimateMemoryRequest(model_path = reference).model_path == path
    checked = TransformersUpgradeCheckRequest(
        model_name = reference,
        model_local_path = reference,
        model_snapshot_path = reference,
        model_snapshot_repo_id = reference,
    )
    assert checked.model_name == path
    assert checked.model_local_path == path
    assert checked.model_snapshot_path == path
    assert checked.model_snapshot_repo_id == path
    assert EstimateMemoryRequest(model_path = "unsloth/Llama-3.2-1B").model_path == (
        "unsloth/Llama-3.2-1B"
    )
    assert EstimateMemoryRequest(model_path = "ref:nope").model_path == "ref:nope"


def test_a_training_run_does_not_carry_the_output_layout():
    row = {
        "id": "run-1",
        "model_name": f"{HOST_ROOT}/my models/base",
        "output_dir": f"{HOST_ROOT}/outputs/run-1",
        "model_local_path": f"{HOST_ROOT}/cache/base",
        "model_snapshot_path": f"{HOST_ROOT}/cache/base/snapshots/abc",
        "dataset_local_path": f"{HOST_ROOT}/data/set",
        "dataset_snapshot_path": f"{HOST_ROOT}/data/set/snapshots/abc",
        "checkpoint_path": f"{HOST_ROOT}/outputs/run-1/checkpoint-10",
        "resume_from_checkpoint": f"{HOST_ROOT}/outputs/run-1/checkpoint-10",
        "output_dirs": [f"{HOST_ROOT}/outputs/run-1", f"{HOST_ROOT}/outputs/run-2"],
        "dataset_name": "acme/dataset",
    }
    redacted = host_paths.redact_host_paths(row, via_api_key = True)
    assert HOST_ROOT not in json.dumps(redacted), redacted
    assert redacted["output_dirs"] == []
    assert host_paths.resolve_host_path_reference(redacted["model_name"]) == row["model_name"]
    # Resume fields are referenced, not blanked, or `can_resume` is true beside no identifier.
    for field in ("output_dir", "checkpoint_path", "resume_from_checkpoint"):
        assert host_paths.resolve_host_path_reference(redacted[field]) == row[field], field
    assert redacted["dataset_name"] == "acme/dataset"
    assert host_paths.redact_host_paths(row, via_api_key = False) == row


def test_the_deferred_five_hundred_restores_the_handle_too():
    import inspect

    from routes import inference as inference_routes

    body = inspect.getsource(inference_routes._tunnel_safe_json)
    generic = body.index("failed after the response was committed")
    # Stop at the success branch, or its restoration would satisfy this assertion.
    tail = body[generic : body.index("else:", generic)]
    assert "_deferred_error_body(" in tail, tail
    assert "restore_inventory_handles(" in tail, tail

    path = f"{HOST_ROOT}/my models/Local-Model"
    reference = host_paths.cache_reference(path)
    token = host_paths._request_handles.set({path: reference})
    try:
        restored = host_paths.restore_inventory_handles(
            f"OSError: [Errno 13] Permission denied: '{path}/model.safetensors'"
        )
        assert HOST_ROOT not in restored, restored
        assert reference in restored, restored
    finally:
        host_paths._request_handles.reset(token)


def test_the_persisted_training_request_keys_are_the_ones_redacted():
    from models.training import TrainingStartRequest

    declared = set(TrainingStartRequest.model_fields)
    covered = (
        host_paths.HOST_PATH_SCALAR_FIELDS
        | host_paths.HOST_PATH_LIST_FIELDS
        | host_paths.HOST_PATH_HANDLE_LIST_FIELDS
    )
    for field in ("local_datasets", "local_eval_datasets", "model_local_path"):
        assert field in declared, field
        assert field in covered, field

    config = {
        "local_datasets": [f"{HOST_ROOT}/data/train.jsonl"],
        "local_eval_datasets": [f"{HOST_ROOT}/data/eval.jsonl"],
        "tensorboard_dir": f"{HOST_ROOT}/outputs/run-1/runs",
        "learning_rate": 0.0002,
    }
    redacted = host_paths.redact_host_paths({"config": config}, via_api_key = True)
    assert HOST_ROOT not in json.dumps(redacted), redacted
    assert redacted["config"]["learning_rate"] == 0.0002


def test_a_persisted_failure_message_keeps_its_reason_and_loses_the_path():
    message = f"FileNotFoundError: no such file: {HOST_ROOT}/data/train.jsonl"
    redacted = host_paths.redact_host_paths(
        {"error_message": message, "status": "error"}, via_api_key = True
    )
    assert HOST_ROOT not in redacted["error_message"], redacted
    assert "FileNotFoundError" in redacted["error_message"], redacted
    assert redacted["status"] == "error"
    assert (
        host_paths.redact_host_paths({"error_message": message}, via_api_key = False)["error_message"]
        == message
    )


def test_the_chat_status_does_not_hand_back_the_path_the_load_resolved(monkeypatch):
    import asyncio

    from models.inference import InferenceStatusResponse
    from routes import inference as inference_routes

    async def _payload(current_subject: str):
        return InferenceStatusResponse(
            active_model = REPO_DIR,
            model_identifier = REPO_DIR,
            is_gguf = True,
            is_local_model = True,
            loaded = [REPO_DIR],
        )

    monkeypatch.setattr(inference_routes, "get_status", _payload)
    answered = asyncio.run(
        inference_routes.inference_status(current_subject = "alice", via_api_key = True)
    )
    body = json.dumps(json.loads(InferenceStatusResponse(**dict(answered)).model_dump_json()))
    assert HOST_ROOT not in body, body
    assert REPO_DIR not in body, body

    ui = asyncio.run(inference_routes.inference_status(current_subject = "alice", via_api_key = False))
    assert ui.model_identifier == REPO_DIR


def test_a_lora_base_model_path_is_referenced_not_returned():
    base = f"{HOST_ROOT}/my models/Llama-3.2-1B"
    details = {"id": "ref:whatever", "is_lora": True, "base_model": base}
    redacted = host_paths.redact_host_paths(details, via_api_key = True)
    assert HOST_ROOT not in json.dumps(redacted), redacted
    # Referenced, not blanked: nothing else in this answer names the base.
    assert redacted["base_model"] == host_paths.cache_reference(base)
    assert (
        host_paths.redact_host_paths({"base_model": "unsloth/Llama-3.2-1B"}, via_api_key = True)[
            "base_model"
        ]
        == "unsloth/Llama-3.2-1B"
    )
    assert host_paths.redact_host_paths(details, via_api_key = False)["base_model"] == base


def test_the_model_details_route_takes_the_caller_class():
    import inspect

    source = inspect.getsource(models_routes.get_model_config)
    assert "via_api_key: bool = Depends(authenticated_via_api_key)" in source
    assert "redact_host_paths(" in source
    assert "restore_inventory_handles(await asyncio.to_thread(_resolve, model_name))" in source


def test_the_upgrade_check_answers_with_the_handle_it_was_sent():
    import inspect

    from routes import inference as inference_routes

    source = inspect.getsource(inference_routes)
    assert "model_name = restore_inventory_handles(model_name)," in source


def test_the_compat_delete_route_resolves_a_cache_reference_too(monkeypatch):
    seen = {}

    async def _delete(repo_id, variant, hf_token, cache_path):
        seen["cache_path"] = cache_path
        return {"status": "deleted", "repo_id": repo_id}

    from hub.services.models import account_access, deletion

    monkeypatch.setattr(deletion, "delete_cached_model_response", _delete)
    monkeypatch.setattr(account_access, "require_installation_owner", lambda: None)
    reference = host_paths.cache_reference(REPO_DIR)
    assert reference != REPO_DIR
    response = _models(via_api_key = True).request(
        "DELETE",
        "/api/models/delete-cached",
        json = {"repo_id": "unsloth/Llama-3.2-1B", "cache_path": reference},
    )
    assert response.status_code == 200, response.text
    assert seen["cache_path"] == REPO_DIR

    seen.clear()
    response = _models(via_api_key = False).request(
        "DELETE",
        "/api/models/delete-cached",
        json = {"repo_id": "unsloth/Llama-3.2-1B", "cache_path": REPO_DIR},
    )
    assert response.status_code == 200, response.text
    assert seen["cache_path"] == REPO_DIR

    # An absent cache_path must reach the service as None, not as some resolved root.
    seen.clear()
    response = _models(via_api_key = True).request(
        "DELETE",
        "/api/models/delete-cached",
        json = {"repo_id": "unsloth/Llama-3.2-1B"},
    )
    assert response.status_code == 200, response.text
    assert seen["cache_path"] is None


def test_a_resumable_run_can_still_be_resumed_by_an_api_key_caller():
    from models.training import DiffusionTrainingStartRequest, TrainingStartRequest

    output_dir = f"{HOST_ROOT}/outputs/run-1"
    detail = host_paths.redact_host_paths(
        {"id": "run-1", "can_resume": True, "output_dir": output_dir, "checkpoint_path": None},
        via_api_key = True,
    )
    assert detail["can_resume"] is True
    handle = detail["output_dir"]
    assert handle.startswith("ref:"), detail
    assert HOST_ROOT not in json.dumps(detail), detail

    started = TrainingStartRequest(
        model_name = "unsloth/Llama-3.2-1B",
        training_type = "LoRA/QLoRA",
        format_type = "chat",
        resume_from_checkpoint = handle,
    )
    assert started.resume_from_checkpoint == output_dir

    diffusion = DiffusionTrainingStartRequest(
        base_model = "unsloth/FLUX.1-dev",
        data_dir = "/data/images",
        output_dir = handle,
        resume_from_checkpoint = handle,
    )
    assert diffusion.resume_from_checkpoint == output_dir
    assert diffusion.output_dir == output_dir

    assert (
        TrainingStartRequest(
            model_name = "unsloth/Llama-3.2-1B",
            training_type = "LoRA/QLoRA",
            format_type = "chat",
        ).resume_from_checkpoint
        is None
    )
    assert (
        TrainingStartRequest(
            model_name = "unsloth/Llama-3.2-1B",
            training_type = "LoRA/QLoRA",
            format_type = "chat",
            resume_from_checkpoint = "ref:nope",
        ).resume_from_checkpoint
        == "ref:nope"
    )

    assert host_paths.redact_host_paths({"output_dir": output_dir}, via_api_key = False) == {
        "output_dir": output_dir
    }


def test_a_local_diffusion_base_is_still_trainable_by_an_api_key_caller():
    from models.training import DiffusionTrainingStartRequest

    base = f"{HOST_ROOT}/my models/SDXL-Local"
    handle = host_paths.cache_reference(base)
    assert handle.startswith("ref:") and HOST_ROOT not in handle

    started = DiffusionTrainingStartRequest(
        base_model = handle,
        data_dir = "/data/images",
        output_dir = f"{HOST_ROOT}/outputs/run-2",
    )
    assert started.base_model == base

    assert (
        DiffusionTrainingStartRequest(
            base_model = "unsloth/FLUX.1-dev",
            data_dir = "/data/images",
            output_dir = "/out",
        ).base_model
        == "unsloth/FLUX.1-dev"
    )
    assert (
        DiffusionTrainingStartRequest(
            base_model = "ref:nope",
            data_dir = "/data/images",
            output_dir = "/out",
        ).base_model
        == "ref:nope"
    )


def test_a_listing_bigger_than_the_reference_table_still_resolves_its_own_rows():
    from hub.utils import host_paths

    host_paths._reference_paths.clear()
    issued = [
        host_paths.cache_reference(f"/srv/models/row-{index}.gguf")
        for index in range(host_paths._REFERENCE_LIMIT + 2000)
    ]
    unresolved = [
        reference
        for reference in issued
        if host_paths.resolve_host_path_reference(reference) is None
    ]
    assert unresolved == [], len(unresolved)


def test_the_table_is_still_bounded_at_its_ceiling(monkeypatch):
    from hub.utils import host_paths

    host_paths._reference_paths.clear()
    monkeypatch.setattr(host_paths, "_REFERENCE_CEILING", host_paths._REFERENCE_LIMIT + 10)
    for index in range(host_paths._REFERENCE_LIMIT + 500):
        host_paths.cache_reference(f"/srv/ceiling/row-{index}.gguf")
    assert len(host_paths._reference_paths) <= host_paths._REFERENCE_CEILING + 1


def test_a_runs_persisted_dataset_name_is_redacted_like_the_list_it_came_from():
    from hub.utils.host_paths import redact_host_paths

    row = {
        "id": "run-1",
        "dataset_name": "/home/operator/datasets/customer-transcripts.jsonl",
        "local_datasets": ["/home/operator/datasets/customer-transcripts.jsonl"],
        "model_name": "unsloth/Llama-3.2-1B",
    }
    redacted = redact_host_paths(row, via_api_key = True)
    assert "/home/operator" not in str(redacted), redacted
    assert redacted["dataset_name"].startswith("ref:"), redacted
    hub_row = {"id": "run-2", "dataset_name": "unsloth/Radiology-mini"}
    assert redact_host_paths(hub_row, via_api_key = True)["dataset_name"] == (
        "unsloth/Radiology-mini"
    )
    assert redact_host_paths(row, via_api_key = False) is row


def test_a_failed_downloads_error_does_not_carry_the_cache_path(monkeypatch):
    from hub.schemas.downloads import DownloadJobStatus
    from hub.services.models import downloads as downloads_service

    async def _status(_repo_id, _variant = ""):
        return DownloadJobStatus(
            repo_id = "acme/model",
            state = "error",
            error = (
                "OSError: [Errno 28] No space left on device: "
                f"'{HOST_ROOT}/hub/models--acme--model/blobs/deadbeef.incomplete'"
            ),
        )

    monkeypatch.setattr(downloads_service, "get_download_status_response", _status)
    answered = _hub(True).get("/api/hub/download-status", params = {"repo_id": "acme/model"})
    assert answered.status_code == 200, answered.text
    body = answered.json()
    assert HOST_ROOT not in answered.text, body
    assert "No space left on device" in body["error"], body

    owner = _hub(False).get("/api/hub/download-status", params = {"repo_id": "acme/model"})
    assert HOST_ROOT in owner.text


def test_an_api_key_caller_can_still_resume_a_run_trained_from_local_data():
    from hub.utils.host_paths import redact_host_paths
    from models.training import TrainingStartRequest

    detail = {
        "id": "run-9",
        "output_dir": f"{HOST_ROOT}/outputs/run-9",
        "local_datasets": [f"{HOST_ROOT}/datasets/train.jsonl"],
        "local_eval_datasets": [f"{HOST_ROOT}/datasets/eval.jsonl", "acme/hub-eval"],
    }
    redacted = redact_host_paths(detail, via_api_key = True)
    assert HOST_ROOT not in str(redacted), redacted
    assert all(entry.startswith("ref:") for entry in redacted["local_datasets"])
    assert redacted["local_eval_datasets"][1] == "acme/hub-eval"

    replayed = TrainingStartRequest(
        model_name = "unsloth/Llama-3.2-1B",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        local_datasets = redacted["local_datasets"],
        local_eval_datasets = redacted["local_eval_datasets"],
        resume_from_checkpoint = redacted["output_dir"],
    )
    assert replayed.local_datasets == [f"{HOST_ROOT}/datasets/train.jsonl"]
    assert replayed.local_eval_datasets == [f"{HOST_ROOT}/datasets/eval.jsonl", "acme/hub-eval"]
    assert replayed.resume_from_checkpoint == f"{HOST_ROOT}/outputs/run-9"


def test_a_dataset_handle_that_was_never_issued_resolves_to_nothing():
    from models.training import TrainingStartRequest

    forged = "ref:" + "0" * 32
    replayed = TrainingStartRequest(
        model_name = "unsloth/Llama-3.2-1B",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        local_datasets = [forged],
    )
    assert replayed.local_datasets == [forged], "a forged handle resolved to a host path"


def test_a_caller_named_path_is_echoed_not_referenced():
    """Referencing an identifier the caller CHOSE turns the route into an online oracle; the
    per-process HMAC key defeats an offline attack and does nothing about this."""
    victim = f"{HOST_ROOT}/models--acme--private/snapshots/deadbeef"
    held = cache_reference(victim)

    def answer(caller_supplied):
        return redact_host_paths(
            {"id": caller_supplied, "model_name": caller_supplied, "config": {}},
            via_api_key = True,
            echo = (caller_supplied,),
        )

    for guess in (f"{HOST_ROOT}/models--acme--private/snapshots/beef", victim):
        out = answer(guess)
        assert out["id"] == guess, "the caller's own identifier came back altered"
        assert out["id"] != held, "the reference function answered for a caller-chosen string"

    assert redact_host_paths({"id": victim}, via_api_key = True)["id"] == held

    out = redact_host_paths(
        {"id": "acme/lora", "base_model": victim}, via_api_key = True, echo = ("acme/lora",)
    )
    assert out["base_model"] == held
    assert response_leaks_host_path(out, [HOST_ROOT]) is None


def test_the_config_route_passes_the_caller_identifier_as_echo():
    import inspect
    source = inspect.getsource(models_routes.get_model_config)
    assert "echo = (model_name,)" in source
