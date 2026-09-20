# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An API key gets the local inventory listing but not the host paths; a browser session still
gets them. Assertions are on the SERIALISED body: a unit test on the helper alone would pass
with the helper wired to nothing.
"""

import ast
import inspect
import json
import re
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
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
    redact_paths_in_text,
    response_leaks_host_path,
    scrub_paths,
    short_path_for_log,
)
from routes import models as models_routes

# Empty: no field is exempt from redaction any more, including ``load_id``.
LOAD_HANDLE: "tuple[str, ...]" = ()

# A root that cannot exist by accident, so finding it in a body is proof and not a coincidence.
HOST_ROOT = "/home/operator-7f3c/.cache/huggingface/hub"
# A constructed path comes back with backslashes on Windows, so a POSIX literal is a leak
# check that can never fire.
HOST_ROOT_NATIVE = str(Path(HOST_ROOT))
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


def jsonable(payload):
    from fastapi.encoders import jsonable_encoder
    return jsonable_encoder(payload)


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

    # BOUNDARY. A browser session keeps every path and gains nothing beside it.
    row = {"repo_id": "acme/model", "cache_path": "/home/op/.cache/huggingface/hub/x"}
    assert redact_host_paths(row, via_api_key = False) is row
    assert host_paths.CACHE_REFERENCE_FIELD not in row


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


# `cache_ref` is the "is this cached" discriminator `cache_path` truthiness no longer can be.


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


def test_the_ambiguous_path_field_is_only_redacted_where_it_is_a_path():
    payload = {"path": REPO_DIR}
    assert redact_host_paths(payload, via_api_key = True)["path"] == REPO_DIR
    assert redact_inventory_host_paths(payload, via_api_key = True)["path"] == ""


def test_scan_root_lists_are_emptied_not_referenced():
    payload = {"lmstudio_dirs": [f"{HOST_ROOT}/lm"], "exact_paths": [REPO_DIR]}
    out = redact_host_paths(payload, via_api_key = True)
    assert out["lmstudio_dirs"] == []
    assert out["exact_paths"] == []


# `base_model` is a host path for one source and a repo id for the next; the sibling decides.
def _adapter(identifier, base_model, source) -> dict:
    return {
        "models": [
            {"id": identifier, "base_model": base_model, "base_model_source": source}
        ]
    }


_LOCAL_ADAPTER = _adapter("my-lora", "/home/op/models/Llama-3.1-8B", "local")
_HUB_ADAPTER = _adapter("other-lora", "meta-llama/Llama-3.1-8B", "huggingface")

# Every route test below trusts the leak detector, so it is pinned on both answers.
_DEFAULT_ROOTS = object()


@pytest.mark.parametrize(
    ("payload", "roots", "leaks"),
    [
        ({"cached": [_cached_row()]}, [HOST_ROOT], True),
        (redact_host_paths({"cached": [_cached_row()]}, via_api_key = True), [HOST_ROOT], False),
        ({"detail": f"could not read {REPO_DIR}"}, [HOST_ROOT], True),
        (_LOCAL_ADAPTER, ["/home/op"], True),
        (_HUB_ADAPTER, ["/home/op"], False),
        ({"load_id": f"{HOST_ROOT}/snapshots/abc"}, _DEFAULT_ROOTS, True),
        ({"load_id": "unsloth/Llama-3.2-1B-Instruct"}, _DEFAULT_ROOTS, False),
        ({"load_id": "ref:0123456789abcdef"}, _DEFAULT_ROOTS, False),
    ],
)
def test_the_leak_detector_finds_what_it_is_for(payload, roots, leaks):
    found = (
        response_leaks_host_path(payload)
        if roots is _DEFAULT_ROOTS
        else response_leaks_host_path(payload, roots)
    )
    assert (found is not None) is leaks, found


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
        (OSError(f"cannot open {REPO_DIR}/blobs/abc"), "cannot open .../blobs/abc"),
        (None, ""),
        (
            r"Skipping \\fileserver\models\hub\models--acme--x: denied",
            "Skipping .../hub/models--acme--x: denied",
        ),
        (r"open \\srv\share\acme\config.json failed", "open .../acme/config.json failed"),
        ("ratio 3/4", "ratio 3/4"),
        ("https://huggingface.co/api/models/a/b", "https://huggingface.co/api/models/a/b"),
        ("Skipping ./models/repo: denied", "Skipping ./models/repo: denied"),
        ("Skipping ../models/repo: denied", "Skipping ../models/repo: denied"),
        ("Skipping ../../a/b/c: denied", "Skipping ../../a/b/c: denied"),
        ("Skipping models/team/repo: denied", "Skipping models/team/repo: denied"),
        # A two-component path keeps its root: rebuilding turned `/srv/cache` into `srv/cache`.
        ("cache root /srv/cache is unreadable", "cache root /srv/cache is unreadable"),
        (
            "Failed /home/op/.cache/huggingface/hub/models--acme--x: EACCES",
            "Failed .../hub/models--acme--x: EACCES",
        ),
        # The label between two paths is not swallowed, nor does a space break the run.
        (
            "Scan folder rejected: /home/jane.doe/.cache/huggingface/hub (path=/home/jane.doe/x)",
            "Scan folder rejected: .../huggingface/hub (path=.../jane.doe/x)",
        ),
        ("Skipping /srv/client(acme)/models: denied", "Skipping .../client(acme)/models: denied"),
        (
            "Skipping /srv/Program Files/models--x: denied",
            "Skipping .../Program Files/models--x: denied",
        ),
    ],
)
def test_log_messages_lose_the_layout_and_keep_the_meaning(message, expected):
    assert scrub_paths(message) == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        (Path(REPO_DIR), ".../hub/models--unsloth--Llama-3.2-1B-Instruct"),
        ("./models/repo", "./models/repo"),
        ("../models/repo", "../models/repo"),
    ],
)
def test_short_path_for_log_keeps_the_tail_and_relative_paths_as_written(path, expected):
    assert short_path_for_log(path) == expected


@pytest.mark.parametrize(
    "scrub, text",
    [
        (scrub_paths, ("word " * 20_000) + "x"),
        (redact_paths_in_text, "/srv/models" + ", filler" * 4000),
    ],
    ids = ("a-long-line-with-no-path", "the-redacted-tail-pass"),
)
def test_the_text_passes_stay_linear(scrub, text):
    started = time.monotonic()
    scrub(text)
    assert time.monotonic() - started < 1.0


# The directory name goes WHOLE: stopping the run at punctuation inside it publishes the rest
# of the layout. `_PATH_COMPONENT` ends at `=`, so the tail rule treats `=` like `:`, `;`, `,`.


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Failed /srv/client(acme)/models", "Failed <path>"),
        ("cannot read /home/o'connor/models", "cannot read <path>"),
        (r"C:\cache\Models (private)", "<path>"),
        ("open /mnt/data[1]/models denied", "open <path>"),
        ("load /home/operator/Acme, Inc/private/model.bin failed", "load <path>"),
        ("read /srv/Acme, Inc, Ltd/private/model.bin", "read <path>"),
        (r"C:\Users\operator\Acme, Inc\private\model.bin", "<path>"),
        ("read /srv/models; Acme; Inc/private/model.bin", "read <path>"),
        (f"Error at {HOST_ROOT}/foo=bar/model/config.json", "Error at <path>"),
        (f"failed: {HOST_ROOT}/models--a--b=v2/snapshots/dead/config.json", "failed: <path>"),
        ("Skipping /a/b: Permission denied", "Skipping <path>: Permission denied"),
        ('opened "/a/b" already', 'opened "<path>" already'),
        ("tried /a/b, /c/d", "tried <path>, <path>"),
        ("Skipping /a/b: denied, see /etc/fstab", "Skipping <path>: denied, see <path>"),
        # BOUNDARY: prose, an `=` outside a path, and a URL are not host paths.
        ("3/4 of the shards", "3/4 of the shards"),
        ("and/or the projector", "and/or the projector"),
        ("no paths here", "no paths here"),
        ("ratio 3/4 and rate=0.5 are fine", "ratio 3/4 and rate=0.5 are fine"),
        (
            "see https://huggingface.co/acme/model for details",
            "see https://huggingface.co/acme/model for details",
        ),
    ],
)
def test_a_directory_name_is_removed_whole_and_prose_is_left_alone(message, expected):
    cleaned = redact_paths_in_text(message)
    assert cleaned == expected, (message, cleaned)


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


# /api/models is the OpenAI-compatible mirror of /api/hub, reachable with the same key.


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
        assert not row["repo_id"].startswith("ref:")


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


# A raised detail is walked like a payload: the message survives, the layout does not.


@pytest.mark.parametrize(
    ("raised", "secret", "kept", "message", "ui_sees_it"),
    [
        (
            f"Failed to create models folder: {HOST_ROOT}/hub: Permission denied",
            HOST_ROOT,
            ("Permission denied", "Failed to create models folder"),
            None,
            True,
        ),
        *[
            (f"Models folder path is not a directory: {root}", root, ("not a directory",), None, False)
            for root in ("/tmp", "/cache", "C:\\cache")
        ],
        (
            {"message": "bad folder", "path": f"{HOST_ROOT}/hub"},
            HOST_ROOT,
            (),
            "bad folder",
            False,
        ),
    ],
)
def test_the_models_folder_error_is_not_disclosed_either(
    monkeypatch, raised, secret, kept, message, ui_sees_it
):
    def _raises():
        raise HTTPException(status_code = 500, detail = raised)

    monkeypatch.setattr(local_inventory, "get_models_folder_response", _raises)

    response = _hub(via_api_key = True).get("/api/hub/models-folder")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert secret not in str(detail), detail
    for fragment in kept:
        assert fragment in str(detail), detail
    if message is not None:
        assert detail["message"] == message, detail

    if ui_sees_it:
        session = _hub(via_api_key = False).get("/api/hub/models-folder")
        assert session.status_code == 500
        assert HOST_ROOT in session.json()["detail"]


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


@pytest.mark.parametrize(
    ("client", "route", "reader", "answers_with_a_reference"),
    [
        (_hub, "/api/hub/download-progress", "get_download_progress_response", True),
        (_models, "/api/models/download-progress", "get_download_progress_response", False),
        (
            _models,
            "/api/models/gguf-download-progress",
            "get_gguf_download_progress_response",
            False,
        ),
    ],
    ids = ("hub", "compat", "compat-gguf"),
)
def test_download_progress_hides_the_cache_dir_it_measured(
    monkeypatch, client, route, reader, answers_with_a_reference
):
    """The measurement is taken FROM the cache directory, so both compat aliases must drop it
    on the way out too."""

    async def _progress(repo_id, **_kwargs):
        return {
            "repo_id": repo_id,
            "downloaded_bytes": 10,
            "expected_bytes": 100,
            "progress": 0.1,
            "complete": False,
            "cache_path": REPO_DIR,
        }

    from hub.services.models import downloads

    monkeypatch.setattr(downloads, reader, _progress)
    payload = client(via_api_key = True).get(route, params = {"repo_id": "org/repo"}).json()
    assert payload["cache_path"] == ""
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None
    if answers_with_a_reference:
        assert payload["cache_ref"].startswith("ref:")


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


def _route_arguments(module, *, router_decorated = False) -> dict:
    """With `router_decorated`, only top-level functions carrying a `@router.<method>(...)`."""
    tree = ast.parse(Path(module.__file__).read_text(encoding = "utf-8"))
    found = {}
    for node in tree.body if router_decorated else ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if router_decorated and not any(
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and isinstance(dec.func.value, ast.Name)
            and dec.func.value.id == "router"
            for dec in node.decorator_list
        ):
            continue
        found[node.name] = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
    return found


def test_every_inventory_route_that_could_answer_a_path_takes_the_caller_class():
    routes = _route_arguments(inventory_routes, router_decorated = True)
    missing = [
        name
        for name, args in routes.items()
        if "via_api_key" not in args and name not in _ROUTES_WITHOUT_HOST_PATHS
    ]
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


def test_a_local_adapter_base_model_is_redacted_but_a_repo_id_is_kept():
    payload = {
        "models": [
            {**row, "path": f"/home/op/models/{row['id']}"}
            for row in _LOCAL_ADAPTER["models"] + _HUB_ADAPTER["models"]
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
    found = _route_arguments(models_routes)
    missing_routes = [name for name in _COMPAT_INVENTORY_ROUTES if name not in found]
    assert not missing_routes, f"these routes were renamed or removed: {missing_routes}"
    without = [
        name
        for name in _COMPAT_INVENTORY_ROUTES
        if "via_api_key" not in found[name]
    ]
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


@pytest.fixture
def _local_inventory(monkeypatch):
    async def _response(models_dir = "./models"):
        return LocalModelListResponse(
            models_dir = f"{HOST_ROOT}/models",
            hf_cache_dir = HOST_ROOT,
            lmstudio_dirs = [f"{HOST_ROOT}/lmstudio"],
            ollama_dirs = [],
            hermes_dirs = [],
            models = [_local_row()],
        )

    monkeypatch.setattr(local_inventory, "list_local_models_response", _response)


def test_a_filesystem_backed_row_is_not_named_by_its_path(_local_inventory):
    from models.inference import LoadRequest

    payload = _hub(via_api_key = True).get("/api/hub/local").json()
    row = payload["models"][0]
    assert row["path"] == ""
    assert row["id"].startswith("ref:"), row["id"]
    assert row["load_id"].startswith("ref:"), row["load_id"]
    assert row["inventory_id"].startswith("models_dir:safetensors:"), row["inventory_id"]
    assert payload["lmstudio_dirs"] == []
    assert HOST_ROOT not in json.dumps(payload), payload
    # Both spellings, or the leak check is vacuous wherever the row carries the other one.
    assert HOST_ROOT_NATIVE not in json.dumps(payload), payload
    assert "my%20models" not in json.dumps(payload), payload
    assert response_leaks_host_path(payload, [HOST_ROOT, HOST_ROOT_NATIVE]) is None

    assert LoadRequest(model_path = row["load_id"]).model_path == str(
        Path(HOST_ROOT) / "my models" / "Llama-3.2-1B"
    )

    # BOUNDARY.
    ui = _hub(via_api_key = False).get("/api/hub/local").json()
    assert ui["models"][0]["load_id"].startswith(HOST_ROOT_NATIVE)
    assert ui["hf_cache_dir"] == HOST_ROOT


def test_a_local_scan_failure_is_redacted_like_its_payload(monkeypatch):
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


def test_a_reference_table_that_fills_up_drops_the_oldest(monkeypatch):
    """Evicted by AGE, not by count, or a listing longer than the table breaks its own rows."""
    clock = {"now": 0.0}
    # Process-global: other tests' entries carry real timestamps this fake clock cannot age out.
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


def _request_model(name: str):
    from models import inference, training

    return getattr(inference, name, None) or getattr(training, name)


@pytest.mark.parametrize(
    ("model_name", "fields", "extra"),
    [
        ("LoadRequest", ("model_path",), {}),
        ("ValidateModelRequest", ("model_path",), {}),
        ("DiffusionLoadRequest", ("model_path",), {}),
        ("VideoLoadRequest", ("model_path",), {}),
        ("UnloadRequest", ("model_path",), {}),
        ("EstimateMemoryRequest", ("model_path",), {}),
        (
            "TransformersUpgradeCheckRequest",
            (
                "model_name",
                "model_local_path",
                "model_snapshot_path",
                "model_snapshot_repo_id",
            ),
            {},
        ),
        (
            "TrainingStartRequest",
            ("model_name", "resume_from_checkpoint"),
            {"training_type": "LoRA/QLoRA", "format_type": "chat"},
        ),
        (
            "DiffusionTrainingStartRequest",
            ("base_model", "output_dir", "resume_from_checkpoint"),
            {"data_dir": "/data/images"},
        ),
    ],
)
def test_every_request_that_consumes_an_inventory_identity_resolves_the_handle(
    model_name, fields, extra
):
    request_model = _request_model(model_name)
    path = f"{HOST_ROOT}/my models/Llama-3.2-1B"
    reference = host_paths.cache_reference(path)

    resolved = request_model(**{field: reference for field in fields}, **extra)
    for field in fields:
        assert getattr(resolved, field) == path, (model_name, field)

    # An unissued reference is left as written, so it fails like an unknown model.
    for written in ("unsloth/Llama-3.2-1B", "ref:" + "0" * 32):
        echoed = request_model(**{field: written for field in fields}, **extra)
        for field in fields:
            assert getattr(echoed, field) == written, (model_name, field, written)


@pytest.mark.parametrize(
    ("client", "route"),
    [(_hub, "/api/hub/delete-cached"), (_models, "/api/models/delete-cached")],
    ids = ("hub", "compat"),
)
def test_a_cache_reference_can_delete_the_copy_it_names(monkeypatch, client, route):
    seen = {}

    async def _delete(repo_id, variant, hf_token, cache_path, only_if_orphan = None):
        seen.update(repo_id = repo_id, variant = variant, cache_path = cache_path)
        return {"status": "deleted", "repo_id": repo_id, "variant": variant}

    from hub.services.models import account_access, deletion

    monkeypatch.setattr(deletion, "delete_cached_model_response", _delete)
    monkeypatch.setattr(account_access, "require_installation_owner", lambda: None)
    reference = host_paths.cache_reference(REPO_DIR)
    assert reference != REPO_DIR

    def _sent(via_api_key, body):
        seen.clear()
        answered = client(via_api_key = via_api_key).request("DELETE", route, json = body)
        assert answered.status_code == 200, answered.text
        return answered

    answered = _sent(True, {"repo_id": "unsloth/Llama-3.2-1B", "cache_path": reference})
    assert answered.json()["status"] == "deleted"
    assert seen["cache_path"] == REPO_DIR

    # An absent `cache_path` reaches the service as None, not as some resolved root.
    _sent(True, {"repo_id": "unsloth/Llama-3.2-1B-Instruct", "variant": None})
    assert seen["cache_path"] is None

    # BOUNDARY. A browser session names the path itself, and it arrives as written.
    _sent(False, {"repo_id": "unsloth/Llama-3.2-1B", "cache_path": REPO_DIR})
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

    # A request that named a path DIRECTLY has no handle of the caller's to give back.
    named = {"model": f"{HOST_ROOT}/models/Llama-3.2-1B"}
    assert host_paths.restore_inventory_handles(named) == named


def test_a_tuple_keeps_its_shape_through_both_walks():
    """A NamedTuple is rebuilt field by field, a plain tuple from the iterable. Deciding that by
    try/except is WRONG: `tuple(*["abc"])` raises nothing and spells the string out."""
    from collections import namedtuple

    Row = namedtuple("Row", "first second")
    One = namedtuple("One", "only")

    path = f"{HOST_ROOT}/models/Llama-3.2-1B"
    token = host_paths._request_handles.set({path: host_paths.cache_reference(path)})
    try:
        assert host_paths.restore_inventory_handles(("abc",)) == ("abc",)
        assert host_paths.restore_inventory_handles(("ab", "cd")) == ("ab", "cd")
        assert host_paths.restore_inventory_handles(One("abc")) == One("abc")
        assert host_paths.restore_inventory_handles(Row("ab", "cd")) == Row("ab", "cd")
        assert host_paths.restore_inventory_handles((path,)) == (host_paths.cache_reference(path),)
    finally:
        host_paths._request_handles.reset(token)

    # This walk runs while a route is already raising: a TypeError here replaces the refusal
    # the caller was being told about with a 500.
    detail = host_paths.raised_inventory_detail(
        Row(f"{HOST_ROOT}/a", f"{HOST_ROOT}/b"), via_api_key = True
    )
    assert isinstance(detail, Row), detail
    assert response_leaks_host_path(detail, [HOST_ROOT]) is None, detail
    assert host_paths.redact_inventory_error_detail(("abc",), via_api_key = True) == ("abc",)
    # The payload walk rebuilds one rather than raising on it.
    assert redact_host_paths({"rows": [Row("acme/x", 5)]}, via_api_key = True)["rows"][0] == Row(
        "acme/x", 5
    )
    assert host_paths.raised_inventory_detail(Row("ab", "cd"), via_api_key = False) == Row(
        "ab", "cd"
    ), "a UI session's detail was rebuilt into something else"


def test_a_sibling_that_merely_starts_with_the_resolved_path_is_not_a_handle():
    """A substring swap turned `<root>/models/Llama-3.2-1B-private` into `ref:<digest>-private`:
    a handle resolving to nothing, and a value the redactor no longer reads as a path, so the
    rest of the layout rode out in it."""
    path = f"{HOST_ROOT}/models/Llama-3.2-1B"
    sibling = f"{path}-private"
    reference = host_paths.cache_reference(path)

    token = host_paths._request_handles.set({path: reference})
    try:
        restored = host_paths.restore_inventory_handles(
            {"active_model": sibling, "model_name": path, "detail": f"{sibling} is not loaded"}
        )
        assert restored["model_name"] == reference, "the path that WAS resolved still comes back"
        assert restored["active_model"] == sibling, restored
        assert restored["detail"] == f"{sibling} is not loaded", restored
        assert host_paths.resolve_host_path_reference(restored["active_model"]) is None

        answered = host_paths.redact_host_paths(restored, via_api_key = True)
        assert response_leaks_host_path(answered, [HOST_ROOT]) is None, answered
        assert host_paths.redact_host_paths(restored, via_api_key = False) == restored
    finally:
        host_paths._request_handles.reset(token)


def _source(target: str) -> str:
    from routes import inference as inference_routes
    from routes import training_history as training_routes
    from routes import video as video_routes

    module, _, attribute = target.partition(".")
    holder = {
        "inference": inference_routes,
        "video": video_routes,
        "training": training_routes,
        "models": models_routes,
    }[module]
    return inspect.getsource(getattr(holder, attribute) if attribute else holder)


# Wiring, invisible in a payload test because the payload is never built. A target marked
# `squeeze` is matched whitespace-insensitively, so a re-wrap is not read as a removal.
_WIRING = {
    ("inference", "squeeze"): (
        "restore_inventory_handles(task.result())",
        "_handle_restored_http_exception(exc)",
        "restore_inventory_handles(exc.detail)",
        "restore_inventory_handles(ValidateModelResponse(",
        "jsonable_encoder(restore_inventory_handles(payload))",
        "restore_inventory_handles(redact_native_paths(str(e)))",
        "raise _handle_restored_http_exception(http_error) from http_error",
        "detail = restore_inventory_handles(str(e))",
    ),
    ("inference", "exact"): (
        "model_name = restore_inventory_handles(model_name),",
        "redact_host_paths(DiffusionStatusResponse(",
        "authenticated_via_api_key",
    ),
    ("video", "exact"): ("redact_host_paths(VideoStatusResponse(", "authenticated_via_api_key"),
    ("training", "exact"): (
        'TrainingRunListResponse(runs = runs, total = result["total"]),',
        "authenticated_via_api_key",
    ),
    ("models.get_model_config", "exact"): (
        "via_api_key: bool = Depends(authenticated_via_api_key)",
        "redact_host_paths(",
        "restore_inventory_handles(await asyncio.to_thread(_resolve, model_name))",
        "echo = (model_name,)",
        "resolve_inventory_handle(",
    ),
    ("models.scan_model_remote_code", "exact"): (
        "resolve_inventory_handle(",
        "restore_inventory_handles(",
    ),
    ("inference.load_diffusion_model", "exact"): ("except HTTPException", "raised_inventory_detail"),
    ("video.load_video_model", "exact"): ("except HTTPException", "raised_inventory_detail"),
    ("inference.diffusion_load_progress", "exact"): (
        "authenticated_via_api_key",
        "redact_load_progress",
    ),
    ("video.video_load_progress", "exact"): ("authenticated_via_api_key", "redact_load_progress"),
}


@pytest.mark.parametrize(("target", "match"), list(_WIRING))
def test_the_routes_are_wired_to_the_redaction_they_depend_on(target, match):
    source = _source(target)
    squeeze = (lambda text: re.sub(r"\s+", "", text)) if match == "squeeze" else (lambda text: text)
    for needle in _WIRING[(target, match)]:
        assert squeeze(needle) in squeeze(source), (target, needle)


def test_a_refusal_names_the_handle_the_caller_sent(monkeypatch):
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


# Referenced rather than blanked, because nothing else in these answers names the model.
# REF means "resolves back to the row's path", KEPT "came back as it was".
REF = object()
KEPT = object()

_MODEL_PATH = f"{HOST_ROOT}/my models/Llama-3.2-1B"
_DATASET_PATH = "/home/operator/datasets/customer-transcripts.jsonl"


def _assert_redacted(row, expect, *, inventory = False, kept_text = (), roots = (HOST_ROOT,)):
    redact = redact_inventory_host_paths if inventory else redact_host_paths
    redacted = redact(row, via_api_key = True)
    body = json.dumps(redacted)
    for root in roots:
        assert root not in body, redacted
    for field, want in expect.items():
        if want is REF:
            assert host_paths.resolve_host_path_reference(redacted[field]) == row[field], field
        elif want is KEPT:
            assert redacted[field] == row[field], field
        else:
            assert redacted[field] == want, field
    for fragment in kept_text:
        assert fragment in body, redacted
    # BOUNDARY.
    assert redact(row, via_api_key = False) is row
    return redacted


@pytest.mark.parametrize(
    ("row", "expect", "kwargs"),
    [
        (
            {"loaded": True, "repo_id": _MODEL_PATH, "device": "cuda"},
            {"repo_id": REF, "device": KEPT}, {},
        ),
        (
            {"run_id": "abc", "model_name": _MODEL_PATH, "status": "completed"},
            {"model_name": REF, "status": KEPT}, {},
        ),
        (
            {"repo_id": "unsloth/Llama-3.2-1B", "model_name": "unsloth/Llama-3.2-1B"},
            {"repo_id": KEPT, "model_name": KEPT}, {},
        ),
        (
            {"id": "ref:whatever", "is_lora": True, "base_model": _MODEL_PATH},
            {"base_model": REF, "is_lora": KEPT}, {},
        ),
        ({"base_model": "unsloth/Llama-3.2-1B"}, {"base_model": KEPT}, {}),
        (
            {
                "id": "run-1",
                "dataset_name": _DATASET_PATH,
                "local_datasets": [_DATASET_PATH],
                "model_name": "unsloth/Llama-3.2-1B",
            },
            {"dataset_name": REF, "model_name": KEPT}, {"roots": ("/home/operator",)},
        ),
        ({"id": "run-2", "dataset_name": "unsloth/Radiology-mini"}, {"dataset_name": KEPT}, {}),
        (
            {
                "error_message": f"FileNotFoundError: no such file: {HOST_ROOT}/data/train.jsonl",
                "status": "error",
            },
            {"status": KEPT}, {"kept_text": ("FileNotFoundError",)},
        ),
        (
            {
                "source": "models_dir",
                "id": f"{HOST_ROOT}/models/Llama-3.2-1B",
                "load_id": f"{HOST_ROOT}/models/Llama-3.2-1B",
                "repo_id": "unsloth/Llama-3.2-1B",
            },
            {"id": REF, "load_id": REF, "repo_id": KEPT}, {"inventory": True},
        ),
    ],
    ids = (
        "loaded-status", "training-run", "hub-row", "lora-base", "hub-base",
        "local-dataset-name", "hub-dataset-name", "failure-message", "local-row",
    ),
)
def test_a_path_that_outlives_its_request_is_still_referenced(row, expect, kwargs):
    _assert_redacted(row, expect, **kwargs)


def test_a_second_load_does_not_hand_back_the_resident_path(monkeypatch):
    import asyncio

    from models.inference import DiffusionStatusResponse
    from routes import inference as inference_routes

    resident = f"{HOST_ROOT}/my models/previous-image-model"
    resident_reference = host_paths.cache_reference(resident)

    async def _gated(request, subject, **kwargs):
        return DiffusionStatusResponse(
            **{"loaded": True, "repo_id": resident, "model_path": resident}
        )

    class _Request:
        model_path = "ref:whatever"
        base_repo = None

    monkeypatch.setattr(inference_routes, "load_diffusion_model_gated", _gated)

    def _loaded(subject, via_api_key):
        return json.dumps(
            jsonable(
                asyncio.run(
                    inference_routes.load_diffusion_model(
                        _Request(), current_subject = subject, via_api_key = via_api_key
                    )
                )
            )
        )

    body = _loaded("api", True)
    assert HOST_ROOT not in body, body
    assert resident_reference in body, body
    assert host_paths.resolve_host_path_reference(resident_reference) == resident

    # BOUNDARY.
    assert resident in _loaded("browser", False)


def test_every_route_that_answers_with_a_persisted_record_redacts():
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


def test_opening_or_renaming_a_run_does_not_hand_back_the_path(monkeypatch):
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

    monkeypatch.setattr(training_routes, "get_run", lambda run_id: dict(row))
    monkeypatch.setattr(training_routes, "get_run_metrics", lambda run_id: {})
    monkeypatch.setattr(training_routes, "get_preview_sharing_enabled", lambda: False)

    async def _opened():
        return await training_routes.get_training_run_detail(
            "run-1", current_subject = "api", no_credential = False, via_api_key = True
        )

    async def _renamed():
        return await training_routes.update_training_run(
            "run-1",
            training_routes.TrainingRunUpdateRequest(),
            current_subject = "api",
            no_credential = False,
            via_api_key = True,
        )

    for answer in (_opened, _renamed):
        body = json.dumps(jsonable(asyncio.run(answer())))
        assert HOST_ROOT not in body, body
        assert reference in body, body


def test_a_prepared_dataset_cache_still_counts_when_the_hub_copy_is_unusable(monkeypatch):
    import sys
    import types

    from hub.utils import hf_tokens

    cache_state = types.ModuleType("hub.utils.hf_cache_state")
    cache_state.iter_repo_cache_dirs = lambda repo_type, repo_id: iter(["a-directory"])
    cache_state.repo_cache_has_usable_snapshot = lambda repo_type, repo_id: False
    dataset_cache = types.ModuleType("hub.utils.dataset_cache")
    dataset_cache.latest_processed_dataset_cache_path = lambda repo_id: "/prepared/here"

    monkeypatch.setitem(sys.modules, "hub.utils.hf_cache_state", cache_state)
    monkeypatch.setitem(sys.modules, "hub.utils.dataset_cache", dataset_cache)

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


def test_the_metadata_route_resolves_and_answers_with_the_handle():
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
    # Resume fields are referenced, not blanked, or `can_resume` is true beside no identifier.
    _assert_redacted(
        row,
        {
            "output_dirs": [],
            "model_name": REF,
            "output_dir": REF,
            "checkpoint_path": REF,
            "resume_from_checkpoint": REF,
            "dataset_name": KEPT,
        },
    )


def test_the_deferred_five_hundred_restores_the_handle_too():
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


def test_a_resumable_run_can_still_be_resumed_by_an_api_key_caller():
    from models.training import TrainingStartRequest

    output_dir = f"{HOST_ROOT}/outputs/run-1"
    detail = host_paths.redact_host_paths(
        {
            "id": "run-1",
            "can_resume": True,
            "output_dir": output_dir,
            "checkpoint_path": None,
            "local_datasets": [f"{HOST_ROOT}/datasets/train.jsonl"],
            "local_eval_datasets": [f"{HOST_ROOT}/datasets/eval.jsonl", "acme/hub-eval"],
        },
        via_api_key = True,
    )
    assert detail["can_resume"] is True
    handle = detail["output_dir"]
    assert handle.startswith("ref:"), detail
    assert HOST_ROOT not in json.dumps(detail), detail
    assert all(entry.startswith("ref:") for entry in detail["local_datasets"])
    assert detail["local_eval_datasets"][1] == "acme/hub-eval"

    replayed = TrainingStartRequest(
        model_name = "unsloth/Llama-3.2-1B",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        local_datasets = detail["local_datasets"],
        local_eval_datasets = detail["local_eval_datasets"],
        resume_from_checkpoint = handle,
    )
    assert replayed.resume_from_checkpoint == output_dir
    assert replayed.local_datasets == [f"{HOST_ROOT}/datasets/train.jsonl"]
    assert replayed.local_eval_datasets == [f"{HOST_ROOT}/datasets/eval.jsonl", "acme/hub-eval"]

    # A handle that was never issued resolves to nothing rather than somebody else's data.
    forged = "ref:" + "0" * 32
    unresumable = TrainingStartRequest(
        model_name = "unsloth/Llama-3.2-1B",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        local_datasets = [forged],
    )
    assert unresumable.local_datasets == [forged], "a forged handle resolved to a host path"
    assert unresumable.resume_from_checkpoint is None


def test_a_listing_bigger_than_the_reference_table_still_resolves_its_own_rows():
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
    host_paths._reference_paths.clear()
    monkeypatch.setattr(host_paths, "_REFERENCE_CEILING", host_paths._REFERENCE_LIMIT + 10)
    for index in range(host_paths._REFERENCE_LIMIT + 500):
        host_paths.cache_reference(f"/srv/ceiling/row-{index}.gguf")
    assert len(host_paths._reference_paths) <= host_paths._REFERENCE_CEILING + 1


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


# `base_repo` is usually a Hub id, but the local form is a host path, and the STATUS routes
# answer it to whoever polls rather than to the caller who supplied it.

BASE_DIR = f"{HOST_ROOT}/flux-base"


def _media_status(cls):
    from models.inference import DiffusionStatusResponse, VideoStatusResponse
    return {"diffusion": DiffusionStatusResponse, "video": VideoStatusResponse}[cls]


def _base_repo(answer):
    return answer.get("base_repo") if isinstance(answer, dict) else answer.base_repo


@pytest.mark.parametrize("kind", ["diffusion", "video"])
def test_a_local_companion_base_is_referenced_not_returned(kind):
    """A row advertised as actionable has to be actionable: the handle the status route hands
    out must resolve back to the path on the load request that takes it."""
    from models.inference import DiffusionLoadRequest, VideoLoadRequest

    response = _media_status(kind)(loaded = True, repo_id = "unsloth/x", base_repo = BASE_DIR)

    api = redact_host_paths(response, via_api_key = True)
    handle = _base_repo(api)
    assert handle.startswith("ref:"), handle
    assert response_leaks_host_path(api, [HOST_ROOT, HOST_ROOT_NATIVE]) is None

    # BOUNDARY.
    assert _base_repo(redact_host_paths(response, via_api_key = False)) == BASE_DIR

    request_cls = {"diffusion": DiffusionLoadRequest, "video": VideoLoadRequest}[kind]
    assert request_cls(model_path = "unsloth/x", base_repo = handle).base_repo == BASE_DIR


@pytest.mark.parametrize("kind", ["diffusion", "video"])
def test_a_hub_companion_base_is_not_a_path_and_is_left_alone(kind):
    """BOUNDARY, and the reason this field is decided by VALUE: referencing the ordinary repo
    id would make the field useless to every caller for nothing."""
    response = _media_status(kind)(
        loaded = True, repo_id = "unsloth/x", base_repo = "black-forest-labs/FLUX.2-klein-4B"
    )

    api = redact_host_paths(response, via_api_key = True)
    assert _base_repo(api) == "black-forest-labs/FLUX.2-klein-4B"


def test_a_media_load_that_raises_redacts_the_path_it_resolved():
    """A load that RAISES skips the route's restore+redact wrappers, and the inner loader only
    redacted NATIVE paths, which do not know inventory handles."""
    from hub.utils.host_paths import raised_inventory_detail

    detail = f"[Errno 2] No such file or directory: '{REPO_DIR}/model.safetensors'"

    assert (
        response_leaks_host_path(
            {"detail": raised_inventory_detail(detail, via_api_key = True)},
            [HOST_ROOT, HOST_ROOT_NATIVE],
        )
        is None
    )
    # BOUNDARY.
    assert raised_inventory_detail(detail, via_api_key = False) == detail


def test_a_media_load_progress_error_does_not_publish_the_resolved_path():
    """The worker stores `str(exc)`, and the progress routes answer with it on a LATER request,
    where the load's handle map is gone: the path has to be removed, not referenced."""
    from hub.utils.host_paths import redact_load_progress

    progress = {"phase": "error", "error": f"could not read {REPO_DIR}/model.safetensors"}

    redacted = redact_load_progress(progress, via_api_key = True)
    assert response_leaks_host_path(redacted, [HOST_ROOT, HOST_ROOT_NATIVE]) is None
    assert redacted["phase"] == "error", "the phase the client polls for was dropped"
    # BOUNDARY.
    assert redact_load_progress(progress, via_api_key = False) == progress
    assert redact_load_progress({"phase": "ready"}, via_api_key = True) == {"phase": "ready"}
