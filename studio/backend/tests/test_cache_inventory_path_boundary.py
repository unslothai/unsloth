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
# Nothing is exempt any more. A cache row is named by its repo id, and where it is not --
# a copy outside the active cache, a `refs/main` on an unusable revision -- `load_id` holds
# an absolute snapshot path, which is the cache root, the home directory and the account
# name. It is referenced like any other identity now that every request that consumes one
# resolves the reference back.
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


def test_the_models_folder_error_is_not_disclosed_either(monkeypatch):
    """The redactors only ever see a payload that was BUILT.

    This route raises with the cache path in the detail when the folder cannot be created
    or turns out to be a file, and an exception raised while evaluating an argument never
    reaches the function it was being passed to. So the one caller the whole redaction
    exists for got the host path out of the 500 instead of out of the 200, on a failure
    that is not exotic: a read-only HF_HOME, or a file where the directory should be.
    """
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
    # The cause survives the redaction, or the error stops being actionable.
    assert "Permission denied" in detail, detail
    assert "Failed to create models folder" in detail, detail

    # A UI session still sees the path, which is the whole point of the distinction.
    session = _hub(via_api_key = False).get("/api/hub/models-folder")
    assert session.status_code == 500
    assert HOST_ROOT in session.json()["detail"]


@pytest.mark.parametrize("root", ["/tmp", "/cache", "C:\\cache"])
def test_a_root_with_one_component_is_redacted_too(monkeypatch, root):
    """`/tmp` is as absolute as `/home/alice/.unsloth/studio/hub`.

    The pattern required a second separator after the root, so exactly the shape a
    configured cache root is usually spelled in was left in the response. This is the value
    the models-folder error puts in its detail, so the incomplete pattern meant the fix did
    not cover the case that prompted it.
    """
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
    """Widening the pattern must not start eating text. A ratio and a slashed conjunction
    are the two shapes that look like a root and are not."""
    from hub.utils.host_paths import redact_paths_in_text
    for kept in ("3/4 of the shards", "and/or the projector", "no paths here"):
        assert redact_paths_in_text(kept) == kept, kept


def test_a_structured_error_detail_is_walked_too(monkeypatch):
    """Nothing stops a route from raising a dict, and a scrubber that only handles strings
    would hand the path straight back."""
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


def test_a_windows_network_share_is_shortened_like_any_other_path():
    """A UNC path is the third spelling of absolute on Windows, and a scan folder on a network
    share is written that way, so the server and share name used to survive in full."""
    assert scrub_paths(r"Skipping \\fileserver\models\hub\models--acme--x: denied") == (
        "Skipping .../hub/models--acme--x: denied"
    )
    assert scrub_paths(r"open \\srv\share\acme\config.json failed") == (
        "open .../acme/config.json failed"
    )
    # And the shapes that are not paths stay exactly as written.
    assert scrub_paths("ratio 3/4") == "ratio 3/4"
    assert scrub_paths("https://huggingface.co/api/models/a/b") == (
        "https://huggingface.co/api/models/a/b"
    )


def test_a_long_line_with_no_path_in_it_is_not_a_stall():
    """The pattern nests a quantifier, so a line built from an exception has to stay linear."""
    import time

    line = ("word " * 20_000) + "x"
    started = time.monotonic()
    assert scrub_paths(line) == line
    assert time.monotonic() - started < 1.0


def test_a_named_tuple_in_a_payload_is_rebuilt_rather_than_raising():
    """Nothing answers one today, and the walk must not turn a future one into a 500: a
    NamedTuple is a tuple whose constructor takes its fields one by one."""
    import collections

    Row = collections.namedtuple("Row", "repo_id size_bytes")
    payload = {"rows": [Row("acme/x", 5)]}
    out = redact_host_paths(payload, via_api_key = True)
    assert out["rows"][0] == Row("acme/x", 5)


# ------------------------------------------------------------ what a client can still tell apart
#
# A redacted scalar is the empty string, not None, and the difference is load bearing: a client
# that tested `if row["cache_path"]` used to read "this row is cached". After the redaction that
# test is false for every row, cached or not, so the discriminator has to be somewhere. It is
# `cache_ref`, which is written exactly when the row HAD a path and omitted when it did not, and
# these cases pin that so it cannot be dropped as an implementation detail.


def test_a_redacted_row_is_still_distinguishable_from_one_with_no_path_at_all():
    """`cache_ref` is the discriminator a truthiness test on `cache_path` used to be."""
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
    """A client groups rows and de-duplicates a repo cached under two roots with it, so it has to
    be the same answer for the same path and a different one for a different path."""
    a = {"cache_path": "/home/op/.cache/huggingface/hub/a"}
    b = {"cache_path": "/home/op/.cache/huggingface/hub/b"}
    ref = lambda row: host_paths.redact_host_paths(row, via_api_key = True)[
        host_paths.CACHE_REFERENCE_FIELD
    ]
    assert ref(a) == ref(dict(a))
    assert ref(a) != ref(b)
    assert "/home/op" not in ref(a) and "hub" not in ref(a)


def test_a_browser_session_keeps_every_path_and_gains_nothing():
    """The other half: the operator looking at their own machine is unchanged, including not
    acquiring a reference field their bundle does not expect."""
    row = {"repo_id": "acme/model", "cache_path": "/home/op/.cache/huggingface/hub/x"}
    same = host_paths.redact_host_paths(row, via_api_key = False)
    assert same is row
    assert host_paths.CACHE_REFERENCE_FIELD not in same


def test_a_local_adapter_base_model_is_redacted_but_a_repo_id_is_kept():
    """`base_model` is a path for one adapter and a repo id for the next.

    It carries `adapter_config.json`'s `base_model_name_or_path` verbatim. For most LoRAs
    that is a Hub repo id, which a caller needs and which must survive. For one trained
    against a local base it is an absolute path to a SEPARATE directory on this machine, and
    an API-key request to `/api/hub/local` disclosed it: `base_model` is not in
    HOST_PATH_SCALAR_FIELDS, and it could not simply be added there without blanking the
    repo id on every other row. The sibling the scan already writes decides it --
    `_base_model_source` answers "local" only after resolving the value on this filesystem.
    """
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

    # The browser session sees the machine it runs on, as everywhere else in this file.
    session = redact_inventory_host_paths(payload, via_api_key = False)["models"]
    assert session[0]["base_model"] == "/home/op/models/Llama-3.1-8B"


def test_the_leak_finder_knows_a_local_base_model_is_a_path():
    """The drift gate has to see it too, or the next field like this is caught by nobody."""
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

    # A repo id in the same field is not a leak, even though the root string is absent.
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


# --------------------------------------------------------------------------------------
# The compatibility router answers the same questions
# --------------------------------------------------------------------------------------
# /api/models is the OpenAI-compatible mirror of /api/hub, mounted in the same app and
# reachable with the same key. A boundary drawn on one router only is not a boundary: the
# caller simply asks the other one.


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


# The drift gate, for the compatibility mirror. Named rather than derived: /api/models
# carries a hundred routes that have nothing to do with the inventory, and a gate over all
# of them would be noise. These are the ones that answer what /api/hub answers.
_COMPAT_INVENTORY_ROUTES = (
    "list_local_models",
    "get_scan_folders",
    "get_download_progress",
    "get_gguf_download_progress",
    "list_cached_gguf",
    "list_cached_models",
)


def test_every_compat_mirror_of_an_inventory_route_takes_the_caller_class():
    source = Path(models_routes.__file__).read_text()
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


# --------------------------------------------------------------------------------------
# A row whose IDENTITY is a path
# --------------------------------------------------------------------------------------


def _local_row():
    """The shape `_local_model_info` builds for a filesystem-backed model."""
    from hub.services.models.common import _local_model_info

    load_path = Path(HOST_ROOT) / "my models" / "Llama-3.2-1B"
    return _local_model_info(
        scan_path = load_path,
        load_path = load_path,
        source = "models_dir",
        model_format = "safetensors",
    )


def test_a_filesystem_backed_row_is_not_named_by_its_path(monkeypatch):
    """`id`, `load_id` and `inventory_id` all spell out the load path for a local row.

    Blanking `path` hides nothing while the same string is still the row's identity, and
    `inventory_id` carries it URL encoded, which reverses. The load-handle exception is
    about CACHED rows, whose handle is a snapshot of a repo the caller already named.
    """

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
    # The inventory id keeps its `<source>:<format>:<identity>` shape, which clients split.
    assert row["inventory_id"].startswith("models_dir:safetensors:"), row["inventory_id"]
    assert HOST_ROOT not in json.dumps(payload), payload
    assert "my%20models" not in json.dumps(payload), payload
    # Nothing is ignored here: the whole row, load handle included, must be free of it.
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None

    ui = _hub(via_api_key = False).get("/api/hub/local").json()
    assert ui["models"][0]["load_id"].startswith(HOST_ROOT)


def test_a_cached_row_keeps_the_repo_id_it_is_named_by(_cached_inventory):
    """The other half. A cached row is named by its repo id, and referencing that would
    take away the only thing a caller can act on."""
    payload = _hub(via_api_key = True).get("/api/hub/cached-models").json()
    assert payload["cached"][0]["repo_id"] == "unsloth/Llama-3.2-1B-Instruct"
    assert not payload["cached"][0]["repo_id"].startswith("ref:")


def test_a_local_scan_failure_is_redacted_like_its_payload(monkeypatch):
    """An exception raised while evaluating the awaited argument never reaches the redactor
    it was being passed to, and the service turns scan and database failures into an
    HTTPException whose detail carries the raw string."""
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
    """A redacted identity that nothing can act on is not a redaction, it is a broken row.

    The picker hands `load_id` straight back when it loads, so the reference the listing
    answered with has to resolve to the path it stands for on the way in.
    """
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

    # A reference this process never issued is left exactly as it arrived, so it fails the
    # way an unknown model does rather than resolving to somebody else's row.
    stranger = "ref:" + "0" * 32
    assert LoadRequest(model_path = stranger).model_path == stranger
    # And an ordinary repo id is untouched.
    assert LoadRequest(model_path = "unsloth/Llama-3.2-1B").model_path == "unsloth/Llama-3.2-1B"


def test_a_reference_table_that_fills_up_drops_the_oldest(monkeypatch):
    """A long-lived server lists a great many rows. The table is bounded, and a reference
    that has aged out does not resolve rather than resolving to the wrong thing."""
    monkeypatch.setattr(host_paths, "_REFERENCE_LIMIT", 4)
    first = host_paths.cache_reference("/host/first")
    for index in range(6):
        host_paths.cache_reference(f"/host/filler-{index}")
    assert host_paths.resolve_host_path_reference(first) is None
    newest = host_paths.cache_reference("/host/newest")
    assert host_paths.resolve_host_path_reference(newest) == "/host/newest"


def test_every_request_that_consumes_an_inventory_identity_resolves_the_handle():
    """One endpoint resolving the handle is not enough: the picker's identity is passed to
    validation, diffusion, video and training too, and `is_local_path` reads the colon in
    `ref:` as local syntax, so those calls failed as a nonexistent path."""
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
        # And an ordinary repo id is untouched everywhere.
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
    """An API-key caller is given `cache_ref` instead of `cache_path`, so the reference is
    the only identifier it has for a specific non-active copy. Sending it produced "Invalid
    cache_path", and omitting it acted on the active root, which is a different copy."""
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

    # A literal path from a browser session still goes through untouched.
    _hub(via_api_key = False).request(
        "DELETE",
        "/api/hub/delete-cached",
        json = {"repo_id": "unsloth/Llama-3.2-1B", "cache_path": REPO_DIR},
    )
    assert seen["cache_path"] == REPO_DIR


def test_a_cache_row_pinned_to_a_snapshot_is_referenced_too(monkeypatch):
    """The load-handle exception was written for a row named by its repo id.

    A cached copy outside the active cache, or a `refs/main` that points at an unusable
    revision, pins `load_id` to an absolute snapshot path instead, and the exception then
    copied that path through into an API-key response: the cache root, the home directory
    and the account name, from a listing that blanks `cache_path` two fields earlier. The
    value decides, not the row's source.
    """
    pinned = f"{HOST_ROOT}/models--unsloth--Llama-3.2-1B-Instruct/snapshots/deadbeef"

    async def _models_response(hf_token = None):
        return {"cached": [_cached_row(load_id = pinned)], "scan_confirmed": True}

    monkeypatch.setattr(cache_inventory, "list_cached_models_response", _models_response)
    payload = _hub(via_api_key = True).get("/api/hub/cached-models").json()
    row = payload["cached"][0]
    # Still named by the repo id, which is the thing a caller can act on.
    assert row["repo_id"] == "unsloth/Llama-3.2-1B-Instruct"
    assert row["load_id"].startswith("ref:"), row["load_id"]
    assert HOST_ROOT not in json.dumps(payload), payload
    assert response_leaks_host_path(payload, [HOST_ROOT]) is None
    # And the reference reverses, so the row is still loadable.
    assert host_paths.resolve_host_path_reference(row["load_id"]) == pinned

    ui = _hub(via_api_key = False).get("/api/hub/cached-models").json()
    assert ui["cached"][0]["load_id"] == pinned


def test_the_leak_finder_knows_a_path_valued_identity_is_a_path():
    """The gate is what stops the next field from arriving unredacted, and it was reading
    the field name: `load_id` is not a path field, so a snapshot path sitting in one was
    reported clean unless the test happened to pass the host root as a needle."""
    assert response_leaks_host_path({"load_id": f"{HOST_ROOT}/snapshots/abc"}) is not None
    assert response_leaks_host_path({"load_id": "unsloth/Llama-3.2-1B-Instruct"}) is None
    assert response_leaks_host_path({"load_id": "ref:0123456789abcdef"}) is None


def test_the_compat_scan_folder_add_does_not_answer_with_the_path(monkeypatch):
    """Both listings hide the normalized absolute path; the POST that creates the folder
    handed it straight back, so submitting a relative directory such as `.` and reading the
    answer recovered the server's working directory."""
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
    """Resolving the reference is what makes a redacted row loadable, and it is also what
    put the path in the ANSWER.

    `ValidateModelResponse.identifier`, `LoadResponse.model` and the label beside it are all
    built from what was asked for, and an error detail quotes it too, so a caller could
    enumerate a redacted listing and read the path straight back out of the load it had just
    performed with the reference. What comes back is the string that went in.
    """
    from models.inference import LoadRequest

    path = f"{HOST_ROOT}/my models/Llama-3.2-1B"
    reference = host_paths.cache_reference(path)
    assert LoadRequest(model_path = reference).model_path == path

    # Everywhere it landed, not only the identity field: the display label, the inference
    # identifier built around it, a list, and the detail of something that went wrong.
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
    """A browser session names paths and expects them back. Only a handle this request
    resolved is put back, so nothing else in the answer is rewritten."""
    answer = {"model": f"{HOST_ROOT}/models/Llama-3.2-1B"}
    assert host_paths.restore_inventory_handles(answer) == answer


def test_the_load_and_validate_answers_go_through_the_restoration():
    """A route that builds the response and returns it unwrapped would leave every
    assertion above passing and the path still going out."""
    import inspect
    from routes import inference as inference_routes

    source = inspect.getsource(inference_routes)
    assert "restore_inventory_handles(task.result())" in source
    assert "restore_inventory_handles(ValidateModelResponse(" in source
    assert "jsonable_encoder(restore_inventory_handles(payload))" in source
    assert "restore_inventory_handles(redact_native_paths(str(e)))" in source
