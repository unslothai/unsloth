# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A flow `dataset` value resolved to rows for the viewers, per source."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.flows import dataset_rows as dr
from hub.services.datasets import formatting
from hub.utils.host_paths import cache_reference
from routes import flows as flows_routes
from utils.paths.storage_roots import dataset_uploads_root, recipe_datasets_root

ROWS = [{"sql_prompt": f"q{i}", "sql": f"SELECT {i}"} for i in range(30)]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    with TestClient(app) as test_client:
        yield test_client


def _rows(
    client,
    value,
    offset = 0,
    limit = 20,
):
    return client.post(
        "/api/flows/dataset-rows", json = {"value": value, "offset": offset, "limit": limit}
    )


def test_each_source_names_what_the_dataset_check_reads():
    hf = dr.dataset_source(
        {"kind": "dataset", "source": "hf", "ref": "org/set", "split": "test", "subset": "a"}
    )
    assert hf == {"dataset_name": "org/set", "subset": "a", "train_split": "test"}
    recipe = dr.dataset_source(
        {"kind": "dataset", "source": "recipe", "ref": "recipes/run1/parquet-files", "job_id": "j"}
    )
    assert recipe["dataset_name"] == str(recipe_datasets_root() / "run1" / "parquet-files")
    for bad in (
        {"kind": "table", "ref": "x"},
        {"kind": "dataset", "source": "hf", "ref": "no-owner"},
        {"kind": "dataset", "source": "local", "ref": "ref:unknown"},
        {"kind": "dataset", "source": "s3", "ref": "x"},
        {"kind": "dataset", "source": "local", "ref": ""},
    ):
        with pytest.raises(dr.DatasetRowsError):
            dr.dataset_source(bad)


def test_a_recipe_dataset_pages_from_its_parquet(client):
    folder = recipe_datasets_root() / "recipe_run1" / "parquet-files"
    folder.mkdir(parents = True)
    pq.write_table(pa.Table.from_pylist(ROWS), folder / "batch_0.parquet")
    value = {"kind": "dataset", "source": "recipe", "ref": "recipes/recipe_run1/parquet-files"}
    first = _rows(client, value).json()
    assert first["total"] == 30 and first["rows"][:2] == ROWS[:2] and len(first["rows"]) == 20
    second = _rows(client, value, offset = 20).json()
    assert second["rows"] == ROWS[20:]


def test_a_managed_account_reads_its_recipe_dataset(client, monkeypatch):
    from core.training import account_jobs

    folder = recipe_datasets_root() / "recipe_run2" / "parquet-files"
    folder.mkdir(parents = True)
    pq.write_table(pa.Table.from_pylist(ROWS[:4]), folder / "batch_0.parquet")
    monkeypatch.setattr(account_jobs, "managed_account", lambda: True)
    value = {"kind": "dataset", "source": "recipe", "ref": "recipes/recipe_run2/parquet-files"}
    response = _rows(client, value)
    assert response.status_code == 200, response.text
    assert response.json()["rows"] == ROWS[:4]


def test_rows_to_dataset_output_reads_from_its_stored_jsonl(client):
    path = dataset_uploads_root() / "flow-rows.jsonl"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("\n".join(json.dumps(r) for r in ROWS[:5]))
    value = {"kind": "dataset", "source": "local", "ref": cache_reference(str(path)), "rows": 5}
    body = _rows(client, value, limit = 3).json()
    assert body["rows"] == ROWS[:3]


def test_a_hub_dataset_is_read_like_the_train_page_check(client, monkeypatch):
    seen = {}

    def preview(request, token, size, **_):
        seen.update(name = request.dataset_name, split = request.train_split, size = size)
        return ROWS[:size], 1234

    monkeypatch.setattr(formatting, "load_preview_slice", preview)
    value = {"kind": "dataset", "source": "hf", "ref": "org/sql", "split": "train"}
    body = _rows(client, value, offset = 10, limit = 5).json()
    assert seen == {"name": "org/sql", "split": "train", "size": 15}
    assert body["rows"] == ROWS[10:15] and body["total"] == 1234


def test_the_viewer_stops_at_its_row_window_and_refuses_what_it_cannot_read(client):
    assert _rows(client, {"kind": "dataset", "source": "s3", "ref": "x"}).status_code == 400
    escape = {"kind": "dataset", "source": "local", "ref": "../../etc/passwd"}
    assert _rows(client, escape).status_code == 400
    hub = {"kind": "dataset", "source": "hf", "ref": "org/sql"}
    assert _rows(client, hub, offset = 100_001).status_code == 422


def test_the_viewer_reads_no_further_than_its_window(client, monkeypatch):
    sizes = []

    def preview(request, token, size, **_):
        sizes.append(size)
        return ROWS[:size], 5000

    monkeypatch.setattr(formatting, "load_preview_slice", preview)
    hub = {"kind": "dataset", "source": "hf", "ref": "org/sql"}
    assert (
        _rows(client, hub, offset = dr.VIEW_ROWS - 10, limit = 50).json()["view_rows"] == dr.VIEW_ROWS
    )
    assert _rows(client, hub, offset = dr.VIEW_ROWS, limit = 50).json()["rows"] == []
    assert max(sizes) == dr.VIEW_ROWS
