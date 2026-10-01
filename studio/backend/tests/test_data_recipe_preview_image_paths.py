# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image cells are row data: a managed account must not preview pixels outside its workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL.Image")

from core.data_recipe.jsonable import to_preview_jsonable_row
from utils.account_context import AccountContext, run_as
from utils.paths.storage_roots import workspace_root

ALICE = AccountContext("a" * 32, "alice")


def _png(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    PIL.new("RGB", (4, 4), (255, 0, 0)).save(path)
    return path


def _is_image_payload(value) -> bool:
    return isinstance(value, dict) and value.get("type") == "image" and bool(value.get("data"))


@pytest.fixture
def studio_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    return tmp_path


def test_managed_account_cannot_preview_a_path_outside_its_workspace(studio_home):
    secret = _png(studio_home / "elsewhere" / "private.png")
    row = {"label": "x", "image": {"bytes": None, "path": str(secret)}}

    preview = run_as(ALICE, to_preview_jsonable_row, row)

    assert preview["label"] == "x"
    assert not _is_image_payload(preview["image"])


def test_managed_account_previews_its_own_workspace_image(studio_home):
    own = run_as(ALICE, lambda: _png(workspace_root() / "data" / "mine.png"))
    row = {"image": {"bytes": None, "path": str(own)}}

    preview = run_as(ALICE, to_preview_jsonable_row, row)

    assert _is_image_payload(preview["image"])


@pytest.mark.parametrize("path", ["~unsloth-no-such-user-xyz/image.png", "data/bad\x00name.png"])
def test_managed_account_unresolvable_path_is_not_an_image(studio_home, path):
    preview = run_as(ALICE, to_preview_jsonable_row, {"image": {"bytes": None, "path": path}})

    assert not _is_image_payload(preview["image"])


def test_owner_previews_any_local_path(studio_home):
    image = _png(studio_home / "elsewhere" / "local.png")

    preview = to_preview_jsonable_row({"image": {"bytes": None, "path": str(image)}})

    assert _is_image_payload(preview["image"])


def _path_only_parquet(
    tmp_path: Path,
    image: Path,
    nested: bool = False,
) -> Path:
    import datasets
    import pyarrow.parquet as pq

    cell = {"bytes": None, "path": str(image)}
    image_feature = [datasets.Image()] if nested else datasets.Image()
    features = datasets.Features({"image": image_feature, "label": datasets.Value("string")})
    out = tmp_path / "rows.parquet"
    dataset = datasets.Dataset.from_dict(
        {"image": [[cell] if nested else cell], "label": ["x"]}, features = features
    )
    # pyarrow directly: to_parquet stats every path, and the Hub one does not resolve offline.
    pq.write_table(dataset.data.table, str(out))
    return out


def _hub_preview(parquet: Path) -> list[dict]:
    from datasets import load_dataset
    from routes.data_recipe.seed import _load_preview_rows, _serialize_preview_rows

    rows = _load_preview_rows(
        load_dataset_fn = load_dataset,
        load_kwargs = {
            "path": "parquet",
            "data_files": [str(parquet)],
            "split": "train",
            "streaming": True,
        },
        preview_size = 1,
    )
    return _serialize_preview_rows(rows)


@pytest.mark.parametrize("nested", [False, True])
def test_managed_account_streamed_image_feature_is_checked_before_it_is_opened(studio_home, nested):
    pytest.importorskip("datasets")
    private = _png(studio_home / "elsewhere" / "private.png")
    parquet = _path_only_parquet(studio_home, private, nested = nested)

    (row,) = run_as(ALICE, _hub_preview, parquet)

    assert row["label"] == "x"
    cells = row["image"] if nested else [row["image"]]
    assert cells and not any(_is_image_payload(cell) for cell in cells)


@pytest.mark.parametrize("nested", [False, True])
def test_managed_account_streamed_image_inside_its_workspace_previews(studio_home, nested):
    pytest.importorskip("datasets")
    own = run_as(ALICE, lambda: _png(workspace_root() / "data" / "mine.png"))
    parquet = _path_only_parquet(studio_home, own, nested = nested)

    (row,) = run_as(ALICE, _hub_preview, parquet)

    cells = row["image"] if nested else [row["image"]]
    assert cells and all(_is_image_payload(cell) for cell in cells)


@pytest.mark.parametrize("nested", [False, True])
def test_owner_streamed_image_feature_still_previews(studio_home, nested):
    pytest.importorskip("datasets")
    image = _png(studio_home / "elsewhere" / "local.png")
    parquet = _path_only_parquet(studio_home, image, nested = nested)

    (row,) = _hub_preview(parquet)

    cells = row["image"] if nested else [row["image"]]
    assert cells and all(_is_image_payload(cell) for cell in cells)


def test_managed_account_image_file_lookup_skips_the_cwd_fallback(studio_home, monkeypatch):
    from core.data_recipe.service import _load_image_file_to_base64

    _png(studio_home / "cwd" / "private.png")
    monkeypatch.chdir(studio_home / "cwd")
    base = run_as(ALICE, lambda: workspace_root() / "data")

    assert run_as(ALICE, _load_image_file_to_base64, "private.png", base_path = str(base)) is None
    assert _load_image_file_to_base64("private.png", base_path = str(base)) is not None


def test_managed_account_hub_hosted_images_still_preview(studio_home, monkeypatch):
    datasets = pytest.importorskip("datasets")
    hub = "hf://datasets/org/repo@0123abc/cat.png"
    parquet = _path_only_parquet(studio_home, hub)
    opened = []

    def fake_decode(
        self,
        value,
        token_per_repo_id = None,
    ):
        opened.append(value["path"])
        return PIL.new("RGB", (4, 4))

    monkeypatch.setattr(datasets.Image, "decode_example", fake_decode)

    (row,) = run_as(ALICE, _hub_preview, parquet)

    assert opened == [hub]
    assert _is_image_payload(row["image"])


def test_managed_account_file_url_is_not_treated_as_hub_hosted(studio_home):
    pytest.importorskip("datasets")
    private = _png(studio_home / "elsewhere" / "private.png")
    parquet = _path_only_parquet(studio_home, f"file://{private.as_posix()}")

    (row,) = run_as(ALICE, _hub_preview, parquet)

    assert not _is_image_payload(row["image"])
