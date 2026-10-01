# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed preview rows can carry Hugging Face image cells that point at a file path.

The path is row data, so a managed account must not get the pixels of a file outside its own
workspace back as a preview. The owner, and paths inside the account's workspace, still preview.
"""

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
