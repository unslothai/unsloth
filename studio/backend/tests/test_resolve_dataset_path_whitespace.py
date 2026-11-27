# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local dataset path pasted with stray whitespace resolves like the same path without it."""

import pytest

from utils.paths import storage_roots


@pytest.fixture(autouse = True)
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    return tmp_path / "studio"


@pytest.mark.parametrize(
    "spelling",
    [
        "uploads/data.jsonl",
        " uploads/data.jsonl",
        "uploads/data.jsonl ",
        "\tassets/datasets/uploads/data.jsonl\n",
        " recipes/data.jsonl ",
        " data.jsonl ",
    ],
)
def test_a_padded_relative_path_resolves_like_the_trimmed_one(spelling):
    assert storage_roots.resolve_dataset_path(spelling) == storage_roots.resolve_dataset_path(
        spelling.strip()
    )


def test_a_padded_relative_path_finds_the_file_it_names():
    uploads = storage_roots.dataset_uploads_root()
    uploads.mkdir(parents = True)
    (uploads / "data.jsonl").write_text("{}")
    assert storage_roots.resolve_dataset_path("  uploads/data.jsonl  ").exists()
