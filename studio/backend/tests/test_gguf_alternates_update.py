# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cached GGUF whose blob the repo lists in unsloth_alternates.json is current: no update badge."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from hub.services.models import gguf_variants as GV
from hub.utils import gguf_alternates as GA
from hub.utils.download_manifest import ExpectedFile
from hub.utils.gguf_plan import GgufVariantPlan

NAME = "Model-UD-Q4_K_XL.gguf"
DRAFTER = "dspark-Model-Q8_0.gguf"
OLD, NEW, OTHER = "a" * 64, "b" * 64, "c" * 64
OLD_DRAFTER, NEW_DRAFTER = "d" * 64, "e" * 64


def _requirement(*expected):
    files = tuple(ExpectedFile(path = p, size = s, sha256 = h) for p, s, h in expected)
    return GgufVariantPlan(
        main_filenames = frozenset(e.path for e in files),
        target_filenames = tuple(e.path for e in files),
        main_hashes = frozenset(e.sha256 for e in files),
        required_hashes = frozenset(e.sha256 for e in files),
        companion_hashes = frozenset(),
        mmproj_filenames = frozenset(),
        mmproj_hashes = frozenset(),
        expected_files = files,
        main_size_bytes = sum(e.size for e in files),
        download_size_bytes = sum(e.size for e in files),
    )


@pytest.fixture(autouse = True)
def _fresh_cache():
    GA._cache.clear()
    yield
    GA._cache.clear()


def _serve(monkeypatch, tmp_path, text):
    calls = []
    manifest = tmp_path / GA.ALTERNATES_FILENAME
    manifest.write_text(text, encoding = "utf-8")

    def _download(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        calls.append((repo_id, filename))
        return str(manifest)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    return calls


def _listing(*names):
    return [SimpleNamespace(rfilename = n, blob_id = "f" * 40) for n in names]


def test_a_cached_alternate_blob_is_current(monkeypatch, tmp_path):
    text = json.dumps({"alternates": {NAME: [OLD], DRAFTER: [OLD_DRAFTER.upper()]}})
    calls = _serve(monkeypatch, tmp_path, text)
    alternates = GA.repo_alternate_blobs("org/repo", None, _listing(NAME, GA.ALTERNATES_FILENAME))
    assert calls == [("org/repo", GA.ALTERNATES_FILENAME)]

    requirement = _requirement((NAME, 4096, NEW), (DRAFTER, 1024, NEW_DRAFTER))
    local = {NAME: {OLD}, DRAFTER: {OLD_DRAFTER}}
    assert GV._variant_update_available_from_requirement(local, requirement, "UD-Q4_K_XL") is True
    assert (
        GV._variant_update_available_from_requirement(local, requirement, "UD-Q4_K_XL", alternates)
        is False
    )

    # The manifest is read once per listing blob, not once per request.
    GA.repo_alternate_blobs("org/repo", None, _listing(NAME, GA.ALTERNATES_FILENAME))
    assert len(calls) == 1


def test_an_unlisted_old_blob_still_reports_an_update(monkeypatch, tmp_path):
    _serve(monkeypatch, tmp_path, json.dumps({"alternates": {NAME: [OLD]}}))
    alternates = GA.repo_alternate_blobs("org/repo", None, _listing(NAME, GA.ALTERNATES_FILENAME))
    requirement = _requirement((NAME, 4096, NEW))

    assert (
        GV._variant_update_available_from_requirement(
            {NAME: {OTHER}}, requirement, "UD-Q4_K_XL", alternates
        )
        is True
    )
    # An alternate listed for another file does not cover this one.
    other = {"Other-Q8_0.gguf": frozenset({OTHER})}
    assert (
        GV._variant_update_available_from_requirement(
            {NAME: {OTHER}}, requirement, "UD-Q4_K_XL", other
        )
        is True
    )


def test_a_repo_without_the_manifest_is_never_fetched(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, tmp_path, json.dumps({"alternates": {NAME: [OLD]}}))
    assert GA.repo_alternate_blobs("org/repo", None, _listing(NAME, "README.md")) == {}
    assert GA.repo_alternate_blobs("org/repo", None, None) == {}
    assert calls == []

    requirement = _requirement((NAME, 4096, NEW))
    assert (
        GV._variant_update_available_from_requirement({NAME: {OLD}}, requirement, "UD-Q4_K_XL", {})
        is True
    )
    assert (
        GV._variant_update_available_from_requirement({NAME: {NEW}}, requirement, "UD-Q4_K_XL", {})
        is False
    )


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        json.dumps([OLD]),
        json.dumps({"alternates": [OLD]}),
        json.dumps({"alternates": {NAME: OLD}}),
        json.dumps({"alternates": {NAME: ["not-a-sha", 7]}}),
    ],
)
def test_a_malformed_manifest_is_ignored(monkeypatch, tmp_path, text):
    _serve(monkeypatch, tmp_path, text)
    alternates = GA.repo_alternate_blobs("org/repo", None, _listing(NAME, GA.ALTERNATES_FILENAME))
    assert alternates == {}
    requirement = _requirement((NAME, 4096, NEW))
    assert (
        GV._variant_update_available_from_requirement(
            {NAME: {OLD}}, requirement, "UD-Q4_K_XL", alternates
        )
        is True
    )


def test_a_failed_download_is_ignored(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)
    assert GA.repo_alternate_blobs("org/repo", None, _listing(GA.ALTERNATES_FILENAME)) == {}
