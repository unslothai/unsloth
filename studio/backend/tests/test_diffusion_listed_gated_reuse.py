# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pipeline load lists its repo with model_info once; the base access check reuses that gated=False answer
instead of asking model_info again. A gated or unknown answer still runs the full check."""

from __future__ import annotations

import ast
import json
import types
from pathlib import Path

import pytest

import core.inference.diffusion as dmod
from core.inference.diffusion import DiffusionBackend, _assert_base_repo_accessible

REPO = "unsloth/some-pipeline"


def _stub_api(monkeypatch, gated):
    calls: list = []

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
            **_kw,
        ):
            calls.append((repo_id, files_metadata))
            return types.SimpleNamespace(gated = gated, siblings = [], sha = "a" * 40)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.get_hf_file_metadata",
        lambda *a, **k: types.SimpleNamespace(etag = "x", size = 1),
    )
    return calls


def test_known_open_base_skips_model_info(monkeypatch):
    calls = _stub_api(monkeypatch, gated = False)
    assert _assert_base_repo_accessible(REPO, None, known_gated = False) is None
    assert calls == []


@pytest.mark.parametrize("known", [None, "auto", "manual"])
def test_gated_or_unknown_still_asks(monkeypatch, known):
    calls = _stub_api(monkeypatch, gated = "auto")
    _assert_base_repo_accessible(REPO, None, known_gated = known)
    assert calls == [(REPO, False)]


def test_known_gated_does_not_bypass_local_files_only(monkeypatch):
    calls = _stub_api(monkeypatch, gated = False)
    assert (
        _assert_base_repo_accessible(REPO, None, local_files_only = True, known_gated = False) is None
    )
    assert calls == []


@pytest.mark.parametrize("gated", [False, "auto", None])
def test_pipeline_listing_reports_gated(monkeypatch, tmp_path, gated):
    manifest = tmp_path / "model_index.json"
    manifest.write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }
        ),
        encoding = "utf-8",
    )
    files = [("model_index.json", 10), ("scheduler/scheduler_config.json", 20)]

    class _Api:
        def model_info(self, repo_id, **_kw):
            return types.SimpleNamespace(
                gated = gated,
                sha = "b" * 40,
                siblings = [types.SimpleNamespace(rfilename = n, size = s) for n, s in files],
            )

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(manifest))
    seen: dict = {}
    DiffusionBackend._estimate_download_bytes(
        REPO, None, REPO, None, kind = "pipeline", gated_out = seen
    )
    assert seen == {REPO.lower(): gated}


def test_run_load_threads_the_listing_into_the_access_check():
    """The load that lists the repo passes the same dict to both calls, keyed by the fetch base."""
    tree = ast.parse(Path(dmod.__file__).read_text(encoding = "utf-8"))
    estimate_kw, access_kw = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        kws = {k.arg: ast.unparse(k.value) for k in node.keywords}
        if name == "_estimate_download_bytes" and "gated_out" in kws:
            estimate_kw.append(kws["gated_out"])
        if name == "_assert_base_repo_accessible" and "known_gated" in kws:
            access_kw.append(kws["known_gated"])
    assert estimate_kw == ["listed_gated"]
    assert access_kw == ["listed_gated.get(fetch_base.lower())"]
