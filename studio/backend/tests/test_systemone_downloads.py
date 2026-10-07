# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.systemone import catalog, native_worker, owned_runtime, runtime
from routes.settings import resolve_systemone_download
from utils import systemone_settings


@pytest.fixture(params = ["clef-flash", "clef"])
def checkpoints(request, monkeypatch, tmp_path):
    torch = catalog.CHECKPOINTS[request.param]
    native = catalog.NATIVE_CHECKPOINTS[request.param]
    roots = {c.source: tmp_path / c.source.replace("/", "--") for c in (torch, native)}
    monkeypatch.setattr(systemone_settings, "get_backend", lambda: "auto")
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: None)
    monkeypatch.setattr(native_worker, "native_availability", lambda: {"available": True})

    def snapshot(repo, **kwargs):
        assert kwargs["local_files_only"] is True
        return str(roots[repo])

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    def cache(checkpoint):
        for file in checkpoint.files:
            path = roots[checkpoint.source] / file
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_bytes(b"fixture")

    return torch, native, cache


@pytest.mark.parametrize("preference", [None, "auto"])
def test_auto_exposes_missing_fallback_before_reporting_cached(checkpoints, preference):
    torch, native, cache = checkpoints

    def resolve():
        return resolve_systemone_download(model = torch.name, backend = preference)

    plan = resolve()
    assert plan.repo == native.source and not plan.cached
    cache(native)
    plan = resolve()
    assert plan.repo == torch.source and not plan.cached
    assert plan.files == list(torch.files)
    assert plan.revision == torch.revision and plan.size_bytes == torch.download_bytes
    cache(torch)
    assert resolve().cached
    for kwargs in (
        {"images": True},
        {"questions": {"q": {"instructions": ""}}},
        {"questions": {"q": {"type": "score", "criteria": ["only"]}}},
    ):
        selected, _ = runtime.select_checkpoint(torch, preference = "auto", **kwargs)
        assert selected == torch and owned_runtime.is_cached(selected)


@pytest.mark.parametrize(
    "preference,stored,torch_available,expected",
    [
        (None, "llama.cpp", True, "native"),
        ("llama.cpp", "auto", True, "native"),
        ("pytorch", "auto", True, "torch"),
        (None, "auto", False, "native"),
    ],
)
def test_explicit_and_no_torch_plans_do_not_download_extra_assets(
    checkpoints, monkeypatch, preference, stored, torch_available, expected
):
    torch, native, cache = checkpoints
    cache(native)
    monkeypatch.setattr(systemone_settings, "get_backend", lambda: stored)
    monkeypatch.setattr(
        systemone_settings,
        "runtime_unavailable_reason",
        lambda: None if torch_available else "PyTorch is not installed",
    )
    plan = resolve_systemone_download(model = torch.name, backend = preference)
    assert plan.repo == (native.source if expected == "native" else torch.source)
    assert plan.cached == (expected == "native")


def test_auto_without_native_only_plans_pytorch(checkpoints, monkeypatch):
    torch, _, _ = checkpoints
    monkeypatch.setattr(
        native_worker, "native_availability", lambda: {"available": False, "reason": "old binary"}
    )
    plan = resolve_systemone_download(model = torch.name, backend = "auto")
    assert plan.repo == torch.source and not plan.cached
