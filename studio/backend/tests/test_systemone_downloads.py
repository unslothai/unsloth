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


@pytest.mark.parametrize("version", ["5.4.0", None, "5.5.0"])
def test_auto_fallback_requires_supported_transformers(checkpoints, monkeypatch, version):
    from importlib.metadata import PackageNotFoundError

    def installed(package):
        assert package == "transformers"
        if version is None:
            raise PackageNotFoundError(package)
        return version

    monkeypatch.setattr("importlib.metadata.version", installed)
    torch, native, cache = checkpoints
    cache(native)
    supported = version == "5.5.0"
    assert runtime.accepts_images(torch) is supported
    assert runtime.select_checkpoint(torch, preference = "auto")[0] == native
    plan = resolve_systemone_download(model = torch.name, backend = "auto")
    assert plan.repo == (torch.source if supported else native.source)
    systemone_settings.validate(enabled = True, model = torch.name, backend = "auto")
    for kwargs in (
        {"images": True},
        {"questions": {"q": {"instructions": ""}}},
        {"questions": {"q": {"type": "score", "criteria": ["only"]}}},
    ):
        if supported:
            assert runtime.select_checkpoint(torch, preference = "auto", **kwargs)[0] == torch
        else:
            with pytest.raises(runtime.Unavailable, match = "Transformers 5.5.0"):
                runtime.select_checkpoint(torch, preference = "auto", **kwargs)
    if not supported:
        with pytest.raises(ValueError, match = "Transformers 5.5.0"):
            systemone_settings.validate(enabled = True, model = torch.name, backend = "pytorch")


@pytest.mark.parametrize("phase", [None, "loaded_model", "loading_model"])
def test_live_laya_status_keeps_cache_protected(monkeypatch, phase):
    from core.systemone import laya_runtime
    from hub.services.models.deletion import _decisions_blocks_delete

    failed = {"loaded_model": None, "loading_model": None, "error": "Clef load failed"}
    live = {"loaded_model": None, "loading_model": None, "error": None}
    name = "laya-multilingual"
    if phase:
        live[phase] = name
    monkeypatch.setattr(owned_runtime, "status", lambda: failed)
    monkeypatch.setattr(laya_runtime, "status", lambda: live)
    expected = live if phase else {**failed, "fallback_reason": runtime._fallback_reason}
    assert runtime.status() == expected
    if phase == "loaded_model":
        assert _decisions_blocks_delete(catalog.CHECKPOINTS[name].source) is not None
