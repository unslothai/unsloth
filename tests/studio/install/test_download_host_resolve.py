"""Routing of iter_resolved_published_releases between the download-host fast
path and the GitHub API enumeration; all I/O monkeypatched."""

import importlib.util
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_dlhost", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

FORK_REPO = ILP.DEFAULT_PUBLISHED_REPO
PrebuiltFallback = ILP.PrebuiltFallback


def _api_raises(*_a, **_k):
    raise AssertionError("GitHub API enumeration was used")


def _fast_path_raises(_repo):
    raise AssertionError("download-host fast path was used")


def _empty_api(*_a, **_k):
    return iter(())


def _resolve(tag = "latest", **kw):
    return list(ILP.iter_resolved_published_releases(tag, FORK_REPO, "", **kw))


def test_fast_path_yields_latest_without_api(monkeypatch):
    sentinel = object()
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", lambda _repo: sentinel)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _api_raises)
    assert _resolve() == [sentinel]


def test_fast_path_disabled_by_caller_uses_api(monkeypatch):
    # macOS passes allow_download_host_fast_path = False to keep the walk-back.
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", _fast_path_raises)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _empty_api)
    with pytest.raises(PrebuiltFallback):
        _resolve(allow_download_host_fast_path = False)


def test_fast_path_disabled_by_env_uses_api(monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", "1")
    monkeypatch.setattr(ILP, "_download_host_resolved_release", _fast_path_raises)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _empty_api)
    with pytest.raises(PrebuiltFallback):
        _resolve()


def test_fast_path_skipped_for_non_latest_request(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", _fast_path_raises)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _empty_api)
    with pytest.raises(PrebuiltFallback):
        _resolve(tag = "b9964")


def test_fast_path_none_falls_back_to_api(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", lambda _repo: None)
    used = {"api": False}

    def _api(*_a, **_k):
        used["api"] = True
        return iter(())

    monkeypatch.setattr(ILP, "iter_published_release_bundles", _api)
    with pytest.raises(PrebuiltFallback):
        _resolve()
    assert used["api"]


def test_fast_path_rejected_checksum_falls_back_to_api(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)

    def _reject(_repo):
        raise PrebuiltFallback("checksum mismatch")

    monkeypatch.setattr(ILP, "_download_host_resolved_release", _reject)
    used = {"api": False}

    def _api(*_a, **_k):
        used["api"] = True
        return iter(())

    monkeypatch.setattr(ILP, "iter_published_release_bundles", _api)
    with pytest.raises(PrebuiltFallback):
        _resolve()
    assert used["api"]
