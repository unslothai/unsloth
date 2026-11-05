# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A scoped job must not complete with no weights when snapshot_download returns an existing folder."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.workers import hf_download


FILES = ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"]
REPO = "black-forest-labs/FLUX.1-dev"
COMMIT = "c" * 40


def _offline_worker(monkeypatch, tmp_path):
    """Metadata lookups all fail and the Hub cache is a tmp dir."""
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no net")),
    )
    monkeypatch.setattr(hf_download, "_protected_blob_hashes", lambda: frozenset())
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *a, **k: None)

    import hub.utils.download_registry as registry_mod

    monkeypatch.setattr(registry_mod, "prepare_cache_for_transport", lambda *a, **k: 0)

    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    from hub.utils.snapshot_reuse import repo_cache_dir

    return repo_cache_dir("model", REPO)


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    """An earlier download left refs/main and its snapshot: offline, the worker pins that commit (#16422)."""
    repo_dir = _offline_worker(monkeypatch, tmp_path)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text(COMMIT, encoding = "utf-8")
    snapshot = repo_dir / "snapshots" / COMMIT
    snapshot.mkdir(parents = True)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: str(snapshot))
    return snapshot


def _run(scope = "diffusion"):
    hf_download._download_scoped_snapshot(REPO, scope, list(FILES), None, "http")


def test_offline_with_no_cached_revision_refuses_before_downloading(monkeypatch, tmp_path):
    """No metadata and no refs/main: there is no immutable revision to pin, so nothing is fetched."""
    _offline_worker(monkeypatch, tmp_path)
    calls = []

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: calls.append(k))
    with pytest.raises(OSError, match = "immutable revision"):
        _run()
    assert calls == []


def test_offline_scoped_download_fails_when_the_files_are_not_on_disk(offline, capsys):
    (offline / "model_index.json").write_text("{}", encoding = "utf-8")

    with pytest.raises(SystemExit) as exit_info:
        _run()
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "incomplete" in err
    assert "diffusion_pytorch_model.safetensors" in err


def test_offline_scoped_download_passes_when_every_file_is_present(offline):
    for rel in FILES:
        path = offline / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text("weights", encoding = "utf-8")

    _run()


def test_a_dangling_symlink_does_not_count_as_present(offline, capsys):
    """Cache entries are symlinks into blobs/; a broken one is a missing file."""
    (offline / "model_index.json").write_text("{}", encoding = "utf-8")
    target = offline / "transformer"
    target.mkdir(parents = True, exist_ok = True)
    (target / "diffusion_pytorch_model.safetensors").symlink_to(offline / "gone.bin")

    with pytest.raises(SystemExit) as exit_info:
        _run()
    assert exit_info.value.code == 1
    assert "incomplete" in capsys.readouterr().err


def test_disk_space_refusal_reports_decimal_gigabytes(monkeypatch, tmp_path, capsys):
    import hub.utils.download_registry as registry_mod
    import hub.utils.hf_cache_state as cache_state_mod

    monkeypatch.setattr(registry_mod, "existing_blob_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(cache_state_mod, "hf_cache_root", lambda **k: tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(free = 1_500_000_000))
    q8 = SimpleNamespace(size = 1_834_426_944, sha256 = "a" * 64)

    with pytest.raises(SystemExit) as exit_info:
        hf_download._preflight_disk_space("model", "unsloth/Qwen3-1.7B-GGUF", [q8])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "need about 1.8 GB free" in err
    assert "only 1.5 GB is available" in err
