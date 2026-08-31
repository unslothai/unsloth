# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scoped download worker must never report success without the files.

``snapshot_download`` returns an existing snapshot folder -- having fetched nothing -- when
its own ``repo_info`` call fails, and with HF metadata unavailable no manifest is written,
so the usual verification is a no-op. A repo already on disk from a full snapshot job (which
ignores ``*.gguf``) would otherwise flip a scoped job to complete with no weights, and the
Images page auto-loads as soon as the job completes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.workers import hf_download


FILES = ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"]


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    """A worker whose metadata lookups all fail, pointed at a snapshot dir we control."""
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no net")),
    )
    monkeypatch.setattr(hf_download, "_protected_blob_hashes", lambda: frozenset())
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *a, **k: None)

    import hub.utils.download_registry as registry_mod

    monkeypatch.setattr(registry_mod, "prepare_cache_for_transport", lambda *a, **k: 0)

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: str(snapshot))
    return snapshot


def _run(scope = "diffusion"):
    hf_download._download_scoped_snapshot(
        "black-forest-labs/FLUX.1-dev", scope, list(FILES), None, "http"
    )


def test_offline_scoped_download_fails_when_the_files_are_not_on_disk(offline, capsys):
    (offline / "model_index.json").write_text("{}", encoding = "utf-8")  # the cheap file only

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

    _run()  # no SystemExit: everything the job asked for is on disk


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
