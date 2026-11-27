# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paths a dataset row, a cached adapter's publisher or a managed MCP caller picks must not reach a filesystem
call as a Windows network share: touching one makes Windows sign in to that host (NTLM)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parents[1]
for root in (_BACKEND_ROOT, _REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from utils.datasets import remote_images  # noqa: E402

UNC = [r"\\evil.example\share\x", "//evil.example/share/x", r"\\?\UNC\evil.example\s\x"]


@pytest.fixture(autouse = True)
def _windows(monkeypatch):
    # Windows path rules on this host; no Windows runner here.
    monkeypatch.setattr(remote_images, "_WINDOWS", True)
    monkeypatch.delenv(remote_images.ALLOW_PRIVATE_ENV, raising = False)


def _shared(value) -> bool:
    return "evil.example" in str(value)


def _record(monkeypatch, target, name):
    seen = []
    real = getattr(target, name)

    def recording(self_or_path, *args, **kwargs):
        seen.append(self_or_path)
        return real(self_or_path, *args, **kwargs)

    monkeypatch.setattr(target, name, recording)
    return seen


@pytest.mark.parametrize("cell", UNC)
def test_cli_media_sniff_never_opens_a_share_named_by_a_row(monkeypatch, cell):
    pytest.importorskip("typer")
    from unsloth_cli.commands.train import _media_columns_hold_media

    seen = _record(monkeypatch, os.path, "isfile")
    rows = [{"image": cell}]
    detected = {"multimodal_columns": ["image"], "audio_columns": []}
    assert _media_columns_hold_media(rows, detected) == (False, False)
    assert not any(_shared(path) for path in seen), seen


def test_cli_media_sniff_still_reads_a_local_file(tmp_path):
    pytest.importorskip("typer")
    from unsloth_cli.commands.train import _media_columns_hold_media

    image = tmp_path / "picture"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(8))
    detected = {"multimodal_columns": ["image"], "audio_columns": []}
    assert _media_columns_hold_media([{"image": str(image)}], detected) == (True, False)


@pytest.mark.parametrize("base", UNC)
def test_cached_adapter_listing_never_probes_a_share_base(monkeypatch, tmp_path, base):
    from hub.services.models import common

    is_dir = _record(monkeypatch, Path, "is_dir")
    exists = _record(monkeypatch, Path, "exists")
    assert common._base_transformers_can_chat(base, None, tmp_path) is None
    common._base_model_source(base, tmp_path)  # a label only; the share is never touched
    assert not any(_shared(path) for path in is_dir + exists), is_dir + exists


def test_cached_adapter_listing_still_sees_a_local_base(tmp_path):
    from hub.services.models import common

    base = tmp_path / "base"
    base.mkdir()
    assert common._base_model_source(str(base), tmp_path / "adapter") == "local"


@pytest.mark.parametrize("name", UNC)
def test_mcp_training_validation_never_probes_a_share(monkeypatch, name):
    from studio_mcp.tools import training as mcp_training
    from utils.models import model_config

    probed = []
    monkeypatch.setattr(
        model_config,
        "detect_audio_type_checked",
        lambda model, **kwargs: probed.append(model) or (None, True),
    )
    validated = SimpleNamespace(
        model_name = name, model_format = None, is_dataset_audio = False, hf_token = None
    )
    caller = SimpleNamespace(hf_token = None)
    result, note = asyncio.run(mcp_training._checked_trainable(validated, caller, name))
    assert result is validated and note is None
    assert probed == []
