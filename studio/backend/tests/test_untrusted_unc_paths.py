# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paths picked by a Hub repo, a dataset row or an unauthenticated URL must not reach a filesystem
call as a Windows network share: touching one makes Windows sign in to that host (NTLM)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from utils.datasets import remote_images

UNC = [r"\\evil.example\share\x", "//evil.example/share/x", r"\\?\UNC\evil.example\s\x"]


@pytest.fixture(autouse = True)
def _windows(monkeypatch):
    # Windows path rules on this host; no Windows runner here.
    monkeypatch.setattr(remote_images, "_WINDOWS", True)
    monkeypatch.delenv(remote_images.ALLOW_PRIVATE_ENV, raising = False)


def _shared(value) -> bool:
    return "evil.example" in str(value)


@pytest.mark.parametrize("base", UNC)
def test_remote_adapter_base_on_a_share_is_dropped(monkeypatch, tmp_path, base):
    from utils.models import model_config
    import huggingface_hub
    import utils.hf_probe as hf_probe

    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": base}))
    monkeypatch.setattr(model_config, "is_local_path", lambda value: False)
    monkeypatch.setattr(hf_probe, "hf_file_definitely_absent", lambda *a, **k: False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(cfg))
    assert model_config.get_base_model_from_lora_identifier("someone/lora") is None


@pytest.mark.parametrize("base", UNC)
def test_activation_tier_never_probes_a_share_base(monkeypatch, base):
    from utils import transformers_version as tv

    monkeypatch.setattr(tv, "_env_offline", lambda: False)
    monkeypatch.setattr(tv, "_hf_json", lambda url, token: {"base_model_name_or_path": base})
    assert tv._remote_lora_base("someone/lora") is None
    seen = []
    monkeypatch.setattr(
        tv, "get_transformers_tier", lambda name, token = None: seen.append(name) or "default"
    )
    tv.get_transformers_activation_tier("someone/lora")
    assert not any(_shared(name) for name in seen), seen


def test_cached_adapter_base_on_a_share_is_dropped(monkeypatch, tmp_path):
    from utils import transformers_version as tv

    snap = tmp_path / "models--someone--lora" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    (snap / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": UNC[0]}))

    class Paths:
        hub_cache = tmp_path

    monkeypatch.setattr(tv, "get_hf_cache_paths", lambda: Paths)
    assert tv._adapter_base_from_hf_cache("someone/lora") is None


@pytest.mark.parametrize("raw", UNC)
def test_dataset_path_on_a_share_is_refused_before_realpath(monkeypatch, raw):
    from hub.utils import paths as hub_paths
    from utils.paths import storage_roots

    touched = []
    real = os.path.realpath
    monkeypatch.setattr(
        os.path, "realpath", lambda p, *a, **k: touched.append(p) or real(p, *a, **k)
    )
    for resolve in (hub_paths.resolve_dataset_path, storage_roots.resolve_dataset_path):
        with pytest.raises(ValueError):
            resolve(raw)
    assert not any(_shared(p) for p in touched), touched


@pytest.mark.parametrize("path", UNC + ["zip://a.wav::" + UNC[1]])
def test_path_only_media_cell_on_a_share_is_not_opened(path):
    from utils.security import dataset_media

    opened = []
    decode = dataset_media.guard_decoder(lambda self, value, *a, **k: opened.append(value))
    with pytest.raises(dataset_media.MediaPathRefused):
        decode(None, {"path": path, "bytes": None})
    assert opened == []
    decode(None, {"path": path, "bytes": b"RIFF"})
    decode(None, {"path": "clips/a.wav", "bytes": None})
    assert len(opened) == 2


@pytest.mark.parametrize("path", UNC)
def test_recipe_preview_image_on_a_share_is_not_opened(monkeypatch, path):
    from PIL import Image

    from core.data_recipe import jsonable

    opened = []
    monkeypatch.setattr(Image, "open", lambda p, *a, **k: opened.append(p))
    assert jsonable._to_pil_from_hf_image_dict({"path": path, "bytes": None}) is None
    assert opened == []


@pytest.mark.parametrize("source", UNC)
def test_modular_index_share_component_is_not_probed(monkeypatch, source):
    from core.inference import media_locality

    touched = []
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: touched.append(str(self)) or is_dir(self))
    assert media_locality._hosted_component_cached(source, "", "", "") is False
    assert not any(_shared(p) for p in touched), touched


@pytest.mark.parametrize("asset", ["//evil.example/share/x.png", r"\\evil.example\share\x.png"])
def test_public_preview_asset_never_resolves_a_share(monkeypatch, asset):
    from routes import preview

    touched = []
    resolve = Path.resolve
    monkeypatch.setattr(
        Path, "resolve", lambda self, *a, **k: touched.append(str(self)) or resolve(self, *a, **k)
    )
    with pytest.raises(HTTPException) as info:
        asyncio.run(preview.preview_asset(asset))
    assert info.value.status_code == 404
    assert not any(_shared(p) for p in touched), touched
