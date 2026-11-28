# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""push_to_hub=True stages the merged checkpoint under tempfile.mkdtemp; a failed merge or a
failed upload must not leave that staging directory behind."""

import contextlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
from types import SimpleNamespace

import huggingface_hub
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("unsloth_zoo.saving_utils")

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _inkling2(monkeypatch):
    loading = types.ModuleType("transformers.core_model_loading")
    for name in (
        "Inkling2LoopDequantizeExperts",
        "Interleave",
        "WeightConverter",
        "WeightRenaming",
    ):
        setattr(loading, name, type(name, (), {}))
    loading.rename_source_key = lambda key, renamings, converters: (key, None)
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", loading)

    unsloth_pkg = types.ModuleType("unsloth")
    unsloth_models = types.ModuleType("unsloth.models")
    unsloth_utils = types.ModuleType("unsloth.models._utils")
    unsloth_utils.lora_relative_to_original_base = contextlib.nullcontext
    unsloth_save = types.ModuleType("unsloth.save")
    unsloth_save._refuse_private_push_to_public_repo = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_pkg)
    monkeypatch.setitem(sys.modules, "unsloth.models", unsloth_models)
    monkeypatch.setitem(sys.modules, "unsloth.models._utils", unsloth_utils)
    monkeypatch.setitem(sys.modules, "unsloth.save", unsloth_save)

    spec = importlib.util.spec_from_file_location(
        "_unsloth_inkling2_staging", _ROOT / "unsloth" / "models" / "inkling2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Api:
    def __init__(self, upload_error = None):
        self.upload_error = upload_error

    def create_repo(self, repo_id, **kw):
        return SimpleNamespace(repo_id = repo_id)

    def list_repo_files(self, *a, **kw):
        return []

    def upload_folder(self, *a, **kw):
        if self.upload_error is not None:
            raise self.upload_error


def _model(source):
    model = SimpleNamespace(
        config = SimpleNamespace(_name_or_path = str(source)),
        named_modules = lambda: iter([]),
        named_parameters = lambda: iter([]),
    )
    model._weight_conversions = [object()]
    return model


def _staged_dirs(monkeypatch):
    made = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking(*a, **kw):
        path = real_mkdtemp(*a, **kw)
        made.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking)
    return made


def test_failed_merge_drops_the_staging_dir(monkeypatch, tmp_path):
    module = _inkling2(monkeypatch)
    made = _staged_dirs(monkeypatch)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **kw: _Api())

    source = tmp_path / "base"
    source.mkdir()
    # The index names a shard that does not exist, so the merge dies inside safe_open.
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00002.safetensors"}})
    )

    with pytest.raises(Exception):
        module.save_inkling2_merged_16bit(
            _model(source), None, "user/repo", push_to_hub = True, token = "tok"
        )
    assert made, "expected a mkdtemp staging dir"
    assert not os.path.exists(made[0]), "staging dir was left behind"


def test_failed_upload_drops_the_staging_dir(monkeypatch, tmp_path):
    import torch
    from safetensors.torch import save_file

    module = _inkling2(monkeypatch)
    made = _staged_dirs(monkeypatch)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda *a, **kw: _Api(upload_error = RuntimeError("boom upload")),
    )

    source = tmp_path / "base"
    source.mkdir()
    save_file({"w": torch.zeros(2, 2)}, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps({"model_type": "inkling2"}))

    with pytest.raises(RuntimeError, match = "boom upload"):
        module.save_inkling2_merged_16bit(
            _model(source), None, "user/repo", push_to_hub = True, token = "tok"
        )
    assert made, "expected a mkdtemp staging dir"
    assert not os.path.exists(made[0]), "staging dir was left behind"
