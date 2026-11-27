# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The Inkling 2 merge copies tensors from the files the base checkpoint's weight index names; an index
naming a file outside the checkpoint folder must be refused before any tensor is read."""

import importlib.util
import json
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
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
    spec = importlib.util.spec_from_file_location(
        "_unsloth_inkling2_paths", _ROOT / "unsloth" / "models" / "inkling2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(tmp_path, shard):
    source = tmp_path / "base"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": shard}}))
    model = SimpleNamespace(config = SimpleNamespace(_name_or_path = str(source)))
    model._weight_conversions = [object()]
    return model


@pytest.mark.parametrize(
    "shard", ["../victim/model.safetensors", "/etc/victim.safetensors", "..\\victim.safetensors"]
)
def test_index_naming_a_file_outside_the_checkpoint_is_refused(monkeypatch, tmp_path, shard):
    module = _inkling2(monkeypatch)
    with pytest.raises(RuntimeError, match = "outside its folder"):
        module.save_inkling2_merged_16bit(
            _checkpoint(tmp_path, shard), None, str(tmp_path / "merged")
        )


def test_plain_shard_names_pass_the_index_check(monkeypatch, tmp_path):
    module = _inkling2(monkeypatch)
    # Gets past the index check and fails later on the stand-in model, not with the refusal.
    with pytest.raises(Exception) as caught:
        module.save_inkling2_merged_16bit(
            _checkpoint(tmp_path, "model-00001-of-00002.safetensors"),
            None,
            str(tmp_path / "merged"),
        )
    assert "outside its folder" not in str(caught.value)
