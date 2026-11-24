# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image pipeline components must not load weights from outside what the account may read:
transformers_weights in a local pipeline's component configs, and text encoder / VAE file names
that climb out of their Hub repo."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.inference.diffusion_comfy_components import ComponentFileError, parse_component_file
from hub.services.models import account_access
from utils.security.file_security import escaping_weight_references


def _pipeline(tmp_path, te_config):
    root = tmp_path / "sdxl-style"
    (root / "text_encoder").mkdir(parents = True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}))
    (root / "text_encoder" / "config.json").write_text(json.dumps(te_config))
    (root / "text_encoder" / "model.safetensors").write_bytes(b"decoy")
    return root


@pytest.mark.parametrize(
    "te_config",
    [
        {"model_type": "clip", "transformers_weights": "/srv/owner/text_encoder/model.safetensors"},
        {"model_type": "clip", "text_config": {"transformers_weights": "../../owner.safetensors"}},
    ],
)
def test_a_local_pipeline_component_cannot_name_outside_weights(tmp_path, te_config):
    from core.inference.diffusion import _refuse_escaping_component_weights
    root = _pipeline(tmp_path, te_config)
    with pytest.raises(ValueError):
        _refuse_escaping_component_weights(str(root))


def test_configuration_files_are_read_too(tmp_path):
    root = _pipeline(tmp_path, {"model_type": "clip", "configuration_files": ["config.4.0.0.json"]})
    (root / "text_encoder" / "config.4.0.0.json").write_text(
        json.dumps({"transformers_weights": "/srv/owner/x.safetensors"})
    )
    assert escaping_weight_references(root / "text_encoder")


def test_an_ordinary_pipeline_still_loads(tmp_path):
    from core.inference.diffusion import _refuse_escaping_component_weights
    root = _pipeline(tmp_path, {"model_type": "clip", "transformers_weights": "model.safetensors"})
    _refuse_escaping_component_weights(str(root), None)


def test_a_hub_component_file_cannot_climb_out_of_its_repo():
    with pytest.raises(ComponentFileError):
        parse_component_file("unsloth/Qwen3-0.6B/../../root/models/te.safetensors")
    ref = parse_component_file("unsloth/Qwen3-0.6B/split/te.safetensors")
    assert ref.repo_id == "unsloth/Qwen3-0.6B" and ref.filename == "split/te.safetensors"


def test_access_check_treats_a_relative_model_path_like_the_loader(tmp_path, monkeypatch):
    mine = tmp_path / "skills" / "mine"
    mine.mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    request = SimpleNamespace(
        model_path = "skills/mine",
        text_encoder_file = "unsloth/Qwen3-0.6B/../../../srv/owner/te.safetensors",
        vae_file = None,
    )
    assert account_access.media_component_file_references(request) == [None]


def test_the_forward_slash_spelling_of_a_folder_is_checked_too(tmp_path, monkeypatch):
    real = tmp_path / "a" / "b"
    real.mkdir(parents = True)
    (real / "config.json").write_text(
        json.dumps({"transformers_weights": "/srv/owner/x.safetensors"})
    )
    monkeypatch.chdir(tmp_path)
    # On POSIX "a\\b" is one folder name the check reads; loaders open "a/b".
    assert escaping_weight_references("a\\b")
