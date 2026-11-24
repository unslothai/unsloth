# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CSM dataset step loaded its processor by the bare repo id, so transformers fetched and imported
whatever processor code the repo's main named by then, past the snapshot the consent gate scanned and the
tokenizer loaded. It must load the same pinned source."""

import pytest
import transformers

from core.training import trainer as tmod


class _Stop(Exception):
    pass


@pytest.fixture(autouse = True)
def _cuda_trainer(monkeypatch):
    # On Apple Silicon UnslothTrainer() hands back the MLX adapter, which has no CSM path.
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda: False)


def test_csm_processor_loads_the_pinned_snapshot(monkeypatch):
    seen = {}

    def from_pretrained(source, **kwargs):
        seen.update(source = source, **kwargs)
        raise _Stop

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    t = tmod.UnslothTrainer()
    t.model_name = "mallory/voice"
    t.trust_remote_code = True
    t._remember_model_source("/cache/models--mallory--voice/snapshots/abc", "abc", True)
    with pytest.raises(_Stop):
        t._preprocess_csm_dataset(None)
    assert seen["source"] == "/cache/models--mallory--voice/snapshots/abc"
    assert seen["revision"] == "abc"
    assert seen["local_files_only"] is True


def test_csm_processor_without_a_recorded_source_uses_the_model_name(monkeypatch):
    seen = {}

    def from_pretrained(source, **kwargs):
        seen.update(source = source, **kwargs)
        raise _Stop

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    t = tmod.UnslothTrainer()
    t.model_name = "org/csm"
    with pytest.raises(_Stop):
        t._preprocess_csm_dataset(None)
    assert seen["source"] == "org/csm" and seen["revision"] is None
