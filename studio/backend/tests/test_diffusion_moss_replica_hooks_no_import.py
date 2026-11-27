# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every CFG replica build calls the MOSS-Image record and rebuild hooks, whatever the model. They must not import
diffusion_moss_image (which needs diffusers at import time): when that import failed, every replica build failed
with it, so CFG parallelism was off for all models on a host without a working diffusers."""

import importlib.abc
import sys
import types

import pytest

from core.inference import diffusion_moss_fused as moss_fused

_MOSS = moss_fused.__name__.rpartition(".")[0] + ".diffusion_moss_image"


class _RefuseMossImage(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.asked = []

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if name == _MOSS or name == "diffusers" or name.startswith("diffusers."):
            self.asked.append(name)
            raise ModuleNotFoundError(f"No module named {name!r}", name = name)
        return None


@pytest.fixture
def no_moss_module(monkeypatch):
    monkeypatch.delitem(sys.modules, _MOSS, raising = False)
    finder = _RefuseMossImage()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    return finder


def test_hooks_answer_without_importing_moss_image(no_moss_module):
    assert moss_fused.is_replica_record("_unsloth_moss_fused_qkv", object()) is False
    assert moss_fused.is_replica_record("anything", object()) is False
    assert moss_fused.rebuild_replica(types.SimpleNamespace(), types.SimpleNamespace()) == 0
    assert no_moss_module.asked == [], "a replica of a non-MOSS model imported diffusion_moss_image"
    assert _MOSS not in sys.modules


def test_a_loaded_moss_image_module_still_names_the_record(monkeypatch):
    fake = types.SimpleNamespace(FUSED_QKV_ATTR = "_fused_record")
    monkeypatch.setitem(sys.modules, _MOSS, fake)
    assert moss_fused.is_replica_record("_fused_record", object()) is True
    assert moss_fused.is_replica_record("other", object()) is False
