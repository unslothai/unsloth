# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""keep_for_training auto mode: FastModel (generic) decode graphs and their StaticCache are released for the
training step; UNSLOTH_DECODE_GRAPHS_KEEP_FOR_TRAINING=1 still keeps them, FastLanguageModel graphs unchanged."""

import importlib.util
import types
from pathlib import Path

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_decode_graphs.py"
ENV = "UNSLOTH_DECODE_GRAPHS_KEEP_FOR_TRAINING"
GB = 1 << 30


@pytest.fixture
def G(monkeypatch):
    spec = importlib.util.spec_from_file_location("_unsloth_decode_graphs_release", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv(ENV, raising = False)
    # Measured step ran without the graphs and plenty is free: auto would keep FastLanguageModel graphs.
    monkeypatch.setattr(module, "_held", lambda model: (GB, torch.device("cuda", 0)))
    monkeypatch.setattr(module, "_optimizer_steps", lambda: 1)
    monkeypatch.setattr(module, "_allocator", lambda device: (2 * GB, 0, 3 * GB))
    monkeypatch.setattr(module, "_free_bytes", lambda device: 40 * GB)
    monkeypatch.setattr(module, "_sleepable", lambda model: False)
    return module


def _model(attr):
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.__dict__[attr] = types.SimpleNamespace(graphs = {(8,): object()}, released_at = 0)
    return model


def test_auto_releases_fastmodel_graphs(G):
    assert G.keep_for_training(_model("_unsloth_generic_decode_graphs")) is False
    assert G.KEEP_DECISIONS["released"] >= 1


def test_env_1_keeps_fastmodel_graphs(G, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert G.keep_for_training(_model("_unsloth_generic_decode_graphs")) is True


def test_auto_still_keeps_fast_language_model_graphs_that_fit(G):
    assert G.keep_for_training(_model("_unsloth_decode_graphs")) is True
