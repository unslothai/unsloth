# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A DDP rank holding its whole bnb model on one card gets no dispatch hooks: on ranks >= 1 they broke the
non-reentrant checkpoint recompute (#3459). A single process on a non-default card (#5068) keeps them."""

import types

import pytest
import torch

pytest.importorskip("accelerate")


def _run(monkeypatch, devices, distributed):
    import accelerate
    from unsloth.models import vision
    from unsloth.models import loader_utils

    calls = []
    monkeypatch.setattr(accelerate, "dispatch_model", lambda model, **kw: calls.append(kw))
    monkeypatch.setattr(vision, "is_distributed", lambda: distributed)
    monkeypatch.setattr(vision, "_repair_dispatch_hooks", lambda model: 0)
    monkeypatch.setattr(loader_utils, "no_placement_tensor_names", lambda model: set())
    monkeypatch.setattr(
        vision,
        "_infer_device_map_from_loaded_model",
        lambda model, skip = None: {f"layer{i}": d for i, d in enumerate(devices)},
    )
    params = [(f"layer{i}.weight", types.SimpleNamespace(device = d)) for i, d in enumerate(devices)]
    model = types.SimpleNamespace(
        is_loaded_in_4bit = True, hf_device_map = None, named_parameters = lambda: iter(params)
    )
    vision._attach_bnb_multidevice_hooks(model, True, False, False, False)
    return calls


def test_a_distributed_rank_on_its_own_card_gets_no_hooks(monkeypatch):
    assert _run(monkeypatch, [torch.device("cuda", 1)] * 2, distributed = True) == []


def test_a_single_process_on_a_non_default_card_keeps_its_hooks(monkeypatch):
    assert len(_run(monkeypatch, [torch.device("cuda", 1)] * 2, distributed = False)) == 1


def test_a_model_split_across_cards_keeps_its_hooks_under_a_distributed_launch(monkeypatch):
    calls = _run(monkeypatch, [torch.device("cuda", 0), torch.device("cuda", 1)], distributed = True)
    assert len(calls) == 1
