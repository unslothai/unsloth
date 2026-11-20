# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The rounded-weight marker in a QAT model's state_dict must name the fake quantization recipe.

It held only a checksum of the values, so an int8-prepared model's state_dict loaded into an int4-prepared model
kept the int8-rounded weights verbatim with the int4 quantizer off, instead of what per call int4 fake
quantization of those weights gives. CPU only."""

import io
import os

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

from tests.test_qat_round_rewrite import DIM, _base, _model, _out, qr  # noqa: E402


def test_imports_this_tree():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(qr.__file__).startswith(here), qr.__file__


@pytest.mark.parametrize("source_scheme, destination_scheme", [("int8", "int4"), ("int4", "int8")])
@pytest.mark.parametrize("through_disk", [False, True])
def test_state_dict_from_another_recipe_is_rounded_for_this_one(
    source_scheme, destination_scheme, through_disk, monkeypatch
):
    x = torch.randn(4, DIM)
    source = _model(source_scheme, monkeypatch, "1", seed = 1)
    ref = _model(destination_scheme, monkeypatch, "0", seed = 2)
    destination = _model(destination_scheme, monkeypatch, "1", seed = 2)
    state = source.state_dict()
    if through_disk:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only = True)
    ref.load_state_dict(state)
    destination.load_state_dict(state)
    # Same as per call fake quantization with this model's recipe (UNSLOTH_QAT_ROUND_INPLACE=0).
    assert torch.equal(_out(destination, x), _out(ref, x))
    assert _base(destination).weight_fake_quantizer.enabled is False
