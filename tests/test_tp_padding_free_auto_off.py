# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel refuses packed / padding-free rows (tp x packing), so SFT never turns padding-free on by
itself under it: packing = False must be enough, as the refusal's alternative says."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

trainer_module = pytest.importorskip("unsloth.trainer")
if not hasattr(trainer_module, "_patch_sft_trainer_auto_packing"):
    pytest.skip("unsloth.trainer is a shim here", allow_module_level = True)


class _TakesKwargs(nn.Module):
    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return None


class _StubSFTTrainer:
    def __init__(
        self,
        model = None,
        args = None,
        **kwargs,
    ):
        self.model = model
        self.args = args


def _init(monkeypatch, tp, packing, padding_free):
    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)
    for name in ("UNSLOTH_RETURN_LOGITS", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE", "UNSLOTH_PARALLEL"):
        monkeypatch.delenv(name, raising = False)
    if tp is None:
        monkeypatch.delenv("UNSLOTH_TP_SIZE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_TP_SIZE", str(tp))
    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    config = SimpleNamespace(packing = packing, padding_free = padding_free, max_length = 64)
    module.SFTTrainer(model = _TakesKwargs(), args = config)
    return config


def test_auto_padding_free_stays_off_under_tensor_parallel(monkeypatch):
    control = _init(monkeypatch, None, False, None)
    if control.padding_free is not True:
        pytest.skip(
            "this setup does not turn padding-free on automatically, so there is nothing to keep off"
        )
    config = _init(monkeypatch, 2, False, None)
    assert not config.padding_free and not config.packing, config


def test_explicit_padding_free_reaches_the_tensor_parallel_refusal(monkeypatch):
    # An explicit request is not silently dropped: the capability check refuses it with the alternative.
    config = _init(monkeypatch, 2, False, True)
    assert config.padding_free is True, config
