# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An explicit padding_free = True that the trainer turns off names its blocker, like packing = True."""

import logging
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


def _init(monkeypatch, caplog, packing, padding_free):
    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.setenv("UNSLOTH_PP_SIZE", "2")
    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    config = SimpleNamespace(packing = packing, padding_free = padding_free, max_length = 64)
    with caplog.at_level(logging.WARNING, logger = trainer_module.logger.name):
        module.SFTTrainer(model = _TakesKwargs(), args = config)
    return config


@pytest.mark.parametrize(
    "packing, padding_free, flag", [(False, True, "padding_free"), (True, None, "packing")]
)
def test_explicit_request_turned_off_under_pipeline_parallel_is_named(
    monkeypatch, caplog, packing, padding_free, flag
):
    config = _init(monkeypatch, caplog, packing, padding_free)
    assert config.packing is False and config.padding_free is False
    assert f"{flag}=True ignored (pipeline parallel" in caplog.text, caplog.text


def test_auto_padding_free_turned_off_stays_quiet(monkeypatch, caplog):
    config = _init(monkeypatch, caplog, False, None)
    assert config.padding_free is False
    assert "ignored" not in caplog.text, caplog.text
