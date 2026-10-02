# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL >= 1.7 enables the router aux loss for any config with output_router_logits; a dense one (0 experts,
e.g. granite-4.0-h-350m) has no router logits and crashes. Unsloth turns it back off only for those, offline."""

from __future__ import annotations

import types

import pytest

from tests.version_compat._rl_anchors import RL_PY, _rl_py_constant

transformers = pytest.importorskip("transformers")


def _trainer_after_trl_init(config):
    # One namespace: rl.py injects this into __init__, and on Python < 3.12 a comprehension in exec
    # with split globals/locals cannot see `_text_config`.
    # What TRL 1.7+ leaves behind for any config carrying output_router_logits with the default coef.
    config.get_text_config().output_router_logits = True
    self = types.SimpleNamespace(aux_loss_enabled = True, model = types.SimpleNamespace(config = config))
    exec(_rl_py_constant("_DENSE_ROUTER_AUX_LOSS_OFF"), {"self": self})
    return self


def test_dense_granite_hybrid_turns_the_aux_loss_off():
    config = transformers.GraniteMoeHybridConfig(num_local_experts = 0, num_experts_per_tok = 0)
    trainer = _trainer_after_trl_init(config)
    assert trainer.aux_loss_enabled is False and config.output_router_logits is False


@pytest.mark.parametrize(
    "config",
    [
        lambda: transformers.GraniteMoeHybridConfig(num_local_experts = 4, num_experts_per_tok = 2),
        lambda: transformers.Qwen3MoeConfig(),
        lambda: transformers.MixtralConfig(),
    ],
    ids = ["granite_hybrid_moe", "qwen3_moe", "mixtral"],
)
def test_real_moe_keeps_the_aux_loss(config):
    config = config()
    trainer = _trainer_after_trl_init(config)
    assert trainer.aux_loss_enabled is True and config.output_router_logits is True


def test_trainer_without_aux_loss_is_untouched():
    self = types.SimpleNamespace(model = None)
    exec(_rl_py_constant("_DENSE_ROUTER_AUX_LOSS_OFF"), {"self": self})
    assert not hasattr(self, "aux_loss_enabled")


def test_rl_py_appends_it_after_trl_init():
    src = RL_PY.read_text("utf-8")
    assert "RLTrainer_post += _DENSE_ROUTER_AUX_LOSS_OFF" in src
