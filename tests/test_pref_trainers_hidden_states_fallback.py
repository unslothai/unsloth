# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Preference, RLOO and Online DPO trainers must wrap a flag-blind fused-head forward at init, as GRPO does."""

import importlib

import pytest

torch = pytest.importorskip("torch")

import unsloth  # noqa: F401,E402  (must be imported before transformers / trl)

STEMS = {
    "dpo": "DPO",
    "kto": "KTO",
    "orpo": "ORPO",
    "cpo": "CPO",
    "bco": "BCO",
    "online_dpo": "OnlineDPO",
    "xpo": "XPO",
    "nash_md": "NashMD",
    "rloo": "RLOO",
}


def _trainer_class(name):
    for module in ("trl", f"trl.experimental.{name}", f"trl.trainer.{name}_trainer"):
        try:
            found = getattr(importlib.import_module(module), f"{STEMS[name]}Trainer", None)
        except Exception:
            continue
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("name", sorted(STEMS))
def test_trainer_init_installs_hidden_states_fallback(name):
    cls = _trainer_class(name)
    if cls is None:
        pytest.skip(f"this TRL has no {STEMS[name]}Trainer")
    if not cls.__name__.startswith("Unsloth"):
        pytest.skip(f"{cls.__name__} is not patched by Unsloth on this TRL")
    assert getattr(
        cls.__init__, "_unsloth_grpo_hidden_states_init_wrapped", False
    ), f"{cls.__name__}.__init__ does not install the hidden-state forward wrapper"


def test_wrapped_forward_returns_hidden_states_for_flag_blind_model():
    """The wrapper the trainers install turns a flag-blind *ForCausalLM into one honouring the flag."""
    import os

    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    from unsloth.models.rl import _install_grpo_hidden_states_forward_wrapper

    config = Qwen2Config(
        num_hidden_layers = 1,
        hidden_size = 32,
        intermediate_size = 64,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 96,
        max_position_embeddings = 32,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).eval()
    ids = torch.randint(0, 96, (2, 7))
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        with torch.no_grad():
            assert model(ids).logits.shape[-1] == 96
            _install_grpo_hidden_states_forward_wrapper(model)
            assert model(ids).logits.shape[-1] == 32
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior
