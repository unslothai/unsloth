"""GRPO logit-scaling helpers must read config through DDP wrappers."""

from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read_source() -> str:
    with open(SOURCE_PATH, "r") as fh:
        return fh.read()


def test_grpo_logit_scaling_uses_model_config_helper():
    src = _read_source()
    # Helper exists and unwraps DDP/Accelerate wrappers via `.module`.
    assert "def _unsloth_get_model_config(model):" in src
    assert 'getattr(model.module, "config", None)' in src
    # Softcapping takes the model and tolerates a missing config.
    assert "logit_softcapping = _unsloth_get_final_logit_softcapping(model)" in src
    assert (
        "if config is None:" in src.split("def _unsloth_get_final_logit_softcapping")[1]
    )
    # Logit scale/divide read through the unwrapped config, not bare model.config.
    assert 'getattr(model_config, "logit_scale", 0)' in src
    assert 'getattr(model_config, "logits_scaling", 0)' in src
    assert src.count("model_config = _unsloth_get_model_config(model)") >= 2
    # Helper source is injected into the compiled GRPO trainer.
    assert "inspect.getsource(_unsloth_get_model_config)" in src
    # No direct model.config access remains in the RL logit path.
    assert "model.config" not in src
