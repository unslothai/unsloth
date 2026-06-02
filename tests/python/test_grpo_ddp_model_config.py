"""GRPO logit-scaling helpers must read config through DDP wrappers."""

from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read_source() -> str:
    with open(SOURCE_PATH, "r") as fh:
        return fh.read()


def test_grpo_logit_scaling_uses_model_config_helper():
    src = _read_source()
    assert "def _unsloth_get_model_config(model):" in src
    assert "_cfg = _unsloth_get_model_config(model)" in src
    assert "logit_softcapping = _unsloth_get_final_logit_softcapping(_cfg)" in src
    assert 'getattr(_cfg, "logit_scale", 0)' in src
    assert 'getattr(_cfg, "logits_scaling", 0)' in src
    assert src.count("_unsloth_get_model_config(model)") >= 2
    assert "inspect.getsource(_unsloth_get_model_config)" in src
    assert "if config is None:" in src.split("def _unsloth_get_final_logit_softcapping")[1]
    assert "model.config" not in src
