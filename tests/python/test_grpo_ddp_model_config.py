"""GRPO logit-scaling helpers must read config through DDP wrappers."""

from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read_source() -> str:
    with open(SOURCE_PATH, "r", encoding = "utf-8") as fh:
        return fh.read()


def test_grpo_logit_scaling_uses_model_config_helper():
    src = _read_source()
    # Helper exists and unwraps DDP/Accelerate wrappers via `.module`.
    assert "def _unsloth_get_model_config(model):" in src
    assert 'getattr(model.module, "config", None)' in src
    # Softcapping takes the model and tolerates a missing config.
    assert "logit_softcapping = _unsloth_get_final_logit_softcapping(model)" in src
    assert "if config is None:" in src.split("def _unsloth_get_final_logit_softcapping")[1]
    # Logit scale/divide read through the unwrapped config, not bare model.config.
    assert 'getattr(model_config, "logit_scale", 0)' in src
    assert 'getattr(model_config, "logits_scaling", 0)' in src
    assert src.count("model_config = _unsloth_get_model_config(model)") >= 2
    # Helper source is injected into the compiled GRPO trainer.
    assert "inspect.getsource(_unsloth_get_model_config)" in src
    # No direct model.config access remains in the RL logit path.
    assert "model.config" not in src


def test_detect_logit_transforms_reads_the_unwrapped_config():
    """The shared helper must be handed model_config, never the bare model.

    detect_logit_transforms only does getattr(x, "config", x), so a DDP or
    Accelerate wrapper (which does not forward .config) makes it read the
    transforms off the wrapper and report zeros. That silently drops Gemma
    softcapping and Cohere/Granite/Falcon-H1 scaling on multi-GPU runs.
    """
    src = _read_source()
    assert "detect_logit_transforms(model)" not in src
    assert src.count("detect_logit_transforms(model_config)") >= 2


def test_detect_logit_transforms_zeroes_out_on_a_wrapped_model():
    """Behavioural counterpart: prove the wrapper really does hide .config."""
    torch = __import__("importlib").import_module("torch")
    transformers = __import__("importlib").import_module("transformers")
    planner = __import__("importlib").import_module("unsloth_zoo.device_map_planner")
    detect = getattr(planner, "detect_logit_transforms", None)
    if detect is None:
        return  # older unsloth_zoo: the fallback branch is in use

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = transformers.Gemma2Config()

    class _Wrapper(torch.nn.Module):
        """Same shape as DistributedDataParallel: real model under .module."""

        def __init__(self, module):
            super().__init__()
            self.module = module

    inner = _Inner()
    wrapped = _Wrapper(inner)
    assert not hasattr(wrapped, "config")
    # The bare wrapper hides the config, so the helper finds nothing.
    assert detect(wrapped)["logit_softcapping"] == 0.0
    # Unwrapped through the helper the real value comes back.
    config = _unsloth_get_model_config_reference(wrapped)
    assert detect(config)["logit_softcapping"] == inner.config.final_logit_softcapping


def _unsloth_get_model_config_reference(model):
    """Mirror of rl_replacements._unsloth_get_model_config, kept local so this
    test does not need to import the heavyweight module."""
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "module"):
        config = getattr(model.module, "config", None)
    return config
