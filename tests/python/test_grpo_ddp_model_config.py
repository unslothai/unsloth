"""GRPO logit-scaling helpers must read config through DDP wrappers."""

from __future__ import annotations

import ast
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read_source() -> str:
    with open(SOURCE_PATH, "r", encoding = "utf-8") as fh:
        return fh.read()


def _model_config_reads(src: str) -> list[int]:
    """Lines where rl_replacements reads model.config: attribute access in its own code, or the text in a
    string that can end up in a patched trainer. The one string exempt is the search half of a
    `.replace(old, new)`: TRL's own `model.config` line being matched there is being removed, not added."""
    tree = ast.parse(src)
    search_targets = {
        id(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and node.args
    }
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "config"
            and isinstance(node.value, ast.Name)
            and node.value.id == "model"
        ):
            hits.append(node.lineno)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "model.config" in node.value
            and id(node) not in search_targets
        ):
            hits.append(node.lineno)
    return sorted(hits)


def test_grpo_logit_scaling_uses_model_config_helper():
    src = _read_source()
    assert "def _unsloth_get_model_config(model):" in src
    assert 'getattr(model.module, "config", None)' in src
    assert "logit_softcapping = _unsloth_get_final_logit_softcapping(model)" in src
    assert "if config is None:" in src.split("def _unsloth_get_final_logit_softcapping")[1]
    # Whitespace-insensitive: the formatter wraps one of the two calls and not the other.
    flat = "".join(src.split())
    assert (
        flat.count("=_unsloth_resolve_logit_scales(model_config)") == 3
    ), "every fallback arm (two GRPO, one GKD) must resolve the scales from the unwrapped config"
    assert "_unsloth_resolve_logit_scales(model)" not in flat
    assert src.count("model_config = _unsloth_get_model_config(model)") >= 2
    assert "inspect.getsource(_unsloth_get_model_config)" in src
    assert "inspect.getsource(_unsloth_resolve_logit_scales)" in src
    assert (
        _model_config_reads(src) == []
    ), "read the config through _unsloth_get_model_config, not model.config"


def test_detect_logit_transforms_reads_the_unwrapped_config():
    """Pass model_config: DDP wrappers do not forward .config, so transforms read as zeros."""
    src = _read_source()
    assert "detect_logit_transforms(model)" not in src
    assert src.count("detect_logit_transforms(model_config)") >= 2


def test_detect_logit_transforms_zeroes_out_on_a_wrapped_model():
    """Checks the resolved config yields the transforms, not what the helper returns for a bare wrapper."""
    torch = __import__("importlib").import_module("torch")
    transformers = __import__("importlib").import_module("transformers")
    planner = __import__("importlib").import_module("unsloth_zoo.device_map_planner")
    detect = getattr(planner, "detect_logit_transforms", None)
    if detect is None:
        return

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
    config = _unsloth_get_model_config_reference(wrapped)
    assert config is inner.config
    assert detect(config)["logit_softcapping"] == inner.config.final_logit_softcapping


def _unsloth_get_model_config_reference(model):
    """Copy of rl_replacements._unsloth_get_model_config, kept local to avoid a heavy import."""
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "module"):
        config = getattr(model.module, "config", None)
    return config


def test_the_model_config_guard_tells_a_search_target_from_a_read():
    """A replace() search target is TRL's text being removed; a replacement, a snippet or real code is a read."""
    removed = 'f = f.replace("x = model.config.model_type", "x = helper(model)")\n'
    assert _model_config_reads(removed) == []
    assert _model_config_reads(
        'f = f.replace("x = helper(model)", "x = model.config.model_type")\n'
    ) == [1]
    assert _model_config_reads('SNIPPET = "scale = model.config.logit_scale"\n') == [1]
    assert _model_config_reads(
        "def g(model):\n    return model.config.final_logit_softcapping\n"
    ) == [2]
