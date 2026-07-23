"""Regression test for `_is_vlm` in `unsloth/save.py`.

The VLM check in `unsloth_save_pretrained_gguf` (and the torchao export path)
used to guard on `hasattr(self.config, "architectures")` and then iterate
`self.config.architectures` directly. That guard is a no-op: transformers'
`PretrainedConfig` always sets `architectures` (defaulting to `None`), so a
config with `architectures = None` passed the guard and hit `for x in None`,
raising `TypeError: 'NoneType' object is not iterable` and aborting the export
before any merge/convert work.

`_is_vlm` centralizes the check and guards `architectures` with
`getattr(config, "architectures", None) or ()`, matching the sibling
`_is_gpt_oss` / `_is_qwen3_5_vlm` helpers. We ast-extract just that function so
the test runs with no GPU and no `import unsloth` (which needs `unsloth_zoo`).
"""

import ast
import os

SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "unsloth", "save.py")


def _load_is_vlm():
    tree = ast.parse(open(SAVE_PATH).read())
    func = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_is_vlm"
    )
    namespace = {}
    module = ast.Module(body = [func], type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, SAVE_PATH, "exec"), namespace)
    return namespace["_is_vlm"]


class _Cfg:
    def __init__(
        self,
        architectures,
        vision_config = False,
    ):
        self.architectures = architectures
        if vision_config:
            self.vision_config = object()


class _Model:
    def __init__(self, config):
        self.config = config


def test_is_vlm_handles_none_architectures():
    is_vlm = _load_is_vlm()
    # architectures = None must not raise (it did before: `for x in None`).
    assert is_vlm(_Model(_Cfg(None))) is False


def test_is_vlm_detects_vision_architecture_and_config():
    is_vlm = _load_is_vlm()
    assert is_vlm(_Model(_Cfg(["Gemma3ForConditionalGeneration"]))) is True
    assert is_vlm(_Model(_Cfg(None, vision_config = True))) is True


def test_is_vlm_false_for_text_model_and_missing_config():
    is_vlm = _load_is_vlm()
    assert is_vlm(_Model(_Cfg(["LlamaForCausalLM"]))) is False
    assert is_vlm(object()) is False
