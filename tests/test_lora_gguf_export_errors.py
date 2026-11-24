# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""LoRA GGUF export: refuse MoE expert (target_parameters) adapters up front; name the converter's error."""

import ast
import pathlib
import re
from types import SimpleNamespace

_SAVE = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "save.py"
_TREE = ast.parse(_SAVE.read_text(encoding = "utf-8"))


def _helpers():
    # By source, so the test needs no `import unsloth` (which wants a GPU).
    names = {"_lora_target_parameters", "_converter_error_line"}
    nodes = [n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"re": re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(_SAVE), "exec"), namespace)
    return namespace["_lora_target_parameters"], namespace["_converter_error_line"]


def test_expert_targets_are_read_from_every_adapter():
    targets, _ = _helpers()
    model = SimpleNamespace(
        peft_config = {
            "default": SimpleNamespace(
                target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
            ),
            "other": SimpleNamespace(target_parameters = None),
        }
    )
    assert targets(model) == ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
    assert targets(SimpleNamespace(peft_config = {"default": SimpleNamespace()})) == []
    assert targets(SimpleNamespace()) == []


def test_converter_error_line_is_the_last_exception_line():
    _, error_line = _helpers()
    output = [
        "INFO:lora-to-gguf:Exporting model...",
        "Traceback (most recent call last):",
        '  File "convert_lora_to_gguf.py", line 506, in modify_tensors',
        '    raise ValueError(f"Can not map tensor {name!r}")',
        "ValueError: Can not map tensor 'model.layers.0.mlp.experts.base_layer.weight'",
    ]
    assert (
        error_line(output)
        == "ValueError: Can not map tensor 'model.layers.0.mlp.experts.base_layer.weight'"
    )
    assert error_line(["INFO: nothing went wrong"]) == ""


def test_refusal_comes_before_the_adapter_is_written():
    fn = next(
        n
        for n in _TREE.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_save_lora_gguf"
    )
    source = ast.get_source_segment(_SAVE.read_text(encoding = "utf-8"), fn)
    refusal = source.index("_lora_target_parameters(model)")
    assert refusal < source.index("os.makedirs(") < source.index("save_lora_to_custom_dir(")
