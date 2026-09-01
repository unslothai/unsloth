# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pins the two rules a pre-quantized bitsandbytes load depends on. No GPU needed.

Rule one: a pre-quantized checkpoint's own ``llm_int8_skip_modules`` is the authority and
Unsloth must not add to it. The list describes how the tensors were actually packed, so
adding a name makes transformers build a dense ``Linear`` for packed weights and the load
dies in ``load_state_dict``:

    size mismatch for weight: copying a param with shape torch.Size([15728640, 1])
    from checkpoint, the shape in current model is torch.Size([4096, 7680])

A real failure, observed on ``unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit``, whose
config ships ``llm_int8_skip_modules: null`` because it quantized everything. ``None``
there is an instruction ("skip nothing"), not an absence, and replacing it with Unsloth's
generic list broke the two ``test_save_merged_*`` cases for that model.

Rule two: what the load used is what gets saved. ``loader.py`` used to stamp ``None`` over
the real list, which for a dynamic-quant repo like ``unsloth/Qwen3-0.6B-unsloth-bnb-4bit``
threw away every per-layer entry and saved a config describing a layout that never existed.

Extracted with ast so nothing in loader.py has to import.
"""

import ast
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "unsloth", "models")
LOADER = os.path.join(MODELS, "loader.py")
VISION = os.path.join(MODELS, "vision.py")


def _load(path, *names):
    source = open(path, encoding = "utf-8").read()
    ns, wanted = {}, set(names)
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(source, node), ns)
            wanted.discard(node.name)
    if wanted:
        raise AssertionError(f"not found in {os.path.basename(path)}: {sorted(wanted)}")
    return ns


loaded_skip = _load(LOADER, "_config_get", "_loaded_skip_modules")["_loaded_skip_modules"]

# What unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit's config.json actually carries.
GLIMMER = [
    "model.language_model.embed_tokens",
    "lm_head",
    "model.vision_tower",
    "model.vision_adapter",
    "model.vision_projection",
]


class _Config:
    """A BitsAndBytesConfig stands in as a plain attribute holder."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_the_vision_loader_does_not_touch_the_checkpoint_skip_list():
    """The regression this file exists for.

    Transformers already prefers a pre-quantized checkpoint's `quantization_config` over
    the runtime one, so there is nothing for the loader to fix. Writing into that config is
    the only way to get it wrong, and it did: on Llama-3.2-11B-Vision-bnb-4bit it turned a
    `null` skip list into Unsloth's generic one and broke the load outright.
    """
    source = open(VISION, encoding = "utf-8").read()
    assert "merge_checkpoint_skip_modules" not in source
    # never do is assign into the config that came off the checkpoint.
    # It may still build its own runtime list (the next test pins that);
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            written = ast.unparse(target)
            assert not (
                "quantization_config" in written and "llm_int8_skip_modules" in written
            ), f"vision.py writes the checkpoint's skip list at line {node.lineno}: {written}"
        if isinstance(node.targets[0], ast.Subscript):
            written = ast.unparse(node.targets[0])
            assert (
                "llm_int8_skip_modules" not in written
            ), f"vision.py writes a skip list by subscript at line {node.lineno}"


def test_the_runtime_skip_list_is_still_built_for_on_the_fly_quantization():
    """The other half: a full-precision checkpoint has no config to defer to, so Unsloth's
    own list keeps heads, routers and towers in compute dtype. Hence the fix was to remove
    a write, not to stop building the list."""
    source = open(VISION, encoding = "utf-8").read()
    assert "_skip_modules = SKIP_QUANTIZATION_MODULES.copy()" in source
    assert "llm_int8_skip_modules = _skip_modules" in source


def test_the_bnb_config_chain_is_still_one_piece():
    """The merge used to sit inside this four-branch if/elif, where a statement dropped in
    the middle silently re-parents the last branch and the "Switching to 16bit LoRA" notice
    fires on every 16-bit load."""
    source = open(VISION, encoding = "utf-8").read()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If) or ast.unparse(node.test) != "load_in_4bit":
            continue
        if "BitsAndBytesConfig" not in ast.unparse(node):
            continue
        tests, current = [], node
        while True:
            tests.append(ast.unparse(current.test))
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
            else:
                break
        assert "load_in_8bit" in tests and "load_in_16bit" in tests, tests
        assert any(
            "full_finetuning" in test for test in tests
        ), f"the 16bit-LoRA notice fell out of the chain: {tests}"
        return
    raise AssertionError("could not find the bnb_config if/elif chain in vision.py")


def test_the_stamp_keeps_the_list_the_load_actually_used():
    """A dynamic-quant repo's per-layer entries have to survive into the saved config, or
    the adapter records a base topology that cannot be rebuilt."""
    real = GLIMMER + ["model.layers.27.mlp.up_proj"]
    config = _Config(quantization_config = {"llm_int8_skip_modules": real})
    assert loaded_skip(config) == real


def test_the_stamp_still_reports_none_when_there_was_no_list():
    """None is the instruction Llama-3.2-11B-Vision-bnb-4bit ships: quantize everything.
    Inventing a list here is the same mistake in a different place."""
    for config in (
        _Config(quantization_config = {}),
        _Config(quantization_config = {"llm_int8_skip_modules": None}),
        _Config(quantization_config = None),
        _Config(),
    ):
        assert loaded_skip(config) is None


def test_the_stamp_preserves_an_explicit_empty_list():
    """[] and None are not interchangeable: None lets transformers pick the output head
    itself, [] says it was told to exclude nothing."""
    config = _Config(quantization_config = {"llm_int8_skip_modules": []})
    assert loaded_skip(config) == []


def test_the_stamp_reads_an_object_config_too():
    config = _Config(quantization_config = _Config(llm_int8_skip_modules = list(GLIMMER)))
    assert loaded_skip(config) == GLIMMER


def test_the_stamp_is_applied_at_every_site_that_writes_a_synthetic_config():
    """There are two of these, one per loader class. Fixing one and not the other leaves
    half the models still saving a config that describes nothing."""
    source = open(LOADER, encoding = "utf-8").read()
    assert source.count('"llm_int8_skip_modules": _loaded_skip_modules(model.config)') == 2
    assert '"llm_int8_skip_modules": None' not in source
