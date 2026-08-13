# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only regression tests for the UEmbed EOS/pooling assembly seam.

The production package cannot be imported on a CPU-only runner, so these tests extract
and execute the small module-selection function from ``sentence_transformer.py``. The
objects passed to it implement the same observable pooling contract as sentence-
transformers modules; no model, network, accelerator, or timing is involved.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path

import pytest

_SOURCE_PATH = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "sentence_transformer.py"


class _StockPooling:
    def __init__(self, dimension: int, mode: str = "lasttoken") -> None:
        self.dimension = dimension
        self.mode = mode

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class _OffsetPooling:
    def __init__(self, word_embedding_dimension: int, num_eos_tokens: int) -> None:
        self.word_embedding_dimension = word_embedding_dimension
        self.num_eos_tokens = num_eos_tokens

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension


class _Transformer:
    pass


class _Normalize:
    pass


def _load_selection_function():
    source = _SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SOURCE_PATH))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_ensure_uembed_offset_pooling"
        ),
        None,
    )
    assert function is not None, "UEmbed pooling assembly safeguard is missing"

    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"OffsetLastTokenPooling": _OffsetPooling}
    exec(compile(isolated, str(_SOURCE_PATH), "exec"), namespace)
    return namespace["_ensure_uembed_offset_pooling"]


@pytest.fixture(scope="module")
def ensure_offset_pooling():
    return _load_selection_function()


def _stock_chain(mode: str = "lasttoken") -> OrderedDict:
    return OrderedDict(
        transformer=_Transformer(),
        pooling=_StockPooling(7, mode=mode),
        normalize=_Normalize(),
    )


@pytest.mark.parametrize("stock_mode", ["lasttoken", "mean"])
def test_positive_num_eos_replaces_stock_modules_json_pooling(
    ensure_offset_pooling, stock_mode
):
    """A UEmbed checkpoint can never leave stock pooling in the assembled chain."""
    modules = _stock_chain(stock_mode)
    transformer = modules["transformer"]
    normalize = modules["normalize"]

    ensure_offset_pooling(
        modules,
        num_eos_tokens=3,
        hidden_size=99,
        pooling_class=_StockPooling,
    )

    assert list(modules) == ["transformer", "pooling", "normalize"]
    assert modules["transformer"] is transformer
    assert modules["normalize"] is normalize
    assert isinstance(modules["pooling"], _OffsetPooling)
    assert modules["pooling"].num_eos_tokens == 3
    assert modules["pooling"].word_embedding_dimension == 7


def test_positive_num_eos_refuses_an_unreplaceable_pooling_chain(ensure_offset_pooling):
    """If safe replacement is impossible, loading fails before encode instead of mispooling."""
    modules = OrderedDict(transformer=_Transformer(), normalize=_Normalize())

    with pytest.raises(RuntimeError) as error:
        ensure_offset_pooling(
            modules,
            num_eos_tokens=2,
            hidden_size=7,
            pooling_class=_StockPooling,
        )

    message = str(error.value)
    assert "Unsloth:" in message
    assert "OffsetLastTokenPooling" in message
    assert "EOS" in message


def test_zero_num_eos_preserves_stock_pooling_exactly(ensure_offset_pooling):
    """Non-UEmbed checkpoints retain the existing stock module and chain unchanged."""
    modules = _stock_chain("mean")
    before_items = list(modules.items())

    ensure_offset_pooling(
        modules,
        num_eos_tokens=0,
        hidden_size=99,
        pooling_class=_StockPooling,
    )

    assert list(modules.items()) == before_items
    assert type(modules["pooling"]) is _StockPooling
    assert modules["pooling"].mode == "mean"


def test_from_pretrained_passes_the_single_resolved_eos_count_into_assembly():
    """The metadata value driving EOS append is also the value driving pooling selection."""
    tree = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    from_pretrained = next(
        node
        for node in ast.walk(class_node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "from_pretrained"
    )

    reads = [
        node
        for node in ast.walk(from_pretrained)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "read_num_eos_tokens"
    ]
    assert len(reads) == 1, "from_pretrained must resolve sparse_info.json exactly once"

    assembly_call = next(
        node
        for node in ast.walk(from_pretrained)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_load_modules"
    )
    keyword = next((kw for kw in assembly_call.keywords if kw.arg == "num_eos_tokens"), None)
    assert keyword is not None
    assert isinstance(keyword.value, ast.Name) and keyword.value.id == "num_eos_tokens"
