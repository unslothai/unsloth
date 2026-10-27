# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Store payloads are JSON, never pickle: the TCPStore is unauthenticated."""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import pickle

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS = REPO / "unsloth" / "models"

_spec = importlib.util.spec_from_file_location("_store_codec", MODELS / "_store_codec.py")
codec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(codec)


@pytest.mark.parametrize(
    "value",
    [
        ("ok", None),
        ("ok", ("a/tokenizer_config.json", "a/vocab.json")),
        ("error", "OSError: disk full"),
        ("ok", ["x.gguf"]),
        ("ok", {"path": "out", "n": 3, "f": [1.5, True]}),
        None,
    ],
)
def test_plain_results_round_trip(value):
    assert codec.decode(codec.encode(value)) == value


def test_a_path_arrives_as_its_string():
    assert codec.decode(codec.encode(("ok", pathlib.Path("/tmp/x")))) == ("ok", os.fspath("/tmp/x"))


def test_an_object_that_is_not_plain_data_arrives_as_none():
    class CommitInfo:
        pass

    assert codec.decode(codec.encode(("ok", CommitInfo()))) == ("ok", None)


class _Planted:
    ran = False

    def __reduce__(self):
        return (setattr, (_Planted, "ran", True))


def test_planted_pickle_bytes_are_refused_without_running():
    with pytest.raises(ValueError):
        codec.decode(pickle.dumps(_Planted()))
    assert _Planted.ran is False


@pytest.mark.parametrize(
    "module, function",
    [
        ("tp_replica.py", "_wait"),
        ("tp_replica.py", "_publish"),
        ("fsdp2_save.py", "_share_from_writer"),
    ],
)
def test_the_store_paths_do_not_use_pickle(module, function):
    tree = ast.parse((MODELS / module).read_text(encoding = "utf-8"))
    (node,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function]
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }
    assert "pickle" not in names and "loads" not in names
    assert "_store_codec" in names
