# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Store payloads are JSON, never pickle: the TCPStore is unauthenticated."""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import pickle
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MODELS = REPO / "unsloth" / "models"

codec = None  # absent before the codec existed: the store tests below still run the old functions
if (MODELS / "_store_codec.py").is_file():
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


class _Store:
    """The c10d store calls the save paths make: set / get / wait."""

    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = bytes(value)

    def get(self, key):
        return self.data[key]

    def wait(
        self,
        keys,
        timeout = None,
    ):
        assert all(k in self.data for k in keys), keys


def _store_functions(module, monkeypatch, store):
    """The real store functions of `module`, run against `store` without importing unsloth."""
    names = {"_publish", "_wait", "_share_from_writer"}
    tree = ast.parse((MODELS / module).read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    package = types.ModuleType("_store_codec_test_pkg")
    package._store_codec = codec
    monkeypatch.setitem(sys.modules, "_store_codec_test_pkg", package)
    dist = types.SimpleNamespace(
        distributed_c10d = types.SimpleNamespace(_get_default_store = lambda: store),
        barrier = lambda: None,
    )
    namespace = {
        "__package__": "_store_codec_test_pkg",
        "_store_codec": codec,
        "pickle": pickle,
        "_dist": lambda: dist,
        "_store": lambda: store,
        "_timeout": lambda: None,
        "_save_timeout": lambda: None,
        "_EXPORT_COUNTER": {},
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(MODELS / module), "exec"), namespace)
    return namespace, dist


def test_tp_rank_waiting_on_planted_pickle_bytes_does_not_run_them(monkeypatch):
    store = _Store()
    tp, _ = _store_functions("tp_replica.py", monkeypatch, store)
    store.set("unsloth_tp_save_1", pickle.dumps(("ok", _Planted())))
    result = tp["_wait"]("unsloth_tp_save_1")
    assert _Planted.ran is False
    assert result[0] == "error", result


def test_fsdp2_rank_waiting_on_planted_pickle_bytes_does_not_run_them(monkeypatch):
    store = _Store()
    fsdp2, dist = _store_functions("fsdp2_save.py", monkeypatch, store)
    store.set("unsloth_fsdp2_export_r0_result_1", pickle.dumps(("ok", _Planted())))
    result = fsdp2["_share_from_writer"](dist, False, 0, "result")
    assert _Planted.ran is False
    assert result is None or result[0] == "error", result


@pytest.mark.parametrize(
    "payload, expected",
    [
        (("ok", None), ("ok", None)),
        (("ok", ("out/tokenizer_config.json", "out/tokenizer.json")), None),
        (("ok", {"gguf_files": ["out/m.Q8_0.gguf"], "is_vlm": False}), None),
        (("error", "OSError: disk full"), None),
        (("ok", object()), ("ok", None)),
    ],
    ids = ["none", "tokenizer_files", "gguf_dict", "error", "not_plain_data"],
)
def test_waiting_ranks_get_what_rank0_returned(monkeypatch, payload, expected):
    """What every caller passes today arrives unchanged; a value that is not plain data arrives as None, as before."""
    store = _Store()
    tp, _ = _store_functions("tp_replica.py", monkeypatch, store)
    tp["_publish"]("unsloth_tp_save_1", payload)
    assert tuple(tp["_wait"]("unsloth_tp_save_1"))[:2] == (expected or payload)
    fsdp2, dist = _store_functions("fsdp2_save.py", monkeypatch, store)
    fsdp2["_share_from_writer"](dist, True, 0, "result", payload)
    fsdp2["_EXPORT_COUNTER"].clear()
    got = fsdp2["_share_from_writer"](dist, False, 0, "result")
    assert got == (expected or payload) or (payload[1].__class__ is object and got == ("ok", None))
