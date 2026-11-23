# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Hub Clef adapter's base must not send the load to a Windows network share (SMB, NTLM leak)."""

import ast
import json
import os
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models"


def _function(path, name, ns):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(path), "exec"), ns)
    return ns[name]


def _is_network_share(path):
    # The Windows branch of unsloth_zoo.vision_utils._is_network_share, forced on.
    p = path.replace("\\", "/")
    return p.startswith("//") and p[2:].split("/", 1)[0].lower() not in ("", "localhost", "wsl$")


@pytest.fixture
def load_clef(monkeypatch):
    refuse = _function(ROOT / "loader.py", "_refuse_network_share_adapter_base", {"os": os})
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.vision_utils",
        types.SimpleNamespace(_is_network_share = _is_network_share),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth.models.loader",
        types.SimpleNamespace(_refuse_network_share_adapter_base = refuse),
    )
    monkeypatch.setitem(
        sys.modules, "unsloth.models.clef", types.SimpleNamespace(JointSchemaHead = object)
    )

    def stop(*args, **kwargs):
        raise RuntimeError("reached the model load")

    ns = {
        "__name__": "unsloth.models.decision",
        "__package__": "unsloth.models",
        "json": json,
        "types": types,
        "CLEF_MAX_LEN": 512,
        "_DECISION_CONFIG": "rl_agent_config.json",
        "_ADAPTER_CONFIG": "adapter_config.json",
        "_is_clef_adapter": lambda folder: True,
        "_device": stop,
    }
    return _function(ROOT / "decision.py", "_load_clef", ns)


def _hub_snapshot(
    tmp_path,
    adapter_base,
    decision_base = None,
):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": adapter_base})
    )
    if decision_base is not None:
        (tmp_path / "rl_agent_config.json").write_text(json.dumps({"base_model": decision_base}))
    return tmp_path


def _call(load_clef, folder, model_name):
    return load_clef(folder, 512, None, False, False, None, False, {}, model_name = model_name)


@pytest.mark.parametrize(
    "adapter_base, decision_base",
    [
        (r"\\evil.example\share\model", None),
        ("//evil.example/share/model", None),
        ("Qwen/Qwen3-0.6B", r"\\evil.example\share\model"),
        (r"\\evil.example\share\model", "Qwen/Qwen3-0.6B"),
    ],
)
def test_hub_clef_adapter_with_unc_base_is_refused(
    load_clef, tmp_path, adapter_base, decision_base
):
    folder = _hub_snapshot(tmp_path, adapter_base, decision_base)
    with pytest.raises(ValueError, match = "network share"):
        _call(load_clef, folder, "someone/clef-adapter")


def test_hub_clef_adapter_with_hub_base_still_loads(load_clef, tmp_path):
    folder = _hub_snapshot(tmp_path, "Qwen/Qwen3-0.6B")
    with pytest.raises(RuntimeError, match = "reached the model load"):
        _call(load_clef, folder, "someone/clef-adapter")


def test_local_clef_adapter_folder_is_the_users_own(load_clef, tmp_path):
    folder = _hub_snapshot(tmp_path, r"\\nas\share\model")
    with pytest.raises(RuntimeError, match = "reached the model load"):
        _call(load_clef, folder, str(folder))
