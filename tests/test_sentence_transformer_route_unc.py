# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Hub sentence-transformers Router route is a folder joined onto the module and probed: a UNC / rooted /
`..` route must be refused before it reaches the filesystem (Windows: SMB + NTLM to a host the repo chose),
as a top-level modules.json path already is. Local folders are the user's own and are left alone."""

import ast
import json
import logging
import os
import pathlib
import posixpath

import pytest

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "sentence_transformer.py"
)
WANTED = {
    "_check_module_config_class_refs",
    "_class_ref_rules",
    "_folder_escapes_model",
    "_MODULE_CONFIG_CLASS_REFS",
    "_LEGACY_MODULE_CONFIG_FILES",
    "_MAX_SUB_MODULE_DEPTH",
}


def _gate():
    """FastSentenceTransformer's route walk, from source (the module needs the GPU package)."""
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "FastSentenceTransformer"
    )
    body = [
        n
        for n in cls.body
        if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in WANTED for t in n.targets))
    ]
    stub = ast.parse(
        "@staticmethod\ndef _import_sentence_transformers_class(value, model_name):\n"
        "    return type('Dense', (), {})\n"
        "@staticmethod\ndef _refuse_whole_model_class(sub_class, value, model_name):\n    return None\n"
    ).body
    new = ast.ClassDef(
        name = "FastSentenceTransformer",
        bases = [],
        keywords = [],
        body = body + stub,
        decorator_list = [],
        type_params = [],
    )
    module = ast.fix_missing_locations(ast.Module(body = [new], type_ignores = []))
    ns = {"os": os, "json": json, "posixpath": posixpath, "logging": logging}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns["FastSentenceTransformer"]


ROUTER = type("Router", (), {"config_file_name": "router_config.json"})


def _router(folder, route):
    folder.mkdir(parents = True, exist_ok = True)
    (folder / "router_config.json").write_text(
        json.dumps({"types": {route: "sentence_transformers.models.Dense"}}), encoding = "utf-8"
    )
    return str(folder)


@pytest.mark.parametrize(
    "route",
    [r"\\evil.example\share\x", "//evil.example/share/x", "/etc", "../../outside", "C:\\x"],
)
def test_hub_router_route_outside_the_model_is_refused(tmp_path, route):
    gate, descended = _gate(), []
    load_path = _router(tmp_path / "snapshot", route)
    with pytest.raises(ValueError, match = "outside the model"):
        gate._check_module_config_class_refs(
            load_path,
            "sentence_transformers.models.Router",
            "someone/st-model",
            False,
            ROUTER,
            descend = lambda *args: descended.append(args),
        )
    assert descended == []


@pytest.mark.parametrize("route", ["query", "modules/../document", ""])
def test_hub_router_routes_inside_the_model_still_descend(tmp_path, route):
    gate, descended = _gate(), []
    load_path = _router(tmp_path / "snapshot", route)
    gate._check_module_config_class_refs(
        load_path,
        "sentence_transformers.models.Router",
        "someone/st-model",
        False,
        ROUTER,
        descend = lambda *args: descended.append(args),
    )
    assert [args[0] for args in descended] == [route]


def test_local_folder_routes_are_unchanged(tmp_path):
    gate, descended = _gate(), []
    load_path = _router(tmp_path / "model", "/elsewhere/query")
    gate._check_module_config_class_refs(
        load_path,
        "sentence_transformers.models.Router",
        load_path,
        False,
        ROUTER,
        descend = lambda *args: descended.append(args),
    )
    assert [args[0] for args in descended] == ["/elsewhere/query"]
