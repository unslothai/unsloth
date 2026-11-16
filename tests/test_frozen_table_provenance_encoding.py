# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The frozen-table provenance sidecar is read and written as UTF-8 (Windows defaults to a code page)."""

import ast
import os


def test_provenance_file_io_names_its_encoding():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "unsloth",
        "models",
        "frozen_table_provenance.py",
    )
    with open(path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    opens = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "open"
    ]
    assert opens
    assert all(any(k.arg == "encoding" for k in n.keywords) for n in opens), [
        n.lineno for n in opens
    ]
