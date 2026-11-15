# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import sys
from pathlib import Path

import pytest

from core.inference import mcp_oauth_issuer
from core.inference.mcp_oauth_issuer import same_issuer

SAME = [
    ("https://accounts.google.com", "https://accounts.google.com"),
    ("https://accounts.google.com/", "https://accounts.google.com/"),
    ("https://accounts.google.com", "https://accounts.google.com/"),
    ("http://localhost:8080", "http://localhost:8080/"),
    ("https://[::1]:8443", "https://[::1]:8443/"),
    ("https://login.example.com/tenant", "https://login.example.com/tenant"),
]

DIFFERENT = [
    ("https://login.example.com/tenant", "https://login.example.com/tenant/"),
    ("https://login.example.com/", "https://login.example.com/tenant"),
    ("https://accounts.google.com", "https://accounts.example.com/"),
    ("https://accounts.google.com", "https://accounts.google.com.example.com/"),
    ("https://accounts.google.com", "http://accounts.google.com/"),
    ("https://accounts.google.com", "https://accounts.google.com:8443/"),
    ("https://accounts.google.com", "https://Accounts.google.com/"),
    ("https://accounts.google.com/", "https://accounts.google.com//"),
    ("https://accounts.google.com?x=1", "https://accounts.google.com?x=1/"),
    ("https://accounts.google.com#x", "https://accounts.google.com#x/"),
    ("https://", "https:///"),
    ("https://[::1", "https://[::1/"),
    ("", "/"),
]


@pytest.mark.parametrize(("a", "b"), SAME)
def test_same_server(a, b):
    assert same_issuer(a, b)
    assert same_issuer(b, a)


@pytest.mark.parametrize(("a", "b"), DIFFERENT)
def test_different_server(a, b):
    assert not same_issuer(a, b)
    assert not same_issuer(b, a)


def test_policy_imports_only_the_standard_library():
    tree = ast.parse(Path(mcp_oauth_issuer.__file__).read_text(encoding = "utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names)
