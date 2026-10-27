# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`has_internet` must not leave `socket.setdefaulttimeout` behind: that capped every later
socket in the process (urllib, ModelScope, user code) at the probe's 3 s. Sliced with `ast`:
importing `_utils` needs a GPU. Probes a loopback listener, so no outside network.
"""

import ast
import functools
import os
import socket
from pathlib import Path

import pytest

UTILS_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _load_has_internet():
    source = UTILS_PATH.read_text(encoding = "utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "has_internet":
            namespace = {"os": os, "socket": socket, "functools": functools}
            exec(compile(ast.Module([node], []), str(UTILS_PATH), "exec"), namespace)
            return namespace["has_internet"]
    raise AssertionError("has_internet not found in unsloth/models/_utils.py")


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(None)
    try:
        yield _load_has_internet()
    finally:
        socket.setdefaulttimeout(previous)


def test_reachable_host_leaves_default_timeout_alone(probe):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        assert probe("127.0.0.1", server.getsockname()[1], 3) is True
    assert socket.getdefaulttimeout() is None


def test_unreachable_host_leaves_default_timeout_alone(probe):
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    assert probe("127.0.0.1", port, 3) is False
    assert socket.getdefaulttimeout() is None
