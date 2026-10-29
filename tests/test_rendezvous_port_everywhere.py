# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every distributed test takes MASTER_PORT from tests/_rendezvous_port.py.

A port from ``bind(("127.0.0.1", 0))`` is in the kernel's ephemeral range, and it is released before rank 0
listens on it, so another xdist worker's gloo connection can take it first and the ranks die with
EADDRINUSE. That happened to test_tp_native_gloo.py in Repo tests (CPU, gloo) after #14334 had moved most,
but not all, of the distributed tests onto the shared helper.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
# Studio tests pick ports for HTTP servers they start themselves, not for a torch rendezvous.
SKIP = {TESTS / "studio", TESTS / "_rendezvous_port.py"}
RENDEZVOUS = ("MASTER_PORT", "master_port", "init_process_group", "TCPStore")


def _test_files():
    for path in sorted(TESTS.rglob("*.py")):
        if any(path == s or s in path.parents for s in SKIP):
            continue
        yield path


def _binds_port_zero(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "bind"
            and node.args
            and isinstance(node.args[0], ast.Tuple)
            and len(node.args[0].elts) == 2
            and isinstance(node.args[0].elts[1], ast.Constant)
            and node.args[0].elts[1].value == 0
        ):
            return True
    return False


def _offenders():
    found = []
    for path in _test_files():
        source = path.read_text(encoding = "utf-8")
        if not any(word in source for word in RENDEZVOUS):
            continue  # a port for a server the test starts itself, not a torch rendezvous
        tree = ast.parse(source)
        name = path.relative_to(TESTS.parent).as_posix()
        free_port = [
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_free_port"
        ]
        if any("_rendezvous_port" not in ast.unparse(fn) for fn in free_port):
            found.append(f"{name}: _free_port does not use tests._rendezvous_port")
        elif _binds_port_zero(tree):
            found.append(f"{name}: binds port 0 for a torch rendezvous")
    return found


def test_distributed_tests_take_their_port_from_the_shared_helper():
    offenders = _offenders()
    assert not offenders, "use tests._rendezvous_port.free_port():\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            "def _free_port():\n    with socket.socket() as s:\n        s.bind(('127.0.0.1', 0))\n"
            "        return s.getsockname()[1]\n\n\nos.environ['MASTER_PORT'] = str(_free_port())\n",
            "_free_port does not use tests._rendezvous_port",
        ),
        (
            "with socket.socket() as s:\n    s.bind(('127.0.0.1', 0))\n    port = s.getsockname()[1]\n"
            "os.environ['MASTER_PORT'] = str(port)\n",
            "binds port 0 for a torch rendezvous",
        ),
    ],
)
def test_the_check_catches_a_raw_bind_zero(tmp_path, monkeypatch, body, reason):
    bad = tmp_path / "test_raw_port.py"
    bad.write_text("import os\nimport socket\n\n" + body, encoding = "utf-8")
    monkeypatch.setitem(globals(), "TESTS", tmp_path)
    monkeypatch.setitem(globals(), "_test_files", lambda: iter([bad]))
    assert _offenders() == [f"{tmp_path.name}/test_raw_port.py: {reason}"]


@pytest.mark.parametrize("worker", ["gw0", "gw3", "gw11"])
def test_the_helper_stays_below_the_ephemeral_range(monkeypatch, worker):
    from tests._rendezvous_port import HIGH, free_port
    monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
    assert free_port() < HIGH
