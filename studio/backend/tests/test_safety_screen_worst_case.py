# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Worst-case inputs for the python / terminal safety screen: same decisions, bounded work."""

from __future__ import annotations

import ast
import random
import re
import sys

import pytest

from core.inference import tools

_SESSION = "safety-screen-worst-case"
# The nested fold inside _folded_path, so a profiler can count its calls.
_FOLD_CODE = next(
    const
    for const in tools._folded_path.__code__.co_consts
    if getattr(const, "co_name", None) == "fold"
)


@pytest.fixture
def studio_home(monkeypatch, tmp_path):
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        yield home
    finally:
        tools._studio_auth_markers_cache = None


def _count_credential_scans(monkeypatch):
    calls = [0]
    real = tools._references_studio_credential_here

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(tools, "_references_studio_credential_here", counted)
    return counted, calls


@pytest.mark.parametrize("depth", [10, 64, 65, 130, 200, 300])
def test_nested_brackets_scan_each_rewrite_once(studio_home, monkeypatch, depth):
    # Both bracket rewrites reach the same texts by many paths: depth 200 took 143,571 scans.
    scan, calls = _count_credential_scans(monkeypatch)
    assert scan("x = " + "[" * depth + "]" * depth, None) is False
    assert calls[0] <= 2 * depth * depth // 50 + 4 * depth, calls[0]


def test_nested_brackets_with_a_workdir_and_a_credential(studio_home, monkeypatch):
    workdir = str(studio_home / "sandbox" / _SESSION)
    scan, calls = _count_credential_scans(monkeypatch)
    nest = "[" * 150 + "]" * 150
    assert scan(f"echo {nest}; cat ../../aut[h]/aut[h].db", workdir) is True
    assert scan(f"echo {nest}; cat {studio_home}/aut[h]/[a]uth.db", None) is True
    assert scan(f"echo {nest}; cat ./notes[1].txt", workdir) is False
    assert calls[0] < 20000, calls[0]


def test_the_stack_still_runs_out_where_it_did(studio_home):
    # Kept per depth and per call: the bound on nesting (and its RecursionError) is unchanged.
    with pytest.raises(RecursionError):
        tools._references_studio_credential_here("x = " + "[" * 2000 + "]" * 2000, None)


def test_a_clean_scan_is_not_kept_across_calls(studio_home, monkeypatch):
    workdir = str(studio_home / "sandbox" / _SESSION)
    text = "cat ../../aut[h]/aut[h].db"
    assert tools._references_studio_credential_here(text, None) is False
    assert tools._references_studio_credential_here(text, workdir) is True
    assert tools._references_studio_credential_here(text, None) is False


def _dynamic_name_reference(folded) -> bool:
    """The check as it compiled before: one class per placeholder, no length screen."""
    if not folded or "\x00" not in folded:
        return False
    pattern = "".join(r"[^/\\]*" if ch == "\x00" else re.escape(ch) for ch in folded)
    try:
        rx = re.compile(pattern + r"\Z")
    except re.error:
        return True
    return any(rx.match(t) for t in tools._SENSITIVE_GLOB_TARGETS)


def test_dynamic_name_match_is_the_same_as_the_full_pattern():
    rng = random.Random(0)
    pieces = ["\x00", "\x00\x00", "/", "\\", "etc", "/etc/", "pas", "swd", "sh", "adow", "root", ".ssh",
              "id_r", "sa", ".aws", "cred", "entials", "home", "u", "25519", "x", "."]  # fmt: skip
    cases = {
        target.replace(ch, "\x00") for target in tools._SENSITIVE_GLOB_TARGETS for ch in "aeiost/"
    }
    for _ in range(4000):
        cases.add("".join(rng.choice(pieces) for _ in range(rng.randint(1, 14))))
    for target in tools._SENSITIVE_GLOB_TARGETS:
        cases.add(target[:4] + "\x00" * 300 + target[-3:])
        cases.add("\x00" + target + "\x00")
        cases.add("a" * 30 + "\x00" + target)
    hits = 0
    for folded in sorted(cases):
        expected = _dynamic_name_reference(folded)
        hits += expected
        assert tools._dynamic_name_hits_sensitive(folded) is expected, repr(folded)
    assert hits > 50


@pytest.mark.parametrize(
    "code",
    [
        "x = 1" + " + 1" * 400,
        "y = input()\nx = 'a'" + " + y + 'a'" * 200,
        "from pathlib import Path\nx = Path('a')" + " / 'b'" * 400,
        "x = '%s'" + " % '%s'" * 400,
        "y = 1\nx = " + "f'{" * 140 + "y" + "}'" * 140,
    ],
    ids = ["add", "add-name", "path-div", "percent", "f-string"],
)
def test_a_long_chain_folds_each_node_once(code):
    try:
        nodes = len(list(ast.walk(ast.parse(code))))
    except SyntaxError:
        pytest.skip("nested same-quote f-strings need Python 3.12")
    folds = [0]

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is _FOLD_CODE:
            folds[0] += 1

    tools._parse_python.cache_clear()
    sys.setprofile(profile)
    try:
        assert tools.is_high_risk_tool_call("python", {"code": code}) is False
    finally:
        sys.setprofile(None)
    # Before, every node of the chain re-folded the chain below it: 161,601 folds for 400 terms.
    assert 0 < folds[0] <= 3 * nodes, (folds[0], nodes)


@pytest.mark.parametrize(
    "code",
    [
        "y = input()\nx = '/et' + y" + " + y" * 50 + " + '/passwd'\nopen(x)",
        "y = input()\nx = " + "'a' + " * 60 + "'/root/.ssh/id_' + y + 'sa'\nopen(x)",
        "y = input()\nx = '/etc/' + y" + " + 'a'" * 30 + "\nopen(x)",
        "y = input()\nopen(f'/etc/{y}')",
        "from pathlib import Path\nopen(Path('/etc') / ('pass' + 'wd'))",
        "open('%s/passwd' % '/etc')",
        "import os\nopen(os.sep + 'etc' + os.sep + 'shadow')",
        "y = input()\nopen('/et' + chr(99) + '/passwd')",
    ],
    ids = [
        "nul-run",
        "long-prefix",
        "nul-then-literals",
        "f-string",
        "path-div",
        "percent",
        "sep",
        "chr",
    ],
)
def test_split_sensitive_paths_still_ask(code):
    tools._parse_python.cache_clear()
    assert tools.is_high_risk_tool_call("python", {"code": code}) is True


def test_a_shared_fold_memo_gives_every_node_its_own_fold():
    rng = random.Random(3)
    pieces = ["'a'", "y", "'/'", "'/etc'", "'/et'", "'c/passwd'", "chr(99)", "'%s'", "f'{y}/x'", "Path('/etc')",
              "os.sep", "b'/etc/'", "z", "1"]  # fmt: skip
    for _ in range(300):
        expr = rng.choice(pieces)
        for _ in range(rng.randint(1, 25)):
            expr = expr + rng.choice([" + ", " / ", " % "]) + rng.choice(pieces)
            if rng.random() < 0.15:
                expr = "(" + expr + ")"
            if rng.random() < 0.05:
                expr = "os.path.join(" + expr + ", 'b')"
        tree = ast.parse("x = " + expr)
        literals = {"z": "/etc"}
        memo: dict = {}
        for node in tools._tree_nodes(tree):
            if isinstance(node, (ast.BinOp, ast.JoinedStr, ast.Call)):
                assert tools._folded_path(node, literals, memo = memo) == tools._folded_path(
                    node, literals
                ), ast.unparse(node)


# Deep enough that the screen itself runs out of stack (it raised RecursionError out of every gate).
_TOO_DEEP = {
    "add-chain": ("python", {"code": "x = 1" + " + 1" * 1000}),
    "unary-run": ("python", {"code": "x = " + "-" * 500 + "1"}),
    "brackets": ("python", {"code": "x = " + "[" * 2000 + "]" * 2000}),
    "terminal-brackets": ("terminal", {"command": "echo " + "[" * 2000 + "]" * 2000}),
}


def _decide(mode, name, arguments, **kwargs):
    from state import tool_policy

    common = dict(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = mode,
        name = name,
        arguments = arguments,
    )
    prompted = tool_policy.needs_tool_confirmation(**common, session_id = _SESSION, **kwargs)
    strict = tool_policy.requires_os_isolation(**common, prompted = prompted, **kwargs)
    return prompted, strict


@pytest.fixture
def launches(monkeypatch):
    calls = []

    def stub(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("launch stubbed")

    monkeypatch.setattr(tools, "_prepare_tool_launch", stub)
    monkeypatch.setattr(tools, "_account_confinement", lambda: None)
    return calls


@pytest.fixture(params = [True, False], ids = ["isolated", "no-isolation"])
def os_isolation(request, monkeypatch):
    from core.inference import os_sandbox

    monkeypatch.setattr(os_sandbox, "cached_tool_isolation", lambda _name: request.param)
    monkeypatch.setattr(tools, "workdir_scan_overran", lambda _session: False)
    return request.param


@pytest.mark.parametrize("case", sorted(_TOO_DEEP))
def test_too_deep_to_screen_is_high_risk(studio_home, case):
    name, arguments = _TOO_DEEP[case]
    assert tools.is_high_risk_tool_call(name, arguments) is True


@pytest.mark.parametrize("case", sorted(_TOO_DEEP))
def test_auto_asks_when_the_screen_cannot_analyse(studio_home, case):
    assert _decide("auto", *_TOO_DEEP[case]) == (True, False)


@pytest.mark.parametrize("case", sorted(_TOO_DEEP))
def test_off_never_runs_unscreened_code_without_the_sandbox(studio_home, os_isolation, case):
    name, arguments = _TOO_DEEP[case]
    prompted, strict = _decide("off", name, arguments)
    if not os_isolation:
        # No OS sandbox to require: "off" asks instead.
        assert (prompted, strict) == (True, False)
    else:
        assert (prompted, strict) == (False, True)


@pytest.mark.parametrize("case", sorted(_TOO_DEEP))
def test_ask_still_asks(studio_home, case):
    assert _decide("ask", *_TOO_DEEP[case]) == (True, False)


def test_only_recursion_fails_closed_in_the_policy():
    def deep(_name, _arguments):
        raise RecursionError

    def broken(_name, _arguments):
        raise ValueError("not a recursion")

    assert _decide("auto", "python", {"code": "1"}, is_high_risk = deep) == (True, False)
    with pytest.raises(ValueError):
        _decide("auto", "python", {"code": "1"}, is_high_risk = broken)


@pytest.mark.parametrize("host_access_approved", [False, True])
@pytest.mark.parametrize("case", sorted(_TOO_DEEP))
def test_an_approved_call_the_screen_cannot_analyse_is_not_run(
    studio_home, launches, case, host_access_approved
):
    name, arguments = _TOO_DEEP[case]
    result = tools.execute_tool(
        name, arguments, session_id = _SESSION, host_access_approved = host_access_approved
    )
    assert result.startswith("Error: the ") and "nested too deeply for the safety check" in result
    assert "not run" in result
    assert launches == []


def test_an_mcp_call_the_screen_cannot_analyse_is_not_sent(studio_home, monkeypatch):
    nested: dict = {}
    cur = nested
    for _ in range(900):
        cur["a"] = {}
        cur = cur["a"]
    sent = []
    monkeypatch.setattr(
        tools.mcp_servers_db, "get_server_for_tool", lambda server_id: sent.append(server_id)
    )
    name = f"{tools.MCP_TOOL_PREFIX}srv__get_item"
    assert tools.is_high_risk_tool_call(name, nested) is True
    result = tools.execute_tool(name, nested, session_id = _SESSION)
    assert "nested too deeply for the safety check" in result and "not made" in result
    assert sent == []


@pytest.mark.parametrize(
    "name, arguments, high_risk",
    [
        ("python", {"code": "print(1 + 1)"}, False),
        ("python", {"code": "x = 1" + " + 1" * 300}, False),
        ("python", {"code": "import os\nos.remove('/etc/passwd')"}, True),
        ("python", {"code": "open('/home/someoneelse/notes.txt', 'w').write('x')"}, True),
        ("terminal", {"command": "ls -la"}, False),
        ("terminal", {"command": "echo " + "[" * 300 + "]" * 300}, False),
        ("terminal", {"command": "rm -rf /"}, True),
    ],
)
def test_ordinary_calls_keep_their_decisions(studio_home, launches, name, arguments, high_risk):
    assert tools.is_high_risk_tool_call(name, arguments) is high_risk
    assert _decide("auto", name, arguments) == (high_risk, False)
    if not high_risk:
        launches.clear()
        tools.execute_tool(name, arguments, session_id = _SESSION)
        assert len(launches) == 1
