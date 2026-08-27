# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for scripts/lint_dynamic_exec.py.

The lint is what stops PR #1083's bug class coming back: an `exec`/`eval`/`compile`
whose first argument is built by interpolation, which is the shape that turns a value
into syntax. Everything already in the tree is allowlisted with a written
justification, so the live tree must pass and any new call must not.

CPU-only and network-free: the lint is stdlib only and runs as a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "lint_dynamic_exec.py"
ALLOWLIST = REPO_ROOT / "scripts" / "dynamic_exec_allowlist.json"


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output = True,
        text = True,
        cwd = REPO_ROOT,
    )


def test_the_script_exists():
    assert SCRIPT.is_file()
    assert ALLOWLIST.is_file()


def test_self_test_passes():
    proc = _run("--self-test")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_live_tree_passes():
    """Every interpolated dynamic-execution call in the tree is reviewed."""
    proc = _run()
    assert (
        proc.returncode == 0
    ), f"the live tree fails the lint:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


# --- the lint actually catches things ----------------------------------------


@pytest.mark.parametrize(
    "body, description",
    [
        ('exec(f"import {x}")', "f-string"),
        ('eval("torch." + x)', "concatenation"),
        ('compile("y = %s" % x, "<x>", "exec")', "%-format"),
        ('exec("a.{}".format(x))', ".format()"),
        # `compile`'s source is not positional-only, so an interpolated payload can
        # reach it with `node.args` empty.
        (
            'compile(source = f"y = {x}", filename = "<x>", mode = "exec")',
            "compile source= keyword",
        ),
        (
            'builtins.compile(source = f"y = {x}", filename = "<x>", mode = "exec")',
            "builtins.compile source= keyword",
        ),
    ],
)
def test_a_new_interpolated_call_fails(body, description, tmp_path):
    offender = tmp_path / "offender.py"
    offender.write_text(f"def f(x):\n    {body}\n")
    proc = _run("--paths", str(offender))
    assert proc.returncode == 1, f"{description} was not caught:\n{proc.stdout}"
    assert "offender.py" in proc.stderr


@pytest.mark.parametrize(
    "body",
    [
        "exec(source, globals())",
        "eval(name)",
        'exec("literal source")',
        'module.exec(f"{x}")',
        # Keyword resolution must not invent findings: a literal source, and a
        # sink call carrying no source at all, both stay quiet.
        'compile(source = "literal", filename = "<x>", mode = "exec")',
        'compile(filename = "<x>", mode = "exec")',
        "exec(**kwargs)",
    ],
)
def test_non_interpolated_calls_are_not_flagged(body, tmp_path):
    """Bare exec of generated source is the normal case here and must stay quiet."""
    clean = tmp_path / "clean.py"
    clean.write_text(f"def f(source, name, x, module, **kwargs):\n    {body}\n")
    proc = _run("--paths", str(clean))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


# --- the allowlist has to stay honest ----------------------------------------


def test_every_allowlist_entry_has_a_justification():
    # An empty allowlist is a legitimate state: it means no interpolated dynamic
    # execution is left in this repo at all.
    entries = json.loads(ALLOWLIST.read_text(encoding = "utf-8"))["allowed"]
    unjustified = [e for e in entries if e.get("reason", "").strip().upper() in ("", "REVIEW ME")]
    assert not unjustified, unjustified


def test_allowlist_entries_are_keyed_on_content_not_lines():
    """Line numbers drift; a justification keyed on one would silently detach."""
    entries = json.loads(ALLOWLIST.read_text(encoding = "utf-8"))["allowed"]
    assert all("hash" in e for e in entries)
    assert all("line" not in e for e in entries)


def test_editing_an_allowlisted_call_revokes_its_justification(tmp_path):
    """The property that makes the allowlist safe: change the call, lose the pass."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import lint_dynamic_exec as lint
    finally:
        sys.path.pop(0)

    original = tmp_path / "a.py"
    original.write_text('def f(x):\n    exec(f"import {x}")\n')
    edited = tmp_path / "b.py"
    edited.write_text('def f(x):\n    exec(f"import os; {x}")\n')
    reflowed = tmp_path / "c.py"
    reflowed.write_text('def f(x):\n    exec(\n        f"import {x}"\n    )\n')

    hashes = {p.name: lint.scan_file(p)[0]["hash"] for p in (original, edited, reflowed)}
    assert hashes["a.py"] != hashes["b.py"], "changing the payload kept the hash"
    assert hashes["a.py"] == hashes["c.py"], "reformatting invalidated the hash"


def _isolated_lint(tmp_path, allowlist):
    """A copy of the lint whose allowlist is `allowlist` (it reads the one beside it)."""
    script = tmp_path / "lint_dynamic_exec.py"
    script.write_text(SCRIPT.read_text(encoding = "utf-8"))
    (tmp_path / "dynamic_exec_allowlist.json").write_text(json.dumps(allowlist))
    return script


def test_a_stale_allowlist_entry_fails(tmp_path):
    """An entry matching nothing means the tree moved on; --update is required."""
    script = _isolated_lint(
        tmp_path,
        {
            "allowed": [
                {
                    "path": "gone.py",
                    "qualname": "f",
                    "sink": "exec",
                    "kind": "f-string",
                    "hash": "0" * 16,
                    "reason": "was reviewed once",
                }
            ]
        },
    )
    # A full scan, not --paths: staleness is only meaningful over the whole tree,
    # and this copy's tree contains none of the package directories.
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 1
    assert "no longer matches any call" in proc.stderr


def test_paths_mode_does_not_report_the_rest_of_the_tree_as_stale(tmp_path):
    """Scanning one file must not invalidate every justification outside it."""
    clean = tmp_path / "clean.py"
    clean.write_text("def f(source):\n    exec(source)\n")
    proc = _run("--paths", str(clean))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_an_unjustified_allowlist_entry_fails(tmp_path):
    """`--update` seeds entries with REVIEW ME; leaving one is a failure, not a pass."""
    offender = tmp_path / "offender.py"
    offender.write_text('def f(x):\n    exec(f"import {x}")\n')

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import lint_dynamic_exec as lint
    finally:
        sys.path.pop(0)
    finding = lint.scan_file(offender)[0]

    script = _isolated_lint(
        tmp_path,
        {
            "allowed": [
                {
                    "path": finding["path"],
                    "qualname": finding["qualname"],
                    "sink": finding["sink"],
                    "kind": finding["reason"],
                    "hash": finding["hash"],
                    "reason": "REVIEW ME",
                }
            ]
        },
    )
    proc = subprocess.run(
        [sys.executable, str(script), "--paths", str(offender)],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 1
    assert "no justification" in proc.stderr


# Grouped by what each case proves, not by the review round that found it.
# Every description is unique across the whole file, so a new case that
# duplicates an existing one is a merge conflict rather than a silent overwrite.

# The callee is written some other way and still resolves to the builtin.
_SINK_IS_STILL_RESOLVED = {
    "a stdlib alias imported below a function still binds for it": 'def f(name):\n    exec(clean(f"result = {name}"))\nfrom textwrap import dedent as clean\nf("42")\n',
    "a signed zero and a signed step still slice the whole string": 'def f(name, scope):\n    exec(f"result = {name}"[-0::+1], scope)\n',
    "a template held by a name still splices its field": 'def f(user):\n    template = "import {m}"\n    exec(template.format(m = user))\n',
    "a field between escaped braces is still a field": 'def f(name):\n    exec("{{{x}}}".format(x = name))\n',
    "a directly imported ast.parse still preserves its source": 'from ast import parse\ndef f(name):\n    compile(parse(f"import {name}"), "<x>", "exec")\n',
    "getattr reads the implicit builtins module too": 'def f(name):\n    getattr(__builtins__, "exec")(f"import {name}")\n',
    "an aliased importlib still names the builtins module": 'import importlib as il\ndef f(name):\n    il.import_module("builtins").exec(f"import {name}")\n',
    "an aliased contextlib still hands its argument over": 'import builtins\nimport contextlib as ctx\ndef f(name):\n    with ctx.nullcontext(builtins.exec) as run:\n        run(f"import {name}")\n',
    "a literal sequence subject hands a sink to its capture": 'import builtins\ndef f(name):\n    match [builtins.exec]:\n        case [run]:\n            run(f"import {name}")\n',
    "an aliased textwrap wrapper still hands the source through": 'from textwrap import dedent as clean\ndef f(name):\n    exec(clean(f"import {name}"))\n',
    "operator.mod is percent formatting written as a call": 'import operator\ndef f(name):\n    exec(operator.mod("import %s", name))\n',
    "the in-place operator spellings concatenate too": 'import operator\ndef f(name):\n    exec(operator.iadd("import ", name))\n',
    "__getitem__ is the mapping subscript written as a call": 'import builtins\ndef f(name):\n    builtins.__dict__.__getitem__("exec")(f"import {name}")\n',
    "a lazy type alias value is evaluated when the alias is read": 'type Payload = (exec(f"import {NAME}") or int)\n',
    "the implicit __builtins__ names the sink as an attribute": 'def f(name):\n    __builtins__.exec(f"import {name}")\n',
    "an aliased codeop still compiles the source it was given": 'import codeop as co\ndef f(name):\n    exec(co.compile_command(f"import {name}"))\n',
    "a mixed literal loop keeps the sink over the constructor": 'import builtins\ndef f(name):\n    for run in [builtins.str, builtins.exec]:\n        run(f"import {name}")\n',
    "a statically false while never takes the builtin away": 'import builtins\ndef f(name, condition):\n    run = builtins.exec\n    while False:\n        run = print\n    run(f"import {name}")\n',
    "a decidable boolean filter binds no comprehension walrus": 'import builtins\ndef f(name):\n    run = builtins.exec\n    [(run := print) for _ in [0] if True and False]\n    run(f"import {name}")\n',
    "a dict constructor literal answers get on the callee side": 'def f(name):\n    dict(run = exec).get("run")(f"import {name}")\n',
    "functools under a module alias still binds the sink": 'import functools as ft\ndef f(name):\n    ft.partial(exec, f"import {name}")()\n',
    "nullcontext hands its argument to the with target": 'import contextlib, builtins\ndef f(name):\n    with contextlib.nullcontext(builtins.exec) as run:\n        run(f"import {name}")\n',
    "an imported alias of functools.partial binds the sink": 'from functools import partial as bind\ndef f(name):\n    bind(exec, f"import {name}")()\n',
    "an inline builtins import is a getattr owner": 'def f(name):\n    getattr(__import__("builtins"), "exec")(f"import {name}")\n',
    "a builtins-qualified getattr names the sink": 'import builtins\ndef f(name):\n    builtins.getattr(builtins, "exec")(f"import {name}")\n',
    "a computed key keeps the get default reachable as a callee": 'def f(name, key):\n    {key: print}.get("run", exec)(f"import {name}")\n',
    "a later module-level def does not apply to earlier calls": 'name = "ok"\nexec(f"import {name}")\ndef exec(source):\n    pass\n',
    "a literal key on the builtins dict names the sink": 'import builtins\ndef f(name):\n    builtins.__dict__["exec"](f"import {name}")\n',
    "a partial assigned to a name keeps the source bound into it": 'import functools\ndef f(name):\n    run = functools.partial(exec, f"import {name}")\n    run()\n',
    "a partial that binds only the sink takes its source at call time": 'import functools\ndef f(name):\n    functools.partial(exec)(f"import {name}")\n',
    "a sink assigned below the function still binds for it": "import builtins\ndef f(name):\n    run(f'import {name}')\nrun = builtins.exec\n",
    "a walrus callee is called before the name it binds means anything": "import builtins\ndef f(name):\n    (run := builtins.exec)(f'import {name}')\n",
    "a walrus inside a lambda is local to the lambda": "def f(name):\n    exec(f'import {name}')\n    g = lambda: (exec := print)\n",
    "an inline builtins import names the sink": 'def f(name):\n    __import__("builtins").exec(f"import {name}")\n',
    "deleting a global shadow puts the builtin back": "exec = print\ndef f(name):\n    global exec\n    del exec\n    exec(f'import {name}')\n",
    "functools.partial binds the sink and its source": 'import functools\ndef f(name):\n    functools.partial(exec, f"import {name}")()\n',
    "get on a literal mapping selects the callee": "import builtins\ndef f(name):\n    {'run': builtins.exec}.get('run')(f\"import {name}\")\n",
    "get on the builtins mapping returns the sink": 'import builtins\ndef f(name):\n    builtins.__dict__.get("exec")(f"import {name}")\n',
    "the implicit builtins mapping holds the sink": "def f(name):\n    __builtins__['exec'](f\"import {name}\")\n",
}


@pytest.mark.parametrize("description", sorted(_SINK_IS_STILL_RESOLVED))
def test_a_sink_reached_through_another_spelling_is_reported(description, tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(_SINK_IS_STILL_RESOLVED[description])
    proc = _run("--paths", str(sample))
    assert proc.returncode == 1, f"{description}:\n{proc.stdout}\n{proc.stderr}"


# `str`, `bytes` and friends reached indirectly still pass the source through.
_CONSTRUCTOR_IS_STILL_RESOLVED = {
    "a boolean callee names every constructor it can select": 'def f(name):\n    exec((memoryview and str)(f"result = {name}"))\n',
    "the decoding form of str hands the text back": 'def f(name):\n    exec(str(f"import {name}".encode(), "utf-8"))\n',
    "an undecidable constructor choice checks both arms": 'def f(name, flag):\n    exec((memoryview if flag else str)(f"import {name}"))\n',
    "a builtins star import hands back the conversions too": "def f(name):\n    str = print\n    from builtins import *\n    exec(str(f'import {name}'))\n",
    "a byte constructor over already encoded source reaches the sink": 'def f(name):\n    exec(bytes(f"import {name}".encode()))\n    exec(bytearray(f"import {name}".encode()))\n    exec(memoryview(f"import {name}".encode()))\n',
    "a class-body global import restores the constructor globally": 'str = print\nclass C:\n    global str\n    from builtins import str\ndef f(name):\n    exec(str(f"import {name}"))\n',
    "a constructor imported below the function still binds for it": "def f(name):\n    exec(s(f'import {name}'))\nfrom builtins import str as s\n",
    "a constructor reached through __call__ still converts": 'def f(name):\n    exec(str.__call__(f"import {name}"))\n',
    "a constructor selected from the builtins mapping still converts": "import builtins\ndef f(name):\n    exec(builtins.__dict__['str'](f\"import {name}\"))\n",
    "a conversion called through __call__ still converts": 'def f(name):\n    exec(f"import {name}".encode.__call__())\n',
    "a lambda default names a constructor": 'import builtins\ng = lambda name, text = builtins.str: exec(text(f"import {name}"))\n',
    "a lambda parameter defaulting to a constructor carries it": "import builtins\nf = (lambda name, s = builtins.str: exec(s(f'import {name}')))\n",
    "a literal loop states the constructor it binds": "import builtins\ndef f(name):\n    for s in [builtins.str]:\n        exec(s(f'import {name}'))\n",
    "a parameter defaulting to a constructor carries the conversion": "import builtins\ndef f(name, text = builtins.str):\n    exec(text(f'import {name}'))\n",
    "a walrus constructor callee converts right there": "def f(name):\n    exec((text := str)(f'import {name}'))\n",
    "an eager comprehension walrus keeps a prefixed alias": 'def f(name):\n    [(text := str) for _ in [0]]\n    exec(text(f"import {name}"))\n',
    "an irrefutable match capture binds the constructor it matched": "import builtins\ndef f(name):\n    match builtins.str:\n        case s:\n            exec(s(f'import {name}'))\n",
    "both arms of a conditional constructor preserve the source": 'def f(name, flag):\n    exec((str if flag else format)(f"import {name}"))\n',
    "getattr names a constructor as plainly as an attribute": 'import builtins\ndef f(name):\n    exec(getattr(builtins, "str")(f"import {name}"))\n',
    "getattr with a default still returns the attribute that exists": "import builtins\ndef f(name):\n    getattr(builtins, 'exec', print)(f'import {name}')\n",
    "memoryview over bytes reaches the sink": 'def f(name):\n    exec(memoryview(bytes(f"import {name}", "utf-8")))\n',
}


@pytest.mark.parametrize("description", sorted(_CONSTRUCTOR_IS_STILL_RESOLVED))
def test_a_text_constructor_reached_through_another_spelling_is_reported(description, tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(_CONSTRUCTOR_IS_STILL_RESOLVED[description])
    proc = _run("--paths", str(sample))
    assert proc.returncode == 1, f"{description}:\n{proc.stdout}\n{proc.stderr}"


# The built string survives a lookup, an operator, a wrapper or a spread.
_SOURCE_SURVIVES_THE_SHAPE = {
    "two calls that read the same source are not the same value": 'def f(values, scope):\n    exec(f"{next(values)}".removeprefix(f"{next(values)}"), scope)\n',
    "an identity translation table changes nothing": 'def f(name):\n    exec(f"import {name}".translate({0: 0}))\n',
    "an ordinary async context manager still runs its body": 'async def f(name, manager):\n    async with manager as payload:\n        exec(f"import {name}")\n',
    "format_map with an empty mapping returns the receiver": 'def f(name):\n    exec(f"import {name}".format_map({}))\n',
    "an opaque expansion leaves compile its source keyword": 'def f(name, args = ()):\n    exec(compile(*args, source = f"import {name}", filename = "<x>", mode = "exec"))\n',
    "nullcontext hands its argument to the as target": 'import contextlib\ndef f(name):\n    with contextlib.nullcontext(f"import {name}") as payload:\n        exec(payload)\n',
    "an empty expansion does not hide the first spread key": 'def f(name):\n    exec(*{**{}, f"import {name}": None})\n',
    "a literal template reached through a name is still a template": 'def f(name):\n    template = "import MODULE"\n    exec(template.replace("MODULE", name))\n',
    "codeop compiles the source it was handed": 'import codeop\ndef f(name):\n    exec(codeop.compile_command(f"import {name}"))\n',
    "operator.add is concatenation under another name": 'import operator\ndef f(name):\n    exec(operator.add("import ", name))\n',
    "operator.concat is the same concatenation": 'import operator\ndef f(name):\n    exec(operator.concat("import ", name))\n',
    "__add__ is concatenation written as a call": 'def f(name):\n    exec("print(".__add__(name).__add__(")"))\n',
    "__mod__ is the percent operator written as a call": 'def f(name):\n    exec("print(%r)".__mod__(name))\n',
    "__str__ hands back the same string": 'def f(name):\n    exec(f"import {name}".__str__())\n',
    "a computed key does not erase an earlier matching entry": "def f(name, key):\n    exec({'source': f'import {name}', key: 'pass'}['source'])\n",
    "a computed key nested under a double star keeps the get default reachable": "def f(name, key):\n    exec({**{key: 'pass'}}.get('source', f\"import {name}\"))\n",
    "a computed key that cannot match still leaves the earlier value": "def f(name):\n    key = 'other'\n    exec({'s': f'import {name}', key: 'pass'}['s'])\n",
    "a computed key under a double star does not erase the outer value": "def f(name, key):\n    exec({'s': f'import {name}', **{key: 'pass'}}['s'])\n",
    "a dict constructor literal answers get": 'def f(name):\n    exec(dict(source = f"import {name}").get("source"))\n',
    "a lazy generator walrus binds nothing": 'def f(name):\n    payload = f"import {name}"\n    generator = ((payload := "pass") for _ in [0])\n    exec(payload)\n',
    "a literal get returns its default when the key is absent": 'def f(name):\n    exec({}.get("source", f"import {name}"))\n',
    "a literal loop that can break leaves some element bound": "import builtins\ndef f(name):\n    run = print\n    for run in [builtins.exec]:\n        break\n    run(f'import {name}')\n",
    "a text-preserving wrapper takes its text by keyword": 'import textwrap\ndef f(name):\n    exec(textwrap.dedent(text = f"import {name}"))\n',
    "a walrus in a definition header binds where it is written": 'import builtins\ndef f(name):\n    run(f"import {name}")\ndef g(value = (run := builtins.exec)):\n    pass\n',
    "a written out getitem selects the source": 'def f(name):\n    exec({"source": f"import {name}"}.__getitem__("source"))\n',
    "an empty comprehension binds no walrus at all": 'import builtins\ndef f(name):\n    run = builtins.exec\n    [(run := print) for _ in []]\n    run(f"import {name}")\n',
    "an opaque double-star expansion does not erase an earlier entry": "def f(name):\n    exec({'source': f\"import {name}\", **dict(other = 'pass')}['source'])\n",
    "an opaque star expansion may contribute nothing": 'def f(name, args = ()):\n    exec(*args, f"import {name}")\n',
    "ast.parse keeps the source it parsed": 'import ast\ndef f(name):\n    compile(ast.parse(f"print({name!r})"), "<x>", "exec")\n',
    "ast.parse takes its source as a keyword": 'import ast\ndef f(name):\n    compile(ast.parse(source = f"import {name}"), "<x>", "exec")\n',
    "augmented multiplication can build a string": 'def f(name):\n    payload = 1\n    payload *= f"import {name}"\n    exec(payload)\n',
    "dict merges a positional mapping with its keywords": "def f(name):\n    compile(**dict({'source': f\"import {name}\"}, filename = '<x>', mode = 'exec'))\n",
    "format with nothing to splice still returns the receiver": "def f(name):\n    exec(f'import {name}'.format())\n",
    "get on a literal mapping reads the same value a subscript would": "def f(name):\n    exec({'s': f'import {name}'}.get('s'))\n",
    "iterating a literal mapping yields its keys": 'def f(name):\n    for payload in {f"import {name}": None}:\n        exec(payload)\n',
    "spreading a literal mapping passes its first key": "def f(name):\n    exec(*{f'import {name}': None})\n",
    "textwrap.dedent keeps what the f-string built": "import textwrap\ndef f(name):\n    exec(textwrap.dedent(f'import {name}'))\n",
    "the unbound descriptor spelling of __str__ preserves the source": 'def f(name):\n    exec(str.__str__(f"import {name}"))\n',
}


@pytest.mark.parametrize("description", sorted(_SOURCE_SURVIVES_THE_SHAPE))
def test_a_source_that_survives_its_wrapper_is_reported(description, tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(_SOURCE_SURVIVES_THE_SHAPE[description])
    proc = _run("--paths", str(sample))
    assert proc.returncode == 1, f"{description}:\n{proc.stdout}\n{proc.stderr}"


# One level of local indirection: an alias of a name that holds a built string.
_TAINT_TRAVELS_THROUGH_A_BINDING = {
    "an unpacking binds each element before the next target runs": 'def f(name, table):\n    payload, table[exec(payload)] = (f"import {name}", None)\n',
    "an augmented subscript evaluates its target first": 'def f(name, table):\n    payload = f"import {name}"\n    table[exec(payload)] += (payload := "pass")\n',
    "a dict evaluates each key beside its own value": 'def f(name):\n    {0: (payload := f"import {name}"), exec(payload): None}\n',
    "a while suite ends at an unconditional break": 'def f(name):\n    payload = f"import {name}"\n    while True:\n        break\n        payload = "pass"\n    exec(payload)\n',
    "a jump under a decidable test ends the suite as well": 'def f(name):\n    payload = f"import {name}"\n    for _ in [0]:\n        if True:\n            continue\n        payload = "pass"\n    exec(payload)\n',
    "a statically false case guard runs no body": 'def f(name, value):\n    payload = f"import {name}"\n    match value:\n        case _ if False:\n            payload = "pass"\n    exec(payload)\n',
    "a comprehension that runs no pass binds no walrus": 'def f(name):\n    payload = f"import {name}"\n    [(payload := "pass") for _ in []]\n    exec(payload)\n',
    "an empty display is a false condition": 'def f(name):\n    payload = f"import {name}"\n    [] and (payload := "pass")\n    exec(payload)\n',
    "a literal sequence subject pairs with its capture": 'def f(name):\n    match [f"import {name}"]:\n        case [payload]:\n            exec(payload)\n',
    "an annotated literal is still a template for replace": 'def f(name):\n    template: str = "import MODULE"\n    exec(template.replace("MODULE", name))\n',
    "a positive augmented repetition keeps the built source": 'def f(name):\n    payload = f"import {name}"\n    payload *= 2\n    exec(payload)\n',
    "an assignment below an unconditional break never runs": 'def f(name):\n    payload = f"import {name}"\n    for _ in [0]:\n        break\n        payload = "pass"\n    exec(payload)\n',
    "an unknown operand before a decisive constant still decides": 'import builtins\ndef f(name, flag):\n    run = builtins.exec\n    while flag and False:\n        run = print\n    run(f"import {name}")\n',
    "a literal loop element carries the taint it holds": 'def f(name):\n    payload = f"import {name}"\n    for source in [payload]:\n        exec(source)\n',
    "a short-circuited walrus never clears the taint": 'def f(name):\n    payload = f"import {name}"\n    False and (payload := "pass")\n    exec(payload)\n',
    "the final loop body wins over the last element": 'def f(name):\n    for payload in ["pass"]:\n        payload = f"import {name}"\n    exec(payload)\n',
    "a walrus alias carries the taint": 'def f(name):\n    payload = f"import {name}"\n    (alias := payload)\n    exec(alias)\n',
    "an annotated alias carries the taint": 'def f(name):\n    payload = f"import {name}"\n    alias: str = payload\n    exec(alias)\n',
    "an unpacked alias carries the taint": 'def f(name):\n    payload = f"import {name}"\n    alias, ignored = payload, None\n    exec(alias)\n',
    "taint travels through a plain alias assignment": 'def f(name):\n    payload = f"import {name}"\n    alias = payload\n    exec(alias)\n',
}


@pytest.mark.parametrize("description", sorted(_TAINT_TRAVELS_THROUGH_A_BINDING))
def test_taint_carried_through_a_binding_is_reported(description, tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(_TAINT_TRAVELS_THROUGH_A_BINDING[description])
    proc = _run("--paths", str(sample))
    assert proc.returncode == 1, f"{description}:\n{proc.stdout}\n{proc.stderr}"


# Shadowed, unreachable, unbound or unable to build a string at all.
_NOTHING_REACHES_THE_SINK = {
    "a field-free template held by a name ignores its arguments": 'def f(user):\n    template = "pass"\n    exec(template.format(user))\n',
    "a field-free format_map template ignores its mapping": 'def f(name):\n    exec("pass".format_map({"x": name}))\n',
    "compile without a filename and a mode raises first": 'def f(name):\n    compile(f"import {name}")\n',
    "a bare partial is only functools.partial when imported": 'def partial(callback, source):\n    return lambda: None\ndef f(name):\n    partial(exec, f"import {name}")()\n',
    "a dedent on something other than textwrap is not the helper": 'def f(cleaner, name):\n    exec(cleaner.dedent(f"import {name}"))\n',
    "a class-body lambda parameter shadows the builtin": 'class C:\n    f = lambda exec, name: exec(f"import {name}")\n',
    "async with cannot enter a synchronous nullcontext": 'import builtins, contextlib\nasync def f(name):\n    async with contextlib.nullcontext(builtins.exec) as run:\n        run(f"import {name}")\n',
    "False is a zero repetition count": 'def f(name):\n    exec(f"import {name}" * False)\n',
    "two written-out containers add to a list, not to source": 'def f(name):\n    exec([f"import {name}"] + [])\n',
    "an async loop over a synchronous literal never reaches its else": 'async def f(name):\n    async for _ in []:\n        pass\n    else:\n        exec(f"import {name}")\n',
    "a deleted nonlocal cell is empty, not the builtin": 'def outer(name):\n    exec = print\n    def inner():\n        nonlocal exec\n        del exec\n        exec(f"import {name}")\n    return inner\n',
    "a zero precision format spec keeps nothing": 'def f(name):\n    exec(format(f"import {name}", ".0"))\n',
    "removing the receiver as its own prefix leaves nothing": 'def f(name):\n    exec(f"import {name}".removeprefix(f"import {name}"))\n',
    "a written-out expansion supplies the source itself": 'def f(name):\n    exec(compile(*["pass"], f"<{name}>", "exec"))\n',
    "an unpacked number stays a number": "def f(name):\n    payload, = (1,)\n    payload += 2\n    exec(payload)\n",
    "a walrus number stays a number": "def f(name):\n    (payload := 1)\n    payload += 2\n    exec(payload)\n",
    "dict takes at most one positional argument": 'def f(name):\n    compile(**dict({"source": f"import {name}"}, {"filename": "<x>", "mode": "exec"}))\n',
    "the decoding form of str rejects text": 'def f(name):\n    exec(str(f"import {name}", "utf-8"))\n',
    "a local contextlib is not the stdlib module": 'def f(contextlib, name):\n    with contextlib.nullcontext(f"import {name}") as payload:\n        exec(payload)\n',
    "a literal template with no field ignores its arguments": 'def f(name):\n    exec("pass".format(name))\n',
    "a local codeop is not the stdlib module": 'def f(codeop, name):\n    exec(codeop.compile_command(f"import {name}"))\n',
    "a local operator is not the stdlib module": 'def f(operator, name):\n    exec(operator.add("import ", name))\n',
    "assigning the builtins module drops the earlier sink alias": 'import builtins\ndef f(name):\n    run = builtins.exec\n    run = builtins\n    run(f"import {name}")\n',
    "an augmented repetition by zero builds nothing": 'def f(name):\n    payload = f"import {name}"\n    payload *= 0\n    exec(payload)\n',
    "a folded unary condition drops the constructor arm": 'def f(name):\n    exec((str if not True else print)(f"import {name}"))\n',
    "an except target takes a prefixed alias away": 'from builtins import str as text\ndef f(name):\n    try:\n        pass\n    except Exception as text:\n        exec(text(f"import {name}"))\n',
    "async for cannot iterate a synchronous literal": 'async def f(name):\n    async for payload in [f"import {name}"]:\n        exec(payload)\n',
    "a constructor assignment clears the earlier sink": 'import builtins\ndef f(name):\n    run = builtins.exec\n    run = builtins.str\n    run(f"import {name}")\n',
    "str of visibly bytes is a repr, not the text": 'def f(name):\n    exec(str(f"import {name}".encode()))\n',
    "a zero repetition builds an empty string": 'def f(name):\n    exec(f"import {name}" * 0)\n',
    "a negative repetition builds an empty string": 'def f(name):\n    exec(f"import {name}" * -1)\n',
    "a partial method on some other object is not functools": 'class R:\n    def partial(self, *a):\n        return lambda: None\ndef f(name, runner):\n    runner.partial(exec, f"import {name}")()\n',
    "the last element wins when the body does not rebind": 'def f(name):\n    for payload in [f"import {name}"]:\n        payload = "pass"\n    exec(payload)\n',
    "an opaque receiver is not a literal template": 'import inspect\ndef f(name, target):\n    exec(inspect.getsource(target).replace("MODULE", name))\n',
    "a rebound partial alias is no longer partial": 'from functools import partial as bind\ndef f(name):\n    bind = print\n    bind(exec, f"import {name}")()\n',
    "operator.add on numbers builds no source": "import operator\ndef f():\n    exec(operator.add(1, 2))\n",
    "a composite default resolving to a shadowed name is not the builtin": 'from re import compile\ndef f(name, run = [compile][0]):\n    run(f"^{name}$")\n',
    "a constant short-circuits a boolean callee": "def f(name):\n    (None and exec)(f'import {name}')\n",
    "a literal condition decides a conditional expression": "def f(name):\n    exec('pass' if True else f'import {name}')\n",
    "a literal false condition decides a conditional callee": 'import builtins\ndef f(name):\n    (builtins.exec if False else print)(f"hello {name}")\n',
    "a literal loop over a shadowed spelling is not the builtin": 'from re import compile\ndef f(name):\n    for run in [compile]:\n        run(f"^{name}$")\n',
    "a literal true condition decides a conditional callee": "def f(name):\n    (print if True else exec)(f'import {name}')\n",
    "a local dict is not the mapping constructor": 'def dict(**kwargs):\n    return {"object": "pass"}\ndef f(name):\n    exec(str(**dict(object = f"import {name}")))\n',
    "a local rebinding hides an enclosing constructor alias": 'from builtins import str as text\ndef f(name):\n    text = print\n    exec(text(f"import {name}"))\n',
    "a loop whose body always breaks cannot run its else": 'def f(name):\n    for payload in [f"import {name}"]:\n        break\n    else:\n        exec(payload)\n',
    "a match subject that is shadowed is not the builtin": 'from re import compile\ndef f(name):\n    match compile:\n        case run:\n            run(f"^{name}$")\n',
    "a truthy literal short-circuits the rest of an or": "def f(name):\n    exec('pass' or f'import {name}')\n",
    "a type alias rebinds its name": 'from builtins import exec as run\ndef f(name):\n    type run = int\n    run(f"import {name}")\n',
    "a visibly empty loop never runs its body": 'def f(name):\n    for _ in []:\n        exec(f"import {name}")\n',
    "an alias bound only in a dead branch is not the builtin": 'if True:\n    from re import compile as comp\nelse:\n    from builtins import compile as comp\ndef f(name):\n    comp(f"{name}")\n',
    "an annotated numeric assignment stays numeric": "def f():\n    payload: int = 1\n    payload += 2\n    exec(payload)\n",
    "an assignment copying a shadowed spelling is not the builtin": 'from re import compile\nrun = compile\ndef f(name):\n    run(f"^{name}$")\n',
    "an attribute call on a parameter named builtins is not the module": 'def f(builtins, name):\n    builtins.exec(f"import {name}")\n',
    "byte constructors without an encoding never reach the sink": 'def f(name):\n    exec(bytes(f"import {name}"))\n    exec(memoryview(f"import {name}"))\n',
    "bytes without an encoding keyword cannot reach the sink": 'def f(name):\n    exec(bytes(source = f"import {name}", errors = "ignore"))\n',
    "decode on a visibly string receiver raises before the sink": 'def f(name):\n    exec(f"import {name}".decode())\n',
    "deleting a module alias unbinds it": 'import builtins as b\ndel b\ndef f(name):\n    b.exec(f"import {name}")\n',
    "deleting a name makes it local for the whole function": "def f(name):\n    exec(f'import {name}')\n    del exec\n",
    "explicit integer addition builds no source": "def f():\n    exec((1).__add__(2))\n",
    "explicit integer modulo builds no source": "def f():\n    exec((7).__mod__(2))\n",
    "format_map without a mapping raises before the sink": 'def f(name):\n    exec(f"import {name}".format_map())\n',
    "getattr on a parameter named builtins is not the module": 'def f(builtins, name):\n    getattr(builtins, "exec")(f"import {name}")\n',
    "modulo between two numbers is arithmetic": "def f():\n    payload = 5\n    payload %= 2\n    exec(payload)\n",
    "not applied to a constant decides a conditional": 'def f(name):\n    exec("pass" if not False else f"import {name}")\n',
    "not applied to a constant decides a conditional callee": 'def f(name):\n    (exec if not True else print)(f"import {name}")\n',
    "rebinding a constructor alias stops the unwrapping": 'from builtins import str as text\ntext = lambda _: "pass"\ndef f(name):\n    exec(text(f"import {name}"))\n',
}


@pytest.mark.parametrize("description", sorted(_NOTHING_REACHES_THE_SINK))
def test_code_that_cannot_execute_a_built_string_is_quiet(description, tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(_NOTHING_REACHES_THE_SINK[description])
    proc = _run("--paths", str(sample))
    assert proc.returncode == 0, f"{description}:\n{proc.stdout}\n{proc.stderr}"


# A magic or interpreter invocation that really does execute the cell body.
_NOTEBOOK_RUNS_PYTHON = {
    "a shell escape running python -c carries a program": '{"cells": [{"cell_type": "code", "source": "!python -c \'name=input(); exec(f\\"import {name}\\")\'\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "a bundled short option carries its attached program": '{"cells": [{"cell_type": "code", "source": "%%script python -uc\'name=\\"x\\";exec(f\\"import {name}\\")\'\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "an assignment-form %time binds the value it timed": '{"cells": [{"cell_type": "code", "source": "name = input()\\npayload = %time f\\"import {name}\\"\\nexec(payload)\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "env -S carries the interpreter command inside its value": '{"cells": [{"cell_type": "code", "source": "%%script env -S \\"python -c \'name=input(); exec(f\\\\\\"import {name}\\\\\\")\'\\"\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "prefix help syntax does not hide the rest of the cell": '{"cells": [{"cell_type": "code", "source": "?len\\nname = input()\\nexec(f\\"import {name}\\")\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "python -i reads the cell body from stdin": '{"cells": [{"cell_type": "code", "source": "%%script python -i helper.py\\nname = input()\\nexec(f\\"import {name}\\")\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "timeit setup code runs before the timed statement": '{"cells": [{"cell_type": "code", "source": "name = input()\\n%timeit -n 1 -s \\"exec(f\'import {name}\')\\" pass\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
}


@pytest.mark.parametrize("description", sorted(_NOTEBOOK_RUNS_PYTHON))
def test_a_notebook_cell_that_runs_python_is_reported(description, tmp_path):
    sample = tmp_path / "sample.ipynb"
    sample.write_text(_NOTEBOOK_RUNS_PYTHON[description])
    proc = _run("--paths", str(sample))
    assert proc.returncode == 1, f"{description}:\n{proc.stdout}\n{proc.stderr}"


# The cell body is data: no interpreter reads it.
_NOTEBOOK_IS_INERT = {
    "a bundled help flag never reads the cell body": '{"cells": [{"cell_type": "code", "source": "%%script python -uh\\nname = input()\\nexec(f\\"import {name}\\")\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "help wins over interactive mode": '{"cells": [{"cell_type": "code", "source": "%%script python -i --help\\nname = input()\\nexec(f\\"import {name}\\")\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "python --help never reads the cell body": '{"cells": [{"cell_type": "code", "source": "%%script python --help\\nname = input()\\nexec(f\\"import {name}\\")\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
    "python -V never reads the cell body": '{"cells": [{"cell_type": "code", "source": "%%script python -V\\nname = input()\\nexec(f\\"import {name}\\")\\n", "metadata": {}, "outputs": [], "execution_count": null}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}',
}


@pytest.mark.parametrize("description", sorted(_NOTEBOOK_IS_INERT))
def test_a_notebook_cell_that_never_runs_is_quiet(description, tmp_path):
    sample = tmp_path / "sample.ipynb"
    sample.write_text(_NOTEBOOK_IS_INERT[description])
    proc = _run("--paths", str(sample))
    assert proc.returncode == 0, f"{description}:\n{proc.stdout}\n{proc.stderr}"


def test_the_scanner_runs_without_the_match_node_types():
    """`ast.Match` arrived in 3.10 and both packages declare `requires-python >= 3.9`.

    Naming it directly made every isinstance tuple holding one raise AttributeError
    there, so the documented command could not scan anything at all. The 3.9 surface is
    simulated by hiding those attributes rather than by needing a 3.9 interpreter.
    """
    import ast
    import importlib.util

    hidden = [
        "Match",
        "MatchAs",
        "MatchStar",
        "MatchClass",
        "MatchMapping",
        "MatchOr",
        "MatchSequence",
        "match_case",
    ]
    saved = {name: getattr(ast, name) for name in hidden if hasattr(ast, name)}
    for name in saved:
        delattr(ast, name)
    try:
        spec = importlib.util.spec_from_file_location("lint_dynamic_exec_39", str(SCRIPT))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        findings = module.scan_file(SCRIPT)
    finally:
        for name, value in saved.items():
            setattr(ast, name, value)
    assert isinstance(findings, list)
