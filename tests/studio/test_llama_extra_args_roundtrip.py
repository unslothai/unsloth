# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavioural round-trip tests for the Custom llama-server Args field.

Typed text is parsed to argv, rendered back on every re-render, re-parsed on the
next blur and normalised again on save. Any asymmetry between those four steps
silently launches llama-server with different arguments than the user entered,
which source greps cannot catch, so run the real module under node (same harness
as test_chat_preset_builtin_invariants.py) and assert the round trips.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


MODULE = _source_path("studio/frontend/src/features/model-picker/model-config/llama-extra-args.ts")
TEMP = WORKDIR / "temp" / "llama_extra_args_roundtrip"


def _run(body: str):
    """Execute ``body`` against the real module; return its last JSON line."""
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not MODULE.exists():
        pytest.skip("studio model-picker sources not present")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")
    TEMP.mkdir(parents = True, exist_ok = True)
    module_path = os.path.relpath(MODULE, TEMP).replace("\\", "/")
    script = TEMP / "run.mts"
    script.write_text(
        "// @ts-nocheck\n"
        "import {\n"
        "  parseLlamaExtraArgsInput,\n"
        "  formatLlamaExtraArgs,\n"
        "  normalizeLlamaExtraArgs,\n"
        "  llamaExtraArgsForLoad,\n"
        f'}} from "{module_path}";\n' + textwrap.dedent(body)
    )
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(TEMP),
        capture_output = True,
        text = True,
        timeout = 60,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_blur_round_trip_is_stable_for_quoted_values():
    """format -> parse must be a fixed point, else every blur mutates the args.

    A formatter that drops quoting turns {"a": "b"} into two mangled tokens.
    """
    out = _run(
        """
        const typed = [
          `--chat-template-kwargs '{"date_string": "July 24"}'`,
          `--chat-template-kwargs '{"date_string":"July"}'`,
          `--chat-template-file "C:\\\\Users\\\\Me\\\\my template.jinja"`,
          `--cpu-moe --no-mmap`,
          `--grammar "root ::= \\\\"yes\\\\" | \\\\"no\\\\""`,
        ];
        const rows = typed.map((text) => {
          const once = parseLlamaExtraArgsInput(text);
          const twice = parseLlamaExtraArgsInput(formatLlamaExtraArgs(once));
          const thrice = parseLlamaExtraArgsInput(formatLlamaExtraArgs(twice));
          return { once, twice, thrice };
        });
        console.log(JSON.stringify({ rows }));
        """
    )
    for row in out["rows"]:
        assert row["twice"] == row["once"], row
        assert row["thrice"] == row["once"], row


def test_quoted_windows_path_keeps_its_backslashes():
    """Only a quote or another backslash escapes, as in a shell.

    Consuming every backslash turns "C:\\Program Files\\t.jinja" into
    "C:Program Filest.jinja", a path the user never typed.
    """
    out = _run(
        """
        console.log(JSON.stringify({
          quoted: parseLlamaExtraArgsInput('--chat-template-file "C:\\\\Users\\\\Me\\\\a b.jinja"'),
          bare: parseLlamaExtraArgsInput("--chat-template-file C:\\\\Users\\\\Me\\\\t.jinja"),
        }));
        """
    )
    assert out["quoted"] == ["--chat-template-file", "C:\\Users\\Me\\a b.jinja"]
    # Unquoted backslashes are literal too, so a Windows path with no spaces
    # survives without the user having to quote it.
    assert out["bare"] == ["--chat-template-file", "C:\\Users\\Me\\t.jinja"]


def test_single_quotes_keep_every_backslash():
    """A shell keeps a single-quoted value verbatim; only double quotes escape.

    Collapsing `\\\\` inside single quotes rewrites the value the user typed:
    `'{"path":"C:\\\\foo"}'` reaches llama-server as `{"path":"C:\\foo"}`, whose
    `\\f` is a JSON formfeed escape, and a single-quoted GBNF loses the literal
    backslash its grammar needs. Cross-checked against shlex.split(posix=True) --
    the same tokenizer the MCP address parser uses.
    """
    typed = [
        r"""--chat-template-kwargs '{"path":"C:\\foo"}'""",
        r"""--grammar 'root ::= "\\n"'""",
        r"--x 'a\\b'",
        r"--x 'a\b'",
        # Controls: double-quoted and unquoted forms must not move.
        r'--x "a\\b"',
        r'--chat-template-file "C:\\Users\\Me\\a b.jinja"',
        r"--chat-template-file C:\Users\Me\t.jinja",
    ]
    out = _run(
        "const typed = "
        + json.dumps(typed)
        + ";\nconsole.log(JSON.stringify({ rows: typed.map((t) => ({\n"
        "  parsed: parseLlamaExtraArgsInput(t),\n"
        "  reparsed: parseLlamaExtraArgsInput(formatLlamaExtraArgs(parseLlamaExtraArgsInput(t))),\n"
        "})) }));\n"
    )
    rows = out["rows"]
    # Single quotes: verbatim, exactly as a POSIX shell would tokenize them.
    for text, row in zip(typed[:4], rows[:4]):
        assert row["parsed"] == shlex.split(text, posix = True), (text, row)
    # Double quotes: a quote or another backslash still escapes.
    assert rows[4]["parsed"] == ["--x", "a\\b"]
    assert rows[5]["parsed"] == ["--chat-template-file", "C:\\Users\\Me\\a b.jinja"]
    # Unquoted backslashes stay literal (deliberately not POSIX; see
    # test_quoted_windows_path_keeps_its_backslashes).
    assert rows[6]["parsed"] == ["--chat-template-file", "C:\\Users\\Me\\t.jinja"]
    # The formatter only ever emits double quotes, so re-parsing is still a
    # fixed point for every one of these.
    for text, row in zip(typed, rows):
        assert row["reparsed"] == row["parsed"], (text, row)


def test_save_persists_the_same_argv_the_load_sent():
    """normalize is a validator, not a rewriter.

    The load path sends the parsed tokens verbatim while the save path stores
    normalizeLlamaExtraArgs(tokens), so a token it edits makes the same saved
    config launch differently after a restart.
    """
    out = _run(
        """
        const cases = [
          '--chat-template "{{ x }} "',
          '--cpu-moe --no-mmap',
          '--chat-template-kwargs \\'{"a": "b"}\\'',
        ];
        const rows = cases.map((text) => {
          const live = parseLlamaExtraArgsInput(text);
          return { live, persisted: normalizeLlamaExtraArgs(live) };
        });
        console.log(JSON.stringify({ rows }));
        """
    )
    for row in out["rows"]:
        assert row["persisted"] == row["live"], row


def test_normalize_drops_non_strings_and_keeps_every_token():
    """normalize validates the container, it does not rewrite the argv.

    Only explicit quoting can produce a blank or edge-padded token, so every
    string entry is deliberate and must survive; non-strings are dropped.
    """
    out = _run(
        """
        console.log(JSON.stringify({
          cleaned: normalizeLlamaExtraArgs(["--cpu-moe", "  ", "", 7, null, undefined]),
          notArray: normalizeLlamaExtraArgs("--cpu-moe") ?? null,
          empty: normalizeLlamaExtraArgs([]),
        }));
        """
    )
    assert out["cleaned"] == ["--cpu-moe", "  ", ""]
    assert out["notArray"] is None
    assert out["empty"] == []


def test_empty_quoted_argument_survives_the_field():
    """Dropping an empty argument shifts every token after it.

    Verified on llama-server b10107: `--chat-template "" --ctx-size BOGUS` fails
    with `error while handling argument "--ctx-size"`, while the argv without the
    empty token fails with `invalid argument: BOGUS` -- llama-server took
    `--ctx-size` as the template string and never applied it.
    """
    out = _run(
        """
        const typed = parseLlamaExtraArgsInput('--chat-template "" --no-mmap');
        const argv = ["--chat-template", "", "--no-mmap"];
        const rendered = formatLlamaExtraArgs(argv);
        console.log(JSON.stringify({
          typed,
          rendered,
          reparsed: parseLlamaExtraArgsInput(rendered),
          persisted: normalizeLlamaExtraArgs(argv),
          blankField: parseLlamaExtraArgsInput("   "),
          adjacent: parseLlamaExtraArgsInput('a""b'),
        }));
        """
    )
    assert out["typed"] == ["--chat-template", "", "--no-mmap"]
    assert out["rendered"] == '--chat-template "" --no-mmap'
    assert out["reparsed"] == ["--chat-template", "", "--no-mmap"]
    assert out["persisted"] == ["--chat-template", "", "--no-mmap"]
    # Whitespace-only yields no tokens; quotes adjacent to bare text join into
    # one token, as a shell would.
    assert out["blankField"] == []
    assert out["adjacent"] == ["ab"]


def test_clearing_the_field_sends_an_explicit_empty_list():
    """[] clears inherited args; undefined means "inherit" to /load.

    Collapsing a cleared field to undefined keeps launching the flags the user
    just deleted, with nothing in the UI to show it.
    """
    out = _run(
        """
        console.log(JSON.stringify({
          cleared: llamaExtraArgsForLoad(parseLlamaExtraArgsInput("   ")),
          absent: llamaExtraArgsForLoad(null) ?? "OMITTED",
          undef: llamaExtraArgsForLoad(undefined) ?? "OMITTED",
          set: llamaExtraArgsForLoad(["--cpu-moe"]),
        }));
        """
    )
    assert out["cleared"] == []
    assert out["absent"] == "OMITTED"
    assert out["undef"] == "OMITTED"
    assert out["set"] == ["--cpu-moe"]
