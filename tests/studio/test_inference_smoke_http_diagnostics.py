# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
The inference smoke probes must say what the server said when a request 4xx's.

#8883 broke the Mac GGUF job's `/v1/chat/completions` call and stayed broken for
three main runs. The only thing CI printed was:

    urllib.error.HTTPError: HTTP Error 400: Bad Request

The server's own explanation went out with the unread response body, so the cause
had to be reconstructed by hand from the workflow source. These probes are the
only place a real llama-server answers a real request, so their diagnostics are
the whole value of a red run; a status line with no body is a red run that costs
an investigation instead of paying for one.

The tests parse the Python actually embedded in the workflows rather than
matching text, so a rewrite that keeps the behaviour keeps passing and a rewrite
that drops it fails.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

SMOKE_WORKFLOWS = (
    "studio-inference-smoke.yml",
    "studio-mac-ui-smoke.yml",
    "studio-windows-inference-smoke.yml",
)

_HEREDOC_START = re.compile(r"^(\s*)python3? - <<'PY'\s*$")


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """Every `python - <<'PY' ... PY` block, as (1-based start line, source)."""
    blocks: list[tuple[int, str]] = []
    lines = path.read_text(encoding = "utf-8").splitlines()
    i = 0
    while i < len(lines):
        match = _HEREDOC_START.match(lines[i])
        if match is None:
            i += 1
            continue
        indent = match.group(1)
        start = i + 1
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() != "PY":
            body.append(lines[i])
            i += 1
        assert i < len(lines), f"{path.name}:{start} heredoc never closed"
        blocks.append((start + 1, textwrap.dedent("\n".join(body)) + "\n"))
        i += 1
    # The indent is stripped by textwrap, so a block whose lines are indented inconsistently would fail to parse below
    del indent
    return blocks


def _handler_reraises_only(handler: ast.ExceptHandler) -> bool:
    """A handler whose entire body is `raise`, i.e. one that adds nothing."""
    return all(isinstance(node, ast.Raise) and node.exc is None for node in handler.body)


def _catches_http_error(handler: ast.ExceptHandler) -> bool:
    types = handler.type
    if types is None:
        return False
    candidates = types.elts if isinstance(types, ast.Tuple) else [types]
    return any(isinstance(node, ast.Attribute) and node.attr == "HTTPError" for node in candidates)


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _sends_a_request(func: ast.FunctionDef) -> bool:
    """A helper that actually opens the URL, as opposed to one of its callers."""
    for call in _calls(func):
        target = call.func
        if isinstance(target, ast.Attribute) and target.attr == "urlopen":
            return True
    return False


def _request_helpers(source: str) -> list[ast.FunctionDef]:
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _sends_a_request(node)
    ]


@pytest.mark.parametrize("name", SMOKE_WORKFLOWS)
def test_every_embedded_probe_is_valid_python(name: str) -> None:
    """The heredocs are shell text to YAML, so nothing else checks they parse."""
    path = WORKFLOWS / name
    blocks = _python_blocks(path)
    assert blocks, f"{name}: no embedded python probe found"
    for line, source in blocks:
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{name}:{line} embedded python does not parse: {exc}")


@pytest.mark.parametrize("name", SMOKE_WORKFLOWS)
def test_every_request_helper_has_an_http_error_handler(name: str) -> None:
    """
    Without a dedicated handler an HTTPError falls into the URLError branch it
    subclasses and gets retried as a transport stall, which spends the job's
    whole timeout budget re-asking a question the server already refused.
    """
    path = WORKFLOWS / name
    helpers = [helper for _, source in _python_blocks(path) for helper in _request_helpers(source)]
    assert helpers, f"{name}: no request helper found"
    for helper in helpers:
        handlers = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.ExceptHandler) and _catches_http_error(node)
        ]
        assert handlers, (
            f"{name}: {helper.name}() opens a URL but does not handle "
            f"urllib.error.HTTPError, so a 4xx is retried as a transport stall"
        )


@pytest.mark.parametrize("name", SMOKE_WORKFLOWS)
def test_an_http_error_reports_the_response_body(name: str) -> None:
    """
    The regression this guards: a handler that is a bare `raise`. CI then prints
    the status line and nothing else, and the server's explanation is lost with
    the unread body.
    """
    path = WORKFLOWS / name
    for _, source in _python_blocks(path):
        for helper in _request_helpers(source):
            for handler in ast.walk(helper):
                if not isinstance(handler, ast.ExceptHandler):
                    continue
                if not _catches_http_error(handler):
                    continue

                assert not _handler_reraises_only(handler), (
                    f"{name}: {helper.name}() re-raises HTTPError without "
                    f"reporting the response body, so a failure prints only "
                    f"'HTTP Error 400: Bad Request'"
                )
                assert handler.name, (
                    f"{name}: {helper.name}() does not bind the HTTPError, so "
                    f"it cannot report the body"
                )

                bound = handler.name
                reads = any(
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "read"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == bound
                    for call in _calls(handler)
                )
                assert reads, (
                    f"{name}: {helper.name}() does not call {bound}.read(), so "
                    f"the server's explanation is discarded"
                )

                prints = [
                    call
                    for call in _calls(handler)
                    if isinstance(call.func, ast.Name) and call.func.id == "print"
                ]
                assert prints, (
                    f"{name}: {helper.name}() reads the body but never prints "
                    f"it, so the diagnosis never reaches the CI log"
                )

                # Reading the body is only useful if the printed text carries it, and the status code alongside it
                printed = "\n".join(ast.dump(call) for call in prints)
                assert (
                    "code" in printed
                ), f"{name}: {helper.name}() prints on HTTPError without the status code"

                # that must not replace the real HTTPError with a confusing one.
                # A read can raise (a truncated or already-consumed body), and that must not replace the real HTTPError
                guarded = any(isinstance(node, ast.Try) for node in ast.walk(handler))
                assert guarded, (
                    f"{name}: {helper.name}() reads the HTTPError body "
                    f"unguarded, so a failed read masks the real status"
                )

                # And the original error must still propagate: reporting is not
                assert any(isinstance(node, ast.Raise) for node in handler.body), (
                    f"{name}: {helper.name}() reports the HTTPError but does "
                    f"not re-raise it, so a 4xx would pass as success"
                )
