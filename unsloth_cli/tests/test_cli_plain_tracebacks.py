# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CLI renders logged exceptions as plain tracebacks, not structlog's slow rich ones."""

import ast
from pathlib import Path

import pytest

structlog = pytest.importorskip("structlog")

from unsloth_cli import _inference

_COMMANDS = Path(__file__).resolve().parents[1] / "commands"


@pytest.fixture(autouse = True)
def _reset_structlog(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising = False)
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _renderers():
    return [
        p
        for p in structlog.get_config()["processors"]
        if isinstance(p, structlog.dev.ConsoleRenderer)
    ]


def _formatter(renderer):
    return getattr(renderer, "exception_formatter", None) or renderer._exception_formatter


def _raise_inner():
    raise ValueError("boom from the backend")


def _log_exception(capsys):
    log = structlog.get_logger("cli-test")
    try:
        _raise_inner()
    except ValueError:
        log.exception("request failed", attempt = 3)
    return capsys.readouterr().out


@pytest.mark.parametrize("quiet", [True, False], ids = ["quiet", "verbose"])
def test_logged_exception_is_plain_traceback(quiet, capsys):
    builtin_before = [_formatter(r) for r in _renderers()]
    if quiet:
        _inference.configure_quiet_logging()
    else:
        _inference.use_plain_tracebacks()

    renderers = _renderers()
    assert renderers
    assert all(_formatter(r) is structlog.dev.plain_traceback for r in renderers)

    out = _log_exception(capsys)
    assert "request failed" in out
    assert "attempt=3" in out
    assert "Traceback (most recent call last):" in out
    assert "in _raise_inner" in out
    assert "ValueError: boom from the backend" in out
    # Rich's boxed traceback and locals dump are gone.
    assert "╭" not in out
    assert "locals" not in out.lower()

    # The builtin default renderer is copied, not mutated, so reset_defaults() restores it.
    structlog.reset_defaults()
    assert [_formatter(r) for r in _renderers()] == builtin_before


def test_quiet_logging_keeps_level_filter(capsys):
    _inference.configure_quiet_logging()
    log = structlog.get_logger("cli-test")
    log.info("hidden info")
    log.warning("shown warning")
    out = capsys.readouterr().out
    assert "hidden info" not in out
    assert "shown warning" in out


@pytest.mark.parametrize("command", ["inference.py", "chat.py"])
def test_verbose_branch_uses_plain_tracebacks(command):
    """Both branches of `if not verbose:` pick a logging setup, so --verbose gets plain tracebacks too."""
    tree = ast.parse((_COMMANDS / command).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and getattr(node.test.operand, "id", None) == "verbose"
        ):
            calls = lambda body: {
                n.func.id
                for stmt in body
                for n in ast.walk(stmt)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if "configure_quiet_logging" in calls(node.body):
                assert "use_plain_tracebacks" in calls(node.orelse)
                return
    pytest.fail(f"no `if not verbose: configure_quiet_logging()` in {command}")
