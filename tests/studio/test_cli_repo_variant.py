"""Tests for the ``repo:variant`` shorthand parser used by ``unsloth studio run``.

Loads ``unsloth_cli/commands/studio.py`` directly via ``importlib`` with a
minimal ``typer`` stub so the test doesn't drag in the rest of
``unsloth_cli`` (which transitively imports the unsloth training stack).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_split_repo_variant():
    """Load ``_split_repo_variant`` from studio.py with typer stubbed.

    studio.py decorates Typer commands at import time, so a stub that
    accepts (and discards) those calls is enough to let module
    execution complete and expose the helper we want to test.
    """
    if "typer" not in sys.modules:
        typer_stub = types.ModuleType("typer")

        class _Typer:
            def __init__(self, **kwargs):
                pass

            def callback(self, *args, **kwargs):
                return lambda fn: fn

            def command(self, *args, **kwargs):
                return lambda fn: fn

        typer_stub.Typer = _Typer
        typer_stub.Option = lambda *args, **kwargs: (args[0] if args else None)
        typer_stub.Context = type("Context", (), {})
        typer_stub.Exit = type("Exit", (Exception,), {})
        typer_stub.echo = lambda *args, **kwargs: None
        sys.modules["typer"] = typer_stub

    studio_py = (
        Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_studio_for_repo_variant_test", studio_py
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._split_repo_variant


_split = _load_split_repo_variant()


# ── HF-style repo:variant inputs -------------------------------------


@pytest.mark.parametrize(
    "model_arg, expected",
    [
        (
            "unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL",
            ("unsloth/gpt-oss-20b-GGUF", "UD-Q4_K_XL"),
        ),
        ("unsloth/gpt-oss-120b-GGUF:Q4_K_XL", ("unsloth/gpt-oss-120b-GGUF", "Q4_K_XL")),
        ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", ("unsloth/Qwen3-0.6B-GGUF", "Q4_K_M")),
        # Variants commonly contain dashes, dots, and underscores.
        ("org/repo:UD-Q5_K_M", ("org/repo", "UD-Q5_K_M")),
        ("org/repo:F16", ("org/repo", "F16")),
    ],
)
def test_repo_variant_split(model_arg, expected):
    assert _split(model_arg) == expected


# ── No variant suffix ------------------------------------------------


@pytest.mark.parametrize(
    "model_arg",
    [
        "unsloth/gpt-oss-20b-GGUF",
        "unsloth/Qwen3-0.6B-GGUF",
        "shorthand-no-org-no-colon",
    ],
)
def test_no_colon_returns_none_variant(model_arg):
    repo, variant = _split(model_arg)
    assert repo == model_arg
    assert variant is None


# ── Local paths must NOT be split ------------------------------------


@pytest.mark.parametrize(
    "local_path",
    [
        "/abs/path/to/model.gguf",
        "/abs/path:with-colon-in-name",
        "./relative/model",
        "../parent/model",
        "~/home/model",
        ".",
        "C:\\Users\\me\\model.gguf",
        "C:/Users/me/model.gguf",
        "D:/data/model:Q4",  # Windows drive + colon-suffixed filename: drive wins
    ],
)
def test_local_path_passthrough(local_path):
    repo, variant = _split(local_path)
    assert repo == local_path
    assert variant is None


# ── Edge cases -------------------------------------------------------


def test_empty_string():
    assert _split("") == ("", None)


def test_trailing_colon_no_variant():
    # "org/repo:" -- no quant label after the colon. Pass through
    # unchanged so the backend's existing validation surfaces a
    # clearer error than "variant ''".
    repo, variant = _split("org/repo:")
    assert repo == "org/repo:"
    assert variant is None


def test_slash_in_variant_disqualifies_split():
    # "foo:bar/baz" -- the suffix has a slash, so this isn't a quant
    # label; treat the whole thing as opaque.
    repo, variant = _split("foo:bar/baz")
    assert repo == "foo:bar/baz"
    assert variant is None


def test_whitespace_stripped():
    assert _split("  org/repo:Q4  ") == ("org/repo", "Q4")
