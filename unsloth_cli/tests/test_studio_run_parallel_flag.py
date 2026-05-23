# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `unsloth studio run --parallel` CLI flag.

Before this commit, `unsloth studio run` always set
``llama_parallel_slots=4`` (hardcoded). Engine + KV-cache math already
supported any N, but the knob was not user-reachable.

These tests pin:
  1. The Typer option exists with the documented aliases.
  2. The default value matches the previous hardcoded value (4),
     so behaviour is unchanged for existing users.
  3. The range guards reject out-of-band values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_run_command():
    """Import the `run` typer command without triggering server start.

    The CLI module pulls in heavy backend imports at function call
    time; we only need the Typer command object to introspect options.
    """
    from unsloth_cli.commands import studio as _studio

    return _studio


def test_parallel_option_is_registered():
    """The `--parallel` flag (with aliases) must be on the `run` command."""
    studio_mod = _load_run_command()
    import inspect

    run_fn = studio_mod.run
    sig = inspect.signature(run_fn)
    assert "parallel" in sig.parameters, "missing `parallel` parameter on run()"

    param = sig.parameters["parallel"]
    # typer.OptionInfo lives in the default value
    opt = param.default
    flags = set()
    # OptionInfo stores param_decls in .param_decls (typer >=0.4)
    decls = getattr(opt, "param_decls", None) or []
    for d in decls:
        flags.add(d)
    for required in ("--parallel", "--n-parallel", "-np"):
        assert required in flags, f"flag {required!r} missing from --parallel option"


def test_parallel_default_is_four():
    """Default must stay at 4 so plain `unsloth studio run` is unchanged."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    default = getattr(opt, "default", None)
    assert (
        default == 4
    ), f"default changed to {default}; would silently alter existing deployments"


def test_parallel_range_guards_are_set():
    """Range guards: 1 <= N <= 64. Outside this is a hard reject."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    assert getattr(opt, "min", None) == 1, "min must be 1 (0 = no decode possible)"
    assert getattr(opt, "max", None) == 64, "max must be 64 (KV split sanity cap)"


def test_run_kwargs_use_parallel_value(monkeypatch):
    """run_kwargs passed to run_server must reflect the --parallel value
    (no hardcoded 4 remaining after the flag was added)."""
    studio_mod = _load_run_command()
    import textwrap

    src = textwrap.dedent(Path(studio_mod.__file__).read_text())
    # Pre-fix line was: run_kwargs = dict(... llama_parallel_slots = 4)
    assert "llama_parallel_slots = 4" not in src, (
        "found hardcoded `llama_parallel_slots = 4` after the parallel "
        "flag landed -- run_kwargs must pull from the typer option"
    )
    assert (
        "llama_parallel_slots = parallel" in src
    ), "run_kwargs must use the parallel variable from the typer option"
