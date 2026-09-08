#!/usr/bin/env python3
"""Run a pre-pass (normalize def-signature magic commas + collapse short
multi-line asserts), then `ruff format`, then the kwarg-spacing / import /
string-merge post-pass."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / ".pre-commit-config.yaml"
# Set to run against whatever ruff is installed. For a one-off experiment; a commit
# made under it will be reformatted by the hook and fail pre-commit.
ANY_VERSION_ENV = "UNSLOTH_RUFF_FORMAT_ANY_VERSION"

# `- ruff==0.6.9` under the hook's additional_dependencies. Read out of the config
# rather than copied here, because a second copy of the pin is a second thing to
# forget; a regex rather than yaml.safe_load because this hook installs ruff and
# nothing else, and adding PyYAML to run a version check would be the tail wagging
# the dog.
_PIN_RE = re.compile(r"^\s*-\s*ruff\s*==\s*([0-9][^\s#]*)\s*(?:#.*)?$", re.MULTILINE)
_VERSION_RE = re.compile(r"^ruff\s+([0-9][^\s]*)")


def pinned_ruff_version(config_text: str) -> str | None:
    """The ruff this repo's formatting was produced with, or None if unpinned."""
    found = {match.group(1) for match in _PIN_RE.finditer(config_text)}
    # Two different pins is not a question this script can answer, and guessing
    # would enforce the wrong one.
    return found.pop() if len(found) == 1 else None


def installed_ruff_version(python: str = sys.executable) -> str | None:
    """The ruff the format below would actually run, or None when it cannot say."""
    try:
        out = subprocess.run(
            [python, "-m", "ruff", "--version"], capture_output = True, text = True, timeout = 60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    match = _VERSION_RE.match(out.stdout.strip())
    return match.group(1) if match else None


def version_mismatch(pinned: str | None, installed: str | None) -> bool:
    """Whether running this ruff would produce formatting the hook then undoes.

    An unreadable pin or an unreadable ruff is not a mismatch: the format below
    fails loudly enough on its own, and refusing on a question we could not ask
    would break the hook wherever the config moves.
    """
    return bool(pinned and installed and pinned != installed)


def main(argv: list[str]) -> int:
    files = [arg for arg in argv if Path(arg).exists()]
    if not files:
        return 0

    # Checked before anything is rewritten. ruff's own formatting is not stable
    # across releases -- 0.9 changed which half of an `assert cond, "msg"` gets
    # wrapped -- so running this with a newer ruff silently produces a style the
    # pinned hook reformats back, and the commit fails pre-commit on files that
    # are otherwise correct. It reached main twice before this check existed.
    pinned = pinned_ruff_version(CONFIG.read_text(encoding = "utf-8")) if CONFIG.exists() else None
    installed = installed_ruff_version()
    if version_mismatch(pinned, installed) and not os.environ.get(ANY_VERSION_ENV):
        print(
            f"run_ruff_format: this would run ruff {installed}, but the repo is formatted "
            f"with ruff {pinned} ({CONFIG.name}).\n"
            f"  Their output differs, so the hook would undo this run and pre-commit would "
            f"fail on files you did not break.\n"
            f"  Fix: pip install ruff=={pinned}, or run the hook itself "
            f"(pre-commit run ruff-format-with-kwargs --files ...).\n"
            f"  Override with {ANY_VERSION_ENV}=1 if you really mean it.",
            file = sys.stderr,
        )
        return 1

    spacing_script = HERE / "enforce_kwargs_spacing.py"

    # Pre-ruff: normalize def-signature magic commas and strip the magic comma
    # from short multi-line asserts so ruff wraps/joins accordingly.
    pre_cmd = [sys.executable, str(spacing_script), "--pre", *files]
    pre_proc = subprocess.run(pre_cmd)
    if pre_proc.returncode != 0:
        return pre_proc.returncode

    ruff_cmd = [sys.executable, "-m", "ruff", "format", *files]
    ruff_proc = subprocess.run(ruff_cmd)
    if ruff_proc.returncode != 0:
        return ruff_proc.returncode

    spacing_cmd = [sys.executable, str(spacing_script), *files]
    spacing_proc = subprocess.run(spacing_cmd)
    return spacing_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
