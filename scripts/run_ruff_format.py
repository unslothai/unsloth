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
USAGE = "usage: run_ruff_format.py FILE [FILE ...]  (formats in place; no options)"

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


def ruff_unavailable_reason(python: str = sys.executable) -> str | None:
    """Why `python -m ruff` cannot run here, or None when it can.

    Separate from the version question because the answers differ. A ruff that
    runs but reports a version this cannot parse is survivable; a ruff that does
    not run at all is not, and the pre-pass below has already rewritten every
    file it was given by the time `ruff format` says so.
    """
    try:
        out = subprocess.run(
            [python, "-m", "ruff", "--version"], capture_output = True, text = True, timeout = 60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return (out.stderr or out.stdout).strip() or f"`ruff --version` exited {out.returncode}"
    return None


def version_mismatch(pinned: str | None, installed: str | None) -> bool:
    """Whether running this ruff would produce formatting the hook then undoes.

    An unreadable pin or an unreadable version string is not a mismatch:
    refusing on a question we could not ask would break the hook wherever the
    config moves. A ruff that cannot run at all is caught before this, by
    ruff_unavailable_reason.
    """
    return bool(pinned and installed and pinned != installed)


def parse_files(argv: list[str]) -> tuple[list[str], str | None]:
    """The paths to format, or an empty list plus a message saying why not.

    Every argument is a path to rewrite. Silently dropping the rest was worse
    than it sounds: `--check FILE` dropped the flag, kept the file, and wrote
    to it, and a typo'd path formatted nothing while exiting 0, which quietly
    passes any "the formatter is a fixed point" check.
    """
    if not argv:
        return [], f"no files given.\n{USAGE}"

    options = [arg for arg in argv if arg.startswith("-")]
    if options:
        message = f"unsupported option{'s' if len(options) > 1 else ''}: {' '.join(options)}"
        if any(opt in ("--check", "--diff") for opt in options):
            message += (
                "\n  There is no check mode: this script always rewrites the files"
                " it is given, and `ruff format --check` is not an equivalent."
                "\n  It checks the middle one of three passes, so a clean ruff says"
                " nothing about the kwarg-spacing passes either side of it."
                "\n  To preview a run, copy the file aside, run this script on the"
                " copy, and diff the two."
            )
        return [], f"{message}\n{USAGE}"

    missing = [arg for arg in argv if not Path(arg).exists()]
    if missing:
        return [], f"no such file{'s' if len(missing) > 1 else ''}: {' '.join(missing)}\n{USAGE}"

    return list(argv), None


def main(argv: list[str]) -> int:
    files, error = parse_files(argv)
    if error is not None:
        print(f"run_ruff_format: {error}", file = sys.stderr)
        return 2

    pinned = pinned_ruff_version(CONFIG.read_text(encoding = "utf-8")) if CONFIG.exists() else None

    # Both checks are made before anything is rewritten, because the pre-pass is
    # itself a rewrite. Without this first one, a missing or broken ruff let the
    # pre-pass strip every magic comma it was given and only then die on `ruff
    # format`, leaving files in a shape the hook rejects -- the opposite of what
    # a full run produces, and blamed on the next person to touch them. The
    # override below is deliberately not honoured here: no ruff formats nothing.
    unavailable = ruff_unavailable_reason()
    if unavailable is not None:
        print(
            f"run_ruff_format: cannot run `python -m ruff` ({unavailable}).\n"
            f"  Refusing before rewriting anything: the passes either side of ruff would "
            f"leave the files half-formatted.\n"
            f"  Fix: pip install ruff=={pinned or '<the pin in .pre-commit-config.yaml>'}",
            file = sys.stderr,
        )
        return 1

    # ruff's own formatting is not stable across releases -- 0.9 changed which
    # half of an `assert cond, "msg"` gets wrapped -- so running this with a newer
    # ruff silently produces a style the pinned hook reformats back, and the
    # commit fails pre-commit on files that are otherwise correct. It reached main
    # twice before this check existed.
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
