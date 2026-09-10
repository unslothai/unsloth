# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""main must be a fixed point of its own formatting hook.

pre-commit runs `ruff-format-with-kwargs` on the files a PR touches, so a file
that lands unformatted is never looked at again: the next PR to edit it inherits
a red `pre-commit.ci - pr` for a diff it did not write, and the author goes
looking for a defect that is not in their change. Two files reached main that
way and sat there, one of them for months.

The check is the real thing rather than `ruff format --check`. The hook is
`enforce_kwargs_spacing --pre`, then `ruff format`, then `enforce_kwargs_spacing`
again, and ruff only covers the middle pass, so a file can pass a ruff check
cleanly and still be rewritten by the hook. It is run over copies, so a failing
run reports the drift instead of quietly fixing it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import enforce_kwargs_spacing  # noqa: E402
from run_ruff_format import (  # noqa: E402
    CONFIG,
    installed_ruff_version,
    pinned_ruff_version,
    version_mismatch,
)

_HOOK_ID = "ruff-format-with-kwargs"
# The spacing pass refuses to rewrite itself ("skip modifying this script to
# avoid self-edit loops"), so it is not a file the hook keeps at a fixed point
# and checking it would fail main over a rewrite that never happens. Taken from
# the module rather than spelled out, so moving or renaming it does not turn
# this into a stale exclusion of nothing.
_SELF_SKIPPED = Path(enforce_kwargs_spacing.__file__).resolve()
# The paths the hook is pointed at. `types: [python]` is what pre-commit filters
# on, and the repo tracks no .pyi, so this is the same set.
_TRACKED_GLOBS = ("*.py", "*.pyi")


def hook_exclude_pattern(config_text: str, hook_id: str) -> str | None:
    """The `exclude:` regex the named hook is configured with, or None.

    Read out of .pre-commit-config.yaml rather than copied here. A second copy of
    the exclusion list is a second thing to forget, and forgetting it in this
    direction is the expensive one: this test would format a file the hook never
    touches and fail main over it.

    Scanned rather than parsed with PyYAML, matching how the version pin is read
    next door: the block is found by its `- id:` and abandoned at the next `- id:`
    or `- repo:`, which keeps the ruff hook's own `exclude: '\\.ipynb$'` out.
    """
    lines = config_text.splitlines()
    inside = False
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(rf"-\s*id:\s*{re.escape(hook_id)}", stripped):
            inside = True
            continue
        if inside:
            if stripped.startswith("- id:") or stripped.startswith("- repo:"):
                break
            # The quoted form is tried first and keeps its contents verbatim: a
            # regex may contain a `#`, and treating that as a comment would
            # silently truncate the exclusion to a prefix that matches nothing.
            quoted = re.fullmatch(r"exclude:\s*(['\"])(.*)\1\s*(?:#.*)?", stripped)
            if quoted:
                return quoted.group(2)
            bare = re.fullmatch(r"exclude:\s*(\S+)\s*(?:#.*)?", stripped)
            if bare:
                return bare.group(1)
    return None


def eligible_files(root: Path) -> list[str]:
    """Every tracked Python file the hook would be handed, repo-relative."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", *_TRACKED_GLOBS],
        capture_output = True,
        text = True,
        check = True,
    )
    tracked = [name for name in out.stdout.split("\0") if name]
    pattern = hook_exclude_pattern(CONFIG.read_text(encoding = "utf-8"), _HOOK_ID)
    assert (
        pattern
    ), f"{CONFIG.name} no longer gives {_HOOK_ID} an exclude; the filter below is blind"
    excluded = re.compile(pattern)
    return [
        name
        for name in tracked
        if not excluded.search(name) and (root / name).resolve() != _SELF_SKIPPED
    ]


def _pinned_ruff_reason() -> str | None:
    """Why this cannot be checked here, or None when it can.

    ruff's formatting is not stable across releases, so another ruff answers a
    different question, and the formatter refuses to run under one anyway.
    """
    pinned = pinned_ruff_version(CONFIG.read_text(encoding = "utf-8")) if CONFIG.exists() else None
    installed = installed_ruff_version()
    if installed is None:
        return "ruff is not installed here, and the formatter cannot run without it"
    if version_mismatch(pinned, installed):
        return f"the repo is formatted with ruff {pinned}, this environment has {installed}"
    return None


class TestTheExcludeComesFromTheConfig:
    """The filter has to track the hook, not a copy of it made once."""

    def test_the_real_hook_still_names_an_exclude(self):
        pattern = hook_exclude_pattern(CONFIG.read_text(encoding = "utf-8"), _HOOK_ID)
        assert pattern, f"no exclude found for {_HOOK_ID}"
        re.compile(pattern)

    def test_it_reads_the_named_hook_and_not_a_neighbour(self):
        # The ruff hook above ours carries `exclude: '\.ipynb$'`. Picking up the
        # first exclude in the file would format the vendored tree and fail main.
        text = (
            "repos:\n"
            "  - repo: https://example.invalid/ruff\n"
            "    hooks:\n"
            "      - id: ruff\n"
            "        exclude: '\\.ipynb$'\n"
            "  - repo: local\n"
            "    hooks:\n"
            "      - id: ruff-format-with-kwargs\n"
            "        exclude: '^vendor/'\n"
            "      - id: something-else\n"
            "        exclude: '^other/'\n"
        )
        assert hook_exclude_pattern(text, "ruff-format-with-kwargs") == "^vendor/"
        assert hook_exclude_pattern(text, "ruff") == "\\.ipynb$"
        assert hook_exclude_pattern(text, "something-else") == "^other/"

    def test_a_hook_without_an_exclude_answers_none(self):
        text = "      - id: ruff-format-with-kwargs\n        entry: python x.py\n      - id: next\n"
        assert hook_exclude_pattern(text, "ruff-format-with-kwargs") is None

    def test_the_vendored_tree_and_the_generated_files_are_out(self):
        # Named because they are the ones the hook skips deliberately: reformatting
        # the vendored copy breaks its digest test.
        names = eligible_files(_ROOT)
        assert names
        assert not [n for n in names if n.startswith("studio/backend/vendor/")]
        assert not [n for n in names if n.endswith("chat_templates.py")]

    def test_the_spacing_pass_is_not_asked_to_rewrite_itself(self):
        # It declines by path identity, so a copy of it under another name would
        # be rewritten and reported as drift the hook will never produce.
        names = eligible_files(_ROOT)
        assert _SELF_SKIPPED.is_file()
        assert not [n for n in names if (_ROOT / n).resolve() == _SELF_SKIPPED]
        # And the rest of scripts/ is still in scope, including the hook entry
        # point, which the spacing pass does rewrite.
        assert "scripts/run_ruff_format.py" in names


@pytest.mark.skipif(_pinned_ruff_reason() is not None, reason = _pinned_ruff_reason() or "")
def test_every_tracked_python_file_is_already_formatted(tmp_path):
    """Run the hook over copies of the whole tracked set and expect no rewrite."""
    names = eligible_files(_ROOT)
    assert len(names) > 1000, f"only {len(names)} files matched; the file list has gone vacuous"

    # ruff reads line-length and extend-exclude from the root pyproject.toml, and
    # finds it by walking up from each file. Without this copy the run happens at
    # ruff's default 88 columns and every long line looks like drift.
    shutil.copy2(_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")

    originals: dict[str, bytes] = {}
    copies: list[str] = []
    for name in names:
        source = _ROOT / name
        target = tmp_path / name
        target.parent.mkdir(parents = True, exist_ok = True)
        originals[name] = source.read_bytes()
        target.write_bytes(originals[name])
        copies.append(str(target))

    run = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_ruff_format.py"), *copies],
        capture_output = True,
        text = True,
    )
    assert run.returncode == 0, f"the formatter itself failed:\n{run.stdout}\n{run.stderr}"

    drifted = [name for name in names if (tmp_path / name).read_bytes() != originals[name]]
    assert not drifted, (
        "these tracked files are not a fixed point of the ruff-format-with-kwargs hook, "
        "so pre-commit.ci will fail the next PR that edits them:\n"
        + "\n".join(f"  {name}" for name in drifted)
        + "\n  Fix: python scripts/run_ruff_format.py "
        + " ".join(drifted)
    )
