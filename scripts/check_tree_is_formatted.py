#!/usr/bin/env python3
"""Is the tree already at `ruff-format-with-kwargs`'s fixed point?

pre-commit.ci is the only thing that checks formatting, and it is also what breaks
when main drifts. Its autofix is computed against the PR merged with main, so three
unformatted files on main made the patch unappliable to every PR head: four open PRs
went red at once, each carrying a one-or-two-line delta it could not self-heal, all
reporting

    a conflict occurred when applying fixes to the pr. try pulling the upstream branch

That is self-sustaining, because a PR that cannot be autofixed gets merged red and
adds more drift. It took a hand-run of the hook to see any of it (#11019).

So run the hook's own formatter over the hook's own file set and report whether
anything moved. Both the ruff pin and the file exclusion are read out of
.pre-commit-config.yaml rather than repeated here: a second copy is a second thing to
forget, and a checker that selects a different set than the hook is worse than none.

Usage: python scripts/check_tree_is_formatted.py [--repo DIR]
       python scripts/check_tree_is_formatted.py --print-pin
Exit 0 when the tree is at the fixed point, 1 when it is not, 2 when it cannot tell.
`--print-pin` writes the pinned ruff version, so a caller can install it without
holding a second copy of the number.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMATTER = HERE / "run_ruff_format.py"
HOOK_ID = "ruff-format-with-kwargs"

# The hook's block in .pre-commit-config.yaml, from its id to the next id or repo.
_HOOK_RE = re.compile(
    rf"^\s*-\s*id:\s*{re.escape(HOOK_ID)}\s*$(?P<body>.*?)(?=^\s*-\s*(?:id|repo):|\Z)",
    re.MULTILINE | re.DOTALL,
)
_EXCLUDE_RE = re.compile(r"^\s*exclude:\s*'(?P<pattern>[^']*)'\s*$", re.MULTILINE)
_TYPES_RE = re.compile(r"^\s*types:\s*\[(?P<types>[^\]]*)\]\s*$", re.MULTILINE)


class CannotTell(Exception):
    """The config does not say what this would have to assume."""


def hook_body(config_text: str) -> str:
    match = _HOOK_RE.search(config_text)
    if match is None:
        raise CannotTell(f"no hook with id {HOOK_ID} in .pre-commit-config.yaml")
    return match.group("body")


def hook_exclude(config_text: str) -> re.Pattern[str]:
    match = _EXCLUDE_RE.search(hook_body(config_text))
    if match is None:
        raise CannotTell(f"{HOOK_ID} has no `exclude:`, so its file set is unknown here")
    return re.compile(match.group("pattern"))


def hook_types(config_text: str) -> list[str]:
    match = _TYPES_RE.search(hook_body(config_text))
    if match is None:
        raise CannotTell(f"{HOOK_ID} has no `types:`, so its file set is unknown here")
    return [t.strip() for t in match.group("types").split(",") if t.strip()]


def selected_files(repo: Path, config_text: str) -> list[str]:
    """The tracked files the hook would run on."""
    types = hook_types(config_text)
    if types != ["python"]:
        # Anything else means the hook grew a file kind this cannot enumerate, and
        # silently checking the old set would be a green that means nothing.
        raise CannotTell(f"{HOOK_ID} now runs on {types}, which this only knows for python")
    exclude = hook_exclude(config_text)
    listed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "*.py"],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.splitlines()
    return [path for path in listed if not exclude.search(path)]


def dirty_paths(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only"],
        capture_output = True,
        text = True,
        check = True,
    ).stdout
    return out.split()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--repo", default = str(HERE.parent), help = "repo root")
    parser.add_argument(
        "--print-pin", action = "store_true", help = "print the pinned ruff version and exit"
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    config_text = (repo / ".pre-commit-config.yaml").read_text()

    if args.print_pin:
        sys.path.insert(0, str(HERE))
        from run_ruff_format import pinned_ruff_version  # noqa: PLC0415

        pin = pinned_ruff_version(config_text)
        if pin is None:
            print("error: .pre-commit-config.yaml does not pin exactly one ruff", file = sys.stderr)
            return 2
        print(pin)
        return 0

    if dirty_paths(repo):
        # Formatting in place and then reading `git diff` cannot separate what this
        # changed from what was already there, so refuse rather than blame the tree.
        print(
            "error: the working tree has unstaged changes; this formats in place", file = sys.stderr
        )
        return 2

    try:
        files = selected_files(repo, config_text)
    except (CannotTell, OSError) as e:
        print(f"error: {e}", file = sys.stderr)
        return 2

    run = subprocess.run([sys.executable, str(FORMATTER), *files], cwd = repo)
    if run.returncode != 0:
        # run_ruff_format.py refuses to run under an unpinned ruff, and says so.
        return 2

    changed = dirty_paths(repo)
    if not changed:
        print(f"{len(files)} files are at the formatter's fixed point")
        return 0
    print(
        f"::error::{len(changed)} file(s) are not formatted the way "
        f"{HOOK_ID} formats them. pre-commit.ci cannot autofix any OTHER pull "
        f"request while this is true: its patch is computed against the merge with "
        f"this branch and will not apply to a head that lacks these. Run "
        f"`pre-commit run {HOOK_ID} --all-files` and commit the result."
    )
    subprocess.run(["git", "-C", str(repo), "diff", "--stat"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
