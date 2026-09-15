#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decide whether a PR touched anything the security audit reads.

This is the `paths:` filter security-audit.yml used to carry, moved into a job so the
workflow can still start and still report a check. A filtered workflow reports nothing when
it is skipped, and a check that never reports can never be required.

The patterns arrive in $PATTERNS, one per line, in the form the filter used: an exact path,
or a directory followed by `/**`. Writes `relevant=true|false` to $GITHUB_OUTPUT.

Self-test: python3 scripts/ci_paths_changed.py --self-test
"""

from __future__ import annotations

import os
import sys


def matches(path: str, pattern: str) -> bool:
    """GitHub's semantics for the two pattern shapes this filter uses."""
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return path == pattern


def relevant(changed, patterns):
    """Every (file, pattern) pair that makes this PR one the audit must run on."""
    hits = []
    for f in changed:
        for p in patterns:
            if matches(f, p):
                hits.append((f, p))
                break
    return hits


def _self_test() -> int:
    pats = [
        "pyproject.toml",
        "studio/backend/requirements/**",
        "studio/frontend/package-lock.json",
        "tests/security/**",
        "unsloth/models/loader_utils.py",
        ".github/workflows/security-audit.yml",
    ]
    cases = [
        # (changed file, should the audit run)
        ("pyproject.toml", True),
        ("studio/backend/requirements/base.txt", True),
        ("studio/backend/requirements/nested/deep.txt", True),
        ("studio/frontend/package-lock.json", True),
        ("tests/security/test_scan_packages.py", True),
        ("unsloth/models/loader_utils.py", True),
        (".github/workflows/security-audit.yml", True),
        # must NOT match: a false skip is the dangerous direction
        ("unsloth/models/llama.py", False),
        ("README.md", False),
        (".github/workflows/lint-ci.yml", False),
        ("vendor/pyproject.toml", False),
        ("tests/security_helpers/foo.py", False),
        ("docs/studio/frontend/package-lock.json", False),
        ("unsloth/models/loader_utils.py.orig", False),
        ("studio/backend/requirements", False),
    ]
    bad = 0
    for path, want in cases:
        got = bool(relevant([path], pats))
        flag = "ok  " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"  {flag} {path:48s} run={got} want={want}")
    # a mixed changeset is relevant if any one file is
    if not relevant(["README.md", "pyproject.toml"], pats):
        print("  FAIL mixed changeset with a dependency file must run"); bad += 1
    else:
        print("  ok   mixed changeset with a dependency file runs")
    if relevant(["README.md", "unsloth/models/llama.py"], pats):
        print("  FAIL wholly unrelated changeset must skip"); bad += 1
    else:
        print("  ok   wholly unrelated changeset skips")
    print("self-test:", "FAILED" if bad else "passed")
    return 1 if bad else 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    changed_file = sys.argv[1]
    patterns = [p.strip() for p in os.environ.get("PATTERNS", "").splitlines() if p.strip()]
    if not patterns:
        print("::error::PATTERNS is empty; refusing to declare this PR irrelevant")
        return 1
    with open(changed_file) as fh:
        changed = [l.strip() for l in fh if l.strip()]
    hits = relevant(changed, patterns)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write("relevant=%s\n" % ("true" if hits else "false"))
    if hits:
        print("=> a scanned path changed; running the audit")
        for f, p in hits:
            print(f"     {f}  (matched {p})")
    else:
        print("=> nothing the audit reads changed; skipping the scans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
