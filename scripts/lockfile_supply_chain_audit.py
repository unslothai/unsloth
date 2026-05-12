#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Lockfile supply-chain audit for the Studio frontend and Tauri shell.

Runs BEFORE `npm ci` / `cargo fetch` in CI. Refuses to proceed when a
lockfile contains patterns that indicate the kind of supply-chain
injection seen in the npm Shai-Hulud waves and the cargo
crates.io brand-squat attempts.

What it checks
==============

studio/frontend/package-lock.json (lockfileVersion 2 or 3):

  1. `resolved` URL origin. Every entry must resolve through
     `https://registry.npmjs.org/`. Direct GitHub-hosted dependencies
     (`git+ssh://`, `git+https://`, `github:owner/repo#sha`,
     `file:`, `http://`) are refused -- npm's TanStack incident used
     exactly this vector to land an unaudited GitHub commit hash as
     an optional dependency.

  2. `integrity` field presence. Every non-workspace entry must carry
     an `integrity` SHA. A missing integrity means the registry can
     swap the tarball after lockfile generation and CI will not
     notice.

  3. Known IOC strings. A hardcoded set of indicator-of-compromise
     substrings is grepped across the entire lockfile body (file
     names, dependency keys, URLs). The list is updated as new
     campaigns surface. Catching one means the local install was
     about to pull a publicly-known malicious release.

studio/src-tauri/Cargo.lock:

  4. `source` field origin. Every entry with a `source` must point at
     `registry+https://github.com/rust-lang/crates.io-index`. Direct
     git sources (`git+https://...`) and `path+...` for cross-crate
     paths warrant manual review and are flagged.

  5. Known cargo IOC strings. Same idea as (3), separate list.

Exit codes
==========

  0  no findings, or an opt-out env var (UNSLOTH_LOCKFILE_AUDIT_SKIP=1)
     is set
  1  one or more findings; stderr lists them with file path and line
     number where derivable
  2  internal error (missing dependency, malformed JSON, etc.)

Operational stance
==================

This scanner only PARSES the lockfiles -- it never executes anything
in them, never resolves anything against the network. Safe to run
ahead of every `npm ci`. The IOC list is short by design; this
complements (not replaces) `npm audit`, OSV-Scanner, and the
advisory-DB pipeline in `.github/workflows/security-audit.yml`. The
shape of the catch is "we refuse to proceed because the lockfile
itself is shaped wrong", which fires before any third-party install
script gets a chance to run on the runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────
# Known IOC strings (case-sensitive substring match).
# ─────────────────────────────────────────────────────────────────────
#
# Keep these short and FACTUAL. Each entry is tied to a public advisory
# and is the literal string an attacker would have to embed for the
# attack to work. Adding speculative or generic patterns here would
# generate false positives on dependency upgrades.
NPM_IOC_STRINGS: tuple[str, ...] = (
    # Shai-Hulud TanStack wave -- May 11, 2026 (GHSA-g7cv-rxg3-hmpx).
    "router_init.js",
    "tanstack_runner.js",
    "router_runtime.js",
    "@tanstack/setup",
    "github:tanstack/router#79ac49eedf774dd4b0cfa308722bc463cfe5885c",
    # Exfiltration endpoints observed across both Shai-Hulud waves.
    "filev2.getsession.org",
    "getsession.org/file/",
    # Campaign markers; the worm tarballs print this to stdout on run.
    "A Mini Shai-Hulud has Appeared",
)

CARGO_IOC_STRINGS: tuple[str, ...] = (
    # Reserved for future cargo-side incidents. Empty by default --
    # `source` origin check below catches the structural pattern.
)


# ─────────────────────────────────────────────────────────────────────
# Allowed lockfile origins.
# ─────────────────────────────────────────────────────────────────────
NPM_REGISTRY_PREFIX = "https://registry.npmjs.org/"

# Tarballs are also fetched from this mirror on some GH Actions cached
# runs (npm rewrites the resolved URL on cache hit). Allow either.
NPM_REGISTRY_PREFIXES_ALLOWED: tuple[str, ...] = (NPM_REGISTRY_PREFIX,)

CARGO_REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"


# ─────────────────────────────────────────────────────────────────────
# Cargo non-registry source allowlist.
# ─────────────────────────────────────────────────────────────────────
#
# Each entry is `(crate_name, exact_source_string)`. The crate must
# match by name AND the source must match the full pinned-SHA string
# verbatim. Bumping the commit SHA forces a re-review here: the
# scanner fires until the new SHA is appended.
#
# Studio's Tauri shell pulls `fix-path-env` directly from
# tauri-apps/fix-path-env-rs because the crate is not published to
# crates.io. The pinned commit (c4c45d5) was reviewed at the time it
# landed; future bumps need explicit approval.
CARGO_SOURCE_ALLOWLIST: tuple[tuple[str, str], ...] = (
    (
        "fix-path-env",
        "git+https://github.com/tauri-apps/fix-path-env-rs#"
        "c4c45d503ea115a839aae718d02f79e7c7f0f673",
    ),
)


# ─────────────────────────────────────────────────────────────────────
# Finding container.
# ─────────────────────────────────────────────────────────────────────


class Finding:
    __slots__ = ("path", "package", "kind", "detail")

    def __init__(self, path: str, package: str, kind: str, detail: str) -> None:
        self.path = path
        self.package = package
        self.kind = kind
        self.detail = detail

    def __str__(self) -> str:
        return (
            f"  [{self.kind}] {self.path}\n"
            f"    package: {self.package}\n"
            f"    detail:  {self.detail}"
        )


# ─────────────────────────────────────────────────────────────────────
# package-lock.json audit.
# ─────────────────────────────────────────────────────────────────────


def audit_npm_lockfile(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not path.exists():
        return findings

    raw = path.read_text(encoding = "utf-8")
    try:
        lock = json.loads(raw)
    except json.JSONDecodeError as exc:
        findings.append(
            Finding(
                path = str(path),
                package = "<root>",
                kind = "malformed-lockfile",
                detail = f"could not parse as JSON: {exc}",
            )
        )
        return findings

    lockfile_version = lock.get("lockfileVersion")
    if lockfile_version not in (2, 3):
        findings.append(
            Finding(
                path = str(path),
                package = "<root>",
                kind = "unsupported-lockfile-version",
                detail = (f"only lockfileVersion 2 or 3 audited; got {lockfile_version}"),
            )
        )

    packages = lock.get("packages") or {}
    for key, entry in packages.items():
        # The empty key "" is the project root; workspace entries use
        # keys like "node_modules/foo" or "studio/frontend/sub-pkg".
        # Skip the project root (it has no `resolved`).
        if key == "":
            continue
        if entry.get("link"):
            # Workspace symlink; no tarball to resolve.
            continue

        resolved = entry.get("resolved")
        # Entries living inside another package's `node_modules/`
        # tree are bundled fold-ins -- the parent's tarball ships
        # their source verbatim and the parent's `integrity` covers
        # the whole subtree. npm represents them in lockfileVersion 3
        # as nested entries with no `resolved` and no `integrity` of
        # their own. Treat them as transparent to this audit.
        nested = key.count("/node_modules/") >= 1

        # 1. resolved-URL origin.
        if resolved is None:
            if nested or entry.get("bundled"):
                # Bundled / fold-in entry; covered by parent integrity.
                pass
            elif entry.get("version"):
                # Top-level entry without a resolved URL is suspicious.
                findings.append(
                    Finding(
                        path = str(path),
                        package = key,
                        kind = "missing-resolved-url",
                        detail = (
                            f"version={entry['version']!r} but no `resolved` "
                            "field; lockfile is incomplete"
                        ),
                    )
                )
        else:
            if not any(resolved.startswith(p) for p in NPM_REGISTRY_PREFIXES_ALLOWED):
                findings.append(
                    Finding(
                        path = str(path),
                        package = key,
                        kind = "non-registry-resolved-url",
                        detail = (
                            f"resolved={resolved!r}; only "
                            f"{NPM_REGISTRY_PREFIX} is permitted. Direct "
                            "GitHub / git / file references are the "
                            "Shai-Hulud injection vector."
                        ),
                    )
                )

        # 2. integrity-hash presence.
        if resolved is not None and not entry.get("integrity"):
            findings.append(
                Finding(
                    path = str(path),
                    package = key,
                    kind = "missing-integrity-hash",
                    detail = (
                        "no `integrity` field; npm cannot verify the "
                        "tarball SHA against the registry-published hash"
                    ),
                )
            )

    # 3. Known IOC strings: scan the raw file body so we hit fields the
    #    structural pass above doesn't enumerate (scripts, optional
    #    dependencies, etc.). Cheap and complete.
    for ioc in NPM_IOC_STRINGS:
        if ioc in raw:
            # Best-effort line number lookup.
            line_no = _first_line_containing(raw, ioc)
            findings.append(
                Finding(
                    path = f"{path}:{line_no}" if line_no else str(path),
                    package = "<ioc-match>",
                    kind = "known-ioc-string",
                    detail = (
                        f"matched known IOC substring {ioc!r}; this is "
                        "a public indicator of a recent supply-chain "
                        "compromise. Refuse to install."
                    ),
                )
            )

    return findings


def _first_line_containing(text: str, needle: str) -> int | None:
    for i, line in enumerate(text.splitlines(), start = 1):
        if needle in line:
            return i
    return None


# ─────────────────────────────────────────────────────────────────────
# Cargo.lock audit.
# ─────────────────────────────────────────────────────────────────────


# Cargo.lock is TOML; parse with stdlib tomllib (Python 3.11+). The
# studio's Tauri shell already requires a modern toolchain so this is
# always available where CI runs.
_PACKAGE_HEADER = re.compile(r"^\[\[package\]\]\s*$")


def audit_cargo_lockfile(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not path.exists():
        return findings

    raw = path.read_text(encoding = "utf-8")
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        # Python <3.11; fall back to a tomli shim if importable.
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            findings.append(
                Finding(
                    path = str(path),
                    package = "<root>",
                    kind = "missing-toml-parser",
                    detail = (
                        "Python 3.11+ tomllib or tomli is required to "
                        "parse Cargo.lock; install tomli or upgrade "
                        "Python before re-running this audit"
                    ),
                )
            )
            return findings

    try:
        lock = tomllib.loads(raw)
    except Exception as exc:
        findings.append(
            Finding(
                path = str(path),
                package = "<root>",
                kind = "malformed-lockfile",
                detail = f"could not parse as TOML: {exc}",
            )
        )
        return findings

    for entry in lock.get("package", []):
        name = entry.get("name") or "<unnamed>"
        version = entry.get("version") or "<unversioned>"
        source = entry.get("source")
        # Workspace-local crates have no `source` field; skip them.
        if source is None:
            continue
        if source != CARGO_REGISTRY_SOURCE:
            if (name, source) in CARGO_SOURCE_ALLOWLIST:
                # Pre-approved non-registry source pinned by SHA.
                pass
            else:
                findings.append(
                    Finding(
                        path = str(path),
                        package = f"{name}@{version}",
                        kind = "non-registry-cargo-source",
                        detail = (
                            f"source={source!r}; only "
                            f"{CARGO_REGISTRY_SOURCE!r} is permitted "
                            "by default, and no allowlist entry covers "
                            "this crate. If the source is legitimate, "
                            "add `(name, source)` to "
                            "CARGO_SOURCE_ALLOWLIST after reviewing the "
                            "pinned commit."
                        ),
                    )
                )
        if not entry.get("checksum") and source == CARGO_REGISTRY_SOURCE:
            findings.append(
                Finding(
                    path = str(path),
                    package = f"{name}@{version}",
                    kind = "missing-cargo-checksum",
                    detail = (
                        "registry crate without checksum; cargo cannot "
                        "verify the downloaded source against the "
                        "registry-published SHA"
                    ),
                )
            )

    for ioc in CARGO_IOC_STRINGS:
        if ioc in raw:
            line_no = _first_line_containing(raw, ioc)
            findings.append(
                Finding(
                    path = f"{path}:{line_no}" if line_no else str(path),
                    package = "<ioc-match>",
                    kind = "known-ioc-string",
                    detail = f"matched known IOC substring {ioc!r}",
                )
            )

    return findings


# ─────────────────────────────────────────────────────────────────────
# CLI.
# ─────────────────────────────────────────────────────────────────────


DEFAULT_NPM_LOCKFILES = ("studio/frontend/package-lock.json",)
DEFAULT_CARGO_LOCKFILES = ("studio/src-tauri/Cargo.lock",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Pre-install lockfile supply-chain audit.",
    )
    parser.add_argument(
        "--root",
        default = str(REPO_ROOT),
        help = "Repo root (default: parent of this script).",
    )
    parser.add_argument(
        "--npm-lockfile",
        action = "append",
        default = None,
        help = (
            "Path to a package-lock.json (repeatable). "
            "Default: studio/frontend/package-lock.json."
        ),
    )
    parser.add_argument(
        "--cargo-lockfile",
        action = "append",
        default = None,
        help = (
            "Path to a Cargo.lock (repeatable). "
            "Default: studio/src-tauri/Cargo.lock."
        ),
    )
    args = parser.parse_args(argv)

    if os.environ.get("UNSLOTH_LOCKFILE_AUDIT_SKIP") == "1":
        print(
            "[lockfile-audit] UNSLOTH_LOCKFILE_AUDIT_SKIP=1; "
            "audit skipped (expected only for local triage)",
            flush = True,
        )
        return 0

    root = Path(args.root).resolve()
    npm_paths = [root / p for p in (args.npm_lockfile or DEFAULT_NPM_LOCKFILES)]
    cargo_paths = [root / p for p in (args.cargo_lockfile or DEFAULT_CARGO_LOCKFILES)]

    all_findings: list[Finding] = []
    for p in npm_paths:
        print(f"[lockfile-audit] npm: {p}", flush = True)
        all_findings.extend(audit_npm_lockfile(p))
    for p in cargo_paths:
        print(f"[lockfile-audit] cargo: {p}", flush = True)
        all_findings.extend(audit_cargo_lockfile(p))

    if not all_findings:
        print(
            f"[lockfile-audit] OK: 0 findings across "
            f"{len(npm_paths)} npm + {len(cargo_paths)} cargo lockfile(s)",
            flush = True,
        )
        return 0

    print(
        f"\n[lockfile-audit] FAIL: {len(all_findings)} finding(s):\n",
        file = sys.stderr,
    )
    for f in all_findings:
        print(str(f), file = sys.stderr)
        print(file = sys.stderr)
    print(
        "[lockfile-audit] Refusing to proceed. Each finding above is "
        "either a structural lockfile anomaly or a public indicator-of-"
        "compromise. Investigate before running `npm ci` or `cargo fetch`.",
        file = sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
