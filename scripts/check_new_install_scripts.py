#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Diff two `package-lock.json` files and flag NEW install-script deps.

A package with `"hasInstallScript": true` runs `preinstall` / `install` /
`postinstall` lifecycle hooks every time `npm ci` lays it down. Every
npm supply-chain compromise of the last 18 months (Shai-Hulud,
TanStack, axios-style, ArmorCode hijacks) leveraged exactly this lever:
the attacker publishes a new malicious version of a dep we already
trust, and the post-install hook runs the next time CI installs.

This scanner refuses to allow a newly-introduced install-script dep to
land without a maintainer eyeball on the lifecycle script body.
Existing install-script deps are NOT re-flagged -- if `node-gyp` has
been in the lockfile since day one, it's not part of this PR's threat
model. Only new entries are surfaced.

Supports lockfileVersion 1 (`dependencies` key, recursive), 2 and 3
(flat `packages` key with `node_modules/<a>/node_modules/<b>` nesting
for transitive entries). For each NEW install-script package we
attempt a stdlib-only fetch of
`https://registry.npmjs.org/<name>/<version>` to recover the actual
postinstall command body. If the network is blocked we still emit the
finding -- the lifecycle command body is informational, not
load-bearing.

Exit codes
==========
  0  no newly-added install-script deps
  1  one or more newly-added install-script deps; listed on stderr
  2  internal error (missing lockfile, malformed JSON, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REGISTRY_BASE = "https://registry.npmjs.org/"
REGISTRY_TIMEOUT_SECS = 5

CRITICAL = "CRITICAL"
HIGH = "HIGH"


class Finding:
    __slots__ = ("severity", "name", "version", "kind", "detail")

    def __init__(
        self, severity: str, name: str, version: str, kind: str, detail: str
    ) -> None:
        self.severity = severity
        self.name = name
        self.version = version
        self.kind = kind
        self.detail = detail

    def __str__(self) -> str:
        return (
            f"  [{self.severity}] {self.name}@{self.version}\n"
            f"    kind:   {self.kind}\n"
            f"    detail: {self.detail}"
        )


# ─────────────────────────────────────────────────────────────────────
# Lockfile parsing.
# ─────────────────────────────────────────────────────────────────────


def _strip_nm_prefix(key: str) -> str:
    """Convert a v2/v3 `packages` key into a bare package name.

    `node_modules/foo` -> `foo`; `node_modules/foo/node_modules/bar` ->
    `bar`. The empty key (`""`) is the project root and returns "".
    """
    if not key:
        return ""
    # Use the LAST `node_modules/` segment so transitives map to their
    # leaf name, matching how npm install resolves a postinstall.
    marker = "node_modules/"
    idx = key.rfind(marker)
    if idx == -1:
        return key
    return key[idx + len(marker) :]


def _collect_install_script_entries(lock: dict) -> dict[str, str]:
    """Walk a parsed lockfile and return {package_name: version} for
    every entry with `hasInstallScript: true` (v2/v3) OR a
    non-empty `scripts.preinstall|install|postinstall` (v1).

    The same package may appear at multiple versions in a single
    lockfile (de-duplicated copies under different parents); we key by
    `name@version` so we don't lose either copy. Returns a dict keyed
    by `name@version` -> the same string for convenience.
    """
    seen: dict[str, str] = {}
    version = lock.get("lockfileVersion")

    # v2 / v3: flat `packages` map.
    packages = lock.get("packages") or {}
    for key, entry in packages.items():
        if key == "" or not isinstance(entry, dict):
            continue
        if entry.get("link"):
            continue
        if not entry.get("hasInstallScript"):
            continue
        name = _strip_nm_prefix(key)
        if not name:
            continue
        ver = entry.get("version") or "<unversioned>"
        seen[f"{name}@{ver}"] = name

    # v1 also embeds a `dependencies` tree; v2/v3 carry both for
    # backwards-compat but `packages` is canonical for them. For v1
    # there is no `hasInstallScript` flag, so look for a non-empty
    # `scripts.preinstall|install|postinstall` directly.
    def _walk_v1(deps: dict, depth: int = 0) -> None:
        if depth > 64 or not isinstance(deps, dict):
            return
        for name, entry in deps.items():
            if not isinstance(entry, dict):
                continue
            scripts = entry.get("scripts") or {}
            lifecycle = any(
                isinstance(scripts, dict) and scripts.get(hook)
                for hook in ("preinstall", "install", "postinstall")
            )
            # v1 also sets `requires` only on the parent, no flag, so
            # the lifecycle-script presence is the only signal.
            if lifecycle:
                ver = entry.get("version") or "<unversioned>"
                seen[f"{name}@{ver}"] = name
            _walk_v1(entry.get("dependencies"), depth = depth + 1)

    if version == 1 or "dependencies" in lock:
        _walk_v1(lock.get("dependencies") or {})

    return seen


def _load_lockfile(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"lockfile not found: {path}")
    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────
# Registry lookup for the postinstall command body (best-effort).
# ─────────────────────────────────────────────────────────────────────


def _fetch_registry_scripts(name: str, version: str) -> dict[str, str] | None:
    """Return {hook: command} for any of preinstall / install /
    postinstall published in the registry metadata for this name@ver.

    Returns None on any error (network blocked, 404, malformed JSON).
    Never raises; the caller treats absence as "could not enrich, emit
    finding anyway".
    """
    safe_name = urllib.parse.quote(name, safe = "@/")
    url = f"{REGISTRY_BASE}{safe_name}/{urllib.parse.quote(version)}"
    try:
        with urllib.request.urlopen(url, timeout = REGISTRY_TIMEOUT_SECS) as resp:
            body = resp.read()
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    try:
        meta = json.loads(body)
    except json.JSONDecodeError:
        return None
    scripts = meta.get("scripts") or {}
    if not isinstance(scripts, dict):
        return None
    keep = {}
    for hook in ("preinstall", "install", "postinstall"):
        cmd = scripts.get(hook)
        if isinstance(cmd, str) and cmd.strip():
            keep[hook] = cmd
    return keep or None


# ─────────────────────────────────────────────────────────────────────
# Diff.
# ─────────────────────────────────────────────────────────────────────


def diff_new_install_scripts(base_lock: dict, head_lock: dict) -> list[Finding]:
    base = _collect_install_script_entries(base_lock)
    head = _collect_install_script_entries(head_lock)
    findings: list[Finding] = []
    for key in sorted(head):
        if key in base:
            continue  # pre-existing install-script dep; not in scope
        name = head[key]
        # key is "name@version"; rsplit("@", 1) handles scoped names.
        version = (
            key[len(name) + 1 :] if key.startswith(name + "@") else "<unversioned>"
        )
        scripts = _fetch_registry_scripts(name, version)
        if scripts:
            detail = "; ".join(f"{h}={cmd!r}" for h, cmd in scripts.items())
        else:
            detail = (
                "newly added with hasInstallScript=true; registry "
                "metadata unreachable -- inspect the package's "
                "scripts.{preinstall,install,postinstall} manually"
            )
        findings.append(
            Finding(
                severity = CRITICAL,
                name = name,
                version = version,
                kind = "new-install-script",
                detail = detail,
            )
        )
    return findings


# ─────────────────────────────────────────────────────────────────────
# CLI.
# ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description = (
            "Diff two package-lock.json files and refuse any newly-"
            "added install-script dep."
        ),
    )
    parser.add_argument(
        "--base",
        required = True,
        help = "Path to the BASE package-lock.json (e.g. main branch).",
    )
    parser.add_argument(
        "--head",
        required = True,
        help = "Path to the HEAD package-lock.json (this PR).",
    )
    args = parser.parse_args(argv)

    try:
        base_lock = _load_lockfile(Path(args.base))
        head_lock = _load_lockfile(Path(args.head))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[install-script-diff] ERROR: {exc}", file = sys.stderr)
        return 2

    findings = diff_new_install_scripts(base_lock, head_lock)
    if not findings:
        print(
            "[install-script-diff] OK: no newly-added install-script "
            "dependencies between base and head",
            flush = True,
        )
        return 0

    print(
        f"\n[install-script-diff] FAIL: {len(findings)} newly-added "
        f"install-script dependency(ies):\n",
        file = sys.stderr,
    )
    for f in findings:
        print(str(f), file = sys.stderr)
        print(file = sys.stderr)
    print(
        "[install-script-diff] Refusing to proceed. Every new "
        "install-script dep is a postinstall lifecycle hook that "
        "would run on the next `npm ci`. Review each finding above, "
        "confirm the maintainer + version, and re-run.",
        file = sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
