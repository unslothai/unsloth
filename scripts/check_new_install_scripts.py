#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fails on newly added hasInstallScript deps in package-lock.json; existing ones are not flagged;
registry metadata fetch is best-effort and the finding is still emitted if unreachable"""

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

    def __init__(self, severity: str, name: str, version: str, kind: str, detail: str) -> None:
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


def _strip_nm_prefix(key: str) -> str:
    """Convert a v2/v3 `packages` key into a bare package name (leaf after last `node_modules/`)."""
    if not key:
        return ""
    marker = "node_modules/"
    idx = key.rfind(marker)
    if idx == -1:
        return key
    return key[idx + len(marker) :]


def _collect_install_script_entries(lock: dict) -> dict[str, str]:
    """Return {name@version: name} for entries with hasInstallScript (v2/v3) or a lifecycle script (v1).

    Keyed by name@version so dup copies at different versions aren't lost.
    """
    seen: dict[str, str] = {}
    version = lock.get("lockfileVersion")

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

    # v1 has no hasInstallScript flag; detect lifecycle scripts directly.
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


def _fetch_registry_scripts(name: str, version: str) -> dict[str, str] | None:
    """Return {hook: command} for lifecycle hooks in registry metadata; None on any error (never raises)."""
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


def diff_new_install_scripts(base_lock: dict, head_lock: dict) -> list[Finding]:
    base = _collect_install_script_entries(base_lock)
    head = _collect_install_script_entries(head_lock)
    findings: list[Finding] = []
    for key in sorted(head):
        if key in base:
            continue
        name = head[key]
        version = key[len(name) + 1 :] if key.startswith(name + "@") else "<unversioned>"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description = (
            "Diff two package-lock.json files and refuse any newly-added install-script dep."
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
