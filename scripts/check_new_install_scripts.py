#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Diff two `package-lock.json` files and flag NEW install-script deps.

Packages with `hasInstallScript: true` run preinstall/install/postinstall
on every `npm ci`. Every npm supply-chain compromise of the last 18 months
(Shai-Hulud, TanStack, axios, ArmorCode) used this lever: malicious
version of a trusted dep, hook runs on next install.

Existing install-script deps are NOT re-flagged; only NEW entries surface.

Supports lockfileVersion 1 (`dependencies`, recursive) and 2/3 (flat
`packages` with `node_modules/<a>/node_modules/<b>` nesting). For each
finding we try a stdlib fetch of `https://registry.npmjs.org/<name>/<version>`
to recover the lifecycle body; network failure is non-fatal.

Exit codes:
  0  no newly-added install-script deps
  1  newly-added install-script deps listed on stderr
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
    """v2/v3 `packages` key -> bare leaf name. ``""`` is the project root."""
    if not key:
        return ""
    # Last `node_modules/` segment: transitives map to leaf name (matches
    # how npm resolves the postinstall).
    marker = "node_modules/"
    idx = key.rfind(marker)
    if idx == -1:
        return key
    return key[idx + len(marker) :]


def _collect_install_script_entries(lock: dict) -> dict[str, str]:
    """Walk a parsed lockfile and return {package_name: version} for
    every entry with `hasInstallScript: true` (v2/v3) OR a
    non-empty `scripts.preinstall|install|postinstall` (v1).

    Keyed by `name@version` so duplicate copies at different versions
    are not collapsed. Returns {`name@version`: name}.
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

    # v1 has no `hasInstallScript` flag -- detect via non-empty
    # scripts.{preinstall,install,postinstall}.
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


# ─────────────────────────────────────────────────────────────────────
# Registry lookup for the postinstall command body (best-effort).
# ─────────────────────────────────────────────────────────────────────


def _fetch_registry_scripts(name: str, version: str) -> dict[str, str] | None:
    """Return {hook: command} from registry metadata, or None on any failure.

    Never raises; absence means "emit finding without enrichment".
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
            continue  # pre-existing dep, out of scope
        name = head[key]
        version = (
            key[len(name) + 1 :] if key.startswith(name + "@") else "<unversioned>"
        )
        scripts = _fetch_registry_scripts(name, version)
        if scripts:
            detail = "; ".join(f"{h}={cmd!r}" for h, cmd in scripts.items())
        else:
            detail = (
                "newly added with hasInstallScript=true; registry "
                "unreachable -- inspect scripts.{preinstall,install,postinstall}"
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


def _load_allowlist(path: Path) -> set[str]:
    """Newline-separated `name@version` entries to skip.

    Each entry MUST be pinned to an exact version; bare names rejected so
    a compromised maintainer cannot ship a malicious later version under
    an existing allowlist line. `#` comments and blank lines ignored.
    Missing file = empty set.
    """
    if not path.exists():
        return set()
    out: set[str] = set()
    try:
        text = path.read_text(encoding = "utf-8")
    except OSError:
        return set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.rpartition("@")
        if not sep or not name or not version:
            raise ValueError(
                f"{path}: allowlist entry {line!r} must be pinned to an "
                "exact version (e.g. 'esbuild@0.21.5'). Bare names are "
                "rejected so we cannot silently approve a later release.",
            )
        out.add(line.lower())
    return out


def _finding_allowlist_key(finding: Finding) -> str:
    return f"{finding.name}@{finding.version}".lower()


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
    parser.add_argument(
        "--allowlist",
        default = None,
        help = (
            "Path to the HEAD newline-separated 'name@version' allowlist "
            "to skip. Defaults to '<head dir>/.install-script-allowlist'."
        ),
    )
    parser.add_argument(
        "--base-allowlist",
        default = None,
        help = (
            "Path to the TRUSTED BASE allowlist. Defaults to "
            "'<base dir>/.install-script-allowlist'. Head-only entries that "
            "self-approve a new postinstall fail the gate."
        ),
    )
    args = parser.parse_args(argv)

    try:
        base_lock = _load_lockfile(Path(args.base))
        head_lock = _load_lockfile(Path(args.head))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[install-script-diff] ERROR: {exc}", file = sys.stderr)
        return 2

    head_allowlist_path = (
        Path(args.allowlist)
        if args.allowlist
        else Path(args.head).parent / ".install-script-allowlist"
    )
    base_allowlist_path = (
        Path(args.base_allowlist)
        if args.base_allowlist
        else Path(args.base).parent / ".install-script-allowlist"
    )

    try:
        head_allowlist = _load_allowlist(head_allowlist_path)
    except ValueError as exc:
        print(f"[install-script-diff] ERROR: {exc}", file = sys.stderr)
        return 2

    raw_findings = diff_new_install_scripts(base_lock, head_lock)
    raw_findings_keys = {_finding_allowlist_key(f) for f in raw_findings}

    # Bootstrap: PR that creates the file. Workflow signals "missing on base"
    # by `rm -f` after a failed `git show`. After landing, head-only / delete
    # rules below kick in.
    if not base_allowlist_path.exists():
        print(
            f"[install-script-diff] bootstrap: {base_allowlist_path} "
            "missing on base; accepting head allowlist as-is for this run.",
            flush = True,
        )
        allowlist = head_allowlist
    else:
        try:
            base_allowlist = _load_allowlist(base_allowlist_path)
        except ValueError as exc:
            print(f"[install-script-diff] ERROR: {exc}", file = sys.stderr)
            return 2

        added_head_only = head_allowlist - base_allowlist
        # Refuse "introduce a new postinstall AND allowlist it in the same
        # diff". Head-only entries that match no current finding are fine
        # (prepare trust for a follow-up PR; reviewer still sees the diff).
        self_approving = sorted(added_head_only & raw_findings_keys)
        if self_approving:
            print(
                "[install-script-diff] FAIL: this PR both introduces an "
                "install-script dependency AND allowlists it in the "
                "same diff. Allowlist entries that approve a NEW "
                "postinstall must land on the base branch first.",
                file = sys.stderr,
            )
            for entry in self_approving:
                print(
                    f"  self-approving allowlist entry: {entry}",
                    file = sys.stderr,
                )
            return 1

        # Refuse PRs that DROP trusted base entries; otherwise a two-step
        # bypass works (PR A removes file -> PR B hits bootstrap and
        # self-allowlists). Deletions go via admin override.
        removed_from_head = sorted(base_allowlist - head_allowlist)
        if removed_from_head:
            print(
                "[install-script-diff] FAIL: PR removes trusted base "
                "allowlist entries. Allowlist deletions must land in a "
                "separate, isolated commit so a follow-up PR cannot "
                "exploit the bootstrap path.",
                file = sys.stderr,
            )
            for entry in removed_from_head:
                print(
                    f"  dropped allowlist entry: {entry}",
                    file = sys.stderr,
                )
            return 1

        # Only base allowlist applies. Head-only entries that survived
        # the self-approval check wait for the next PR to take effect.
        allowlist = base_allowlist

    findings = raw_findings
    if allowlist:
        skipped = [f for f in findings if _finding_allowlist_key(f) in allowlist]
        findings = [f for f in findings if _finding_allowlist_key(f) not in allowlist]
        for f in skipped:
            print(
                f"[install-script-diff] SKIP {_finding_allowlist_key(f)} "
                "(allowlisted via trusted base allowlist)",
                flush = True,
            )
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
        "[install-script-diff] Refusing to proceed. Each new install-script "
        "dep ships a postinstall hook that runs on next `npm ci`. Review, "
        "confirm maintainer + version, and re-run.",
        file = sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
