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

  0  no findings, or an opt-out env var (UNSLOTH_LOCKFILE_AUDIT_SKIP)
     is set to a justification string (>=5 chars, not '1'/'true'/etc).
     A value like '1' or 'true' is now REJECTED loudly and the audit
     runs normally
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
    # Mini Shai-Hulud May-12 2026 wave.
    "git-tanstack.com",
    "transformers.pyz",
    "/tmp/transformers.pyz",
    "With Love TeamPCP",
    # Aikido (May-12 wave): payload SHA-256 hashes + Bun marker.
    "ab4fcadaec49c03278063dd269ea5eef82d24f2124a8e15d7b90f2fa8601266c",
    "2ec78d556d696e208927cc503d48e4b5eb56b31abc2870c2ed2e98d6be27fc96",
    "bun run tanstack_runner.js",
    "We've been online over 2 hours",
)

# Hard pin-blocks for publicly confirmed malicious versions.
# keep in sync with scripts/scan_npm_packages.py
BLOCKED_NPM_VERSIONS: dict[str, set[str]] = {
    # GHSA-g7cv-rxg3-hmpx -- TanStack May-11 2026 (84 versions).
    "@tanstack/arktype-adapter": {"1.166.12", "1.166.15"},
    "@tanstack/eslint-plugin-router": {"1.161.9", "1.161.12"},
    "@tanstack/eslint-plugin-start": {"0.0.4", "0.0.7"},
    "@tanstack/history": {"1.161.9", "1.161.12"},
    "@tanstack/nitro-v2-vite-plugin": {"1.154.12", "1.154.15"},
    "@tanstack/react-router": {"1.169.5", "1.169.8"},
    "@tanstack/react-router-devtools": {"1.166.16", "1.166.19"},
    "@tanstack/react-router-ssr-query": {"1.166.15", "1.166.18"},
    "@tanstack/react-start": {"1.167.68", "1.167.71"},
    "@tanstack/react-start-client": {"1.166.51", "1.166.54"},
    "@tanstack/react-start-rsc": {"0.0.47", "0.0.50"},
    "@tanstack/react-start-server": {"1.166.55", "1.166.58"},
    "@tanstack/router-cli": {"1.166.46", "1.166.49"},
    "@tanstack/router-core": {"1.169.5", "1.169.8"},
    "@tanstack/router-devtools": {"1.166.16", "1.166.19"},
    "@tanstack/router-devtools-core": {"1.167.6", "1.167.9"},
    "@tanstack/router-generator": {"1.166.45", "1.166.48"},
    "@tanstack/router-plugin": {"1.167.38", "1.167.41"},
    "@tanstack/router-ssr-query-core": {"1.168.3", "1.168.6"},
    "@tanstack/router-utils": {"1.161.11", "1.161.14"},
    "@tanstack/router-vite-plugin": {"1.166.53", "1.166.56"},
    "@tanstack/solid-router": {"1.169.5", "1.169.8"},
    "@tanstack/solid-router-devtools": {"1.166.16", "1.166.19"},
    "@tanstack/solid-router-ssr-query": {"1.166.15", "1.166.18"},
    "@tanstack/solid-start": {"1.167.65", "1.167.68"},
    "@tanstack/solid-start-client": {"1.166.50", "1.166.53"},
    "@tanstack/solid-start-server": {"1.166.54", "1.166.57"},
    "@tanstack/start-client-core": {"1.168.5", "1.168.8"},
    "@tanstack/start-fn-stubs": {"1.161.9", "1.161.12"},
    "@tanstack/start-plugin-core": {"1.169.23", "1.169.26"},
    "@tanstack/start-server-core": {"1.167.33", "1.167.36"},
    "@tanstack/start-static-server-functions": {"1.166.44", "1.166.47"},
    "@tanstack/start-storage-context": {"1.166.38", "1.166.41"},
    "@tanstack/valibot-adapter": {"1.166.12", "1.166.15"},
    "@tanstack/virtual-file-routes": {"1.161.10", "1.161.13"},
    "@tanstack/vue-router": {"1.169.5", "1.169.8"},
    "@tanstack/vue-router-devtools": {"1.166.16", "1.166.19"},
    "@tanstack/vue-router-ssr-query": {"1.166.15", "1.166.18"},
    "@tanstack/vue-start": {"1.167.61", "1.167.64"},
    "@tanstack/vue-start-client": {"1.166.46", "1.166.49"},
    "@tanstack/vue-start-server": {"1.166.50", "1.166.53"},
    "@tanstack/zod-adapter": {"1.166.12", "1.166.15"},
    # Mini Shai-Hulud May-12 wave: OpenSearch JS client.
    "@opensearch-project/opensearch": {"3.5.3", "3.6.2", "3.7.0", "3.8.0"},
    # Mini Shai-Hulud May-12 wave: @squawk/* (22 packages, 5 versions each;
    # https://safedep.io/mass-npm-supply-chain-attack-tanstack-mistral/).
    "@squawk/airport-data": {"0.7.4", "0.7.5", "0.7.6", "0.7.7", "0.7.8"},
    "@squawk/airports": {"0.6.2", "0.6.3", "0.6.4", "0.6.5", "0.6.6"},
    "@squawk/airspace": {"0.8.1", "0.8.2", "0.8.3", "0.8.4", "0.8.5"},
    "@squawk/airspace-data": {"0.5.3", "0.5.4", "0.5.5", "0.5.6", "0.5.7"},
    "@squawk/airway-data": {"0.5.4", "0.5.5", "0.5.6", "0.5.7", "0.5.8"},
    "@squawk/airways": {"0.4.2", "0.4.3", "0.4.4", "0.4.5", "0.4.6"},
    "@squawk/fix-data": {"0.6.4", "0.6.5", "0.6.6", "0.6.7", "0.6.8"},
    "@squawk/fixes": {"0.3.2", "0.3.3", "0.3.4", "0.3.5", "0.3.6"},
    "@squawk/flight-math": {"0.5.4", "0.5.5", "0.5.6", "0.5.7", "0.5.8"},
    "@squawk/flightplan": {"0.5.2", "0.5.3", "0.5.4", "0.5.5", "0.5.6"},
    "@squawk/geo": {"0.4.4", "0.4.5", "0.4.6", "0.4.7", "0.4.8"},
    "@squawk/icao-registry": {"0.5.2", "0.5.3", "0.5.4", "0.5.5", "0.5.6"},
    "@squawk/icao-registry-data": {"0.8.4", "0.8.5", "0.8.6", "0.8.7", "0.8.8"},
    "@squawk/mcp": {"0.9.1", "0.9.2", "0.9.3", "0.9.4", "0.9.5"},
    "@squawk/navaid-data": {"0.6.4", "0.6.5", "0.6.6", "0.6.7", "0.6.8"},
    "@squawk/navaids": {"0.4.2", "0.4.3", "0.4.4", "0.4.5", "0.4.6"},
    "@squawk/notams": {"0.3.6", "0.3.7", "0.3.8", "0.3.9", "0.3.10"},
    "@squawk/procedure-data": {"0.7.3", "0.7.4", "0.7.5", "0.7.6", "0.7.7"},
    "@squawk/procedures": {"0.5.2", "0.5.3", "0.5.4", "0.5.5", "0.5.6"},
    "@squawk/types": {"0.8.1", "0.8.2", "0.8.3", "0.8.4", "0.8.5"},
    "@squawk/units": {"0.4.3", "0.4.4", "0.4.5", "0.4.6", "0.4.7"},
    "@squawk/weather": {"0.5.6", "0.5.7", "0.5.8", "0.5.9", "0.5.10"},
    # Mini Shai-Hulud May-12 wave: @uipath/* (64 packages, single version each;
    # https://www.aikido.dev/blog/mini-shai-hulud-is-back-tanstack-compromised).
    "@uipath/apollo-react": {"4.24.5"},
    "@uipath/apollo-wind": {"2.16.2"},
    "@uipath/cli": {"1.0.1"},
    "@uipath/rpa-tool": {"0.9.5"},
    "@uipath/apollo-core": {"5.9.2"},
    "@uipath/filesystem": {"1.0.1"},
    "@uipath/solutionpackager-tool-core": {"0.0.34"},
    "@uipath/solution-tool": {"1.0.1"},
    "@uipath/maestro-tool": {"1.0.1"},
    "@uipath/codedapp-tool": {"1.0.1"},
    "@uipath/agent-tool": {"1.0.1"},
    "@uipath/orchestrator-tool": {"1.0.1"},
    "@uipath/integrationservice-tool": {"1.0.2"},
    "@uipath/rpa-legacy-tool": {"1.0.1"},
    "@uipath/vertical-solutions-tool": {"1.0.1"},
    "@uipath/flow-tool": {"1.0.2"},
    "@uipath/codedagent-tool": {"1.0.1"},
    "@uipath/common": {"1.0.1"},
    "@uipath/resource-tool": {"1.0.1"},
    "@uipath/auth": {"1.0.1"},
    "@uipath/docsai-tool": {"1.0.1"},
    "@uipath/case-tool": {"1.0.1"},
    "@uipath/api-workflow-tool": {"1.0.1"},
    "@uipath/test-manager-tool": {"1.0.2"},
    "@uipath/robot": {"1.3.4"},
    "@uipath/traces-tool": {"1.0.1"},
    "@uipath/agent-sdk": {"1.0.2"},
    "@uipath/integrationservice-sdk": {"1.0.2"},
    "@uipath/maestro-sdk": {"1.0.1"},
    "@uipath/data-fabric-tool": {"1.0.2"},
    "@uipath/tasks-tool": {"1.0.1"},
    "@uipath/insights-tool": {"1.0.1"},
    "@uipath/insights-sdk": {"1.0.1"},
    "@uipath/uipath-python-bridge": {"1.0.1"},
    "@uipath/ap-chat": {"1.5.7"},
    "@uipath/project-packager": {"1.1.16"},
    "@uipath/packager-tool-case": {"0.0.9"},
    "@uipath/packager-tool-workflowcompiler-browser": {"0.0.34"},
    "@uipath/packager-tool-connector": {"0.0.19"},
    "@uipath/packager-tool-workflowcompiler": {"0.0.16"},
    "@uipath/packager-tool-webapp": {"1.0.6"},
    "@uipath/packager-tool-apiworkflow": {"0.0.19"},
    "@uipath/packager-tool-functions": {"0.1.1"},
    "@uipath/widget.sdk": {"1.2.3"},
    "@uipath/resources-tool": {"0.1.11"},
    "@uipath/agent.sdk": {"0.0.18"},
    "@uipath/codedagents-tool": {"0.1.12"},
    "@uipath/aops-policy-tool": {"0.3.1"},
    "@uipath/solution-packager": {"0.0.35"},
    "@uipath/packager-tool-bpmn": {"0.0.9"},
    "@uipath/packager-tool-flow": {"0.0.19"},
    "@uipath/telemetry": {"0.0.7"},
    "@uipath/tool-workflowcompiler": {"0.0.12"},
    "@uipath/vss": {"0.1.6"},
    "@uipath/solutionpackager-sdk": {"1.0.11"},
    "@uipath/ui-widgets-multi-file-upload": {"1.0.1"},
    "@uipath/access-policy-tool": {"0.3.1"},
    "@uipath/context-grounding-tool": {"0.1.1"},
    "@uipath/gov-tool": {"0.3.1"},
    "@uipath/admin-tool": {"0.1.1"},
    "@uipath/identity-tool": {"0.1.1"},
    "@uipath/llmgw-tool": {"1.0.1"},
    "@uipath/resourcecatalog-tool": {"0.1.1"},
    "@uipath/functions-tool": {"1.0.1"},
    "@uipath/access-policy-sdk": {"0.3.1"},
    "@uipath/platform-tool": {"1.0.1"},
    # Mini Shai-Hulud May-12 wave: @mistralai/* (npm) — separate from PyPI mistralai
    # (https://www.aikido.dev/blog/mini-shai-hulud-is-back-tanstack-compromised).
    "@mistralai/mistralai": {"2.2.2", "2.2.3", "2.2.4"},
    "@mistralai/mistralai-gcp": {"1.7.1", "1.7.2", "1.7.3"},
    "@mistralai/mistralai-azure": {"1.7.1", "1.7.2", "1.7.3"},
    # Mini Shai-Hulud May-12 wave: @tallyui/* (30 entries, 10 packages)
    # (Aikido enumeration).
    "@tallyui/components": {"1.0.1", "1.0.2", "1.0.3"},
    "@tallyui/connector-medusa": {"1.0.1", "1.0.2", "1.0.3"},
    "@tallyui/connector-shopify": {"1.0.1", "1.0.2", "1.0.3"},
    "@tallyui/connector-vendure": {"1.0.1", "1.0.2", "1.0.3"},
    "@tallyui/connector-woocommerce": {"1.0.1", "1.0.2", "1.0.3"},
    "@tallyui/core": {"0.2.1", "0.2.2", "0.2.3"},
    "@tallyui/database": {"1.0.1", "1.0.2", "1.0.3"},
    "@tallyui/pos": {"0.1.1", "0.1.2", "0.1.3"},
    "@tallyui/storage-sqlite": {"0.2.1", "0.2.2", "0.2.3"},
    "@tallyui/theme": {"0.2.1", "0.2.2", "0.2.3"},
    # Mini Shai-Hulud May-12 wave: @beproduct/nestjs-auth (18 versions)
    # (Aikido enumeration).
    "@beproduct/nestjs-auth": {
        "0.1.2",
        "0.1.3",
        "0.1.4",
        "0.1.5",
        "0.1.6",
        "0.1.7",
        "0.1.8",
        "0.1.9",
        "0.1.10",
        "0.1.11",
        "0.1.12",
        "0.1.13",
        "0.1.14",
        "0.1.15",
        "0.1.16",
        "0.1.17",
        "0.1.18",
        "0.1.19",
    },
    # Mini Shai-Hulud May-12 wave: @draftlab/* + @draftauth/*
    # (Aikido enumeration).
    "@draftauth/client": {"0.2.1", "0.2.2"},
    "@draftauth/core": {"0.13.1", "0.13.2"},
    "@draftlab/auth": {"0.24.1", "0.24.2"},
    "@draftlab/auth-router": {"0.5.1", "0.5.2"},
    "@draftlab/db": {"0.16.1"},
    # Mini Shai-Hulud May-12 wave: @taskflow-corp/cli + @tolka/cli
    # (Aikido enumeration).
    "@taskflow-corp/cli": {"0.1.24", "0.1.25", "0.1.26", "0.1.27", "0.1.28", "0.1.29"},
    "@tolka/cli": {"1.0.2", "1.0.3", "1.0.4", "1.0.5", "1.0.6"},
    # Mini Shai-Hulud May-12 wave: @ml-toolkit-ts/* + @mesadev/* + @dirigible-ai/sdk + @supersurkhet/*
    # (Aikido enumeration).
    "@dirigible-ai/sdk": {"0.6.2", "0.6.3"},
    "@mesadev/rest": {"0.28.3"},
    "@mesadev/saguaro": {"0.4.22"},
    "@mesadev/sdk": {"0.28.3"},
    "@ml-toolkit-ts/preprocessing": {"1.0.2", "1.0.3"},
    "@ml-toolkit-ts/xgboost": {"1.0.3", "1.0.4"},
    "@supersurkhet/cli": {"0.0.2", "0.0.3", "0.0.4", "0.0.5", "0.0.6", "0.0.7"},
    "@supersurkhet/sdk": {"0.0.2", "0.0.3", "0.0.4", "0.0.5", "0.0.6", "0.0.7"},
    # Mini Shai-Hulud May-12 wave: Unscoped packages (10 entries)
    # (Aikido enumeration).
    "safe-action": {"0.8.3", "0.8.4"},
    "ts-dna": {"3.0.1", "3.0.2", "3.0.3", "3.0.4"},
    "cross-stitch": {"1.1.3", "1.1.4", "1.1.5", "1.1.6"},
    "cmux-agent-mcp": {"0.1.3", "0.1.4", "0.1.5", "0.1.6", "0.1.7", "0.1.8"},
    "agentwork-cli": {"0.1.4", "0.1.5"},
    "git-branch-selector": {"1.3.3", "1.3.4", "1.3.5", "1.3.6", "1.3.7"},
    "wot-api": {"0.8.1", "0.8.2", "0.8.3", "0.8.4"},
    "git-git-git": {"1.0.8", "1.0.9", "1.0.10", "1.0.11", "1.0.12"},
    "nextmove-mcp": {"0.1.3", "0.1.4", "0.1.5", "0.1.6", "0.1.7"},
    "ml-toolkit-ts": {"1.0.4", "1.0.5"},
    # Cross-ecosystem Mini Shai-Hulud (Apr-30 wave): npm counterpart of
    # PyPI lightning 2.6.2/2.6.3. Same threat actor (TeamPCP) per Semgrep,
    # Aikido, OX Security, Resecurity. Safe version: 7.0.3 and earlier.
    "intercom-client": {"7.0.4"},
}

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

        # 3. Blocked malicious version list.
        nm_prefix = "node_modules/"
        pkg_name = key[len(nm_prefix) :] if key.startswith(nm_prefix) else key
        version = entry.get("version")
        blocked = BLOCKED_NPM_VERSIONS.get(pkg_name, set())
        if version and version in blocked:
            findings.append(
                Finding(
                    path = str(path),
                    package = key,
                    kind = "blocked-known-malicious",
                    detail = (
                        f"{pkg_name}@{version} is on the " "BLOCKED_NPM_VERSIONS list"
                    ),
                )
            )

    # 4. Known IOC strings: scan the raw file body so we hit fields the
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

    # SF4: require a real justification (e.g. JIRA ticket id) for the
    # skip env var. Treat the trivially-set values ("1", "true", "yes",
    # "on", empty) as INVALID -- they look like accidental flips and
    # silently bypassed the supply-chain audit. A valid value is a
    # non-empty string >=5 chars after stripping that does not match
    # any of the boolean-shaped tokens above. An invalid value emits a
    # loud GitHub Actions warning to stderr and FALLS THROUGH to run
    # the audit normally (fail-safe). A valid value emits a warning
    # naming the reason and skips with rc=0 (compat).
    _skip_raw = os.environ.get("UNSLOTH_LOCKFILE_AUDIT_SKIP")
    if _skip_raw is not None:
        _skip = _skip_raw.strip()
        _invalid_tokens = {"", "1", "0", "true", "false", "yes", "no", "on", "off"}
        if _skip.lower() in _invalid_tokens or len(_skip) < 5:
            print(
                "::warning::Lockfile audit skip REQUIRES a justification "
                f"value (>=5 chars, not '{_skip_raw}'). Proceeding with "
                "audit. Use e.g. UNSLOTH_LOCKFILE_AUDIT_SKIP=ticket-1234.",
                file = sys.stderr,
                flush = True,
            )
        else:
            print(
                f"::warning::Lockfile audit skipped: reason='{_skip}'",
                file = sys.stderr,
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
