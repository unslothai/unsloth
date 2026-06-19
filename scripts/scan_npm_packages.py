#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# .github/workflows/security-audit.yml's npm-scan-packages job depends
# on this file existing at scripts/scan_npm_packages.py.

"""scan_npm_packages.py -- npm-side content scanner.

Counterpart to scripts/scan_packages.py for the pip ecosystem. Reads
studio/frontend/package-lock.json, downloads each resolved tarball
DIRECTLY from registry.npmjs.org (never via `npm install` -- no
lifecycle scripts ever run), verifies the lockfile integrity hash,
unpacks each tarball into a sandboxed temp dir behind size / count /
path-escape / symlink guards, and pattern-scans the extracted file
contents for the signatures common to npm supply-chain attacks:

  - Lifecycle (preinstall / install / postinstall / prepare) scripts
    in any package.json that fetch + execute external code.
  - C2 / exfiltration hosts (getsession.org, AWS IMDS endpoints,
    Kubernetes ServiceAccount token paths, GitHub Actions OIDC,
    HashiCorp Vault endpoints).
  - Credential-stealing references (~/.npmrc, ~/.aws/credentials,
    GITHUB_TOKEN / NPM_TOKEN in JS sources).
  - Known IOC filenames from public advisories
    (router_init.js, tanstack_runner.js, router_runtime.js).
  - Obfuscation shapes (large single JS in package root with a low
    whitespace ratio + Function/eval against a base64-decoded blob).

Safety stance
=============

This script ingests attacker-controlled archives. Every parse path
assumes the worst:

  1. Downloads ONLY from `registry.npmjs.org`. Any tarball URL with a
     different hostname is refused without fetching.
  2. Tarball download is size-capped (HARD_MAX_TARBALL_BYTES default
     64 MiB). HEAD-style probe via the Content-Length response header
     plus a chunked read that aborts on overflow.
  3. SHA-512 integrity verified against the lockfile entry BEFORE the
     tarball is even opened. A mismatch aborts that package -- the
     scanner does not "fall back" to the registry-published hash.
  4. tar extraction goes through `safe_extract`:
        - rejects symbolic links (`SYMTYPE`, `LNKTYPE`)
        - rejects absolute paths, `..` traversal, paths outside the
          extract root after resolution
        - rejects character / block / FIFO devices
        - per-file uncompressed size cap (HARD_MAX_FILE_BYTES, default
          8 MiB) AND cumulative cap (HARD_MAX_TOTAL_BYTES, default
          128 MiB) AND member-count cap (HARD_MAX_MEMBERS, default
          50_000)
        - tar reads happen via `tarfile.open(mode='r|gz')` streaming
          so an oversized file is detected before write
  5. NOTHING from the extracted tree is ever executed. Files are read
     as raw bytes, decoded with `errors='replace'`, and grepped. We
     never call `node`, `eval`, `compile`, `subprocess.run`,
     `os.system`, or anything that would touch the tarball's
     declared scripts.
  6. Tempdir is created with `tempfile.mkdtemp(prefix='npm-scan-')`,
     fully resolved with .resolve(), and registered with atexit to be
     wiped on every termination path.
  7. Stdlib only. No third-party deps -- adding one would itself be a
     supply-chain liability.

Exit codes
==========

  0  no findings of severity HIGH or higher
  1  one or more HIGH/CRITICAL findings (or pre-scan structural
     anomalies -- non-registry resolved URL, missing integrity)
  2  internal error (lockfile missing, integrity mismatch on
     download, malformed tarball, etc.)

The script is meant to be run in CI on every PR that touches
package-lock.json and on a nightly schedule.
"""

from __future__ import annotations

import argparse
import atexit
import base64 as _b64  # imported only so the IOC string-scan can detect it
import hashlib
import io
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
# Caps calibrated against the real Studio frontend transitive closure:
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

# Credentials a frontend package should NEVER need to read. Bare
# substring match is too noisy (object-treeify ships a `docker` dev
# script that mounts ~/.npmrc -- legitimate dev tooling, never run
# at install time). We instead surface these only when they appear
# inside a LIFECYCLE script (preinstall / install / postinstall /
# prepare), which is the only path that runs automatically on
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

# `process.env.GITHUB_TOKEN` / `NPM_TOKEN` / `AWS_*` access in
# top-level / install-time code is suspicious. We also catch
# `os.environ["GITHUB_TOKEN"]` for the rare Python-in-npm postinstall.
_JS_ENV_TOKEN = re.compile(
    r"""(process\.env\.|os\.environ\[?['"])(?:
        GITHUB_TOKEN | GH_TOKEN | NPM_TOKEN | NODE_AUTH_TOKEN
        | AWS_ACCESS_KEY_ID | AWS_SECRET_ACCESS_KEY | AWS_SESSION_TOKEN
        | GOOGLE_APPLICATION_CREDENTIALS
        | DOCKER_AUTH_CONFIG | VAULT_TOKEN
    )['"]?\]?""",
    re.VERBOSE,
)

# Suspicious lifecycle-script payloads. Anything in a package.json
# `scripts` field that wgets/curls an external resource and executes
# it. We do NOT block ALL curl/wget in scripts (some legit packages
# fetch test fixtures into devDependencies), but we DO block the
# fetch+exec chain.
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

# Obfuscation: large JS file that is mostly one line of base64-ish
# blob with a Function() / eval() bookend. Tuned against the
# router_init.js shape (2.3 MB obfuscated single-blob).
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

    Structural findings here are HIGH-severity refusals that should
    short-circuit the scan -- a lockfile with non-registry resolved
    URLs is itself a finding (covered by scripts/lockfile_supply_chain
    _audit.py in detail; we surface a summary here so this scanner is
    standalone-runnable).
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
        # Strict registry origin check. lockfile_supply_chain_audit
        # already catches this; double-defend here so this scanner
        # cannot be tricked into fetching from an attacker-chosen URL.
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
    """Stream-download entry.resolved to dest. Verify SRI integrity.

    Returns (downloaded_path, error_or_none). On any error the
    returned path may not exist. Network access is restricted to
    https://{ALLOWED_DOWNLOAD_HOST}/ -- the caller passes a Request
    we already validated.
    """
    # Re-assert hostname; the entry was validated at parse time but a
    # defence-in-depth check here means a future refactor cannot
    # accidentally bypass it.
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
                            f"download exceeded cap {max_bytes} bytes "
                            f"after {written} bytes"
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
        # Open in streaming mode so we never seek backwards in the
        # input. `r|gz` rejects malformed gzip frames immediately.
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
                # Cumulative cap is checked against DECLARED size up
                # front to short-circuit obvious bombs without reading
                # the body.
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
                # Strip leading "package/" -- the npm convention. We do
                # NOT trust npm to be right, so we explicitly resolve
                # the destination and refuse anything that escapes.
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
                # Sniff first 16 bytes to classify text vs binary.
                # Text-cap members get the tight 16 MiB limit; binary
                # members (executables, .node, .wasm, native libs)
                # get the generous binary cap. We bound BOTH cases.
                header = src.read(16)
                is_binary = _looks_binary(name, header)
                file_cap = (
                    HARD_MAX_BINARY_FILE_BYTES
                    if is_binary
                    else HARD_MAX_TEXT_FILE_BYTES
                )
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
                # Write with restrictive mode (rw-r--r--) so even if
                # someone runs the extract dir nothing is executable.
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


def _evidence(text: str, pat: re.Pattern, max_chars: int = 200) -> str:
    m = pat.search(text)
    if not m:
        return ""
    start = max(0, m.start() - 30)
    end = min(len(text), m.end() + 30)
    snippet = text[start:end].replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return snippet


LIFECYCLE_HOOKS = ("preinstall", "install", "postinstall", "prepare")


def scan_package_json(
    pkg: PackageEntry,
    rel: str,
    text: str,
) -> list[Finding]:
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
        if _LIFECYCLE_FETCH_EXEC.search(body):
            findings.append(
                Finding(
                    severity = CRITICAL,
                    package = pkg.display,
                    filename = rel,
                    pattern = f"lifecycle-fetch-exec ({hook})",
                    evidence = body,
                    detail = (
                        f"`scripts.{hook}` fetches an external "
                        "resource and pipes/chains it to an "
                        "interpreter; this is the install-time RCE "
                        "vector. Refusing to install."
                    ),
                )
            )
        # Credential file paths inside a lifecycle script are
        # exfiltration prep -- npm runs these scripts automatically
        # on `npm ci`. Manual `scripts.*` entries (like a `docker`
        # dev script) are out of scope: npm does not run them.
        for path_substr, why in CRED_PATH_SUBSTRINGS:
            if path_substr in body:
                findings.append(
                    Finding(
                        severity = HIGH,
                        package = pkg.display,
                        filename = rel,
                        pattern = f"cred-path-in-lifecycle ({hook})",
                        evidence = body,
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
                    evidence = _evidence(body, _JS_ENV_TOKEN),
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
                v.startswith("github:")
                or v.startswith("git+")
                or v.startswith("git://")
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
    """True if `host` appears in a way consistent with an outbound call.

    A bare `"169.254.169.254"` array literal (defensive blocklist) is
    safe; a `fetch("http://169.254.169.254/...")` is not. The signal
    is co-occurrence with either an HTTP URL scheme or a fetch verb
    within a short window.

    A defensive blocklist looks like:
        const CLOUD_METADATA_IPS = ["169.254.169.254", "169.254.170.2"];
    An exfil call looks like:
        fetch("http://169.254.169.254/latest/meta-data/...")
        http.request({ host: "169.254.169.254", path: "/..." })
    """
    # Esc for use in a regex (IPs contain dots).
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


def scan_text_blob(
    pkg: PackageEntry,
    rel: str,
    text: str,
) -> list[Finding]:
    findings: list[Finding] = []

    # IOC substrings (literal, case-sensitive).
    for needle, (sev, why) in KNOWN_IOC_STRINGS.items():
        if needle in text:
            findings.append(
                Finding(
                    severity = sev,
                    package = pkg.display,
                    filename = rel,
                    pattern = "known-ioc-string",
                    evidence = needle,
                    detail = f"{why}: {needle!r}",
                )
            )

    # Credential surfaces. Tier 1: hosts with no legitimate use,
    # bare substring is enough.
    for needle, why in CRED_HOST_ALWAYS_BAD:
        if needle in text:
            findings.append(
                Finding(
                    severity = HIGH,
                    package = pkg.display,
                    filename = rel,
                    pattern = "cred-surface-host (always-bad)",
                    evidence = needle,
                    detail = (
                        f"references {why} ({needle!r}); no legitimate "
                        "frontend use of this surface"
                    ),
                )
            )

    # Credential surfaces. Tier 2: hosts that do appear in defensive
    # code; require co-occurrence with a fetch verb or URL prefix.
    for needle, why in CRED_HOST_NEEDS_CONTEXT:
        if needle in text and _host_in_outbound_context(text, needle):
            findings.append(
                Finding(
                    severity = HIGH,
                    package = pkg.display,
                    filename = rel,
                    pattern = "cred-surface-host (outbound)",
                    evidence = needle,
                    detail = (
                        f"references {why} ({needle!r}) in an outbound "
                        "call / URL / host config; a defensive blocklist "
                        "literal would not match this rule"
                    ),
                )
            )

    # Credential PATHS are deliberately not scanned here; they have
    # too high a false-positive rate at file scope (defensive code,
    # docker mounts, AWS SDK docs strings). `scan_package_json`
    # catches the malicious case -- credential paths inside a
    # lifecycle script run automatically on `npm ci`.

    # JS-specific regex.
    if _JS_FETCH_EVAL.search(text):
        findings.append(
            Finding(
                severity = HIGH,
                package = pkg.display,
                filename = rel,
                pattern = "js-fetch-eval",
                evidence = _evidence(text, _JS_FETCH_EVAL),
                detail = (
                    "Function/eval against base64-decoded payload "
                    "(obfuscated dropper shape)"
                ),
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


# Filename suffix decides which scanners run. We deliberately treat
# *.cjs/*.mjs/*.ts the same as *.js -- attackers use whichever
# extension the consumer's bundler / loader resolves.
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


def scan_extracted_tree(
    pkg: PackageEntry,
    root: Path,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        lower = rel.lower()
        if not lower.endswith(_TEXT_SUFFIXES):
            # Skip native binaries entirely -- regex over compiled
            # machine code is just noise (false positives in WASM
            # opcodes, .node BSS segments, image pixel data). Use
            # content-magic detection so extensionless executables
            # (eg `package/biome`) and versioned shared libraries
            # are also skipped.
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
                            evidence = needle,
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


def scan_one(
    pkg: PackageEntry,
    workspace: Path,
) -> tuple[list[Finding], str | None]:
    """Download + extract + scan a single package. Cleans up its dir.

    Returns (findings, error). `error` is non-None only on hard
    failures (download error, integrity mismatch, malformed tarball);
    on a clean run with findings the error is None and the caller
    decides exit code based on severity.
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
    blocking = [f for f in all_findings if _SEVERITY_RANK[f.severity] <= threshold_rank]
    if hard_errors or blocking:
        if blocking:
            print(
                f"\n[scan-npm] FAIL: {len(blocking)} finding(s) "
                f"at or above {threshold}",
                file = sys.stderr,
            )
        return 1
    print("\n[scan-npm] OK", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
