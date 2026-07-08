#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# .github/workflows/security-audit.yml's pip-scan-packages job depends
# on this file existing at scripts/scan_packages.py.
"""
scan_packages.py -- Standalone pre-install package scanner.

Downloads PyPI packages WITHOUT installing them and inspects archive
contents for malicious patterns: weaponized .pth files, credential
stealers, obfuscated payloads, install-time droppers.

Motivated by the litellm 1.82.7/1.82.8 supply chain attack (March 2026).
Single file, stdlib only, Python 3.10+.

Examples:
    # Scan specific packages
    python scan_packages.py requests==2.32.5
    python scan_packages.py fastapi uvicorn pydantic

    # Scan requirements files
    python scan_packages.py -r requirements.txt
    python scan_packages.py -r base.txt -r extras.txt

    # Auto-discover requirements files in a project
    python scan_packages.py -d ./my-project/

    # Scan with full transitive dependency tree
    python scan_packages.py --with-deps unsloth unsloth-zoo

    # Scan + auto-fix CRITICAL findings in requirements files
    python scan_packages.py --fix -r requirements.txt
    python scan_packages.py --fix --max-search 20 -r requirements.txt

    # Triage to a baseline once, then gate on anything NEW
    python scan_packages.py -r requirements.txt --write-baseline scripts/scan_packages_baseline.json
    python scan_packages.py -r requirements.txt   # auto-loads the baseline, exits 0 if only baselined findings remain

False positives:
    .py files are scanned code-only: comments and bare docstrings/doctests are
    blanked before pattern matching (line numbers preserved), so prose, usage
    examples and `>>>` doctests cannot trip a finding. Residual findings that
    are genuine library behavior (a HTTP client reading HF_TOKEN, a vendored
    test fixture) are suppressed via a reviewed baseline allowlist, matched on
    (package, package-relative file, check, evidence hash). A new check, or
    changed flagged code under the same check, reopens the finding; version
    bumps and line shifts do not. This mirrors the Hugging Face Hub approach
    (ClamAV/picklescan: low-FP, signature/structural, surface status).

Exit codes:
    0 -- no non-baselined CRITICAL or HIGH findings (or --write-baseline)
    1 -- non-baselined CRITICAL or HIGH findings detected
    2 -- no packages specified, or scan incomplete (pip download failure)
"""

import argparse
import atexit
import bisect
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tokenize
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


# Severity
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"

SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2}

# Hard pin-blocks for confirmed malicious PyPI versions (Socket.dev 2026-05-12
# Mini Shai-Hulud wave; earlier Semgrep/Endor reports for `lightning`).
BLOCKED_PYPI_VERSIONS: dict[str, set[str]] = {
    "guardrails-ai": {"0.10.1"},
    "mistralai": {"2.4.6"},
    "lightning": {"2.6.2", "2.6.3"},
}

# Pattern definitions

# Subprocess / OS exec patterns
RE_SUBPROCESS = re.compile(
    r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
    r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b",
)

# Encoding / obfuscation
RE_BASE64 = re.compile(
    r"\bbase64\s*\.\s*(b64decode|decodebytes|b32decode|b16decode)\b|\bcodecs\s*\.\s*decode\b",
)

# exec / eval
RE_EXEC_EVAL = re.compile(r"\b(exec|eval)\s*\(")

# Network APIs (excludes urllib.parse which is pure string manipulation)
RE_NETWORK = re.compile(
    r"\burllib\.request\b"
    r"|\burlopen\s*\("
    r"|\brequests\s*\.\s*(get|post|put|patch|delete|head|Session)\b"
    r"|\bhttpx\s*\.\s*(get|post|put|patch|delete|Client|AsyncClient)\b"
    r"|\bsocket\s*\.\s*(socket|create_connection)\b"
    r"|\bhttp\.client\b"
    r"|\bhttp\.server\b",
)

# Large base64 blob (>200 chars of contiguous base64 alphabet)
RE_LARGE_BLOB = re.compile(r"[A-Za-z0-9+/=]{200,}")

# Credential path access (requires file-access context, not just string mentions)
RE_CRED_ACCESS = re.compile(
    r"(?:open|Path|read_text|read_bytes)\s*\([^)]*?"
    r"(?:\.ssh[/\\]|\.aws[/\\]|\.kube[/\\]|\.gnupg[/\\]|\.docker[/\\]"
    r"|\.azure[/\\]|\.gcp[/\\]"
    r"|credentials\.json|\.git-credentials|\.npmrc|\.pypirc|wallet\.dat"
    r"|/etc/shadow|/etc/passwd"
    r"|id_rsa|id_ed25519|id_ecdsa"
    r"|kubeconfig|service-account-token)"
    r"|os\.path\.(?:join|expanduser)\([^)]*?"
    r"(?:\.ssh|\.aws|\.kube|\.gnupg|\.docker|\.azure|\.gcp|credentials)"
    r"|(?:open|Path)\(\s*['\"]\.env['\"]\s*[,)]",
    re.DOTALL,
)

# Chained / advanced obfuscation (marshal, compile, zlib, nested decode)
RE_OBFUSCATION = re.compile(
    r"\bmarshal\s*\.\s*(loads|load)\b"
    r"|\bcompile\s*\([^)]*['\"]exec['\"]\s*\)"
    r"|\bzlib\s*\.\s*decompress\b"
    r"|\blzma\s*\.\s*decompress\b"
    r"|\bbz2\s*\.\s*decompress\b"
    r"|\bbytearray\s*\(\s*\[.*?\]\s*\)"  # bytearray([104,101,...])
    r"|\bchr\s*\(\s*\d+\s*\).*chr\s*\(\s*\d+\s*\)"  # chr() obfuscation chains
    r"|\b__import__\s*\("  # dynamic import
    r"|\bgetattr\s*\(\s*__builtins__"  # getattr(__builtins__, ...)
    r"|\brotate\s*=.*\blambda\b.*\bchr\b"  # rotation ciphers
    r"|\b(?:b64decode|decodebytes)\s*\(.*(?:b64decode|decodebytes)\s*\(",  # double base64
    re.DOTALL,
)

# Embedded cryptographic keys (PEM-encoded)
RE_EMBEDDED_KEYS = re.compile(
    r"-----BEGIN\s+(?:RSA\s+)?(?:PUBLIC|PRIVATE|ENCRYPTED|EC|DSA|OPENSSH)\s+KEY-----"
    r"|\bRSA\s+PUBLIC\s+KEY\b.*[A-Za-z0-9+/=]{64,}"
    r"|\bMII[A-Za-z0-9+/]{20,}",  # DER-encoded key prefix (base64)
    re.DOTALL,
)

# Full PEM block (BEGIN..END), used to pin a multiline key body in evidence.
RE_PEM_BLOCK = re.compile(r"-----BEGIN[^\n]*KEY-----.*?-----END[^\n]*KEY-----", re.DOTALL)

# Cloud metadata / IMDS endpoints
RE_CLOUD_METADATA = re.compile(
    r"169\.254\.169\.254"  # AWS/Azure/GCP IMDS
    r"|metadata\.google\.internal"  # GCP metadata
    r"|169\.254\.170\.2"  # AWS ECS task metadata
    r"|100\.100\.100\.200"  # Alibaba Cloud metadata
    r"|/latest/meta-data"  # AWS IMDS path
    r"|/metadata/instance"  # GCP metadata path
    r"|/metadata/identity"  # Azure managed identity
    r"|\bIMDSv[12]\b",
)

# Persistence mechanisms (systemd, cron, launchd, registry, startup dirs)
RE_PERSISTENCE = re.compile(
    r"/etc/systemd/"
    r"|systemctl\s+(enable|start|daemon-reload)"
    r"|\.service\b.*\[Service\]"  # systemd unit content
    r"|/etc/cron"
    r"|crontab\s"
    r"|/etc/init\.d/"
    r"|/Library/LaunchDaemons"
    r"|/Library/LaunchAgents"
    r"|~/\.config/autostart"
    r"|~/.local/share/systemd"
    r"|~/\.config/systemd/user/"  # user-level systemd
    r"|HKEY_LOCAL_MACHINE.*\\\\Run"  # Windows registry autorun
    r"|HKEY_CURRENT_USER.*\\\\Run"
    r"|\\\\Start Menu\\\\Programs\\\\Startup"
    r"|schtasks\s",  # Windows scheduled tasks
    re.IGNORECASE,
)

# Container / orchestration abuse
RE_CONTAINER_ABUSE = re.compile(
    r"/var/run/docker\.sock"
    r"|\bdocker\s+(run|exec|cp|build)\b"
    r"|\bkubectl\s+(apply|create|exec|run|cp)\b"
    r"|\bkubernetes\.client\b"
    r"|\bfrom_incluster_config\b"
    r"|\blist_namespaced_secret\b"
    r"|\bcreate_namespaced_pod\b"
    r"|\bcreate_namespaced_daemon_set\b"
    r"|\bcreate_namespaced_secret\b"
    r"|\bkube-system\b"
    r"|\bhostPID\s*:\s*true"
    r"|\bprivileged\s*:\s*true"
    r"|\bhostNetwork\s*:\s*true"
    r"|\bhostPath\b.*\bpath\s*:\s*/",  # k8s hostPath mounts
    re.IGNORECASE,
)

# Environment variable harvesting (bulk access or known secret vars)
RE_ENV_HARVEST = re.compile(
    r"\bos\.environ\s*\.\s*copy\s*\("  # full env copy
    r"|\bdict\s*\(\s*os\.environ\s*\)"
    r"|\bjson\.dumps\s*\(\s*(?:dict\s*\(\s*)?os\.environ"
    r"|\bfor\s+\w+\s*,\s*\w+\s+in\s+os\.environ\.items\(\)"  # iterating all env vars
    r"|\bos\.environ\b.*(?:SECRET|TOKEN|KEY|PASSWORD|CREDENTIAL|API_KEY|PRIVATE)"
    r"|\b(?:SECRET|TOKEN|PASSWORD|API_KEY|PRIVATE_KEY)\b.*os\.environ",
    re.IGNORECASE,
)

# Archive staging / exfiltration prep (create archive + network send)
RE_ARCHIVE_STAGING = re.compile(
    r"\btarfile\s*\.\s*open\s*\("
    r"|\bzipfile\s*\.\s*ZipFile\s*\([^)]*['\"]w['\"]\s*\)"
    r"|\bshutil\s*\.\s*make_archive\b"
    r"|\b\.add\s*\([^)]*(?:\.ssh|\.aws|\.env|\.kube|credentials|\.gnupg|\.docker)"
    r"|\b\.write\s*\([^)]*(?:\.ssh|\.aws|\.env|\.kube|credentials|\.gnupg|\.docker)",
    re.DOTALL,
)

# Anti-analysis / sandbox evasion / debugger detection
# NB: deliberately does NOT include a bare ``platform.system() ... Linux/Windows
# /Darwin`` branch. Under re.DOTALL that matched across the whole file -- any
# cross-platform library (typer, packaging, pandas, pymupdf, ...) trips it -- so
# it had ~zero precision and only generated false positives. OS detection alone
# is not an anti-analysis signal; the debugger/VM/long-sleep signals below are.
RE_ANTI_ANALYSIS = re.compile(
    r"\bptrace\b"
    r"|\bsys\s*\.\s*gettrace\s*\("
    r"|\bsys\s*\.\s*settrace\b"
    r"|\bTracerPid\b"
    # /proc/self/status is read to scrape TracerPid for anti-debug. A leading
    # \b here is unsatisfiable (\b never holds between a non-word boundary and
    # "/"), so the old pattern was dead; a lookbehind that only forbids a
    # preceding word char or path separator lets `open("/proc/self/status")`
    # and `cat /proc/self/status` match while avoiding mid-path partials.
    r"|(?<![\w/])/proc/self/status\b"
    r"|\bIsDebuggerPresent\b"
    r"|\bvirtualbox\b.*\bhardware\b"
    r"|\bvmware\b.*\bdetect\b"
    r"|\btime\.sleep\s*\(\s*(?:[3-9]\d{2,}|[1-9]\d{3,})\s*\)",  # long sleep (anti-sandbox)
    re.IGNORECASE | re.DOTALL,
)

# DNS exfiltration / tunneling
RE_DNS_EXFIL = re.compile(
    r"\bdns\.resolver\b"
    r"|\bsocket\.getaddrinfo\s*\([^)]*\+[^)]*\)"  # dynamic hostname construction
    r"|\bdnspython\b"
    r"|\bTXT\b.*\bresolver\b"
    r"|\bresolver\b.*\bTXT\b"
    r"|\bnslookup\b"
    r"|\bdig\s+",
)

# File system enumeration / bulk file theft
RE_FS_ENUM = re.compile(
    r"\bos\.walk\s*\(\s*['\"](?:/|~|/home|/root|/Users|C:\\\\)"
    r"|\bglob\s*\.\s*glob\s*\([^)]*(?:\*\*|\*\.pem|\*\.key|\*\.cer|\*\.pfx|\*\.p12)"
    r"|\bos\.listdir\s*\(\s*['\"](?:/home|/root|/Users|/etc)"
    r"|\bPath\s*\(\s*['\"]~['\"]\s*\)\s*\.\s*glob\b"
    r"|\bhistory\b.*\bread\b"  # reading shell history
    r"|\b\.bash_history\b"
    r"|\b\.zsh_history\b"
    r"|/etc/shadow"
    r"|/etc/passwd",
    re.DOTALL,
)

# Reverse shell / bind shell patterns
RE_REVERSE_SHELL = re.compile(
    r"\bsocket\b.*\bconnect\b.*\bsubprocess\b"
    r"|\bsocket\b.*\bconnect\b.*\b(?:sh|bash|cmd)\b"
    r"|\b/bin/(?:sh|bash)\b.*\bsocket\b"
    r"|\bpty\s*\.\s*spawn\b"
    r"|\bos\s*\.\s*dup2\s*\("
    r"|\bwebbrowser\s*\.\s*open\b.*\bdata:\b",  # data: URI abuse
    re.DOTALL,
)

# Process injection / code loading from remote
RE_REMOTE_CODE = re.compile(
    r"\bexec\s*\(\s*(?:urllib|requests|httpx|urlopen)"  # exec(requests.get(...))
    r"|\bexec\s*\([^)]*\.(?:text|content|read)\s*\("
    r"|\beval\s*\([^)]*\.(?:text|content|read)\s*\("
    r"|\bimportlib\s*\.\s*import_module\s*\([^)]*\+"  # dynamic import with concatenation
    r"|\b__import__\s*\([^)]*\+",  # __import__ with concatenation
    re.DOTALL,
)

# Crypto wallet / cryptocurrency theft
RE_CRYPTO_THEFT = re.compile(
    r"\bwallet\.dat\b"
    r"|\b\.bitcoin[/\\]"
    r"|\b\.ethereum[/\\]"
    r"|\b\.solana[/\\]"
    r"|\b\.monero[/\\]"
    r"|\b\.litecoin[/\\]"
    r"|\b\.config/solana[/\\]"
    r"|\bkeystore[/\\]UTC--"
    r"|\bseed\s*phrase\b"
    r"|\bmnemonic\b.*\b(?:word|phrase|recover|restore)\b"
    r"|\b(?:xprv|xpub|bc1|0x[a-fA-F0-9]{40})\b",
    re.IGNORECASE,
)

# Import line in .pth (Python site.py only exec()s lines starting with "import")
RE_PTH_IMPORT = re.compile(r"^\s*import\s+", re.MULTILINE)

# openssl CLI invocations via subprocess (encrypted exfiltration)
RE_OPENSSL_CLI = re.compile(r"\bopenssl\s+(enc|rand|rsautl|pkeyutl|genrsa|dgst|s_client)\b")

# Write to /tmp then execute (staged dropper)
RE_TEMP_EXEC = re.compile(
    r"/tmp/\S+.*(?:subprocess|os\.system|os\.popen|Popen|chmod.*\+x)",
    re.DOTALL,
)

# C2 polling / beaconing loop
RE_C2_POLLING = re.compile(
    r"while\s+True.*(?:time\.sleep|sleep)\s*\(.*(?:urlopen|requests\.|httpx\.)",
    re.DOTALL,
)

# Developer-tool persistence hooks. Lightning 2.6.x planted SessionStart hooks
# into Claude Code / VS Code / Cursor so the payload re-attached on editor open.
RE_DEV_TOOL_HIJACK = re.compile(
    r"\.claude/settings\.json"
    r"|\.cursor/.*hooks"
    r"|\.vscode/(?:tasks|settings|launch)\.json"
    r"|SessionStart|folderOpen|onCommand:.*runTask"
    r"|/etc/profile\.d/"
    r"|\b\.bashrc\b|\b\.zshrc\b|\b\.profile\b"
    r"|\bautomator\b.*\.workflow\b",
)

# Hard-coded credential / API-token regexes embedded in source. Packages that
# ship regexes for OTHER people's secrets are nearly always stealers.
RE_TOKEN_REGEX = re.compile(
    r"\bgh[psoru]_[A-Za-z0-9_]{20,}"  # GitHub PAT/OAuth/etc.
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\bnpm_[A-Za-z0-9]{30,}"  # npm token
    r"|\bsk-[A-Za-z0-9]{20,}"  # OpenAI / Anthropic
    r"|\bxox[bpaesr]-"  # Slack
    r"|\bAIza[0-9A-Za-z_-]{20,}"  # Google API key
    r"|\bAKIA[0-9A-Z]{16}"  # AWS access key id
    r"|\bASIA[0-9A-Z]{16}"  # AWS STS
    r"|\bgithub.com/login/oauth/access_token"
    r"|\bglpat-[0-9A-Za-z_-]{20,}",  # GitLab PAT
)

# Mini Shai-Hulud May-12 2026 wave indicators. `transformers.pyz` dropper name
# is high-confidence; the host + slogans are CRITICAL.
RE_MAY12_IOC = re.compile(
    r"(git-tanstack\.com|/tmp/transformers\.pyz|transformers\.pyz"
    r"|With Love TeamPCP|We've been online over 2 hours)",
    re.IGNORECASE,
)

# JavaScript-side obfuscation. A bundle full of `_0x1f2e3d` hex-var identifiers
# is a near-universal tell for a malicious npm payload, rare in legit wheels.
RE_JS_OBFUSCATION = re.compile(
    r"_0x[a-f0-9]{4,6}\s*=\s*function"
    r"|var\s+_0x[a-f0-9]{4,6}\b"
    r"|(?:\\x[0-9a-f]{2}){10,}"  # \x-escape strings
    r"|String\.fromCharCode\s*\(\s*\d+\s*(?:,\s*\d+\s*){10,}\)",
)

# Web3 / wallet-hijack pattern. The Qix npm phish overrode fetch/XMLHttpRequest
# and swapped recipient addresses via a `window.ethereum` listener.
RE_WEB3_HIJACK = re.compile(
    r"\bwindow\.ethereum\b"
    r"|\bweb3\.eth\.\w+\s*\("
    r"|XMLHttpRequest\.prototype\.(?:open|send)\s*="
    r"|(?:^|\s)fetch\s*=\s*\(?\s*async"
    r"|TronWeb|solanaWeb3",
)

# Self-propagating worms (Shai-Hulud, ForceMemo) plant their own GitHub workflow
# in every repo they reach and use trufflehog/gitleaks for credential discovery.
# Any of these strings in a package payload is strong repo-takeover evidence.
RE_WORKFLOW_INJECT = re.compile(
    r"\.github/workflows/[^\"\']*\.ya?ml"
    r"|\btrufflehog\b|\bgitleaks\b"
    r"|/user/repos\?affiliation=.*owner.*collaborator"
    r"|\bshai-hulud\b|EveryBoiWeBuildIsAWormyBoi"
    r"|\bgit\s+push\s+--force\b.*--no-verify",
    re.IGNORECASE | re.DOTALL,
)

# install.sh / postinstall scripts piping remote code into a shell.
# `curl ... | sh` is the canonical npm postinstall dropper.
RE_SHELL_DROPPER = re.compile(
    r"\bcurl\b[^\n|]*\|\s*(?:sh|bash|zsh)\b"
    r"|\bwget\b[^\n|]*-O-\s*\|\s*(?:sh|bash|zsh)\b"
    r"|\bnpx\b\s+-y\s+[^\s]+@latest\s*\|"
    r"|\beval\s+\$\(\s*curl\b"
    r"|\bbash\s+<\(\s*curl\b",
)


@dataclass
class Finding:
    severity: str
    package: str
    filename: str
    check: str
    evidence: str = ""


# Checkers


def check_pth_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run all .pth-specific checks.

    Executable .pth files run on every Python startup, so any suspicious
    pattern in a .pth is treated as CRITICAL.
    """
    findings = []

    # Only .pth files with import lines are executable
    import_lines = [line for line in content.splitlines() if RE_PTH_IMPORT.match(line)]
    if not import_lines:
        return findings  # Pure path entries, inert

    # All patterns are CRITICAL inside executable .pth files
    _pth_checks = [
        (RE_SUBPROCESS, ".pth has subprocess/os exec calls"),
        (RE_BASE64, ".pth has base64/encoding obfuscation"),
        (RE_EXEC_EVAL, ".pth has exec()/eval()"),
        (RE_NETWORK, ".pth has network API calls"),
        (
            RE_OBFUSCATION,
            ".pth has advanced obfuscation (marshal/compile/zlib/__import__)",
        ),
        (RE_EMBEDDED_KEYS, ".pth has embedded cryptographic key material"),
        (RE_CLOUD_METADATA, ".pth accesses cloud metadata / IMDS endpoints"),
        (RE_PERSISTENCE, ".pth installs persistence (systemd/cron/launchd/registry)"),
        (RE_CONTAINER_ABUSE, ".pth interacts with container/orchestration runtime"),
        (RE_ENV_HARVEST, ".pth harvests environment variables / secrets"),
        (RE_ARCHIVE_STAGING, ".pth stages archive for exfiltration"),
        (RE_ANTI_ANALYSIS, ".pth has anti-analysis / sandbox evasion"),
        (RE_DNS_EXFIL, ".pth has DNS exfiltration / tunneling patterns"),
        (RE_FS_ENUM, ".pth enumerates filesystem / steals files"),
        (RE_REVERSE_SHELL, ".pth has reverse/bind shell patterns"),
        (RE_REMOTE_CODE, ".pth loads and executes remote code"),
        (RE_CRYPTO_THEFT, ".pth targets cryptocurrency wallets / keys"),
        (RE_CRED_ACCESS, ".pth accesses credential files"),
        (RE_OPENSSL_CLI, ".pth invokes openssl CLI (encrypted exfil pattern)"),
        (RE_TEMP_EXEC, ".pth writes to /tmp and executes (staged dropper)"),
        (RE_C2_POLLING, ".pth has C2 polling/beaconing loop"),
    ]

    for pattern, description in _pth_checks:
        if pattern.search(content):
            findings.append(
                Finding(
                    CRITICAL,
                    package,
                    filename,
                    description,
                    _extract_evidence(content, pattern),
                )
            )

    # Large base64 blob
    if RE_LARGE_BLOB.search(content):
        # Digest every blob (not just the first 120 chars, and not just the
        # first blob), so a later payload that keeps the prefix or appends a
        # second encoded blob reopens.
        blob, digest = _blob_digest(content)
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                f".pth has large base64-like blob ({len(blob)} chars)",
                f"{blob[:120]}... sha256:{digest}",
            )
        )

    # Catch-all: any import line in .pth if nothing else triggered. Bind every
    # line through a digest so an appended/swapped import reopens the key, but cap
    # the displayed text so a large .pth of benign-looking imports cannot dump up
    # to the archive member cap into the logs or baseline JSON.
    if not findings and import_lines:
        evidence = _cap_line("\n".join(import_lines))
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                f".pth has {len(import_lines)} executable import line(s)",
                evidence,
            )
        )

    # Unusually large executable .pth (litellm's was 34 KB; legit ones are <100 bytes)
    size = len(content)
    if size > 500 and import_lines:
        # Pin the content so a different payload of the same size/import count reopens.
        digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                f"Unusually large executable .pth ({size} bytes)",
                f"{len(import_lines)} import line(s) in {size}-byte .pth file sha256:{digest}",
            )
        )

    return findings


# A STRING after one of these tokens (and before a NEWLINE) is a bare
# docstring/doctest/prose statement -- the dominant FP source -- so we blank it.
# A string after `=` or `(` is real code and is never blanked.
_LINE_START_TOKENS = frozenset({tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT})


def _is_fstring(tok_string: str) -> bool:
    """True if a STRING token is an f-string (3.10/3.11 emit one STRING token).

    A bare f-string statement evaluates its expressions at import, so unlike an
    inert docstring it must never be blanked.
    """
    q = min((tok_string.find(c) for c in "'\"" if c in tok_string), default = -1)
    return q > 0 and "f" in tok_string[:q].lower()


def _strip_noncode(content: str, blank_comments: bool = True) -> str:
    """Blank comments and bare docstrings so IOC patterns see code only.

    Removed regions become spaces (newlines kept) so line numbers stay exact for
    _extract_evidence. Fails open on tokenizer errors (the raw text is still
    fully scanned, so a real detection is never lost). ``blank_comments=False``
    keeps comments (only strings/docstrings blanked) to isolate the span that
    exec() could actually run.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(content).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return content

    spans: list[tuple[int, int, int, int]] = []  # (srow, scol, erow, ecol)
    prev_significant = tokenize.NEWLINE  # start-of-file behaves like a new line
    n = len(toks)
    for i, tok in enumerate(toks):
        ttype = tok.type
        if ttype == tokenize.COMMENT:
            if blank_comments:
                spans.append((*tok.start, *tok.end))
            continue  # transparent; never advances prev_significant
        if (
            ttype == tokenize.STRING
            and prev_significant in _LINE_START_TOKENS
            and not _is_fstring(tok.string)  # f-strings execute; never blank them
        ):
            # Bare string only if it is the whole statement: next significant
            # token must close the logical line.
            j = i + 1
            while j < n and toks[j].type in (tokenize.COMMENT, tokenize.NL):
                j += 1
            if j < n and toks[j].type == tokenize.NEWLINE:
                spans.append((*tok.start, *tok.end))
                prev_significant = ttype
                continue
        if ttype in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENCODING,
        ):
            prev_significant = ttype
            continue
        prev_significant = ttype

    if not spans:
        return content

    buf = content.splitlines(keepends = True)
    for srow, scol, erow, ecol in spans:
        for row in range(srow, erow + 1):
            line = buf[row - 1]
            if line.endswith("\n"):
                body, nl = line[:-1], "\n"
            elif line.endswith("\r"):
                body, nl = line[:-1], "\r"
            else:
                body, nl = line, ""
            start = scol if row == srow else 0
            end = ecol if row == erow else len(body)
            end = min(end, len(body))
            if start < end:
                body = body[:start] + (" " * (end - start)) + body[end:]
            buf[row - 1] = body + nl
    return "".join(buf)


# Payload carriers that are suspicious when hidden in a blanked region (a
# docstring/string) of a file that can dynamically execute strings.
_HIDDEN_PAYLOAD_PATTERNS = (
    (RE_LARGE_BLOB, "large base64 blob"),
    (RE_EMBEDDED_KEYS, "embedded key material"),
    (RE_MAY12_IOC, "Shai-Hulud IOC string"),
    (RE_OBFUSCATION, "marshal/compile/obfuscation"),
)


def _hidden_payload_findings(
    original: str, stripped: str, filename: str, package: str
) -> list[Finding]:
    """Flag payloads that live only in the blanked (docstring/string) region of
    a file that contains exec/eval. Such a string is invisible to code-only
    scanning yet ``exec(__doc__)`` / ``exec(<str>)`` could still run it."""
    if not RE_EXEC_EVAL.search(stripped):
        return []
    # Only docstrings/strings run via exec(__doc__)/exec(<str>); comments cannot.
    # Isolate that span: keep comments as real code, take what string-blanking
    # removed (length-preserved, so offsets stay exact for _extract_evidence).
    code = _strip_noncode(original, blank_comments = False)
    removed = "".join(o if o != s else " " for o, s in zip(original, code))
    out = []

    # The visible exec/eval line is what makes the hidden string executable, so
    # bind it into every finding's evidence: otherwise a reviewed false positive
    # that keeps the same hidden text but flips a harmless `eval("1+1")` to
    # `exec(__doc__)` (now running the payload) keeps the same key and stays
    # suppressed. Taken from `stripped` (real code), where the exec/eval lives.
    trigger = _extract_evidence(stripped, RE_EXEC_EVAL)

    def _hidden(pat):
        # Carrier present in a blanked region but NOT in real code. A carrier in
        # real code is already caught by the normal check, so restricting to
        # blanked-only avoids re-flagging legitimate in-code constants.
        return bool(pat.search(removed)) and not pat.search(stripped)

    for pat, label in _HIDDEN_PAYLOAD_PATTERNS:
        if _hidden(pat):
            out.append(
                Finding(
                    HIGH,
                    package,
                    filename,
                    "exec/eval with payload hidden in a docstring/string",
                    f"exec: {trigger}\n{label}: {_extract_evidence(removed, pat)}",
                )
            )
    # Fetch-then-run dropper: a network call AND an os/subprocess exec that both
    # live in the blanked region. Search the removed span directly (not "absent
    # from real code") so a benign visible network/subprocess call cannot mask
    # the docstring payload.
    if RE_NETWORK.search(removed) and RE_SUBPROCESS.search(removed):
        out.append(
            Finding(
                HIGH,
                package,
                filename,
                "exec/eval with hidden network+exec payload",
                f"exec: {trigger}\n"
                f"network+exec: {_extract_evidence(removed, RE_NETWORK)} | "
                f"{_extract_evidence(removed, RE_SUBPROCESS)}",
            )
        )
    return out


def check_py_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run all .py-specific checks."""
    # Code-only scanning: strip comments/docstrings up front so prose, doctests
    # and usage examples cannot manufacture false positives. Aligns with the
    # Hugging Face Hub model (ClamAV/picklescan: low-FP, signature/structural).
    original = content
    content = _strip_noncode(content)
    findings = _hidden_payload_findings(original, content, filename, package)
    basename = os.path.basename(filename)
    is_setup = basename in ("setup.py", "setup.cfg")
    is_init = basename == "__init__.py"

    # Pre-compute pattern matches
    has_network = bool(RE_NETWORK.search(content))
    has_subprocess = bool(RE_SUBPROCESS.search(content))
    has_base64 = bool(RE_BASE64.search(content))
    has_exec_eval = bool(RE_EXEC_EVAL.search(content))
    has_creds = bool(RE_CRED_ACCESS.search(content))
    has_blob = bool(RE_LARGE_BLOB.search(content))
    has_obfuscation = bool(RE_OBFUSCATION.search(content))
    has_keys = bool(RE_EMBEDDED_KEYS.search(content))
    has_cloud_meta = bool(RE_CLOUD_METADATA.search(content))
    has_persistence = bool(RE_PERSISTENCE.search(content))
    has_container = bool(RE_CONTAINER_ABUSE.search(content))
    has_env_harvest = bool(RE_ENV_HARVEST.search(content))
    has_archive = bool(RE_ARCHIVE_STAGING.search(content))
    has_anti = bool(RE_ANTI_ANALYSIS.search(content))
    has_dns_exfil = bool(RE_DNS_EXFIL.search(content))
    has_fs_enum = bool(RE_FS_ENUM.search(content))
    has_rev_shell = bool(RE_REVERSE_SHELL.search(content))
    has_remote_code = bool(RE_REMOTE_CODE.search(content))
    has_crypto_theft = bool(RE_CRYPTO_THEFT.search(content))
    has_openssl_cli = bool(RE_OPENSSL_CLI.search(content))
    has_temp_exec = bool(RE_TEMP_EXEC.search(content))
    has_c2_polling = bool(RE_C2_POLLING.search(content))
    has_may12_ioc = bool(RE_MAY12_IOC.search(content))

    # CRITICAL: combination patterns that strongly indicate malice

    # base64 decode + subprocess execution (staged payload)
    if has_base64 and has_subprocess:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "base64 decode + subprocess execution (staged payload)",
                f"Base64: {_extract_evidence(content, RE_BASE64)}\n"
                f"Subprocess: {_extract_evidence(content, RE_SUBPROCESS)}",
            )
        )

    # openssl encryption + network/key material (encrypted exfiltration)
    if has_openssl_cli and (has_network or has_keys):
        # Bind whichever side(s) co-occur so a changed endpoint or key reopens.
        evidence = [f"OpenSSL: {_extract_evidence(content, RE_OPENSSL_CLI)}"]
        if has_network:
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if has_keys:
            evidence.append(f"Key: {_embedded_key_evidence(content)}")
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "openssl encryption + network/key material (encrypted exfiltration)",
                "\n".join(evidence),
            )
        )

    # Writes to /tmp and executes (staged dropper)
    if has_temp_exec:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Writes to /tmp and executes (staged dropper)",
                _extract_evidence(content, RE_TEMP_EXEC),
            )
        )

    # May-12 Shai-Hulud IOC string in Python source.
    if has_may12_ioc:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "May-12 Shai-Hulud IOC string present in Python file",
                _extract_evidence(content, RE_MAY12_IOC),
            )
        )

    # C2 polling/beaconing loop
    if has_c2_polling:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "C2 polling/beaconing loop detected",
                _extract_evidence(content, RE_C2_POLLING),
            )
        )

    # Credential stealer: reads cred paths AND phones home
    if has_creds and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Reads credential paths AND makes network calls",
                f"Creds: {_extract_evidence(content, RE_CRED_ACCESS)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Reverse / bind shell
    if has_rev_shell:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Reverse shell / bind shell pattern",
                _extract_evidence(content, RE_REVERSE_SHELL),
            )
        )

    # Remote code execution: exec/eval on HTTP response
    if has_remote_code:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Downloads and executes remote code",
                _extract_evidence(content, RE_REMOTE_CODE),
            )
        )

    # Env harvest + network exfil
    if has_env_harvest and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Harvests environment variables/secrets AND makes network calls",
                f"Env: {_extract_evidence(content, RE_ENV_HARVEST)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Filesystem enum + network exfil
    if has_fs_enum and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Enumerates filesystem AND makes network calls",
                f"FS: {_extract_evidence(content, RE_FS_ENUM)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Cloud metadata access + network (exfil IMDS tokens)
    if has_cloud_meta and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Accesses cloud metadata/IMDS AND makes network calls",
                f"IMDS: {_extract_evidence(content, RE_CLOUD_METADATA)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Crypto wallet theft + network
    if has_crypto_theft and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Targets cryptocurrency wallets AND makes network calls",
                f"Crypto: {_extract_evidence(content, RE_CRYPTO_THEFT)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Archive staging with credential content + network
    if has_archive and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Creates archive with sensitive data AND makes network calls",
                f"Archive: {_extract_evidence(content, RE_ARCHIVE_STAGING)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Persistence + network (dropper that persists)
    if has_persistence and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Installs persistence AND makes network calls (backdoor pattern)",
                f"Persist: {_extract_evidence(content, RE_PERSISTENCE)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Container/k8s abuse + network
    if has_container and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Container/orchestration abuse AND makes network calls",
                f"Container: {_extract_evidence(content, RE_CONTAINER_ABUSE)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # HIGH: single strong signals or weaker combinations

    # Obfuscated payload: base64 + exec/eval + large blob
    if has_base64 and has_exec_eval and has_blob:
        # Digest every blob too: a payload may sit on a separate line from the
        # decode call, and a second encoded blob may be appended later, so
        # binding only the base64/exec lines or the first blob would miss it.
        _, blob_digest = _blob_digest(content)
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "base64 decode + exec/eval + large encoded blob",
                f"Base64: {_extract_evidence(content, RE_BASE64)}\n"
                f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}\n"
                f"Blob: sha256:{blob_digest}",
            )
        )

    # Advanced obfuscation + exec/eval
    if has_obfuscation and has_exec_eval:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Advanced obfuscation (marshal/compile/zlib) + exec/eval",
                f"Obfusc: {_extract_evidence(content, RE_OBFUSCATION)}\n"
                f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}",
            )
        )

    # Embedded crypto key + network (hardcoded key for encrypted exfil)
    if has_keys and has_network:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Embedded cryptographic key + network calls (encrypted exfil pattern)",
                f"Key: {_embedded_key_evidence(content)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Anti-analysis + any other suspicious pattern
    if has_anti and (has_network or has_subprocess or has_exec_eval):
        # Bind the suspicious side too so a changed payload reopens.
        evidence = [f"Anti: {_extract_evidence(content, RE_ANTI_ANALYSIS)}"]
        if has_network:
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if has_subprocess:
            evidence.append(f"Subprocess: {_extract_evidence(content, RE_SUBPROCESS)}")
        if has_exec_eval:
            evidence.append(f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}")
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Anti-analysis/sandbox evasion + suspicious behavior",
                "\n".join(evidence),
            )
        )

    # DNS exfiltration with dynamic hostnames
    if has_dns_exfil and (has_base64 or has_network or has_creds):
        # Bind the co-occurring side so a changed exfil channel reopens.
        evidence = [f"DNS: {_extract_evidence(content, RE_DNS_EXFIL)}"]
        if has_base64:
            evidence.append(f"Base64: {_extract_evidence(content, RE_BASE64)}")
        if has_network:
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if has_creds:
            evidence.append(f"Creds: {_extract_evidence(content, RE_CRED_ACCESS)}")
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "DNS exfiltration / tunneling patterns",
                "\n".join(evidence),
            )
        )

    # Cloud metadata standalone (IMDS access in a PyPI package is suspicious)
    if has_cloud_meta and not findings:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Accesses cloud metadata / IMDS endpoints",
                _extract_evidence(content, RE_CLOUD_METADATA),
            )
        )

    # Persistence standalone (a PyPI package installing systemd/cron is suspicious)
    if has_persistence and not has_network:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Installs persistence mechanism (systemd/cron/launchd/registry)",
                _extract_evidence(content, RE_PERSISTENCE),
            )
        )

    # Container abuse standalone
    if has_container and not has_network:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Interacts with container/orchestration runtime",
                _extract_evidence(content, RE_CONTAINER_ABUSE),
            )
        )

    # openssl CLI standalone (uncommon in PyPI packages)
    if has_openssl_cli and not (has_network or has_keys):
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Invokes openssl CLI (uncommon in PyPI packages)",
                _extract_evidence(content, RE_OPENSSL_CLI),
            )
        )

    # setup.py checks
    if is_setup:
        if has_network and has_subprocess:
            findings.append(
                Finding(
                    HIGH,
                    package,
                    filename,
                    "setup.py has network calls + subprocess (dropper pattern)",
                    f"Network: {_extract_evidence(content, RE_NETWORK)}\n"
                    f"Subprocess: {_extract_evidence(content, RE_SUBPROCESS)}",
                )
            )
        elif has_network:
            findings.append(
                Finding(
                    MEDIUM,
                    package,
                    filename,
                    "setup.py makes network calls at install time",
                    _extract_evidence(content, RE_NETWORK),
                )
            )

    # MEDIUM: standalone signals (informational, may be legitimate)

    # base64 + exec/eval without blob
    if has_base64 and has_exec_eval and not has_blob:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "base64 decode + exec/eval (no large blob)",
                f"Base64: {_extract_evidence(content, RE_BASE64)}\n"
                f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}",
            )
        )

    # Standalone obfuscation without exec
    if has_obfuscation and not has_exec_eval:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Advanced obfuscation patterns (marshal/compile/zlib/__import__)",
                _extract_evidence(content, RE_OBFUSCATION),
            )
        )

    # Embedded crypto keys standalone
    if has_keys and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Embedded cryptographic key material",
                _embedded_key_evidence(content),
            )
        )

    # Env harvest standalone
    if has_env_harvest and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Harvests environment variables / secrets",
                _extract_evidence(content, RE_ENV_HARVEST),
            )
        )

    # Filesystem enum standalone
    if has_fs_enum and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Enumerates filesystem / reads sensitive file paths",
                _extract_evidence(content, RE_FS_ENUM),
            )
        )

    # Crypto wallet references standalone
    if has_crypto_theft and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "References cryptocurrency wallets / keys",
                _extract_evidence(content, RE_CRYPTO_THEFT),
            )
        )

    return findings


_MAX_MULTILINE_LINES = 12
# How far a single matched call is followed over its bracket continuations. A call
# that genuinely closes is bound all the way to its real close, up to the hard
# limit, so a ``requests.post(`` with many option/header lines before ``data=``
# binds its whole argument list in the digest and a changed payload on a late
# continuation line reopens (a 40-line soft cap would hash only the first 40 lines
# and let a later ``data=``/headers change ride the baseline key). A bracket that
# never closes within the hard limit is a miscount (a multi-line string the
# single-line blanker cannot mask) or a stray opener, so it is bound only to the
# soft cap and cannot swallow unrelated code.
_MAX_CALL_LINES = 40  # soft cap: how far a NEVER-closing opener is followed
_MAX_CALL_HARD_LINES = 200  # hard cap: how far a closing call is followed to bind it

# Cap a single rendered line. A short line is shown verbatim; a long (e.g.
# minified one-liner) line is shown as a bounded prefix plus a sha256 of the full
# line, so a packed payload cannot dump unbounded content into the evidence and
# baseline while a change past the cutoff still changes the digest and reopens the
# finding. The npm scanner bounds its snippets the same way.
_MAX_LINE_CHARS = 200
# Cap on recorded spans in one evidence string; beyond it the remaining spans are
# folded into a digest so a file with thousands of matching lines cannot build a
# multi-megabyte evidence blob, while an added/removed span past the cap still
# changes the key. Comfortably above the largest real baseline entry.
_MAX_EVIDENCE_SPANS = 96


def _cap_line(code: str) -> str:
    """Bound a single line's displayed code: return it verbatim when short, else a
    ``_MAX_LINE_CHARS`` prefix plus a digest of the whole line so the tail is still
    pinned (fail-closed) without recording the entire line."""
    if len(code) <= _MAX_LINE_CHARS:
        return code
    digest = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
    return f"{code[:_MAX_LINE_CHARS]} sha256:{digest}"


_PY_TRIPLE = ("'''", '"""')


def _ends_with_odd_backslash(s: str) -> bool:
    """True if ``s`` ends with an odd run of backslashes, i.e. a trailing
    backslash that escapes the newline (a string/line continuation) rather than a
    literal ``\\\\`` pair."""
    return (len(s) - len(s.rstrip("\\"))) % 2 == 1


# Single-line quoted string literal; blanks complete one-line strings (the legacy
# view) so the single-line and multi-line blanked spans can be unioned below.
_RE_STR_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def _blank_code_strings(lines: list[str]) -> list[str]:
    """Replace string contents (single- and triple-quoted, escapes honoured) with
    spaces across ``lines``, keeping the line count and every bracket OUTSIDE a
    string intact. Bracket counting then never miscounts a ``)`` that lives inside
    a string -- including a triple-quoted string spanning several lines, which a
    per-line regex cannot blank."""
    out: list[str] = []
    in_triple: str | None = None  # active ''' or \"\"\" delimiter, or None
    in_string: str | None = None  # active ' or " continued via a trailing backslash
    for line in lines:
        buf: list[str] = []
        i, n = 0, len(line)
        while i < n:
            if in_triple is not None:
                end = line.find(in_triple, i)
                if end == -1:
                    buf.append(" " * (n - i))
                    i = n
                else:
                    buf.append(" " * (end - i + 3))
                    i = end + 3
                    in_triple = None
                continue
            if in_string is not None:
                # A single-/double-quoted string continued onto this line by a
                # backslash-escaped newline. Resume blanking until its closing quote;
                # if this line also ends on an odd trailing backslash the string
                # continues again, otherwise it closes (or is unterminated) here. A
                # per-line regex blanker cannot see this, so a `)` on the
                # continuation line would otherwise be counted as code and close the
                # call early -- dropping the URL/body lines that follow.
                j, closed = i, False
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == in_string:
                        j += 1
                        closed = True
                        break
                    j += 1
                buf.append(" " * (min(j, n) - i))
                if closed:
                    in_string = None
                    i = j
                else:
                    i = n
                    if not _ends_with_odd_backslash(line):
                        in_string = None  # unterminated without continuation; stop
                continue
            ch = line[i]
            if ch in "'\"":
                if line[i : i + 3] in _PY_TRIPLE:
                    delim = line[i : i + 3]
                    end = line.find(delim, i + 3)
                    if end == -1:  # opens a triple string that runs past this line
                        buf.append(" " * (n - i))
                        in_triple = delim
                        i = n
                    else:
                        buf.append(" " * (end - i + 3))
                        i = end + 3
                    continue
                j = i + 1  # single-line string; skip to its closing quote
                closed = False
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == ch:
                        j += 1
                        closed = True
                        break
                    j += 1
                buf.append(" " * (min(j, n) - i))
                if closed:
                    i = j
                else:
                    # Ran off the line without closing: an odd trailing backslash
                    # escapes the newline and continues the string onto the next
                    # line, so remember the quote; otherwise it is just unterminated.
                    i = n
                    if _ends_with_odd_backslash(line):
                        in_string = ch
                continue
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return out


_RE_BRACKETS = re.compile(r"[()\[\]{}]")
_OPENERS = frozenset("([{")


def _bracket_lr(line: str) -> tuple[int, int]:
    """Order-aware bracket reduction of one already-string-blanked line: ``(L, R)``
    where ``L`` is the count of closers with no opener earlier on the line (they
    need an opener to the LEFT / a prior line) and ``R`` is the count of openers
    with no closer later on the line (they need a closer to the RIGHT / a later
    line). A plain net count (opens minus closes) collapses order and so masks a
    trailing opener that follows leading closers on the same line, e.g.
    ``]; requests.post(`` nets to 0 and hides the ``(`` that opens the flagged
    call; tracking the running minimum keeps that opener visible so the call's
    argument lines still bind. Only bracket characters are walked (pulled out with
    one C-level regex pass) so a long minified line stays cheap."""
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


def _scan_line_end(view: list[str], start: int) -> int:
    """1-based line where the statement at ``start`` closes its brackets in
    ``view`` (one blanked view of the file). A call that closes is followed to its
    real close up to ``_MAX_CALL_HARD_LINES`` so its whole argument list binds; a
    bracket that never closes within that hard limit (a stray/miscounted opener) is
    bound only to the ``_MAX_CALL_LINES`` soft cap so it cannot swallow the file.
    Brackets are applied in order via ``_bracket_lr`` (leading closers clamp at 0)
    so a closer that precedes the opener on the same line does not cancel it."""
    depth = 0
    hard = min(len(view), start + _MAX_CALL_HARD_LINES - 1)
    for j in range(start, hard + 1):
        ln = view[j - 1]
        left, right = _bracket_lr(ln)
        depth = max(0, depth - left) + right
        if ln.rstrip().endswith("\\"):
            continue  # explicit backslash continuation: the call (e.g. its `(` and
            # URL/body) is on the next physical line, so do not close here
        if depth <= 0:
            return j
    # Never closed within the hard limit: bind only the soft cap so a stray opener
    # cannot bind a giant unrelated span.
    return min(len(view), start + _MAX_CALL_LINES - 1)


def _logical_line_end(sl_blanked: list[str], ml_blanked: list[str], start: int) -> int:
    """1-based line where the statement opened at ``start`` closes, so a multi-line
    call binds its argument lines (a changed URL/body on a continuation line
    reopens, not just the API line). Returns the LARGER of the spans found in the
    single-line-blanked view (legacy: a payload embedded inside a string still
    counts, so its brackets bind the call) and the multi-line-blanked view (a
    bracket inside a triple-quoted string argument no longer closes the call
    early). Taking the union never shrinks the bound span below either view, so
    neither blanking strategy can drop a continuation line a malicious change
    relies on."""
    return max(_scan_line_end(sl_blanked, start), _scan_line_end(ml_blanked, start))


def _extract_evidence(
    content: str,
    pattern: re.Pattern,
    max_matches: int = 0,
) -> str:
    """Pull matching lines as evidence snippets (``max_matches=0`` means all).

    Records every matching line in full, not a truncated sample, so an extra
    match (or extra code on a long line) appended to an already-flagged file
    changes the evidence and the baseline key instead of riding the first few.
    Leading whitespace is kept so a flagged line moved out of a guarded block
    reads as changed. Each single-line match is extended over bracket
    continuations so a multi-line call binds its argument lines too. Cross-line
    matches the per-line scan cannot see (DOTALL IOC regexes, or a multi-line
    construct appended under a check that already had a one-line match) are
    recorded afterwards, so an added multiline payload reopens the finding. A
    pathological greedy span is bounded to its head line plus a digest of the
    rest.
    """
    lines = content.splitlines()
    sl_blanked = [_RE_STR_LITERAL.sub("", ln) for ln in lines]
    ml_blanked = _blank_code_strings(lines)
    out = []
    seen: set[tuple[int, int]] = set()
    # Overflow is streamed, not buffered: once `out` holds _MAX_EVIDENCE_SPANS
    # rendered spans, every further span is folded straight into a running digest
    # instead of being materialized and sliced off at the end. On a minified or
    # padded file with hundreds of thousands of matching lines that keeps memory
    # and work bounded to the display cap rather than the match count, while the
    # digest still covers every overflow span so an over-cap payload change
    # reopens. The fold reproduces _canon_evidence(" | ".join(overflow)) exactly
    # (strip each span to its non-empty L<NN>-less code lines, join with "\n"), so
    # the digest is identical to buffering the whole list and canonicalizing once.
    overflow_count = 0
    overflow_hash = hashlib.sha256()
    overflow_started = False

    def _emit(rendered: str) -> None:
        nonlocal overflow_count, overflow_started
        if len(out) < _MAX_EVIDENCE_SPANS:
            out.append(rendered)
            return
        overflow_count += 1
        for piece in _RE_EVIDENCE_SPLIT.split(rendered):
            piece = _RE_EVIDENCE_PREFIX.sub("", piece, count = 1).rstrip()
            if not piece:
                continue
            if overflow_started:
                overflow_hash.update(b"\n")
            overflow_hash.update(piece.encode("utf-8", "replace"))
            overflow_started = True

    def _render(start: int, end: int) -> str:
        span = lines[start - 1 : end] or ["<multiline match>"]
        if len(span) > _MAX_MULTILINE_LINES:
            # Digest the code without the L<NN>: markers so a pure line shift of
            # the same span stays stable while a code change still reopens. The
            # head is truncated for display only; the span digest already binds
            # its full content, so no per-line digest is needed here.
            code = "\n".join(ln.rstrip() for ln in span)
            digest = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
            head = span[0].rstrip()
            if len(head) > _MAX_LINE_CHARS:
                head = head[:_MAX_LINE_CHARS] + "..."
            return f"L{start}: {head} sha256:{digest}"
        return "\n".join(f"L{start + i}: {_cap_line(ln.rstrip())}" for i, ln in enumerate(span))

    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            span = (i, _logical_line_end(sl_blanked, ml_blanked, i))
            if span in seen:
                continue
            # Only track spans while still filling the display list: past the cap
            # every span is folded into the overflow digest, so growing `seen` with
            # all of them would keep memory proportional to the match count (the
            # behavior this cap exists to bound) on a generated file with millions
            # of one-line matches. The per-line spans are unique by line number, so
            # dropping them from `seen` past the cap cannot cause a missed dedup
            # here; at worst the fallback re-folds an over-cap span into the same
            # digest, which stays deterministic and still reopens on a change.
            if len(out) < _MAX_EVIDENCE_SPANS:
                seen.add(span)
            _emit(_render(*span))
            if max_matches and len(out) >= max_matches:
                return " | ".join(out)

    # Precompute newline offsets once so mapping a match offset to its 1-based line
    # is O(log n) (bisect) rather than O(n) (content.count) per match; the latter
    # made this fallback quadratic on a minified file with thousands of matches.
    nl = [p for p, ch in enumerate(content) if ch == "\n"]
    for m in pattern.finditer(content):
        start = bisect.bisect_left(nl, m.start()) + 1
        end = bisect.bisect_left(nl, m.end()) + 1
        if end <= start or (start, end) in seen:
            continue  # single-line matches are already covered by the pass above
        # A giant greedy DOTALL span is bound by the full digest of its content
        # (via _render, which renders a >12-line span as a head line plus a sha256
        # of the whole span). Binding only the anchors leaves the bridged interior
        # unhashed, so an attacker could insert a new cross-line payload (a `/tmp`
        # line and a later `subprocess` line, sharing no single line so the
        # per-line pass never binds them) between unchanged outer anchors and keep
        # the same key. Digesting the interior reopens on any such change; a pure
        # line shift stays stable because the digest is over the markerless code.
        if len(out) < _MAX_EVIDENCE_SPANS:
            seen.add((start, end))
        _emit(_render(start, end))
        if max_matches and len(out) >= max_matches:
            break
    if overflow_count:
        # The overflow digest was accumulated from the canonicalized (L<NN>:-less)
        # spans as they were emitted, so a pure line shift above the overflow
        # region does not change it and reopen an otherwise-unchanged finding,
        # matching the per-span key's line-shift stability.
        out.append(f"(+{overflow_count} more) sha256:{overflow_hash.hexdigest()}")
    return " | ".join(out)


def _embedded_key_evidence(content: str) -> str:
    """Key evidence that also pins the full PEM block(s) via a digest, so a key
    body swapped under the same BEGIN marker reopens the finding (single-line and
    DER keys are already bound by their full matched line)."""
    ev = _extract_evidence(content, RE_EMBEDDED_KEYS)
    blocks = RE_PEM_BLOCK.findall(content)
    if blocks:
        digest = hashlib.sha256("\n".join(blocks).encode("utf-8", "replace")).hexdigest()
        ev = f"{ev} sha256:{digest}" if ev else f"sha256:{digest}"
    return ev


def _blob_digest(content: str) -> tuple[str, str]:
    """First large blob (for display) plus a digest binding EVERY large blob, so
    an appended or swapped encoded payload reopens the finding rather than riding
    an unchanged first blob. Assumes at least one blob is present (single-blob
    files keep the prior single-blob digest, so the baseline does not drift)."""
    blobs = RE_LARGE_BLOB.findall(content)
    digest = hashlib.sha256("\n".join(blobs).encode("utf-8", "replace")).hexdigest()
    return blobs[0], digest


# Non-Python checkers
# Recent PyPI compromises (Lightning 2.6.x, ForceMemo) carried the payload in a
# bundled .js / .sh / workflow yaml so the Python imports looked clean. These
# checkers scan those file types when they appear inside a wheel/sdist.


def check_js_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run JS-side checks. Triggered by .js / .mjs / .cjs / .ts."""
    findings = []

    # A >100 KB JS file inside a Python wheel is anomalous: CRITICAL combined
    # with any other JS heuristic, HIGH standalone.
    is_large = len(content) > 100 * 1024
    has_obf = bool(RE_JS_OBFUSCATION.search(content))
    has_web3 = bool(RE_WEB3_HIJACK.search(content))
    has_token_regex = bool(RE_TOKEN_REGEX.search(content))
    has_workflow_inj = bool(RE_WORKFLOW_INJECT.search(content))
    has_network = bool(RE_NETWORK.search(content))

    if has_obf:
        sev = CRITICAL if (is_large or has_web3 or has_token_regex) else HIGH
        findings.append(
            Finding(
                sev,
                package,
                filename,
                "JS minifier-style hex-var obfuscation (npm-payload signature)",
                _extract_evidence(content, RE_JS_OBFUSCATION),
            )
        )
    if has_web3:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "JS Web3 / wallet hijack (window.ethereum or fetch override)",
                _extract_evidence(content, RE_WEB3_HIJACK),
            )
        )
    if has_token_regex and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "JS embeds credential regexes AND makes network calls (stealer)",
                f"Token: {_extract_evidence(content, RE_TOKEN_REGEX)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )
    if has_workflow_inj:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "JS self-propagation: workflow injection / repo takeover signature",
                _extract_evidence(content, RE_WORKFLOW_INJECT),
            )
        )
    # Pin the whole file's content digest to EVERY JS finding (not just large
    # bundles). _extract_evidence blanks only Python string forms before counting
    # brackets, so a JS backtick template literal that contains `)` can close a
    # call's span early and omit the option/body lines that follow; binding the
    # full content means a change to those omitted lines still reopens instead of
    # riding the matched-line evidence. A large bundle with no other heuristic is a
    # standalone HIGH.
    if findings or is_large:
        digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        if findings:
            for f in findings:
                f.evidence = f"{f.evidence} bundle-sha256:{digest}"
        else:
            findings.append(
                Finding(
                    HIGH,
                    package,
                    filename,
                    # Size stays out of the check label (from main) so the baseline
                    # key does not drift when a benign bundle grows; the full-content
                    # digest below still binds the bytes so a payload swap reopens.
                    "Python wheel ships large JS bundle (uncommon; manually review)",
                    f"sha256: {digest}",
                )
            )
    return findings


def check_shell_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run shell-side checks. Triggered by .sh / .bash / install scripts."""
    findings = []
    if RE_SHELL_DROPPER.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell pipes remote code into an interpreter (curl|sh dropper)",
                _extract_evidence(content, RE_SHELL_DROPPER),
            )
        )
    if RE_DEV_TOOL_HIJACK.search(content) and (
        RE_NETWORK.search(content) or RE_SUBPROCESS.search(content)
    ):
        # Bind the hook AND the network/exec signal so a changed exfil reopens.
        evidence = [f"Hook: {_extract_evidence(content, RE_DEV_TOOL_HIJACK)}"]
        if RE_NETWORK.search(content):
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if RE_SUBPROCESS.search(content):
            evidence.append(f"Exec: {_extract_evidence(content, RE_SUBPROCESS)}")
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell installs developer-tool persistence hook (.bashrc / "
                "profile.d / vscode tasks) AND has network or exec",
                "\n".join(evidence),
            )
        )
    if RE_TOKEN_REGEX.search(content) and RE_NETWORK.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell embeds credential regexes AND makes network calls",
                f"Token: {_extract_evidence(content, RE_TOKEN_REGEX)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )
    if RE_WORKFLOW_INJECT.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell self-propagation: workflow injection / repo takeover signature",
                _extract_evidence(content, RE_WORKFLOW_INJECT),
            )
        )
    if RE_MAY12_IOC.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "May-12 Shai-Hulud IOC string present in shell script",
                _extract_evidence(content, RE_MAY12_IOC),
            )
        )
    return findings


def check_workflow_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run GitHub-Actions workflow checks. Triggered by .github/workflows/*.yml."""
    findings = []
    # A workflow file inside a PyPI package is suspicious (Shai-Hulud plants
    # `shai-hulud.yml` everywhere); injection-signature matches are CRITICAL.
    if RE_WORKFLOW_INJECT.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Workflow file inside PyPI package matches self-propagation signature",
                _extract_evidence(content, RE_WORKFLOW_INJECT),
            )
        )
    if RE_TOKEN_REGEX.search(content):
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Workflow file embeds credential regexes (token harvesting?)",
                _extract_evidence(content, RE_TOKEN_REGEX),
            )
        )
    if RE_SHELL_DROPPER.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Workflow pipes remote code into a shell (curl|sh dropper)",
                _extract_evidence(content, RE_SHELL_DROPPER),
            )
        )
    if RE_MAY12_IOC.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "May-12 Shai-Hulud IOC string present in workflow file",
                _extract_evidence(content, RE_MAY12_IOC),
            )
        )
    return findings


# Archive handling

# Tarbomb caps, mirrored from scripts/scan_npm_packages.py::safe_extract.
# Refuses zip/tar-of-death so a hostile archive cannot exhaust memory before
# scanning. Keep in sync with the npm side; duplicated to stay standalone.
HARD_MAX_FILE_BYTES = 64 * 1024 * 1024  # 64 MiB per member
HARD_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB cumulative
HARD_MAX_MEMBERS = 50_000  # entries per archive


def _refuse_unsafe_member_name(name: str) -> str | None:
    """Return a refusal reason for a member name, or None if safe.

    Mirrors `safe_extract`: no absolute paths, no `..` traversal. We never write
    to disk, so the name-shape check plus the in-memory size cap is sufficient.
    """
    if name.startswith("/") or ".." in Path(name).parts:
        return f"unsafe member name {name!r}"
    return None


def iter_archive_files(archive_path: str):
    """Yield (filename, text_content) for every file in a wheel/sdist.

    Streams members with per-member size + count caps so a tarbomb/zipbomb can't
    blow the memory budget. On cap breach, emits a `[WARN]` and short-circuits.
    """
    path = Path(archive_path)

    if path.suffix == ".whl" or path.suffix == ".zip":
        total = 0
        count = 0
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                count += 1
                if count > HARD_MAX_MEMBERS:
                    print(
                        f"  [WARN] {path.name}: refused; member count "
                        f"{count} exceeds cap {HARD_MAX_MEMBERS}",
                        file = sys.stderr,
                    )
                    return
                reason = _refuse_unsafe_member_name(info.filename)
                if reason is not None:
                    print(
                        f"  [WARN] {path.name}: refused member ({reason})",
                        file = sys.stderr,
                    )
                    continue
                # Declared (uncompressed) size cap
                if info.file_size > HARD_MAX_FILE_BYTES:
                    print(
                        f"  [WARN] {path.name}: skipped {info.filename!r} "
                        f"(declared {info.file_size} > cap {HARD_MAX_FILE_BYTES})",
                        file = sys.stderr,
                    )
                    continue
                if total + info.file_size > HARD_MAX_TOTAL_BYTES:
                    print(
                        f"  [WARN] {path.name}: cumulative bytes cap "
                        f"{HARD_MAX_TOTAL_BYTES} hit at {info.filename!r}",
                        file = sys.stderr,
                    )
                    return
                try:
                    data = zf.read(info.filename)
                    total += len(data)
                    text = data.decode("utf-8", errors = "replace")
                    yield info.filename, text
                except Exception:
                    continue

    elif path.name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        total = 0
        count = 0
        # Streaming open so we never read the whole archive into memory.
        with tarfile.open(path, mode = "r|*") as tf:
            for member in tf:
                count += 1
                if count > HARD_MAX_MEMBERS:
                    print(
                        f"  [WARN] {path.name}: refused; member count "
                        f"{count} exceeds cap {HARD_MAX_MEMBERS}",
                        file = sys.stderr,
                    )
                    return
                # Refuse symlinks/hardlinks/devices: tar parsers have
                # historically dereferenced them on extract.
                if member.issym() or member.islnk():
                    print(
                        f"  [WARN] {path.name}: refused link member " f"{member.name!r}",
                        file = sys.stderr,
                    )
                    continue
                if member.isdev() or member.isfifo():
                    print(
                        f"  [WARN] {path.name}: refused special member " f"{member.name!r}",
                        file = sys.stderr,
                    )
                    continue
                if not member.isfile():
                    continue
                reason = _refuse_unsafe_member_name(member.name)
                if reason is not None:
                    print(
                        f"  [WARN] {path.name}: refused member ({reason})",
                        file = sys.stderr,
                    )
                    continue
                declared = max(member.size, 0)
                if declared > HARD_MAX_FILE_BYTES:
                    print(
                        f"  [WARN] {path.name}: skipped {member.name!r} "
                        f"(declared {declared} > cap {HARD_MAX_FILE_BYTES})",
                        file = sys.stderr,
                    )
                    continue
                if total + declared > HARD_MAX_TOTAL_BYTES:
                    print(
                        f"  [WARN] {path.name}: cumulative bytes cap "
                        f"{HARD_MAX_TOTAL_BYTES} hit at {member.name!r}",
                        file = sys.stderr,
                    )
                    return
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    # Bound the read: a tar header may lie about size
                    data = f.read(HARD_MAX_FILE_BYTES + 1)
                    if len(data) > HARD_MAX_FILE_BYTES:
                        print(
                            f"  [WARN] {path.name}: body of "
                            f"{member.name!r} exceeded declared cap",
                            file = sys.stderr,
                        )
                        continue
                    total += len(data)
                    text = data.decode("utf-8", errors = "replace")
                    yield member.name, text
                except Exception:
                    continue
    else:
        print(f"  [WARN] Unknown archive format: {path.name}", file = sys.stderr)


def scan_archive(archive_path: str, package: str) -> list[Finding]:
    """Scan all files in an archive for malicious patterns.

    A corrupted archive container (truncated wheel, bad gzip header, etc.) emits
    a CRITICAL ``archive_corrupted`` finding rather than being silently skipped
    and reported as "0 findings" (silent-failure hardening SF1).
    """
    findings: list[Finding] = []
    try:
        for filename, content in iter_archive_files(archive_path):
            lower = filename.lower()
            if lower.endswith(".pth"):
                findings.extend(check_pth_file(content, filename, package))
            elif lower.endswith(".py"):
                findings.extend(check_py_file(content, filename, package))
            elif lower.endswith((".js", ".mjs", ".cjs", ".ts")):
                # Lightning 2.6.x hid its payload in a 14.8 MB router_runtime.js;
                # without this branch we'd only see the small Python loader.
                findings.extend(check_js_file(content, filename, package))
            elif lower.endswith((".sh", ".bash")):
                findings.extend(check_shell_file(content, filename, package))
            elif "/.github/workflows/" in lower and lower.endswith((".yml", ".yaml")):
                # Shai-Hulud/ForceMemo plant their own GHA workflow
                findings.extend(check_workflow_file(content, filename, package))
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as exc:
        # Archive cannot be opened / is structurally broken: either transport
        # corruption or a deliberate attempt to bypass error-swallowing scanners.
        findings.append(
            Finding(
                CRITICAL,
                package,
                os.path.basename(archive_path),
                "archive_corrupted",
                f"{type(exc).__name__}: {exc}"[:240],
            )
        )
    return findings


# Download packages


_RE_PYPI_SPEC_VERSION = re.compile(r"==\s*([A-Za-z0-9_.\-+!]+)")


def _check_blocked_pypi_versions(specs: list[str]) -> tuple[list[str], list[Finding]]:
    """Filter ``specs`` against ``BLOCKED_PYPI_VERSIONS``.

    Returns ``(safe_specs, findings)``. Each blocked spec emits a CRITICAL
    ``Finding`` and is dropped so the malicious tarball is never fetched. Specs
    without an ``==X.Y.Z`` pin pass through; the IOC regexes catch them later.
    """
    safe: list[str] = []
    findings: list[Finding] = []
    for spec in specs:
        name = _extract_pkg_name(spec).lower()
        blocked = BLOCKED_PYPI_VERSIONS.get(name, set())
        if not blocked:
            safe.append(spec)
            continue
        m = _RE_PYPI_SPEC_VERSION.search(spec)
        version = m.group(1) if m else None
        if version is not None and version in blocked:
            findings.append(
                Finding(
                    CRITICAL,
                    f"{name}=={version}",
                    "<spec>",
                    "blocked-known-malicious",
                    f"{name}=={version} is on the BLOCKED_PYPI_VERSIONS list",
                )
            )
            # Drop the spec; do not download
            continue
        safe.append(spec)
    return safe, findings


def _pip_download_env() -> dict[str, str]:
    """Return a scrubbed environment for invoking `pip download`.

    Strips every PIP_* override and forces the resolver at PyPI; PIP_CONFIG_FILE
    is /dev/null so a stray pip.conf extra-index-url cannot bypass the pin.
    """
    env = {**os.environ}
    # Drop any user override
    for key in [k for k in env if k.startswith("PIP_")]:
        env.pop(key, None)
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"
    env["PIP_EXTRA_INDEX_URL"] = ""
    env["PIP_CONFIG_FILE"] = "/dev/null"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


# Pip resolver flags shared by both download branches. CLI index-URL pin is
# belt + braces with the env scrub; `--only-binary :all:` avoids running setup.py.
_PIP_DOWNLOAD_PIN_FLAGS = [
    "--index-url",
    "https://pypi.org/simple",
    "--only-binary",
    ":all:",
]


# Strip characters that could escape `dest` via `os.path.join`, so a spec like
# `../../etc/foo==1.0` cannot land outside the temp tree.
_RE_PKG_NAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]")


# sdist fallback. `--only-binary :all:` never builds an sdist (no setup.py
# exec), but a wheel-less project then can't be fetched at all and one such
# package fails the whole --with-deps resolve (exit 2) -- a coverage hole. So on
# resolve failure we drop to per-spec and fetch any sdist-only package's raw
# tarball from the PyPI JSON API for scan_archive() to read statically: no pip,
# no build, same no-exec guarantee. Transport failures are still exit 2; only
# "no wheel" is downgraded to a direct fetch.

# How many levels of indirect-dep recovery to chase (a wheel dep whose own child
# is sdist-only, and so on). Bounded with dedup so recovery always terminates.
_MAX_DEP_FOLLOWUP_DEPTH = 2
_SDIST_DOWNLOAD_TIMEOUT = 180
# Never fetch an archive larger than we would be willing to scan (iter_archive_files cap).
_MAX_SDIST_BYTES = HARD_MAX_TOTAL_BYTES
# Direct sdist bytes only ever come from PyPI's own CDN; refuse anything else.
_TRUSTED_PYPI_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org", "pypi.python.org"})


def _spec_pin_version(spec: str) -> str | None:
    """Return the ``==X.Y.Z`` pin from a spec, or None if unpinned."""
    m = _RE_PYPI_SPEC_VERSION.search(spec)
    return m.group(1) if m else None


def _pypi_json(name: str, version: str | None = None) -> dict | None:
    """Fetch PyPI metadata JSON (read-only HTTPS GET, no exec); None on error.
    With ``version`` it fetches that release's document, whose ``requires_dist``
    is accurate for the pin (the project-level doc describes only the latest)."""
    url = "https://pypi.org/pypi/" + urllib.parse.quote(name, safe = "")
    if version:
        url += "/" + urllib.parse.quote(version, safe = "")
    url += "/json"
    try:
        req = urllib.request.Request(url, headers = {"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout = 30) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            data = resp.read(16 * 1024 * 1024)  # metadata is small; cap regardless
        return json.loads(data.decode("utf-8", errors = "replace"))
    except Exception:
        return None


def _release_files(meta: dict, version: str | None) -> list[dict]:
    """Files for a pinned version, else the latest release's. A pin that is
    absent or empty returns [] (never the latest) so a yanked/bad pin fails
    closed instead of a different artifact being scanned in its place."""
    if version is not None:
        return meta.get("releases", {}).get(version) or []
    return meta.get("urls", []) or []


def _release_has_wheel(meta: dict, version: str | None) -> bool:
    """True if the (pinned or latest) release publishes any bdist_wheel."""
    return any(f.get("packagetype") == "bdist_wheel" for f in _release_files(meta, version))


def _is_trusted_pypi_url(url: str) -> bool:
    """Only download sdist bytes from PyPI's own hosts, over HTTPS."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    return parsed.scheme == "https" and parsed.hostname in _TRUSTED_PYPI_HOSTS


_MARKER_ENV_VARS = (
    "sys_platform",
    "platform_system",
    "platform_machine",
    "platform_release",
    "platform_version",
    "platform_python_implementation",
    "os_name",
    "python_version",
    "python_full_version",
    "implementation_name",
    "implementation_version",
)


def _marker_holds_by_default(marker: str) -> bool:
    """Keep (scan) a dep unless its marker is purely ``extra``-gated. The scanner
    runs on one OS/Python but a package may be installed on another, so a marker
    that can be true on a different target (``sys_platform == 'win32'``,
    ``python_version == '3.13'``) is always kept; only a marker depending solely
    on ``extra`` and false with no extra requested is dropped. Conservative: on
    any uncertainty, keep (over-scan, never silently skip)."""
    m = marker.strip()
    if not m or "extra" not in m:
        return True  # no extra gate: installed by default on some target -> scan
    if any(v in m for v in _MARKER_ENV_VARS):
        return True  # also platform/python gated: true on some target -> scan
    # Pure extra marker: decide by evaluating with no extra requested.
    try:
        from packaging.markers import Marker, default_environment

        env = default_environment()
        env["extra"] = ""
        return bool(Marker(m).evaluate(env))
    except Exception:
        # packaging missing/unparseable: drop only a pure positive extra-equality.
        return re.fullmatch(r"\s*extra\s*==\s*['\"][^'\"]+['\"]\s*", m) is None


def _requires_dist_names(meta: dict) -> list[str]:
    """Transitive dep specs (name + version specifier) from metadata, to recover
    a sdist-only package's tree. The specifier is kept so a pinned malicious
    version is fetched, not latest. Drops deps whose marker cannot hold for a
    default install."""
    info = meta.get("info", {}) or {}
    reqs = info.get("requires_dist") or []
    specs: list[str] = []
    for r in reqs:
        if not isinstance(r, str):
            continue
        head = r
        if ";" in r:
            head, marker = r.split(";", 1)
            if not _marker_holds_by_default(marker):
                continue
        if not _RE_NAME.match(head.strip()):
            continue
        # "torch (>=1.10)" / "torch >=1.10" -> "torch>=1.10" (pip-friendly).
        specs.append(re.sub(r"\s+", "", head).replace("(", "").replace(")", ""))
    return specs


def _requires_dist_for(
    name: str,
    version: str | None,
    project_meta: dict,
    errors: list[str] | None = None,
) -> list[str]:
    """Declared deps for the pinned version, read from that release's metadata
    (its ``requires_dist`` can differ from latest). Unpinned uses the
    project-level (latest) document. A pinned version whose own metadata cannot
    be fetched returns [] (never latest's deps) and, when ``errors`` is given,
    records an incomplete-scan error so a partial tree is not read as "no deps"."""
    if not version:
        return _requires_dist_names(project_meta)
    vmeta = _pypi_json(name, version)
    if vmeta is None:
        msg = f"metadata fetch failed for pinned {name}=={version}; dependency scan incomplete"
        if errors is None:
            print(f"  [WARN] {msg}", file = sys.stderr)
        else:
            errors.append(msg)
        return []
    return _requires_dist_names(vmeta)


def _download_sdist_direct(
    name: str,
    version: str | None,
    dest: str,
    *,
    meta: dict | None = None,
) -> tuple[str | None, str | None]:
    """Fetch a project's sdist tarball directly from PyPI (no pip, no build).

    Returns ``(filepath, error)``, one non-None. Suffix preserved for the archive
    reader; bounded by ``_MAX_SDIST_BYTES`` and restricted to PyPI's CDN.
    """
    if meta is None:
        meta = _pypi_json(name)
    if meta is None:
        return None, f"PyPI metadata fetch failed for {name}"
    picked: tuple[str, str] | None = None
    for f in _release_files(meta, version):
        if f.get("packagetype") == "sdist" and f.get("url") and f.get("filename"):
            picked = (f["filename"], f["url"])
            break
    if picked is None:
        return None, f"no sdist published for {name} (version={version or 'latest'})"
    fname, url = picked
    if not _is_trusted_pypi_url(url):
        return None, f"refusing non-PyPI sdist URL for {name}: {url[:80]}"
    # basename + sanitize keeps the path inside dest; the char class preserves
    # the real `.tar.gz` / `.zip` suffix so the archive reader picks the format.
    safe_fname = _RE_PKG_NAME_SANITIZE.sub("_", os.path.basename(fname)) or "sdist.tar.gz"
    out = os.path.join(dest, safe_fname)
    try:
        req = urllib.request.Request(url, headers = {"Accept": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout = _SDIST_DOWNLOAD_TIMEOUT) as resp:
            if getattr(resp, "status", 200) != 200:
                return None, f"sdist HTTP {getattr(resp, 'status', '?')} for {name}"
            data = resp.read(_MAX_SDIST_BYTES + 1)
        if len(data) > _MAX_SDIST_BYTES:
            return None, f"sdist for {name} exceeds {_MAX_SDIST_BYTES} byte cap"
        with open(out, "wb") as fh:
            fh.write(data)
        print(
            f"  [INFO] fetched sdist directly (no build) for {name}: {safe_fname}",
            file = sys.stderr,
        )
        return out, None
    except Exception as exc:
        return None, f"sdist download failed for {name}: {type(exc).__name__}: {str(exc)[:120]}"


def _pip_download_with_deps(
    specs: list[str],
    dest: str,
    env: dict,
    *,
    timeout: int = 600,
) -> tuple[int, str]:
    """One `pip download --with-deps --only-binary :all:` call. Returns (rc, stderr)."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        *_PIP_DOWNLOAD_PIN_FLAGS,
        "--dest",
        dest,
    ] + list(specs)
    try:
        proc = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout, env = env)
        return proc.returncode, proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "pip download (with deps) timed out"


def _collect_flat_dir(dest: str, results: list[tuple[str, str]]) -> None:
    """Append every archive in a flat dest dir as (pkg_name, path)."""
    for fname in sorted(os.listdir(dest)):
        fpath = os.path.join(dest, fname)
        if os.path.isfile(fpath):
            pkg_name = fname.split("-")[0].replace("_", "-").lower()
            results.append((pkg_name, fpath))


def _resolve_per_spec_with_deps(
    specs: list[str], dest: str, env: dict, download_errors: list[str]
) -> None:
    """Fallback when the bulk --with-deps resolve fails: resolve each spec alone.

    A still-failing spec is probed against PyPI: sdist-only -> direct fetch (deps
    recovered one level); wheel-present but tree-unresolvable -> a --no-deps fetch
    of just that package. Only a genuine fetch failure errors (caller exits 2);
    unfetchable indirect deps are warned, since the named package is still scanned.
    """
    sdist_dep_followups: list[str] = []
    for spec in specs:
        name = _extract_pkg_name(spec)
        version = _spec_pin_version(spec)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            spec,
        ]
        try:
            proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 300, env = env)
        except subprocess.TimeoutExpired:
            download_errors.append(f"per-spec --with-deps timed out for {spec}")
            continue
        if proc.returncode == 0:
            continue  # archives landed in dest; collected by the caller
        meta = _pypi_json(name)
        if meta is not None and not _release_has_wheel(meta, version):
            fpath, serr = _download_sdist_direct(name, version, dest, meta = meta)
            if fpath is None:
                download_errors.append(serr or f"sdist fetch failed for {name}")
                continue
            sdist_dep_followups.extend(_requires_dist_for(name, version, meta, download_errors))
            continue
        # Has a wheel but the full transitive tree won't co-resolve
        # (ResolutionImpossible) -- typically a package the requirement file
        # installs with --no-deps by design (e.g. descript-audio-codec, whose
        # own pins conflict). Fetch just the package itself with --no-deps so it
        # is still scanned; its conflicting deps are out of scope here (the file
        # excludes them on purpose). Only a genuine fetch failure is an error.
        nd_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            spec,
        ]
        try:
            nd = subprocess.run(nd_cmd, capture_output = True, text = True, timeout = 180, env = env)
        except subprocess.TimeoutExpired:
            download_errors.append(f"per-spec --no-deps timed out for {spec}")
            continue
        if nd.returncode == 0:
            print(
                f"  [INFO] {name}: full tree unresolvable; scanned the package "
                f"alone (--no-deps), recovering deps individually.",
                file = sys.stderr,
            )
            # The --with-deps failure may have been a sdist-only TRANSITIVE dep,
            # which --no-deps skips. Recover the declared deps so that class is
            # still scanned (each is fetched as a wheel or direct sdist below).
            if meta is not None:
                sdist_dep_followups.extend(_requires_dist_for(name, version, meta, download_errors))
            continue
        # --no-deps also failed: last-ditch sdist fetch at the pinned version.
        if meta is not None:
            fpath, _serr = _download_sdist_direct(name, version, dest, meta = meta)
            if fpath is not None:
                continue
        download_errors.append(
            f"per-spec failed for {spec} (with-deps and --no-deps): " f"{nd.stderr.strip()[:240]}"
        )

    # Recover the transitive deps of sdist-only packages. A depth-bounded,
    # deduped worklist so a wheel dep whose own child is sdist-only is itself
    # fetched (--no-deps) and scanned -- not silently dropped -- and that child
    # is then recovered in turn. `dep` carries the version specifier so a pinned
    # version is fetched.
    seen: set[str] = set()
    worklist: list[tuple[str, int]] = [(d, 0) for d in sdist_dep_followups]
    while worklist:
        dep, depth = worklist.pop()
        dep_name = _extract_pkg_name(dep)
        key = _norm_pkg(dep_name)
        if key in seen:
            continue
        seen.add(key)
        dep_ver = _spec_pin_version(dep)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            dep,
        ]
        try:
            proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 300, env = env)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] dep download timed out for {dep}", file = sys.stderr)
            continue
        if proc.returncode == 0:
            continue
        meta = _pypi_json(dep_name)
        if meta is None:
            print(f"  [WARN] could not resolve indirect dep {dep}; skipping", file = sys.stderr)
            continue
        if not _release_has_wheel(meta, dep_ver):
            fpath, serr = _download_sdist_direct(dep_name, dep_ver, dest, meta = meta)
            if fpath is None:
                print(f"  [WARN] could not fetch sdist dep {dep}: {serr}", file = sys.stderr)
            elif depth < _MAX_DEP_FOLLOWUP_DEPTH:
                worklist.extend((d, depth + 1) for d in _requires_dist_for(dep_name, dep_ver, meta))
            continue
        # Wheel published but its tree won't co-resolve (a sdist-only child).
        # Fetch the dep alone so it is scanned, then chase its own declared deps.
        nd_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            dep,
        ]
        try:
            nd = subprocess.run(nd_cmd, capture_output = True, text = True, timeout = 180, env = env)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] dep --no-deps timed out for {dep}", file = sys.stderr)
            continue
        if nd.returncode == 0:
            if depth < _MAX_DEP_FOLLOWUP_DEPTH:
                worklist.extend((d, depth + 1) for d in _requires_dist_for(dep_name, dep_ver, meta))
            continue
        fpath, _serr = _download_sdist_direct(dep_name, dep_ver, dest, meta = meta)
        if fpath is None:
            print(f"  [WARN] could not resolve indirect dep {dep}; skipping", file = sys.stderr)
        elif depth < _MAX_DEP_FOLLOWUP_DEPTH:
            worklist.extend((d, depth + 1) for d in _requires_dist_for(dep_name, dep_ver, meta))


def download_packages(
    specs: list[str],
    dest: str,
    *,
    with_deps: bool = False,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Download packages to dest using pip download. NEVER installs.

    Returns ``(results, download_errors)``: ``results`` is ``(spec_or_name,
    filepath)`` per archive; ``download_errors`` is one-line transport-failure
    summaries. A non-empty ``download_errors`` MUST make the caller exit
    non-zero so a partial scan can't masquerade as "0 findings, all clean".

    with_deps=True downloads the full transitive tree (flat dir); a bulk resolve
    failure (sdist-only package or version conflict) degrades to per-spec
    resolution + direct sdist fetch rather than blanking the shard.
    with_deps=False (default) downloads each spec individually with --no-deps,
    also falling back to a direct sdist fetch when no wheel exists.
    """
    results: list[tuple[str, str]] = []
    download_errors: list[str] = []
    env = _pip_download_env()

    if with_deps:
        os.makedirs(dest, exist_ok = True)
        # Fast path: resolve + download the whole transitive tree in one call.
        # `--only-binary :all:` refuses sdists so we never build for metadata.
        rc, stderr = _pip_download_with_deps(specs, dest, env)
        if rc != 0:
            # Atomic resolve failed -- a sdist-only package, or a cross-package
            # version conflict (ResolutionImpossible). Degrade to per-spec
            # resolution so one bad spec can't blank the shard, then direct-fetch
            # any sdist-only holdouts (no build). Genuine failures still record an
            # error so the caller exits 2.
            print(
                f"  [INFO] bulk --with-deps resolve failed "
                f"({stderr.strip()[:160]}); falling back to per-spec resolution "
                f"for {len(specs)} spec(s).",
                file = sys.stderr,
            )
            _resolve_per_spec_with_deps(specs, dest, env, download_errors)
        # Collect everything that landed (bulk OR per-spec OR direct sdist).
        _collect_flat_dir(dest, results)
    else:
        for spec in specs:
            raw_name = _extract_pkg_name(spec)
            # Sanitize before joining into `dest` to prevent path traversal
            safe_name = _RE_PKG_NAME_SANITIZE.sub("_", raw_name) or "_pkg"
            pkg_dir = os.path.join(dest, safe_name)
            os.makedirs(pkg_dir, exist_ok = True)
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                *_PIP_DOWNLOAD_PIN_FLAGS,
                "--dest",
                pkg_dir,
                spec,
            ]
            try:
                proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 120, env = env)
            except subprocess.TimeoutExpired:
                download_errors.append(f"pip download timed out for {spec}")
                continue
            if proc.returncode != 0:
                # No wheel? Direct-fetch the sdist (no build) before erroring.
                name = _extract_pkg_name(spec)
                version = _spec_pin_version(spec)
                meta = _pypi_json(name)
                if meta is not None and not _release_has_wheel(meta, version):
                    fpath, serr = _download_sdist_direct(name, version, pkg_dir, meta = meta)
                    if fpath is not None:
                        results.append((spec, fpath))
                        continue
                    download_errors.append(serr or f"sdist fetch failed for {name}")
                    continue
                download_errors.append(
                    f"pip download failed for {spec}: {proc.stderr.strip()[:300]}"
                )
                continue

            for fname in os.listdir(pkg_dir):
                fpath = os.path.join(pkg_dir, fname)
                if os.path.isfile(fpath):
                    results.append((spec, fpath))
    return results, download_errors


# Parse requirements files

_RE_NAME = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _extract_pkg_name(spec: str) -> str:
    """Extract the package name from a pip spec string."""
    m = _RE_NAME.match(spec)
    return (
        m.group(1) if m else spec.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
    )


def parse_requirements(req_files: list[str]) -> list[dict]:
    """Parse requirements files into a list of dicts with source tracking.

    Each dict has keys: spec, name, source_file, line_num, raw_line, is_git.
    """
    results = []
    for req_file in req_files:
        abs_path = os.path.abspath(req_file)
        try:
            with open(req_file) as f:
                for line_num, raw_line in enumerate(f, 1):
                    line = raw_line.strip()
                    # Skip blanks, comments, options, nested -r
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    is_git = line.startswith("git+") or "git+" in line.split("#")[0]
                    # Strip inline comments and env markers
                    spec = line.split("#")[0].strip()
                    spec = spec.split(";")[0].strip()
                    if not spec:
                        continue
                    name = _extract_pkg_name(spec) if not is_git else spec
                    results.append(
                        {
                            "spec": spec,
                            "name": name,
                            "source_file": abs_path,
                            "line_num": line_num,
                            "raw_line": raw_line.rstrip("\n"),
                            "is_git": is_git,
                        }
                    )
        except FileNotFoundError:
            print(f"  [ERROR] Requirements file not found: {req_file}", file = sys.stderr)
    return results


def get_downloaded_version(archive_path: str) -> str | None:
    """Extract version from wheel/sdist filename.

    Wheel: {name}-{version}(-...).whl
    Sdist: {name}-{version}.tar.gz / .zip
    """
    basename = os.path.basename(archive_path)
    # Wheel: name-version-pytag-abitag-platform.whl
    if basename.endswith(".whl"):
        parts = basename[:-4].split("-")
        if len(parts) >= 2:
            return parts[1]
    # Sdist: name-version.<ext>
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar", ".zip"):
        if basename.endswith(ext):
            stem = basename[: -len(ext)]
            parts = stem.rsplit("-", 1)
            if len(parts) == 2:
                return parts[1]
    return None


# Display


def severity_color(sev: str) -> str:
    colors = {CRITICAL: "\033[91m", HIGH: "\033[93m", MEDIUM: "\033[33m"}
    return colors.get(sev, "")


RESET = "\033[0m"


def print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("\n  All clean. No suspicious patterns found.")
        return

    findings.sort(key = lambda f: SEVERITY_ORDER.get(f.severity, 99))

    print(f"\n  {'=' * 72}")
    print(f"  SCAN RESULTS: {len(findings)} finding(s)")
    print(f"  {'=' * 72}")

    for i, f in enumerate(findings, 1):
        color = severity_color(f.severity)
        print(f"\n  [{i}] {color}{f.severity}{RESET}  {f.check}")
        print(f"      Package:  {f.package}")
        print(f"      File:     {f.filename}")
        if f.evidence:
            for eline in f.evidence.split("\n"):
                print(f"      Evidence: {eline}")

    print(f"\n  {'=' * 72}")
    crits = sum(1 for f in findings if f.severity == CRITICAL)
    highs = sum(1 for f in findings if f.severity == HIGH)
    meds = sum(1 for f in findings if f.severity == MEDIUM)
    parts = []
    if crits:
        parts.append(f"{crits} CRITICAL")
    if highs:
        parts.append(f"{highs} HIGH")
    if meds:
        parts.append(f"{meds} MEDIUM")
    print(f"  Summary: {', '.join(parts)}")


# PyPI version queries and --fix logic


def version_sort_key(v: str) -> tuple:
    """PEP 440-ish sort key using stdlib only.

    Handles: epoch!, major.minor.patch, pre/post/dev suffixes.
    Returns a tuple that sorts in ascending version order.
    """
    epoch = 0
    if "!" in v:
        epoch_str, v = v.split("!", 1)
        try:
            epoch = int(epoch_str)
        except ValueError:
            pass

    # Split off pre/post/dev suffixes
    v_clean = re.split(
        r"[-_.]?(a|alpha|b|beta|rc|c|pre|preview|dev|post)", v, maxsplit = 1, flags = re.I
    )
    base = v_clean[0]
    suffix = v[len(base) :]

    parts = []
    for seg in base.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:  # pad to at least 3 parts
        parts.append(0)

    # Suffix ordering: dev < alpha < beta < rc < (none) < post
    suffix_lower = suffix.lower().lstrip(".-_")
    if suffix_lower.startswith("dev"):
        suffix_rank = -4
    elif suffix_lower.startswith(("a", "alpha")):
        suffix_rank = -3
    elif suffix_lower.startswith(("b", "beta")):
        suffix_rank = -2
    elif suffix_lower.startswith(("rc", "c", "pre", "preview")):
        suffix_rank = -1
    elif suffix_lower.startswith("post"):
        suffix_rank = 1
    else:
        suffix_rank = 0  # stable

    return (epoch, tuple(parts), suffix_rank, suffix)


def fetch_pypi_versions(name: str) -> list[str]:
    """Fetch all available versions for a package from PyPI JSON API.

    Returns versions sorted ascending by version_sort_key.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        req = urllib.request.Request(url, headers = {"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout = 30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [ERROR] Failed to query PyPI for {name}: {e}", file = sys.stderr)
        return []

    versions = list(data.get("releases", {}).keys())
    versions.sort(key = version_sort_key)
    return versions


def find_safe_version(
    name: str,
    bad_ver: str,
    tmpdir: str,
    max_search: int = 10,
) -> str | None:
    """Search backward from bad_ver for a clean version.

    Downloads and scans up to max_search older versions.
    Returns the first clean version found, or None.
    """
    versions = fetch_pypi_versions(name)
    if not versions:
        print(f"  [WARN] No versions found on PyPI for {name}", file = sys.stderr)
        return None

    try:
        bad_idx = versions.index(bad_ver)
    except ValueError:
        # bad_ver may resolve to a different string; search by sort key
        bad_key = version_sort_key(bad_ver)
        bad_idx = None
        for i, v in enumerate(versions):
            if version_sort_key(v) >= bad_key:
                bad_idx = i
                break
        if bad_idx is None:
            bad_idx = len(versions) - 1

    # Search backward from the version before bad_ver
    candidates = versions[:bad_idx]
    candidates.reverse()  # newest-first among older versions
    candidates = candidates[:max_search]

    if not candidates:
        print(f"  [WARN] No older versions to scan for {name}", file = sys.stderr)
        return None

    print(f"  Searching {len(candidates)} older version(s) of {name}...")

    for ver in candidates:
        spec = f"{name}=={ver}"
        scan_dir = os.path.join(tmpdir, f"{name}_{ver}")
        os.makedirs(scan_dir, exist_ok = True)

        downloaded, download_errors = download_packages([spec], scan_dir)
        if not downloaded:
            for err in download_errors:
                print(f"    [WARN] {err}", file = sys.stderr)
            continue

        clean = True
        for _, archive_path in downloaded:
            findings = scan_archive(archive_path, name)
            # Delete archive immediately after scanning
            try:
                os.remove(archive_path)
            except OSError:
                pass
            crit_findings = [f for f in findings if f.severity == CRITICAL]
            if crit_findings:
                clean = False
                print(f"    {ver} -- CRITICAL finding(s), skipping")
                break

        shutil.rmtree(scan_dir, ignore_errors = True)

        if clean:
            print(f"    {ver} -- clean!")
            return ver

    return None


def update_req_line(raw_line: str, safe_ver: str, old_ver: str | None) -> str:
    """Rewrite a single requirements line to pin to safe_ver.

    Preserves env markers, inline comments, and line format.
    Appends a comment noting the pin.
    """
    # Split off inline comment
    comment = ""
    if " #" in raw_line:
        code_part, comment = raw_line.split(" #", 1)
        comment = " #" + comment
    else:
        code_part = raw_line

    # Split off env markers (after semicolon)
    marker = ""
    if ";" in code_part:
        code_part, marker = code_part.split(";", 1)
        marker = ";" + marker

    # Replace version specifier (==1.2.3, >=1.2, ~=1.0, !=1.1, or bare name)
    rewritten = re.sub(
        r"([A-Za-z0-9._-]+)\s*(?:[><=!~]=?[^;#,\s]*(?:\s*,\s*[><=!~]=?[^;#,\s]*)*)?",
        lambda m: f"{m.group(1)}=={safe_ver}",
        code_part.strip(),
        count = 1,
    )

    was_note = f" (was {old_ver})" if old_ver else ""
    pin_comment = f"  # pinned by pth_scanner{was_note}"

    return f"{rewritten}{marker}{pin_comment}"


def update_req_file(filepath: str, updates: dict[int, str]) -> None:
    """Apply line-level updates to a requirements file.

    updates: {line_num (1-indexed): new_line_text}

    Writes atomically (sibling tmp file, fsync, os.replace) so a crash mid-write
    never leaves a half-written file that re-introduces a malicious pin.
    """
    with open(filepath) as f:
        lines = f.readlines()

    for line_num, new_text in updates.items():
        idx = line_num - 1
        if 0 <= idx < len(lines):
            ending = "\n" if lines[idx].endswith("\n") else ""  # preserve line ending
            lines[idx] = new_text + ending

    dirpath = os.path.dirname(os.path.abspath(filepath)) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix = ".req_fix.",
        dir = dirpath,
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        # Best effort cleanup; the destination was never touched.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _run_fix(critical_pkgs: set[str], entries: list[dict], max_search: int) -> None:
    """Run the --fix flow: find safe versions, update requirements files."""
    # Map package names to entries for source tracking
    pkg_entries: dict[str, list[dict]] = {}
    for e in entries:
        norm = e["name"].lower().replace("-", "_").replace(".", "_")
        pkg_entries.setdefault(norm, []).append(e)

    changes_summary: list[str] = []

    with tempfile.TemporaryDirectory(prefix = "pth_fix_") as tmpdir:
        for pkg_name in sorted(critical_pkgs):
            norm = pkg_name.lower().replace("-", "_").replace(".", "_")
            related = pkg_entries.get(norm, [])

            # Check if any are git deps
            git_entries = [e for e in related if e["is_git"]]
            if git_entries:
                for e in git_entries:
                    src = e["source_file"] or "CLI"
                    print(f"  [SKIP] {pkg_name} is a git URL dep in {src}, cannot auto-update")
                    changes_summary.append(f"  SKIP  {pkg_name} (git URL)")
                continue

            # Resolved version: try to extract from the spec (name==1.2.3)
            current_ver = None
            for e in related:
                spec = e["spec"]
                if "==" in spec:
                    current_ver = spec.split("==", 1)[1].split(";")[0].strip()
                    break

            if not current_ver:
                # If no pinned version, download to find what pip resolves
                dl_dir = os.path.join(tmpdir, f"resolve_{pkg_name}")
                os.makedirs(dl_dir, exist_ok = True)
                downloaded, download_errors = download_packages([pkg_name], dl_dir)
                if downloaded:
                    current_ver = get_downloaded_version(downloaded[0][1])
                else:
                    for err in download_errors:
                        print(f"  [WARN] {err}", file = sys.stderr)
                shutil.rmtree(dl_dir, ignore_errors = True)

            if not current_ver:
                print(f"  [WARN] Cannot determine current version of {pkg_name}, skipping fix")
                changes_summary.append(f"  SKIP  {pkg_name} (version unknown)")
                continue

            print(f"\n  Fixing {pkg_name} (current: {current_ver})...")
            safe_ver = find_safe_version(pkg_name, current_ver, tmpdir, max_search)

            if not safe_ver:
                print(
                    f"  [FAIL] No safe version found for {pkg_name} within {max_search} older versions"
                )
                changes_summary.append(
                    f"  FAIL  {pkg_name}=={current_ver} -> no safe version found"
                )
                continue

            print(f"  [OK]   {pkg_name}: {current_ver} -> {safe_ver}")
            changes_summary.append(f"  FIX   {pkg_name}=={current_ver} -> {pkg_name}=={safe_ver}")

            # Update all occurrences in requirements files
            file_updates: dict[str, dict[int, str]] = {}
            for e in related:
                if e["source_file"] is None:
                    # CLI arg, no file to update
                    print(f"         (CLI arg, no file to update)")
                    continue
                new_line = update_req_line(e["raw_line"], safe_ver, current_ver)
                file_updates.setdefault(e["source_file"], {})[e["line_num"]] = new_line
                print(f"         {e['source_file']}:{e['line_num']}")
                print(f"           - {e['raw_line']}")
                print(f"           + {new_line}")

            for filepath, updates in file_updates.items():
                update_req_file(filepath, updates)

    print(f"\n  {'=' * 72}")
    print(f"  FIX SUMMARY")
    print(f"  {'=' * 72}")
    for line in changes_summary:
        print(line)
    print(f"\n  Re-run without --fix to verify the scan is clean.")


# Directory scanning


def _find_requirements_files(root: str) -> list[str]:
    """Recursively find pip requirements files under root.

    Matches:
      - requirements*.txt (e.g. requirements.txt, requirements-dev.txt)
      - *.txt inside directories named 'requirements' (e.g. requirements/base.txt)
    Skips:
      - .egg-info dirs, venvs, hidden dirs, __pycache__, node_modules
    """
    import fnmatch

    skip_dirs = {"__pycache__", "node_modules", "venv", ".venv", "site-packages"}
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden and known non-requirement dirs
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in skip_dirs and not d.endswith(".egg-info")
        ]
        dirname = os.path.basename(dirpath)
        for fname in sorted(filenames):
            if not fname.endswith(".txt"):
                continue
            if fnmatch.fnmatch(fname.lower(), "requirements*.txt"):
                results.append(os.path.join(dirpath, fname))
            # *.txt inside a directory named "requirements"
            elif dirname == "requirements":
                results.append(os.path.join(dirpath, fname))
    return sorted(results)


# Baseline allowlist: triaged known-good CRITICAL/HIGH findings so the gate can
# enforce without drowning in legitimate-library noise. Matched on
# (package, package-relative file, check, evidence hash); the hash strips
# ``L<NN>:`` markers so version bumps and line shifts do not reopen an entry,
# but changed flagged code does. Regenerate with ``--write-baseline``.

_DEFAULT_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scan_packages_baseline.json"
)


def _norm_pkg(name: str) -> str:
    """PEP 503-style normalization so requests/Requests/req_uests collapse."""
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower())


# Leading "<name>-<version>/" archive root of an sdist member, which carries the
# version. Stripping it (but keeping the rest of the path) gives a key that is
# stable across version bumps yet still distinguishes same-named files.
_RE_SDIST_ROOT = re.compile(r"^[^/]+-\d[^/]*/")


def _relpath_in_package(filename: str) -> str:
    """Package-relative path: drop an sdist's version-carrying archive root.

    Wheel members are already package-relative (``numba/cuda/utils.py``); sdist
    members sit under ``numba-0.60.0/...``, so strip that one leading segment.
    """
    return _RE_SDIST_ROOT.sub("", filename, count = 1)


# Evidence joins matched spans with " | " and a newline between labelled groups,
# each span tagged "L<NN>: ". Split only on those real delimiters (a " | " before
# a marker, or a newline), never on a bare "|" -- matched code may contain a
# bitwise-or or union type. The prefix strips only a genuine leading marker, an
# optional "Label: " then "L<NN>: "; a marker-like "L<NN>:" inside raw code (e.g.
# a .pth import line) has no leading marker and is left intact.
_RE_EVIDENCE_SPLIT = re.compile(r" \| (?=L\d+:)|\n")
_RE_EVIDENCE_PREFIX = re.compile(r"^(?:[A-Za-z][A-Za-z0-9 _/+.-]*:\s*)?L\d+:\s?")


def _canon_evidence(evidence: str) -> str:
    """Matched code lines in discovery order (markers removed), duplicates kept.

    Splits evidence on its real span delimiters, drops each span's leading
    label / line-number marker, and keeps the code with its indentation. Line
    shifts are absorbed by stripping the L<NN>: markers, not by sorting, so order
    stays significant: reordering matched lines (executable context, e.g. the
    arguments of a multi-line call) reopens the finding. Keeping duplicates means
    an appended identical occurrence still changes the key."""
    spans = []
    for s in _RE_EVIDENCE_SPLIT.split(evidence or ""):
        s = _RE_EVIDENCE_PREFIX.sub("", s, count = 1).rstrip()
        if s:
            spans.append(s)
    return "\n".join(spans)


def _evidence_hash(evidence: str) -> str:
    """Stable digest of the canonical matched evidence."""
    return hashlib.sha256(_canon_evidence(evidence).encode("utf-8", "replace")).hexdigest()


def _finding_key(f: Finding) -> tuple[str, str, str, str]:
    """Allowlist key: package, package-relative path, check, evidence hash.

    The evidence hash is over the set of matched code, so the key survives version
    bumps, line shifts and reordering but reopens when the flagged code changes --
    so a future payload in a baselined file/check is not auto-suppressed.
    """
    return (
        _norm_pkg(f.package),
        _relpath_in_package(f.filename),
        f.check,
        _evidence_hash(f.evidence),
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
    keys: set[tuple[str, str, str, str]] = set()
    legacy = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            # Use the reviewed hash; else recompute it from the stored evidence.
            evidence_hash = e.get("evidence_hash") or _evidence_hash(e.get("evidence") or "")
            if not e.get("evidence_hash"):
                legacy += 1
            keys.add(
                (
                    _norm_pkg(e["package"]),
                    _relpath_in_package(e["file"]),
                    e["check"],
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


def _write_baseline(path: str, findings: list[Finding]) -> None:
    """Persist CRITICAL/HIGH findings as an allowlist for human triage."""
    entries = []
    seen: set[tuple[str, str, str, str]] = set()
    for f in sorted(findings, key = lambda f: SEVERITY_ORDER.get(f.severity, 99)):
        if f.severity not in (CRITICAL, HIGH):
            continue
        key = _finding_key(f)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "package": f.package,
                "file": _relpath_in_package(f.filename),
                "check": f.check,
                "severity": f.severity,
                "evidence": f.evidence,
                "evidence_hash": _evidence_hash(f.evidence),
            }
        )
    doc = {
        "_comment": (
            "scan_packages.py allowlist. Each entry is a CRITICAL/HIGH finding "
            "manually judged benign. Matched on (package, package-relative file, "
            "check, evidence_hash); evidence_hash is over the matched code with "
            "L<NN>: markers stripped, so version bumps and line shifts do not "
            "reopen an entry but changed code does. severity and evidence are for "
            "review only. Regenerate with --write-baseline AFTER reviewing every line."
        ),
        "version": 1,
        "entries": entries,
    }
    with open(path, "w", encoding = "utf-8") as fh:
        json.dump(doc, fh, indent = 2, sort_keys = False)
        fh.write("\n")
    print(f"  Wrote {len(entries)} baseline entr(y/ies) to {path}")


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


# Main


def main() -> int:
    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "packages",
        nargs = "*",
        help = "Package specs (e.g. requests==2.32.5 fastapi)",
    )
    parser.add_argument(
        "-r",
        "--requirements",
        action = "append",
        default = [],
        metavar = "FILE",
        help = "Requirements file(s) to scan",
    )
    parser.add_argument(
        "-d",
        "--scan-dir",
        action = "append",
        default = [],
        metavar = "DIR",
        help = "Recursively find requirements*.txt files in DIR",
    )
    parser.add_argument(
        "--with-deps",
        action = "store_true",
        help = "Also download and scan transitive dependencies (full dependency tree)",
    )
    parser.add_argument(
        "--fix",
        action = "store_true",
        help = "Auto-search for safe versions and update requirements files",
    )
    parser.add_argument(
        "--max-search",
        type = int,
        default = 10,
        metavar = "N",
        help = "Max older versions to scan when searching for safe version (default: 10)",
    )
    parser.add_argument(
        "--baseline",
        metavar = "FILE",
        default = None,
        help = (
            "Allowlist JSON of triaged known-good findings to suppress. "
            f"Defaults to {os.path.basename(_DEFAULT_BASELINE_PATH)} next to this "
            "script if present."
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
            "Write the current CRITICAL/HIGH findings to FILE as an allowlist, "
            "then exit 0. Review every entry before committing it."
        ),
    )
    args = parser.parse_args()

    # --scan-dir: auto-discover requirements files
    req_files = list(args.requirements)
    for scan_dir in args.scan_dir:
        found = _find_requirements_files(scan_dir)
        if found:
            print(f"  Found {len(found)} requirements file(s) in {scan_dir}/")
            for f in found:
                print(f"    {f}")
            req_files.extend(found)
        else:
            print(f"  [WARN] No requirements files found in {scan_dir}/", file = sys.stderr)

    # Build unified entry list: list of dicts with source tracking
    entries: list[dict] = []

    # CLI args -> entries with no source file
    for pkg in args.packages or []:
        entries.append(
            {
                "spec": pkg,
                "name": _extract_pkg_name(pkg),
                "source_file": None,
                "line_num": None,
                "raw_line": pkg,
                "is_git": pkg.startswith("git+") or "git+" in pkg,
            }
        )

    # Requirements files -> entries with source tracking
    if req_files:
        entries.extend(parse_requirements(req_files))

    if not entries:
        parser.print_help()
        return 2

    # Deduplicate by normalized name, preserving first occurrence
    seen: set[str] = set()
    unique_entries: list[dict] = []
    for e in entries:
        key = e["name"].lower().replace("-", "_").replace(".", "_")
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)

    specs = [e["spec"] for e in unique_entries]
    mode_label = " (with transitive deps)" if args.with_deps else ""
    print(f"  Scanning {len(specs)} package(s){mode_label}...")

    all_findings: list[Finding] = []

    # Hard pin-block: refuse to download known-malicious PyPI versions
    specs, blocked_findings = _check_blocked_pypi_versions(specs)
    all_findings.extend(blocked_findings)

    tmpdir = tempfile.mkdtemp(prefix = "pth_scan_")
    atexit.register(lambda d = tmpdir: shutil.rmtree(d, ignore_errors = True))
    download_errors: list[str] = []
    try:
        downloaded, download_errors = download_packages(
            specs,
            tmpdir,
            with_deps = args.with_deps,
        )
        print(f"  Downloaded {len(downloaded)} archive(s).")

        for spec, archive_path in downloaded:
            pkg_name = _extract_pkg_name(spec)
            findings = scan_archive(archive_path, pkg_name)
            all_findings.extend(findings)
            # Delete archive immediately after scanning
            try:
                os.remove(archive_path)
            except OSError:
                pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors = True)

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

    print_findings(active)
    if suppressed:
        crit_s = sum(1 for f in suppressed if f.severity == CRITICAL)
        high_s = sum(1 for f in suppressed if f.severity == HIGH)
        med_s = sum(1 for f in suppressed if f.severity == MEDIUM)
        print(
            f"\n  {len(suppressed)} finding(s) suppressed by baseline "
            f"{baseline_path} "
            f"({crit_s} CRITICAL, {high_s} HIGH, {med_s} MEDIUM)."
        )

    # --fix mode: auto-search for safe versions (only real, non-baselined ones)
    if args.fix and active:
        critical_pkgs = {f.package for f in active if f.severity == CRITICAL}
        if critical_pkgs:
            print(
                f"\n  --fix: Searching for safe versions of {len(critical_pkgs)} CRITICAL package(s)..."
            )
            _run_fix(critical_pkgs, entries, args.max_search)

    # Surface pip-download failures BEFORE the exit code so a partial download
    # can't masquerade as "0 findings, all clean" (silent-failure hardening 4).
    # Also keeps us from writing a baseline from an incomplete scan.
    if download_errors:
        print(
            f"\n  {'=' * 72}\n"
            f"  SCAN INCOMPLETE: {len(download_errors)} pip download "
            f"failure(s):\n"
            f"  {'=' * 72}",
            file = sys.stderr,
        )
        for err in download_errors:
            print(f"  [ERROR] {err}", file = sys.stderr)
        print(
            "  Refusing to report 'all clean' on a partial scan; exiting 2.",
            file = sys.stderr,
        )
        return 2

    # --write-baseline: persist the full current CRITICAL/HIGH set as the new
    # allowlist (ignoring any loaded baseline), then exit 0. Only reached once
    # the scan is known complete.
    if args.write_baseline:
        _write_baseline(args.write_baseline, all_findings)
        return 0

    # Exit code: 1 only if a NON-baselined CRITICAL or HIGH remains. This is the
    # signal CI gates on once the baseline reaches a clean run.
    if any(f.severity in (CRITICAL, HIGH) for f in active):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
