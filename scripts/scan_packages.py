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

Exit codes:
    0 -- no CRITICAL or HIGH findings
    1 -- CRITICAL or HIGH findings detected
    2 -- no packages specified
"""

import argparse
import atexit
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"

SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2}

# Hard pin-blocks for publicly confirmed malicious PyPI versions.
# Source: Socket.dev 2026-05-12 disclosure (Mini Shai-Hulud May-12 wave) and
# earlier Semgrep / Endor reports for the `lightning` entries.
BLOCKED_PYPI_VERSIONS: dict[str, set[str]] = {
    "guardrails-ai": {"0.10.1"},
    "mistralai": {"2.4.6"},
    "lightning": {"2.6.2", "2.6.3"},
}

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Subprocess / OS exec patterns
RE_SUBPROCESS = re.compile(
    r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
    r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b",
)

# Encoding / obfuscation
RE_BASE64 = re.compile(
    r"\bbase64\s*\.\s*(b64decode|decodebytes|b32decode|b16decode)\b"
    r"|\bcodecs\s*\.\s*decode\b",
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
RE_ANTI_ANALYSIS = re.compile(
    r"\bptrace\b"
    r"|\bsys\s*\.\s*gettrace\s*\("
    r"|\bsys\s*\.\s*settrace\b"
    r"|\bTracerPid\b"
    r"|\b/proc/self/status\b"
    r"|\bIsDebuggerPresent\b"
    r"|\bvirtualbox\b.*\bhardware\b"
    r"|\bvmware\b.*\bdetect\b"
    r"|\btime\.sleep\s*\(\s*(?:[3-9]\d{2,}|[1-9]\d{3,})\s*\)"  # long sleep (anti-sandbox)
    r"|\bplatform\.\s*system\b.*\bif\b.*\b(?:Linux|Windows|Darwin)\b",
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
RE_OPENSSL_CLI = re.compile(
    r"\bopenssl\s+(enc|rand|rsautl|pkeyutl|genrsa|dgst|s_client)\b"
)

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

# Developer-tool persistence hooks. The PyTorch Lightning 2.6.x compromise
# planted SessionStart hooks into Claude Code, VS Code tasks, and Cursor
# settings so the payload re-attached on every editor open. Catches any
# package writing into a known dev-tool config that supports auto-run.
RE_DEV_TOOL_HIJACK = re.compile(
    r"\.claude/settings\.json"
    r"|\.cursor/.*hooks"
    r"|\.vscode/(?:tasks|settings|launch)\.json"
    r"|SessionStart|folderOpen|onCommand:.*runTask"
    r"|/etc/profile\.d/"
    r"|\b\.bashrc\b|\b\.zshrc\b|\b\.profile\b"
    r"|\bautomator\b.*\.workflow\b",
)

# Hard-coded credential / API-token regexes embedded in source. Packages
# that ship regexes for OTHER people's secrets are nearly always
# stealers (litellm 1.82.7, elementary-data 0.23.3, Shai-Hulud).
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

# Mini Shai-Hulud May-12 2026 wave indicators. The dropper artifact name
# `transformers.pyz` is high-confidence (no legit PyPI package ships a `.pyz`
# named after `transformers`); the host + slogans are CRITICAL.
RE_MAY12_IOC = re.compile(
    r"(git-tanstack\.com|/tmp/transformers\.pyz|transformers\.pyz"
    r"|With Love TeamPCP|We've been online over 2 hours)",
    re.IGNORECASE,
)

# JavaScript-side obfuscation. The npm chalk/debug compromise and the
# Lightning router_runtime.js use the same minifier-style hex-var name
# pattern; a bundle full of `_0x1f2e3d` identifiers is a near-universal
# tell for a malicious npm payload (and very rare in legit minified code
# that ships in PyPI wheels).
RE_JS_OBFUSCATION = re.compile(
    r"_0x[a-f0-9]{4,6}\s*=\s*function"
    r"|var\s+_0x[a-f0-9]{4,6}\b"
    r"|(?:\\x[0-9a-f]{2}){10,}"  # \x-escape strings
    r"|String\.fromCharCode\s*\(\s*\d+\s*(?:,\s*\d+\s*){10,}\)",
)

# Web3 / wallet-hijack pattern. The Qix npm phish overrode fetch /
# XMLHttpRequest and attached a `window.ethereum` listener that
# Levenshtein-swapped recipient addresses on the way to the network.
RE_WEB3_HIJACK = re.compile(
    r"\bwindow\.ethereum\b"
    r"|\bweb3\.eth\.\w+\s*\("
    r"|XMLHttpRequest\.prototype\.(?:open|send)\s*="
    r"|(?:^|\s)fetch\s*=\s*\(?\s*async"
    r"|TronWeb|solanaWeb3",
)

# Self-propagating supply-chain worms (Shai-Hulud, ForceMemo) plant
# their own GitHub workflow in every repo they can reach, and lean on
# trufflehog/gitleaks for credential discovery. The combo of any of
# these strings inside a *package payload* is overwhelming evidence of
# repo-takeover intent.
RE_WORKFLOW_INJECT = re.compile(
    r"\.github/workflows/[^\"\']*\.ya?ml"
    r"|\btrufflehog\b|\bgitleaks\b"
    r"|/user/repos\?affiliation=.*owner.*collaborator"
    r"|\bshai-hulud\b|EveryBoiWeBuildIsAWormyBoi"
    r"|\bgit\s+push\s+--force\b.*--no-verify",
    re.IGNORECASE | re.DOTALL,
)

# Shell-side patterns specific to install.sh / postinstall scripts that
# pipe remote code into a shell. `curl ... | sh` and friends are the
# canonical npm postinstall dropper.
RE_SHELL_DROPPER = re.compile(
    r"\bcurl\b[^\n|]*\|\s*(?:sh|bash|zsh)\b"
    r"|\bwget\b[^\n|]*-O-\s*\|\s*(?:sh|bash|zsh)\b"
    r"|\bnpx\b\s+-y\s+[^\s]+@latest\s*\|"
    r"|\beval\s+\$\(\s*curl\b"
    r"|\bbash\s+<\(\s*curl\b",
)


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    severity: str
    package: str
    filename: str
    check: str
    evidence: str = ""


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------


def check_pth_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run all .pth-specific checks.

    Executable .pth files run on every Python startup, so any suspicious
    pattern in a .pth is treated as CRITICAL.
    """
    findings = []

    # Only care about .pth files that have import lines (executable)
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

    # Large base64 blob (special handling for blob size)
    if RE_LARGE_BLOB.search(content):
        blob = RE_LARGE_BLOB.search(content).group()
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                f".pth has large base64-like blob ({len(blob)} chars)",
                blob[:120] + "...",
            )
        )

    # Catch-all: any import line at all in .pth (if nothing else triggered)
    if not findings and import_lines:
        evidence = "\n".join(import_lines[:5])
        if len(import_lines) > 5:
            evidence += f"\n... ({len(import_lines)} import lines total)"
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
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                f"Unusually large executable .pth ({size} bytes)",
                f"{len(import_lines)} import line(s) in {size}-byte .pth file",
            )
        )

    return findings


def check_py_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run all .py-specific checks."""
    findings = []
    basename = os.path.basename(filename)
    is_setup = basename in ("setup.py", "setup.cfg")
    is_init = basename == "__init__.py"

    # Pre-compute all pattern matches
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

    # ---------------------------------------------------------------
    # CRITICAL: combination patterns that strongly indicate malice
    # ---------------------------------------------------------------

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
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "openssl encryption + network/key material (encrypted exfiltration)",
                f"OpenSSL: {_extract_evidence(content, RE_OPENSSL_CLI)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
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

    # ---------------------------------------------------------------
    # HIGH: single strong signals or weaker combinations
    # ---------------------------------------------------------------

    # Obfuscated payload: base64 + exec/eval + large blob
    if has_base64 and has_exec_eval and has_blob:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "base64 decode + exec/eval + large encoded blob",
                f"Base64: {_extract_evidence(content, RE_BASE64)}\n"
                f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}",
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
                f"Key: {_extract_evidence(content, RE_EMBEDDED_KEYS)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Anti-analysis + any other suspicious pattern
    if has_anti and (has_network or has_subprocess or has_exec_eval):
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Anti-analysis/sandbox evasion + suspicious behavior",
                f"Anti: {_extract_evidence(content, RE_ANTI_ANALYSIS)}",
            )
        )

    # DNS exfiltration with dynamic hostnames
    if has_dns_exfil and (has_base64 or has_network or has_creds):
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "DNS exfiltration / tunneling patterns",
                _extract_evidence(content, RE_DNS_EXFIL),
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

    # ---------------------------------------------------------------
    # MEDIUM: standalone signals (informational, may be legitimate)
    # ---------------------------------------------------------------

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
                _extract_evidence(content, RE_EMBEDDED_KEYS),
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


def _extract_evidence(content: str, pattern: re.Pattern, max_matches: int = 3) -> str:
    """Pull matching lines as evidence snippets."""
    lines = content.splitlines()
    matches = []
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            snippet = line.strip()
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            matches.append(f"L{i}: {snippet}")
            if len(matches) >= max_matches:
                break
    return " | ".join(matches) if matches else ""


# ---------------------------------------------------------------------------
# Non-Python checkers
# ---------------------------------------------------------------------------
# Several recent PyPI compromises (PyTorch Lightning 2.6.x, ForceMemo)
# carried the active payload in a bundled .js / .sh / workflow yaml so
# the Python imports looked clean on first glance. These checkers scan
# those file types when they appear inside a Python wheel/sdist.


def check_js_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run JS-side checks. Triggered by .js / .mjs / .cjs / .ts."""
    findings = []

    # A JS file *inside a Python wheel* that's larger than 100 KB is
    # itself anomalous (legit Python packages don't ship hand-written
    # JS bundles). Combined with ANY of the other JS heuristics it is
    # CRITICAL; standalone it is HIGH.
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
                _extract_evidence(content, RE_TOKEN_REGEX),
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
    if is_large and not findings:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                f"Python wheel ships large ({len(content) // 1024} KB) JS bundle "
                "(uncommon; manually review)",
                "",
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
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell installs developer-tool persistence hook (.bashrc / "
                "profile.d / vscode tasks) AND has network or exec",
                _extract_evidence(content, RE_DEV_TOOL_HIJACK),
            )
        )
    if RE_TOKEN_REGEX.search(content) and RE_NETWORK.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell embeds credential regexes AND makes network calls",
                _extract_evidence(content, RE_TOKEN_REGEX),
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
    # A GitHub workflow file inside a *PyPI package* is itself
    # suspicious (Shai-Hulud's whole MO is to plant `shai-hulud.yml`
    # in every repo it can write to). Anything matching the workflow
    # injection signature gets flagged CRITICAL.
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


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------

# Tarbomb caps, mirrored from scripts/scan_npm_packages.py::safe_extract.
# Refuses zip-of-death / tar-of-death archives so a hostile sdist or
# wheel cannot exhaust memory or fill the temp dir before content
# scanning even starts. Keep these constants in sync with the npm side;
# we duplicate rather than import to keep `scan_packages.py` standalone.
HARD_MAX_FILE_BYTES = 64 * 1024 * 1024  # 64 MiB per member
HARD_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB cumulative
HARD_MAX_MEMBERS = 50_000  # entries per archive


def _refuse_unsafe_member_name(name: str) -> str | None:
    """Return a refusal reason for a member name, or None if safe.

    Mirrors `scan_npm_packages.py::safe_extract` semantics: no absolute
    paths, no `..` traversal segments. The caller is responsible for
    checking the resolved path lands inside the extract root, but for
    iter_archive_files we never write to disk so the name-shape check
    plus the in-memory size cap is sufficient.
    """
    if name.startswith("/") or ".." in Path(name).parts:
        return f"unsafe member name {name!r}"
    return None


def iter_archive_files(archive_path: str):
    """Yield (filename, text_content) for every file in a wheel/sdist.

    Streams members with size + count caps applied at the member level
    so a tarbomb / zipbomb cannot blow up the scanner's memory budget.
    On cap breach we emit a `[WARN]` log and short-circuit the archive.
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
                # Declared (uncompressed) size cap.
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
                # Refuse symlinks / hardlinks / devices outright -- the
                # scanner never writes them anyway, but tar parsers
                # have historically dereferenced them on extract.
                if member.issym() or member.islnk():
                    print(
                        f"  [WARN] {path.name}: refused link member "
                        f"{member.name!r}",
                        file = sys.stderr,
                    )
                    continue
                if member.isdev() or member.isfifo():
                    print(
                        f"  [WARN] {path.name}: refused special member "
                        f"{member.name!r}",
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
                    # Bound the read so a tar header that lies about
                    # size cannot OOM us.
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

    A corrupted archive container (truncated wheel, bad gzip header,
    etc.) used to be silently skipped by an ``except Exception: continue``
    inside ``iter_archive_files``. Per the silent-failure hardening
    (SF1) it now emits a CRITICAL ``archive_corrupted`` finding so the
    main loop counts and surfaces it rather than reporting "0 findings".
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
                # Lightning 2.6.x hid its real payload in a 14.8 MB
                # router_runtime.js inside a Python wheel. Without this
                # branch we'd have only seen the small Python loader.
                findings.extend(check_js_file(content, filename, package))
            elif lower.endswith((".sh", ".bash")):
                findings.extend(check_shell_file(content, filename, package))
            elif "/.github/workflows/" in lower and lower.endswith((".yml", ".yaml")):
                # Shai-Hulud / ForceMemo plant their own GHA workflow.
                # A workflow file inside a *PyPI package* is on its own
                # already a yellow flag; pattern-match the worm signatures.
                findings.extend(check_workflow_file(content, filename, package))
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as exc:
        # The archive cannot be opened or is structurally broken. A
        # benign wheel/sdist always opens; a malformed one is either a
        # transport corruption (treat as scan failure) or a deliberate
        # attempt to bypass scanners that swallow archive errors.
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


# ---------------------------------------------------------------------------
# Download packages
# ---------------------------------------------------------------------------


_RE_PYPI_SPEC_VERSION = re.compile(r"==\s*([A-Za-z0-9_.\-+!]+)")


def _check_blocked_pypi_versions(
    specs: list[str],
) -> tuple[list[str], list[Finding]]:
    """Filter ``specs`` against ``BLOCKED_PYPI_VERSIONS``.

    Returns ``(safe_specs, findings)``. Each blocked spec emits a CRITICAL
    ``Finding`` and is removed from the returned spec list so the caller
    never fetches the malicious tarball. Specs without an ``==X.Y.Z`` pin
    pass through unchanged -- pip will resolve them at download time and
    the existing scanners will catch the payload via the IOC regexes.
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
            # Drop the spec; do not download.
            continue
        safe.append(spec)
    return safe, findings


def _pip_download_env() -> dict[str, str]:
    """Return a scrubbed environment for invoking `pip download`.

    Hostile shells / CI configs can override the index with PIP_INDEX_URL,
    PIP_EXTRA_INDEX_URL, or a user `pip.conf`. We strip every PIP_*
    override and route the resolver explicitly at PyPI. PIP_CONFIG_FILE
    is forced to /dev/null so a stray ~/.pip/pip.conf with an
    extra-index-url cannot bypass the pin.
    """
    env = {**os.environ}
    # Drop any user override.
    for key in [k for k in env if k.startswith("PIP_")]:
        env.pop(key, None)
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"
    env["PIP_EXTRA_INDEX_URL"] = ""
    env["PIP_CONFIG_FILE"] = "/dev/null"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


# Pip resolver flags shared by both download branches. Pinning the
# index URL on the CLI is belt + braces with the env scrub above.
# `--no-build-isolation` is deliberately NOT set; we never invoke
# setup.py at all because of `--only-binary :all:`.
_PIP_DOWNLOAD_PIN_FLAGS = [
    "--index-url",
    "https://pypi.org/simple",
    "--only-binary",
    ":all:",
]


# Strip any character that could escape `dest` via `os.path.join`. This
# is the last line of defence before `pkg_dir = os.path.join(dest, ...)`
# so a spec like `../../etc/foo==1.0` cannot land outside the temp tree.
_RE_PKG_NAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]")


def download_packages(
    specs: list[str],
    dest: str,
    *,
    with_deps: bool = False,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Download packages to dest using pip download. NEVER installs.

    Returns ``(results, download_errors)`` where ``results`` is a list of
    ``(spec_or_name, filepath)`` for every downloaded archive and
    ``download_errors`` is a list of one-line transport-failure summaries.
    A non-empty ``download_errors`` MUST cause the caller to exit non-zero
    even if no findings were produced; a silent ``0 findings, scan
    incomplete`` is the bug class this return-shape was widened to fix.

    When with_deps=True, downloads the full transitive dependency tree
    in a single pip invocation (all archives land in one flat dir).
    When with_deps=False (default), downloads each spec individually
    with --no-deps.
    """
    results: list[tuple[str, str]] = []
    download_errors: list[str] = []
    env = _pip_download_env()

    if with_deps:
        # Single pip download call for all specs + their transitive deps.
        # `--only-binary :all:` refuses sdists so we never execute a
        # setup.py just to learn dependency metadata; combined with the
        # scrubbed env, pip is wired hard at pypi.org.
        os.makedirs(dest, exist_ok = True)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
        ] + specs
        try:
            proc = subprocess.run(
                cmd,
                capture_output = True,
                text = True,
                timeout = 600,  # transitive resolution can be slow
                env = env,
            )
            if proc.returncode != 0:
                msg = (
                    f"pip download (with deps) failed: " f"{proc.stderr.strip()[:500]}"
                )
                print(f"  [ERROR] {msg}", file = sys.stderr)
                download_errors.append(msg)
        except subprocess.TimeoutExpired:
            msg = "pip download (with deps) timed out"
            print(f"  [ERROR] {msg}", file = sys.stderr)
            download_errors.append(msg)

        # Collect every archive that landed in dest
        for fname in sorted(os.listdir(dest)):
            fpath = os.path.join(dest, fname)
            if os.path.isfile(fpath):
                # Derive package name from filename
                pkg_name = fname.split("-")[0].replace("_", "-").lower()
                results.append((pkg_name, fpath))
    else:
        for spec in specs:
            raw_name = _extract_pkg_name(spec)
            # Sanitize before joining into `dest` so a hostile spec
            # cannot path-traverse out of the destination directory.
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
                proc = subprocess.run(
                    cmd,
                    capture_output = True,
                    text = True,
                    timeout = 120,
                    env = env,
                )
                if proc.returncode != 0:
                    msg = (
                        f"pip download failed for {spec}: "
                        f"{proc.stderr.strip()[:500]}"
                    )
                    print(f"  [ERROR] {msg}", file = sys.stderr)
                    download_errors.append(msg)
                    continue
            except subprocess.TimeoutExpired:
                msg = f"pip download timed out for {spec}"
                print(f"  [ERROR] {msg}", file = sys.stderr)
                download_errors.append(msg)
                continue

            # Find downloaded file(s)
            for fname in os.listdir(pkg_dir):
                fpath = os.path.join(pkg_dir, fname)
                if os.path.isfile(fpath):
                    results.append((spec, fpath))
    return results, download_errors


# ---------------------------------------------------------------------------
# Parse requirements files
# ---------------------------------------------------------------------------

_RE_NAME = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _extract_pkg_name(spec: str) -> str:
    """Extract the package name from a pip spec string."""
    m = _RE_NAME.match(spec)
    return (
        m.group(1)
        if m
        else spec.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
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
                    # Strip inline comments and environment markers for spec
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
    # Sdist: name-version.tar.gz / .tar.bz2 / .zip
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar", ".zip"):
        if basename.endswith(ext):
            stem = basename[: -len(ext)]
            parts = stem.rsplit("-", 1)
            if len(parts) == 2:
                return parts[1]
    return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def severity_color(sev: str) -> str:
    colors = {CRITICAL: "\033[91m", HIGH: "\033[93m", MEDIUM: "\033[33m"}
    return colors.get(sev, "")


RESET = "\033[0m"


def print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("\n  All clean. No suspicious patterns found.")
        return

    # Sort by severity
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


# ---------------------------------------------------------------------------
# PyPI version queries and --fix logic
# ---------------------------------------------------------------------------


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

    # Parse numeric parts
    parts = []
    for seg in base.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            parts.append(0)
    # Pad to at least 3 parts
    while len(parts) < 3:
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
        suffix_rank = 0  # stable release

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

    # Find index of bad version
    try:
        bad_idx = versions.index(bad_ver)
    except ValueError:
        # bad_ver might have been resolved to a different string; search by sort key
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

        downloaded = download_packages([spec], scan_dir)
        if not downloaded:
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

        # Clean up scan dir for this version
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

    # Replace version specifier
    # Match patterns like ==1.2.3, >=1.2, ~=1.0, <=2.0, !=1.1, or bare name
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

    Writes atomically: stage in a sibling tmp file on the same
    filesystem, fsync, then `os.replace` over the original. A SIGKILL
    or power loss mid-write therefore either leaves the original
    intact or leaves the fully new file -- never a half-written
    requirements file (which would silently re-introduce a malicious
    pin).
    """
    with open(filepath) as f:
        lines = f.readlines()

    for line_num, new_text in updates.items():
        idx = line_num - 1
        if 0 <= idx < len(lines):
            # Preserve original line ending
            ending = "\n" if lines[idx].endswith("\n") else ""
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


def _run_fix(
    critical_pkgs: set[str],
    entries: list[dict],
    max_search: int,
) -> None:
    """Run the --fix flow: find safe versions, update requirements files."""
    # Map package names to their entries for source tracking
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
                    print(
                        f"  [SKIP] {pkg_name} is a git URL dep in {src}, cannot auto-update"
                    )
                    changes_summary.append(f"  SKIP  {pkg_name} (git URL)")
                continue

            # Get the currently resolved version
            # Try to extract from the spec (e.g. name==1.2.3)
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
                downloaded = download_packages([pkg_name], dl_dir)
                if downloaded:
                    current_ver = get_downloaded_version(downloaded[0][1])
                # Delete resolution download immediately
                shutil.rmtree(dl_dir, ignore_errors = True)

            if not current_ver:
                print(
                    f"  [WARN] Cannot determine current version of {pkg_name}, skipping fix"
                )
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
            changes_summary.append(
                f"  FIX   {pkg_name}=={current_ver} -> {pkg_name}=={safe_ver}"
            )

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

    # Print summary
    print(f"\n  {'=' * 72}")
    print(f"  FIX SUMMARY")
    print(f"  {'=' * 72}")
    for line in changes_summary:
        print(line)
    print(f"\n  Re-run without --fix to verify the scan is clean.")


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


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
        # Skip hidden dirs and known non-requirement dirs
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in skip_dirs
            and not d.endswith(".egg-info")
        ]
        dirname = os.path.basename(dirpath)
        for fname in sorted(filenames):
            if not fname.endswith(".txt"):
                continue
            # Match requirements*.txt anywhere
            if fnmatch.fnmatch(fname.lower(), "requirements*.txt"):
                results.append(os.path.join(dirpath, fname))
            # Match *.txt inside a directory named "requirements"
            elif dirname == "requirements":
                results.append(os.path.join(dirpath, fname))
    return sorted(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
            print(
                f"  [WARN] No requirements files found in {scan_dir}/", file = sys.stderr
            )

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

    # Hard pin-block: refuse to download known-malicious PyPI versions.
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

    print_findings(all_findings)

    # --fix mode: auto-search for safe versions
    if args.fix and all_findings:
        critical_pkgs = {f.package for f in all_findings if f.severity == CRITICAL}
        if critical_pkgs:
            print(
                f"\n  --fix: Searching for safe versions of {len(critical_pkgs)} CRITICAL package(s)..."
            )
            _run_fix(critical_pkgs, entries, args.max_search)

    # Surface any pip-download failures BEFORE the scan-result exit code so
    # an empty / partial download cannot mask itself as "0 findings, all
    # clean". This is item (4) of the silent-failure hardening: an
    # unresolvable spec or PyPI timeout used to print to stderr and exit 0.
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
            "  Refusing to report 'all clean' on a partial scan; " "exiting 2.",
            file = sys.stderr,
        )
        return 2

    # Exit code: 1 if any CRITICAL or HIGH
    if any(f.severity in (CRITICAL, HIGH) for f in all_findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
