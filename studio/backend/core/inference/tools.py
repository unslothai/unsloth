# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool definitions and executors for LLM tool calling: web search
(DuckDuckGo), Python code execution, and terminal commands."""

import ast
import http.client
import os
import signal

os.environ["UNSLOTH_IS_PRESENT"] = "1"

import asyncio
import base64
import binascii
import codecs
import random
import re
import shlex
import ssl
import subprocess
import sys
import tempfile
import threading
import urllib.request
import zlib

from core.inference.mcp_client import (
    MCP_TOOL_PREFIX,
    TOOL_CACHE_INVALIDATING_FIELDS,
    cache_tools,
    call_tool_sync,
    get_cached_tools,
    in_failure_cooloff,
    is_stdio,
    list_tools_async,
    parse_server_headers,
    probe_timeout,
    record_probe_failure,
    stdio_mcp_enabled,
)
from storage import mcp_servers_db

from loggers import get_logger

logger = get_logger(__name__)

_EXEC_TIMEOUT = 300  # 5 minutes

# Splits the UI source-map from the result; loops strip it (like __IMAGES__).
RAG_SOURCES_SENTINEL = "\n__RAG_SOURCES__:"

# Import these at module level so the preexec_fn closure triggers no imports in
# the forked child (which can deadlock multi-threaded servers).
_libc = None
if sys.platform == "linux":
    try:
        import ctypes
        import ctypes.util

        _libc_name = ctypes.util.find_library("c")
        if _libc_name:
            _libc = ctypes.CDLL(_libc_name, use_errno = True)
    except (OSError, AttributeError):
        pass

_resource = None
if sys.platform != "win32":
    try:
        import resource as _resource
    except ImportError:
        pass

# Raster-image allowlist for sandbox file serving.
# No .svg (XSS via embedded scripts), no .html, no .pdf.
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_MAX_OUTPUT_CHARS = 8000  # truncate long output
_BLOCKED_COMMANDS_COMMON = frozenset(
    {
        "rm",
        "dd",
        "chmod",
        "chown",
        "mkfs",
        "mount",
        "umount",
        "fdisk",
        "sudo",
        "su",
        "doas",
        "pkexec",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "kill",
        "killall",
        "pkill",
        "passwd",
        "curl",
        "wget",
        "nc",
        "ncat",
        "netcat",
        "socat",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "eval",
        "source",
        "ln",
        # flock [options] <file>|<fd> <command> (or flock -c <command>) runs an arbitrary
        # command in an unguarded child while holding a lock; its file/fd operand + -c forms
        # make the command word hard to resolve, so block the wrapper outright.
        "flock",
    }
)
# Language interpreters that run inline / file / stdin code in a FRESH child process.
# The runtime filesystem backstop only patches the current interpreter, so a spawned
# `python -c '...'`, `perl -e '...'`, `node -e '...'`, etc. runs with none of the guard
# monkeypatches and can write/delete outside the session workdir. Blocking the
# interpreter at shell command position closes that child-process escape; the sandbox's
# own python_execute tool is the supported way to run Python (it IS guarded). Argument
# position (`echo python`, `ls /usr/bin/python3`) is unaffected by the command-position
# scanner.
_INTERPRETER_COMMANDS = frozenset(
    {
        "python",
        "python2",
        "python3",
        "pythonw",
        "perl",
        "ruby",
        "node",
        "nodejs",
        "php",
        "deno",
        "lua",
        "luajit",
        "rscript",
        # awk variants run an inline program that can write files (print > "/path")
        # in an unguarded child without any shell redirection token the scanner sees.
        "awk",
        "gawk",
        "mawk",
        "nawk",
    }
)
# Python console-script entry points that START A FRESH, UNGUARDED Python interpreter (their
# shebang is the same interpreter whose bin dir the safe env prepends). Running a workdir file
# through one -- subprocess.run(['pytest', 'test_evil.py']) / pip install <local sdist> -- is
# the same child-process escape as a bare `python foo.py`, which is already blocked above, so
# the launcher entry points are denied for consistency. In-workdir Python belongs in the
# guarded python_execute tool. (This is deliberately tight to well-known launchers; a broader
# allowlisted-tooling relaxation is tracked separately.)
_PYTHON_LAUNCHER_COMMANDS = frozenset(
    {
        "pip",
        "pip2",
        "pip3",
        "pipx",
        "pytest",
        "py.test",
        "ipython",
        "ipython3",
    }
)
# Recipe / task runners that execute shell commands read from a workdir control file (a
# Makefile recipe, etc.) in an unguarded child, the same escape as the Python launchers: a
# sandboxed snippet can write a Makefile whose recipe runs `echo x > /tmp/p` and then run
# `make`. Deny the runner; in-workdir work belongs in the guarded tools. (Kept tight to the
# common ones; a broader allowlisted-tooling relaxation is tracked separately.)
_RECIPE_RUNNER_COMMANDS = frozenset({"make", "gmake"})
# File-creating / writing coreutils. Same rationale as the interpreters: a spawned child
# runs without the in-process realpath backstop, so subprocess.run(['touch', '/tmp/x']),
# tee, cp, mv, ... write / create / delete outside the session workdir. In-workdir file
# work should go through the guarded Python file APIs. (dd / ln / rm are already denied
# above.) Native / unknown binaries the sandbox cannot enumerate remain an OS-isolation
# residual.
_CHILD_WRITE_COMMANDS = frozenset(
    {
        "touch",
        "tee",
        "cp",
        "mv",
        "mkdir",
        "install",
        "truncate",
        "mkfifo",
        "mknod",
        "shred",
        "unlink",
        # patch applies a diff in an unguarded child; patch -o /tmp/x writes the result outside
        # the workdir, and a patch targeting ../../tmp/x escapes even without -o. Same native
        # writer class as touch / cp / tar.
        "patch",
        # rmdir removes (empty) directories; a bash child gets no realpath guard, so
        # rmdir /tmp/some-empty-dir deletes a host directory outside the workdir.
        "rmdir",
        # split / csplit slice a file into PREFIXaa, PREFIXab, ... at an arbitrary prefix
        # path, creating files outside the workdir in an unguarded child.
        "split",
        "csplit",
        # Archive / compression tools create files in an unguarded child (tar -cf out,
        # zip out, unzip extracts, gzip file). In-workdir archiving should go through the
        # guarded Python APIs.
        "tar",
        "zip",
        "unzip",
        "gzip",
        "gunzip",
        "bzip2",
        "bunzip2",
        "xz",
        "unxz",
        "zstd",
        "7z",
        "7za",
        "rar",
        "unrar",
        "cpio",
        "rsync",
        # mktemp creates a file / dir at a caller-chosen template path (mktemp
        # /tmp/x.XXXXXX, mktemp -d), writing outside the workdir in an unguarded child.
        "mktemp",
        # sponge (moreutils) soaks up stdin and writes it to a file argument
        # (printf x | sponge /tmp/probe), an unguarded-child write outside the workdir.
        "sponge",
    }
)
_BLOCKED_COMMANDS_COMMON = (
    _BLOCKED_COMMANDS_COMMON
    | _INTERPRETER_COMMANDS
    | _PYTHON_LAUNCHER_COMMANDS
    | _RECIPE_RUNNER_COMMANDS
    | _CHILD_WRITE_COMMANDS
)
_BLOCKED_COMMANDS_WIN = frozenset(
    {
        "rmdir",
        "takeown",
        "icacls",
        "runas",
        "powershell",
        "pwsh",
    }
)
# Commands that take a path as DATA but never read its CONTENTS: echo / printf print their args,
# the no-ops do nothing, and test / [ only stat. A literal sensitive path handed to one of these
# (echo /etc/passwd) is not an exfiltration, so the literal-sensitive-path scan skips it. Any
# OTHER (unknown) command word still fails closed -- only this explicit allowlist is exempt.
_SHELL_NON_READER_COMMANDS = frozenset(
    {
        "echo",
        "printf",
        ":",
        "true",
        "false",
        "test",
        "[",
        "[[",
    }
)
_BLOCKED_COMMANDS = (
    _BLOCKED_COMMANDS_COMMON | _BLOCKED_COMMANDS_WIN
    if sys.platform == "win32"
    else _BLOCKED_COMMANDS_COMMON
)


_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "\n", "(", ")", "`", "{", "}"})
# Bash keywords whose FOLLOWING word is a new command position: the compound-statement
# headers (if / while / until / elif run their CONDITION command) and the body markers
# (then / do / else). `if touch x; then :; fi` executes `touch` as the condition command, so
# these must reset command position -- otherwise the header word is mistaken for the command
# and the real command it precedes is skipped as an argument.
_SHELL_KEYWORDS_AS_SEP = frozenset({"then", "do", "else", "elif", "if", "while", "until", "coproc"})
# POSIX / common shell binaries. A shell without an inline `-c` payload runs unscanned
# code (a script file, -s / stdin, or a bare stdin-reading shell), so it is denied.
_SHELL_BINARIES = frozenset({"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish"})
# Utilities whose LATER argv elements are actions / write flags, not inert arguments
# (find -exec/-delete, sed -i / w, sort -o). A non-shell argv resolving to one of these is
# re-scanned as a reconstructed command line so those dangerous flags are caught.
_ARGV_TAIL_SCAN_COMMANDS = frozenset(
    {"find", "sed", "gsed", "ssed", "perl", "sort", "git", "openssl", "sqlite3"}
)
# openssl option flags whose VALUE is an output file the unguarded openssl child writes (rand
# -out, req -keyout, ca -CAout / -CAserial, ...). A value that escapes the workdir writes a host
# file the realpath guard never sees; a workdir-local -out and the no-output forms stay allowed.
_OPENSSL_WRITE_FLAGS = frozenset(
    {"-out", "-writerand", "-keyout", "-CAout", "-CAkeyout", "-CAserial"}
)
# iconv writes its converted output to the -o / --output file in an unguarded child, so an
# escaping value writes a host path the realpath guard never sees (printf x | iconv -o /tmp/p).
_ICONV_WRITE_FLAGS = frozenset({"-o", "--output"})
# sqlite3 CLI dot-commands that WRITE (or read) an arbitrary file argument in the unguarded
# child: `.output FILE` / `.once FILE` redirect query output to FILE, `.excel` / `.import` /
# `.backup FILE` / `.save FILE` / `.dump FILE` / `.clone FILE` create files, `.log FILE` writes
# a log, and `.read FILE` sources SQL from FILE. A FILE that escapes the workdir writes / reads a
# host path the realpath guard never sees. The group captures the FILE operand for a path check.
_SQLITE_DOTFILE_RE = re.compile(
    r"(?m)^\s*\.(?:output|once|excel|import|dump|clone|log|read)\b\s+(?:-{1,2}\S+\s+)*"
    r"(?P<f>(?:'[^']*'|\"[^\"]*\"|\S+))"
)
# .backup ?DB? FILE / .save ?DB? FILE put the written FILE LAST (an optional schema name precedes
# it), and .open ?OPTIONS? FILE puts the opened/created database file after its options. The first
# operand of these (a schema name, or an option token) is not the file, so capture the whole tail
# and check the LAST bare operand instead of the first.
_SQLITE_LASTFILE_RE = re.compile(r"(?m)^\s*\.(?:backup|save|open)\b[^\n]*")
# sqlite3 dot-commands that RUN a system shell command in the unguarded child: `.shell CMD` /
# `.system CMD` ("Run CMD ARGS... in a system shell"), and `.excel` (opens the result in a
# system program). These execute regardless of any path check, so match the command itself.
_SQLITE_SHELL_RE = re.compile(r"(?m)^\s*\.(?:shell|system|excel)\b")
# sqlite3 CLI options that consume a SEPARATED operand (so the value after them is NOT the
# database filename). Only -init also reads a file (its value is path-checked at the call site).
_SQLITE_OPERAND_OPTS = frozenset(
    {
        "-init",
        "-cmd",
        "-mode",
        "-separator",
        "-newline",
        "-nullvalue",
        "-lookaside",
        "-mmap",
        "-maxsize",
    }
)


def _is_versioned_interpreter(base: str) -> bool:
    """True when ``base`` is a version-suffixed interpreter name (python3.14, python3.11,
    perl5.36, ruby3.0) whose unversioned stem is a blocked interpreter. Those binaries are
    commonly on the sandbox PATH and start the same unguarded child as the bare name."""
    stem = re.sub(r"[0-9][0-9.]*$", "", base)
    return stem != base and stem in _INTERPRETER_COMMANDS


# Absolute paths under a standard system bin dir are trusted as real system commands (their
# basename is still interpreter-checked separately); every OTHER explicit path is a local file.
_SYSTEM_BIN_PREFIXES = ("/bin/", "/usr/bin/", "/usr/local/bin/", "/sbin/", "/usr/sbin/")


def _is_local_executable_path(tok: str) -> bool:
    """True when a command word is an explicit path to a LOCAL executable file (./evil, ../x,
    subdir/tool, /tmp/x). Running such a file executes whatever its shebang names in an
    UNGUARDED child -- a sandboxed snippet can create + chmod ./evil with `#!/usr/bin/python3`
    and run it, starting an interpreter the argv basename scan never sees. A bare command name
    resolved via PATH (no slash) and an absolute system-bin path are not treated as local."""
    t = tok.replace("\\", "/")
    if "/" not in t:
        return False
    # Collapse .. before the system-bin exemption so a workdir shebang cannot masquerade as a
    # trusted binary via /usr/bin/../../<workdir>/evil (normpath -> /<workdir>/evil, not exempt).
    # normpath keeps the leading ./ -> bare-name collapse harmless: the "/" check above already
    # ran on the original token, so ./evil (has a slash) still reaches here and stays local.
    norm = os.path.normpath(t)
    return not norm.startswith(_SYSTEM_BIN_PREFIXES)


def _split_path_entries(value: str):
    """Split a PATH value on ':' separators, but NOT on a ':' inside a ${...} expansion (so a
    ${VAR:-default} default operator is not mistaken for a list separator)."""
    entries = []
    cur = []
    depth = 0
    i = 0
    v = value.replace("\\", "/")
    while i < len(v):
        c = v[i]
        if c == "$" and i + 1 < len(v) and v[i + 1] == "{":
            depth += 1
            cur.append("${")
            i += 2
            continue
        if c == "}" and depth > 0:
            depth -= 1
            cur.append(c)
            i += 1
            continue
        if c == ":" and depth == 0:
            entries.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    entries.append("".join(cur))
    return entries


def _path_var_resolves_unsafe(var, assignments):
    """Whether a PATH component expanded from shell variable ``var`` can resolve to the workdir.
    HOME / PWD are the session workdir; PATH is the trusted search list; a var assigned a
    relative / cwd value earlier in the same command (P=.; PATH=$P) is unsafe; an unknown
    external var (CONDA_PREFIX) is assumed to expand to a trusted absolute path."""
    if var in ("HOME", "PWD"):
        return True
    if var == "PATH":
        return False
    if assignments and var in assignments:
        return _path_value_is_unsafe(assignments[var], assignments)
    return False


def _path_var_is_unknown_external(var, assignments):
    """True when ``var`` is neither a workdir alias (HOME/PWD), the trusted inherited PATH, nor a
    variable assigned earlier in the same command. In the sandbox such a variable is UNSET, so its
    expansion is EMPTY -- not a trusted absolute path."""
    return var not in ("HOME", "PWD", "PATH") and not (assignments and var in assignments)


def _path_entry_empty_expansion_unsafe(entry: str, var: str) -> bool:
    """Model an unknown/unset ``$var`` in a PATH ENTRY as EMPTY (the sandbox reality) and report
    whether the entry then collapses to an empty or RELATIVE path (both search the cwd). A bare
    ``$EVIL`` -> ``''`` and ``${X}bin`` -> ``bin`` are unsafe; ``$CONDA_PREFIX/bin`` -> ``/bin``
    stays absolute and is safe."""
    blanked = re.sub(r"\$\{?" + re.escape(var) + r"\}?", "", entry)
    return blanked == "" or not blanked.startswith(("/", "%"))


def _path_value_is_unsafe(value: str, assignments = None) -> bool:
    """True when a PATH search list would let a BARE (no-slash) command resolve to a workdir
    executable: any entry that is ``.``, empty (``:`` = cwd), a relative directory, or one that
    expands to the session workdir. In the sandbox ``HOME`` and the child cwd ARE the workdir,
    so ``~`` / ``~user``, ``$HOME`` / ``$PWD``, a ``${VAR:-.}`` default that is relative, and a
    ``$VAR`` bound to a relative value earlier in the same command (``P=.; PATH=$P``) are unsafe.
    An absolute (``/...``), ``%VAR%``, or unknown external ``$VAR`` entry (``$PATH``,
    ``$CONDA_PREFIX/bin``, assumed to expand to a trusted absolute path) is safe. ``assignments``
    maps local shell VAR=value bindings so a locally-controlled expansion can be resolved."""
    for entry in _split_path_entries(value):
        e = entry.strip()
        if e in ("", "."):
            return True
        # ~ / ~user expand to HOME, which is the session workdir in the sandbox.
        if e.startswith("~"):
            return True
        # A command substitution $(...) / `...` in a PATH entry is a DYNAMIC value the analyzer
        # cannot resolve (PATH=$(pwd) points the search list at the cwd, where an earlier sandboxed
        # step may have planted an executable), so treat it as unsafe rather than a trusted $VAR
        # expansion -- otherwise the $ branch below swallows $( as a non-matching variable.
        if "$(" in e or "`" in e:
            return True
        if e.startswith("${"):
            inner = e[2:]
            if inner.endswith("}"):
                inner = inner[:-1]
            # ${VAR-def} / ${VAR:-def} / ${VAR=def} / ${VAR:=def}: def applies when VAR is unset/
            # empty, so a relative default is unsafe. ${VAR:+alt} / ${VAR:?msg} carry no path.
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(:?[-=?+])(.*)$", inner)
            if m:
                var, op, default = m.group(1), m.group(2), m.group(3)
                if op in (":-", "-", ":=", "=") and _path_value_is_unsafe(default, assignments):
                    return True
                if _path_var_resolves_unsafe(var, assignments):
                    return True
                continue
            var = re.split(r"[/}]", inner, maxsplit = 1)[0]
            if _path_var_resolves_unsafe(var, assignments):
                return True
            if _path_var_is_unknown_external(
                var, assignments
            ) and _path_entry_empty_expansion_unsafe(e, var):
                return True
            continue
        if e.startswith("$"):
            m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)", e)
            if m and _path_var_resolves_unsafe(m.group(1), assignments):
                return True
            # An unknown/unset $VAR expands to EMPTY in the sandbox, so a bare `$EVIL` (or one that
            # leaves a relative remainder, `${X}bin`) collapses the entry to the cwd; only an entry
            # that stays ABSOLUTE with the var blanked ($CONDA_PREFIX/bin -> /bin) is trusted.
            if (
                m
                and _path_var_is_unknown_external(m.group(1), assignments)
                and _path_entry_empty_expansion_unsafe(e, m.group(1))
            ):
                return True
            continue  # $PATH / $CONDA_PREFIX/bin / $1: a trusted absolute expansion
        if e.startswith(("/", "%")):
            continue
        return True  # a relative directory (relbin, ./tools)
    return False


def _dynamic_path_value_unsafe(value_node, env) -> bool:
    """A NON-literal PATH assignment value (os.environ['PATH'] = '.:' + os.environ['PATH'],
    f'.:{x}') is unsafe when a COMPLETE, fully-literal PATH entry it contributes is a relative /
    cwd / empty entry. Operands are const-folded; an OPAQUE segment (os.environ['PATH'], a
    variable) taints only the entry that spans it, so a dynamic ABSOLUTE extension
    (venv + ':' + $PATH, '/usr/local/bin:' + $PATH) stays allowed. Returns True only for a
    provable unsafe entry -- the folded literal case is handled by the caller."""
    segments: list = []  # ("lit", str) or ("opaque",)

    def _flatten(n):
        folded = _const_fold(n, env)
        if isinstance(folded, str):
            segments.append(("lit", folded))
            return
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            _flatten(n.left)
            _flatten(n.right)
            return
        if isinstance(n, ast.JoinedStr):
            for _p in n.values:
                if isinstance(_p, ast.Constant) and isinstance(_p.value, str):
                    segments.append(("lit", _p.value))
                else:
                    _fv = _const_fold(getattr(_p, "value", _p), env)
                    segments.append(("lit", _fv) if isinstance(_fv, str) else ("opaque",))
            return
        segments.append(("opaque",))

    _flatten(value_node)
    entries: list = []  # (text, complete, tainted)
    cur = ""
    tainted = False
    for seg in segments:
        if seg[0] == "opaque":
            tainted = True
            continue
        parts = seg[1].split(":")
        for j, part in enumerate(parts):
            if j == 0:
                cur += part
            else:
                entries.append((cur, True, tainted))
                cur = part
                tainted = False
    _last_opaque = bool(segments) and segments[-1][0] == "opaque"
    entries.append((cur, not _last_opaque, tainted))
    for text, complete, taint in entries:
        if complete and not taint and _path_value_is_unsafe(text):
            return True
    return False


def _arg_escapes_workdir(tok: str) -> bool:
    """True when a path-like argument can point OUTSIDE the session workdir: an absolute path
    (``/tmp/x``), a ``~`` / ``~user`` home path (home == workdir, but a shell child follows the
    real HOME), or any path with a ``..`` component that can traverse above the workdir. A
    workdir-relative name (``sub/out``, ``repo``) stays inside and returns False. Used to confine
    file-creating child commands (git init/clone <dir>, ...) that the runtime guard cannot see."""
    t = tok.replace("\\", "/")
    if t.startswith("/") or t.startswith("~"):
        return True
    return ".." in t.split("/")


def _git_operand_escapes(tok: str, assigns = None) -> bool:
    """As _arg_escapes_workdir, but resolves a ``$VAR`` / ``${VAR}`` bound to an escaping value
    earlier in the SAME command, as the WHOLE token (``OUT=/tmp/repo; git init $OUT``) OR as a
    PREFIX (``P=/tmp; git init $P/repo``, ``openssl rand -out $P/key``). An unknown external
    expansion is left to the literal check (so ``git clone $REPO_URL`` is not a false positive)."""
    # A command substitution $(...) / `...` operand (git init $(printf /tmp/x)) is a DYNAMIC path
    # the analyzer cannot resolve: the real shell expands it and native git creates the result
    # outside the workdir, so fail closed. Tokenization splits `$(` into a bare `$` and `(`
    # (and backticks into their own tokens), so the fragment left as the operand is `$` / `` ` ``.
    if tok in ("$", "`") or "$(" in tok or "`" in tok:
        return True
    m = re.match(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?(.*)$", tok)
    if m and assigns and m.group(1) in assigns:
        return _arg_escapes_workdir(assigns[m.group(1)] + m.group(2))
    return _arg_escapes_workdir(tok)


def _cwd_wrapper_escapes(tokens, cmd_idx) -> bool:
    """True when an ``env -C DIR`` / ``--chdir DIR`` / ``--chdir=DIR`` / glued ``-CDIR`` wrapper
    in the SAME command segment BEFORE ``cmd_idx`` changes the child's cwd to a directory that
    escapes the workdir (a literal escaping ``cwd=`` on a subprocess call reaches here as the same
    synthetic ``env -C <dir>`` prefix). Under such a cwd even a workdir-RELATIVE write operand
    (openssl -out key, sqlite3 db.sqlite) lands outside the session. Scans back to the previous
    shell separator; a workdir-local chdir (env -C sub) returns False."""
    for _bk in range(cmd_idx - 1, -1, -1):
        _bt = tokens[_bk]
        if _bt in _SHELL_SEPARATORS or _bt in _SHELL_KEYWORDS_AS_SEP:
            break
        if _bt in ("-C", "--chdir") and _bk + 1 < len(tokens):
            if _arg_escapes_workdir(tokens[_bk + 1]):
                return True
        elif _bt.startswith("--chdir=") and _arg_escapes_workdir(_bt.split("=", 1)[1]):
            return True
        elif _bt.startswith("-C") and len(_bt) > 2 and _arg_escapes_workdir(_bt[2:]):
            return True
    return False


def _operand_relative_local(tok: str) -> bool:
    """A literal RELATIVE path operand that resolves under the child cwd, so it escapes the workdir
    when the cwd itself escapes (paired with _cwd_wrapper_escapes). Absolute (``/x``), home (``~``),
    ``$``/backtick expansions (unknown -- left to _git_operand_escapes), option flags, empty, and
    the sqlite in-memory forms return False so they are handled by their own checks."""
    if not tok:
        return False
    _u = tok
    if len(_u) >= 2 and _u[0] == _u[-1] and _u[0] in ("'", '"'):
        _u = _u[1:-1]
    if not _u or _u[0] in ("/", "~", "-") or "$" in _u or "`" in _u:
        return False
    _ul = _u.lower()
    if _u == ":memory:" or _ul.startswith("file::memory:") or "mode=memory" in _ul:
        return False
    return True


# git options whose VALUE is a path that a native git child writes to / operates in (the runtime
# realpath backstop never sees a native git process). A value that escapes the workdir lets git
# write outside the session: -C / --git-dir / --work-tree / --separate-git-dir (repo location),
# and -o / --output / -O / --output-directory (git archive / format-patch write their output
# file there). Handled for `-x val`, `--opt val`, and inline `--opt=val` forms.
_GIT_PATH_VALUE_OPTIONS = frozenset(
    {
        "-C",
        "--git-dir",
        "--work-tree",
        "--separate-git-dir",
        "-o",
        "--output",
        "-O",
        "--output-directory",
        # fast-export / fast-import marks files: git writes / reads the given path from its
        # unguarded child (git fast-export --export-marks=/tmp/marks HEAD).
        "--export-marks",
        "--import-marks",
        "--import-marks-if-exists",
    }
)
# git config keys whose value is a COMMAND git runs in an unguarded child (git -c KEY=CMD ... /
# git config KEY CMD). core.fsmonitor / sshCommand / pager / editor / credential.helper /
# diff.external / gpg.program / sequence.editor / uploadpack.packObjectsHook run their value;
# core.hooksPath / init.templateDir re-point hooks (undoing the sandbox hook suppression).
_GIT_EXEC_CONFIG_KEYS = frozenset(
    {
        "core.fsmonitor",
        "core.sshcommand",
        "core.pager",
        "core.editor",
        "core.hookspath",
        "core.askpass",
        "sequence.editor",
        "diff.external",
        "gpg.program",
        "credential.helper",
        "init.templatedir",
        "uploadpack.packobjectshook",
        "ssh.variant",
    }
)


def _git_config_key_is_exec(key: str) -> bool:
    """True for a git config key whose value git executes as a command (or that re-points hooks)."""
    k = key.strip().lower()
    if k in _GIT_EXEC_CONFIG_KEYS:
        return True
    # include.path / includeIf.<cond>.path pull in another config file whose contents git then
    # honors, so an included workdir config can set core.hooksPath / core.fsmonitor (re-enabling a
    # planted hook) even though the direct key is blocked. Treat any include*.path key as exec.
    if k == "include.path" or (k.startswith("includeif.") and k.endswith(".path")):
        return True
    # filter.<name>.clean/smudge/process, diff.<name>.command, merge.<name>.driver take commands.
    parts = k.split(".")
    if len(parts) == 3:
        section, _, leaf = parts
        if section == "filter" and leaf in ("clean", "smudge", "process"):
            return True
        if section == "diff" and leaf == "command":
            return True
        if section == "merge" and leaf == "driver":
            return True
    return False


# The only shell redirection targets trusted without a realpath check: standard device
# sinks that cannot escape the workdir. Every other target (relative or absolute) fails
# closed, because the unguarded child follows symlinks and resolves relative names against a
# cwd the static scanner cannot verify (a pre-existing `out -> /tmp/host` symlink escapes).
_SAFE_REDIRECT_TARGETS = frozenset(
    {"/dev/null", "/dev/zero", "/dev/full", "/dev/stdout", "/dev/stderr", "/dev/tty"}
)
# Coreutils that read + print file contents. A shell-expanded ($VAR / `cmd`) path passed
# to one of these can exfiltrate a host secret whose name the static scan cannot resolve.
_SHELL_READ_COMMANDS = frozenset(
    {
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "od",
        "xxd",
        "hexdump",
        "strings",
        "nl",
        "tac",
        "cut",
        "sort",
        "uniq",
        "wc",
        "base64",
        "base32",
        "sed",
        "grep",
        "egrep",
        "fgrep",
        "rev",
        "fold",
        "paste",
        "comm",
        "tr",
        "dd",
        "readlink",
        "realpath",
        # diff-style utilities print file contents in their output: `diff SECRET /dev/null`
        # (or `cmp -l SECRET /dev/null`) leaks the file line-by-line / byte-by-byte, so a
        # shell-expanded ($VAR / glob / `cmd`) path handed to one exfiltrates a host secret.
        "diff",
        "sdiff",
        "diff3",
        "colordiff",
        "cmp",
        # directory / file enumerators: an EXPANDED root (find ${P:-/root/.ssh} -exec cat {} \;,
        # ls $SECRET) enumerates a host path the static scan cannot resolve, and find's -exec
        # can then read every match. Literal find / ls (find . -name '*.py', ls -la) carry no
        # expansion and stay allowed; only a $ / backtick / escaping-glob operand fails closed.
        "find",
        "ls",
        # openssl can READ + print a file's contents (openssl base64 -in SECRET, openssl enc -d
        # -in SECRET, openssl x509 -in SECRET), so an EXPANDED / sensitive -in path exfiltrates a
        # host secret. Literal in-workdir input (openssl base64 -in data.txt) carries no expansion
        # and stays allowed; only a $ / backtick / escaping-glob / sensitive operand fails closed.
        "openssl",
    }
)
# Wrappers whose next non-flag argument is the command Bash will exec.
_COMMAND_PREFIXES = frozenset(
    {
        "env",
        "command",
        "builtin",
        "exec",
        "time",
        "nohup",
        "nice",
        "setsid",
        "stdbuf",
        "timeout",
        "ionice",
        "chroot",
        "sudo",
        "doas",
        "su",
        "xargs",
        # chrt [options] <priority> <command> [<arg>...]: util-linux scheduler wrapper that
        # execs the following command, so chrt -o 0 touch /tmp/x must resolve to touch.
        "chrt",
        # watch [options] command: repeatedly runs command (via sh -c, or exec with -x), so
        # watch -x touch /tmp/x / watch -n 2 rm -rf / must resolve to the wrapped command.
        "watch",
        # taskset [options] <mask | -c cpu-list> <command> [<arg>...]: util-linux affinity
        # wrapper that execs the following command, so taskset 1 touch /tmp/x / taskset -c 0,1
        # rm -rf must resolve to the wrapped command (the mask / cpu-list is skipped as a
        # numeric operand). The -p PID form operates on an existing process and execs nothing.
        "taskset",
    }
)
# A shell assignment prefix: NAME=value or NAME+=value (bash append). The optional `+` is part
# of the operator, so `PATH+=:. cmd` is recognized as an assignment prefix, not a command word.
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
# Per-wrapper option flags that take a SEPARATED operand (the NEXT token is the flag's value,
# not the command). Anything not listed -- a no-operand flag (env -i, xargs -0), a GLUED short
# flag (stdbuf -oL), or a --long=value -- does NOT consume the next token, so the real command
# after it is still analysed. Wrappers absent from the map default to no operand-taking flags
# (their numeric args, nice -n 5 / timeout 5, are skipped separately).
_WRAPPER_OPERAND_FLAGS = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "stdbuf": frozenset({"-i", "--input", "-o", "--output", "-e", "--error"}),
    "ionice": frozenset({"-c", "--class", "-n", "--classdata", "-p", "--pid"}),
    "sudo": frozenset(
        {
            "-u",
            "--user",
            "-g",
            "--group",
            "-C",
            "--close-from",
            "-h",
            "--host",
            "-p",
            "--prompt",
            "-r",
            "--role",
            "-t",
            "--type",
            "-U",
            "--other-user",
            "-T",
            "--command-timeout",
            "-R",
            "--chroot",
            "-D",
            "--chdir",
        }
    ),
    "xargs": frozenset(
        {
            "-n",
            "--max-args",
            "-P",
            "--max-procs",
            "-L",
            "--max-lines",
            "-s",
            "--max-chars",
            "-I",
            "--replace",
            "-E",
            "-d",
            "--delimiter",
            "-a",
            "--arg-file",
            # --process-slot-var VAR sets an env var for the child; the separated operand VAR
            # would otherwise be mistaken for the command word (xargs --process-slot-var V touch).
            "--process-slot-var",
        }
    ),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
    "chrt": frozenset({"-T", "--sched-runtime", "-P", "--sched-period", "-D", "--sched-deadline"}),
    "watch": frozenset({"-n", "--interval"}),
    # taskset -c CPU-LIST cmd (the cpu-list is a separated operand); -p PID targets an existing
    # process (no command follows). The bare hex / decimal mask form is skipped as a numeric arg.
    "taskset": frozenset({"-c", "--cpu-list", "-p", "--pid"}),
}


def _wrapper_flag_takes_operand(wrapper, flag: str) -> bool:
    """True when a wrapper option FLAG consumes the NEXT token as a separated operand
    (env -u NAME, nice -n 5, stdbuf -o L). A glued short flag (-oL), a --long=value, or any
    flag not listed for the wrapper does NOT, so the command word after it is still analysed
    (stdbuf -oL sed -i ..., xargs -0 sed ...)."""
    if "=" in flag:
        return False
    if not flag.startswith("--") and len(flag) > 2:
        return False  # glued short flag: -oL already carries its value
    return flag in _WRAPPER_OPERAND_FLAGS.get(wrapper, frozenset())


# GNU sed can WRITE files (`w FILE`, `W FILE`, `s///w FILE`) or EXECUTE shell commands
# (`e COMMAND`, `s///e`) straight from its SCRIPT even without -i, escaping the workdir in an
# unguarded child. The filename/command may follow immediately (GNU accepts `w/tmp/x`) or
# after whitespace. A plain `s/word/x/` has `w`/`e` inside the pattern/replacement (a letter or
# closing delimiter follows), so these patterns are shaped to skip that.
_SED_WRITE_RE = re.compile(r"(?<![A-Za-z])[wW](?:[ \t]|/|~)")
_SED_EXEC_RE = re.compile(r"(?:^|[;\n{}]|[0-9$])[ \t]*e(?:[ \t;}\n]|$)")
# GNU sed runs `e COMMAND` after an ADDRESS too (`/regex/e cmd`, `/a/,/b/e cmd`, `1,/x/e cmd`).
# Match a `/regex/` that sits at a command boundary (start / `;` / `{` / `}`) OR right after a
# range comma, followed by an optional `!` and the `e` command with a trailing separator. The `e`
# must be followed by whitespace / `;` / `}` / EOL, so a substitution replacement that merely ends
# in `.../e/` (e before the closing delimiter) does not match; and requiring a boundary before the
# opening `/` means `s/a/e /` (its `/` preceded by `s`) is not mistaken for an address. A
# line-number address then `e` (`1e`, `1,/x/` handled via the comma) is already covered above.
_SED_ADDR_EXEC_RE = re.compile(
    r"(?:^|[;\n{},])[ \t]*!?[ \t]*/(?:[^/\\]|\\.)*/[ \t]*!?[ \t]*e(?:[ \t;}\n]|$)"
)
# A completed s/PATTERN/REPL/FLAGS whose FLAGS include w (write) or e (execute).
_SED_SFLAG_RE = re.compile(r"s(.)(?:(?!\1).)*\1(?:(?!\1).)*\1[A-Za-z0-9]*[we]")
_FIND_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
# find's -exec / -ok command runs up to a `;` or `+` terminator; everything between
# is a full command line (may itself begin with a wrapper like env/timeout or sh -c).
_FIND_EXEC_TERMINATORS = frozenset({";", "+"})


def _is_wrapper_numeric_arg(token: str) -> bool:
    """A wrapper's numeric argument (`nice -n 5`, `timeout 5m`, `timeout 0.5`).

    Accepts a plain int/float, optionally with a single trailing GNU ``timeout``
    duration unit (s/m/h/d). Used only to decide whether to skip a token while a
    command-prefix wrapper is still awaiting its real command, so being permissive
    keeps the scan on the following command rather than dropping out of command
    position.
    """
    t = token.lstrip("-")
    if not t:
        return False
    # A hex affinity mask (taskset 0x3 cmd).
    if t[:2].lower() == "0x" and len(t) > 2:
        try:
            int(t, 16)
            return True
        except ValueError:
            return False
    # Strip a single trailing GNU timeout duration unit (timeout 5m / 0.5s).
    if len(t) > 1 and t[-1] in "smhd":
        t = t[:-1]
    # A cpu-list / affinity mask of digits with , and - separators (taskset -c 0,1 / 0-3 cmd).
    if (
        any(c in ",-" for c in t)
        and all(c in "0123456789,-" for c in t)
        and any(c.isdigit() for c in t)
    ):
        return True
    try:
        float(t)
        return True
    except ValueError:
        return False


_ANSI_C_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}


def _decode_ansi_c(body: str) -> str:
    """Decode the escape sequences bash resolves inside a $'...' word (\\n, \\t, \\xHH,
    octal \\NNN, \\uHHHH, ...) so the resulting command word matches what actually runs."""
    out = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        d = body[i + 1]
        if d in _ANSI_C_ESCAPES:
            out.append(_ANSI_C_ESCAPES[d])
            i += 2
        elif d == "x":
            j, h = i + 2, ""
            while j < n and len(h) < 2 and body[j] in "0123456789abcdefABCDEF":
                h += body[j]
                j += 1
            if h:
                out.append(chr(int(h, 16)))
                i = j
            else:
                out.append(c)
                out.append(d)
                i += 2
        elif d in "01234567":
            j, o = i + 1, ""
            while j < n and len(o) < 3 and body[j] in "01234567":
                o += body[j]
                j += 1
            out.append(chr(int(o, 8) & 0xFF))
            i = j
        elif d in ("u", "U"):
            width = 4 if d == "u" else 8
            j, h = i + 2, ""
            while j < n and len(h) < width and body[j] in "0123456789abcdefABCDEF":
                h += body[j]
                j += 1
            if h:
                out.append(chr(int(h, 16)))
                i = j
            else:
                out.append(c)
                out.append(d)
                i += 2
        else:
            out.append(c)
            out.append(d)
            i += 2
    return "".join(out)


def _normalize_ansi_c_quotes(command: str) -> str:
    """Rewrite bash ANSI-C ($'...') and locale ($"...") quoted words to plain quoted words
    so shlex sees the token bash actually executes. shlex leaves `$'touch'` as the literal
    `$touch`, so a writer/interpreter hidden behind ANSI-C quoting (`$'touch' x`,
    `$'\\x74ouch' x`) never matches the command blocklist otherwise."""
    if "$'" not in command and '$"' not in command:
        return command
    res = []
    i, n = 0, len(command)
    while i < n:
        if command[i] == "$" and i + 1 < n and command[i + 1] == '"':
            res.append('"')  # locale translation: bash just strips the leading $
            i += 2
            continue
        if command[i] == "$" and i + 1 < n and command[i + 1] == "'":
            j, buf = i + 2, []
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    buf.append(command[j])
                    buf.append(command[j + 1])
                    j += 2
                    continue
                if command[j] == "'":
                    break
                buf.append(command[j])
                j += 1
            decoded = _decode_ansi_c("".join(buf))
            # Re-emit as a single-quoted shlex token (escaping embedded single quotes).
            res.append("'" + decoded.replace("'", "'\\''") + "'")
            i = j + 1  # skip the closing quote
            continue
        res.append(command[i])
        i += 1
    return "".join(res)


_IFS_RE = re.compile(r"\$\{IFS[^}]*\}|\$IFS\b")


def _expand_ifs(command: str) -> str:
    """bash expands ${IFS} / $IFS to whitespace (default space/tab/newline) BEFORE word
    splitting, so cat${IFS}/etc/shadow runs `cat /etc/shadow` in the child. Replace an IFS
    reference with a space so the scanner tokenizes the command bash actually executes."""
    if "IFS" not in command:
        return command
    return _IFS_RE.sub(" ", command)


def _rewrite_unquoted_newlines(command: str) -> str:
    """Rewrite only UNQUOTED newline runs to ` ; ` (a bash command separator). A newline INSIDE
    quotes is data (echo "ok\\nrm" is one argument), so a blanket regex would split a quoted
    multiline string into a spurious command position and mis-block the later line."""
    out = []
    q = None
    esc = False
    prev_nl = False
    for ch in command:
        if esc:
            if ch in ("\n", "\r"):
                # A backslash immediately before a newline is a bash LINE CONTINUATION: both are
                # removed before command lookup, so `tou\<nl>ch` runs `touch`. Drop the backslash
                # we already emitted and the newline so the joined word is tokenized (outside
                # single quotes; single-quoted text never sets esc, so it stays literal).
                if out and out[-1] == "\\":
                    out.pop()
                esc = False
                prev_nl = False
                continue
            out.append(ch)
            esc = False
            prev_nl = False
            continue
        if q == "'":
            out.append(ch)
            if ch == "'":
                q = None
            prev_nl = False
            continue
        if q == '"':
            out.append(ch)
            if ch == "\\":
                esc = True
            elif ch == '"':
                q = None
            prev_nl = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            prev_nl = False
            continue
        if ch in ("'", '"'):
            out.append(ch)
            q = ch
            prev_nl = False
            continue
        if ch in ("\r", "\n"):
            if not prev_nl:
                out.append(" ; ")
            prev_nl = True
            continue
        out.append(ch)
        prev_nl = False
    return "".join(out)


def _strip_bash_comments(command: str) -> str:
    """Remove bash ``#`` comments, respecting quotes / escapes. A ``#`` begins a comment only at a
    WORD BOUNDARY (start of string, or after unquoted whitespace / a metacharacter) and runs to the
    end of the PHYSICAL line; a ``#`` inside a word (``echo ok#``) or inside quotes is literal. Run
    BEFORE newline rewriting so each comment terminates at its real line break rather than a
    synthesized ``;`` separator, and pair it with ``lexer.commenters = ""`` so shlex (whose default
    ``#`` handling is not bash-accurate and fires mid-word) does not re-introduce the miss."""
    out = []
    q = None
    i = 0
    n = len(command)
    boundary = True  # the start of the string is a word boundary
    while i < n:
        ch = command[i]
        if q == "'":
            out.append(ch)
            if ch == "'":
                q = None
            boundary = False
            i += 1
            continue
        if q == '"':
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(command[i + 1])
                i += 2
                continue
            if ch == '"':
                q = None
            boundary = False
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(command[i + 1])
            boundary = False
            i += 2
            continue
        if ch in ("'", '"'):
            out.append(ch)
            q = ch
            boundary = False
            i += 1
            continue
        if ch == "#" and boundary:
            # A comment runs to the end of the physical line; drop it but KEEP the newline so it
            # still separates the following command.
            while i < n and command[i] not in ("\n", "\r"):
                i += 1
            continue
        out.append(ch)
        boundary = ch in (" ", "\t", "\n", "\r", ";", "&", "|", "(", ")", "<", ">")
        i += 1
    return "".join(out)


def _mask_quoted_separators(command: str) -> str:
    """Neutralize command-boundary characters that are DATA inside quotes (blank them to a
    space) so the regex command-position scan does not treat a quoted separator -- echo
    "ok\\nrm" or 'a;rm' -- as a fresh command word. Command substitution ($(...) / backticks)
    still runs inside DOUBLE quotes, so those are preserved; single-quoted text is fully
    literal. The result is used only for the boundary regex, not for tokenization."""
    out = []
    q = None
    esc = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue
        if q == "'":
            out.append(" " if ch in ";&|(\n\r`$" else ch)
            if ch == "'":
                q = None
            i += 1
            continue
        if q == '"':
            if ch == "\\":
                out.append(ch)
                esc = True
                i += 1
                continue
            if ch == '"':
                out.append(ch)
                q = None
                i += 1
                continue
            if ch == "$" and i + 1 < n and command[i + 1] == "(":
                out.append("$(")  # command substitution runs inside double quotes; keep it
                i += 2
                continue
            if ch == "`":
                out.append("`")
                i += 1
                continue
            out.append(" " if ch in ";&|(\n\r" else ch)
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue
        if ch in ("'", '"'):
            out.append(ch)
            q = ch
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _iter_unquoted_chars(s):
    """Yield (index, char) for every character OUTSIDE single / double quotes (a backslash
    escape and the char it escapes are skipped inside double quotes / unquoted text). Used to
    locate brace-expansion syntax that bash would act on, ignoring quoted braces."""
    q = None
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if q == "'":
            if ch == "'":
                q = None
            continue
        if q == '"':
            if ch == "\\":
                esc = True
            elif ch == '"':
                q = None
            continue
        if ch == "\\":
            esc = True
            yield i, ch
            continue
        if ch in ("'", '"'):
            q = ch
            continue
        yield i, ch


def _brace_first_comma_group(s):
    """Return (open, close) indices of the first UNQUOTED ``{...}`` that contains a top-level
    comma (the shape bash expands), else None. ``{}`` / ``${x}`` / ``{1..5}`` have no top-level
    comma and are left untouched, as are quoted braces."""
    idxset = {i: ch for i, ch in _iter_unquoted_chars(s)}
    for o, ch in list(idxset.items()):
        if ch != "{":
            continue
        depth = 0
        has_comma = False
        for i in range(o, len(s)):
            c = idxset.get(i)
            if c is None:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    if has_comma:
                        return o, i
                    break
            elif c == "," and depth == 1:
                has_comma = True
    return None


def _brace_split_top_commas(content):
    """Split a brace group's inner text on top-level (unnested, unquoted) commas."""
    parts = []
    cur = []
    depth = 0
    q = None
    esc = False
    for ch in content:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if q == "'":
            cur.append(ch)
            if ch == "'":
                q = None
            continue
        if q == '"':
            cur.append(ch)
            if ch == "\\":
                esc = True
            elif ch == '"':
                q = None
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if ch in ("'", '"'):
            cur.append(ch)
            q = ch
            continue
        if ch == "{":
            depth += 1
            cur.append(ch)
            continue
        if ch == "}":
            depth -= 1
            cur.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def _brace_expand_word(word, budget):
    """Recursively expand a single word's comma brace groups (bash-style, quote-aware,
    cartesian across multiple groups), returning the list of expansions. Bounded by budget."""
    grp = _brace_first_comma_group(word)
    if grp is None:
        return [word]
    o, c = grp
    pre, content, post = word[:o], word[o + 1 : c], word[c + 1 :]
    out = []
    for opt in _brace_split_top_commas(content):
        for opt_exp in _brace_expand_word(opt, budget):
            for post_exp in _brace_expand_word(post, budget):
                out.append(pre + opt_exp + post_exp)
                if len(out) >= budget[0]:
                    return out
    return out


def _split_words_unquoted_ws(s):
    """Split ``s`` into words on UNQUOTED space / tab; emit an unquoted newline as its own
    token so it survives as a command separator. Quotes and their contents stay intact."""
    words = []
    cur = []
    q = None
    esc = False
    for ch in s:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if q == "'":
            cur.append(ch)
            if ch == "'":
                q = None
            continue
        if q == '"':
            cur.append(ch)
            if ch == "\\":
                esc = True
            elif ch == '"':
                q = None
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if ch in ("'", '"'):
            cur.append(ch)
            q = ch
            continue
        if ch == "\n":
            if cur:
                words.append("".join(cur))
                cur = []
            words.append("\n")
            continue
        if ch in " \t":
            if cur:
                words.append("".join(cur))
                cur = []
            continue
        cur.append(ch)
    if cur:
        words.append("".join(cur))
    return words


def _expand_braces(command: str) -> str:
    """Model bash brace expansion (comma lists) before the block / read scans so a payload such
    as ``{touch,/tmp/escape}`` or ``{python3,-c} '...'`` is seen as the writer / interpreter bash
    would actually run, instead of a single opaque ``{...}`` token. Only unquoted groups with a
    top-level comma are expanded; ``{}`` (find -exec), ``${VAR}`` parameter expansion, numeric
    ``{1..5}`` sequences and quoted braces are left intact. Expansion is bounded to avoid blowup;
    if the bound is hit the (partial) expansion is still scanned."""
    if "{" not in command:
        return command
    budget = [4096]
    out = []
    for w in _split_words_unquoted_ws(command):
        if "{" in w and "}" in w and "," in w:
            out.extend(_brace_expand_word(w, budget))
        else:
            out.append(w)
        if len(out) >= 8192:
            break
    return " ".join(out)


def _find_blocked_commands(command: str) -> set[str]:
    """Detect blocked commands at shell command position only.

    A token is at command position if it is the first token, or follows a
    shell separator / brace-group opener / new-command keyword (`then`, `do`,
    etc.), or a command-prefix wrapper like `env` / `time` / `xargs` (next
    token is the real command). Tokens in argument position (`grep -r curl .`,
    `echo source the data`, `ls /usr/bin/curl`) pass through. Also scans
    `find ... -exec CMD` and recurses into bash -c / cmd /c.
    """
    blocked: set[str] = set()

    # Strip bash # comments FIRST (at their real physical-line boundaries), so a comment does not
    # swallow the ` ; ` synthesized from a following newline (echo ok #\nsed -i ...) and a mid-word
    # # (echo ok#; rm ...) is not mistaken by shlex for a comment. commenters is cleared below too.
    command = _strip_bash_comments(command)
    # Normalize bash ANSI-C ($'...') / locale ($"...") quoting first: shlex leaves
    # `$'touch'` as `$touch`, so a writer/interpreter hidden behind ANSI-C quoting would
    # never match the blocklist even though bash decodes and runs it. Then expand ${IFS} to
    # whitespace so a separator-obfuscated command (rm${IFS}-rf${IFS}/) is tokenized.
    command = _expand_ifs(_normalize_ansi_c_quotes(command))
    # bash treats an unquoted newline as a command separator, but shlex's whitespace_split
    # folds it into ordinary whitespace, so `echo ok\nsed -i ...` would read `sed` as an
    # argument of `echo` and miss the write. Rewrite UNQUOTED newlines to `;` so each line
    # starts a fresh command position; a newline inside quotes stays data (echo "ok\nrm" is one
    # argument), so it is not turned into a spurious `; rm` command position.
    command = _rewrite_unquoted_newlines(command)
    # bash performs brace expansion before command lookup, so `{touch,/tmp/x}` /
    # `{python3,-c} '...'` run the writer / interpreter even though the raw string has no
    # blocked token. Expand comma brace groups so the produced command words are scanned.
    command = _expand_braces(command)

    # punctuation_chars splits separators into their own tokens, so command
    # position is detected even in `echo done; rm -rf x` (no whitespace) or
    # quote-split names (`r''m` collapses to `rm` after `;`). Including `<` splits an INPUT
    # redirect / here-string glued to the command word (sh<<<'...', cat</etc/passwd) so the
    # shell / reader is still recognized. (`>` is left out so the regex-based output-redirect
    # scan keeps seeing `>&` as one operator.)
    try:
        if sys.platform == "win32":
            tokens = shlex.split(command, posix = False)
        else:
            lexer = shlex.shlex(command, posix = True, punctuation_chars = ";&|()`<")
            lexer.whitespace_split = True
            lexer.commenters = (
                ""  # bash comments are pre-stripped; shlex's # handling is not bash-accurate
            )
            tokens = list(lexer)
    except ValueError:
        tokens = command.split()

    def _token_basename(tok: str) -> str:
        # Strip glued-on meta-chars (`rm;`) so the basename still matches `rm`.
        tok = tok.strip(";&|()`{}")
        base = os.path.basename(tok).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        return base

    expect_command = True  # start of string is a command position
    prefix_pending = False  # last cmd-position token was a wrapper (env/time/xargs/...)
    prev_was_flag = False  # previous token (while a wrapper is pending) takes an operand
    cur_wrapper = None  # the active wrapper's basename (drives per-wrapper option arity)
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            expect_command = True
            prefix_pending = False
            prev_was_flag = False
            cur_wrapper = None
            continue
        if token in _SHELL_KEYWORDS_AS_SEP:
            # if / while / until / then / do / else / elif start a NEW command position ONLY when
            # they appear at command position (the compound-statement header: `if touch x; then`).
            # After a command word they are ordinary arguments -- bash does not run the next word as
            # a command in `echo if touch`, so only reset there. Real separators (; | && ...) above
            # always reset regardless of position.
            if expect_command:
                prefix_pending = False
                prev_was_flag = False
                cur_wrapper = None
            continue
        if token.startswith("-"):
            # Flags belong to the active command, but keep expect_command while a
            # wrapper prefix awaits its command. Only a flag that actually takes a SEPARATED
            # operand (env -u NAME) marks the next token as its value; a glued / no-operand
            # flag (stdbuf -oL sed, xargs -0 sed) does not, so the command that follows is
            # still analysed.
            if not prefix_pending:
                expect_command = False
            elif _wrapper_flag_takes_operand(cur_wrapper, token):
                prev_was_flag = True
            continue
        if not expect_command:
            continue
        # A leading `!` negates the pipeline exit status, but the following word is still the
        # command bash executes (`! touch x`, `! python3 -c ...`). Keep command position so the
        # real command is scanned, rather than mistaking `!` for the command and its command for
        # an argument.
        if token == "!":
            continue
        # FOO=bar assignment prefix; next non-assignment token is the command.
        if _ASSIGNMENT_RE.match(token):
            continue
        # Numeric wrapper arg: `timeout 1 cmd` / `nice -n 5 cmd`, plus GNU `timeout`
        # duration forms (`5m`, `0.5`, `2h`). Skipping it keeps prefix_pending so the
        # real command that follows is still analysed at command position; over-
        # accepting a numeric-looking token is safe (we only skip, never stop scanning),
        # whereas the old int-only check let `timeout 5m rm -rf /` slip through.
        if prefix_pending and _is_wrapper_numeric_arg(token):
            prev_was_flag = False
            continue
        base = _token_basename(token)
        # A wrapper's separated option ARGUMENT (`stdbuf -o L cmd`, `ionice -c 2 cmd`):
        # an operand right after a wrapper flag that is NOT itself a blocked command /
        # prefix is the flag's value, so skip it and keep scanning for the real command
        # instead of mistaking it for the command and stopping. If it IS a blocked
        # command / prefix it is treated as the command below (never miss `env -i rm`).
        if (
            prefix_pending
            and prev_was_flag
            and base not in _BLOCKED_COMMANDS
            and base not in _COMMAND_PREFIXES
        ):
            prev_was_flag = False
            continue
        prev_was_flag = False
        # An expansion sitting AT the resolved command word -- behind a wrapper
        # (env $CMD -c ...) or after a leading assignment -- runs whatever it expands to as
        # the command name and cannot be proven safe, so fail closed. The separator-anchored
        # regex below misses the wrapper case because $CMD is not right after a separator.
        if "$" in token or "`" in token:
            blocked.add("command-expansion")
        # Glob metacharacters in a command NAME (/bin/s?, touc?, /bin/[bd]ash) are expanded by
        # the shell to a matching path BEFORE command lookup, so the literal basename compared
        # against the blocklist (s?, touc?) never matches the shell / writer it resolves to.
        # The resolved binary cannot be proven safe, so fail closed. A bare `[` is the test
        # builtin (not a glob), so exclude it.
        if "*" in token or "?" in token or (token != "[" and "[" in token):
            blocked.add("command-glob")
        if base in _BLOCKED_COMMANDS or _is_versioned_interpreter(base):
            blocked.add(base)
        # The `.` builtin is bash's `source`: `. evil.sh` runs an unscanned script in the
        # shell, the same escape as `source`, but its basename is not a blocklist word.
        if base == ".":
            blocked.add("source")
        # An explicit path to a LOCAL executable at command position (./evil, subdir/tool) runs
        # whatever its shebang names in an unguarded child, so treat it like a blocked command.
        if _is_local_executable_path(token):
            blocked.add("local-exec:" + base)
        # Wrappers (env/time/xargs/sudo) consume one command; the next non-flag,
        # non-numeric token is the real command. sudo is also in _BLOCKED_COMMANDS.
        if base in _COMMAND_PREFIXES:
            prefix_pending = True
            cur_wrapper = base
            continue
        expect_command = False
        prefix_pending = False
        cur_wrapper = None

    # `find ... -exec CMD ... ;` / `-execdir CMD ... +` invoke CMD directly. CMD may
    # itself be a wrapper (`env rm`, `timeout 5 rm`) or a nested shell (`sh -c '...'`),
    # so rescan the whole slice up to the `;`/`+` terminator through the full command-
    # position analyzer instead of only basename-matching the immediate next token.
    for i, tok in enumerate(tokens):
        if tok in _FIND_EXEC_FLAGS:
            seg = []
            j = i + 1
            while j < len(tokens) and tokens[j] not in _FIND_EXEC_TERMINATORS:
                seg.append(tokens[j])
                j += 1
            if seg:
                # find substitutes `{}` with each matched path, so `-exec {} ;` EXECUTES the matched
                # file -- a prior step can plant an executable ./evil and `find . -name evil -exec {}
                # ';'` then runs that workdir shebang in an unguarded child. The reconstructed
                # segment scan sees only the harmless-looking `{}`, so fail closed when the exec
                # command word itself is (or starts with) the placeholder.
                _ecw = seg[0]
                if len(_ecw) >= 2 and _ecw[0] == _ecw[-1] and _ecw[0] in ("'", '"'):
                    _ecw = _ecw[1:-1]
                if _ecw == "{}" or _ecw.startswith("{}"):
                    blocked.add("find-exec-placeholder")
                blocked |= _find_blocked_commands(" ".join(seg))

    # Regex catches blocked words at command boundaries shlex misses: inside
    # $(rm -rf), <(rm), backtick chains, or "foo;rm". Anchored to command-position
    # delimiters, so it doesn't match in argument position. Quoted separators are neutralized
    # first so a quoted multiline string (echo "ok\nrm") is not read as a command boundary.
    lowered = _mask_quoted_separators(command).lower()
    if _BLOCKED_COMMANDS:
        words_alt = "|".join(re.escape(w) for w in sorted(_BLOCKED_COMMANDS))
        pattern = (
            rf"(?:^|[;&|`\n(]\s*|[$]\(\s*|<\(\s*)"
            rf"(?:[\w./\\-]*/|[a-zA-Z]:[/\\][\w./\\-]*)?"
            rf"({words_alt})(?:\.(?:exe|com|bat|cmd))?\b"
        )
        blocked.update(re.findall(pattern, lowered))

    # Nested shell invocations (bash -c '...', bash -lc '...', cmd /c '...'):
    # on a -c/-/c flag, look back for a shell name (skipping flags) and
    # recursively scan the nested command string.
    _SHELLS = _SHELL_BINARIES
    _SHELLS_WIN = {"cmd", "cmd.exe"}
    for i, token in enumerate(tokens):
        tok_lower = token.lower()
        # Match -c exactly, or combined flags ending in c (e.g. -lc, -xc)
        is_unix_c = tok_lower == "-c" or (
            tok_lower.startswith("-") and tok_lower.endswith("c") and not tok_lower.startswith("--")
        )
        is_win_c = tok_lower == "/c"
        if not (is_unix_c or is_win_c) or i < 1 or i + 1 >= len(tokens):
            continue
        # Look back past flags for the shell binary. Windows flags and absolute
        # paths both start with /, so only skip short /X flags (not /bin/bash).
        for j in range(i - 1, -1, -1):
            prev = tokens[j]
            if prev.startswith("-"):
                continue  # skip Unix flags like --login, -l
            if is_win_c and prev.startswith("/") and len(prev) <= 3:
                continue  # skip Windows flags like /s, /q (not /bin/bash)
            prev_base = os.path.basename(prev).lower()
            if is_unix_c and prev_base in _SHELLS:
                blocked |= _find_blocked_commands(tokens[i + 1])
            elif is_win_c and prev_base in _SHELLS_WIN:
                blocked |= _find_blocked_commands(tokens[i + 1])
            break  # stop at first non-flag token

    # `env -S 'cmd ...'` / `env --split-string='cmd'` splits the string and runs it as a
    # fresh command, so a bare `env -S` operand is NOT just a flag value -- recurse into
    # it (it can invoke an unguarded interpreter or another blocked command).
    for i, token in enumerate(tokens):
        tl = token.lower()
        payload = None
        if tl in ("-s", "--split-string") and i + 1 < len(tokens):
            payload = tokens[i + 1]
        elif tl.startswith("-s") and tl != "-s" and not tl.startswith("--"):
            payload = token[2:]  # glued short form: env -S'cmd' / -Scmd
        elif tl.startswith("--split-string="):
            payload = token[len("--split-string=") :]
        if not payload:
            continue
        for j in range(i - 1, -1, -1):
            prev = tokens[j]
            if prev.startswith("-"):
                continue
            if os.path.basename(prev).lower() == "env":
                blocked |= _find_blocked_commands(payload)
            break

    def _command_word_indices():
        # Indices of the REAL command word at each command position, skipping FOO=bar
        # assignments and wrapper prefixes (env / nice / timeout / xargs / ...) plus their
        # numeric / separated-option arguments, so `env sed`, `timeout 5 bash` resolve to
        # sed / bash. Mirrors the main command-position scan above.
        out = []
        expect = True
        pending = False
        prev_flag = False
        wrapper = None
        for _i, _tok in enumerate(tokens):
            if _tok in _SHELL_SEPARATORS:
                expect = True
                pending = False
                prev_flag = False
                wrapper = None
                continue
            if _tok in _SHELL_KEYWORDS_AS_SEP:
                # if / while / until / then / do (etc.) begin a new command position ONLY at
                # command position (the compound-statement header); after a command word they are
                # ordinary arguments, so `echo if sed -i ...` must not record sed as a command.
                # Mirrors the round-44 fix in the main scanner above.
                if expect:
                    pending = False
                    prev_flag = False
                    wrapper = None
                continue
            if _tok.startswith("-"):
                if not pending:
                    expect = False
                elif _wrapper_flag_takes_operand(wrapper, _tok):
                    prev_flag = True
                continue
            if not expect:
                continue
            if _tok == "!":
                continue  # pipeline negation keeps command position (! bash s.sh)
            if _ASSIGNMENT_RE.match(_tok):
                continue
            if pending and _is_wrapper_numeric_arg(_tok):
                prev_flag = False
                continue
            _base = _token_basename(_tok)
            if (
                pending
                and prev_flag
                and _base not in _BLOCKED_COMMANDS
                and _base not in _COMMAND_PREFIXES
            ):
                prev_flag = False
                continue
            prev_flag = False
            if _base in _COMMAND_PREFIXES:
                pending = True
                wrapper = _base
                continue
            out.append(_i)
            expect = False
            pending = False
            wrapper = None
        return out

    _cmd_word_idx = _command_word_indices()

    def _wrapper_prefix_indices():
        # Indices where a _COMMAND_PREFIXES wrapper (env / xargs / watch / ...) sits AT command
        # position. _command_word_indices SKIPS these (it records the RESOLVED command), but the
        # watch / xargs handlers below key off the wrapper token itself, so track them here with
        # the same command-position rules -- so `echo watch rm` (watch in ARGUMENT position) is
        # not mistaken for a wrapper.
        out = []
        expect = True
        pending = False
        prev_flag = False
        wrapper = None
        for _i, _tok in enumerate(tokens):
            if _tok in _SHELL_SEPARATORS:
                expect = True
                pending = False
                prev_flag = False
                wrapper = None
                continue
            if _tok in _SHELL_KEYWORDS_AS_SEP:
                if expect:
                    pending = False
                    prev_flag = False
                    wrapper = None
                continue
            if _tok.startswith("-"):
                if not pending:
                    expect = False
                elif _wrapper_flag_takes_operand(wrapper, _tok):
                    prev_flag = True
                continue
            if not expect:
                continue
            if _tok == "!":
                continue
            if _ASSIGNMENT_RE.match(_tok):
                continue
            if pending and _is_wrapper_numeric_arg(_tok):
                prev_flag = False
                continue
            _base = _token_basename(_tok)
            if (
                pending
                and prev_flag
                and _base not in _BLOCKED_COMMANDS
                and _base not in _COMMAND_PREFIXES
            ):
                prev_flag = False
                continue
            prev_flag = False
            if _base in _COMMAND_PREFIXES:
                out.append(_i)
                pending = True
                wrapper = _base
                continue
            expect = False
            pending = False
            wrapper = None
        return out

    _wrapper_prefix_idx = _wrapper_prefix_indices()

    # trap 'CMD' SIGSPEC registers CMD to run (in the unguarded shell) on EXIT / a signal, so
    # the quoted handler is unscanned shell code. Scan the handler operand of a command-position
    # `trap` recursively; a reset (trap - EXIT) / ignore (trap '' EXIT) has nothing to run.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "trap":
            continue
        # Skip trap options / the -- terminator (trap -- 'CMD' EXIT, trap -p) so the handler
        # operand is not mistaken for -- and left unscanned.
        _j = i + 1
        while _j < len(tokens) and tokens[_j].startswith("-") and len(tokens[_j]) > 1:
            _j += 1
        if _j >= len(tokens):
            continue
        _h = tokens[_j]
        if _h and _h != "-" and _h not in _SHELL_SEPARATORS and _h not in _SHELL_KEYWORDS_AS_SEP:
            blocked |= _find_blocked_commands(_h)

    # A shell binary invoked with a SCRIPT FILE (`bash s.sh`) or `-s` (read the script from
    # stdin) runs unscanned shell code in the same unguarded environment; only the inline
    # `-c '...'` form is statically analyzable (handled above). Block a command-position
    # shell whose operands include a non-flag argument (the script) and no -c/-lc flag. Using
    # the wrapper-aware command-word indices so `env bash s.sh` / `timeout 5 bash s.sh` are
    # not hidden behind the wrapper prefix.
    for i in _cmd_word_idx:
        tok = tokens[i]
        if os.path.basename(tok).lower() not in _SHELLS:
            continue
        _has_c = False
        _script = None
        _interactive = False
        for k in range(i + 1, len(tokens)):
            t = tokens[k]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            tl = t.lower()
            # An interactive shell (bash -i, sh -i, or a combined short flag like -ic) SOURCES
            # the user's rc files (.bashrc / ENV) before running any -c payload, executing
            # unscanned workdir startup code in the unguarded child. Treat -i as unscanned
            # startup like BASH_ENV.
            if tl.startswith("-") and not tl.startswith("--") and "i" in tl[1:]:
                _interactive = True
            if tl == "-c" or (tl.startswith("-") and not tl.startswith("--") and tl.endswith("c")):
                _has_c = True
                break
            if tl in ("-s", "--"):  # -s reads the script from stdin (unscanned)
                _script = t
                break
            if t.startswith("-"):
                continue  # other shell flags: -l, -x, --login, --norc, ...
            _script = t  # first non-flag operand is the script file
            break
        if _interactive:
            blocked.add("shell-interactive-rc:" + _token_basename(tok))
        # Any command-position shell WITHOUT an inline `-c` payload runs unscanned code:
        # a script file (bash s.sh), stdin via -s, or a bare shell that reads stdin
        # (`printf 'evil' | bash`). Only the `-c '...'` form is statically analyzable, so
        # block everything else.
        if not _has_c:
            blocked.add("shell-script:" + (_script or _token_basename(tok)))
        # BASH_ENV=script / ENV=script assignment prefix before a shell makes bash / sh SOURCE
        # that workdir file before the scanned -c payload runs, executing unscanned commands in
        # the unguarded child (BASH_ENV=env.sh bash -c 'echo ok', env BASH_ENV=env.sh bash -c).
        # Scan the command segment before this shell word for a non-empty startup-env assignment.
        for k in range(i - 1, -1, -1):
            pk = tokens[k]
            if pk in _SHELL_SEPARATORS or pk in _SHELL_KEYWORDS_AS_SEP:
                break
            if _ASSIGNMENT_RE.match(pk):
                _an, _, _av = pk.partition("=")
                if _an in ("BASH_ENV", "ENV") and _av != "":
                    blocked.add("shell-startup-env:" + _an)

    # A NAME=value token is an ENVIRONMENT assignment only in the command-PREFIX position (before
    # the command word of its segment); after the command word it is an ARGUMENT the shell does not
    # export (echo GIT_CONFIG_COUNT=0, printf %s PATH=.:/bin). Map each leading assignment token to
    # its segment's command-word index (None if the segment is pure assignments).
    def _assignment_prefix_map():
        _cmd_sorted = sorted(_cmd_word_idx)
        _bounds = []
        _seg_start = 0
        for _j in range(len(tokens) + 1):
            if (
                _j == len(tokens)
                or tokens[_j] in _SHELL_SEPARATORS
                or tokens[_j] in _SHELL_KEYWORDS_AS_SEP
            ):
                if _j > _seg_start:
                    _bounds.append((_seg_start, _j))
                _seg_start = _j + 1
        _out = {}
        for _a, _b in _bounds:
            _cw = None
            for _w in _cmd_sorted:
                if _a <= _w < _b:
                    _cw = _w
                    break
            # `export NAME=value` / `declare -x` / `typeset` set an env var even though NAME=value
            # follows the command word, so their NAME=value ARGS are assignments too.
            _exporter = _cw is not None and _token_basename(tokens[_cw]) in (
                "export",
                "declare",
                "typeset",
            )
            for _j in range(_a, _b):
                if _ASSIGNMENT_RE.match(tokens[_j]) and (_cw is None or _j < _cw or _exporter):
                    _out[_j] = _cw
        return _out

    _assign_prefix = _assignment_prefix_map()

    # Local VAR=value bindings in this command, so a PATH component expanded from a locally-set
    # variable (P=.; PATH=$P evil) can be resolved to its (unsafe) value. Only real prefix
    # assignments count (not a NAME=value printed as an argument).
    _local_assigns = {}
    for _ei in _assign_prefix:
        _n, _, _v = tokens[_ei].partition("=")
        _local_assigns[_n.rstrip("+")] = _v

    # Assignment prefixes that persist for the command's child: a non-empty BASH_ENV / ENV (sourced
    # by a later shell), a PATH with a cwd entry (a bare command resolves to a workdir shebang), and
    # git path / config environment variables -- GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE /
    # GIT_OBJECT_DIRECTORY / GIT_COMMON_DIR point git's repo / objects outside the workdir, and
    # GIT_CONFIG_* override the sandbox's env-based hook suppression. Handle NAME=value / NAME+=value.
    _GIT_EXEC_ENV_VARS = frozenset(
        {
            "GIT_EXTERNAL_DIFF",
            "GIT_ASKPASS",
            "GIT_SSH",
            "GIT_SSH_COMMAND",
            "GIT_PROXY_COMMAND",
            "GIT_EDITOR",
            "GIT_SEQUENCE_EDITOR",
            "GIT_PAGER",
        }
    )
    # git path-valued repository env vars whose escaping value writes outside the workdir from an
    # unguarded git child (GIT_OBJECT_DIRECTORY=/tmp git hash-object -w --stdin).
    _GIT_PATH_ENV_VARS = frozenset(
        {
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_COMMON_DIR",
        }
    )
    for _ei, _cwidx in _assign_prefix.items():
        _et = tokens[_ei]
        _an, _, _av = _et.partition("=")
        _append = _an.endswith("+")
        _an = _an.rstrip("+")
        _cmd_base = _token_basename(tokens[_cwidx]) if _cwidx is not None else None
        _cmd_is_git = _cmd_base == "git"
        _is_exporter = _cmd_base in ("export", "declare", "typeset")
        if _an in ("BASH_ENV", "ENV") and _av != "":
            blocked.add("shell-startup-env:" + _an)
        # PATH=. cmd / PATH+=:. cmd: a relative / cwd entry lets a bare command word resolve to a
        # workdir shebang. For += the value is APPENDED to the existing PATH, so evaluate
        # "$PATH" + value (a trailing / doubled separator or . entry is then the unsafe one).
        elif _an == "PATH":
            _pval = ("$PATH" + _av) if _append else _av
            # PATH=$(pwd) / PATH=/x:$(cmd): a command substitution in the value is a DYNAMIC search
            # path (it can point at the cwd where an earlier step planted an exe). Tokenization
            # splits `$(` into a trailing `$` on this token and a following `(`, so detect that
            # shape here; a backtick form leaves an empty value token which _path_value_is_unsafe
            # already flags.
            _pathsub = _av.endswith("$") and _ei + 1 < len(tokens) and tokens[_ei + 1] == "("
            if _pathsub or _path_value_is_unsafe(_pval, _local_assigns):
                blocked.add("unsafe-path-assign")
        # GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE / GIT_OBJECT_DIRECTORY / ... set git's repo /
        # tree / index / object-store path directly, so an escaping value writes outside the workdir
        # (GIT_DIR=/tmp/x git init, GIT_OBJECT_DIRECTORY=/tmp git hash-object -w) with no CLI flag.
        elif _an in _GIT_PATH_ENV_VARS and _arg_escapes_workdir(_av):
            blocked.add("git-write-outside")
        # GIT_CONFIG[_GLOBAL/_SYSTEM/_COUNT/_KEY_*/_VALUE_*] re-point git config or drop the
        # sandbox's env-based hook suppression (GIT_CONFIG_COUNT=0 git ...), re-enabling a planted
        # .git/hooks/* in an unguarded git child.
        elif _an == "GIT_CONFIG" or _an.startswith("GIT_CONFIG_"):
            blocked.add("git-config-env-override")
        # git runs the program named by these env vars (GIT_EXTERNAL_DIFF / GIT_ASKPASS /
        # GIT_SSH[_COMMAND] / GIT_PROXY_COMMAND / GIT_EDITOR / GIT_PAGER), and for a git child the
        # standard EDITOR / VISUAL fallbacks name the commit-message editor too. Block a value that
        # points at a WORKDIR executable (GIT_EXTERNAL_DIFF=./evil), a ~ path, OR whose command the
        # scanner flags (GIT_EXTERNAL_DIFF='touch /tmp/p' -> touch writes outside). A bare system
        # command (GIT_PAGER=cat, EDITOR=vim) stays allowed.
        elif _an in _GIT_EXEC_ENV_VARS or (
            _an in ("EDITOR", "VISUAL") and (_cmd_is_git or _is_exporter)
        ):
            _gev = _av
            if len(_gev) >= 2 and _gev[0] == _gev[-1] and _gev[0] in ("'", '"'):
                _gev = _gev[1:-1]
            _gecmd = _gev.split()[0] if _gev.split() else ""
            if _is_local_executable_path(_gecmd) or _gecmd.startswith("~"):
                blocked.add("git-exec-env")
            elif _gev and _find_blocked_commands(_gev):
                blocked.add("git-exec-env")

    # git -c alias.X='!CMD' X / git config alias.X '!CMD': a git alias whose value starts with
    # `!` runs CMD through an unguarded shell, but the scanner sees only `git`. Flag the shell-
    # dispatch alias form (the ! marker) so the aliased writer / reader is not smuggled past.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "git":
            continue
        # An env -C DIR / --chdir DIR wrapper BEFORE git changes git's cwd, so even a bare or
        # relative write subcommand (env -C /tmp git init) resolves under DIR. Scan back to the
        # previous separator for such a wrapper; if DIR escapes the workdir, git operates outside.
        _git_cwd_escapes = False
        _env_suppress_dropped = False
        _seg_has_env = False
        for _bk in range(i - 1, -1, -1):
            _bt = tokens[_bk]
            if _bt in _SHELL_SEPARATORS or _bt in _SHELL_KEYWORDS_AS_SEP:
                break
            if _bt in ("-C", "--chdir") and _bk + 1 < len(tokens):
                if _arg_escapes_workdir(tokens[_bk + 1]):
                    _git_cwd_escapes = True
            elif _bt.startswith("--chdir=") and _arg_escapes_workdir(_bt.split("=", 1)[1]):
                _git_cwd_escapes = True
            # GNU env glues the short chdir operand directly onto the flag (env -C/tmp git init),
            # which the separated / --chdir= forms above miss. Only -C takes a dir here.
            elif _bt.startswith("-C") and len(_bt) > 2 and _arg_escapes_workdir(_bt[2:]):
                _git_cwd_escapes = True
            # env -i / --ignore-environment / a bare `-` start git with an EMPTY environment, and
            # env -u NAME / --unset NAME / --unset=NAME strip just the suppression var; either
            # removes the injected core.hooksPath suppression so a planted .git/hooks/* runs in
            # the unguarded git child. Handle the separated and glued long forms and the bare `-`.
            if _bt in ("-i", "--ignore-environment", "-"):
                _env_suppress_dropped = True
            elif (
                _bt in ("-u", "--unset")
                and _bk + 1 < len(tokens)
                and tokens[_bk + 1].startswith("GIT_CONFIG")
            ):
                _env_suppress_dropped = True
            elif _bt.startswith("--unset=") and _bt.split("=", 1)[1].startswith("GIT_CONFIG"):
                _env_suppress_dropped = True
            # GNU env glues the short unset operand onto the flag (env -uGIT_CONFIG git ...).
            elif _bt.startswith("-u") and len(_bt) > 2 and _bt[2:].startswith("GIT_CONFIG"):
                _env_suppress_dropped = True
            elif _token_basename(_bt) == "env":
                _seg_has_env = True
        if _git_cwd_escapes:
            blocked.add("git-write-outside")
        if _env_suppress_dropped and _seg_has_env:
            blocked.add("git-config-env-override")
        _seg = []
        for k in range(i + 1, len(tokens)):
            if tokens[k] in _SHELL_SEPARATORS or tokens[k] in _SHELL_KEYWORDS_AS_SEP:
                # A command substitution ( `...` / $(...) ) used as a git operand (git init
                # `printf /tmp/x` / git worktree add $(pwd)/out) is split by tokenization into
                # separator tokens; re-inject a backtick marker so the operand scan flags it as a
                # dynamic escaping path. A backtick starts one directly; `$(` leaves a trailing `$`.
                if tokens[k] == "`" or (tokens[k] == "(" and _seg and _seg[-1].endswith("$")):
                    _seg.append("`")
                break
            _seg.append(tokens[k])
        _joined = " ".join(_seg)
        if re.search(r"alias\.[^=\s]+=\s*!", _joined):
            blocked.add("git-shell-alias")
        else:
            for _k, _t in enumerate(_seg):
                if _t.startswith("alias.") and _k + 1 < len(_seg) and _seg[_k + 1].startswith("!"):
                    blocked.add("git-shell-alias")
                    break
        # git init /tmp/x, git clone url /tmp/x, git worktree add /tmp/x, git -C /outside ...
        # all create / operate on files outside the workdir in an unguarded native git child.
        # Flag a path OPERAND (bare, non-flag) or a -C / --git-dir / --work-tree value that
        # escapes the workdir. Workdir-relative git usage (git init, git clone url, git -C sub)
        # and non-path operands (a clone URL, a config name=value) stay allowed.
        _gk = 0
        while _gk < len(_seg):
            _gt = _seg[_gk]
            # git -c KEY=VALUE: an execution-capable config (core.fsmonitor / sshCommand / ...)
            # runs VALUE in an unguarded child; core.hooksPath / init.templateDir re-enable
            # planted hooks. Block the exec-capable configs (alias.*=! handled above).
            if _gt == "-c" and _gk + 1 < len(_seg):
                if _git_config_key_is_exec(_seg[_gk + 1].split("=", 1)[0]):
                    blocked.add("git-exec-config")
                _gk += 2
                continue
            # git --exec-path=<path> re-points where git looks for its git-<cmd> helpers, so
            # `git --exec-path=. evil` runs a workdir git-evil in an unguarded child. Any value
            # redirects the core path (the no-value form just prints it), so flag it.
            if _gt.startswith("--exec-path=") and _gt.split("=", 1)[1]:
                blocked.add("git-exec-config")
                _gk += 1
                continue
            # git --config-env=KEY=ENVVAR sets a config KEY from an env var, so an execution-
            # capable / alias KEY (git --config-env=alias.x=P with P='!cmd') runs a command.
            if _gt.startswith("--config-env="):
                _cekey = _gt.split("=", 1)[1].split("=", 1)[0]
                if _git_config_key_is_exec(_cekey) or _cekey.startswith("alias."):
                    blocked.add("git-exec-config")
                _gk += 1
                continue
            if _gt in _GIT_PATH_VALUE_OPTIONS and _gk + 1 < len(_seg):
                if _git_operand_escapes(_seg[_gk + 1], _local_assigns):
                    blocked.add("git-write-outside")
                _gk += 2
                continue
            # Stuck short form: git archive -o/tmp/x (and -O.. / -C/outside) glue the path value
            # directly onto the short option with no space, which the separated / --opt=val scans
            # above miss. Only the path-valued SHORT options take a glued value; a non-escaping
            # value (-oout.tar, -C90 for find-copies) is left alone by _git_operand_escapes.
            if len(_gt) > 2 and _gt[:2] in ("-C", "-o", "-O"):
                if _git_operand_escapes(_gt[2:], _local_assigns):
                    blocked.add("git-write-outside")
                _gk += 1
                continue
            _oeq = None
            for _opt in _GIT_PATH_VALUE_OPTIONS:
                if _gt.startswith(_opt + "="):
                    _oeq = _gt.split("=", 1)[1]
                    break
            if _oeq is not None:
                if _git_operand_escapes(_oeq, _local_assigns):
                    blocked.add("git-write-outside")
            elif not _gt.startswith("-") and _git_operand_escapes(_gt, _local_assigns):
                blocked.add("git-write-outside")
            _gk += 1
        # git apply --unsafe-paths lets a patch write to targets OUTSIDE the working tree (a
        # +++ ../../tmp/x hunk), which the native git child applies with no realpath guard. The
        # patch body is not statically visible, so deny the unsafe mode outright; a plain
        # git apply p.patch (in-tree targets) stays allowed.
        if "apply" in _seg and "--unsafe-paths" in _seg:
            blocked.add("git-write-outside")
        # git config [options] KEY [VALUE]: setting an execution-capable config key (git config
        # core.pager 'sh -c ...') runs its value on later git operations, like the -c form; and
        # git config --file <path> / -f <path> writes the config to an arbitrary file, escaping
        # the workdir (git config --file=/tmp/gitcfg ...).
        for _ci, _ct in enumerate(_seg):
            if _ct == "config":
                _cj = _ci + 1
                # --system / --global select the host system / user config file (/etc/gitconfig,
                # ~/.gitconfig), both OUTSIDE the workdir. A WRITE there (KEY VALUE, or a write
                # flag / --edit) escapes the sandbox; a pure read (--get* / --list / -l / a bare
                # KEY) does not, so only writes are blocked.
                _host_scope = False
                _write_flag = False
                while _cj < len(_seg):
                    _cw = _seg[_cj]
                    if _cw in ("--file", "-f") and _cj + 1 < len(_seg):
                        if _arg_escapes_workdir(_seg[_cj + 1]):
                            blocked.add("git-write-outside")
                        _cj += 2
                        continue
                    if _cw.startswith("--file="):
                        if _arg_escapes_workdir(_cw.split("=", 1)[1]):
                            blocked.add("git-write-outside")
                        _cj += 1
                        continue
                    if _cw in ("--system", "--global"):
                        _host_scope = True
                        _cj += 1
                        continue
                    if _cw in (
                        "--add",
                        "--unset",
                        "--unset-all",
                        "--replace-all",
                        "--remove-section",
                        "--rename-section",
                        "-e",
                        "--edit",
                    ):
                        _write_flag = True
                        _cj += 1
                        continue
                    if not _cw.startswith("-"):
                        if _git_config_key_is_exec(_cw.split("=", 1)[0]):
                            blocked.add("git-exec-config")
                        # A host-scope write: an explicit write flag, or a KEY followed by a VALUE
                        # operand (git config --global user.name x). A bare KEY read is left alone.
                        if _host_scope and (
                            _write_flag
                            or (_cj + 1 < len(_seg) and not _seg[_cj + 1].startswith("-"))
                        ):
                            blocked.add("git-write-outside")
                        break
                    _cj += 1
                if _host_scope and _write_flag:
                    blocked.add("git-write-outside")  # --global --edit / --unset with no inline KEY
                break

    # hash -p PATHNAME NAME binds the command NAME to PATHNAME in the shell's hash table, so a
    # later bare `NAME` runs PATHNAME. With a local executable (hash -p ./evil ls; ls) that
    # launches an unguarded workdir shebang under a benign-looking command word. Block hash -p
    # when its pathname operand is a local executable path.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "hash":
            continue
        for k in range(i + 1, len(tokens)):
            t = tokens[k]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            if t == "-p" and k + 1 < len(tokens) and _is_local_executable_path(tokens[k + 1]):
                blocked.add("hash-p-local-exec")
                break

    # alias x='touch /tmp/p'; ...; x (with expand_aliases) runs the alias BODY at execution
    # time, but the command word `x` is unknown to the scanner. Scan the body of each alias
    # definition so a blocked writer / interpreter in it is caught at the definition site.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "alias":
            continue
        for k in range(i + 1, len(tokens)):
            t = tokens[k]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            if t.startswith("-"):
                continue  # alias -p (print)
            if "=" in t:
                _body = t.split("=", 1)[1]
                if _body:
                    blocked |= _find_blocked_commands(_body)

    # openssl <subcmd> ... -out FILE writes FILE in an unguarded openssl child (openssl rand
    # -out /tmp/p 4), which the realpath guard never sees. Block when an output-file flag names a
    # path that escapes the workdir; a workdir-local -out (openssl rand -out key.bin) and the
    # no-output forms (openssl rand -hex 16, openssl dgst file) stay allowed.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "openssl":
            continue
        # An env -C DIR / subprocess cwd= (reconstructed as env -C DIR) that escapes the workdir
        # makes even a RELATIVE -out operand (openssl rand -out key, cwd=/tmp) land outside.
        _ossl_cwd_escapes = _cwd_wrapper_escapes(tokens, i)
        for k in range(i + 1, len(tokens)):
            t = tokens[k]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            _op = None
            if t in _OPENSSL_WRITE_FLAGS and k + 1 < len(tokens):
                _op = tokens[k + 1]  # separated form: -out FILE
            else:
                # glued form: -out=FILE / -writerand=FILE (openssl accepts -out outfile and =).
                _oflag, _oeq, _oval = t.partition("=")
                if _oeq and _oflag in _OPENSSL_WRITE_FLAGS:
                    _op = _oval
            if _op is not None and (
                _git_operand_escapes(_op, _local_assigns)
                or (_ossl_cwd_escapes and _operand_relative_local(_op))
            ):
                blocked.add("openssl-write-outside")

    # iconv -o FILE / --output FILE / --output=FILE / -oFILE writes FILE in an unguarded iconv
    # child. Block when the output path escapes the workdir; a workdir-local -o and the no-output
    # forms (iconv -f utf8 -t utf16 file, printing to stdout) stay allowed.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "iconv":
            continue
        _ic_cwd_escapes = _cwd_wrapper_escapes(tokens, i)
        for k in range(i + 1, len(tokens)):
            t = tokens[k]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            _op = None
            if t in _ICONV_WRITE_FLAGS and k + 1 < len(tokens):
                _op = tokens[k + 1]  # separated form: -o FILE / --output FILE
            elif t.startswith("--output="):
                _op = t[len("--output=") :]
            elif t.startswith("-o") and len(t) > 2:
                _op = t[2:]  # glued short form: -oFILE
            if _op is not None and (
                _git_operand_escapes(_op, _local_assigns)
                or (_ic_cwd_escapes and _operand_relative_local(_op))
            ):
                blocked.add("iconv-write-outside")

    # sqlite3 <DBFILE> creates / opens a database in an unguarded child (no realpath guard), and
    # its dot-commands (.output / .backup / .dump / .read ...) read + write arbitrary files. Flag
    # a DBFILE operand that escapes the workdir, and any dot-file target that escapes. A local DB
    # (sqlite3 local.db 'create ...'), :memory:, and an in-memory URI carry no escape and stay
    # allowed. -init / -cmd option values are option operands, not the DBFILE.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "sqlite3":
            continue
        # env -C DIR / subprocess cwd= (reconstructed as env -C DIR) that escapes the workdir makes
        # even a RELATIVE DBFILE / dot-file / -init operand (sqlite3 db.sqlite ..., cwd=/tmp) land
        # outside; combine the escaping cwd with a relative operand below.
        _sqlite_cwd_escapes = _cwd_wrapper_escapes(tokens, i)
        # sqlite3 [OPTIONS] [FILENAME [SQL]] reads SQL from STDIN when no SQL argv is given, so a
        # dot-command fed via a pipe or `<` redirect (printf '.shell touch /tmp/p\n' | sqlite3
        # :memory:) runs unscanned in the unguarded child. Detect a stdin source (this command is a
        # pipe target, or has a `<` / heredoc input redirect) with no inline SQL and fail closed.
        _sqlite_pipe_target = False
        for _bk in range(i - 1, -1, -1):
            _bt = tokens[_bk]
            if _bt in _SHELL_SEPARATORS or _bt in _SHELL_KEYWORDS_AS_SEP:
                _sqlite_pipe_target = _bt == "|"
                break
        _sqlite_stdin_redirect = False
        _seen_sql = False
        _seen_db = False
        _sk = i + 1
        while _sk < len(tokens):
            t = tokens[_sk]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            if t in ("<", "<<", "<<<", "0<"):
                _sqlite_stdin_redirect = True
                _sk += 2  # skip the redirect target too
                continue
            # sqlite3 options that consume a SEPARATED operand; skip the value so it is not
            # mistaken for the DBFILE (only -init reads a file, checked via its own value here).
            if t in _SQLITE_OPERAND_OPTS:
                if t == "-init" and _sk + 1 < len(tokens):
                    _iv = tokens[_sk + 1]
                    if _git_operand_escapes(_iv, _local_assigns) or (
                        _sqlite_cwd_escapes and _operand_relative_local(_iv)
                    ):
                        blocked.add("sqlite3-write-outside")
                _sk += 2
                continue
            # Any dot-command file target that escapes the workdir (.output /tmp/leak, .backup
            # ../x, .read $P) writes / reads a host path; scan the (possibly quoted, multi-line
            # SQL) operand for one.
            _unq = t
            if len(_unq) >= 2 and _unq[0] == _unq[-1] and _unq[0] in ("'", '"'):
                _unq = _unq[1:-1]
            # .shell CMD / .system CMD run an arbitrary command in the unguarded child shell.
            if _SQLITE_SHELL_RE.search(_unq):
                blocked.add("sqlite3-shell")
            for _m in _SQLITE_DOTFILE_RE.finditer(_unq):
                _dot_f = _m.group("f")
                if len(_dot_f) >= 2 and _dot_f[0] == _dot_f[-1] and _dot_f[0] in ("'", '"'):
                    _dot_f = _dot_f[1:-1]
                # .output |CMD / .once |CMD open CMD as a PIPE (a shell command), not a file.
                if _dot_f.startswith("|"):
                    blocked.add("sqlite3-shell")
                elif _dot_f not in ("stdout", "stderr", "off") and (
                    _git_operand_escapes(_dot_f, _local_assigns)
                    or (_sqlite_cwd_escapes and _operand_relative_local(_dot_f))
                ):
                    blocked.add("sqlite3-write-outside")
            # .backup / .save / .open put the target FILE as the LAST operand (an optional schema
            # name or option tokens precede it), so check the last bare operand for an escape.
            for _m in _SQLITE_LASTFILE_RE.finditer(_unq):
                try:
                    _ops = shlex.split(_m.group(0).strip())
                except ValueError:
                    _ops = _m.group(0).split()
                _tail = [
                    _o for _o in _ops[1:] if not _o.startswith("-")
                ]  # drop the dot-command word and option flags
                if _tail:
                    _bk_f = _tail[-1]
                    if _bk_f not in ("stdout", "stderr", "off") and (
                        _git_operand_escapes(_bk_f, _local_assigns)
                        or (_sqlite_cwd_escapes and _operand_relative_local(_bk_f))
                    ):
                        blocked.add("sqlite3-write-outside")
            if t.startswith("-"):
                _sk += 1
                continue
            # First bare operand is the DBFILE. :memory: / '' / file::memory: never touch disk.
            if not _seen_db:
                _seen_db = True
                _dbn = t
                if len(_dbn) >= 2 and _dbn[0] == _dbn[-1] and _dbn[0] in ("'", '"'):
                    _dbn = _dbn[1:-1]
                _dblow = _dbn.lower()
                _is_mem = (
                    _dbn in ("", ":memory:")
                    or _dblow.startswith("file::memory:")
                    or "mode=memory" in _dblow
                )
                if not _is_mem and (
                    _git_operand_escapes(_dbn, _local_assigns)
                    or (_sqlite_cwd_escapes and _operand_relative_local(_dbn))
                ):
                    blocked.add("sqlite3-write-outside")
            else:
                # A bare operand after the DBFILE is inline SQL, so sqlite3 runs it and exits
                # WITHOUT reading stdin (already scanned for dot-commands above).
                _seen_sql = True
            _sk += 1
        # No inline SQL argv + a stdin source (pipe / redirect) means the dot-commands come from
        # unscanned stdin; fail closed (the .shell / .import / .output there are uninspectable).
        if not _seen_sql and (_sqlite_pipe_target or _sqlite_stdin_redirect):
            blocked.add("sqlite3-stdin-sql")

    # watch runs its command via `sh -c '<operands joined>'` UNLESS -x/--exec is given (then it
    # execs argv directly, resolved by the wrapper handling above). So a quoted payload
    # (watch 'python3 -c ...', watch -n 0.1 'rm -rf /') is shell CODE, not one inert command
    # word; scan it recursively. A bare `watch date` / `watch -n 1 date` just re-scans `date`.
    for i in _wrapper_prefix_idx:
        if _token_basename(tokens[i]) != "watch":
            continue
        _has_x = False
        _ops = []
        _skip_val = False
        _wk = i + 1
        while _wk < len(tokens):
            t = tokens[_wk]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            if _skip_val:
                _skip_val = False
                _wk += 1
                continue
            if t in ("-x", "--exec"):
                _has_x = True
            elif t in ("-n", "--interval"):
                _skip_val = True
            elif not t.startswith("-"):
                _ops.append(t)
            _wk += 1
        if not _has_x and _ops:
            _payload = " ".join(
                (o[1:-1] if len(o) >= 2 and o[0] == o[-1] and o[0] in ("'", '"') else o)
                for o in _ops
            )
            blocked |= _find_blocked_commands(_payload)

    # xargs -I{} / -i / --replace substitutes UNSCANNED stdin into the command at runtime. When
    # the replacement token becomes the command word (xargs -I{} {}) or flows into an interpreter
    # code string (xargs -I{} sh -c '{}', xargs -I% python3 -c %), stdin executes as code -- the
    # `{}` payload the scanner sees is inert. Fail closed on those forms; a replacement used only
    # as a data ARGUMENT to a non-interpreter (xargs -I{} cp {} dir/) is left to the normal
    # command-word scan, and xargs without a replace flag (xargs echo hi) is unaffected.
    _XARGS_INTERP = _SHELL_BINARIES | _INTERPRETER_COMMANDS
    for i in _wrapper_prefix_idx:
        if _token_basename(tokens[i]) != "xargs":
            continue
        _xseg = []
        _xk = i + 1
        while _xk < len(tokens):
            t = tokens[_xk]
            if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                break
            _xseg.append(t)
            _xk += 1
        _repl = None
        _xj = 0
        while _xj < len(_xseg):
            t = _xseg[_xj]
            if t == "-I" and _xj + 1 < len(_xseg):
                _repl = _xseg[_xj + 1]
                _xj += 2
                continue
            if t.startswith("-I") and len(t) > 2:
                _repl = t[2:]
            elif t in ("-i", "--replace"):
                _repl = "{}"
            elif t.startswith("--replace="):
                _repl = t.split("=", 1)[1] or "{}"
            elif t.startswith("-i") and len(t) > 2:
                _repl = t[2:]
            _xj += 1
        if not _repl:
            continue
        # Resolve the wrapped command word (skip xargs flags + their separated operands).
        _cwidx = None
        _cwj = 0
        while _cwj < len(_xseg):
            t = _xseg[_cwj]
            if t.startswith("-"):
                _cwj += 2 if _wrapper_flag_takes_operand("xargs", t) else 1
                continue
            _cwidx = _cwj
            break
        if _cwidx is None:
            continue
        _cw = os.path.basename(_xseg[_cwidx]).lower()
        if _repl in _xseg[_cwidx]:
            blocked.add("xargs-replace-exec")  # stdin becomes the command itself
        elif _cw in _XARGS_INTERP:
            for _ci2 in range(_cwidx + 1, len(_xseg)):
                _ct2 = _xseg[_ci2].lower()
                _is_code_flag = _ct2 in ("-c", "-e", "--eval") or (
                    _ct2.startswith("-") and not _ct2.startswith("--") and _ct2.endswith("c")
                )
                if _is_code_flag and _ci2 + 1 < len(_xseg) and _repl in _xseg[_ci2 + 1]:
                    blocked.add("xargs-replace-exec")  # stdin flows into interpreter code
                    break

    # Output redirection (> / >> / &> / N>) runs in an unguarded child shell that follows
    # symlinks before any Python guard, so no filename target can be trusted: a relative
    # single-component name (> out) may be a pre-existing symlink to an outside file, a
    # relative multi-component name (> sub/out) may traverse a symlinked subdir, an absolute
    # / ~ / .. target is plainly outside, and a $ / backtick target can expand anywhere.
    # Fail closed on every real-file target; only fd duplications (>&2) and the standard
    # device sinks (/dev/null, ...) are allowed. Scanning tokens (not the raw string) avoids
    # matching a `>` inside a quoted argument.
    for i, tok in enumerate(tokens):
        rm = re.search(r">{1,2}([^\s>]*)$", tok)
        if rm is None:
            continue
        tgt = rm.group(1)
        j = i
        # `>|` (noclobber override) and `>&` (stdout+stderr / fd-or-file redirect) tokenize
        # as `>` then `|` / `&`, so that punctuation is part of the redirect operator, not a
        # pipeline / background op; skip it and take the real target after.
        if not tgt and j + 1 < len(tokens) and tokens[j + 1] in ("|", "&"):
            j += 1
        if not tgt and j + 1 < len(tokens):
            tgt = tokens[j + 1]
        if not tgt:
            continue
        tn = tgt.replace("\\", "/")
        # Allowed: a pure fd duplication (>&2, >&1 -> `&2` / a bare digit) and the safe
        # device sinks. Everything else is a file target that fails closed.
        if tgt.startswith("&") or tgt.isdigit() or tn in _SAFE_REDIRECT_TARGETS:
            continue
        blocked.add("redirect:" + tgt)

    # `cd` / `pushd` to a dir OUTSIDE the workdir moves the child shell's cwd so a later
    # relative redirect / write escapes (`cd /tmp; echo x > p`, `pushd /tmp; echo x > p`).
    # Block a command-position cwd change to an absolute / .. / ~ / variable target; a
    # relative in-workdir `cd data` stays allowed.
    _at_cmd = True
    for i, tok in enumerate(tokens):
        if tok in _SHELL_SEPARATORS or tok in _SHELL_KEYWORDS_AS_SEP:
            _at_cmd = True
            continue
        if _at_cmd and _token_basename(tok) in ("command", "builtin"):
            # `command` / `builtin` run the following shell builtin with its args, so a
            # `command cd /tmp` still changes the cwd. Stay at command position so the cd
            # behind the wrapper is inspected (bash `help command`/`help builtin`).
            continue
        if _at_cmd and _token_basename(tok) in ("cd", "pushd"):
            _cwd_kw = _token_basename(tok)
            for k in range(i + 1, len(tokens)):
                t = tokens[k]
                if t.startswith("-") or t.startswith("+"):
                    continue  # cd flags (-P/-L/-e/-@) and pushd rotation (+N/-N)
                tnn = t.replace("\\", "/")
                if (
                    t.startswith("~")
                    or tnn.startswith("/")
                    or ".." in tnn.split("/")
                    or "$" in t
                    or "`" in t
                ):
                    blocked.add(_cwd_kw + ":" + t)
                break
            _at_cmd = False
            continue
        if not tok.startswith("-"):
            _at_cmd = False

    # An EXPANSION in COMMAND POSITION runs whatever it expands to as the command name and
    # cannot be proven safe: a command substitution ($(printf touch) / `printf touch`), a
    # variable-expanded command word (p=python3; $p -c ...), or a ${VAR} parameter expansion.
    # Fail closed. (An argument-position expansion -- echo $(date), echo $HOME, x=$(cmd) -- is
    # not at command position, so it stays allowed. ${IFS} is already expanded to whitespace
    # above, so a `cat${IFS}x` command word is not misread as an expansion here.)
    if re.search(r"(?:^|[\n;&|(])\s*(?:\$|`)", command):
        blocked.add("command-expansion")

    # Some normally read-only utilities MUTATE files with certain flags (sed -i, sort -o
    # FILE, find ... -delete, dd of=FILE, tee FILE, truncate), writing/deleting OUTSIDE the
    # workdir in an unguarded child that no redirect token exposes. Treat the mutating
    # invocation as a child writer. Uses the wrapper-aware command-word indices so a wrapper
    # prefix (env sed -i ..., nice sed -i ...) does not hide the mutating utility.
    for i in _cmd_word_idx:
        tok = tokens[i]
        _base = _token_basename(tok)
        if _base not in (
            "sed",
            "gsed",
            "ssed",
            "perl",
            "sort",
            "shuf",
            "find",
            "dd",
            "tee",
            "truncate",
            "history",
        ):
            continue
        if _base == "truncate":
            blocked.add("mutating:truncate")
            continue
        for k in range(i + 1, len(tokens)):
            a = tokens[k]
            if a in _SHELL_SEPARATORS or a in _SHELL_KEYWORDS_AS_SEP:
                break
            al = a.lower()
            _short = al.startswith("-") and not al.startswith("--")
            if _base in ("sed", "gsed", "ssed", "perl"):
                if al.startswith("--in-place") or (_short and "i" in al[1:]):
                    blocked.add("mutating:" + _base)
                    break
                # A sed SCRIPT can write files (`w FILE` / `W FILE` / `s///w`) or execute shell
                # commands (`e CMD` / `s///e`) even without -i: sed -n '1w /tmp/escape' file,
                # sed -n 'w/tmp/probe' file (no space), sed '1e touch /tmp/x' file. The script may
                # be a bare positional OR provided via -e / --expression (sed -e'w /tmp/x' /dev/null,
                # sed --expression='w /tmp/x'). Detect the write / execute commands and flags in the
                # script text; a plain s/word/x/ is not matched.
                if _base in ("sed", "gsed", "ssed"):
                    # A -f / --file script file is loaded from disk and can carry the same
                    # w / W / e / r mutating + exec commands as an inline script, but its
                    # contents are not statically visible (a planted workdir evil.sed with
                    # `1w /tmp/p`). Fail closed on any -f / --file form (separated, glued, or
                    # combined short group like -nf).
                    if (
                        a in ("-f", "--file")
                        or al.startswith("--file=")
                        or (_short and "f" in al[1:])
                    ):
                        blocked.add("sed-script-file:" + _base)
                        break
                    _sed_script = None
                    if a in ("-e", "--expression") and k + 1 < len(tokens):
                        _sed_script = tokens[k + 1]  # -e SCRIPT (separated)
                    elif al.startswith("-e") and not al.startswith("--") and len(a) > 2:
                        _sed_script = a[2:]  # glued -e'w /tmp/x'
                    elif a.startswith("--expression="):
                        _sed_script = a.split("=", 1)[1]
                    elif not a.startswith("-"):
                        _sed_script = a  # bare positional script
                    if _sed_script is not None and (
                        _SED_WRITE_RE.search(_sed_script)
                        or _SED_EXEC_RE.search(_sed_script)
                        or _SED_ADDR_EXEC_RE.search(_sed_script)
                        or _SED_SFLAG_RE.search(_sed_script)
                    ):
                        blocked.add("mutating:" + _base)
                        break
            elif _base == "sort":
                if al.startswith("--output") or (_short and "o" in al[1:]):
                    blocked.add("mutating:sort")
                    break
            elif _base == "shuf":
                # shuf -o FILE / --output=FILE writes its shuffled output to FILE in an unguarded
                # child, escaping the workdir just like sort -o.
                if al.startswith("--output") or (_short and "o" in al[1:]):
                    blocked.add("mutating:shuf")
                    break
            elif _base == "find":
                # -delete removes; -fprint/-fprintf/-fprint0 and -fls write their listing to a
                # named FILE (find . -fls /tmp/escape truncates/creates it in an unguarded child).
                if al == "-delete" or al.startswith("-fprint") or al == "-fls":
                    blocked.add("mutating:find")
                    break
            elif _base == "dd":
                if al.startswith("of="):
                    blocked.add("mutating:dd")
                    break
            elif _base == "tee" and not a.startswith("-"):
                blocked.add("mutating:tee")
                break
            elif _base == "history" and _short and any(_c in al[1:] for _c in "warn"):
                # bash's history builtin reads/writes an arbitrary file: `history -w FILE`
                # (or -a append) creates/overwrites an absolute host path, and `-r` / `-n`
                # read a file into the history buffer. Even without a FILE operand it targets
                # $HISTFILE, which the caller can point outside the workdir. -c / -d / -p / -s
                # do not touch a file, so only w / a / r / n are blocked.
                blocked.add("mutating:history")
                break

    # uniq [OPTION]... [INPUT [OUTPUT]] writes to its SECOND positional operand (uniq in out /
    # uniq /dev/null /tmp/p) in an unguarded child -- a native writer no redirect token exposes,
    # like sort -o. Block when a second bare operand is present; a single INPUT (or none) reads
    # to stdout and stays allowed. -f / -s / -w take a separated numeric value, so skip it.
    for i in _cmd_word_idx:
        if _token_basename(tokens[i]) != "uniq":
            continue
        _uniq_ops = 0
        _skip_val = False
        for k in range(i + 1, len(tokens)):
            a = tokens[k]
            if a in _SHELL_SEPARATORS or a in _SHELL_KEYWORDS_AS_SEP:
                break
            if _skip_val:
                _skip_val = False
                continue
            if a.startswith("-") and a != "-":
                if a in ("-f", "-s", "-w", "--skip-fields", "--skip-chars", "--check-chars"):
                    _skip_val = True  # separated numeric value belongs to the flag, not an operand
                continue
            _uniq_ops += 1
            if _uniq_ops == 2:  # the OUTPUT operand
                blocked.add("mutating:uniq")
                break

    return blocked


def _blocked_in_argv(str_elts: list[str | None]) -> tuple[set[str], int | None]:
    """Scan the command WORDS of a non-shell argv vector (subprocess.run(['rm', '-rf', '/'])).
    Only element 0 -- and the real command after any wrapper prefix (env / nice / timeout /
    xargs / ...) -- is executed by the OS; every later element is a literal argument that is
    never run. Scanning just the command word keeps `env rm -rf /` blocked (rm resolved through
    the wrapper) while a benign argument such as subprocess.run(['echo', 'python']) is not
    misread as invoking `python`.

    Returns (blocked_basenames, cmd_index): cmd_index is the position of the resolved command
    word (or None), so the caller can hand a wrapper-hidden shell binary (env bash s.sh) to the
    shell-argv analyzer."""
    blocked: set[str] = set()
    idx, n = 0, len(str_elts)
    prefix_pending = False  # a wrapper is awaiting its real command word
    prev_was_flag = False  # last token (under a wrapper) was an option flag with an operand
    cur_wrapper = None  # the active wrapper's basename (env / nice / timeout / ...)
    while idx < n:
        tok = str_elts[idx]
        if tok is None:
            return blocked, None  # a non-literal element hides the command word; stop
        # env FOO=bar assignments precede the command word.
        if _ASSIGNMENT_RE.match(tok):
            idx += 1
            continue
        if prefix_pending and tok.startswith("-"):
            # env -S CMD / --split-string=CMD splits its operand into a command line, so
            # scan that operand with the full command scanner (env -S 'bash -c ...').
            if cur_wrapper == "env":
                if tok in ("-S", "--split-string"):
                    _nxt = str_elts[idx + 1] if idx + 1 < n else None
                    if _nxt is not None:
                        blocked |= _find_blocked_commands(_nxt)
                    return blocked, None
                if tok.startswith("--split-string="):
                    blocked |= _find_blocked_commands(tok[len("--split-string=") :])
                    return blocked, None
                if tok.startswith("-S") and len(tok) > 2:
                    blocked |= _find_blocked_commands(tok[2:])
                    return blocked, None
            # Only a flag that takes a SEPARATED operand (env -u NAME, nice -n 5) marks the
            # next token as its value; a no-operand flag (env -i, xargs -0) or a glued short
            # flag (stdbuf -oL) does not, so the real command after it is still analysed.
            prev_was_flag = _wrapper_flag_takes_operand(cur_wrapper, tok)
            idx += 1
            continue
        # A wrapper's numeric arg (`timeout 5 cmd`).
        if prefix_pending and _is_wrapper_numeric_arg(tok):
            prev_was_flag = False
            idx += 1
            continue
        base = os.path.basename(tok).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        # A wrapper flag's SEPARATED operand (`env -u FOO python3`, `env -C DIR cmd`): the
        # token after a wrapper option flag that is not itself a blocked command / prefix /
        # shell is the flag's value -- skip it and keep scanning so the real command (python3,
        # bash) is not missed. A blocked command / prefix / shell is treated as the command.
        if (
            prefix_pending
            and prev_was_flag
            and base not in _BLOCKED_COMMANDS
            and base not in _COMMAND_PREFIXES
            and base not in _SHELL_BINARIES
        ):
            prev_was_flag = False
            idx += 1
            continue
        prev_was_flag = False
        if base in _BLOCKED_COMMANDS or _is_versioned_interpreter(base):
            blocked.add(base)
        if base in _COMMAND_PREFIXES:
            prefix_pending = True
            cur_wrapper = base
            idx += 1
            continue  # wrapper consumes one command; the next word is the real one
        if _is_local_executable_path(tok):
            blocked.add("local-exec:" + base)  # runs an unguarded shebang interpreter
        return blocked, idx  # reached the executed command word
    return blocked, None


def _build_safe_env(workdir: str) -> dict[str, str]:
    """Build a minimal, credential-free environment for sandboxed subprocesses.

    Whitelist-built from scratch (parent env NOT inherited): only PATH/HOME/
    TMPDIR/LANG/TERM/PYTHONIOENCODING (+VIRTUAL_ENV or Windows SystemRoot) reach
    the child; all credential vars (HF_TOKEN, AWS_*, etc.) are absent. HOME
    points at the sandbox workdir so SDKs can't read the operator's cached creds.
    """
    # Start from the running interpreter's dir so 'python'/'pip' resolve to the
    # same environment the Studio server runs in.
    exe_dir = os.path.dirname(sys.executable)
    path_entries = [exe_dir] if exe_dir else []

    # If a virtualenv is active, include its bin/Scripts directory.
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_bin = os.path.join(venv, "Scripts" if sys.platform == "win32" else "bin")
        if venv_bin not in path_entries:
            path_entries.append(venv_bin)

    if sys.platform == "win32":
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        path_entries.extend([os.path.join(sysroot, "System32"), sysroot])
    else:
        path_entries.extend(["/usr/local/bin", "/usr/bin", "/bin"])

    # Deduplicate, preserving order.
    deduped = list(dict.fromkeys(p for p in path_entries if p))

    env = {
        "PATH": os.pathsep.join(deduped),
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": "dumb",
        "PYTHONIOENCODING": "utf-8",
        # HOME points at the workdir, so a prior run could plant
        # .local/.../site-packages/usercustomize.py that runs (unguarded) at the next child's
        # startup. Disable the per-user site directory here too (belt-and-suspenders with the
        # interpreter's -s flag) so a sandboxed child never imports it.
        "PYTHONNOUSERSITE": "1",
        # git runs repository hooks (.git/hooks/pre-commit, post-checkout, ...) as executable
        # files in an UNGUARDED child; a sandboxed snippet could plant one and trigger it via a
        # benign-looking git commit / merge / checkout. Point core.hooksPath at a non-directory
        # (via git's env-config mechanism) so NO repository hook runs, for every git subcommand,
        # without having to block git itself. Neutralizing hooks is the sandbox-correct default.
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": os.devnull,
    }
    if venv:
        env["VIRTUAL_ENV"] = venv
    # Windows needs SystemRoot for Python/subprocess to work.
    if sys.platform == "win32":
        env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
    return env


# Credential env vars dropped even in bypass mode so tool code cannot read the
# operator's keys. Over-strips on purpose (a benign var is harmless to lose).
_BYPASS_ENV_SECRET_NAMES = frozenset(
    {
        "HF_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "WANDB_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "REPLICATE_API_TOKEN",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "NGC_API_KEY",
        "KAGGLE_KEY",
        "MYSQL_PWD",  # exact name: markers use PASSWD, not PWD (PWD is the cwd var)
        "LD_PRELOAD",
        # Auth brokers / capability handles: not secrets by value, but they
        # hand the child the operator's live agent (ssh/gpg), kube config, or
        # docker daemon. Names are listed because there is no value signal to
        # key off. URL config vars (HTTP_PROXY, PIP_INDEX_URL, DATABASE_URL,
        # ...) are intentionally NOT name-listed: a benign proxy/index without
        # credentials must keep working in bypass mode, while a credentialed
        # value is dropped by _is_secret_env_value() regardless of its name.
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GPG_AGENT_INFO",
        "GNUPGHOME",
        "KUBECONFIG",
        "DOCKER_HOST",
    }
)
_BYPASS_ENV_SECRET_PREFIXES = ("AWS_", "AZURE_", "GOOGLE_", "GCP_", "GCLOUD_", "DYLD_")
_BYPASS_ENV_SECRET_MARKERS = (
    "TOKEN",
    "API_KEY",
    "APIKEY",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "AUTH",  # e.g. NPM_CONFIG__AUTH (npm _auth), REDISCLI_AUTH
    # Azure App Service connection strings: SQLCONNSTR_/CUSTOMCONNSTR_/... and
    # WEBSITE_CONTENTAZUREFILECONNECTIONSTRING carry DB/storage credentials.
    "CONNSTR",
    "CONNECTIONSTRING",
)
# Non-secret hardening flags that match a secret prefix/marker but must be KEPT
# so bypass mode does not silently undo an operator's opt-out. AWS_EC2_METADATA_
# DISABLED tells the AWS SDK/CLI not to pull instance-role creds from IMDS;
# dropping it would re-open that path for a bypassed tool.
_BYPASS_ENV_KEEP_NAMES = frozenset(
    {
        "AWS_EC2_METADATA_DISABLED",
        "AWS_EC2_METADATA_V1_DISABLED",
    }
)
# Matches a URL that embeds userinfo before the host, covering both
# "scheme://user:pass@host" and token-only "scheme://token@host" (and
# percent-encoded variants). The userinfo must precede the first '/', so an '@'
# in a path or query does not false-positive. Used to scrub credential-bearing
# URL values regardless of the variable's name.
_URL_USERINFO_RE = re.compile(r"://[^/\s@]+@")
# Connection-string credential fields (ADO.NET / Azure storage / Service Bus):
# "...;Password=...", "...;AccountKey=...", "...;SharedAccessKey=...". Catches
# credential-bearing values whose names dodge the name classifier. "accesskey"
# also covers Shared/Secret AccessKey via substring; the Name fields (e.g.
# SharedAccessKeyName=) do not match since "=" must follow the keyword.
_SECRET_VALUE_RE = re.compile(r"(?i)(?:password|pwd|accountkey|accesskey)\s*=\s*[^\s;]")

# Names that hold no secret value but point SDKs at the operator's real
# home/cache/config (cached tokens, cred files), defeating the HOME repoint.
# Startup always sets HF_HOME (-> $HF_HOME/token), so this is the live leak.
# Dropped in bypass mode so tools fall back to the empty repointed HOME.
_BYPASS_ENV_CRED_LOCATION_NAMES = frozenset(
    {
        # HF cache roots (token lives under $HF_HOME/token)
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_ASSETS_CACHE",
        # XDG base dirs (resolved before $HOME)
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        # explicit cred/config file pointers honoured before $HOME
        "NETRC",
        "PGPASSFILE",
        "BOTO_CONFIG",
        "PIP_CONFIG_FILE",
        "CLOUDSDK_CONFIG",
        "KAGGLE_CONFIG_DIR",
        "DOCKER_CONFIG",
        "WANDB_DIR",
        "WANDB_CONFIG_DIR",
        "WANDB_CACHE_DIR",
        # package-manager / git / cloud config pointers to real cred files
        "NPM_CONFIG_USERCONFIG",
        "NPM_CONFIG_GLOBALCONFIG",
        "YARN_RC_FILENAME",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "CARGO_HOME",
        "RCLONE_CONFIG",
        # auth-helper scripts that hand creds to git/ssh
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        # shell startup hook: bash -c sources $BASH_ENV (can re-export secrets)
        "BASH_ENV",
        # Windows: HOMEDRIVE+HOMEPATH compose a home that bypasses HOME
        "HOMEDRIVE",
        "HOMEPATH",
    }
)
# Windows profile dirs SDKs read creds under; repointed (not dropped) since
# callers expect them present.
_BYPASS_ENV_WINDOWS_PROFILE_VARS = ("USERPROFILE", "APPDATA", "LOCALAPPDATA")


def _is_secret_env_name(name: str) -> bool:
    """True if an env var name looks like it carries a credential."""
    upper = name.upper()
    if upper in _BYPASS_ENV_KEEP_NAMES:
        return False  # non-secret hardening flag; keep it
    if upper in _BYPASS_ENV_SECRET_NAMES:
        return True
    if any(upper.startswith(p) for p in _BYPASS_ENV_SECRET_PREFIXES):
        return True
    return any(marker in upper for marker in _BYPASS_ENV_SECRET_MARKERS)


def _is_cred_location_env_name(name: str) -> bool:
    """True for vars that point SDKs at the real home/cache/config (cached creds)."""
    return name.upper() in _BYPASS_ENV_CRED_LOCATION_NAMES


def _is_secret_env_value(value: str) -> bool:
    """True if a value embeds credentials regardless of its name.

    Catches URL userinfo (``scheme://user:token@host`` in DATABASE_URL /
    PIP_INDEX_URL / HTTP_PROXY) and connection-string credential fields
    (``...;Password=...`` / ``...;AccountKey=...``) whose names dodge the name
    classifier.
    """
    if not value:
        return False
    return _URL_USERINFO_RE.search(value) is not None or _SECRET_VALUE_RE.search(value) is not None


def _build_bypass_env(workdir: str) -> dict[str, str]:
    """Env for bypass exec: full host env (unrestricted) minus credential vars,
    with HOME/TMPDIR repointed at the workdir so SDKs cannot read cached creds.

    Note: stripping the child env is necessary but not sufficient on its own -
    a same-UID child can still read the parent's environment via procfs, so
    callers also harden the parent (see _harden_parent_against_proc_env_leak).
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not _is_secret_env_name(k)
        and not _is_secret_env_value(v)
        and not _is_cred_location_env_name(k)
    }
    env["HOME"] = workdir
    env["TMPDIR"] = workdir
    # Windows tempfile / SDKs honour TEMP/TMP, not TMPDIR; repoint all three so
    # the bypassed tool writes under the per-session sandbox dir on every OS.
    env["TEMP"] = workdir
    env["TMP"] = workdir
    # Windows SDKs read creds under the profile dirs, not $HOME; repoint set
    # ones to the workdir (HOMEDRIVE/HOMEPATH are dropped above).
    for var in _BYPASS_ENV_WINDOWS_PROFILE_VARS:
        if var in os.environ:
            env[var] = workdir
    return env


def _sandbox_preexec():
    """Best-effort sandbox setup for sandboxed subprocesses (modules are
    resolved at import time so the forked child runs no imports)."""
    try:
        os.setsid()
    except OSError:
        pass

    try:
        os.umask(0o077)
    except OSError:
        pass

    if _libc is not None:
        try:
            _libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
        except (OSError, AttributeError):
            pass

        try:
            _libc.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG = SIGKILL
        except (OSError, AttributeError):
            pass

        # CLONE_NEWNET not applied: with userns enabled it blocks all egress,
        # including allowlisted hosts. Network policy is enforced by the AST
        # host check and the bash blocklist.

    if _resource is not None:
        # RLIMIT_NPROC is per-real-UID, so the cap is well above normal usage.
        try:
            nproc = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NPROC", "10000"))
            _resource.setrlimit(_resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            _resource.setrlimit(_resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
        except (ValueError, OSError):
            pass
        try:
            as_bytes = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_AS_GB", "8")) * 1024 * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (as_bytes, as_bytes))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            cpu_s = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_CPU_S", "600"))
            _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            # High enough for multi-shard safetensors mmaps; tunable via env.
            # Clamp to the inherited hard limit so setrlimit doesn't ValueError
            # when the parent's hard cap is below the request.
            nofile = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NOFILE", "16384"))
            _soft_cur, hard_cur = _resource.getrlimit(_resource.RLIMIT_NOFILE)
            target = nofile if hard_cur == _resource.RLIM_INFINITY else min(nofile, hard_cur)
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (target, target))
        except (ValueError, OSError, AttributeError):
            pass


def _bypass_preexec():
    """Minimal pre-exec for bypass exec: os.setsid() only.

    Required, not a restriction: _kill_process_tree does killpg(getpgid(child)),
    so without a new session a timeout/cancel would kill the Studio server too.
    """
    try:
        os.setsid()
    except OSError:
        pass


# Hardening the Studio parent is done once (PR_SET_DUMPABLE is process-global
# and sticky); guarded so repeated bypass calls do not re-issue the prctl.
_parent_proc_hardened = False


def _harden_parent_against_proc_env_leak() -> bool:
    """Make the Studio process's /proc/<pid>/environ unreadable to its children.

    Stripping the child env is not enough on Linux: a bypassed same-UID child
    runs unsandboxed and can read /proc/<getppid()>/environ to recover the
    tool-executing process's *unfiltered* secrets (HF_TOKEN, cloud keys, ...).
    Clearing the dumpable flag (PR_SET_DUMPABLE=0) reparents this process's
    /proc entries to root, so a same-UID child can no longer read its environ.

    Returns True when the process is hardened or hardening is unnecessary (no
    /proc leak off Linux), and False when it is needed but could not be applied
    (e.g. prctl denied by a seccomp policy). Callers must fail closed - refuse
    the unsandboxed exec - when this returns False, rather than running with the
    parent environ still readable.

    Scope: this closes the direct parent read (the demonstrated leak). It is a
    mitigation, not a full boundary - a bypassed tool is unsandboxed by design,
    so it can still walk /proc to a same-UID *ancestor* (e.g. the launching
    shell) or read on-disk credentials by absolute path. Complete isolation
    needs a separate uid / PID+mount namespace, which is out of scope here; the
    UI already warns the mode is dangerous. Applied lazily on first bypass exec
    so non-bypass operation is unchanged.
    """
    global _parent_proc_hardened
    if _parent_proc_hardened:
        return True
    if sys.platform != "linux":
        return True  # no /proc/<pid>/environ same-UID leak to close
    if _libc is None:
        return False  # on Linux but cannot issue prctl -> cannot harden
    try:
        # prctl(PR_SET_DUMPABLE=4, SUID_DUMP_DISABLE=0). ctypes returns the
        # syscall result (-1 on failure) and does NOT raise, so check it.
        ret = _libc.prctl(4, 0, 0, 0, 0)
    except (OSError, AttributeError):
        return False
    if ret != 0:
        return False
    _parent_proc_hardened = True
    return True


def _get_shell_cmd(command: str) -> list[str]:
    """Return the platform-appropriate shell invocation for a command string."""
    if sys.platform == "win32":
        return ["cmd", "/c", command]
    return ["bash", "-c", command]


# Per-session working directories so each chat thread gets its own sandbox.
# Falls back to ~/studio_sandbox/_default for callers without a session_id.
_workdirs: dict[str, str] = {}


# Non-matching session_ids collapse to ``_invalid`` to block cross-session escapes.
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9_\-]{1,64}\Z")
_PROJECT_SESSION_PREFIX = "project-"


def _get_project_workdir(session_id: str) -> str | None:
    if not session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    project_id = session_id[len(_PROJECT_SESSION_PREFIX) :]
    if not project_id or not _SESSION_ID_RE.match(project_id):
        return None
    try:
        from storage.studio_db import ensure_chat_project_workspace
        project = ensure_chat_project_workspace(project_id)
    except Exception:
        logger.warning("Failed to resolve project sandbox for %s", session_id, exc_info = True)
        return None
    if not project:
        return None
    root_path = project.get("rootPath")
    sandbox_path = project.get("sandboxPath")
    if not root_path or not sandbox_path:
        return None
    root_real = os.path.realpath(root_path)
    sandbox_real = os.path.realpath(sandbox_path)
    if sandbox_real != root_real and not sandbox_real.startswith(root_real + os.sep):
        return None
    return sandbox_real


def _get_workdir(session_id: str | None = None) -> str:
    """Return a per-session sandbox dir at mode 0o700."""
    global _workdirs
    key = session_id or "_default"
    if key not in _workdirs or not os.path.isdir(_workdirs[key]):
        home = os.path.expanduser("~")
        sandbox_root = os.path.join(home, "studio_sandbox")
        project_workdir = (
            _get_project_workdir(session_id)
            if session_id and _SESSION_ID_RE.match(session_id)
            else None
        )
        if project_workdir:
            workdir = project_workdir
        elif session_id and _SESSION_ID_RE.match(session_id):
            workdir = os.path.join(sandbox_root, session_id)
            if not os.path.realpath(workdir).startswith(os.path.realpath(sandbox_root) + os.sep):
                workdir = os.path.join(sandbox_root, "_invalid")
        elif session_id:
            workdir = os.path.join(sandbox_root, "_invalid")
        else:
            workdir = os.path.join(sandbox_root, "_default")
        os.makedirs(workdir, exist_ok = True)
        try:
            os.chmod(sandbox_root, 0o700)
        except OSError:
            pass
        try:
            os.chmod(workdir, 0o700)
        except OSError:
            pass
        _workdirs[key] = workdir
    return _workdirs[key]


def get_sandbox_workdir(session_id: str | None = None) -> str:
    return _get_workdir(session_id)


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and fetch page content. Returns snippets for all results. "
            "Use the url parameter to fetch full page text from a specific URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "url": {
                    "type": "string",
                    "description": "A URL to fetch full page content from (instead of searching). Use this to read a page found in search results.",
                },
            },
            "required": [],
        },
    },
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code in a sandbox and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to run",
                }
            },
            "required": ["code"],
        },
    },
}

TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Execute a terminal command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run",
                }
            },
            "required": ["command"],
        },
    },
}

RENDER_HTML_TOOL = {
    "type": "function",
    "function": {
        "name": "render_html",
        "description": (
            "Render a self-contained HTML/CSS/JavaScript canvas for the user. "
            "Call this at most once per assistant response unless the user "
            "explicitly asks for changes in that response. Future user requests "
            "for new canvases may call render_html once. Put the entire document "
            "in code, including any CSS in <style> tags and JavaScript in <script> tags."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "A complete self-contained HTML document.",
                },
                "title": {
                    "type": "string",
                    "description": "Short display title for the canvas.",
                },
            },
            "required": ["code"],
        },
    },
}

# Duplicated (not imported from core.rag.tool) so the registry never pulls in
# the RAG stack; dispatch imports it lazily.
SEARCH_KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the user's uploaded documents and knowledge bases for "
            "relevant passages. Use this whenever the question may be answered "
            "by the attached documents, then cite the returned chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max chunks to return.",
                },
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    PYTHON_TOOL,
    TERMINAL_TOOL,
    RENDER_HTML_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
]


# OpenAI's function.name regex ^[a-zA-Z0-9_-]{1,64}$, enforced before streaming.
# MCP tool names with '.', '/', spaces, etc. would 400 the whole request, so we
# validate up front and skip with a warning.
_OPENAI_FN_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _mcp_specs_for_server(server: dict, mcp_tools: list[dict]) -> list[dict]:
    """Convert an MCP server's tool list into OpenAI function specs."""
    display = server.get("display_name") or server["id"]
    specs: list[dict] = []
    seen_names: set[str] = set()
    for tool in mcp_tools:
        raw_name = tool.get("name") or ""
        if not raw_name:
            logger.warning("Skipping MCP tool on '%s': empty name.", display)
            continue
        name = f"{MCP_TOOL_PREFIX}{server['id']}__{raw_name}"
        # Bad chars or oversized names would 400 the whole request; skip + warn
        # so the rest of the tools still ship.
        if not _OPENAI_FN_NAME_RE.fullmatch(name):
            logger.warning(
                "Skipping MCP tool '%s' on '%s': composed name '%s' is not "
                "valid OpenAI function.name (regex ^[a-zA-Z0-9_-]{1,64}$).",
                raw_name,
                display,
                name,
            )
            continue
        # Duplicate tool names would also 400 OpenAI; drop dupes.
        if name in seen_names:
            logger.warning("Skipping duplicate MCP tool '%s' on '%s'.", raw_name, display)
            continue
        seen_names.add(name)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"[{display}] {tool.get('description') or ''}".strip(),
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
        )
    return specs


async def get_enabled_mcp_tools() -> list[dict]:
    servers = [s for s in mcp_servers_db.list_servers() if s.get("is_enabled")]
    # Never spawn stdio servers when stdio is disabled on this host (e.g. a DB
    # carried from a desktop install onto a Colab/network deployment).
    if not stdio_mcp_enabled():
        servers = [s for s in servers if not is_stdio(s["url"])]
    if not servers:
        return []

    # Skip servers still in their post-failure cool-off, otherwise a down
    # server gets re-probed -- and blocks the send for the full timeout -- on
    # every message.
    uncached = [
        s for s in servers if get_cached_tools(s["id"]) is None and not in_failure_cooloff(s["id"])
    ]
    if uncached:
        results = await asyncio.gather(
            *(
                list_tools_async(
                    url = s["url"],
                    headers = parse_server_headers(s),
                    timeout = probe_timeout(s["url"], bool(s.get("use_oauth"))),
                    use_oauth = bool(s.get("use_oauth")),
                )
                for s in uncached
            ),
            return_exceptions = True,
        )
        # An edit/delete can land while we await a probe (up to 305 s for
        # OAuth); its cache eviction is a no-op against an entry we haven't
        # written yet. Re-read and drop a result whose server changed or
        # was removed mid-probe, else a stale tool list caches indefinitely.
        current = {s["id"]: s for s in mcp_servers_db.list_servers()}
        for server, payload in zip(uncached, results):
            # Guard the failure branch too: a stale failure must not park a
            # cool-off on the fresh config, or the server the user just fixed
            # is skipped for the whole window.
            fresh = current.get(server["id"])
            if fresh is None or any(
                fresh.get(k) != server.get(k) for k in TOOL_CACHE_INVALIDATING_FIELDS
            ):
                continue
            if isinstance(payload, BaseException):
                logger.warning(
                    "MCP server '%s' (%s) discovery failed: %s",
                    server.get("display_name") or server["id"],
                    server.get("url"),
                    payload,
                )
                # Failures aren't cached, but record one so a down server
                # isn't re-probed every send during the cool-off.
                record_probe_failure(server["id"], bool(fresh.get("use_oauth")))
                continue
            cache_tools(server["id"], payload)

    specs: list[dict] = []
    for server in servers:
        payload = get_cached_tools(server["id"])
        if payload is None:
            continue
        specs.extend(_mcp_specs_for_server(server, payload))
    return specs


_TIMEOUT_UNSET = object()


def _render_html_result(arguments: dict) -> str:
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        return "Error: render_html requires a non-empty code string."
    title = arguments.get("title")
    if isinstance(title, str) and title.strip():
        safe_title = title.strip()[:120]
        return (
            f"Rendered HTML canvas: {safe_title}. Do not call render_html "
            "again in this response unless the user asks for changes. For a later "
            "user request for a new canvas, call render_html once."
        )
    return (
        "Rendered HTML canvas. Do not call render_html again in this response "
        "unless the user asks for changes. For a later user request for a new "
        "canvas, call render_html once."
    )


def execute_tool(
    name: str,
    arguments: dict,
    cancel_event = None,
    timeout: int | None = _TIMEOUT_UNSET,
    session_id: str | None = None,
    rag_scope: dict | None = None,
    disable_sandbox: bool = False,
) -> str:
    """Execute a tool by name with the given arguments; returns a string.

    ``timeout``: int seconds, ``None`` = no limit, unset = ``_EXEC_TIMEOUT``.
    ``session_id``: optional ID for per-conversation sandbox isolation.
    ``rag_scope``: hidden per-request RAG context the model never sees; consumed
    by ``search_knowledge_base``.
    ``disable_sandbox``: Bypass Permissions; run python/terminal without the
    safety checks, blocklist, or resource caps (secrets still stripped). Only
    affects local code tools; web_search / MCP are unchanged.
    """
    logger.info(f"execute_tool: name={name}, session_id={session_id}, timeout={timeout}")
    effective_timeout = _EXEC_TIMEOUT if timeout is _TIMEOUT_UNSET else timeout
    if name == "search_knowledge_base":
        return _search_knowledge_base(arguments, rag_scope)
    if name == "render_html":
        return _render_html_result(arguments)
    if name.startswith(MCP_TOOL_PREFIX):
        try:
            _, server_id, tool_name = name.split("__", 2)
        except ValueError:
            return f"Error: malformed MCP tool name '{name}'"
        server = mcp_servers_db.get_server(server_id)
        if not server:
            return f"Error: MCP server '{server_id}' not found"
        if not server.get("is_enabled"):
            return f"Error: MCP server '{server_id}' is disabled"
        if is_stdio(server["url"]) and not stdio_mcp_enabled():
            return f"Error: stdio MCP server '{server_id}' is disabled on this host"
        return call_tool_sync(
            url = server["url"],
            headers = parse_server_headers(server),
            name = tool_name,
            args = arguments,
            timeout = effective_timeout,
            use_oauth = bool(server.get("use_oauth")),
            cancel_event = cancel_event,
        )
    if name == "web_search":
        return _web_search(
            arguments.get("query", ""),
            url = arguments.get("url"),
            timeout = effective_timeout,
        )
    if name == "python":
        return _python_exec(
            arguments.get("code", ""),
            cancel_event,
            effective_timeout,
            session_id,
            disable_sandbox = disable_sandbox,
        )
    if name == "terminal":
        return _bash_exec(
            arguments.get("command", ""),
            cancel_event,
            effective_timeout,
            session_id,
            disable_sandbox = disable_sandbox,
        )
    return f"Unknown tool: {name}"


def _opt_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _scope_retrieval_kwargs(scope: dict) -> dict:
    """Retrieval mode from rag_scope; candidate pools and RRF come from config."""
    mode = scope.get("mode")
    return {"mode": mode if mode in ("hybrid", "dense", "lexical") else "hybrid"}


def _search_knowledge_base(arguments: dict, rag_scope: dict | None) -> str:
    """Run the RAG search bound to the hidden per-request ``rag_scope`` (the model
    supplies only ``query``/``top_k``). Lazy import; missing sqlite-vec degrades
    to a friendly message."""
    scope = rag_scope or {}
    query = (arguments or {}).get("query", "")
    if not query or not str(query).strip():
        return "Error: query is empty."
    try:
        from storage import rag_db
        if not rag_db.RAG_AVAILABLE:
            return "Knowledge base search is unavailable on this server."
        from core.rag.tool import search_knowledge_base_with_sources
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG tool unavailable: %s", exc)
        return "Knowledge base search is unavailable on this server."

    top_k = _opt_int((arguments or {}).get("top_k") or scope.get("default_top_k"))
    text, sources = search_knowledge_base_with_sources(
        query = str(query),
        scope_kb_id = scope.get("kb_id"),
        scope_thread_id = scope.get("thread_id"),
        scope_project_id = scope.get("project_id"),
        top_k = top_k,
        **_scope_retrieval_kwargs(scope),
    )
    # Append the UI source-map after the sentinel; loops strip it before the model.
    if sources:
        import json as _json
        return text + RAG_SOURCES_SENTINEL + _json.dumps(sources, ensure_ascii = False)
    return text


# Forced first-pass RAG retrieval: a high cosine floor keeps it precise (fires on
# on-topic queries, skips weak ones) and helps small models that under-call the tool.
# Tunable via RAG_AUTOINJECT_MIN_SCORE.
_AUTOINJECT_DEFAULT_FLOOR = 0.70


def _autoinject_enabled() -> bool:
    return os.environ.get("RAG_AUTOINJECT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _autoinject_floor() -> float:
    raw = os.environ.get("RAG_AUTOINJECT_MIN_SCORE")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return _AUTOINJECT_DEFAULT_FLOOR


# Lean: injecting the full top_k every turn prefills thousands of tokens.
_AUTOINJECT_DEFAULT_TOP_K = 4


def _autoinject_top_k() -> int:
    raw = os.environ.get("RAG_AUTOINJECT_TOP_K")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _AUTOINJECT_DEFAULT_TOP_K


def _thread_whole_doc_enabled(scope: dict) -> bool:
    """Whether a thread-attached file should be injected in full rather than
    retrieved top-K. ``rag_scope.whole_doc=False`` disables it for this request."""
    override = scope.get("whole_doc")
    if override is False:
        return False
    try:
        from core.rag import config as _rag_config
    except Exception:  # noqa: BLE001
        return True
    return _rag_config.THREAD_WHOLE_DOC


_IMAGE_PART_TOKEN_ESTIMATE = 1024


def _message_token_estimate(conversation: list[dict]) -> int:
    """Cheap prompt-size estimate for budget guards; exact tokenization happens later."""
    total = 0
    for msg in conversation:
        content = msg.get("content")
        if isinstance(content, str):
            total += max(1, len(content) // 4)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("image_url", "input_image"):
                        total += _IMAGE_PART_TOKEN_ESTIMATE
                    else:
                        total += max(1, len(str(part.get("text") or "")) // 4)
        total += 4  # chat-template role / separator overhead estimate
    return total


def _whole_doc_budget(scope: dict | None = None, conversation: list[dict] | None = None) -> int:
    try:
        from core.rag import config as _rag_config
    except Exception:  # noqa: BLE001
        budget = 6000
    else:
        budget = _rag_config.WHOLE_DOC_MAX_TOKENS
    if not scope:
        return budget
    context = _opt_int(scope.get("context_length") or scope.get("max_context_tokens"))
    if context is None or context <= 0:
        return budget
    headroom = _opt_int(scope.get("response_headroom"))
    if headroom is None:
        headroom = max(1024, context // 4)
    used = _message_token_estimate(conversation or [])
    # Leave room for tool XML wrappers, citation metadata, and chat-template overhead.
    available = context - headroom - used - 512
    return min(budget, max(0, available))


def _last_user_text(conversation: list[dict]) -> str:
    """Plain text of the most recent user turn (text parts only)."""
    for msg in reversed(conversation):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") in ("text", "input_text")
            ]
            return " ".join(t for t in parts if t).strip()
        return ""
    return ""


def build_rag_autoinject(conversation: list[dict], rag_scope: dict | None) -> dict | None:
    """Pre-retrieve the latest user turn; if a hit clears the cosine floor return
    ``{"events": [...], "messages": [...]}`` to splice into the loop, else ``None``.
    Toggle via ``rag_scope.autoinject`` (else env ``RAG_AUTOINJECT``); floor via
    ``rag_scope.autoinject_min_score`` (else env ``RAG_AUTOINJECT_MIN_SCORE``).

    Also the small-model fallback: models below ~4B often answer from memory
    instead of calling ``search_knowledge_base``, so forcing retrieval here keeps
    attachments consulted regardless of model size."""
    if not rag_scope:
        return None
    enabled = rag_scope.get("autoinject")
    if enabled is None:
        enabled = _autoinject_enabled()
    thread_id = rag_scope.get("thread_id")
    whole_doc_requested = (
        bool(thread_id) and not rag_scope.get("kb_id") and _thread_whole_doc_enabled(rag_scope)
    )
    if not enabled and not whole_doc_requested:
        return None
    query = _last_user_text(conversation)
    if not query:
        return None
    try:
        from storage import rag_db
        if not rag_db.RAG_AVAILABLE:
            return None
        from core.rag.tool import render_sources, search_for_autoinject, whole_document_context
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG auto-inject unavailable: %s", exc)
        return None

    text: str | None = None
    sources: list[dict] = []

    floor_override = rag_scope.get("autoinject_min_score")
    floor = float(floor_override) if floor_override is not None else _autoinject_floor()
    # Cap at the lean top_k, but honor a lower user setting.
    lean_k = _autoinject_top_k()
    sidebar_k = _opt_int(rag_scope.get("default_top_k"))
    top_k = min(sidebar_k, lean_k) if sidebar_k is not None else lean_k

    # Whole-document mode: a thread-attached file under budget is injected in full so
    # the model reads everything. A KB selection is exclusive, so whole-doc never
    # preempts it; in a project chat the project sources are still retrieved top-K and
    # appended under one citation numbering. Oversized files (or no thread doc) fall
    # through to the combined top-K retrieval below.
    if whole_doc_requested:
        try:
            budget = _whole_doc_budget(rag_scope, conversation)

            whole = whole_document_context(
                scope_thread_id = thread_id,
                max_tokens = budget,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG whole-document context failed: %s", exc)
            whole = None
        if whole is not None:
            text, sources = whole
            project_id = rag_scope.get("project_id")
            if project_id:
                try:
                    proj = search_for_autoinject(
                        query = query,
                        scope_project_id = project_id,
                        top_k = top_k,
                        min_dense_score = floor,
                        **_scope_retrieval_kwargs(rag_scope),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("RAG project retrieval (whole-doc companion) failed: %s", exc)
                    proj = None
                if proj is not None:
                    merged = sources + proj[1]
                    merged_text = render_sources(merged)
                    if max(1, len(merged_text) // 4) <= budget:
                        sources = merged
                        text = merged_text
            logger.info("RAG auto-inject: whole-document context (%d chunk(s))", len(sources))

    if text is None and enabled:
        try:
            found = search_for_autoinject(
                query = query,
                scope_kb_id = rag_scope.get("kb_id"),
                scope_thread_id = rag_scope.get("thread_id"),
                scope_project_id = rag_scope.get("project_id"),
                top_k = top_k,
                min_dense_score = floor,
                **_scope_retrieval_kwargs(rag_scope),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG auto-inject retrieval failed: %s", exc)
            return None
        if not found:
            logger.info("RAG auto-inject: no passage >= %.2f; skipping", floor)
            return None
        text, sources = found
    if text is None:
        return None

    import json as _json
    import uuid as _uuid

    call_id = "rag_auto_" + _uuid.uuid4().hex[:12]
    args = {"query": query}
    full_result = text + RAG_SOURCES_SENTINEL + _json.dumps(sources, ensure_ascii = False)
    events = [
        {"type": "status", "text": f"Searching documents: {query[:60]}"},
        {
            "type": "tool_start",
            "tool_name": "search_knowledge_base",
            "tool_call_id": call_id,
            "arguments": args,
        },
        {
            "type": "tool_end",
            "tool_name": "search_knowledge_base",
            "tool_call_id": call_id,
            "result": full_result,
        },
        {"type": "status", "text": ""},
    ]
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": _json.dumps(args, ensure_ascii = False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "search_knowledge_base",
            "tool_call_id": call_id,
            "content": text,
        },
    ]
    logger.info("RAG auto-inject: %d passage(s) for %r", len(sources), query[:80])
    return {"events": events, "messages": messages}


_MAX_PAGE_CHARS = 16000  # cap fetched page text (after HTML-to-MD conversion)
# Raw download cap > _MAX_PAGE_CHARS because SSR pages embed large <head>
# sections stripped during conversion; 512 KB reaches article content even
# where <head> alone is ~200 KB.
_MAX_FETCH_BYTES = 512 * 1024

_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
)

_tls_ctx = ssl.create_default_context()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection to a pinned IP, using a different hostname for SNI and
    cert verification.

    SSRF IP-pinning rewrites URLs to raw IPs; a normal HTTPSConnection would then
    send no SNI and verify the cert against the IP (both fail). This splits the
    concerns: TCP connects to the pinned IP (``host``), TLS uses ``sni_hostname``.
    """

    def __init__(self, host: str, *, sni_hostname: str, **kwargs):
        super().__init__(host, **kwargs)
        self._sni_hostname = sni_hostname

    def connect(self):
        # TCP connect to the pinned IP in self.host.
        http.client.HTTPConnection.connect(self)
        # TLS handshake with the real hostname for SNI + cert verification.
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname = self._sni_hostname,
        )


class _SNIHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler sending the correct SNI hostname during TLS handshake.

    SSRF IP-pinning breaks SNI and cert verification; this returns a
    ``_PinnedHTTPSConnection`` that connects to the pinned IP but verifies TLS
    against the original hostname.
    """

    def __init__(self, hostname: str):
        super().__init__(context = _tls_ctx)
        self._sni_hostname = hostname

    def https_open(self, req):
        return self.do_open(self._sni_connection, req)

    def _sni_connection(self, host, **kwargs):
        kwargs["context"] = _tls_ctx
        return _PinnedHTTPSConnection(host, sni_hostname = self._sni_hostname, **kwargs)


def _validate_and_resolve_host(hostname: str, port: int) -> tuple[bool, str, str]:
    """Resolve *hostname*, reject non-public IPs, return a pinned IP string.

    Returns ``(ok, reason_or_empty, resolved_ip)``. The caller should connect
    to *resolved_ip* (with a ``Host`` header) to prevent DNS rebinding between
    validation and the actual fetch.
    """
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(hostname, port, type = socket.SOCK_STREAM)
    except OSError as e:
        return False, f"Failed to resolve host: {e}", ""

    if not infos:
        return False, f"Failed to resolve host: no addresses for {hostname!r}", ""

    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        # `not ip.is_global` is the source of truth: it rejects every category
        # below PLUS shared/CGNAT (100.64.0.0/10) and benchmarking/doc ranges
        # Python marks is_private=False and is_global=False. The explicit
        # predicates only give human-readable categories in the error message.
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"Blocked: refusing to fetch non-public address {ip}.", ""

    # Return the first resolved address for pinning.
    first_ip = infos[0][4][0]
    return True, "", first_ip


def _fetch_page_text(
    url: str,
    max_chars: int = _MAX_PAGE_CHARS,
    timeout: int = 30,
) -> str:
    """Fetch a URL and return plain text content (HTML tags stripped).

    Blocks private/loopback/link-local targets (SSRF protection) and caps
    the download size to avoid unbounded memory usage.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Blocked: only http/https URLs are allowed (got {parsed.scheme!r})."
    if not parsed.hostname:
        return "Blocked: URL is missing a hostname."

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ok, reason, pinned_ip = _validate_and_resolve_host(parsed.hostname, port)
    if not ok:
        return reason

    try:
        from urllib.error import HTTPError as _HTTPError
        from urllib.parse import urljoin, urlunparse

        max_bytes = _MAX_FETCH_BYTES
        current_url = url
        current_host = parsed.hostname
        ua = random.choice(_USER_AGENTS)

        for _hop in range(5):
            # Pin to the validated IP (prevents DNS rebinding): rewrite URL to
            # the IP, set the Host header.
            cp = urlparse(current_url)
            # Bracket IPv6 addresses so the netloc is valid in a URL.
            ip_str = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
            ip_netloc = f"{ip_str}:{cp.port}" if cp.port else ip_str
            pinned_url = urlunparse(cp._replace(netloc = ip_netloc))

            opener = urllib.request.build_opener(
                _NoRedirect,
                _SNIHTTPSHandler(current_host),
            )

            req = urllib.request.Request(
                pinned_url,
                headers = {
                    "User-Agent": ua,
                    "Host": current_host,
                },
            )
            try:
                resp = opener.open(req, timeout = timeout)
            except _HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    return f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}"
                location = e.headers.get("Location")
                if not location:
                    return "Failed to fetch URL: redirect missing Location header."
                current_url = urljoin(current_url, location)
                rp = urlparse(current_url)
                if rp.scheme not in ("http", "https") or not rp.hostname:
                    return "Blocked: redirect target is not a valid http/https URL."
                rp_port = rp.port or (443 if rp.scheme == "https" else 80)
                ok2, reason2, pinned_ip = _validate_and_resolve_host(
                    rp.hostname,
                    rp_port,
                )
                if not ok2:
                    return reason2
                current_host = rp.hostname
                continue
            # Success: read capped body.
            raw_bytes = resp.read(max_bytes)
            break
        else:
            return "Failed to fetch URL: too many redirects."

        charset = resp.headers.get_content_charset() or "utf-8"
        raw_html = raw_bytes.decode(charset, errors = "replace")
    except _HTTPError as e:
        return f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}"
    except Exception as e:
        return f"Failed to fetch URL: {e}"

    # Convert HTML to Markdown with the builtin converter (no external deps).
    from ._html_to_md import html_to_markdown

    text = html_to_markdown(raw_html)

    if not text:
        return "(page returned no readable text)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


def _web_search(
    query: str,
    max_results: int = 5,
    timeout: int = _EXEC_TIMEOUT,
    url: str | None = None,
) -> str:
    """Search the web using DuckDuckGo and return formatted results.

    If ``url`` is provided, fetches that page directly instead of searching.
    """
    # Direct URL fetch mode.
    if url and url.strip():
        fetch_timeout = 60 if timeout is None else min(timeout, 60)
        return _fetch_page_text(url.strip(), timeout = fetch_timeout)

    if not query or not query.strip():
        return "No query provided."
    try:
        from ddgs import DDGS

        results = DDGS(timeout = timeout).text(query, max_results = max_results)
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"Snippet: {r.get('body', '')}"
            )
        text = "\n\n---\n\n".join(parts)
        text += (
            "\n\n---\n\nIMPORTANT: These are only short snippets. "
            "To get the full page content, call web_search with "
            'the url parameter (e.g. {"url": "<URL>"}).'
        )
        return text
    except Exception as e:
        return f"Search failed: {e}"


# ==========================================================================
# Sandbox static-analysis hardening (feature-flagged; see UNSLOTH_STUDIO_SINK_ANALYZER)
#
# A pure, whitelist-only constant folder plus a filesystem-confinement path
# resolver back the eval/exec payload recursion and the destructive-op gate.
# Everything here recomputes pure transforms on *literals only* and never runs,
# imports, or reflects on user code. All limits are bounded so the analyzer can
# never be slower or crashier than the legacy syntactic checks; on any breach a
# folder returns None (opaque) and the caller fails safe.
# ==========================================================================

# Folder bounds (Stage 1). Breaching any of these yields None ("un-foldable").
_FOLD_DEPTH = 24
_FOLD_MAXLEN = 65536
_FOLD_OPS = 4000
_FOLD_MAX_SEQ = 4096
_FOLD_MAXINT = 1 << 64

_UNKNOWN = object()  # sentinel: "not statically decidable"


class _ConstEnv(dict):
    """A const-prop env (name -> RHS node) that also carries the set of names REBOUND away from
    their canonical builtin / stdlib module in the snippet, so the folder can refuse to fold a
    shadowed helper (str = lambda _: '...'; eval(str(1))) as the real builtin."""

    __slots__ = ("shadowed",)

    def __init__(
        self,
        *a,
        shadowed = None,
        **k,
    ):
        super().__init__(*a, **k)
        self.shadowed = shadowed or frozenset()


class _FoldState:
    """Shared op counter + single-assignment const-prop environment."""

    __slots__ = ("ops", "names", "shadowed")

    def __init__(self, names = None):
        self.ops = 0
        self.names = names or {}
        self.shadowed = getattr(names, "shadowed", None) or frozenset()


def _fold_cap(value):
    """Return value unless a str/bytes exceeds the size cap or an int the magnitude cap."""
    if isinstance(value, (str, bytes, bytearray)) and len(value) > _FOLD_MAXLEN:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > _FOLD_MAXINT:
        return None
    return value


def _too_wide(n):
    """A format width / precision / size arg large enough to OOM the folder."""
    return isinstance(n, int) and not isinstance(n, bool) and n > _FOLD_MAXLEN


def _fold_repr_len_exceeds(value, budget = _FOLD_MAXLEN):
    """A cheap upper-bound walk of the str()/repr() length of an already-folded value, returning
    True as soon as the estimate exceeds ``budget``. str(container) builds the WHOLE repr before
    _fold_cap can reject it -- e.g. str(['x' * 65536] * 4096) is a list of 4096 refs to one 64KB
    string (cheap) whose repr is ~256MB -- so estimate the size WITHOUT materializing it, which is
    exactly the allocation the fold caps exist to prevent. A scalar (str/bytes/int/float/bool/None)
    is already bounded by the input caps and never trips this."""
    _stack = [value]
    _total = 0
    _seen = 0
    while _stack:
        _v = _stack.pop()
        _seen += 1
        if _seen > _FOLD_OPS:
            return True
        if isinstance(_v, (str, bytes, bytearray)):
            _total += len(_v) + 3  # quotes / b'' overhead
        elif isinstance(_v, bool):
            _total += 5
        elif isinstance(_v, int):
            _total += 20 if abs(_v) <= _FOLD_MAXINT else budget + 1
        elif isinstance(_v, float):
            _total += 24
        elif _v is None:
            _total += 4
        elif isinstance(_v, (list, tuple, set, frozenset)):
            _total += 2 + 2 * len(_v)  # brackets + ", " separators
            _stack.extend(_v)
        elif isinstance(_v, dict):
            _total += 2 + 4 * len(_v)  # braces + ": " / ", " separators
            for _dk, _dv in _v.items():
                _stack.append(_dk)
                _stack.append(_dv)
        else:
            _total += 16
        if _total > budget:
            return True
    return False


# Format-spec mini-language: reject an oversized width or precision BEFORE format()
# allocates the padded string. format()/str.format()/f-strings all run in the Studio
# process during static analysis, ahead of the child-subprocess rlimits.
_FMT_SPEC_RE = re.compile(r"^(?:.?[<>=^])?[+\- ]?z?#?0?(\d+)?[,_]?(?:\.(\d+))?[a-zA-Z%]?$")


def _format_spec_ok(spec):
    if not isinstance(spec, str) or not spec:
        return True
    m = _FMT_SPEC_RE.match(spec)
    if not m:
        return True  # unrecognized spec: let format() itself decide at runtime
    return not any(g and _too_wide(int(g)) for g in m.groups())


def _format_template_ok(template):
    """Every replacement field of a str.format template has a bounded width."""
    if not isinstance(template, str):
        return True
    try:
        import string as _string
        for _lit, _field, _spec, _conv in _string.Formatter().parse(template):
            if _spec and not _format_spec_ok(_spec):
                return False
    except Exception:
        return True
    return True


def _format_has_nested_spec(template):
    """A replacement field whose spec itself contains a field ({:{}}): the width is
    supplied by an argument, so a large numeric arg drives the allocation."""
    if not isinstance(template, str):
        return False
    try:
        import string as _string
        for _lit, _field, _spec, _conv in _string.Formatter().parse(template):
            if _spec and "{" in _spec:
                return True
    except Exception:
        return False
    return False


_PRINTF_WIDTH_RE = re.compile(r"%[-+ #0]*(\d+)?(?:\.(\d+))?[hlL]?[diouxXeEfFgGcrsab%]")


def _printf_ok(fmt):
    """Percent-format string with no oversized (or dynamic '*') width / precision."""
    if isinstance(fmt, (bytes, bytearray)):
        try:
            fmt = fmt.decode("latin-1")
        except Exception:
            return True
    if not isinstance(fmt, str):
        return True
    # A '*' width or precision ('%*s', '%.*f') pulls its size from a runtime argument,
    # so it cannot be bounded statically -- refuse rather than risk a large allocation.
    for m in re.finditer(r"%[-+ #0]*(\*)?(?:\.(\*)?\d*)?", fmt):
        if m.group(1) == "*" or m.group(2) == "*":
            return False
    for m in _PRINTF_WIDTH_RE.finditer(fmt):
        if any(g and _too_wide(int(g)) for g in m.groups()):
            return False
    return True


def _replace_output_ok(recv, call_args):
    """Bound str.replace/bytes.replace output before it allocates: replacing many
    occurrences with a long replacement can build a multi-gigabyte string."""
    if len(call_args) < 2:
        return True
    old, new = call_args[0], call_args[1]
    if not isinstance(new, (str, bytes, bytearray)):
        return True
    lo = len(old) if isinstance(old, (str, bytes, bytearray)) else 1
    n_repl = (len(recv) + 1) if lo == 0 else (len(recv) // max(lo, 1) + 1)
    if len(call_args) >= 3 and isinstance(call_args[2], int) and call_args[2] >= 0:
        n_repl = min(n_repl, call_args[2])
    return len(recv) + n_repl * len(new) <= _FOLD_MAXLEN


def _join_output_ok(sep, call_args):
    """Bound str.join/bytes.join output before it allocates."""
    if not call_args or not isinstance(call_args[0], (list, tuple)):
        return True
    items = call_args[0]
    total = len(sep) * max(len(items) - 1, 0)
    for x in items:
        if not isinstance(x, (str, bytes, bytearray)):
            return True  # a real join would TypeError; not an allocation concern
        total += len(x)
        if total > _FOLD_MAXLEN:
            return False
    return True


def _fold_apply_codec(name, data):
    """Pure data transforms only (rot13/hex/base64/zlib/text codecs). Bounded zlib."""
    name = name.lower().replace("-", "_")
    try:
        if name in ("rot_13", "rot13"):
            text = data if isinstance(data, str) else data.decode("latin-1")
            return codecs.decode(text, "rot_13")
        if name == "hex":
            return codecs.decode(data, "hex")
        if name in ("base64", "base_64"):
            return base64.b64decode(data if isinstance(data, (bytes, bytearray)) else data.encode())
        if name == "zlib":
            payload = data if isinstance(data, (bytes, bytearray)) else str(data).encode()
            d = zlib.decompressobj()
            out = d.decompress(payload, _FOLD_MAXLEN)
            if d.unconsumed_tail:  # would exceed the cap -> refuse
                return None
            return out
        if name in ("utf_8", "utf8", "latin_1", "latin1", "ascii"):
            if isinstance(data, (bytes, bytearray)):
                return data.decode(name)
            return data.encode(name)
    except Exception:
        return None
    return None  # bz2/lzma/gzip and unknowns: bomb-unsafe / opaque -> refuse


_FOLD_PURE_BUILTINS = frozenset(
    {"chr", "ord", "str", "int", "bytes", "bytearray", "hex", "oct", "bin", "bool", "float", "len"}
)
_FOLD_STR_METHODS = frozenset(
    {
        "join",
        "replace",
        "upper",
        "lower",
        "strip",
        "lstrip",
        "rstrip",
        "swapcase",
        "title",
        "capitalize",
        "format",
        "zfill",
        "ljust",
        "rjust",
        "center",
        "encode",
        "decode",
    }
)
_FOLD_B64_FUNCS = frozenset(
    {
        "b64decode",
        "b64encode",
        "urlsafe_b64decode",
        "standard_b64decode",
        "b32decode",
        "b16decode",
        "a85decode",
        "b85decode",
    }
)


def _const_fold(
    node,
    env = None,
    _state = None,
    _depth = 0,
):
    """Fold an AST expression to a concrete str/bytes/int/list value, else None.

    Whitelist-only and pure: it never executes user code, never imports, never
    reflects. Only a fixed set of pure transforms over already-folded literals
    (concat/repeat/join/format/slice/reverse, base64/hex/rot13/zlib decode, and
    a handful of pure builtins/str methods) is supported; anything else returns
    None. ``env`` maps single-assignment module-level names to their RHS nodes.
    """
    if _state is None:
        _state = _FoldState(env)
    _state.ops += 1
    if node is None or _depth > _FOLD_DEPTH or _state.ops > _FOLD_OPS:
        return None

    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, (str, bytes, bytearray, int, float)) or v is None:
            return _fold_cap(v)
        return None

    if isinstance(node, ast.Name):
        rhs = _state.names.get(node.id)
        if rhs is None:
            return None
        return _const_fold(rhs, None, _state, _depth + 1)

    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > _FOLD_MAX_SEQ:
            return None
        vals = []
        for e in node.elts:
            v = _const_fold(e, None, _state, _depth + 1)
            if v is None and not (isinstance(e, ast.Constant) and e.value is None):
                return None
            vals.append(v)
        return vals

    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                out.append(str(part.value))
            elif isinstance(part, ast.FormattedValue):
                v = _const_fold(part.value, None, _state, _depth + 1)
                if v is None:
                    return None
                spec = ""
                if part.format_spec is not None:
                    spec = _const_fold(part.format_spec, None, _state, _depth + 1)
                    if spec is None:
                        return None
                if part.conversion and part.conversion != -1:
                    try:
                        v = {114: repr, 115: str, 97: ascii}[part.conversion](v)
                    except Exception:
                        return None
                if not _format_spec_ok(spec if isinstance(spec, str) else ""):
                    return None  # oversized f-string width/precision: refuse pre-format
                try:
                    out.append(format(v, spec if isinstance(spec, str) else ""))
                except Exception:
                    return None
            else:
                return None
        return _fold_cap("".join(out))

    if isinstance(node, ast.BinOp):
        left = _const_fold(node.left, None, _state, _depth + 1)
        right = _const_fold(node.right, None, _state, _depth + 1)
        if left is None or right is None:
            return None
        op = node.op
        try:
            if isinstance(op, ast.Mult):
                if isinstance(left, (str, bytes, bytearray)) and isinstance(right, int):
                    if len(left) * max(right, 0) > _FOLD_MAXLEN:
                        return None
                if isinstance(right, (str, bytes, bytearray)) and isinstance(left, int):
                    if len(right) * max(left, 0) > _FOLD_MAXLEN:
                        return None
                # list/tuple repetition allocates len(seq)*n elements before _fold_cap
                # (which only sizes str/bytes) can reject it -- cap it here too.
                if isinstance(left, (list, tuple)) and isinstance(right, int):
                    if len(left) * max(right, 0) > _FOLD_MAX_SEQ:
                        return None
                if isinstance(right, (list, tuple)) and isinstance(left, int):
                    if len(right) * max(left, 0) > _FOLD_MAX_SEQ:
                        return None
                return _fold_cap(left * right)
            if isinstance(op, ast.Add):
                # str/bytes concat is sized by _fold_cap, but list/tuple concatenation is
                # not, so a chain (a + a + a + ...) materializes an oversized sequence in the
                # parent process before child rlimits apply. Cap the combined length.
                if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                    if len(left) + len(right) > _FOLD_MAX_SEQ:
                        return None
                return _fold_cap(left + right)
            if isinstance(op, ast.Mod):
                if isinstance(left, (str, bytes, bytearray)) and not _printf_ok(left):
                    return None  # oversized %-format width/precision: refuse pre-format
                return _fold_cap(left % right)
            if isinstance(op, ast.Sub):
                return _fold_cap(left - right)
            if isinstance(op, ast.FloorDiv):
                return _fold_cap(left // right)
            if isinstance(op, ast.Div):
                return _fold_cap(left / right)
            if isinstance(op, ast.BitXor):
                return _fold_cap(left ^ right)
            if isinstance(op, ast.BitOr):
                return _fold_cap(left | right)
            if isinstance(op, ast.BitAnd):
                return _fold_cap(left & right)
            if isinstance(op, ast.LShift) and isinstance(right, int) and 0 <= right < 64:
                return _fold_cap(left << right)
            if isinstance(op, ast.RShift) and isinstance(right, int) and 0 <= right < 64:
                return _fold_cap(left >> right)
        except Exception:
            return None
        return None  # Pow and others: refuse (bignum DoS)

    if isinstance(node, ast.UnaryOp):
        v = _const_fold(node.operand, None, _state, _depth + 1)
        if v is None:
            return None
        try:
            return {
                ast.USub: lambda x: -x,
                ast.UAdd: lambda x: +x,
                ast.Invert: lambda x: ~x,
                ast.Not: lambda x: not x,
            }[type(node.op)](v)
        except Exception:
            return None

    if isinstance(node, ast.Subscript):
        base = _const_fold(node.value, None, _state, _depth + 1)
        if base is None or not isinstance(base, (str, bytes, bytearray, list, tuple)):
            return None
        sl = node.slice
        try:
            if isinstance(sl, ast.Slice):
                lo = _const_fold(sl.lower, None, _state, _depth + 1) if sl.lower else None
                hi = _const_fold(sl.upper, None, _state, _depth + 1) if sl.upper else None
                st = _const_fold(sl.step, None, _state, _depth + 1) if sl.step else None
                if (
                    (sl.lower is not None and lo is None)
                    or (sl.upper is not None and hi is None)
                    or (sl.step is not None and st is None)
                ):
                    return None
                return _fold_cap(base[lo:hi:st])
            idx = _const_fold(sl, None, _state, _depth + 1)
            if not isinstance(idx, int):
                return None
            return _fold_cap(base[idx])
        except Exception:
            return None

    if isinstance(node, ast.Call):
        return _fold_call(node, _state, _depth)

    return None


def _is_path_join_owner(nv):
    """AST for ``os.path`` (Attribute) or ``posixpath`` / ``ntpath`` (Name)."""
    if (
        isinstance(nv, ast.Attribute)
        and nv.attr == "path"
        and isinstance(nv.value, ast.Name)
        and nv.value.id == "os"
    ):
        return True
    return isinstance(nv, ast.Name) and nv.id in ("posixpath", "ntpath")


def _fold_call(node, _state, _depth):
    """Fold a whitelisted pure builtin / method / decode call, else None."""
    f = node.func
    args = []
    for a in node.args:
        v = _const_fold(a, None, _state, _depth + 1)
        if v is None and not (isinstance(a, ast.Constant) and a.value is None):
            return None
        args.append(v)

    if isinstance(f, ast.Name):
        name = f.id
        if name not in _FOLD_PURE_BUILTINS:
            return None
        # A snippet that rebinds the builtin name (str = lambda _: '...'; eval(str(1))) makes the
        # real-builtin fold diverge from runtime; refuse so the payload stays opaque (fail closed).
        if name in _state.shadowed:
            return None
        try:
            if name == "chr":
                if len(args) == 1 and isinstance(args[0], int) and 0 <= args[0] <= 0x10FFFF:
                    return chr(args[0])
                return None
            if name == "ord":
                if (
                    len(args) == 1
                    and isinstance(args[0], (str, bytes, bytearray))
                    and len(args[0]) == 1
                ):
                    return ord(args[0])
                return None
            fn = {
                "str": str,
                "bytes": bytes,
                "bytearray": bytearray,
                "int": int,
                "hex": hex,
                "oct": oct,
                "bin": bin,
                "bool": bool,
                "float": float,
                "len": len,
            }[name]
            if name in ("bytes", "bytearray") and len(args) == 1:
                # bytes(n) / bytearray(n) allocate n zero bytes; refuse an oversized
                # integer size before constructing it so static analysis of e.g.
                # bytes(2_000_000_000) cannot OOM the Studio process (the child
                # sandbox rlimits never get a chance to help during analysis).
                if (
                    isinstance(args[0], int)
                    and not isinstance(args[0], bool)
                    and args[0] > _FOLD_MAXLEN
                ):
                    return None
            if name == "str" and len(args) == 1 and _fold_repr_len_exceeds(args[0]):
                # str(container) materializes the ENTIRE repr before _fold_cap sees its length;
                # a small aliased container (['x' * 65536] * 4096) expands to hundreds of MB and
                # OOMs the Studio process ahead of the child rlimits. Estimate the size first and
                # refuse (leaving the payload opaque, which already fails closed) if it exceeds cap.
                return None
            return _fold_cap(fn(*args))
        except Exception:
            return None

    if isinstance(f, ast.Attribute):
        attr = f.attr
        owner = f.value
        # os.path.join('/etc', 'passwd') / posixpath.join(...) / ntpath.join(...):
        # fold literal path builders so the sensitive-read scanner sees the concrete
        # path (open(os.path.join('/etc','passwd')) must not be treated as opaque).
        if attr == "join" and _is_path_join_owner(owner):
            if args and all(isinstance(x, str) for x in args):
                try:
                    return _fold_cap(os.path.join(*args))
                except Exception:
                    return None
            return None
        # os.path.normpath / abspath on a literal reveal the same sensitive / traversal
        # path a direct string would (open(os.path.normpath('a/../../etc/passwd')) must
        # not stay opaque). normpath is a pure string transform; abspath is only foldable
        # for an already-absolute arg (a relative abspath depends on the runtime cwd,
        # which is the in-workdir sandbox cwd, so it need not be folded).
        if attr in ("normpath", "abspath") and _is_path_join_owner(owner):
            if len(args) == 1 and isinstance(args[0], str):
                try:
                    if attr == "normpath" or os.path.isabs(args[0]):
                        return _fold_cap(os.path.normpath(args[0]))
                except Exception:
                    return None
            return None
        if isinstance(owner, ast.Name):
            mod = owner.id
            # A rebound module receiver (base64 = <fake>; eval(base64.b64decode('...'))) would fold
            # through the real stdlib module while runtime uses the user binding; refuse the fold.
            if mod in _state.shadowed:
                return None
            try:
                if mod == "base64" and attr in _FOLD_B64_FUNCS and len(args) >= 1:
                    return _fold_cap(getattr(base64, attr)(args[0]))
                if (
                    mod == "codecs"
                    and attr in ("decode", "encode")
                    and len(args) >= 2
                    and isinstance(args[1], str)
                ):
                    return _fold_cap(_fold_apply_codec(args[1], args[0]))
                if mod == "binascii" and attr in ("unhexlify", "a2b_hex") and len(args) >= 1:
                    return _fold_cap(binascii.unhexlify(args[0]))
                if (
                    mod in ("bytes", "bytearray")
                    and attr == "fromhex"
                    and len(args) >= 1
                    and isinstance(args[0], str)
                ):
                    return _fold_cap(bytes.fromhex(args[0]))
            except Exception:
                return None
        recv = _const_fold(owner, None, _state, _depth + 1)
        if isinstance(recv, (str, bytes, bytearray)) and attr in _FOLD_STR_METHODS:
            try:
                kwargs = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        return None
                    kv = _const_fold(kw.value, None, _state, _depth + 1)
                    if kv is None:
                        return None
                    kwargs[kw.arg] = kv
                call_args = []
                for a in args:
                    call_args.append(
                        list(a) if attr == "join" and isinstance(a, (list, tuple)) else a
                    )
                # Padding methods take a width as their first arg; str.format takes a
                # template with per-field widths. Reject an oversized width before the
                # method allocates the padded string during folding.
                if attr in ("center", "ljust", "rjust", "zfill") and call_args:
                    if _too_wide(call_args[0]):
                        return None
                if attr == "replace" and not _replace_output_ok(recv, call_args):
                    return None
                if attr == "join" and not _join_output_ok(recv, call_args):
                    return None
                if attr == "format":
                    if not _format_template_ok(recv):
                        return None
                    # Nested-width field ({:{}}) whose width comes from a large numeric
                    # arg: refuse before format() allocates.
                    if _format_has_nested_spec(recv) and any(
                        _too_wide(a) for a in list(call_args) + list(kwargs.values())
                    ):
                        return None
                return _fold_cap(getattr(recv, attr)(*call_args, **kwargs))
            except Exception:
                return None
    return None


def _build_const_prop_env(tree):
    """Names bound exactly once by a module-level ``name = <expr>`` (single Name
    target), never re-assigned / aug-assigned / declared global-nonlocal / used as
    a loop / comprehension / with / except target. Maps name -> RHS node.

    Conservative: any ambiguity excludes the name. Only module-level statements are
    considered so a name shadowed inside a def / loop is never folded.
    """
    assigned_once: dict[str, ast.expr] = {}
    disqualified: set[str] = set()

    def _disqualify_targets(target):
        for n in ast.walk(target):
            if isinstance(n, ast.Name):
                disqualified.add(n.id)

    # Module-level single assignments.
    for stmt in getattr(tree, "body", []):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            name = stmt.targets[0].id
            if name in assigned_once or name in disqualified:
                disqualified.add(name)
                assigned_once.pop(name, None)
            else:
                assigned_once[name] = stmt.value
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                _disqualify_targets(t)
        elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
            if getattr(stmt, "target", None) is not None:
                _disqualify_targets(stmt.target)

    # Any name that is ALSO written anywhere else (loops, defs, walrus, aug, params,
    # comprehension targets, with/except/for) is disqualified.
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            nm = n.id
            if nm in assigned_once:
                # It is stored somewhere; allow only if that single store is the
                # module-level assign we recorded (identity check below).
                pass
        if isinstance(n, (ast.AugAssign,)):
            _disqualify_targets(n.target)
        elif isinstance(n, ast.NamedExpr):
            _disqualify_targets(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            _disqualify_targets(n.target)
        elif isinstance(n, ast.comprehension):
            _disqualify_targets(n.target)
        elif isinstance(n, ast.withitem):
            if n.optional_vars is not None:
                _disqualify_targets(n.optional_vars)
        elif isinstance(n, ast.ExceptHandler):
            if n.name:
                disqualified.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            for nm in n.names:
                disqualified.add(nm)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            disqualified.add(n.name)
            args = getattr(n, "args", None)
            if args is not None:
                for a in list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs):
                    disqualified.add(a.arg)
                for extra in (args.vararg, args.kwarg):
                    if extra is not None:
                        disqualified.add(extra.arg)

    # A write THROUGH the namespace dict (globals()['x'] = BAD, vars()['x'] = BAD, locals()[...]
    # = ..., or globals().update(...) / .setdefault(...) / .__setitem__(...)) mutates a module
    # variable with NO Name Store, so a folded constant would be stale and the recovered exec/eval
    # payload wrong. Invalidate the affected name (constant key) or, for a dynamic key / bulk
    # update, every recorded name -- the snippet is manipulating the namespace opaquely.
    def _is_namespace_call(nv):
        return (
            isinstance(nv, ast.Call)
            and isinstance(nv.func, ast.Name)
            and nv.func.id in ("globals", "vars", "locals")
        )

    _ns_write_all = False
    _ns_write_names: set[str] = set()
    for n in ast.walk(tree):
        # globals()[key] = ... (Assign target or AugAssign target).
        _subs = []
        if isinstance(n, ast.Assign):
            _subs = [t for t in n.targets if isinstance(t, ast.Subscript)]
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(getattr(n, "target", None), ast.Subscript):
                _subs = [n.target]
        for _t in _subs:
            if not _is_namespace_call(_t.value):
                continue
            _key = _t.slice.value if isinstance(_t.slice, ast.Constant) else None
            if isinstance(_key, str):
                _ns_write_names.add(_key)
            else:
                _ns_write_all = True
        # globals().update(...) / .setdefault(...) / .__setitem__(...) -- an opaque bulk write.
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("update", "setdefault", "__setitem__", "pop", "clear")
            and _is_namespace_call(n.func.value)
        ):
            _ns_write_all = True

    # Count how many module-level stores each recorded name really has; if more
    # than one Store target references it anywhere, drop it.
    store_counts: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            store_counts[n.id] = store_counts.get(n.id, 0) + 1

    # Names REBOUND to something OTHER than their canonical builtin / stdlib module: a fold that
    # calls the real builtin (str(...), len(...)) or hard-coded module (base64.b64decode(...))
    # would diverge from runtime, which calls the user binding. A plain `import name` keeps the
    # canonical module (NOT shadowing); every other binding -- assignment, def/class, param,
    # from-import, an aliased import that rebinds the name, or a loop/with/except/comprehension
    # target -- is. The folder consults this set before folding a Name builtin / module receiver.
    shadowed: set[str] = set()

    def _shadow_targets(t):
        for nn in ast.walk(t):
            if isinstance(nn, ast.Name) and isinstance(nn.ctx, (ast.Store, ast.Del)):
                shadowed.add(nn.id)

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                _shadow_targets(t)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            if getattr(n, "target", None) is not None:
                _shadow_targets(n.target)
        elif isinstance(n, ast.NamedExpr):
            _shadow_targets(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            _shadow_targets(n.target)
        elif isinstance(n, ast.comprehension):
            _shadow_targets(n.target)
        elif isinstance(n, ast.withitem):
            if n.optional_vars is not None:
                _shadow_targets(n.optional_vars)
        elif isinstance(n, ast.ExceptHandler):
            if n.name:
                shadowed.add(n.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(n.name)
            _a = getattr(n, "args", None)
            if _a is not None:
                for _p in list(_a.args) + list(_a.posonlyargs) + list(_a.kwonlyargs):
                    shadowed.add(_p.arg)
                for _extra in (_a.vararg, _a.kwarg):
                    if _extra is not None:
                        shadowed.add(_extra.arg)
        elif isinstance(n, ast.ImportFrom):
            for _al in n.names:
                shadowed.add(_al.asname or _al.name)
        elif isinstance(n, ast.Import):
            for _al in n.names:
                # import os as base64 rebinds `base64` to a different module; a plain
                # `import base64` (asname None) keeps the canonical module and does not shadow.
                if _al.asname is not None and _al.asname != _al.name:
                    shadowed.add(_al.asname)

    env = _ConstEnv(shadowed = frozenset(shadowed))
    if _ns_write_all:
        return env  # an opaque namespace mutation could rebind any recorded constant
    disqualified |= _ns_write_names
    for name, rhs in assigned_once.items():
        if name in disqualified:
            continue
        if store_counts.get(name, 0) != 1:
            continue
        env[name] = rhs
    return env


# --------------------------------------------------------------------------
# Stage 2: eval / exec / compile recursive payload analysis.
# --------------------------------------------------------------------------
_MAX_UNWRAP_DEPTH = 5
_MAX_INNER_SRC = 200 * 1024
_MAX_INNER_PARSES = 25
_MAX_TOTAL_INNER_CHARS = 2 * 1024 * 1024
_MAX_BRACKET_DEPTH = 200
_MAX_ANALYZER_NODES = 200_000

_EXEC_BUILTINS = frozenset({"eval", "exec", "compile"})
# Deserialization sinks that reconstruct/execute arbitrary objects from bytes.
_CODE_DESERIALIZE_SINKS = frozenset(
    {
        "pickle.loads",
        "pickle.load",
        "marshal.loads",
        "marshal.load",
        "dill.loads",
        "dill.load",
        "cloudpickle.loads",
        "cloudpickle.load",
        "_pickle.loads",
        "_pickle.load",
        "jsonpickle.decode",
        # PyYAML: unsafe_load / full_load construct arbitrary Python objects from
        # `!!python/object/apply:os.system [...]`. yaml.load is conditional (safe only with a
        # SafeLoader) and is handled separately.
        "yaml.unsafe_load",
        "yaml.unsafe_load_all",
        "yaml.full_load",
        "yaml.full_load_all",
        # shelve is a dbm-backed dict that UNPICKLES a value on every read (shelf[key],
        # shelf.get(key)), so shelve.open() on an attacker-planted dbm runs a pickle reduce
        # payload just like pickle.load. The read can be aliased (d = shelve.open(...); d[k]),
        # so the open() gateway call is flagged rather than only the direct-chain subscript.
        # (A pure write-only shelf never unpickles; blocking it is an accepted narrow tradeoff.)
        "shelve.open",
    }
)
# Modules whose load/loads/decode entry points run a pickle reduce payload; used to
# resolve `import pickle as p; p.loads(x)` and `from pickle import loads as l`.
_DESERIALIZE_MODULES = frozenset(
    {"pickle", "marshal", "dill", "cloudpickle", "_pickle", "jsonpickle", "yaml", "shelve"}
)
# Modules exposing an Unpickler class whose .load() runs the same reduce payload as *.load:
# pickle.Unpickler(f).load() / dill.Unpickler(f).load() bypass the *.load sink-name check.
_UNPICKLER_MODULES = frozenset({"pickle", "_pickle", "dill", "cloudpickle"})
# yaml.load / yaml.load_all construct arbitrary objects UNLESS given a safe loader; flag them
# when the Loader is absent or is one of the unsafe loader classes.
_YAML_SAFE_LOADERS = frozenset({"SafeLoader", "CSafeLoader", "BaseLoader"})
_YAML_LOAD_METHODS = frozenset({"load", "load_all"})

# Pickle-backed loaders whose reduce payload executes arbitrary code, gated by a keyword like
# yaml.load: torch.load runs a pickle unless weights_only is True (torch>=2.6 default), numpy.load
# only unpickles with allow_pickle=True, and joblib.load is always pickle-backed. Tracked
# separately from _CODE_DESERIALIZE_SINKS because the safe default forms must stay allowed.
_PICKLE_LOADER_MODULES = frozenset({"torch", "numpy", "joblib"})


def _kw_constant_truthy(node, name):
    """The literal truthiness of keyword ``name`` in a call: True/False when it is a constant,
    None when absent or non-constant. Used for the weights_only / allow_pickle gates."""
    for kw in node.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
            return None
    return None


def _kw_present(node, name):
    """True when keyword ``name`` is passed in the call (any value)."""
    return any(kw.arg == name for kw in node.keywords)


def _numpy_allow_pickle_unsafe(node):
    """True when numpy.load may unpickle: allow_pickle is a constant truthy, a present-but-non-
    constant value (flag=True), or hidden in a **kwargs splat we cannot prove absent/False.
    allow_pickle absent (default False) or a constant False stays safe."""
    for kw in node.keywords:
        if kw.arg == "allow_pickle":
            if isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
            return True  # non-literal value cannot be proven False
        if kw.arg is None:
            # **{...} / **var splat: inspect a literal dict, else fail closed.
            if isinstance(kw.value, ast.Dict):
                for _k, _v in zip(kw.value.keys, kw.value.values):
                    if isinstance(_k, ast.Constant) and _k.value == "allow_pickle":
                        if not isinstance(_v, ast.Constant) or bool(_v.value):
                            return True
            else:
                return True  # opaque **var could carry allow_pickle=True
    return False


def _pickle_loader_is_unsafe(fq, node):
    """True when a torch.load / numpy.load / joblib.load call runs an UNVERIFIED pickle payload:
    joblib.load always does; torch.load when weights_only is EXPLICITLY not-True (torch>=2.6
    defaults it to True, so the bare torch.load(f) form relies on that safe default and stays
    allowed); numpy.load when allow_pickle is truthy / non-literal / splatted (see
    _numpy_allow_pickle_unsafe). So the safe forms (torch.load(f), torch.load(f,
    weights_only=True), numpy.load(f), numpy.load(f, allow_pickle=False)) return False."""
    if fq == "joblib.load":
        return True
    if fq == "torch.load":
        return (
            _kw_present(node, "weights_only")
            and _kw_constant_truthy(node, "weights_only") is not True
        )
    if fq == "numpy.load":
        return _numpy_allow_pickle_unsafe(node)
    return False


def _yaml_loader_class_name(value):
    """Terminal attribute/name of a Loader= argument (yaml.SafeLoader -> SafeLoader)."""
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Name):
        return value.id
    return None


def _yaml_call_has_safe_loader(node):
    """True only when a yaml.load(...) call passes an explicit safe loader.

    PyYAML's signature is load(stream, Loader), so the loader may be the SECOND POSITIONAL
    argument (yaml.load(data, yaml.SafeLoader)) or the Loader= keyword. A missing loader (older
    PyYAML defaults to the full, unsafe loader), an unknown/computed loader, or a **kwargs splat
    all fail closed so the call is treated as an unsafe sink.
    """
    if len(node.args) >= 2:
        return _yaml_loader_class_name(node.args[1]) in _YAML_SAFE_LOADERS
    for kw in node.keywords:
        if kw.arg == "Loader":
            return _yaml_loader_class_name(kw.value) in _YAML_SAFE_LOADERS
        if kw.arg is None:
            # **kwargs unpacking hides the loader; cannot prove it is safe.
            return False
    return False


# Attribute names of pure decode/decompress primitives used to hide a payload.
_DECODE_ATTRS = frozenset(
    {
        "b64decode",
        "b64encode",
        "urlsafe_b64decode",
        "standard_b64decode",
        "b32decode",
        "b16decode",
        "a85decode",
        "b85decode",
        "decodebytes",
        "fromhex",
        "unhexlify",
        "a2b_hex",
        "a2b_base64",
        "decompress",
    }
)
_FETCH_FQ_PREFIXES = (
    "requests.",
    "urllib.",
    "httpx.",
    "socket.",
    "aiohttp.",
    "urllib3.",
    "http.client.",
)
# Constant attribute names that, resolved off a sensitive module via getattr, still
# reach shell / process / delete / dynamic-import / code-exec capabilities. A benign
# constant attr (getpid, path, sep, getcwd, ...) is allowed; a dynamic attr blocks.
_DANGEROUS_ATTR_NAMES = frozenset(
    {
        "system",
        "popen",
        "popen2",
        "popen3",
        "popen4",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "posix_spawn",
        "posix_spawnp",
        "startfile",
        "fork",
        "forkpty",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "renames",
        "replace",
        "truncate",
        "chmod",
        "lchmod",
        "chown",
        "lchown",
        "chflags",
        "mkdir",
        "makedirs",
        "mknod",
        "symlink",
        "link",
        "chdir",
        "chroot",
        "import_module",
        "__import__",
        "reload",
        "eval",
        "exec",
        "compile",
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "getoutput",
        "getstatusoutput",
        "load_module",
        "exec_module",
        "loads",
        "load",
    }
)


class _AnalyzerBudget:
    """Shared, bounded counters across one classification (incl. exec recursion)."""

    __slots__ = ("inner_parses", "inner_chars", "nodes")

    def __init__(self):
        self.inner_parses = 0
        self.inner_chars = 0
        self.nodes = 0


def _reexport_dangerous_module_name(expr):
    """For a re-export gadget that fetches a submodule by NAME off some object and then calls a
    sink on it, return the fetched module name normalized to 'os' / 'subprocess' (or None).

    Covers ``getattr(<mod>, 'os')``, ``vars(<mod>)['os']`` and ``<mod>.__dict__['os']`` -- the
    call / subscript twins of the plain ``<mod>.os`` attribute form (pathlib.os.system), which
    stay reachable off a call-returned module (``getattr(__import__('pathlib'), 'os').system``)."""

    def _str_const(n):
        return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None

    key = None
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "getattr"
        and len(expr.args) >= 2
    ):
        key = _str_const(expr.args[1])
    elif isinstance(expr, ast.Subscript):
        base = expr.value
        if (
            isinstance(base, ast.Call)
            and isinstance(base.func, ast.Name)
            and base.func.id == "vars"
        ) or (isinstance(base, ast.Attribute) and base.attr == "__dict__"):
            key = _str_const(expr.slice)
    if key in ("os", "posix"):
        return "os"
    if key == "subprocess":
        return "subprocess"
    return None


def _fq_attr_name(node):
    """Return the dotted name for a Name/Attribute chain, else ''."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _bracket_depth(s):
    """Linear max bracket-nesting scan; never invokes the C parser (DoS-safe)."""
    depth = mx = 0
    for ch in s:
        if ch in "([{":
            depth += 1
            if depth > mx:
                mx = depth
        elif ch in ")]}":
            depth = depth - 1 if depth > 0 else 0
    return mx


def _to_text(value):
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("latin-1", "replace")
    return value


# PEP 263 source-encoding cookie ("# -*- coding: utf-8 -*-", "# coding: utf_7").
_CODING_COOKIE_RE = re.compile(rb"coding[:=]\s*([-\w.]+)")
_CODING_COOKIE_TEXT_RE = re.compile(r"coding[:=]\s*([-\w.]+)")


def _decode_source_bytes(data):
    """Decode an exec/compile *bytes* payload the way CPython would.

    exec()/eval()/compile() honor a PEP 263 coding cookie on bytes, so the analyzer
    must decode with that cookie's codec (not a fixed UTF-8 view) or a snippet like
    ``exec(b"# coding: utf_7\\n#+AAo-__import__('os').system('id')")`` reads as pure
    comments under UTF-8 while actually running hidden code. Detect the encoding,
    decode, then neutralize the cookie so ast.parse(str) does not reject the decoded
    text (a str carrying a coding declaration raises SyntaxError), preserving line
    numbers so the recursive analysis sees the real source.
    """
    data = bytes(data)
    enc = "utf-8"
    try:
        import io as _io_mod
        import tokenize as _tok
        enc, _ = _tok.detect_encoding(_io_mod.BytesIO(data).readline)
    except Exception:
        enc = "utf-8"
    for _cand in (enc, "utf-8"):
        try:
            text = data.decode(_cand)
            break
        except Exception:
            text = None
    if text is None:
        text = data.decode("latin-1", "replace")
    lines = text.split("\n")
    for _i in range(min(2, len(lines))):
        if _CODING_COOKIE_TEXT_RE.search(lines[_i]):
            lines[_i] = _CODING_COOKIE_TEXT_RE.sub("coding_neutralized", lines[_i], count = 1)
    return "\n".join(lines)


def _recovered_source(v):
    """Text an exec/compile sink actually runs: cookie-aware decode for bytes."""
    if isinstance(v, (bytes, bytearray)):
        return _decode_source_bytes(v)
    return _to_text(v)


def _compile_source_node(node):
    """The SOURCE argument of a compile() call: the 1st positional arg or the ``source=``
    keyword. compile() accepts its payload either way, so a keyword-only call
    (compile(source='...', filename='<p>', mode='exec')) must still be recovered."""
    if node.args:
        return node.args[0]
    for kw in node.keywords or []:
        if kw.arg == "source":
            return kw.value
    return None


def _compile_mode(node, const_env):
    """Recover a compile()'s literal mode= (3rd positional or keyword), else 'exec'."""
    mode_node = None
    if len(node.args) >= 3:
        mode_node = node.args[2]
    for kw in node.keywords or []:
        if kw.arg == "mode":
            mode_node = kw.value
    if mode_node is not None:
        v = _const_fold(mode_node, const_env)
        if v in ("eval", "exec", "single"):
            return "eval" if v == "eval" else "exec"
    return "exec"


def _walk_scope_local(scope):
    """Yield descendants of ``scope``'s body that share its namespace, WITHOUT
    descending into nested def / lambda / class / comprehension (each of which is a
    new scope). Used so single-assignment alias detection is scope-correct.

    When ``scope`` is itself a lambda or comprehension, walk its own namespace: a lambda
    body is a single expression, and a comprehension's namespace holds its element
    expression plus the generator iterables / conditions (target bindings are collected
    separately). ``getattr(scope, "body", [])`` only applies to def / class / module."""
    if isinstance(scope, ast.Lambda):
        stack = [scope.body]
    elif isinstance(scope, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        stack = [scope.elt] + [g for gen in scope.generators for g in [gen.iter, *gen.ifs]]
    elif isinstance(scope, ast.DictComp):
        stack = [scope.key, scope.value] + [
            g for gen in scope.generators for g in [gen.iter, *gen.ifs]
        ]
    else:
        stack = list(getattr(scope, "body", []))
    _NESTED = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    while stack:
        n = stack.pop()
        # A nested def / lambda / class / comprehension opens its OWN scope: its body
        # neither shares this namespace nor should its stores be counted here. Skip it
        # entirely -- do not yield it or descend into it. (Checking the popped node,
        # not just its children, is what keeps a nested `def f(): s = print` from
        # inflating the outer count of an `s = os.system` single-assignment alias.)
        if isinstance(n, _NESTED):
            continue
        yield n
        for child in ast.iter_child_nodes(n):
            stack.append(child)


class _ScopeAliasIndex:
    """Per-scope single-assignment aliases (shell sink / exec builtin / compiled
    source) resolved with Python lexical scoping. Counting and resolution are per
    function scope, so two functions binding the same local name neither cancel out
    (a real sink would be missed) nor cross-contaminate (a benign call in one function
    would be flagged, or a dynamic exec in another wrongly treated as a safe alias)."""

    __slots__ = (
        "tree",
        "node_scope",
        "enclosing",
        "shell",
        "execb",
        "compiled",
        "compiledany",
        "impf",
        "deser",
        "strconst",
        "rhsnode",
        "assigned",
        "class_shell",
        "class_execb",
        "class_deser",
        "class_bases",
        "instance_shell",
        "instance_execb",
        "instance_deser",
    )

    def __init__(self, tree):
        self.tree = tree
        self.node_scope: dict = {tree: tree}
        self.enclosing: dict = {tree: None}
        self.shell: dict = {}
        self.execb: dict = {}
        self.compiled: dict = {}
        # name -> True: bound to a compile() result (foldable OR dynamic). Used to catch
        # a code object executed through types.FunctionType(c, ...) after `c = compile(...)`.
        self.compiledany: dict = {}
        self.impf: dict = {}  # name -> True: alias of __import__ / importlib.import_module
        self.deser: dict = {}  # name -> fq deserializer sink (pickle.loads, ...)
        self.strconst: dict = {}  # name -> folded str/bytes constant (for read scanning)
        self.rhsnode: dict = {}  # name -> single-assignment RHS node (for pathlib reads)
        self.assigned: dict = {}
        # class NAME -> {attr: sink}: a class-body alias (class C: f = os.system) accessed
        # as C.f from outside the class, which lexical scope resolution does not cover.
        self.class_shell: dict = {}
        self.class_execb: dict = {}
        self.class_deser: dict = {}
        # class NAME -> [base class names]: literal Name bases, so a subclass access
        # (class D(C): pass; D.s) can follow inheritance to a base's class-body sink alias.
        self.class_bases: dict = {}
        # (receiver_name, attr) -> sink: a simple instance-attribute alias assigned a
        # dangerous callable (c.e = exec; c.e(payload) / obj.s = os.system; obj.s('rm -rf /')).
        # Tracked tree-wide as a fail-closed over-approximation (attribute values are not
        # lexically scoped), so a call through the same receiver name + attr is analyzed.
        self.instance_shell: dict = {}
        self.instance_execb: dict = {}
        self.instance_deser: dict = {}

    def resolve_class_attr(
        self,
        cname,
        attr,
        kind,
        _seen = None,
    ):
        table = getattr(self, "class_" + kind)
        m = table.get(cname)
        if m and attr in m:
            return m[attr]
        # Follow literal base classes so an inherited alias (class C: s = os.system;
        # class D(C): pass; D.s(...)) resolves through C. Cycle-guarded.
        if _seen is None:
            _seen = set()
        if cname in _seen:
            return None
        _seen.add(cname)
        for _base in self.class_bases.get(cname, ()):
            hit = self.resolve_class_attr(_base, attr, kind, _seen)
            if hit is not None:
                return hit
        return None

    def resolve_instance_attr(self, recv, attr, kind):
        return getattr(self, "instance_" + kind).get((recv, attr))

    def _chain(self, node):
        s = self.node_scope.get(node, self.tree)
        while s is not None:
            yield s
            s = self.enclosing.get(s)

    def resolve(self, name, node, kind):
        maps = getattr(self, kind)
        for s in self._chain(node):
            m = maps.get(s)
            if m and name in m:
                return m[name]
            if name in self.assigned.get(s, ()):  # locally shadowed by a non-alias
                return None
        return None

    def effective(self, node, kind):
        maps = getattr(self, kind)
        result: dict = {}
        shadowed: set = set()
        for s in self._chain(node):
            for k, v in maps.get(s, {}).items():
                if k not in shadowed and k not in result:
                    result[k] = v
            shadowed |= self.assigned.get(s, set())
        return result


def _build_scope_alias_index(tree, const_env):
    idx = _ScopeAliasIndex(tree)

    def _rec(node, scope, func_enclose):
        # scope: namespace the direct children belong to (for node_scope + counting).
        # func_enclose: the scope a nested FUNCTION / class body encloses to. Python skips
        # class scope for nested functions, so inside a class body this stays the class's
        # own lexical function/module parent rather than the class.
        for child in ast.iter_child_nodes(node):
            idx.node_scope[child] = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # A lambda, like a def, opens its own scope: a default-bound param alias
                # ((lambda e=exec: e(payload))()) lives in the lambda body's namespace, and
                # anything nested inside encloses to the lambda itself.
                idx.enclosing[child] = func_enclose
                _rec(child, child, child)
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # A comprehension opens its own scope in Python 3; its target bindings
                # ([e(p) for e in [exec]]) belong to that scope, enclosing to func_enclose.
                idx.enclosing[child] = func_enclose
                _rec(child, child, func_enclose)
            elif isinstance(child, ast.ClassDef):
                # A class body executes immediately with its OWN namespace, so it is a
                # real alias scope (class C: e = eval; e(...) runs eval), but its names
                # are not visible to methods defined inside it -- those enclose to
                # func_enclose, skipping the class.
                idx.enclosing[child] = func_enclose
                _rec(child, child, func_enclose)
            else:
                _rec(child, scope, func_enclose)

    _rec(tree, tree, tree)

    # os / subprocess import + from-import aliases are collected tree-wide (imports
    # are lexically visible module-wide in practice) and shared across scopes.
    os_aliases = {"os"}
    subprocess_aliases = {"subprocess"}
    from_aliases: dict[str, str] = {}
    builtins_aliases = {"builtins", "__builtins__"}
    importlib_aliases = {"importlib"}
    # `compile` bound by name (bare builtin or `from builtins import compile as comp`).
    compile_aliases = {"compile"}
    deser_module_aliases: dict[str, str] = {m: m for m in _DESERIALIZE_MODULES}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "os":
                    os_aliases.add(a.asname or "os")
                elif a.name in ("posix", "nt"):
                    # posix / nt are the C backend os wraps (posix.system == os.system), so a
                    # single-assignment alias s = posix.system resolves to an os shell sink.
                    os_aliases.add(a.asname or a.name)
                elif a.name == "subprocess":
                    subprocess_aliases.add(a.asname or "subprocess")
                elif a.name == "builtins":
                    builtins_aliases.add(a.asname or "builtins")
                elif a.name == "importlib":
                    importlib_aliases.add(a.asname or "importlib")
                if a.name in _DESERIALIZE_MODULES:
                    deser_module_aliases[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module in ("os", "subprocess", "posix", "nt"):
            _eff = "subprocess" if n.module == "subprocess" else "os"
            for a in n.names:
                fq = f"{_eff}.{a.name}"
                if fq in _SHELL_SINK_FUNCS:
                    from_aliases[a.asname or a.name] = fq
        elif isinstance(n, ast.ImportFrom) and n.module == "builtins":
            for a in n.names:
                if a.name == "compile":
                    compile_aliases.add(a.asname or "compile")

    def _rhs_is_compile_call(rhs):
        # `compile(...)` reached as the bare builtin, `builtins.compile(...)`, or a
        # `from builtins import compile as comp` alias -- the callee forms that produce a
        # code object bound to a name (for the types.FunctionType(c) execution gadget).
        if not isinstance(rhs, ast.Call):
            return False
        f = rhs.func
        if isinstance(f, ast.Name):
            return f.id in compile_aliases
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "compile"
            and isinstance(f.value, ast.Name)
            and f.value.id in builtins_aliases
        ):
            return True
        return False

    def _rhs_exec_builtin(rhs):
        # bare `exec` / `eval` / `compile`, or `builtins.eval` (attribute form).
        if isinstance(rhs, ast.Name) and rhs.id in _EXEC_BUILTINS:
            return rhs.id
        if (
            isinstance(rhs, ast.Attribute)
            and rhs.attr in _EXEC_BUILTINS
            and isinstance(rhs.value, ast.Name)
            and rhs.value.id in builtins_aliases
        ):
            return rhs.attr
        return None

    def _rhs_import_func(rhs):
        # `__import__` / `importlib.import_module` (+ reload) bound to a name.
        if isinstance(rhs, ast.Name) and rhs.id in ("__import__", "import_module"):
            return True
        if (
            isinstance(rhs, ast.Attribute)
            and rhs.attr in ("import_module", "reload", "__import__")
            and isinstance(rhs.value, ast.Name)
            and rhs.value.id in importlib_aliases
        ):
            return True
        return False

    def _rhs_deserializer(rhs):
        # `pickle.loads` (+ aliased module) bound to a name.
        if isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name):
            canon = deser_module_aliases.get(rhs.value.id)
            if canon is not None:
                fq = f"{canon}.{rhs.attr}"
                if fq in _CODE_DESERIALIZE_SINKS:
                    return fq
        return None

    def _unwrap_container_index(rhs):
        # `s = [os.system][0]` / `e = {'e': exec}['e']` / `l = (pickle.loads,)[0]`: the
        # unwrapped callable is assigned first, then called. Resolve the inline
        # literal-container index to the element node so the sink resolvers below see the
        # real callable instead of an opaque Subscript.
        if not isinstance(rhs, ast.Subscript):
            return rhs
        container = rhs.value
        ci = _const_fold(rhs.slice, const_env)
        if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
            if -len(container.elts) <= ci < len(container.elts):
                return container.elts[ci]
        if isinstance(container, ast.Dict) and ci is not None:
            for k, v in zip(container.keys, container.values):
                if k is not None and _const_fold(k, const_env) == ci:
                    return v
        return rhs

    scopes = [tree] + [
        n
        for n in ast.walk(tree)
        if isinstance(
            n,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        )
    ]
    for scope in scopes:
        counts: dict[str, int] = {}
        rebound: set[str] = set()
        global_names: set[str] = set()
        nonlocal_names: set[str] = set()
        assigns: list[tuple[str, ast.expr]] = []
        allnames: set[str] = set()
        # Function parameters bind local names that lexically shadow an outer alias of
        # the same name, so count them as local assignments for the shadowing rules.
        _sargs = getattr(scope, "args", None)
        if _sargs is not None:
            for _a in list(_sargs.posonlyargs) + list(_sargs.args) + list(_sargs.kwonlyargs):
                allnames.add(_a.arg)
            for _extra in (_sargs.vararg, _sargs.kwarg):
                if _extra is not None:
                    allnames.add(_extra.arg)
        for n in _walk_scope_local(scope):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                counts[n.id] = counts.get(n.id, 0) + 1
                allnames.add(n.id)
            elif isinstance(n, ast.Global):
                rebound.update(n.names)
                global_names.update(n.names)
            elif isinstance(n, ast.Nonlocal):
                rebound.update(n.names)
                nonlocal_names.update(n.names)
            elif (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                assigns.append((n.targets[0].id, n.value))
            elif (
                # An annotated single-assignment (e: object = exec) is a real binding whose
                # RHS must be recorded like a plain Assign, else e(payload) skips analysis.
                isinstance(n, ast.AnnAssign)
                and n.value is not None
                and isinstance(n.target, ast.Name)
            ):
                assigns.append((n.target.id, n.value))
            elif (
                # A parallel unpacking assignment binds each name to the matching RHS element
                # ((s,) = (os.system,); [e] = [exec]; a, b = os.system, 1), which then reaches
                # a sink call the same way a plain alias does. Pair a literal tuple/list target
                # with a literal tuple/list RHS of equal length element-wise so those aliases
                # are recorded; a starred / mismatched / non-literal RHS is left alone.
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], (ast.Tuple, ast.List))
                and isinstance(n.value, (ast.Tuple, ast.List))
                and len(n.targets[0].elts) == len(n.value.elts)
                and not any(isinstance(_t, ast.Starred) for _t in n.targets[0].elts)
            ):
                for _tgt, _val in zip(n.targets[0].elts, n.value.elts):
                    if isinstance(_tgt, ast.Name):
                        assigns.append((_tgt.id, _val))
        # A comprehension generator binds its target like a single-assignment alias when the
        # iterable is a one-element literal: [e(p) for e in [exec]] binds e to exec, so the
        # payload passed through e must still get eval/exec recursion.
        for _gen in getattr(scope, "generators", []):
            if (
                isinstance(_gen.target, ast.Name)
                and isinstance(_gen.iter, (ast.List, ast.Tuple, ast.Set))
                and len(_gen.iter.elts) == 1
            ):
                _tn = _gen.target.id
                counts[_tn] = counts.get(_tn, 0) + 1
                allnames.add(_tn)
                assigns.append((_tn, _gen.iter.elts[0]))
        # A `global`/`nonlocal`-declared name is NOT a local binding, so it must not shadow an
        # outer alias here (its assignment rebinds the target scope instead).
        allnames -= rebound
        idx.assigned[scope] = allnames
        smap: dict[str, str] = {}
        emap: dict[str, str] = {}
        cmap: dict[str, tuple] = {}
        camap: dict[str, bool] = {}
        imap: dict[str, bool] = {}
        dmap: dict[str, str] = {}
        scmap: dict[str, object] = {}
        rnmap: dict[str, ast.expr] = {}
        # Process assignments in SOURCE order so a chained single-assignment alias resolves
        # against the earlier binding it copies (s = os.system; t = s -> t is os.system). The
        # scope walk yields assignments out of order, so sort by the RHS position; Python
        # binds top-to-bottom, so the aliased name is always defined on an earlier line.
        assigns.sort(key = lambda _p: (getattr(_p[1], "lineno", 0), getattr(_p[1], "col_offset", 0)))
        for name, rhs in assigns:
            if counts.get(name) != 1 or name in rebound:
                continue
            # Single-assignment RHS node, used by the read scanner to resolve a pathlib
            # expression bound to a name (p = Path('..') / 'etc' / 'passwd'; p.read_text()).
            rnmap[name] = rhs
            # An inline-container index RHS (s = [os.system][0]) hides the callable; resolve
            # it to the element so the sink resolvers below see the real sink.
            rhs_eff = _unwrap_container_index(rhs)
            fq = _resolve_static_shell_sink(rhs_eff, os_aliases, subprocess_aliases, from_aliases)
            # A chained single-assignment alias (s = os.system; t = s; t('rm -rf /')): the RHS
            # is a bare Name already resolved to a sink earlier in this scope (assigns are in
            # source order), so propagate its sink identity instead of dropping it.
            if fq is None and isinstance(rhs_eff, ast.Name) and rhs_eff.id in smap:
                fq = smap[rhs_eff.id]
            if fq:
                smap[name] = fq
            eb = _rhs_exec_builtin(rhs_eff)
            if eb is None and isinstance(rhs_eff, ast.Name) and rhs_eff.id in emap:
                eb = emap[rhs_eff.id]  # chained alias e = exec; f = e; f(payload)
            if eb is not None:
                emap[name] = eb
            elif (
                _rhs_is_compile_call(rhs_eff)
                or (
                    # A local alias of compile (cfn = compile; co = cfn(src, ...)) is not in
                    # compile_aliases, so resolve the callee through this scope's exec-builtin
                    # map (built in source order, cfn precedes co) before giving up.
                    isinstance(rhs_eff, ast.Call)
                    and isinstance(rhs_eff.func, ast.Name)
                    and emap.get(rhs_eff.func.id) == "compile"
                )
            ) and _compile_source_node(rhs_eff) is not None:
                # Any `c = compile(...)` (bare / builtins.compile / from-import alias)
                # binds a code object, tracked for the types.FunctionType(c) execution
                # gadget below (dynamic or foldable payload). The source may be positional
                # OR the source= keyword.
                camap[name] = True
                v = _const_fold(_compile_source_node(rhs_eff), const_env)
                if isinstance(v, (str, bytes, bytearray)):
                    cmap[name] = (
                        _recovered_source(v),
                        _compile_mode(rhs_eff, const_env),
                        isinstance(v, (bytes, bytearray)),
                    )
            if _rhs_import_func(rhs_eff):
                imap[name] = True
            dfq = _rhs_deserializer(rhs_eff)
            if dfq is None and isinstance(rhs_eff, ast.Name) and rhs_eff.id in dmap:
                dfq = dmap[rhs_eff.id]  # chained alias d = pickle.loads; e = d; e(payload)
            if dfq is not None:
                dmap[name] = dfq
            # Single-assignment string/bytes path constant (p = '/etc/passwd'), used by
            # the sensitive-read scanner to fold function-local read paths.
            cv = _const_fold(rhs, const_env)
            if isinstance(cv, (str, bytes, bytearray)):
                scmap[name] = cv
        # A parameter DEFAULT that is a dangerous callable acts as an alias inside the body:
        # def f(e=exec): e(payload) / def f(s=os.system): s('rm -rf /'). Bind it like a
        # single-assignment alias unless the parameter is reassigned in the body.
        if _sargs is not None:
            _pos = list(_sargs.posonlyargs) + list(_sargs.args)
            _paired = list(zip(_pos[len(_pos) - len(_sargs.defaults) :], _sargs.defaults))
            _paired += [
                (a, d) for a, d in zip(_sargs.kwonlyargs, _sargs.kw_defaults) if d is not None
            ]
            for _p, _d in _paired:
                _pn = _p.arg
                if counts.get(_pn, 0) != 0 or _pn in rebound:
                    continue
                _de = _unwrap_container_index(_d)
                _dfq_sh = _resolve_static_shell_sink(
                    _de, os_aliases, subprocess_aliases, from_aliases
                )
                if _dfq_sh and _pn not in smap:
                    smap[_pn] = _dfq_sh
                _deb = _rhs_exec_builtin(_de)
                if _deb is not None and _pn not in emap:
                    emap[_pn] = _deb
                _ddfq = _rhs_deserializer(_de)
                if _ddfq is not None and _pn not in dmap:
                    dmap[_pn] = _ddfq
                if _rhs_import_func(_de) and _pn not in imap:
                    imap[_pn] = True
        # A `global name = <sink>` (or `nonlocal name = <sink>`) rebinds the name in the
        # TARGET scope (module for global, nearest enclosing scope for nonlocal), NOT locally,
        # so `global s; s = os.system; s('rm -rf /')` must record the alias in that target
        # scope -- otherwise the local pass skips it (name in rebound) and the call resolves to
        # nothing. Target scopes are processed before nested scopes, so setdefault preserves
        # any alias they already hold.
        if global_names or nonlocal_names:

            def _chain_lookup(nm, table, local_map):
                # Resolve a bare-name RHS alias (global t; t = s) against this scope's just-built
                # local map, then the already-indexed enclosing / module scopes -- so a
                # `s = os.system` at module scope copied into a `global t; t = s` is propagated,
                # matching the chained-alias resolution the local pass does via smap[rhs.id].
                _sc = scope
                while _sc is not None:
                    _m = local_map if _sc is scope else table.get(_sc, {})
                    if nm in _m:
                        return _m[nm]
                    _sc = idx.enclosing.get(_sc)
                return None

            for name, rhs in assigns:
                if name in global_names:
                    _target = tree
                elif name in nonlocal_names:
                    _target = idx.enclosing.get(scope)
                else:
                    continue
                if _target is None:
                    continue
                _rhs_eff = _unwrap_container_index(rhs)
                _gfq = _resolve_static_shell_sink(
                    _rhs_eff, os_aliases, subprocess_aliases, from_aliases
                )
                if _gfq is None and isinstance(_rhs_eff, ast.Name):
                    _gfq = _chain_lookup(_rhs_eff.id, idx.shell, smap)
                if _gfq:
                    idx.shell.setdefault(_target, {}).setdefault(name, _gfq)
                _geb = _rhs_exec_builtin(_rhs_eff)
                if _geb is None and isinstance(_rhs_eff, ast.Name):
                    _geb = _chain_lookup(_rhs_eff.id, idx.execb, emap)
                if _geb is not None:
                    idx.execb.setdefault(_target, {}).setdefault(name, _geb)
                _gdfq = _rhs_deserializer(_rhs_eff)
                if _gdfq is None and isinstance(_rhs_eff, ast.Name):
                    _gdfq = _chain_lookup(_rhs_eff.id, idx.deser, dmap)
                if _gdfq is not None:
                    idx.deser.setdefault(_target, {}).setdefault(name, _gdfq)
        if smap:
            idx.shell[scope] = smap
        if emap:
            idx.execb[scope] = emap
        if cmap:
            idx.compiled[scope] = cmap
        if camap:
            idx.compiledany[scope] = camap
        if imap:
            idx.impf[scope] = imap
        if dmap:
            idx.deser[scope] = dmap
        if scmap:
            idx.strconst[scope] = scmap
        if rnmap:
            idx.rhsnode[scope] = rnmap
        # Class-body aliases are also reachable as ClassName.attr from OUTSIDE the class
        # (class C: f = os.system; C.f('rm -rf /')), which lexical scope resolution does not
        # cover, so index them by the class name too.
        if isinstance(scope, ast.ClassDef):
            if smap:
                idx.class_shell[scope.name] = dict(smap)
            if emap:
                idx.class_execb[scope.name] = dict(emap)
            if dmap:
                idx.class_deser[scope.name] = dict(dmap)
            _bases = [b.id for b in scope.bases if isinstance(b, ast.Name)]
            if _bases:
                idx.class_bases[scope.name] = _bases
    # Instance-attribute sink aliases (c.e = exec; c.e(payload) / obj.s = os.system;
    # obj.s('rm -rf /')): a simple `Name.attr = <sink>` store binds the attribute to a
    # dangerous callable. Tracked tree-wide by (receiver_name, attr) as a fail-closed
    # over-approximation, so a later call through the same receiver name gets analyzed.
    for n in ast.walk(tree):
        if not (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Attribute)
            and isinstance(n.targets[0].value, ast.Name)
        ):
            continue
        key = (n.targets[0].value.id, n.targets[0].attr)
        rhs_eff = _unwrap_container_index(n.value)
        _fq = _resolve_static_shell_sink(rhs_eff, os_aliases, subprocess_aliases, from_aliases)
        if _fq:
            idx.instance_shell[key] = _fq
        _eb = _rhs_exec_builtin(rhs_eff)
        if _eb is not None:
            idx.instance_execb[key] = _eb
        _dfq = _rhs_deserializer(rhs_eff)
        if _dfq is not None:
            idx.instance_deser[key] = _dfq
    return idx


def _payload_has_obfuscation_primitive(node):
    """True when a (non-plain) exec/eval payload is assembled from decode / fetch /
    runtime-assembly primitives -- the canonical loader shapes that are essentially
    never benign inside a sandbox."""
    if node is None or isinstance(node, (ast.Name, ast.Constant)):
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            fq = _fq_attr_name(fn)
            attr = (
                fn.attr
                if isinstance(fn, ast.Attribute)
                else (fn.id if isinstance(fn, ast.Name) else "")
            )
            if fq in _CODE_DESERIALIZE_SINKS:
                return True
            # A dynamic exec payload produced by another eval/exec/compile is a
            # nested-dynamic-exec obfuscation (also fails closed on eval(eval(...))).
            if isinstance(fn, ast.Name) and fn.id in _EXEC_BUILTINS:
                return True
            if attr in _DECODE_ATTRS or attr in ("decode", "translate"):
                return True
            if fq and any(fq.startswith(p) for p in _FETCH_FQ_PREFIXES):
                return True
            if attr == "join" and sub.args:
                a0 = sub.args[0]
                if isinstance(a0, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                    return True
                if (
                    isinstance(a0, ast.Call)
                    and isinstance(a0.func, ast.Name)
                    and a0.func.id in ("map", "filter")
                ):
                    return True
            if isinstance(fn, ast.Name) and fn.id in ("bytes", "bytearray") and sub.args:
                a0 = sub.args[0]
                if isinstance(a0, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                    return True
        elif isinstance(sub, ast.Attribute) and sub.attr in ("text", "content"):
            if isinstance(sub.value, ast.Call):
                return True
        elif isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
            if sub.slice.step is not None and _const_fold(sub.slice.step) == -1:
                return True
        elif isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mult):
            # Large string/bytes repetition assembles an oversized payload (parse
            # bomb) that the folder refuses on size; fail closed.
            for a, b in ((sub.left, sub.right), (sub.right, sub.left)):
                sv = _const_fold(a)
                nv = _const_fold(b)
                if isinstance(sv, (str, bytes, bytearray)) and isinstance(nv, int) and nv >= 1024:
                    return True
    return False


def _safe_parse_inner(src, mode, depth, budget):
    """DoS-safe gateway to ast.parse on an attacker-influenced payload string.

    Returns one of ("PARSED", tree|None), ("SYNTAX_BAD", None), ("BOUND_HIT", None).
    A linear bracket-depth pre-scan rejects pathological nesting *before* the C
    parser runs (defends the CPython C-stack overflow on deeply-nested input)."""
    if depth >= _MAX_UNWRAP_DEPTH:
        return ("BOUND_HIT", None)
    if len(src) > _MAX_INNER_SRC:
        return ("BOUND_HIT", None)
    if budget.inner_parses >= _MAX_INNER_PARSES:
        return ("BOUND_HIT", None)
    if budget.inner_chars + len(src) > _MAX_TOTAL_INNER_CHARS:
        return ("BOUND_HIT", None)
    if _bracket_depth(src) > _MAX_BRACKET_DEPTH:
        return ("BOUND_HIT", None)
    budget.inner_parses += 1
    budget.inner_chars += len(src)
    try:
        return ("PARSED", ast.parse(src, mode = mode))
    except SyntaxError:
        try:
            ast.parse(src, mode = ("exec" if mode == "eval" else "eval"))
            return ("PARSED", None)
        except SyntaxError:
            return ("SYNTAX_BAD", None)
    except (RecursionError, MemoryError, ValueError):
        return ("BOUND_HIT", None)


def _first_unsafe_reason(info):
    for key in (
        "shell_escapes",
        "dynamic_exec",
        "network_calls",
        "sensitive_file_reads",
        "filesystem_violations",
        "signal_tampering",
        "exception_catching",
    ):
        for item in info.get(key, []) or []:
            desc = item.get("description")
            if desc:
                return desc
    return "unsafe operation"


def _recover_exec_payload(node, func_id, const_env, compiled_env):
    """Recover a statically foldable source string for eval/exec/compile.

    Returns ("RECOVERED", src, mode, is_bytes) / ("DYNAMIC", None, None, False) /
    ("NO_PAYLOAD", None, None, False). ``is_bytes`` records that the payload folded to
    a bytes/bytearray literal -- ``exec``/``compile`` honor PEP 263 coding cookies on
    bytes, so a bytes payload that fails to parse as UTF-8 Python is treated as an
    obfuscation vector by the caller rather than a harmless SyntaxError.
    """
    if node.args:
        arg0 = node.args[0]
    elif func_id == "compile":
        # compile(source=..., filename=..., mode=...) passes its payload as the source=
        # keyword with no positional args. compile() alone does not execute, but its code
        # object runs via a gadget (types.FunctionType(c)(); fn.__code__ = c; fn()), so the
        # keyword-only source must be analyzed exactly like the positional form rather than
        # slipping through as having no payload. (eval / exec take no keyword arguments in
        # CPython, so an empty node.args there is genuinely payload-less.)
        arg0 = _compile_source_node(node)
        if arg0 is None:
            return ("NO_PAYLOAD", None, None, False)
    else:
        return ("NO_PAYLOAD", None, None, False)
    base_mode = "eval" if func_id == "eval" else "exec"

    # exec(compile("...", ...)) / eval(compile("...", "<s>", "eval")) -- the compile source may be
    # positional or the source= keyword.
    if (
        isinstance(arg0, ast.Call)
        and isinstance(arg0.func, ast.Name)
        and arg0.func.id == "compile"
        and _compile_source_node(arg0) is not None
    ):
        v = _const_fold(_compile_source_node(arg0), const_env)
        if isinstance(v, (str, bytes, bytearray)):
            return (
                "RECOVERED",
                _recovered_source(v),
                _compile_mode(arg0, const_env),
                isinstance(v, (bytes, bytearray)),
            )
        return ("DYNAMIC", None, None, False)

    # c = compile("..."); exec(c)
    if isinstance(arg0, ast.Name) and arg0.id in compiled_env:
        csrc, cmode, cbytes = compiled_env[arg0.id]
        return ("RECOVERED", csrc, cmode, cbytes)

    v = _const_fold(arg0, const_env)
    if isinstance(v, (str, bytes, bytearray)):
        mode = _compile_mode(node, const_env) if func_id == "compile" else base_mode
        return ("RECOVERED", _recovered_source(v), mode, isinstance(v, (bytes, bytearray)))
    return ("DYNAMIC", None, None, False)


# --------------------------------------------------------------------------
# Stage 3: static sensitive-read scanner.
#
# Filesystem WRITE confinement is enforced at runtime by the realpath backstop
# (Stage 5), which is strictly more robust than static path proving. Reads are not
# confined there, so this small static pass blocks sandboxed code from reading host
# secrets: any call arg that folds to a sensitive host path (covers open()/os.open
# and library loaders like pandas.read_csv('/etc/shadow')), plus `..`/`~` traversal
# on the dedicated open()/read callees. Dynamic paths are left to the backstop.
# --------------------------------------------------------------------------

# Sensitive read targets: exact host-identity / credential files, credential dirs,
# and the classic /proc self-inspection paths. Substring tokens are only consulted
# for absolute or ~-rooted paths with no whitespace (avoids sentence false positives).
_SANDBOX_SENSITIVE_EXACT = frozenset(
    {"/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/gshadow", "/etc/master.passwd"}
)
_SANDBOX_SENSITIVE_DIR_PARTS = (
    "/etc/ssh/",
    "/root/",
    "/.ssh/",
    "/.aws/",
    "/.config/gcloud",
    "/.kube/",
    "/.docker/",
    # In-cluster Kubernetes service-account credentials (token / ca.crt / namespace)
    # mounted into every pod; reading the token impersonates the pod to the API server.
    "/var/run/secrets/kubernetes.io/",
    "/run/secrets/kubernetes.io/",
)
_SANDBOX_SENSITIVE_TOKENS = (
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".netrc",
    "credentials",
    ".git-credentials",
    "/.huggingface/token",
    ".kube/config",
)
_SANDBOX_SENSITIVE_RE = re.compile(
    r"^/proc/(?:self|\d+)/(?:environ|cmdline|maps|mem|task/\d+/environ)$"
)


def _is_sensitive_abs_path(s):
    """Provably-sensitive absolute (or ~-rooted) path, whitespace-free."""
    if not isinstance(s, str) or not s:
        return False
    norm = s.replace("\\", "/")
    if any(ch.isspace() for ch in norm):
        return False
    if not (norm.startswith("/") or norm.startswith("~")):
        return False
    if norm in _SANDBOX_SENSITIVE_EXACT:
        return True
    # The directory markers carry a trailing slash to match descendants (/root/id_rsa);
    # append one to the candidate so the directory ITSELF (an unguarded `ls /root` /
    # `find /etc/ssh`) matches too, without loosening the component boundary.
    if any(part in (norm + "/") for part in _SANDBOX_SENSITIVE_DIR_PARTS):
        return True
    if _SANDBOX_SENSITIVE_RE.match(norm):
        return True
    low = norm.lower()
    return any(tok in low for tok in _SANDBOX_SENSITIVE_TOKENS)


# Punctuation separators plus the bash compound-statement keywords (if/while/until/then/do/
# else/elif), so a reader in a CONDITION body (`if cat $SECRET; then :; fi`) is scanned at
# command position rather than treated as an argument of the keyword.
_READ_SCAN_SEPARATORS = (
    ";",
    "&&",
    "||",
    "|",
    "&",
    "(",
    ")",
    "`",
    "{",
    "}",
    "\n",
) + tuple(_SHELL_KEYWORDS_AS_SEP)


def _join_chdir(base, newdir):
    """Resolve an ``env -C DIR`` / ``--chdir=DIR`` operand against the current child cwd.

    A relative DIR chdirs relative to wherever the child already is (an ambient
    subprocess ``cwd=`` or a prior ``env -C``), so ``env -C . cat passwd`` under
    ``cwd=/etc`` still reads ``/etc/passwd``; join it onto the current base rather
    than replacing the base with the bare relative fragment. An absolute / ``~``
    DIR overrides the base outright."""
    if newdir.startswith("/") or newdir.startswith("~"):
        return newdir
    return os.path.join(base, newdir) if base else newdir


def _extract_command_subs(s):
    """Extract the inner payloads of ``$(...)`` and backtick command substitutions from a
    shell string. Substitutions run inside DOUBLE quotes (``echo "$(head /etc/passwd)"``) but
    are suppressed entirely inside SINGLE quotes (``echo '$(head /etc/passwd)'`` is a literal),
    so single-quoted spans are skipped to avoid over-blocking benign literals. Returns a list of
    inner command strings for recursive read scanning. ``$((arith))`` yields a harmless
    ``(arith)`` payload that scans clean."""
    subs = []
    i, n = 0, len(s)
    in_double = False
    while i < n:
        c = s[i]
        # A single quote OUTSIDE double quotes opens a literal span in which $() / backticks do
        # not expand; skip to its close. (Inside double quotes a `'` is an ordinary character.)
        if c == "'" and not in_double:
            j = s.find("'", i + 1)
            if j == -1:
                break  # unterminated single quote: rest is literal
            i = j + 1
        elif c == '"':
            in_double = not in_double
            i += 1
        elif c == "`":
            j = s.find("`", i + 1)
            if j == -1:
                break
            subs.append(s[i + 1 : j])
            i = j + 1
        elif c == "$" and i + 1 < n and s[i + 1] == "(":
            depth = 1
            k = i + 2
            while k < n and depth:
                if s[k] == "(":
                    depth += 1
                elif s[k] == ")":
                    depth -= 1
                k += 1
            subs.append(s[i + 2 : k - 1] if depth == 0 else s[i + 2 : k])
            i = k
        elif c in "<>" and not in_double and i + 1 < n and s[i + 1] == "(":
            # Process substitution <(cmd) / >(cmd): bash runs cmd in a child even when the OUTER
            # command is a non-reader (echo <(cat /etc/passwd >&2) leaks the file), so scan the
            # inner payload too. It is a word-level construct (not performed inside quotes), so
            # single-quoted spans are already skipped and double-quoted text is left literal.
            depth = 1
            k = i + 2
            while k < n and depth:
                if s[k] == "(":
                    depth += 1
                elif s[k] == ")":
                    depth -= 1
                k += 1
            subs.append(s[i + 2 : k - 1] if depth == 0 else s[i + 2 : k])
            i = k
        else:
            i += 1
    return subs


def _resolve_read_chdir(operand, assigns):
    """Resolve an ``env -C DIR`` operand for the read scanner. Returns ``(dir, dynamic)``:

    - a ``$VAR`` / ``${VAR}`` that a preceding assignment in the same command bound (``P=/etc;
      env -C $P cat passwd``) resolves to that value so the read is combined and caught;
    - any other operand carrying ``$`` / backtick is an UNKNOWN expansion that the unguarded
      child can point outside the workdir, so ``dynamic=True`` (fail closed for relative reads);
    - a plain literal DIR resolves to itself."""
    m = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", operand)
    if m:
        if assigns and m.group(1) in assigns:
            return assigns[m.group(1)], False
        return None, True
    if "$" in operand or "`" in operand:
        return None, True
    return operand, False


def _argv_env_chdir(str_elts):
    """Extract an ``env -C DIR`` / ``--chdir[=DIR]`` target from a folded argv vector.

    A wrapper-prefixed child that chdirs before the reader (``['env', '-C', '/etc', 'cat',
    'passwd']``) reads ``/etc/passwd`` in the unguarded child, so the relative reader arg must
    be resolved against DIR. Returns the DIR string (or None). Only the ``env`` wrapper honors
    ``-C``; other flags / wrappers are skipped until the real command word is reached."""
    i, n = 0, len(str_elts)
    wrapper = None
    while i < n:
        tok = str_elts[i]
        if tok is None:
            return None
        if _ASSIGNMENT_RE.match(tok):
            i += 1
            continue
        if tok.startswith("-"):
            if wrapper == "env":
                if tok in ("-C", "--chdir"):
                    return str_elts[i + 1] if i + 1 < n else None
                if tok.startswith("--chdir="):
                    return tok.split("=", 1)[1]
            # A wrapper flag that consumes the NEXT token (env -u NAME, sudo -u user, nice -n 5,
            # timeout -k 5): skip both so the operand is not mistaken for the command word.
            if wrapper is not None and _wrapper_flag_takes_operand(wrapper, tok):
                i += 2
                continue
            i += 1
            continue
        base = os.path.basename(tok).lower()
        if base in _COMMAND_PREFIXES:
            wrapper = base
            i += 1
            continue
        # A wrapper's numeric / duration operand (timeout 1, nice 5): skip it and keep scanning
        # for a following env -C, rather than stopping at it as the executed command word. Without
        # this, `env -C` hidden behind `timeout 1 env -C /etc ...` was never reached.
        if wrapper is not None and _is_wrapper_numeric_arg(tok):
            i += 1
            continue
        return None  # reached the executed command word before any env -C
    return None


def _argv_command_word_index(str_elts):
    """Index of the EXECUTED command word in a folded argv vector, skipping VAR=value
    assignments and command wrappers (env / sudo / nice / timeout / ...) with their flags and
    numeric / separated operands. None if a token is unresolved (None) or no command word is
    reached. Lets a wrapper-hidden reader (['timeout', '1', 'cat', 'x']) be found at its real
    position instead of stopping at argv[0]."""
    i, n = 0, len(str_elts)
    wrapper = None
    while i < n:
        tok = str_elts[i]
        if tok is None:
            return None
        if _ASSIGNMENT_RE.match(tok):
            i += 1
            continue
        if tok.startswith("-"):
            if wrapper is not None and _wrapper_flag_takes_operand(wrapper, tok):
                i += 2
                continue
            i += 1
            continue
        base = os.path.basename(tok).lower()
        if base in _COMMAND_PREFIXES:
            wrapper = base
            i += 1
            continue
        if wrapper is not None and _is_wrapper_numeric_arg(tok):
            i += 1
            continue
        return i
    return None


def _scan_command_string_for_reads(
    command,
    *,
    strict_traversal,
    cwd = None,
    cwd_dynamic = False,
    _depth = 0,
):
    """Scan a shell command STRING for an embedded host-secret read; return a short reason or
    None. Covers literal sensitive / traversal paths, input redirects, and $ / backtick /
    escaping-glob expansions on file-reading commands. Reads from a shell child are not
    runtime-confined, so these are refused statically.

    strict_traversal=True blocks ANY ``..`` / ``~`` read path (the os.system() shell-string
    policy in the Python tool); False blocks only a traversal that resolves onto a sensitive
    path, so ordinary terminal relative navigation (``ls ../src``) is not flagged.

    The reader / command word is resolved past leading ``VAR=value`` assignments and command
    wrappers (env / sudo / nice / timeout / ...), a ``VAR=value`` assignment whose value is a
    sensitive path is flagged, and a nested ``bash -c '<payload>'`` shell has its payload
    recursively scanned, so a read hidden behind a normal command-prefix form is still caught."""
    if not command or _depth > 6:
        return None
    # Strip bash # comments and rewrite unquoted newlines to `;` so each physical shell line starts
    # a fresh command context: without this, `echo ok\ncat /etc/passwd` reads `cat /etc/passwd` as
    # arguments of the non-reader `echo` and the real second-line read is missed.
    # Model bash brace expansion so a brace-hidden reader / path (`{cat,/etc/passwd}`) is seen.
    cmd = _expand_braces(
        _expand_ifs(
            _normalize_ansi_c_quotes(_rewrite_unquoted_newlines(_strip_bash_comments(command)))
        )
    )

    def _traversal_hits_sensitive(norm):
        # A relative path that climbs out of the workdir with '..' can name a host secret
        # (../../../../etc/passwd). Resolve the climb and test whether the descent lands on a
        # sensitive path; a plain in-tree relative path (../sibling/file.txt) is left alone.
        if ".." not in norm.split("/"):
            return False
        try:
            canon = os.path.normpath(norm)
        except Exception:
            canon = norm
        parts = [p for p in canon.split("/") if p not in ("", ".", "..")]
        return bool(parts) and _is_sensitive_abs_path("/" + "/".join(parts))

    def _flag(s):
        norm = s.replace("\\", "/")
        try:
            canon = os.path.normpath(norm)
        except Exception:
            canon = norm
        if _is_sensitive_abs_path(norm) or _is_sensitive_abs_path(canon):
            return f"{s!r} is a sensitive host identity / credential file"
        if strict_traversal:
            if s[:1] == "~" or ".." in norm.split("/"):
                return f"{s!r} escapes the session workdir via path traversal"
        elif _traversal_hits_sensitive(norm):
            return f"{s!r} reads a sensitive host path via directory traversal"
        return None

    def _escaping_glob(tok):
        if not any(g in tok for g in "*?["):
            return False
        tn = tok.replace("\\", "/")
        return tok[:1] == "~" or tn.startswith("/")

    def _check_assignment_rhs(tok):
        # FOO=/etc/passwd binds a sensitive path into a variable that a later reader dereferences
        # (P=/etc/passwd cat ${P}); flag the assigned value directly.
        _rhs = tok.split("=", 1)[1] if "=" in tok else ""
        for _pc in re.split(r"[;|&<>()`{}]+", _rhs):
            if _pc and not _pc.startswith("-"):
                _r = _flag(_pc)
                if _r is not None:
                    return _r
        return None

    # Re-tokenize keeping redirects / separators for the input-redirect + expansion scan.
    try:
        _lx = shlex.shlex(cmd, posix = True, punctuation_chars = ";&|()`<>")
        _lx.whitespace_split = True
        _lx.commenters = (
            ""  # bash comments are pre-stripped; shlex's # handling is not bash-accurate
        )
        ptoks = list(_lx)
    except ValueError:
        ptoks = cmd.split()

    # Literal-path token scan (absolute-sensitive + traversal), COMMAND-WORD aware: a sensitive
    # path that a non-reader merely prints / passes as data (echo /etc/passwd, printf %s
    # /etc/passwd) is not a read, so its args are exempt. Every other command word -- readers AND
    # unknown commands -- still fails closed. A prefix assignment binding a sensitive path
    # (P=/etc/passwd cat ${P}) is flagged at the command position; punctuation glued to a word
    # (cat /etc/passwd|wc) is split so the path piece is still checked.
    _lit_at_cmd = True
    _lit_wrapper = None
    _lit_is_reader_ctx = True  # unknown command word -> still fail closed
    for _lt in ptoks:
        if _lt in _READ_SCAN_SEPARATORS:
            _lit_at_cmd = True
            _lit_wrapper = None
            _lit_is_reader_ctx = True
            continue
        if _lit_at_cmd:
            if _ASSIGNMENT_RE.match(_lt):
                _r = _check_assignment_rhs(_lt)
                if _r is not None:
                    return _r
                continue  # assignment prefix; command word still ahead
            if _lt.startswith("-"):
                continue  # wrapper flag before the command word
            _ltbase = os.path.basename(_lt).lower()
            if _ltbase in _COMMAND_PREFIXES:
                _lit_wrapper = _ltbase
                continue
            if _lit_wrapper is not None and _is_wrapper_numeric_arg(_lt):
                continue
            # command word resolved: a known non-reader exempts its args from the literal scan.
            _lit_is_reader_ctx = _ltbase not in _SHELL_NON_READER_COMMANDS
            _lit_at_cmd = False
            continue
        if not _lit_is_reader_ctx:
            continue  # echo / printf / ... argument: the path is data, not a read
        for _piece in re.split(r"[;|&<>()`{}]+", _lt):
            if _piece and not _piece.startswith("-"):
                _r = _flag(_piece)
                if _r is not None:
                    return _r

    # find <ROOT...> ... -exec READER {} \; loses the search ROOT in the -exec segment, so a reader
    # over an ABSOLUTE / sensitive root (find /etc -name passwd -exec cat {} ;) reads a host secret
    # the segment scan alone cannot see ({} carries no path). Compute the find roots (the leading
    # operands before the first predicate) and note whether any escapes the workdir.
    _find_roots = []
    for _fi2, _ft2 in enumerate(ptoks):
        if os.path.basename(_ft2).lower() == "find":
            _rj = _fi2 + 1
            while _rj < len(ptoks):
                _rt = ptoks[_rj]
                if _rt.startswith("-") or _rt in ("(", "!", ",") or _rt in _READ_SCAN_SEPARATORS:
                    break
                _find_roots.append(_rt)
                _rj += 1
            break
    _find_escaping_root = any(
        _arg_escapes_workdir(_r) or _is_sensitive_abs_path(_r.replace("\\", "/"))
        for _r in _find_roots
    )

    # find ... -exec CMD ... ; runs CMD directly on each match; CMD may be a nested shell
    # (sh -c 'cat /etc/passwd') or a reader, so scan each -exec segment through this scanner
    # (mirrors the blocked-command find -exec handling). The main command-word loop below only
    # recurses into a shell that IS the command word, so the quoted -c payload would otherwise
    # be treated as one inert argument.
    for _fi, _ft in enumerate(ptoks):
        if _ft in _FIND_EXEC_FLAGS:
            _seg = []
            _fj = _fi + 1
            while _fj < len(ptoks) and ptoks[_fj] not in _FIND_EXEC_TERMINATORS:
                _seg.append(ptoks[_fj])
                _fj += 1
            if _seg:
                # A reader -exec that references {} over an escaping find root reads host files.
                if _find_escaping_root and "{}" in _seg:
                    _si = 0
                    while _si < len(_seg) and (
                        _seg[_si].startswith("-")
                        or os.path.basename(_seg[_si]).lower() in _COMMAND_PREFIXES
                        or _ASSIGNMENT_RE.match(_seg[_si])
                    ):
                        _si += 1
                    _segcmd = os.path.basename(_seg[_si]).lower() if _si < len(_seg) else ""
                    if _segcmd in _SHELL_READ_COMMANDS:
                        return (
                            f"find -exec {_segcmd} {{}} over an escaping search root "
                            f"reads a host file the {{}} placeholder hides"
                        )
                _r = _scan_command_string_for_reads(
                    shlex.join(_seg),
                    strict_traversal = strict_traversal,
                    cwd = cwd,
                    cwd_dynamic = cwd_dynamic,
                    _depth = _depth + 1,
                )
                if _r is not None:
                    return _r
        # trap 'CMD' SIG: the quoted handler runs as shell code on EXIT / a signal; scan it.
        # Skip trap options / the -- terminator (trap -- 'CMD' EXIT, trap -p) so the handler
        # operand is not mistaken for -- and left unscanned.
        if _ft == "trap":
            _tj = _fi + 1
            while _tj < len(ptoks) and ptoks[_tj].startswith("-") and len(ptoks[_tj]) > 1:
                _tj += 1
            _th = ptoks[_tj] if _tj < len(ptoks) else None
            if _th and _th != "-" and _th not in _READ_SCAN_SEPARATORS:
                _r = _scan_command_string_for_reads(
                    _th,
                    strict_traversal = strict_traversal,
                    cwd = cwd,
                    cwd_dynamic = cwd_dynamic,
                    _depth = _depth + 1,
                )
                if _r is not None:
                    return _r

    # A command substitution ($(...) / `...`) runs its payload as a shell command regardless of
    # surrounding quotes, so `echo "$(head /etc/passwd)"` reads the file even though the outer
    # command is not a reader and the tokenizer keeps the quoted substitution as one argument.
    # Scan each substitution payload recursively, independent of the outer command word.
    for _cs in _extract_command_subs(cmd):
        if _cs.strip():
            _r = _scan_command_string_for_reads(
                _cs,
                strict_traversal = strict_traversal,
                cwd = cwd,
                cwd_dynamic = cwd_dynamic,
                _depth = _depth + 1,
            )
            if _r is not None:
                return _r

    # env -S 'cmd' / --split-string='cmd' splits its operand into a fresh command line that runs
    # as the child, so a reader-only payload (env --split-string='cat /etc/passwd') is a host-file
    # read even though env is a wrapper. Recurse each env split-string payload into the read scan.
    for _si, _st in enumerate(ptoks):
        _sl = _st.lower()
        _spayload = None
        if _sl in ("-s", "--split-string") and _si + 1 < len(ptoks):
            _spayload = ptoks[_si + 1]
        elif _sl.startswith("-s") and _sl != "-s" and not _sl.startswith("--"):
            _spayload = _st[2:]  # glued short form: -S'cmd' / -Scmd
        elif _sl.startswith("--split-string="):
            _spayload = _st[len("--split-string=") :]
        if not _spayload:
            continue
        # Confirm the split-string belongs to an `env` wrapper (not a -s flag of another command).
        _is_env = False
        for _sj in range(_si - 1, -1, -1):
            _sp = ptoks[_sj]
            if _sp in _READ_SCAN_SEPARATORS:
                break
            if _sp.startswith("-"):
                continue
            _is_env = os.path.basename(_sp).lower() == "env"
            break
        if _is_env:
            _r = _scan_command_string_for_reads(
                _spayload,
                strict_traversal = strict_traversal,
                cwd = cwd,
                cwd_dynamic = cwd_dynamic,
                _depth = _depth + 1,
            )
            if _r is not None:
                return _r

    def _risky_read_target(tgt):
        if not tgt:
            return False
        if "$" in tgt or "`" in tgt or _escaping_glob(tgt):
            return True
        tn = tgt.replace("\\", "/")
        if _is_sensitive_abs_path(tgt):
            return True
        return ".." in tn.split("/") if strict_traversal else _traversal_hits_sensitive(tn)

    _at_cmd = True
    _cur_reader = False
    _wrapper = None
    _skip_operand = False
    # The child's cwd for a relative read: seeded from an ambient cwd (a subprocess cwd=), and
    # overridable per-command by env -C DIR. Resets to the ambient cwd at each separator.
    _chdir = cwd
    _pending_chdir = False
    _pending_argfile = False
    # env -C DIR whose DIR is an unknown expansion ($UNRESOLVED / backtick): the child's cwd is
    # unprovable, so a later relative reader arg fails closed. Reset per command, like _chdir.
    _chdir_dynamic = False
    # Shell VAR=value bindings seen so far (P=/etc; env -C $P ...), so an env -C $P operand
    # resolves to /etc and the read is combined + caught. Persists across separators.
    _local_assigns = {}
    # jq reads files through explicit options, but its positional FILTER legitimately contains
    # `$` (jq variables: jq -n --rawfile x f '$x'), so jq is NOT a generic reader -- that would
    # misfire on every filter. Scan only jq's file-valued options: --rawfile NAME FILE /
    # --slurpfile NAME FILE read FILE into a variable, and -f / --from-file FILE read the program
    # file. A sensitive / expanded / escaping FILE operand exfiltrates a host secret.
    for _ji, _jt in enumerate(ptoks):
        if os.path.basename(_jt).lower() != "jq":
            continue
        _jk = _ji + 1
        while _jk < len(ptoks) and ptoks[_jk] not in _READ_SCAN_SEPARATORS:
            _jw = ptoks[_jk]
            if _jw in ("--rawfile", "--slurpfile") and _jk + 2 < len(ptoks):
                if _risky_read_target(ptoks[_jk + 2]):
                    return f"jq reads a sensitive file {ptoks[_jk + 2]!r}"
                _jk += 3
                continue
            if _jw in ("-f", "--from-file") and _jk + 1 < len(ptoks):
                if _risky_read_target(ptoks[_jk + 1]):
                    return f"jq reads a program file {ptoks[_jk + 1]!r}"
                _jk += 2
                continue
            if _jw.startswith("--from-file="):
                if _risky_read_target(_jw.split("=", 1)[1]):
                    return f"jq reads a program file {_jw.split('=', 1)[1]!r}"
            _jk += 1
    for _pi, _pt in enumerate(ptoks):
        if _pt in _READ_SCAN_SEPARATORS:
            # env -C `...` / env -C $(...): the substitution operand STARTS with a punctuation
            # token ( ` or ( ) that the tokenizer emits as a separator, so the pending env -C
            # never captured it. Mark the cwd dynamic here so the trailing reader fails closed.
            if _pending_chdir and _pt in ("(", "`"):
                _chdir_dynamic = True
            _at_cmd = True
            _cur_reader = False
            _wrapper = None
            _skip_operand = False
            _pending_argfile = False
            _chdir = cwd
            # A command-substitution punctuation token ( ( ) ` ) does NOT end the current
            # command, so it must not drop a pending env -C dynamic-cwd flag: env -C $(printf
            # /etc) cat passwd tokenizes the operand into $ ( printf /etc ), and the ( / )
            # would otherwise reset _chdir_dynamic before the trailing reader is scanned. Only a
            # real command separator ( ; | & newline / keyword ) ends the env -C scope.
            if _pt not in ("(", ")", "`"):
                _chdir_dynamic = False
            _pending_chdir = False
            continue
        if _pt.startswith("<"):
            _rt = _pt.lstrip("<") or (ptoks[_pi + 1] if _pi + 1 < len(ptoks) else "")
            if _risky_read_target(_rt):
                return f"shell input redirect from a non-literal / sensitive path {_rt!r}"
            continue
        if _pt.startswith(">"):
            continue  # output redirects are handled by _find_blocked_commands
        if _at_cmd:
            if _skip_operand:  # a wrapper flag's separated operand (env -u NAME)
                if _pending_chdir:  # ...but env -C DIR's operand is the child cwd
                    _rdir, _rdyn = _resolve_read_chdir(_pt, _local_assigns)
                    if _rdyn:
                        _chdir_dynamic = True
                    else:
                        _chdir = _join_chdir(_chdir, _rdir)
                    _pending_chdir = False
                elif _pending_argfile:  # ...and xargs -a FILE reads FILE
                    _pending_argfile = False
                    if _risky_read_target(_pt):
                        return f"xargs reads arguments from a sensitive path {_pt!r}"
                _skip_operand = False
                continue
            if _ASSIGNMENT_RE.match(_pt):
                _r = _check_assignment_rhs(_pt)
                if _r is not None:
                    return _r
                _an, _, _av = _pt.partition("=")
                _local_assigns[_an.rstrip("+")] = _av
                continue  # assignment prefix; the command word is still ahead
            if _pt.startswith("-"):
                # env -C DIR / --chdir DIR changes the child's cwd before the command runs, so
                # a later relative reader arg (env -C /etc cat passwd -> /etc/passwd) resolves
                # against DIR, not the workdir. Capture DIR instead of just skipping it.
                if _wrapper == "env" and _pt in ("-C", "--chdir"):
                    _pending_chdir = True
                    _skip_operand = True
                elif _wrapper == "env" and _pt.startswith("--chdir="):
                    _rdir, _rdyn = _resolve_read_chdir(_pt.split("=", 1)[1], _local_assigns)
                    if _rdyn:
                        _chdir_dynamic = True
                    else:
                        _chdir = _join_chdir(_chdir, _rdir)
                # xargs -a FILE / --arg-file[=]FILE reads its argument list FROM that file, so a
                # sensitive / expanded target is a host-file read even though xargs is a wrapper.
                elif _wrapper == "xargs" and _pt in ("-a", "--arg-file"):
                    _pending_argfile = True
                    _skip_operand = True
                elif _wrapper == "xargs" and _pt.startswith("--arg-file="):
                    if _risky_read_target(_pt.split("=", 1)[1]):
                        return (
                            f"xargs reads arguments from a sensitive path {_pt.split('=', 1)[1]!r}"
                        )
                elif _wrapper and _wrapper_flag_takes_operand(_wrapper, _pt):
                    _skip_operand = True
                continue  # wrapper flag; still before the command word
            # A wrapper's numeric operand (`timeout 1 bash -c ...`, `nice 5 cat ...`) is not
            # the command word; skip it so the real command (bash / cat) after it is scanned.
            if _wrapper and _is_wrapper_numeric_arg(_pt):
                continue
            _base = os.path.basename(_pt).lower()
            if _base in _COMMAND_PREFIXES:
                _wrapper = _base
                continue  # env / sudo / nice / timeout ...; command word is still ahead
            if _base in _SHELL_BINARIES:
                # A nested shell runs its -c payload as fresh shell code; scan it recursively.
                for _k in range(_pi + 1, len(ptoks)):
                    _ft = ptoks[_k]
                    if _ft in _READ_SCAN_SEPARATORS:
                        break
                    _fl = _ft.lower()
                    if _fl == "-c" or (
                        _fl.startswith("-") and not _fl.startswith("--") and _fl.endswith("c")
                    ):
                        if _k + 1 < len(ptoks):
                            # Propagate the CURRENT env -C cwd AND its dynamic flag: env -C
                            # ${X:-/etc} bash -c 'cat passwd' chdirs to an unprovable dir, so the
                            # nested payload's relative reads must fail closed too (not just the
                            # original ambient cwd_dynamic).
                            _r = _scan_command_string_for_reads(
                                ptoks[_k + 1],
                                strict_traversal = strict_traversal,
                                cwd = _chdir,
                                cwd_dynamic = cwd_dynamic or _chdir_dynamic,
                                _depth = _depth + 1,
                            )
                            if _r is not None:
                                return _r
                        break
                _at_cmd = False
                _cur_reader = False
                _wrapper = None
                continue
            _cur_reader = _base in _SHELL_READ_COMMANDS
            _at_cmd = False
            _wrapper = None
            continue
        if _cur_reader and not _pt.startswith("-"):
            if "$" in _pt or "`" in _pt or _escaping_glob(_pt):
                return f"shell read command reads an expanded path {_pt!r}"
            _rel = not _pt.startswith("/") and not _pt.startswith("~")
            # A relative reader arg under an env -C whose DIR was an UNKNOWN expansion (env -C
            # $UNRESOLVED cat passwd) cannot be proven sandbox-local, so fail closed.
            if _chdir_dynamic and _rel:
                return f"shell read command reads {_pt!r} under an expanded chdir"
            # Under a known chdir (env -C DIR or an ambient subprocess cwd=), a relative reader
            # arg resolves against DIR (cat passwd + cwd=/etc -> /etc/passwd).
            if _chdir and _rel:
                _r = _flag(os.path.join(_chdir, _pt))
                if _r is not None:
                    return _r
            # A relative reader arg under a NON-literal cwd cannot be proven sandbox-local, so
            # fail closed (subprocess.run('cat passwd', shell=True, cwd=P)).
            if cwd_dynamic and _chdir is None and _rel:
                return f"shell read command reads {_pt!r} under a non-literal cwd"
    return None


def _command_reads_sensitive(command: str) -> str | None:
    """Scan a raw terminal-tool shell command STRING for an embedded host-secret read; return a
    short reason or None. The terminal tool runs the command in an unguarded shell child (the
    runtime open() backstop only wraps the Python tool), so a read of an identity / credential
    file, a sensitive-target ``..`` traversal, or an escaping glob / ``$()`` / backtick
    expansion on a file-reading command must be refused statically. Benign in-tree relative
    navigation is left alone (strict_traversal disabled)."""
    return _scan_command_string_for_reads(command, strict_traversal = False)


# Stage 4: pragmatic aliasing (single-assignment alias + inline literal container).
# Catches `s = os.system; s('rm -rf /')` and `[os.system][0](...)` feeding the
# existing shell-command denylist. Deliberately low-FP: only unambiguous single
# assignments (a name stored exactly once) and inline literal containers, never a
# flow-insensitive union (so `s = os.system; s = print; s('hi')` is NOT aliased).
# --------------------------------------------------------------------------
_SHELL_SINK_FUNCS = frozenset(
    {
        "os.system",
        "os.popen",
        "os.popen2",
        "os.popen3",
        "os.popen4",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.posix_spawn",
        "os.posix_spawnp",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
    }
)


def _resolve_static_shell_sink(node, os_aliases, subprocess_aliases, from_aliases):
    """Resolve an expression to a shell-sink fully-qualified name, else None."""
    if isinstance(node, ast.Attribute):
        v = node.value
        # os / posix / nt receiver as a simple name (os.system, posix.system) or a re-exported
        # module reached as an attribute (pathlib.os.system, tempfile.os.system).
        is_os = (isinstance(v, ast.Name) and v.id in os_aliases) or (
            isinstance(v, ast.Attribute) and v.attr in ("os", "posix", "nt")
        )
        is_sp = (isinstance(v, ast.Name) and v.id in subprocess_aliases) or (
            isinstance(v, ast.Attribute) and v.attr == "subprocess"
        )
        if is_os:
            fq = f"os.{node.attr}"
            if fq in _SHELL_SINK_FUNCS:
                return fq
        if is_sp:
            fq = f"subprocess.{node.attr}"
            if fq in _SHELL_SINK_FUNCS:
                return fq
    if isinstance(node, ast.Name):
        return from_aliases.get(node.id)
    return None


_PY_NESTED_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _module_stmt_names(node, loads, stores):
    """Append Name loads / stores GOVERNED by the current (module) scope from ``node`` WITHOUT
    descending into nested function / class / lambda / comprehension scopes (which have their own
    scope). A def / class / import binds its name in the current scope; its body is skipped."""
    for _child in ast.iter_child_nodes(node):
        if isinstance(_child, ast.Name):
            if isinstance(_child.ctx, ast.Load):
                loads.append(_child.id)
            elif isinstance(_child.ctx, (ast.Store, ast.Del)):
                stores.append(_child.id)
        elif isinstance(_child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.append(_child.name)  # binds its name; the body is a nested scope (skipped)
        elif isinstance(
            _child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)
        ):
            pass  # nested scope: contributes no binding to the current scope
        elif isinstance(_child, (ast.Import, ast.ImportFrom)):
            for _al in _child.names:
                stores.append((_al.asname or _al.name).split(".")[0])
        else:
            _module_stmt_names(_child, loads, stores)


def _module_toplevel_free_loads(module_node):
    """(early_loads, all_bound) for a payload Module top level: ``early_loads`` are names Loaded at
    module top level BEFORE that name's first top-level binding, in SOURCE order (these resolve to
    the enclosing global scope at runtime -- ``f('x'); f = None`` still calls the caller's ``f``),
    and ``all_bound`` is every name bound at module top level."""
    early: set = set()
    bound: set = set()
    for _stmt in module_node.body:
        _loads: list = []
        _stores: list = []
        _module_stmt_names(_stmt, _loads, _stores)
        for _nm in _loads:
            if _nm not in bound:
                early.add(_nm)
        bound.update(_stores)
    return early, bound


def _payload_outward_load_names(src, mode):
    """Names in an exec/eval payload whose Load can resolve to the ENCLOSING (caller / provided-
    namespace / global) scope rather than a payload-local binding. exec/eval run at module scope,
    so this is ORDER-sensitive: a top-level Load before the name's first top-level binding resolves
    outward, as does a free (non-local) Load inside any nested function/class scope -- that can run
    after a later top-level rebind. A name bound at module top level is treated as payload-local for
    nested references (its own binding shadows the caller), and a payload-local sink is caught by
    the inner recursive scan instead. Returns a set of names."""
    try:
        inner = ast.parse(src, mode = "eval" if mode == "eval" else "exec")
    except Exception:
        return set()
    # eval: a single expression with no bindings -- every loaded name resolves outward.
    if isinstance(inner, ast.Expression):
        names = {
            _n.id
            for _n in ast.walk(inner)
            if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Load)
        }
    else:
        names, module_bound = _module_toplevel_free_loads(inner)
        try:
            import symtable as _symtable
            _stack = list(_symtable.symtable(src, "<payload>", "exec").get_children())
            while _stack:
                _s = _stack.pop()
                for _sym in _s.get_symbols():
                    # A nested-scope reference that is free / global resolves to the module (caller)
                    # scope UNLESS the payload binds it at module top level (then the payload
                    # controls it, and any payload-local sink is caught by the inner scan).
                    if (
                        _sym.is_referenced()
                        and (_sym.is_free() or _sym.is_global())
                        and _sym.get_name() not in module_bound
                    ):
                        names.add(_sym.get_name())
                _stack.extend(_s.get_children())
        except Exception:  # pragma: no cover - defensive: fail closed by flagging every load
            for _n in ast.walk(inner):
                if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Load):
                    names.add(_n.id)
    # A constant-key namespace-dict lookup (globals()['f'] / locals()['f'] / vars()['f']) reads a
    # name from the caller's namespace WITHOUT a Name node, so its key is an outward reference too:
    # exec("globals()['f']('rm -rf /')") reaches the caller's f = os.system. Add the literal keys.
    for _n in ast.walk(inner):
        if (
            isinstance(_n, ast.Subscript)
            and isinstance(_n.value, ast.Call)
            and isinstance(_n.value.func, ast.Name)
            and _n.value.func.id in ("globals", "locals", "vars")
            and not _n.value.args
            and not _n.value.keywords
        ):
            _key = _n.slice
            if isinstance(_key, ast.Constant) and isinstance(_key.value, str):
                names.add(_key.value)
    return names


def _payload_calls_nonbuiltin_free_name(src, mode, free):
    """True when the payload calls (``f(...)``) a FREE name that is not a Python builtin -- the
    sink-execution vector when an OPAQUE exec/eval namespace could map that name to a hidden sink."""
    try:
        inner = ast.parse(src, mode = "eval" if mode == "eval" else "exec")
    except Exception:
        return True  # fail closed
    import builtins as _bpy

    _bi_names = set(dir(_bpy))
    for _n in ast.walk(inner):
        if (
            isinstance(_n, ast.Call)
            and isinstance(_n.func, ast.Name)
            and _n.func.id in free
            and _n.func.id not in _bi_names
        ):
            return True
    return False


# Native FFI modules that make UNGUARDED libc / syscall calls (ctypes.CDLL('libc').system,
# cffi.FFI().dlopen), bypassing the Python open / os.open monkeypatches, so a literal `import
# ctypes` in the submitted snippet is denied the same way as a dynamic import or a workdir helper
# module importing one. (numpy / other compiled wheels are NOT here: they expose no raw-syscall API.)
_NATIVE_ESCAPE_MODULES = frozenset({"ctypes", "_ctypes", "cffi"})


def _check_signal_escape_patterns(
    code: str,
    _depth: int = 0,
    _budget = None,
):
    """Check for patterns that could escape signal-based timeouts. Returns
    (safe: bool, details: dict). Vendored from unsloth_zoo.rl_environments to
    avoid importing unsloth_zoo (needs GPU drivers; fails on Apple Silicon)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "signal_tampering": [],
            "exception_catching": [],
            "warnings": [],
        }

    # Charge this tree's node count against the shared analyzer budget BEFORE the several
    # unbounded ast.walk / visitor passes below. A syntactically valid file (or a huge
    # recovered exec/eval payload, since the budget is shared across the recursion) with
    # hundreds of thousands of nodes would otherwise tie up the Studio parent process
    # before the child rlimits apply. Fail closed (block) when the budget is exceeded.
    if _budget is None:
        _budget = _AnalyzerBudget()
    try:
        _budget.nodes += sum(1 for _ in ast.walk(tree))
    except Exception:  # pragma: no cover - defensive
        pass
    if _budget.nodes > _MAX_ANALYZER_NODES:
        return False, {
            "error": None,
            "signal_tampering": [],
            "exception_catching": [],
            "shell_escapes": [],
            "dynamic_exec": [
                {
                    "type": "analyzer_budget",
                    "line": -1,
                    "description": (
                        "code exceeds the static-analysis node budget (too large to verify safely)"
                    ),
                }
            ],
            "network_calls": [],
            "sensitive_file_reads": [],
            "filesystem_violations": [],
            "warnings": [],
        }

    signal_tampering = []
    exception_catching = []
    shell_escapes = []
    dynamic_exec = []
    filesystem_violations = []
    warnings = []

    # Feature flag + shared budget for the recursive sink analyzer (Stages 2-4).
    # Default on; UNSLOTH_STUDIO_SINK_ANALYZER=0 reverts to the legacy blanket
    # eval/exec ban and disables filesystem-confinement + aliasing analysis.
    _analyzer_on = os.environ.get("UNSLOTH_STUDIO_SINK_ANALYZER", "1") != "0"
    if _analyzer_on:
        try:
            _const_env = _build_const_prop_env(tree)
            _scope_idx = _build_scope_alias_index(tree, _const_env)
        except Exception:  # pragma: no cover - defensive: never crashier than legacy
            logger.warning("sandbox analyzer context build failed; legacy fallback", exc_info = True)
            _analyzer_on = False
            _const_env = {}
            _scope_idx = _ScopeAliasIndex(tree)
    else:
        _const_env = {}
        _scope_idx = _ScopeAliasIndex(tree)

    def _payload_free_name_hits_caller_alias(src, mode, node):
        """A FREE name in an exec/eval payload that resolves, in the CALLER's scope at ``node``, to
        a shell / exec-builtin / deserialize / import alias -- exec/eval run in the caller
        namespace, so ``f`` in ``exec('f(...)')`` is the caller's ``f = os.system``. Returns the
        offending name or None. The outward-name analysis is ORDER-aware (``f('x'); f = None`` still
        calls the caller's ``f`` before the rebind) and scope-aware (a free load inside a nested
        function resolves outward too). Builtins / undefined names never resolve, so
        ``exec('print(1)')`` and ``exec('x = 1')`` stay allowed."""
        for nm in _payload_outward_load_names(src, mode):
            for _kind in ("shell", "execb", "deser", "impf"):
                if _scope_idx.resolve(nm, node, _kind):
                    return nm
        return None

    def _namespace_value_is_sink(_v):
        """True when an exec/eval namespace VALUE node ({'f': os.system}) resolves to a shell /
        exec-builtin / deserialize / import sink."""
        if isinstance(_v, ast.Name):
            if _v.id in _DYNAMIC_EXEC_BUILTINS or _v.id == "__import__":
                return True
            for _kind in ("shell", "execb", "deser", "impf"):
                if _scope_idx.resolve(_v.id, _v, _kind):
                    return True
            return False
        _fq = _fq_attr_name(_v)
        if _fq in _SHELL_SINK_FUNCS or _fq in _CODE_DESERIALIZE_SINKS:
            return True
        _last = _fq.rsplit(".", 1)[-1] if _fq else ""
        return _last in _DYNAMIC_EXEC_BUILTINS or _last == "__import__"

    def _exec_namespace_alias_hit(node, src, mode):
        """exec/eval with an EXPLICIT globals/locals namespace resolves the payload's free names
        from that mapping, not the caller scope. Inspect a literal-dict namespace precisely (a free
        name mapped to a sink blocks) and fail closed on an OPAQUE namespace when a non-builtin free
        name is CALLED (it could map to a hidden sink). Returns the offending key / marker or None."""
        _ns_nodes = []
        for _i in (1, 2):  # exec(obj, globals, locals) / eval(expr, globals, locals)
            if len(node.args) > _i:
                _ns_nodes.append(node.args[_i])
        for _kw in node.keywords:
            if _kw.arg in ("globals", "locals"):
                _ns_nodes.append(_kw.value)
        if not _ns_nodes:
            return None
        _free = _payload_outward_load_names(src, mode)
        if not _free:
            return None
        for _ns in _ns_nodes:
            if isinstance(_ns, ast.Dict):
                for _k, _v in zip(_ns.keys, _ns.values):
                    _ks = _extract_string_from_node(_k) if _k is not None else None
                    if _ks is not None and _ks in _free and _namespace_value_is_sink(_v):
                        return _ks
                    if _k is None and not isinstance(_v, ast.Dict):
                        # a **opaque splat could carry a sink alias for a called free name
                        if _payload_calls_nonbuiltin_free_name(src, mode, _free):
                            return "<opaque-namespace>"
            elif _payload_calls_nonbuiltin_free_name(src, mode, _free):
                return "<opaque-namespace>"
        return None

    def _analyze_exec_call(node, func_id):
        """Stage 2 driver: recover + recurse a foldable payload, else dynamic policy."""
        try:
            # Resolve compiled-code aliases (c = compile(...)) in the CALL's scope so a
            # safe alias in one function cannot shadow a dynamic exec(c) in another.
            _compiled_here = _scope_idx.effective(node, "compiled")
            kind, src, mode, is_bytes = _recover_exec_payload(
                node, func_id, _const_env, _compiled_here
            )
            if kind == "NO_PAYLOAD":
                return
            if kind == "RECOVERED":
                parsed_kind, _ = _safe_parse_inner(src, mode, _depth, _budget)
                if parsed_kind == "PARSED":
                    inner_safe, inner_info = _check_signal_escape_patterns(src, _depth + 1, _budget)
                    if not inner_safe and not inner_info.get("error"):
                        dynamic_exec.append(
                            {
                                "type": "dynamic_exec",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    f"{func_id}() payload reaches unsafe operation: "
                                    f"{_first_unsafe_reason(inner_info)}"
                                ),
                            }
                        )
                        return
                    # exec()/eval() run in the CALLER namespace, so the payload -- scanned above as a
                    # fresh module -- can reference a caller-scope alias the inner pass cannot see
                    # (import os; f = os.system; exec("f('rm -rf /')")). Fail closed when a payload
                    # FREE name resolves to a shell / exec / deserialize / import alias at this call.
                    _alias = _payload_free_name_hits_caller_alias(src, mode, node)
                    if _alias is not None:
                        dynamic_exec.append(
                            {
                                "type": "dynamic_exec",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    f"{func_id}() payload references caller alias {_alias!r} "
                                    "bound to a shell / exec / deserialize sink"
                                ),
                            }
                        )
                        return
                    # exec("f(...)", {'f': os.system}) resolves the payload's free names from the
                    # EXPLICIT namespace, not the caller scope; inspect a literal-dict namespace for
                    # a sink alias and fail closed on an opaque one.
                    _ns_hit = _exec_namespace_alias_hit(node, src, mode)
                    if _ns_hit is not None:
                        dynamic_exec.append(
                            {
                                "type": "dynamic_exec",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    f"{func_id}() payload free name resolves to a shell / exec / "
                                    f"deserialize sink in the supplied namespace ({_ns_hit})"
                                ),
                            }
                        )
                    return
                if parsed_kind == "BOUND_HIT":
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                f"{func_id}() payload exceeds static-analysis bounds "
                                "(oversized / too-deeply-nested / too-many-layers)"
                            ),
                        }
                    )
                    return
                # SYNTAX_BAD on a fully RECOVERED literal: we hold the exact
                # source and it simply is not valid Python for this sink's mode,
                # so it raises SyntaxError at runtime (same mode as the static
                # parse) -- harmless, not an ACE vector. Allow it; only truly
                # opaque (non-recoverable) payloads fall to the dynamic policy.
                #
                # Exception: a *bytes* payload for an executing sink. exec()/eval()/
                # compile() honor PEP 263 coding cookies (e.g. "# coding: utf-7") on
                # bytes, decoding them through a codec this static pass does not
                # replicate -- the UTF-8 view we parsed is SYNTAX_BAD precisely
                # because the real (cookie-decoded) source is hidden. A legitimate
                # exec(b"...") uses plain ASCII/UTF-8 that parses cleanly, so blocking
                # the unparseable-bytes case closes the codec-smuggling vector with
                # negligible false positives.
                if is_bytes and func_id != "compile":
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                f"{func_id}() of a bytes payload that is not valid UTF-8 "
                                "Python (may smuggle code via a PEP 263 coding cookie)"
                            ),
                        }
                    )
                return
            payload = node.args[0] if node.args else None
            if _payload_has_obfuscation_primitive(payload):
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": (
                            f"{func_id}() of a runtime-decoded / fetched / assembled payload"
                        ),
                    }
                )
            else:
                # An opaque, non-recoverable payload for an executing sink (eval/exec/
                # runpy) is a universal ACE bypass: it can synthesize any shell/network/
                # filesystem escape at runtime, invisibly to every static check. compile()
                # does not itself run, but its CODE OBJECT can be executed without exec/eval
                # (fn.__code__ = compile(src, '<p>', 'exec'); fn()), so an opaque compile
                # source is equally unverifiable and is blocked too. A literal source is
                # analyzed recursively above; ast.literal_eval / json.loads cover data.
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": (
                            f"{func_id}() of a non-literal payload cannot be statically "
                            "verified (use ast.literal_eval / json.loads for data)"
                        ),
                    }
                )
        except Exception:  # pragma: no cover - fail closed, never crashier than legacy
            dynamic_exec.append(
                {
                    "type": "dynamic_exec",
                    "line": getattr(node, "lineno", -1),
                    "description": f"dynamic code execution via {func_id}()",
                }
            )

    def _analyze_timeit_code_arg(call_node, arg_node, label):
        """Analyze a timeit stmt / setup argument, which timeit COMPILES and EXECUTES. A string
        (literal or foldable) is recursed like an exec payload -- a benign body (sum(range(10)))
        passes, a shell / escape body blocks. A non-string arg (a callable stmt, timeit's other
        supported form) carries no source and is left alone, so ordinary timeit use is not blocked."""
        if arg_node is None:
            return
        try:
            _src = _const_fold(arg_node, _const_env)
            if not isinstance(_src, str):
                # A bare string literal that _const_fold declined (kept for clarity) is still source.
                if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                    _src = arg_node.value
                else:
                    return  # a callable stmt / opaque non-string arg carries no analyzable source
            parsed_kind, _ = _safe_parse_inner(_src, "exec", _depth, _budget)
            if parsed_kind == "PARSED":
                _inner_safe, _inner_info = _check_signal_escape_patterns(_src, _depth + 1, _budget)
                if not _inner_safe and not _inner_info.get("error"):
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(call_node, "lineno", -1),
                            "description": (
                                f"timeit {label} string reaches unsafe operation: "
                                f"{_first_unsafe_reason(_inner_info)}"
                            ),
                        }
                    )
            elif parsed_kind == "BOUND_HIT":
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(call_node, "lineno", -1),
                        "description": f"timeit {label} string exceeds static-analysis bounds",
                    }
                )
        except Exception:
            dynamic_exec.append(
                {
                    "type": "dynamic_exec",
                    "line": getattr(call_node, "lineno", -1),
                    "description": f"timeit {label} string could not be statically verified",
                }
            )

    def _ast_name_matches(node, names):
        if isinstance(node, ast.Name):
            return node.id in names
        elif isinstance(node, ast.Attribute):
            full_name = []
            current = node
            while isinstance(current, ast.Attribute):
                full_name.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                full_name.append(current.id)
            full_name = ".".join(reversed(full_name))
            return full_name in names
        return False

    # Dangerous os/subprocess functions that can execute shell commands
    # (defined at module scope as _SHELL_SINK_FUNCS so Stage 4 alias resolution
    # can share it).
    _SHELL_EXEC_FUNCS = _SHELL_SINK_FUNCS

    # Dynamic-execution / obfuscation primitives that defeat the static (name-based) checks
    # above: they build or reach a dangerous callable at runtime, so a bare name match cannot
    # see the payload. eval/exec/compile are direct code-execution builtins; __import__ /
    # importlib load a module by (possibly computed) name; getattr/setattr on a sensitive
    # module implement `getattr(os, 'sys'+'tem')(...)`.
    _DYNAMIC_EXEC_BUILTINS = frozenset({"eval", "exec", "compile"})
    _DYNAMIC_IMPORT_FUNCS = frozenset(
        {"importlib.import_module", "importlib.reload", "importlib.__import__"}
    )
    # Dynamic import is a real workflow (e.g. importing huggingface_hub), so it is flagged only
    # when the target is computed (non-literal name = obfuscation) or names a module that can
    # reach code execution / shell / builtins. A benign literal (json, numpy, huggingface_hub)
    # passes; the HF upload gate below still validates its call args separately.
    _DANGEROUS_IMPORT_NAMES = frozenset(
        {
            "os",
            "posix",  # the C module os wraps; __import__('posix').system(...) == os.system
            "nt",  # Windows analogue of posix
            "subprocess",
            "sys",
            "builtins",
            "importlib",
            "ctypes",
            "pty",
            "socket",
            "signal",
            "resource",
            "shutil",
            "multiprocessing",
            "runpy",
            "code",
            "codeop",
            "pdb",
            "mmap",
            "fcntl",
        }
    )
    # Attribute-name obfuscation via getattr/setattr is only flagged when aimed at a module
    # that can execute code or reach builtins (keeps ordinary getattr(obj, "field") benign).
    _DYNAMIC_ATTR_TARGETS = frozenset({"os", "subprocess", "sys", "builtins", "importlib"})
    # Introspection "gadget" dunders used to walk from a harmless object to os/builtins
    # (``().__class__.__bases__[0].__subclasses__()``). ``__class__`` / ``__dict__`` are
    # intentionally excluded (too common); the chain still trips on the others.
    # __mro__ and __code__ are deliberately EXCLUDED here: on their own they do not reach
    # an execution primitive, and they are read by ordinary ML/debugging code
    # (trainer_class.__mro__, fn.__code__), so flagging them over-blocks legitimate
    # snippets. The terminal escape primitives below still trip on the real gadget
    # chains (().__class__.__bases__[0].__subclasses__(), f.__globals__['os']). A
    # SUBSCRIPTED __mro__ (cls.__mro__[1], the base-class extraction shape) is flagged
    # separately in visit_Subscript so plain iteration stays allowed.
    # cell_contents is the ONLY way to read a closure cell's value, so it is the terminal
    # step of recovering a guarded wrapper's original callable via __closure__; flagging
    # it closes that recovery even when the __closure__ name was built dynamically.
    _GADGET_DUNDERS = frozenset(
        {
            "__subclasses__",
            "__bases__",
            "__base__",
            "__globals__",
            "__builtins__",
            "__closure__",
            "cell_contents",
            # Frame / traceback introspection recovers a runtime-guard wrapper's ORIGINAL
            # unguarded callable: it is a free variable (`real`) in the wrapper's frame, so a
            # snippet that triggers a denied open() can read it back via a trace hook or the
            # caught exception's traceback (frame.f_locals['real'], tb.tb_frame.f_locals) and
            # call it directly, escaping the filesystem boundary. __closure__ / cell_contents
            # are already blocked, so the frame path is the remaining channel; close it by
            # flagging frame acquisition (settrace / _getframe / currentframe / tb_frame) and
            # the f_locals / f_globals value read for ANY receiver.
            "f_locals",
            "f_globals",
            "f_back",
            "f_builtins",
            "tb_frame",
            "tb_next",
            "gi_frame",
            "cr_frame",
            "ag_frame",
            "settrace",
            "setprofile",
            "_getframe",
            "_current_frames",
            "currentframe",
        }
    )

    def _extract_string_from_node(node):
        """Extract a plain string value from an AST node, if it is a constant."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _extract_folded_string(node):
        """Like _extract_string_from_node but also const-folds a computed string, so a folded
        dynamic-attribute name is recognized: getattr(sys, 'meta_' + 'path') / a const-var
        alias resolve to 'meta_path' exactly like the raw literal."""
        _s = _extract_string_from_node(node)
        if _s is not None:
            return _s
        _f = _const_fold(node, _const_env)
        return _f if isinstance(_f, str) else None

    def _extract_env_scalar(node):
        """A str constant, a const-folded string (a const-var / concatenation via the module const
        env, ``P='.:/usr/bin'; ... P``), or a bytes constant / folded bytes decoded to str
        (os.environb byte keys / values and ``env={'PATH': b'.:'}`` are the same inherited
        environment), else None."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, (bytes, bytearray)):
                try:
                    return bytes(node.value).decode("utf-8", "surrogateescape")
                except Exception:
                    return None
        _f = _const_fold(node, _const_env)
        if isinstance(_f, str):
            return _f
        if isinstance(_f, (bytes, bytearray)):
            try:
                return bytes(_f).decode("utf-8", "surrogateescape")
            except Exception:
                return None
        return None

    def _env_mapping_pairs(node):
        """Flatten a subprocess env= mapping (a literal dict, a dict(...) call, or a nested
        ``**{...}`` splat) into ([(key_str, value_node), ...], opaque). ``opaque`` is True when
        any entry's KEY cannot be resolved to a constant string -- a computed key, or a
        non-literal ``**mapping`` splat -- since such an entry could carry BASH_ENV / ENV /
        GIT_CONFIG_COUNT. Only literal-keyed entries appear in the pair list."""
        pairs = []
        opaque = False
        if isinstance(node, ast.Dict):
            for _k, _v in zip(node.keys, node.values):
                if _k is None:  # **mapping splat
                    if isinstance(_v, ast.Dict):
                        _ip, _io = _env_mapping_pairs(_v)
                        pairs.extend(_ip)
                        opaque = opaque or _io
                    else:
                        opaque = True
                else:
                    _ks = _extract_string_from_node(_k)
                    if _ks is None:
                        opaque = True
                    else:
                        pairs.append((_ks, _v))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            for _kw in node.keywords:
                if _kw.arg is None:  # dict(**mapping)
                    if isinstance(_kw.value, ast.Dict):
                        _ip, _io = _env_mapping_pairs(_kw.value)
                        pairs.extend(_ip)
                        opaque = opaque or _io
                    else:
                        opaque = True
                else:
                    pairs.append((_kw.arg, _kw.value))
        return pairs, opaque

    def _extract_strings_from_list(node):
        """Extract string elements from an AST List or Tuple node."""
        if isinstance(node, (ast.List, ast.Tuple)):
            parts = []
            for elt in node.elts:
                s = _extract_string_from_node(elt)
                if s is not None:
                    parts.append(s)
            return parts
        return []

    # Kwarg names that carry command content (not control flags like
    # check=True, text=True, capture_output=True).
    _CMD_KWARGS = frozenset({"args", "command", "executable", "path", "file"})

    def _check_shell_argv(elts):
        """Analyze a subprocess argv VECTOR (['bash', '-c', '...'], ['sh', 's.sh']) as a
        whole. A shell argv is only safe when it carries an inline -c payload that scans
        clean; a script-file / -s / bare-shell / dynamic-payload form runs unscanned code
        and is denied. Returns a set of blocked markers (empty if not a shell argv or the
        scanned -c payload is benign)."""
        if not elts:
            return set()
        first = _extract_string_from_node(elts[0])
        if first is None or os.path.basename(first).lower() not in _SHELL_BINARIES:
            return set()
        found = set()
        i = 1
        while i < len(elts):
            f = _extract_string_from_node(elts[i])
            # -i / -ic (combined short flag) makes the shell INTERACTIVE, sourcing the user's rc
            # files (.bashrc, with HOME = the workdir) before any -c payload runs -- unscanned
            # startup code in the unguarded child. Mirror the shell-string interactive-rc block.
            if f is not None and f.startswith("-") and not f.startswith("--") and "i" in f[1:]:
                found.add("shell-interactive-rc:" + os.path.basename(first).lower())
            if f is not None and (
                f == "-c" or (f.startswith("-") and not f.startswith("--") and f.endswith("c"))
            ):
                if i + 1 < len(elts):
                    payload = _extract_string_from_node(elts[i + 1])
                    if payload is None:
                        found.add("shell-dynamic-c")  # unanalyzable inline payload
                    else:
                        found |= _find_blocked_commands(payload)
                else:
                    found.add("shell-script:" + first)  # -c with no payload
                return found
            i += 1
        # No -c: a script file, -s (stdin), or a bare shell that reads stdin.
        found.add("shell-script:" + first)
        return found

    def _check_args_for_blocked(
        args_nodes,
        shell_maybe_true = False,
        cwd_prefix = "",
    ):
        """Check if any call arguments contain blocked commands. ``cwd_prefix`` is a synthetic
        ``env -C <dir> `` wrapper string prepended to a reconstructed argv command when the call
        has a literal escaping ``cwd=`` (subprocess.run(['git','init','repo'], cwd='/tmp')), so the
        git cwd backscan resolves the child's real working directory."""
        found = set()
        for arg in args_nodes:
            s = _extract_string_from_node(arg)
            if s is not None:
                found |= _find_blocked_commands(s)
                continue
            if isinstance(arg, (ast.List, ast.Tuple)):
                str_elts = [_extract_string_from_node(e) for e in arg.elts]
                first = str_elts[0] if str_elts else None
                # With shell=True, POSIX subprocess passes the FIRST sequence element to
                # /bin/sh -c as the command string (the rest become $0, $1, ...); so
                # subprocess.run(['echo x > /tmp/p'], shell=True) runs a full shell command,
                # not an argv vector. Scan elts[0] with the shell parser in that case.
                if shell_maybe_true and first is not None:
                    found |= _find_blocked_commands(first)
                # A shell argv vector is analyzed as a whole so `['bash', '-c', 'echo hi']`
                # scans the payload instead of tripping the bare-shell block on the 'bash'
                # element. A non-shell argv only executes its command word (argv[0] plus any
                # wrapper prefix), so scan just that -- scanning every element would misread a
                # benign argument (subprocess.run(['echo', 'python'])) as a blocked command.
                elif first is not None and os.path.basename(first).lower() in _SHELL_BINARIES:
                    found |= _check_shell_argv(arg.elts)
                else:
                    _argv_blocked, _cmd_idx = _blocked_in_argv(str_elts)
                    found |= _argv_blocked
                    if _cmd_idx is not None and str_elts[_cmd_idx] is not None:
                        _cmd_base = os.path.basename(str_elts[_cmd_idx]).lower()
                        # A wrapper-hidden shell binary (env bash s.sh, nice sh -c '...')
                        # resolves to a shell as its command word; analyze the shell + its args
                        # (script file or -c payload) so the bare-shell / unscanned-script forms
                        # are caught.
                        if _cmd_base in _SHELL_BINARIES:
                            found |= _check_shell_argv(arg.elts[_cmd_idx:])
                        # find -exec/-delete, sed -i, sort -o interpret LATER argv elements as
                        # actions / write flags, and git -c alias.X=!CMD hides a shell dispatch
                        # in a config operand, so reconstruct a command line and reuse the full
                        # scanner (which handles those forms). Reconstruct from the FULL argv (not
                        # just the command word onward) so a preceding wrapper -- e.g. an escaping
                        # env -C /tmp before git -- is still seen by the git cwd backscan.
                        elif _cmd_base in _ARGV_TAIL_SCAN_COMMANDS:
                            found |= _find_blocked_commands(
                                cwd_prefix
                                + " ".join(shlex.quote(s) for s in str_elts if s is not None)
                            )
                    # An env WRAPPER in the argv applies NAME=value assignments before the command
                    # (env PATH=. evil, env BASH_ENV=env.sh bash -c ..., env GIT_DIR=/tmp git init);
                    # the command-word resolution skips those assignment operands, so reconstruct
                    # the full argv and reuse the unsafe-PATH / startup-env / git-env checks.
                    if any(
                        s is not None and os.path.basename(s).lower() == "env" for s in str_elts
                    ) and any(s is not None and _ASSIGNMENT_RE.match(s) for s in str_elts):
                        found |= _find_blocked_commands(
                            " ".join(shlex.quote(s) for s in str_elts if s is not None)
                        )
                continue
            for s in _extract_strings_from_list(arg):
                found |= _find_blocked_commands(s)
        return found

    class SignalEscapeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.imports_signal = False
            self.signal_aliases = {"signal"}
            self.os_aliases = {"os"}
            # Names bound to os.environ / os.environb (bare, from-imported as X, or e = os.environ),
            # so an aliased env mutation (e['BASH_ENV'] = ...) is still recognized.
            self.environ_aliases = {"environ", "environb"}
            # from os import putenv as p -> {"putenv", "p"}: os.putenv(key, value) sets an
            # inherited env var through the C-level setter (not os.environ), a later-child escape.
            self.putenv_aliases: set[str] = set()
            self.subprocess_aliases = {"subprocess"}
            # import asyncio as aio -> {"asyncio", "aio"}. asyncio.create_subprocess_shell /
            # create_subprocess_exec start the SAME unguarded child as subprocess.run/Popen.
            self.asyncio_aliases = {"asyncio"}
            # from asyncio import create_subprocess_shell as s -> {"s": "create_subprocess_shell"}.
            self.asyncio_subprocess_from_aliases: dict[str, str] = {}
            self.importlib_aliases = {"importlib"}
            self.sys_aliases = {"sys"}
            # __builtins__ is the builtins *module* in __main__ (how the sandbox runs
            # user code as `python <file>.py`), so builtins.eval / __builtins__.eval work.
            self.builtins_aliases = {"builtins", "__builtins__"}
            # Bare name -> fully-qualified form for from-import tracking
            # (e.g. "system" -> "os.system").
            self.shell_exec_aliases: dict[str, str] = {}
            # from importlib import import_module as im  ->  {"im"}
            self.import_func_aliases: set[str] = set()
            # from builtins import exec as e  ->  {"e": "exec"}
            self.exec_from_aliases: dict[str, str] = {}
            # import pickle as p  ->  {"p": "pickle"}; from pickle import loads as l -> {"l": "pickle.loads"}
            self.deserialize_module_aliases: dict[str, str] = {}
            self.deserialize_aliases: dict[str, str] = {}
            # `from yaml import load [as X]` / load_all: yaml.load is conditional (safe only
            # with a SafeLoader) so it is not a static sink; track the bare-name alias so the
            # safe-loader check can be applied to the direct-call form.
            self.yaml_load_aliases: dict[str, str] = {}
            # `from pickle import Unpickler [as X]`: Unpickler(f).load() reaches the same reduce
            # path as pickle.load; track the ctor alias so the .load() method call is flagged.
            self.unpickler_aliases: set[str] = set()
            # import torch / numpy as np / joblib -> {alias: module}; from joblib import load ->
            # {load: "joblib.load"}. Conditional pickle-backed loaders (see _pickle_loader_is_unsafe).
            self.pickle_loader_module_aliases: dict[str, str] = {}
            self.pickle_loader_func_aliases: dict[str, str] = {}
            # import types as t -> {"types", "t"}; from types import FunctionType as F -> {"F"}.
            # FunctionType(code, globals)() runs a code object WITHOUT eval/exec, so a
            # dynamic compile() result reaches execution through it (see visit_Call).
            self.types_aliases = {"types"}
            self.functiontype_aliases: set[str] = set()
            # import runpy as r -> {"runpy", "r"}. runpy.run_path/run_module execute a
            # file/module in the guarded interpreter without the recursive source
            # analysis exec/eval receive, so treat those calls as execution sinks.
            self.runpy_aliases = {"runpy"}
            # from runpy import run_path as X / run_module as Y -> {"X", "Y"}.
            self.runpy_func_aliases: set[str] = set()
            # import timeit as t -> {"timeit", "t"}. timeit.timeit / .repeat / Timer(...) COMPILE
            # and EXECUTE their stmt / setup STRING args, so a string payload there is analyzed
            # like an exec payload (a callable stmt carries no source and stays allowed).
            self.timeit_aliases = {"timeit"}
            # from timeit import timeit as X / repeat as Y / Timer as Z -> {"X", "Y", "Z"}.
            self.timeit_func_aliases: set[str] = set()
            # import inspect as i -> {"inspect", "i"}. inspect.getclosurevars(fn) hands back
            # the cells a guard wrapper closes over (the original unguarded callable), so
            # treat it as a closure-recovery gadget like __closure__ / cell_contents.
            self.inspect_aliases = {"inspect"}
            # from inspect import getclosurevars as g -> {"g"}.
            self.getclosurevars_aliases: set[str] = set()
            # import operator as op -> {"operator", "op"}. operator.attrgetter('name')(obj)
            # is the same attribute-fetch obfuscation as getattr(obj, 'name').
            self.operator_aliases = {"operator"}
            # from operator import attrgetter as ag -> {"ag"}.
            self.attrgetter_aliases: set[str] = set()
            # from operator import methodcaller as mc -> {"mc"}. methodcaller('__getattribute__',
            # 'system')(os) fetches os.system, the same obfuscation as attrgetter.
            self.methodcaller_aliases: set[str] = set()
            # import gc as g -> {"gc", "g"}. gc.get_referents / get_referrers / get_objects
            # walk the object graph to a guard wrapper's closure cell (the original unguarded
            # callable) without spelling __closure__, so treat them as recovery gadgets.
            self.gc_aliases = {"gc"}
            # from gc import get_referents as gr -> {"gr"}.
            self.gc_walk_aliases: set[str] = set()
            # import pty as p -> {"pty", "p"}. pty.spawn([...]) / pty.fork() run an unguarded
            # child process (a shell) outside the sandbox.
            self.pty_aliases: set[str] = set()
            # from pty import spawn as s -> {"s"}: bare-name aliases of the pty child sinks.
            self.pty_func_aliases: set[str] = set()
            self.loop_depth = 0

        def visit_Import(self, node):
            for alias in node.names:
                # import ctypes / import ctypes.util / import cffi: native FFI, unguardable.
                if alias.name.split(".")[0] in _NATIVE_ESCAPE_MODULES:
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                f"import of native FFI module {alias.name!r} makes unguarded "
                                "libc / syscall calls that bypass the sandbox filesystem confinement"
                            ),
                        }
                    )
                if alias.name == "signal":
                    self.imports_signal = True
                    if alias.asname:
                        self.signal_aliases.add(alias.asname)
                elif alias.name == "os":
                    self.os_aliases.add(alias.asname or "os")
                elif alias.name in ("posix", "nt"):
                    # posix / nt are the C backend os wraps: posix.system(...) == os.system,
                    # and posix.exec*/spawn*/popen mirror os. Model them as os aliases so a
                    # direct `import posix; posix.system('...')` resolves to an os shell sink.
                    self.os_aliases.add(alias.asname or alias.name)
                elif alias.name == "subprocess":
                    self.subprocess_aliases.add(alias.asname or "subprocess")
                elif alias.name == "asyncio":
                    self.asyncio_aliases.add(alias.asname or "asyncio")
                elif alias.name == "pty":
                    # pty.spawn([...]) / pty.fork() run an unguarded child process.
                    self.pty_aliases.add(alias.asname or "pty")
                elif alias.name == "importlib":
                    self.importlib_aliases.add(alias.asname or "importlib")
                elif alias.name == "sys":
                    self.sys_aliases.add(alias.asname or "sys")
                elif alias.name == "builtins":
                    self.builtins_aliases.add(alias.asname or "builtins")
                elif alias.name == "types":
                    self.types_aliases.add(alias.asname or "types")
                elif alias.name == "runpy":
                    self.runpy_aliases.add(alias.asname or "runpy")
                elif alias.name == "timeit":
                    self.timeit_aliases.add(alias.asname or "timeit")
                elif alias.name == "inspect":
                    self.inspect_aliases.add(alias.asname or "inspect")
                elif alias.name == "operator":
                    self.operator_aliases.add(alias.asname or "operator")
                elif alias.name == "gc":
                    self.gc_aliases.add(alias.asname or "gc")
                if alias.name in _DESERIALIZE_MODULES:
                    self.deserialize_module_aliases[alias.asname or alias.name] = alias.name
                # import torch / import numpy as np / import joblib: the top-level name (or its
                # alias) is the receiver for torch.load / np.load / joblib.load.
                _pl_top = alias.name.split(".")[0]
                if _pl_top in _PICKLE_LOADER_MODULES and "." not in (alias.asname or alias.name):
                    self.pickle_loader_module_aliases[alias.asname or alias.name] = _pl_top
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            # from ctypes import CDLL / from cffi import FFI: native FFI, unguardable.
            if node.module and node.module.split(".")[0] in _NATIVE_ESCAPE_MODULES:
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": (
                            f"import from native FFI module {node.module!r} makes unguarded "
                            "libc / syscall calls that bypass the sandbox filesystem confinement"
                        ),
                    }
                )
            if node.module == "signal":
                self.imports_signal = True
                for alias in node.names:
                    if alias.name in (
                        "signal",
                        "SIGALRM",
                        "SIG_IGN",
                        "setitimer",
                        "ITIMER_REAL",
                        "pthread_sigmask",
                        "SIG_BLOCK",
                        "alarm",
                    ):
                        self.signal_aliases.add(alias.asname or alias.name)
            elif node.module in ("os", "subprocess", "posix", "nt"):
                if node.module == "subprocess":
                    self.subprocess_aliases.add("subprocess")
                    _eff_mod = "subprocess"
                else:
                    # posix / nt are the C backend os wraps: `from posix import system` is
                    # os.system, so model the sink under os so it is caught the same way.
                    self.os_aliases.add("os")
                    _eff_mod = "os"
                # Track from-imports of dangerous functions.
                for alias in node.names:
                    fq = f"{_eff_mod}.{alias.name}"
                    if fq in _SHELL_EXEC_FUNCS:
                        self.shell_exec_aliases[alias.asname or alias.name] = fq
                    # from os import environ as e / from os import environb as eb.
                    if _eff_mod == "os" and alias.name in ("environ", "environb"):
                        self.environ_aliases.add(alias.asname or alias.name)
                    # from os import putenv as p.
                    if _eff_mod == "os" and alias.name == "putenv":
                        self.putenv_aliases.add(alias.asname or alias.name)
            elif node.module == "asyncio":
                for alias in node.names:
                    if alias.name in ("create_subprocess_shell", "create_subprocess_exec"):
                        self.asyncio_subprocess_from_aliases[alias.asname or alias.name] = (
                            alias.name
                        )
            elif node.module == "importlib":
                for alias in node.names:
                    if alias.name in ("import_module", "reload", "__import__"):
                        self.import_func_aliases.add(alias.asname or alias.name)
            elif node.module == "builtins":
                for alias in node.names:
                    if alias.name in _DYNAMIC_EXEC_BUILTINS:
                        self.exec_from_aliases[alias.asname or alias.name] = alias.name
                    elif alias.name == "__import__":
                        # `from builtins import __import__ as imp; imp('os').system(...)`
                        # is a dynamic import exactly like a bare __import__ call.
                        self.import_func_aliases.add(alias.asname or alias.name)
            elif node.module in _DESERIALIZE_MODULES:
                for alias in node.names:
                    fq = f"{node.module}.{alias.name}"
                    if fq in _CODE_DESERIALIZE_SINKS:
                        self.deserialize_aliases[alias.asname or alias.name] = fq
                    elif node.module == "yaml" and alias.name in _YAML_LOAD_METHODS:
                        self.yaml_load_aliases[alias.asname or alias.name] = alias.name
                    elif alias.name == "Unpickler" and node.module in _UNPICKLER_MODULES:
                        self.unpickler_aliases.add(alias.asname or alias.name)
            elif node.module in _PICKLE_LOADER_MODULES:
                # from joblib import load / from torch import load: the bare-name form of the
                # conditional pickle-backed loader; the safe-flag gate is applied at the call.
                for alias in node.names:
                    if alias.name == "load":
                        self.pickle_loader_func_aliases[alias.asname or alias.name] = (
                            f"{node.module}.load"
                        )
            elif node.module == "types":
                for alias in node.names:
                    if alias.name == "FunctionType":
                        self.functiontype_aliases.add(alias.asname or alias.name)
            elif node.module == "runpy":
                for alias in node.names:
                    if alias.name in ("run_path", "run_module"):
                        self.runpy_func_aliases.add(alias.asname or alias.name)
            elif node.module == "pty":
                # from pty import spawn / fork: bare-name aliases of the pty child sinks.
                for alias in node.names:
                    if alias.name in ("spawn", "fork"):
                        self.pty_func_aliases.add(alias.asname or alias.name)
            elif node.module == "timeit":
                # from timeit import timeit / repeat / Timer: bare-name aliases of the
                # string-executing entry points.
                for alias in node.names:
                    if alias.name in ("timeit", "repeat", "Timer"):
                        self.timeit_func_aliases.add(alias.asname or alias.name)
            elif node.module == "inspect":
                for alias in node.names:
                    if alias.name == "getclosurevars":
                        self.getclosurevars_aliases.add(alias.asname or alias.name)
            elif node.module == "operator":
                for alias in node.names:
                    if alias.name == "attrgetter":
                        self.attrgetter_aliases.add(alias.asname or alias.name)
                    elif alias.name == "methodcaller":
                        self.methodcaller_aliases.add(alias.asname or alias.name)
            elif node.module == "gc":
                for alias in node.names:
                    if alias.name in ("get_referents", "get_referrers", "get_objects"):
                        self.gc_walk_aliases.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_While(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_For(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def _resolve_container_sink(self, sub):
            """Resolve an inline literal-container index callee to a shell sink fq.

            Covers ``[os.system][0]``, ``(os.system,)[0]`` and ``{'k': os.system}['k']``.
            """

            def _elt(elt):
                fq = _resolve_static_shell_sink(
                    elt, self.os_aliases, self.subprocess_aliases, self.shell_exec_aliases
                )
                if fq:
                    return fq
                if isinstance(elt, ast.Name):
                    return _scope_idx.resolve(elt.id, elt, "shell")
                return None

            container = sub.value
            # Resolve a single-assignment NAME container (d = [os.system]; d[0]('rm -rf /')) to its
            # literal, mirroring the exec-container resolver -- a shell sink hidden in an assigned
            # container was otherwise missed because the callee is an indexed Name.
            if isinstance(container, ast.Name) and container.id in _const_env:
                container = _const_env[container.id]
            ci = _const_fold(sub.slice, _const_env)
            if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
                if -len(container.elts) <= ci < len(container.elts):
                    return _elt(container.elts[ci])
            if isinstance(container, ast.Dict) and ci is not None:
                for k, v in zip(container.keys, container.values):
                    if k is not None and _const_fold(k, _const_env) == ci:
                        return _elt(v)
            return None

        def _resolve_container_exec(self, sub):
            """Resolve an inline literal-container index callee to a dynamic-exec builtin.

            Covers ({'e': exec}['e'])(...), [exec][0](...), (eval,)[0](...): an inline
            container hiding an eval/exec/compile sink from the bare-name recursion."""

            def _elt(elt):
                if isinstance(elt, ast.Name):
                    if elt.id in _DYNAMIC_EXEC_BUILTINS:
                        return elt.id
                    if elt.id in self.exec_from_aliases:
                        return self.exec_from_aliases[elt.id]
                    if _analyzer_on:
                        return _scope_idx.resolve(elt.id, elt, "execb")
                if (
                    isinstance(elt, ast.Attribute)
                    and elt.attr in _DYNAMIC_EXEC_BUILTINS
                    and _ast_name_matches(elt.value, self.builtins_aliases)
                ):
                    return elt.attr
                return None

            container = sub.value
            # Resolve a subscript into a container bound to a single-assignment NAME
            # (d = {'e': exec}; d['e'](...), xs = [eval]; xs[0](...)) to the literal container,
            # so the exec/eval sink hidden inside it is not missed just because the callee is an
            # indexed Name rather than an inline literal.
            if isinstance(container, ast.Name) and container.id in _const_env:
                container = _const_env[container.id]
            ci = _const_fold(sub.slice, _const_env)
            if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
                if -len(container.elts) <= ci < len(container.elts):
                    return _elt(container.elts[ci])
            if isinstance(container, ast.Dict) and ci is not None:
                for k, v in zip(container.keys, container.values):
                    if k is not None and _const_fold(k, _const_env) == ci:
                        return _elt(v)
            return None

        def _direct_exec_callee_id(self, func):
            """Resolve a NON-composite callee expression to an eval/exec/compile id, else None.

            Composite forms (a ternary `a if c else b`, a boolean fallback `x or eval`) are
            peeled by _resolve_exec_callee, which delegates each branch here."""
            if isinstance(func, ast.Name):
                if func.id in _DYNAMIC_EXEC_BUILTINS:
                    return func.id
                if func.id in self.exec_from_aliases:
                    return self.exec_from_aliases[func.id]  # from builtins import exec as e
                if _analyzer_on:
                    # single-assignment `e = exec` alias, resolved in the call's scope.
                    return _scope_idx.resolve(func.id, func, "execb")
                return None
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _DYNAMIC_EXEC_BUILTINS
                and _ast_name_matches(func.value, self.builtins_aliases)
            ):
                return func.attr  # builtins.eval(...) / __builtins__.exec(...)
            if isinstance(func, ast.Attribute) and func.attr == "__call__":
                # eval.__call__("...") / exec.__call__(...) / builtins.eval.__call__(...)
                _base = func.value
                if isinstance(_base, ast.Name):
                    if _base.id in _DYNAMIC_EXEC_BUILTINS:
                        return _base.id
                    if _base.id in self.exec_from_aliases:
                        return self.exec_from_aliases[_base.id]
                    if _analyzer_on:
                        return _scope_idx.resolve(_base.id, _base, "execb")
                    return None
                if (
                    isinstance(_base, ast.Attribute)
                    and _base.attr in _DYNAMIC_EXEC_BUILTINS
                    and _ast_name_matches(_base.value, self.builtins_aliases)
                ):
                    return _base.attr
                return None
            if (
                _analyzer_on
                and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
            ):
                # class-body alias reached as ClassName.attr (class C: e = eval; C.e('...')),
                # or an instance-attribute alias (c.e = exec; c.e('...')).
                _eid = _scope_idx.resolve_class_attr(func.value.id, func.attr, "execb")
                if _eid is None:
                    _eid = _scope_idx.resolve_instance_attr(func.value.id, func.attr, "execb")
                return _eid
            if isinstance(func, ast.Subscript):
                # __builtins__['exec'] / builtins['eval']: a subscript of a builtins alias by a
                # constant exec-builtin name. The container resolver below only walks user
                # literals ({'e': exec}['e']), so the builtins mapping is handled explicitly.
                if _ast_name_matches(func.value, self.builtins_aliases):
                    _key = _const_fold(func.slice, _const_env)
                    if isinstance(_key, str) and _key in _DYNAMIC_EXEC_BUILTINS:
                        return _key
                # ({'e': exec}['e'])(...) / [exec][0](...): an inline container hides the
                # sink from the bare-name / attribute checks above.
                return self._resolve_container_exec(func)
            return None

        def _resolve_exec_callee(
            self,
            func,
            _depth = 0,
        ):
            """Resolve a callee expression to an eval/exec/compile id, peeling composites.

            A ternary ((eval if c else exec)('...')) or a boolean fallback
            ((getattr(__builtins__, 'ev', None) or eval)('...')) evaluates to a dynamic-exec
            builtin without the callee being a bare Name / Attribute. Fail closed: for a
            ternary or an and/or chain, ANY branch that can resolve to an exec builtin taints
            the whole call, since which branch runs is not statically known."""
            if _depth > 8 or func is None:
                return None
            if isinstance(func, ast.IfExp):
                return self._resolve_exec_callee(
                    func.body, _depth + 1
                ) or self._resolve_exec_callee(func.orelse, _depth + 1)
            if isinstance(func, ast.BoolOp):
                for _v in func.values:
                    _hit = self._resolve_exec_callee(_v, _depth + 1)
                    if _hit is not None:
                        return _hit
                return None
            return self._direct_exec_callee_id(func)

        def _resolve_container_deser(self, sub):
            """Resolve an inline literal-container index callee to a deserializer sink fq.

            Covers ([pickle.loads][0])(payload), (pickle.loads,)[0](...) and
            {'k': pickle.loads}['k'](...): an inline container hiding a pickle/marshal
            reduce sink from the attribute/name deserializer checks."""

            def _elt(elt):
                if isinstance(elt, ast.Attribute):
                    if isinstance(elt.value, ast.Name):
                        canon = self.deserialize_module_aliases.get(elt.value.id)
                        if canon is not None:
                            cand = f"{canon}.{elt.attr}"
                            if cand in _CODE_DESERIALIZE_SINKS:
                                return cand
                    fq = _fq_attr_name(elt)
                    if fq in _CODE_DESERIALIZE_SINKS:
                        return fq
                elif isinstance(elt, ast.Name):
                    fq = self.deserialize_aliases.get(elt.id)
                    if fq is not None:
                        return fq
                    if _analyzer_on:
                        return _scope_idx.resolve(elt.id, elt, "deser")
                return None

            container = sub.value
            # Resolve a single-assignment NAME container (d = {'k': pickle.loads}; d['k'](payload))
            # to its literal, mirroring the exec-container resolver above.
            if isinstance(container, ast.Name) and container.id in _const_env:
                container = _const_env[container.id]
            ci = _const_fold(sub.slice, _const_env)
            if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
                if -len(container.elts) <= ci < len(container.elts):
                    return _elt(container.elts[ci])
            if isinstance(container, ast.Dict) and ci is not None:
                for k, v in zip(container.keys, container.values):
                    if k is not None and _const_fold(k, _const_env) == ci:
                        return _elt(v)
            return None

        def _is_unpickler_ctor(self, recv):
            """True when ``recv`` constructs a pickle/dill Unpickler instance.

            Covers pickle.Unpickler(f) (incl. an aliased module: import pickle as p;
            p.Unpickler(f)) and a from-imported ctor (from pickle import Unpickler;
            Unpickler(f)); its .load() runs the same reduce payload as pickle.load."""
            if not isinstance(recv, ast.Call):
                return False
            cf = recv.func
            if isinstance(cf, ast.Attribute) and cf.attr == "Unpickler":
                if isinstance(cf.value, ast.Name):
                    return self.deserialize_module_aliases.get(cf.value.id) in _UNPICKLER_MODULES
                return _fq_attr_name(cf) in {m + ".Unpickler" for m in _UNPICKLER_MODULES}
            if isinstance(cf, ast.Name):
                return cf.id in self.unpickler_aliases
            return False

        def _attrgetter_name(self, n):
            """Return the single attribute name for an ``operator.attrgetter('name')``
            call (or a ``from operator import attrgetter`` alias), else None. A dotted or
            multi-attr getter (attrgetter('a.b'), attrgetter('a', 'b')) returns None."""
            if not isinstance(n, ast.Call) or len(n.args) != 1 or n.keywords:
                return None
            af = n.func
            is_attrgetter = (
                isinstance(af, ast.Attribute)
                and af.attr == "attrgetter"
                and _ast_name_matches(af.value, self.operator_aliases)
            ) or (isinstance(af, ast.Name) and af.id in self.attrgetter_aliases)
            if not is_attrgetter:
                return None
            name = _const_fold(n.args[0], _const_env)
            if isinstance(name, str) and "." not in name:
                return name
            return None

        def _methodcaller_getattr_name(self, n):
            """Return the attribute name for an ``operator.methodcaller('__getattribute__',
            'name')`` / ``__getattr__`` call (or a from-import alias), else None. This form
            fetches ``obj.name`` exactly like attrgetter, so it needs the same normalization."""
            if not isinstance(n, ast.Call) or len(n.args) != 2 or n.keywords:
                return None
            af = n.func
            is_mc = (
                isinstance(af, ast.Attribute)
                and af.attr == "methodcaller"
                and _ast_name_matches(af.value, self.operator_aliases)
            ) or (isinstance(af, ast.Name) and af.id in self.methodcaller_aliases)
            if not is_mc:
                return None
            meth = _const_fold(n.args[0], _const_env)
            if meth in ("__getattribute__", "__getattr__"):
                name = _const_fold(n.args[1], _const_env)
                if isinstance(name, str) and "." not in name:
                    return name
            return None

        def _methodcaller_module_call(self, node):
            """Rewrite ``operator.methodcaller('meth', *args)(receiver)`` into the equivalent
            ``receiver.meth(*args)`` Call when the receiver is an os/subprocess module
            reference, so a methodcaller-hidden sink -- methodcaller('system', 'rm -rf /')(os)
            -- is analyzed exactly like the direct os.system('rm -rf /') call. Returns the
            synthetic Call node (with the original location) or None when the pattern does not
            apply. Only os/subprocess receivers are rewritten; a methodcaller aimed at some
            other object is left untouched so benign method calls are not misread."""
            if len(node.args) != 1 or node.keywords:
                return None
            mc = node.func
            while isinstance(mc, ast.Attribute) and mc.attr == "__call__":
                mc = mc.value
            if not isinstance(mc, ast.Call) or not mc.args:
                return None
            mf = mc.func
            is_mc = (
                isinstance(mf, ast.Attribute)
                and mf.attr == "methodcaller"
                and _ast_name_matches(mf.value, self.operator_aliases)
            ) or (isinstance(mf, ast.Name) and mf.id in self.methodcaller_aliases)
            if not is_mc:
                return None
            meth = _const_fold(mc.args[0], _const_env)
            if not isinstance(meth, str) or not meth.isidentifier():
                return None
            receiver = node.args[0]
            if not (
                isinstance(receiver, ast.Name)
                and (receiver.id in self.os_aliases or receiver.id in self.subprocess_aliases)
            ):
                return None
            synth = ast.Call(
                func = ast.Attribute(value = receiver, attr = meth, ctx = ast.Load()),
                args = list(mc.args[1:]),
                keywords = list(mc.keywords),
            )
            ast.copy_location(synth, node)
            ast.fix_missing_locations(synth)
            return synth

        def _asyncio_subprocess_rewrite(self, node):
            """Rewrite ``asyncio.create_subprocess_shell(cmd, ...)`` /
            ``asyncio.create_subprocess_exec(prog, *args, ...)`` into the equivalent
            ``subprocess.run(cmd, shell=True)`` / ``subprocess.run([prog, *args])`` Call, so the
            SAME child-process command analysis (blocked commands, shell payload, argv / cwd / env
            escape) applies -- asyncio starts the same unguarded child the runtime open/os guards
            never see. Covers the module-attribute form and a `from asyncio import
            create_subprocess_shell` bare alias. Returns the synthetic Call or None."""
            _fn = node.func
            while isinstance(_fn, ast.Attribute) and _fn.attr == "__call__":
                _fn = _fn.value
            _kind = None
            if (
                isinstance(_fn, ast.Attribute)
                and _fn.attr in ("create_subprocess_shell", "create_subprocess_exec")
                and isinstance(_fn.value, ast.Name)
                and _fn.value.id in self.asyncio_aliases
            ):
                _kind = "shell" if _fn.attr == "create_subprocess_shell" else "exec"
            elif isinstance(_fn, ast.Name) and _fn.id in self.asyncio_subprocess_from_aliases:
                _kind = (
                    "shell"
                    if self.asyncio_subprocess_from_aliases[_fn.id] == "create_subprocess_shell"
                    else "exec"
                )
            if _kind is None or not node.args:
                return None
            # Carry cwd= / env= so the cwd-escape and startup-env (BASH_ENV / PATH) analysis fires.
            _carry = [_kw for _kw in node.keywords if _kw.arg in ("cwd", "env")]
            if _kind == "shell":
                # create_subprocess_shell(cmd) runs cmd via /bin/sh -c, i.e. shell=True.
                _sargs = [node.args[0]]
                _skw = [ast.keyword(arg = "shell", value = ast.Constant(value = True))] + _carry
            else:
                # create_subprocess_exec(prog, *args) is the argv vector (shell=False).
                _sargs = [ast.List(elts = list(node.args), ctx = ast.Load())]
                _skw = list(_carry)
            _synth = ast.Call(
                func = ast.Attribute(
                    value = ast.Name(id = "subprocess", ctx = ast.Load()),
                    attr = "run",
                    ctx = ast.Load(),
                ),
                args = _sargs,
                keywords = _skw,
            )
            ast.copy_location(_synth, node)
            ast.fix_missing_locations(_synth)
            return _synth

        def _sink_ref_desc(self, n):
            """Describe ``n`` when it is a bare reference to a dangerous callable used as a
            first-class VALUE (map/reduce/partial argument): a dynamic-exec builtin, a shell
            sink (os.system / subprocess.*), a dynamic-import function, or a code
            deserializer. Returns a short description or None. The payloads such a sink runs
            never reach the recursive analyzer, so passing one by reference is unsafe."""
            if isinstance(n, ast.Subscript):
                # An inline literal-container index hides the sink from the name/attribute
                # checks: map([eval][0], [...]) / partial({'e': exec}['e'], ...). Resolve the
                # element node and describe it, the same unwrap direct calls already apply.
                container = n.value
                ci = _const_fold(n.slice, _const_env)
                elt = None
                if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
                    if -len(container.elts) <= ci < len(container.elts):
                        elt = container.elts[ci]
                elif isinstance(container, ast.Dict) and ci is not None:
                    for _k, _v in zip(container.keys, container.values):
                        if _k is not None and _const_fold(_k, _const_env) == ci:
                            elt = _v
                            break
                return self._sink_ref_desc(elt) if elt is not None else None
            if isinstance(n, ast.Name):
                if n.id in _DYNAMIC_EXEC_BUILTINS:
                    return f"{n.id} (dynamic exec)"
                if n.id in self.exec_from_aliases:
                    return f"{self.exec_from_aliases[n.id]} (dynamic exec)"
                if n.id in ("__import__", "import_module") or n.id in self.import_func_aliases:
                    return "dynamic import"
                _sh = self.shell_exec_aliases.get(n.id)
                if _sh in _SHELL_EXEC_FUNCS:
                    return f"{_sh} (shell)"
                _ds = self.deserialize_aliases.get(n.id)
                if _ds is not None:
                    return f"{_ds} (deserialize)"
                # The subprocess / pty MODULE passed by reference (f(subprocess)) is a child-spawn
                # primitive a callee can invoke as subprocess.run(...) with an unguarded escape;
                # the recursive analyzer never sees that call, and a workdir helper receiving the
                # module as a parameter cannot resolve it. Flag the module reference itself.
                if n.id in self.subprocess_aliases:
                    return "subprocess module (child spawn)"
                if n.id in self.pty_aliases:
                    return "pty module (child spawn)"
                if _analyzer_on:
                    _r = _scope_idx.resolve(n.id, n, "shell")
                    if _r in _SHELL_EXEC_FUNCS:
                        return f"{_r} (shell)"
                    if _scope_idx.resolve(n.id, n, "execb") in _DYNAMIC_EXEC_BUILTINS:
                        return f"{_scope_idx.resolve(n.id, n, 'execb')} (dynamic exec)"
                    _rd = _scope_idx.resolve(n.id, n, "deser")
                    if _rd:
                        return f"{_rd} (deserialize)"
                return None
            if isinstance(n, ast.Attribute):
                if n.attr in _DYNAMIC_EXEC_BUILTINS and _ast_name_matches(
                    n.value, self.builtins_aliases
                ):
                    return f"{n.attr} (dynamic exec)"
                _sh = _resolve_static_shell_sink(
                    n, self.os_aliases, self.subprocess_aliases, self.shell_exec_aliases
                )
                if _sh in _SHELL_EXEC_FUNCS:
                    return f"{_sh} (shell)"
                if isinstance(n.value, ast.Name):
                    _c = self.deserialize_module_aliases.get(n.value.id)
                    if _c is not None and f"{_c}.{n.attr}" in _CODE_DESERIALIZE_SINKS:
                        return f"{_c}.{n.attr} (deserialize)"
                _fq = _fq_attr_name(n)
                if _fq in _CODE_DESERIALIZE_SINKS:
                    return f"{_fq} (deserialize)"
                if _fq in _SHELL_EXEC_FUNCS:
                    return f"{_fq} (shell)"
                return None
            return None

        def _rhs_module_attr(self, func, attrs, mod_aliases):
            """A single-assignment alias (x = mod.attr; x(...)) resolving to an attribute in
            ``attrs`` on a module in ``mod_aliases``. Returns the attribute name or None, so a
            re-bound execution sink (r = runpy.run_path, s = pty.spawn) is caught the same as
            the direct call."""
            if not (_analyzer_on and isinstance(func, ast.Name)):
                return None
            rhs = _scope_idx.resolve(func.id, func, "rhsnode")
            if (
                isinstance(rhs, ast.Attribute)
                and rhs.attr in attrs
                and _ast_name_matches(rhs.value, mod_aliases)
            ):
                return rhs.attr
            return None

        def _is_sys_modules_expr(self, n):
            # `sys.modules` (attribute form) or getattr(sys, 'modules') (the getattr-
            # obfuscated form) -- both denote the loader table itself.
            if (
                isinstance(n, ast.Attribute)
                and n.attr == "modules"
                and _ast_name_matches(n.value, self.sys_aliases)
            ):
                return True
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "getattr"
                and len(n.args) >= 2
                and _ast_name_matches(n.args[0], self.sys_aliases)
                and _extract_folded_string(n.args[1]) == "modules"
            ):
                return True
            # object.__getattribute__(sys, 'modules') / type(sys).__getattribute__(sys,
            # 'modules'): the unbound dunder accessor reaches the loader table exactly like
            # getattr, so treat it the same.
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("__getattribute__", "__getattr__")
                and len(n.args) >= 2
                and _ast_name_matches(n.args[0], self.sys_aliases)
                and _extract_folded_string(n.args[1]) == "modules"
            ):
                return True
            return False

        def _is_sys_modules(self, n):
            # `sys.modules` / getattr(sys, 'modules'), or a single-assignment alias of
            # either (m = sys.modules; m.pop('_io')). Used by the loader-table mutation
            # checks (sys.modules.pop('posix'); import posix drops the guard-patched module).
            if self._is_sys_modules_expr(n):
                return True
            if _analyzer_on and isinstance(n, ast.Name):
                rhs = _scope_idx.resolve(n.id, n, "rhsnode")
                if rhs is not None and self._is_sys_modules_expr(rhs):
                    return True
            return False

        def _is_sys_meta_path_expr(self, n):
            # `sys.meta_path` (attribute form) or getattr(sys, 'meta_path') -- the import
            # finder chain into which the sandbox installs its workdir-module vetter.
            if (
                isinstance(n, ast.Attribute)
                and n.attr == "meta_path"
                and _ast_name_matches(n.value, self.sys_aliases)
            ):
                return True
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "getattr"
                and len(n.args) >= 2
                and _ast_name_matches(n.args[0], self.sys_aliases)
                and _extract_folded_string(n.args[1]) == "meta_path"
            ):
                return True
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("__getattribute__", "__getattr__")
                and len(n.args) >= 2
                and _ast_name_matches(n.args[0], self.sys_aliases)
                and _extract_folded_string(n.args[1]) == "meta_path"
            ):
                return True
            return False

        def _is_sys_meta_path(self, n):
            # `sys.meta_path` (or getattr form), or a single-assignment alias of either
            # (mp = sys.meta_path; mp.pop(0)). Used by the import-hook mutation checks: removing
            # / reordering the vetter lets a planted workdir module import without source review.
            if self._is_sys_meta_path_expr(n):
                return True
            if _analyzer_on and isinstance(n, ast.Name):
                rhs = _scope_idx.resolve(n.id, n, "rhsnode")
                if rhs is not None and self._is_sys_meta_path_expr(rhs):
                    return True
            return False

        def _is_namespace_dict_expr(self, n):
            # globals() / locals() / vars() with no args, or a single-assignment alias of one
            # (g = globals(); g['__builtins__']). Used by the namespace-dict subscript check.
            def _direct(x):
                return (
                    isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Name)
                    and x.func.id in ("globals", "locals", "vars")
                    and not x.args
                )

            if _direct(n):
                return True
            if _analyzer_on and isinstance(n, ast.Name):
                rhs = _scope_idx.resolve(n.id, n, "rhsnode")
                if rhs is not None and _direct(rhs):
                    return True
            return False

        def _is_builtins_ref(self, n):
            # The builtins module (builtins / __builtins__ / an import alias), or a
            # single-assignment alias of one (b = __builtins__; b.__import__('os')).
            if _ast_name_matches(n, self.builtins_aliases):
                return True
            if _analyzer_on and isinstance(n, ast.Name):
                rhs = _scope_idx.resolve(n.id, n, "rhsnode")
                if rhs is not None and _ast_name_matches(rhs, self.builtins_aliases):
                    return True
            return False

        def _functiontype_arg_is_vetted(self, arg):
            """True only when a ``types.FunctionType(code, ...)`` first arg is statically KNOWN to
            be an ordinary in-source function's code object -- ``fn.__code__`` / ``meth.__func__``
            -- whose body the analyzer already walked. Fails CLOSED for everything else: a
            ``compile(...)`` result, a loader's ``get_code()``, ``codeop.compile_command()``,
            ``marshal.loads()``, a bare name / alias, or a container unwrap all yield a code object
            running source the recursive eval/exec analysis never saw, so FunctionType is the
            execution gadget and must be blocked. (Denylisting only ``compile`` left other producers
            open; an allowlist of the one benign shape is robust against new producers.)"""
            return isinstance(arg, ast.Attribute) and arg.attr in ("__code__", "__func__")

        def _attr_obfuscation_targets(self):
            # Modules whose DYNAMIC attribute / dict access (getattr, vars, __dict__) is
            # obfuscation that reaches code execution: the exec / import / shell modules
            # PLUS the deserializer modules -- getattr(pickle, 'loads')(x) and
            # vars(pickle)['loads'](x) are just pickle.loads(x) with the name hidden.
            return (
                _DYNAMIC_ATTR_TARGETS
                | self.os_aliases
                | self.subprocess_aliases
                | self.importlib_aliases
                | self.sys_aliases
                | self.builtins_aliases
                | set(self.deserialize_module_aliases)
            )

        def _stores_hidden_sink(self, value):
            # A dynamic-exec builtin (exec / eval / compile / __import__), bare / builtins-attr /
            # scope alias, being stashed for later obfuscated invocation.
            if isinstance(value, ast.Name):
                if value.id in _DYNAMIC_EXEC_BUILTINS or value.id == "__import__":
                    return True
                if value.id in self.exec_from_aliases:
                    return True
                if _analyzer_on and _scope_idx.resolve(value.id, value, "execb"):
                    return True
            if (
                isinstance(value, ast.Attribute)
                and value.attr in (_DYNAMIC_EXEC_BUILTINS | {"__import__"})
                and _ast_name_matches(value.value, self.builtins_aliases)
            ):
                return True
            return False

        def _code_store_rhs_vetted(self, rhs):
            # A value assigned to fn.__code__ that we can prove is safe to execute. An in-source
            # function's code (g.__code__ / meth.__func__) is analyzed normally, and a compile()
            # result -- direct call or a c = compile(...) alias -- has its SOURCE analyzed at the
            # compile site (a malicious / opaque source is flagged there). Everything else (a
            # producer code object from codeop / a loader's get_code() / marshal, or an opaque
            # name) is unvetted and fails closed.
            if isinstance(rhs, ast.Attribute) and rhs.attr in ("__code__", "__func__"):
                return True
            if isinstance(rhs, ast.Call):
                rf = rhs.func
                if isinstance(rf, ast.Name) and (
                    rf.id == "compile" or self.exec_from_aliases.get(rf.id) == "compile"
                ):
                    return True
                if (
                    isinstance(rf, ast.Attribute)
                    and rf.attr == "compile"
                    and _ast_name_matches(rf.value, self.builtins_aliases)
                ):
                    return True
            if _analyzer_on and isinstance(rhs, ast.Name):
                if _scope_idx.resolve(rhs.id, rhs, "compiledany"):
                    return True
                if _scope_idx.resolve(rhs.id, rhs, "execb") == "compile":
                    return True
            return False

        def _is_environ_receiver(self, _v):
            # os.environ / os.environb (or a bare `environ` / `environb`, a `from os import environ
            # as e` alias, or a single-assignment `e = os.environ`). environb is the SAME inherited
            # process environment via byte keys/values.
            if (
                isinstance(_v, ast.Attribute)
                and _v.attr in ("environ", "environb")
                and isinstance(_v.value, ast.Name)
                and _v.value.id in self.os_aliases
            ):
                return True
            if isinstance(_v, ast.Name):
                if _v.id in self.environ_aliases:
                    return True
                if _analyzer_on:
                    _rhs = _scope_idx.resolve(_v.id, _v, "rhsnode")
                    if (
                        isinstance(_rhs, ast.Attribute)
                        and _rhs.attr in ("environ", "environb")
                        and isinstance(_rhs.value, ast.Name)
                        and _rhs.value.id in self.os_aliases
                    ):
                        return True
            return False

        def _environ_subscript_key(self, target):
            # The literal key of an os.environ[...] / os.environb[...] (or a bare environ[...] /
            # environb[...]) subscript assignment target; None otherwise. A bytes key is decoded.
            if not isinstance(target, ast.Subscript):
                return None
            if not self._is_environ_receiver(target.value):
                return None
            return _extract_env_scalar(target.slice)

        def _env_mutation_escape(self, key, value_node):
            # A short reason when setting env var ``key`` to ``value_node`` is a child-escape
            # prelude (mirrors the subprocess env={...} mapping analysis), else None. The mutated
            # process environment is inherited by a later unguarded child.
            _vs = _extract_env_scalar(value_node)
            if key == "PATH":
                if isinstance(_vs, str):
                    if _path_value_is_unsafe(_vs):
                        return "PATH set to a relative / cwd entry (a bare argv resolves to a workdir exec)"
                    return None
                # A non-literal PATH value that provably prepends / embeds a relative / cwd entry
                # ('.:' + os.environ['PATH'], f'.:{x}'); a dynamic ABSOLUTE extension stays allowed.
                if _dynamic_path_value_unsafe(value_node, _const_env):
                    return "PATH prepends a relative / cwd entry (dynamic value)"
                return None
            if key in ("BASH_ENV", "ENV"):
                return None if _vs == "" else "a shell startup file a child shell sources"
            if isinstance(key, str) and key.startswith("GIT_CONFIG"):
                return "overrides git config / drops the sandbox hook suppression"
            if key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
                if isinstance(_vs, str) and _arg_escapes_workdir(_vs):
                    return "points git's repo / tree outside the workdir"
                return None
            return None

        def _env_removal_escape(self, key):
            # A short reason when REMOVING inherited env var ``key`` (del / pop / unsetenv) is a
            # child-escape prelude, else None: dropping a GIT_CONFIG* var re-enables the
            # sandbox-suppressed git hooks (core.hooksPath) in a later unguarded git child.
            if isinstance(key, str) and key.startswith("GIT_CONFIG"):
                return "removes the sandbox git hook suppression"
            return None

        def visit_Assign(self, node):
            # e = os.environ (or os.environb) binds a NEW name to the same inherited-env mapping, so
            # a later e['BASH_ENV'] = ... escape reads as a plain-name subscript. Record the alias
            # (source order puts this assignment before the mutation) so _is_environ_receiver treats
            # `e` as the environ mapping. A bare `environ` RHS (from-import alias) is covered too.
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                _rv = node.value
                if (
                    isinstance(_rv, ast.Attribute)
                    and _rv.attr in ("environ", "environb")
                    and isinstance(_rv.value, ast.Name)
                    and _rv.value.id in self.os_aliases
                ) or (isinstance(_rv, ast.Name) and _rv.id in self.environ_aliases):
                    self.environ_aliases.add(node.targets[0].id)
            # os.environ['PATH'] = '.' (or BASH_ENV / ENV / GIT_CONFIG* / GIT_DIR) mutates the
            # INHERITED environment a later unguarded subprocess child reads, the same escape as
            # passing env={...} to the child: a bare-argv workdir exec via PATH='.', a sourced
            # BASH_ENV script, or a dropped GIT_CONFIG hook suppression. Flag the mutation itself.
            for _t in node.targets:
                _envkey = self._environ_subscript_key(_t)
                if _envkey is not None:
                    _reason = self._env_mutation_escape(_envkey, node.value)
                    if _reason is not None:
                        shell_escapes.append(
                            {
                                "type": "shell_escape",
                                "line": getattr(node, "lineno", -1),
                                "description": f"os.environ[{_envkey!r}] mutation: {_reason}",
                            }
                        )
            # d['e'] = exec / lst[0] = eval -- storing a dynamic-exec builtin into a CONTAINER
            # element (not a plain name, which the alias tracker already follows) hides the sink
            # from the name / attribute call checks, and the later d['e'](payload) then runs
            # unreviewed. There is no benign reason to stash exec / eval / compile / __import__ in
            # a container slot, so flag the store itself.
            if any(isinstance(t, ast.Subscript) for t in node.targets) and self._stores_hidden_sink(
                node.value
            ):
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": (
                            "a dynamic-exec builtin stored into a container element "
                            "(obfuscated exec alias)"
                        ),
                    }
                )
            # fn.__code__ = <code object> rebinds a function's body, so fn() then runs that code
            # WITHOUT eval / exec. A code object from an unvetted producer (codeop.compile_command,
            # a loader's get_code(), marshal) runs source the recursive analysis never saw, the
            # __code__ twin of the FunctionType gadget. Flag a __code__ store whose RHS is not a
            # vetted in-source / compile()-analyzed code object.
            if any(
                isinstance(t, ast.Attribute) and t.attr == "__code__" for t in node.targets
            ) and not self._code_store_rhs_vetted(node.value):
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": (
                            "an unvetted code object assigned to __code__ "
                            "(executes via the function without eval/exec)"
                        ),
                    }
                )
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            # os.environ['PATH'] += ':.' (or BASH_ENV / GIT_CONFIG*) mutates the inherited env in
            # place; model the result as (old value + appended) and run the same policy as a plain
            # assignment, so a relative / cwd PATH entry appended to $PATH is caught while a dynamic
            # ABSOLUTE extension (+= ':/usr/local/bin') stays allowed.
            _envkey = self._environ_subscript_key(node.target)
            if _envkey is not None:
                _synth = ast.BinOp(left = node.target, op = node.op, right = node.value)
                _reason = self._env_mutation_escape(_envkey, _synth)
                if _reason is not None:
                    shell_escapes.append(
                        {
                            "type": "shell_escape",
                            "line": getattr(node, "lineno", -1),
                            "description": f"os.environ[{_envkey!r}] augmented mutation: {_reason}",
                        }
                    )
            self.generic_visit(node)

        def visit_Delete(self, node):
            # del os.environ['GIT_CONFIG_COUNT'] removes an inherited env var without an assignment;
            # dropping a GIT_CONFIG* var re-enables the sandbox-suppressed git hooks in a later
            # unguarded git child.
            for _t in node.targets:
                _envkey = self._environ_subscript_key(_t)
                if _envkey is not None:
                    _reason = self._env_removal_escape(_envkey)
                    if _reason is not None:
                        shell_escapes.append(
                            {
                                "type": "shell_escape",
                                "line": getattr(node, "lineno", -1),
                                "description": f"del os.environ[{_envkey!r}]: {_reason}",
                            }
                        )
            self.generic_visit(node)

        def visit_Call(self, node):
            # operator.methodcaller('system', 'rm -rf /')(os) applies a deferred method to a
            # module receiver; rewrite it to the direct os.system('rm -rf /') call and analyze
            # that instead so the hidden shell/exec sink is not missed.
            _mc_rewrite = self._methodcaller_module_call(node)
            if _mc_rewrite is not None:
                self.visit_Call(_mc_rewrite)
                return
            # asyncio.create_subprocess_shell / create_subprocess_exec start the same unguarded
            # child as subprocess.run/Popen; rewrite to the subprocess form and analyze that.
            _aio_rewrite = self._asyncio_subprocess_rewrite(node)
            if _aio_rewrite is not None:
                self.visit_Call(_aio_rewrite)
                return
            # os.environ.update({'PATH': '.:...'}) / .update(PATH='...') / .setdefault('PATH', ...)
            # (and the os.environb byte forms) mutate the inherited environment WITHOUT a subscript
            # assignment, the same child escape as os.environ['PATH'] = ...; run each (key, value)
            # pair through the mutation policy.
            _mf = node.func
            if isinstance(_mf, ast.Attribute) and _mf.attr in ("update", "setdefault"):
                if self._is_environ_receiver(_mf.value):
                    _pairs = []
                    if _mf.attr == "setdefault" and len(node.args) >= 2:
                        _sk = _extract_env_scalar(node.args[0])
                        if _sk is not None:
                            _pairs.append((_sk, node.args[1]))
                    elif _mf.attr == "update":
                        if node.args and isinstance(node.args[0], ast.Dict):
                            for _kn, _vn in zip(node.args[0].keys, node.args[0].values):
                                if _kn is not None:
                                    _dk = _extract_env_scalar(_kn)
                                    if _dk is not None:
                                        _pairs.append((_dk, _vn))
                        for _kw in node.keywords:
                            if _kw.arg is not None:
                                _pairs.append((_kw.arg, _kw.value))
                    for _pk, _pvn in _pairs:
                        _preason = self._env_mutation_escape(_pk, _pvn)
                        if _preason is not None:
                            shell_escapes.append(
                                {
                                    "type": "shell_escape",
                                    "line": getattr(node, "lineno", -1),
                                    "description": (
                                        f"os.environ.{_mf.attr}({_pk!r}) mutation: {_preason}"
                                    ),
                                }
                            )
            # os.environ.pop('GIT_CONFIG_COUNT') / .clear() / .popitem() REMOVE an inherited env var
            # without an assignment or del; dropping a GIT_CONFIG* var (or clearing the whole env)
            # re-enables the sandbox-suppressed git hooks in a later unguarded git child.
            if isinstance(_mf, ast.Attribute) and self._is_environ_receiver(_mf.value):
                if _mf.attr in ("clear", "popitem"):
                    shell_escapes.append(
                        {
                            "type": "shell_escape",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                f"os.environ.{_mf.attr}() drops inherited env "
                                "(incl. the git hook suppression)"
                            ),
                        }
                    )
                elif _mf.attr == "pop" and node.args:
                    _rk = _extract_env_scalar(node.args[0])
                    _rreason = self._env_removal_escape(_rk)
                    if _rreason is not None:
                        shell_escapes.append(
                            {
                                "type": "shell_escape",
                                "line": getattr(node, "lineno", -1),
                                "description": f"os.environ.pop({_rk!r}): {_rreason}",
                            }
                        )
            # os.unsetenv('GIT_CONFIG_COUNT') is the C-level twin of os.putenv that removes an
            # inherited var, dropping the git hook suppression the same way as del os.environ[...].
            if (
                isinstance(_mf, ast.Attribute)
                and _mf.attr == "unsetenv"
                and isinstance(_mf.value, ast.Name)
                and _mf.value.id in self.os_aliases
                and node.args
            ):
                _xk = _extract_env_scalar(node.args[0])
                _xreason = self._env_removal_escape(_xk)
                if _xreason is not None:
                    shell_escapes.append(
                        {
                            "type": "shell_escape",
                            "line": getattr(node, "lineno", -1),
                            "description": f"os.unsetenv({_xk!r}): {_xreason}",
                        }
                    )
            # os.putenv(key, value) sets an inherited env var through the C-level setter (NOT via
            # os.environ), so the subscript / update checks miss it; a later child still inherits it
            # (os.putenv('BASH_ENV', 'evil.sh') then subprocess.run(['bash','-c',...])). Run the
            # (key, value) pair through the same mutation policy. Cover os.putenv (os alias) and a
            # bare `putenv` from `from os import putenv`.
            _is_putenv = (
                isinstance(_mf, ast.Attribute)
                and _mf.attr == "putenv"
                and isinstance(_mf.value, ast.Name)
                and _mf.value.id in self.os_aliases
            ) or (isinstance(_mf, ast.Name) and _mf.id in self.putenv_aliases)
            if _is_putenv and len(node.args) >= 2:
                _uk = _extract_env_scalar(node.args[0])
                if _uk is not None:
                    _ureason = self._env_mutation_escape(_uk, node.args[1])
                    if _ureason is not None:
                        shell_escapes.append(
                            {
                                "type": "shell_escape",
                                "line": getattr(node, "lineno", -1),
                                "description": f"os.putenv({_uk!r}) mutation: {_ureason}",
                            }
                        )
            if self._is_unbound_mro_gadget(node):
                # type.mro(io.FileIO) / type.__getattribute__(io.FileIO, '__mro__') /
                # getattr(io.FileIO, 'mro'): reaches the unguarded MRO without a .mro / .__mro__
                # attribute for visit_Attribute to see.
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": "unbound MRO access on a file class recovers an unguarded base (gadget)",
                    }
                )
            func = node.func
            # A trailing `.__call__` invokes the underlying callable through its bound
            # method: os.system.__call__(cmd), __import__.__call__('os'),
            # pickle.loads.__call__(blob). Strip it so the shell / import / deserializer /
            # attribute resolvers below see the real sink instead of a plain attribute.
            _ecf = func
            while isinstance(_ecf, ast.Attribute) and _ecf.attr == "__call__":
                _ecf = _ecf.value
            func_name = None
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in self.signal_aliases:
                        func_name = f"signal.{func.attr}"
            elif isinstance(func, ast.Name):
                if func.id in ("signal", "setitimer", "alarm", "pthread_sigmask"):
                    func_name = func.id

            if func_name:
                if func_name in ("signal.signal", "signal"):
                    if len(node.args) >= 1:
                        if _ast_name_matches(node.args[0], ("SIGALRM", "signal.SIGALRM")):
                            signal_tampering.append(
                                {
                                    "type": "signal_handler_override",
                                    "line": node.lineno,
                                    "description": "Overrides SIGALRM handler",
                                }
                            )
                elif func_name in ("signal.setitimer", "setitimer"):
                    if len(node.args) >= 1:
                        if _ast_name_matches(node.args[0], ("ITIMER_REAL", "signal.ITIMER_REAL")):
                            signal_tampering.append(
                                {
                                    "type": "timer_manipulation",
                                    "line": node.lineno,
                                    "description": "Manipulates ITIMER_REAL timer",
                                }
                            )
                elif func_name in ("signal.alarm", "alarm"):
                    signal_tampering.append(
                        {
                            "type": "alarm_manipulation",
                            "line": node.lineno,
                            "description": "Manipulates alarm timer",
                        }
                    )
                elif func_name in ("signal.pthread_sigmask", "pthread_sigmask"):
                    signal_tampering.append(
                        {
                            "type": "signal_mask",
                            "line": node.lineno,
                            "description": "Modifies signal mask (may block SIGALRM)",
                        }
                    )

            # --- Shell escape detection ---
            # Resolve the FQ function name for os.*/subprocess.* (via the __call__-stripped
            # effective callee so os.system.__call__(cmd) resolves to os.system).
            shell_func = None
            if isinstance(_ecf, ast.Attribute):
                if isinstance(_ecf.value, ast.Name):
                    if _ecf.value.id in self.os_aliases:
                        shell_func = f"os.{_ecf.attr}"
                    elif _ecf.value.id in self.subprocess_aliases:
                        shell_func = f"subprocess.{_ecf.attr}"
                    # class-body alias reached as ClassName.attr (class C: f = os.system;
                    # C.f('rm -rf /')), or an instance-attribute alias (obj.s = os.system;
                    # obj.s('rm -rf /')).
                    elif _analyzer_on:
                        shell_func = _scope_idx.resolve_class_attr(
                            _ecf.value.id, _ecf.attr, "shell"
                        ) or _scope_idx.resolve_instance_attr(_ecf.value.id, _ecf.attr, "shell")
                elif (
                    _analyzer_on
                    and isinstance(_ecf.value, ast.Call)
                    and isinstance(_ecf.value.func, ast.Name)
                ):
                    # An instance built inline, ClassName().attr: instance lookup still returns
                    # the class-body sink alias. class C: s = os.system; C().s('rm -rf /').
                    shell_func = _scope_idx.resolve_class_attr(
                        _ecf.value.func.id, _ecf.attr, "shell"
                    )
                elif isinstance(_ecf.value, ast.Attribute) and _ecf.value.attr in ("os", "posix"):
                    # A stdlib module that re-exports os as an attribute (pathlib.os.system,
                    # tempfile.os.system, subprocess.os.system): the `.os` attribute IS the os
                    # module, so treat the chain as an os.* sink.
                    shell_func = f"os.{_ecf.attr}"
                elif isinstance(_ecf.value, ast.Attribute) and _ecf.value.attr == "subprocess":
                    # ...and *.subprocess.run (a module re-exporting subprocess).
                    shell_func = f"subprocess.{_ecf.attr}"
                if shell_func is None:
                    # getattr(<mod>, 'os').system(...) / vars(<mod>)['os'].system(...) /
                    # <mod>.__dict__['subprocess'].run(...): a re-export fetched by NAME -- the
                    # call / subscript twin of the <mod>.os.system attribute form, and reachable
                    # off a call-returned module (getattr(__import__('pathlib'), 'os')). Map the
                    # fetched os / posix / subprocess to the sink module so the chain is a sink.
                    _rx = _reexport_dangerous_module_name(_ecf.value)
                    if _rx:
                        shell_func = f"{_rx}.{_ecf.attr}"
            elif isinstance(_ecf, ast.Name):
                # from-import aliases: from os import system; system(...)
                shell_func = self.shell_exec_aliases.get(_ecf.id)
                # Stage 4: single-assignment alias `s = os.system; s('rm -rf /')`,
                # resolved in the call's own scope (per-function).
                if shell_func is None and _analyzer_on:
                    shell_func = _scope_idx.resolve(_ecf.id, _ecf, "shell")
            elif _analyzer_on and isinstance(_ecf, ast.Subscript):
                # Stage 4: inline literal container index `[os.system][0](...)`.
                shell_func = self._resolve_container_sink(_ecf)

            if shell_func and shell_func in _SHELL_EXEC_FUNCS:
                # Expand **kwargs dicts to inspect their keys.
                expanded_kwargs: dict[str, ast.AST] = {}
                has_opaque_kwargs = False
                for kw in node.keywords:
                    if kw.arg is not None:
                        expanded_kwargs[kw.arg] = kw.value
                    elif isinstance(kw.value, ast.Dict):
                        for k, v in zip(kw.value.keys, kw.value.values):
                            key = _extract_string_from_node(k) if k else None
                            if key is not None:
                                expanded_kwargs[key] = v
                    else:
                        has_opaque_kwargs = True

                cmd_kw_values = [v for k, v in expanded_kwargs.items() if k in _CMD_KWARGS]
                all_call_args = list(node.args) + cmd_kw_values
                # A non-literal-False shell= is treated as potentially True (conservative), so a
                # sequence's first element is the shell -c command string, not an argv vector.
                _shell_node = expanded_kwargs.get("shell")
                _shell_maybe_true = not (
                    _shell_node is None
                    or (isinstance(_shell_node, ast.Constant) and _shell_node.value is False)
                )
                # A literal cwd= that escapes the workdir sets the child's real working directory,
                # so a relative write operand (subprocess.run(['git','init','repo'], cwd='/tmp'))
                # lands OUTSIDE the session. Model it as a synthetic `env -C <cwd>` wrapper so the
                # git cwd backscan resolves the escape; a workdir-relative / in-tree cwd adds no
                # prefix and stays allowed.
                _cwd_node = expanded_kwargs.get("cwd")
                _cwd_str = _extract_string_from_node(_cwd_node) if _cwd_node is not None else None
                _cwd_prefix = ""
                if _cwd_str is not None and _arg_escapes_workdir(_cwd_str):
                    _cwd_prefix = "env -C " + shlex.quote(_cwd_str) + " "
                blocked_in_args = _check_args_for_blocked(
                    all_call_args, _shell_maybe_true, _cwd_prefix
                )

                # The argv sequence can be given positionally (run(['bash', ...])) or through the
                # public args= keyword (run(args=['bash', ...])), which this analyzer already
                # collects in _CMD_KWARGS. Resolve either form so the executable= and shell-child
                # checks below are not bypassed by moving the command into args=.
                _argv0_node = node.args[0] if node.args else expanded_kwargs.get("args")

                # subprocess(..., executable=PROG) makes PROG the real program while the argv
                # TAIL still supplies its flags/args, so scanning executable and argv separately
                # misses run(['x', '-i', 's/a/b/', '/f'], executable='/usr/bin/sed') (child runs
                # sed -i). Reconstruct PROG + argv[1:] and scan the effective command line.
                _exe_node = expanded_kwargs.get("executable")
                _exe = _extract_string_from_node(_exe_node) if _exe_node is not None else None
                if (
                    _exe is not None
                    and not _shell_maybe_true
                    and isinstance(_argv0_node, (ast.List, ast.Tuple))
                ):
                    _tail = [_extract_string_from_node(e) for e in _argv0_node.elts[1:]]
                    _combined = [_exe] + _tail
                    if all(_c is not None for _c in _combined):
                        blocked_in_args = blocked_in_args | _find_blocked_commands(
                            " ".join(shlex.quote(_c) for _c in _combined)
                        )

                # A shell startup variable (BASH_ENV / ENV) in the env= mapping names a script
                # bash / sh SOURCES before the -c payload runs, executing unscanned code
                # (subprocess.run(['bash','-c','echo OK'], env={'BASH_ENV':'env.sh'})). Flag a
                # non-empty value; an empty string is inert. Cover a literal dict, a dict(...)
                # call, and -- for a shell child -- a non-literal mapping we cannot prove free
                # of BASH_ENV / ENV (fail closed). Whether the child is a shell: shell=True, or
                # the argv command word resolves to bash / sh.
                _env_node = expanded_kwargs.get("env")
                # A single-assignment env mapping (e = {'PATH': '.'}; run(['evil'], env=e)) reaches
                # here as a Name; resolve it to its literal dict / dict() so the BASH_ENV and
                # unsafe-PATH checks below still apply instead of silently passing.
                if isinstance(_env_node, ast.Name) and _analyzer_on:
                    _renv = _scope_idx.resolve(_env_node.id, _env_node, "rhsnode")
                    if isinstance(_renv, (ast.Dict, ast.Call)):
                        _env_node = _renv
                if _env_node is not None:
                    _is_shell_child = _shell_maybe_true
                    _is_git_child = False
                    if isinstance(_argv0_node, (ast.List, ast.Tuple)):
                        _elts0 = [_extract_string_from_node(_e) for _e in _argv0_node.elts]
                        _ci0 = _blocked_in_argv(_elts0)[1]
                        if _ci0 is not None and _ci0 < len(_elts0) and _elts0[_ci0]:
                            _cw0 = os.path.basename(_elts0[_ci0]).lower()
                            if not _is_shell_child:
                                _is_shell_child = _cw0 in _SHELL_BINARIES
                            _is_git_child = _cw0 == "git"
                    _is_env_mapping = isinstance(_env_node, ast.Dict) or (
                        isinstance(_env_node, ast.Call)
                        and isinstance(_env_node.func, ast.Name)
                        and _env_node.func.id == "dict"
                    )
                    if _is_env_mapping:
                        # A literal dict, a dict(...) call, and any nested **{...} splat are
                        # flattened together so the BASH_ENV / PATH / GIT_* checks (and the git
                        # hook-suppression check) apply uniformly; a computed / non-literal-**
                        # key marks the mapping opaque (fail closed for a shell child).
                        _epairs, _opaque_key = _env_mapping_pairs(_env_node)
                        for _ekey, _ev in _epairs:
                            # Const-fold / decode the value so a const-var, a concatenation, or a
                            # POSIX bytes value (env={'PATH': P}, {'PATH': '.:' + x}, {'PATH':
                            # b'.:'}) is analyzed, not just an inline str constant.
                            _evstr = _extract_env_scalar(_ev)
                            if _ekey in ("BASH_ENV", "ENV") and _evstr != "":
                                blocked_in_args = blocked_in_args | {"shell-startup-env:" + _ekey}
                            elif _ekey == "PATH":
                                # env={'PATH': '.'} lets a bare argv[0] resolve to a workdir exec.
                                # A folded literal is checked directly; a non-literal value that
                                # provably contributes a relative / cwd entry ('.:' + $PATH) fails
                                # closed, while a dynamic ABSOLUTE extension stays allowed.
                                if isinstance(_evstr, str):
                                    if _path_value_is_unsafe(_evstr):
                                        blocked_in_args = blocked_in_args | {"unsafe-path-assign"}
                                elif _dynamic_path_value_unsafe(_ev, _const_env):
                                    blocked_in_args = blocked_in_args | {"unsafe-path-assign"}
                            elif (
                                _ekey in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
                                and _is_git_child
                                and isinstance(_evstr, str)
                                and _arg_escapes_workdir(_evstr)
                            ):
                                # env={'GIT_DIR': '/tmp/x'} points git's repo outside the workdir.
                                blocked_in_args = blocked_in_args | {"git-write-outside"}
                            elif (
                                _ekey is not None
                                and _ekey.startswith("GIT_CONFIG")
                                and _is_git_child
                            ):
                                # env={'GIT_CONFIG_COUNT': '0'} drops the sandbox hook suppression.
                                blocked_in_args = blocked_in_args | {"git-config-env-override"}
                        if _opaque_key and _is_shell_child:
                            blocked_in_args = blocked_in_args | {"shell-startup-env:opaque"}
                        # A git child whose replaced env drops the sandbox's GIT_CONFIG_COUNT hook
                        # suppression (env={} / dict(PATH=...) / any mapping without a literal
                        # GIT_CONFIG_COUNT) re-enables a planted .git/hooks/* in the unguarded
                        # child. An OPAQUE mapping (env={**d}) cannot PROVE the suppression is
                        # present, so fail closed too.
                        if _is_git_child and (
                            _opaque_key or not any(_k == "GIT_CONFIG_COUNT" for _k, _ in _epairs)
                        ):
                            blocked_in_args = blocked_in_args | {"git-config-env-override"}
                    elif _is_shell_child or _is_git_child:
                        # A non-literal env mapping (env=e, env=f(), a comprehension) cannot be
                        # proven free of BASH_ENV / ENV (shell child) nor proven to carry the
                        # GIT_CONFIG_COUNT hook suppression (git child), so fail closed.
                        if _is_shell_child:
                            blocked_in_args = blocked_in_args | {"shell-startup-env:non-literal"}
                        if _is_git_child:
                            blocked_in_args = blocked_in_args | {"git-config-env-override"}

                # os.execl(path, a0, a1, ...) / os.execv(path, [a0, ...]) / os.spawnl(mode,
                # path, a0, ...) / os.posix_spawn(path, argv, env) spread the child's argv across
                # separate positional args (or a single list), so scanning each string alone
                # misses a mutating tail like `sed -i ...`. posix_spawn(p) executes `path` while
                # argv[0] is only cosmetic, so a literal-env form (env=() / a byte list) otherwise
                # slips the non-literal-env fallback. Reconstruct the executed command line
                # (program path + argv[1:], since argv[0] is cosmetic) and run the full scanner.
                if (
                    shell_func.startswith("os.exec")
                    or shell_func.startswith("os.spawn")
                    or shell_func.startswith("os.posix_spawn")
                ):
                    _name = shell_func.split(".", 1)[1]
                    if _name.startswith("spawn"):  # spawn*(mode, path, ...)
                        _path_node = node.args[1] if len(node.args) > 1 else None
                        _tail = node.args[2:]
                    else:  # exec*(path, ...) / posix_spawn(path, argv, env)
                        _path_node = node.args[0] if node.args else None
                        _tail = node.args[1:]
                    _is_v = "execv" in _name or "spawnv" in _name or _name.startswith("posix_spawn")
                    if _is_v:
                        _argv = (
                            [_extract_string_from_node(e) for e in _tail[0].elts]
                            if _tail and isinstance(_tail[0], (ast.List, ast.Tuple))
                            else []
                        )
                    else:
                        _argv = [_extract_string_from_node(a) for a in _tail]
                    _parts = [p for p in ([_extract_string_from_node(_path_node)] + _argv[1:]) if p]
                    if _parts:
                        blocked_in_args = blocked_in_args | _find_blocked_commands(
                            " ".join(shlex.quote(p) for p in _parts)
                        )

                if has_opaque_kwargs:
                    # Can't inspect dynamic **kwargs; flag as unsafe.
                    shell_escapes.append(
                        {
                            "type": "shell_escape_dynamic",
                            "line": node.lineno,
                            "description": (f"{shell_func}() called with dynamic **kwargs"),
                        }
                    )
                elif blocked_in_args:
                    shell_escapes.append(
                        {
                            "type": "shell_escape",
                            "line": node.lineno,
                            "description": (
                                f"{shell_func}() invokes blocked command(s): "
                                f"{', '.join(sorted(blocked_in_args))}"
                            ),
                        }
                    )
                else:
                    # Only flag dynamic args for funcs that interpret strings as
                    # shell commands, or when shell= might be on. Any non-literal-
                    # False shell= is treated as potentially True (conservative).
                    _STRING_SHELL_FUNCS = frozenset(
                        {
                            "os.system",
                            "os.popen",
                            "os.popen2",
                            "os.popen3",
                            "os.popen4",
                            "subprocess.getoutput",
                            "subprocess.getstatusoutput",
                        }
                    )
                    shell_node = expanded_kwargs.get("shell")
                    shell_safe = shell_node is None or (
                        isinstance(shell_node, ast.Constant) and shell_node.value is False
                    )
                    # Dynamic shell-exec args (chr/format/concat bypasses).
                    if (
                        shell_func in _STRING_SHELL_FUNCS
                        or shell_func in _SHELL_EXEC_FUNCS
                        or not shell_safe
                    ):

                        def _is_safe_literal(n):
                            if _extract_string_from_node(n) is not None:
                                return True
                            if isinstance(n, (ast.List, ast.Tuple)):
                                return all(_extract_string_from_node(e) is not None for e in n.elts)
                            return False

                        has_non_literal = any(not _is_safe_literal(a) for a in all_call_args)
                        if has_non_literal:
                            shell_escapes.append(
                                {
                                    "type": "shell_escape_dynamic",
                                    "line": node.lineno,
                                    "description": (
                                        f"{shell_func}() called with non-literal "
                                        f"shell command (potential shell escape)"
                                    ),
                                }
                            )

            # --- Dynamic execution / obfuscation primitives ---
            # eval / exec / compile (bare builtin, single-assignment alias, builtins
            # attribute / subscript, inline container, or an indirect callee expression --
            # a ternary / boolean fallback -- that evaluates to one of them).
            # timeit.timeit / .repeat / Timer(...) (attribute, from-import, or single-assignment
            # alias) COMPILE and EXECUTE their stmt (arg0 / kw 'stmt') and setup (arg1 / kw 'setup')
            # STRING args, so analyze those like exec payloads (a callable stmt carries no source).
            if _analyzer_on and (
                (
                    isinstance(func, ast.Attribute)
                    and func.attr in ("timeit", "repeat", "Timer")
                    and _ast_name_matches(func.value, self.timeit_aliases)
                )
                or (isinstance(func, ast.Name) and func.id in self.timeit_func_aliases)
                or self._rhs_module_attr(func, ("timeit", "repeat", "Timer"), self.timeit_aliases)
            ):
                _t_stmt = node.args[0] if node.args else None
                _t_setup = node.args[1] if len(node.args) > 1 else None
                for _kw in node.keywords:
                    if _kw.arg == "stmt":
                        _t_stmt = _kw.value
                    elif _kw.arg == "setup":
                        _t_setup = _kw.value
                _analyze_timeit_code_arg(node, _t_stmt, "stmt")
                _analyze_timeit_code_arg(node, _t_setup, "setup")

            exec_func_id = self._resolve_exec_callee(func)

            if exec_func_id is not None:
                if _analyzer_on:
                    # Stage 2: recover + recurse the payload instead of a blanket ban,
                    # so eval("2+2") passes while obfuscated escapes still block.
                    _analyze_exec_call(node, exec_func_id)
                else:
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": f"dynamic code execution via {exec_func_id}()",
                        }
                    )
            else:
                dynamic_desc = None
                # A dangerous sink passed as a first-class VALUE (not called here) runs its
                # payloads through a higher-order applier the recursive analyzer never sees:
                # list(map(eval, ["..."])), functools.reduce(exec, ...),
                # list(map(os.system, ['rm -rf /'])), functools.partial(subprocess.getoutput,
                # 'wget ...')(), list(map(pickle.loads, [blob])). Flag any bare reference to a
                # dynamic-exec / shell / import / deserializer sink appearing as a call
                # argument, unpacking a literal *[...] / *(...) starred arg too.
                _cand_args = []
                for _a in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(_a, ast.Starred) and isinstance(_a.value, (ast.List, ast.Tuple)):
                        _cand_args.extend(_a.value.elts)
                    elif isinstance(_a, ast.Starred):
                        _cand_args.append(_a.value)
                    else:
                        _cand_args.append(_a)
                _indirect_sink = None
                for _t in _cand_args:
                    _indirect_sink = self._sink_ref_desc(_t)
                    if _indirect_sink is not None:
                        break
                if _indirect_sink is not None:
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                f"{_indirect_sink} passed as a value to a higher-order call "
                                "(indirect execution of an un-analyzed payload)"
                            ),
                        }
                    )
                # An attribute-access call whose (receiver, attr-name) pair is the same
                # obfuscation as getattr(): the builtin getattr/setattr, or the dunder
                # forms object.__getattribute__(obj, 'name') / type.__getattribute__(...)
                # / obj.__getattr__('name') that fetch an attribute without matching the
                # bare getattr name. Normalized here so the gadget + sensitive-module
                # checks below cover all of them.
                _attr_call = None
                _attr_dunder = False  # True when reached via __getattribute__/__getattr__
                if (
                    isinstance(func, ast.Name)
                    and func.id in ("getattr", "setattr")
                    and len(node.args) >= 2
                ):
                    _attr_call = (node.args[0], node.args[1])
                elif isinstance(func, ast.Attribute) and func.attr in (
                    "__getattribute__",
                    "__getattr__",
                ):
                    # Unbound form object.__getattribute__(obj, 'name') carries the receiver
                    # as arg0; the BOUND form obj.__getattribute__('name') carries it as the
                    # attribute's own value (builtins.open.__getattribute__('__closure__')).
                    _attr_dunder = True
                    if len(node.args) >= 2:
                        _attr_call = (node.args[0], node.args[1])
                    elif len(node.args) == 1:
                        _attr_call = (func.value, node.args[0])
                elif self._attrgetter_name(func) is not None and len(node.args) == 1:
                    # operator.attrgetter('name')(obj) evaluates to obj.name -- the same
                    # attribute-fetch obfuscation as getattr(obj, 'name'). Detect the
                    # attrgetter APPLICATION call itself (node.func is the attrgetter,
                    # node.args[0] is the object) so it is caught whether or not the result
                    # is immediately invoked: attrgetter('__closure__')(open)[0] and the
                    # chained attrgetter('system')(os)('rm -rf /') both normalize here.
                    _attr_call = (node.args[0], ast.Constant(value = self._attrgetter_name(func)))
                elif self._methodcaller_getattr_name(func) is not None and len(node.args) == 1:
                    # operator.methodcaller('__getattribute__', 'name')(obj) fetches obj.name,
                    # the same attribute obfuscation as attrgetter/getattr.
                    _attr_call = (
                        node.args[0],
                        ast.Constant(value = self._methodcaller_getattr_name(func)),
                    )
                is_dynamic_import = (
                    _ast_name_matches(_ecf, _DYNAMIC_IMPORT_FUNCS)
                    or (
                        isinstance(_ecf, ast.Name)
                        and (
                            _ecf.id in ("__import__", "import_module")
                            or _ecf.id in self.import_func_aliases
                        )
                    )
                    or (
                        isinstance(_ecf, ast.Attribute)
                        and _ecf.attr in ("import_module", "reload", "__import__")
                        and _ast_name_matches(_ecf.value, self.importlib_aliases)
                    )
                    or (
                        # builtins.__import__('os') / __builtins__.__import__(...), incl. a
                        # single-assignment alias (b = __builtins__; b.__import__('os')).
                        isinstance(_ecf, ast.Attribute)
                        and _ecf.attr == "__import__"
                        and self._is_builtins_ref(_ecf.value)
                    )
                    or (
                        # single-assignment `im = importlib.import_module` in scope.
                        _analyzer_on
                        and isinstance(_ecf, ast.Name)
                        and bool(_scope_idx.resolve(_ecf.id, _ecf, "impf"))
                    )
                )
                # Deserialization sinks reconstruct arbitrary objects/code from bytes.
                # Resolve aliased imports (from pickle import loads as l), module aliases
                # (import pickle as p; p.loads) and the file-based *.load variants -- not
                # just the exact pickle.loads name. Uses the __call__-stripped effective
                # callee so pickle.loads.__call__(blob) resolves like pickle.loads(blob).
                _deser_fq = None
                if isinstance(_ecf, ast.Attribute) and isinstance(_ecf.value, ast.Name):
                    _canon = self.deserialize_module_aliases.get(_ecf.value.id)
                    if _canon is not None:
                        _cand = f"{_canon}.{_ecf.attr}"
                        if _cand in _CODE_DESERIALIZE_SINKS:
                            _deser_fq = _cand
                    if _deser_fq is None and _analyzer_on:
                        # class-body alias reached as ClassName.attr (class C: l = pickle.loads).
                        _deser_fq = _scope_idx.resolve_class_attr(_ecf.value.id, _ecf.attr, "deser")
                elif isinstance(_ecf, ast.Name):
                    _deser_fq = self.deserialize_aliases.get(_ecf.id)
                    if _deser_fq is None and _analyzer_on:
                        # single-assignment `l = pickle.loads` in the call's scope.
                        _deser_fq = _scope_idx.resolve(_ecf.id, _ecf, "deser")
                elif _analyzer_on and isinstance(_ecf, ast.Subscript):
                    # ([pickle.loads][0])(payload) / {'k': pickle.loads}['k'](payload):
                    # an inline container hides the sink from the attribute / name checks.
                    _deser_fq = self._resolve_container_deser(_ecf)
                if _deser_fq is None:
                    _fq_func = _fq_attr_name(_ecf)
                    if _fq_func in _CODE_DESERIALIZE_SINKS:
                        _deser_fq = _fq_func
                if _deser_fq is None and isinstance(_ecf, ast.Attribute):
                    # yaml.load(...) / yaml.load_all(...) reconstruct arbitrary objects unless
                    # handed a safe loader. Resolve the (possibly module-aliased) yaml receiver
                    # and only flag when no explicit SafeLoader is passed, so yaml.load(data,
                    # Loader=yaml.SafeLoader) and yaml.safe_load(data) stay allowed.
                    if _ecf.attr in _YAML_LOAD_METHODS and isinstance(_ecf.value, ast.Name):
                        if self.deserialize_module_aliases.get(_ecf.value.id) == "yaml":
                            if not _yaml_call_has_safe_loader(node):
                                _deser_fq = "yaml." + _ecf.attr
                    # pickle.Unpickler(f).load() / dill.Unpickler(f).load(): the reduce payload
                    # runs on .load(); the sink-name check misses it because the callee is a
                    # method on an Unpickler instance, not a *.load module function.
                    if (
                        _deser_fq is None
                        and _ecf.attr in ("load", "load_all")
                        and self._is_unpickler_ctor(_ecf.value)
                    ):
                        _deser_fq = "pickle.Unpickler.load"
                    # torch.load(f, weights_only=False) / np.load(f, allow_pickle=True) /
                    # joblib.load(f) run a pickle reduce payload; flag only the unsafe forms so
                    # the safe defaults (torch.load(f), np.load(f)) stay allowed.
                    if (
                        _deser_fq is None
                        and _ecf.attr == "load"
                        and isinstance(_ecf.value, ast.Name)
                    ):
                        _pcanon = self.pickle_loader_module_aliases.get(_ecf.value.id)
                        if _pcanon is not None:
                            _plfq = f"{_pcanon}.load"
                            if _pickle_loader_is_unsafe(_plfq, node):
                                _deser_fq = _plfq
                if _deser_fq is None and isinstance(_ecf, ast.Name):
                    # from yaml import load; load(data): apply the same safe-loader check to the
                    # bare-name alias so importing the function directly is not a bypass.
                    _ym = self.yaml_load_aliases.get(_ecf.id)
                    if _ym is not None and not _yaml_call_has_safe_loader(node):
                        _deser_fq = "yaml." + _ym
                    # from joblib import load; load(f): the bare-name conditional pickle loader.
                    if _deser_fq is None:
                        _plf = self.pickle_loader_func_aliases.get(_ecf.id)
                        if _plf is not None and _pickle_loader_is_unsafe(_plf, node):
                            _deser_fq = _plf
                if _analyzer_on and _deser_fq is not None:
                    dynamic_desc = f"{_deser_fq}() deserializes an unverifiable code payload"
                elif is_dynamic_import:
                    # Computed module name (obfuscation) or a dangerous target is unsafe; a
                    # benign literal import (huggingface_hub, json, ...) passes. With the
                    # analyzer on, the name is constant-folded first so `__import__(
                    # "hugging"+"face_hub")` resolves to a real module instead of blocking.
                    if node.args:
                        if _analyzer_on:
                            folded = _const_fold(node.args[0], _const_env)
                            mod = folded if isinstance(folded, str) else None
                        else:
                            mod = _extract_string_from_node(node.args[0])
                    else:
                        mod = None
                    _mod_top = mod.split(".")[0] if mod else None
                    if (
                        mod is None
                        or _mod_top in _DANGEROUS_IMPORT_NAMES
                        or _mod_top in _DESERIALIZE_MODULES
                    ):
                        # Deserializer modules (pickle/marshal/...) are dangerous import
                        # targets too: __import__('pickle').loads(blob) executes a reduce
                        # payload even though a plain `import pickle` is benign.
                        dynamic_desc = "dynamic import of a computed or sensitive module name"
                elif _attr_call is not None and (
                    (
                        isinstance(_const_fold(_attr_call[1], _const_env), str)
                        and _const_fold(_attr_call[1], _const_env) in _GADGET_DUNDERS
                    )
                    or (
                        # A __getattribute__/__getattr__ dunder call whose attribute name is
                        # not constant-foldable hides a gadget dunder behind a runtime
                        # expression (open.__getattribute__(''.join(map(chr, ...)))), which
                        # recovers a guarded wrapper's __closure__/cell_contents. No program
                        # legitimately spells __getattribute__ with a computed name, so fail
                        # closed on the dynamic form.
                        _attr_dunder and not isinstance(_const_fold(_attr_call[1], _const_env), str)
                    )
                ):
                    # getattr(anything, '__globals__' / '__subclasses__' / ...) or the
                    # object.__getattribute__ equivalent reaches an introspection gadget
                    # with no ast.Attribute for visit_Attribute to catch. Direct
                    # x.__globals__ is already flagged for ANY receiver, so flag the
                    # dynamic-attr-name form regardless of receiver too. (Also closes the
                    # __closure__ recovery of a guarded wrapper's original callable.)
                    _gv = _const_fold(_attr_call[1], _const_env)
                    if isinstance(_gv, str):
                        dynamic_desc = (
                            f"dynamic attribute access of an introspection gadget dunder ({_gv})"
                        )
                    else:
                        dynamic_desc = (
                            "computed attribute name via __getattribute__/__getattr__ "
                            "(obfuscated introspection gadget)"
                        )
                elif (
                    isinstance(func, ast.Name)
                    and func.id == "vars"
                    and node.args
                    and _ast_name_matches(node.args[0], self._attr_obfuscation_targets())
                ):
                    # vars(os) / vars(__builtins__) returns the module __dict__, the same
                    # obfuscation as os.__dict__['system'] but without the attribute access.
                    dynamic_desc = "vars() on a sensitive module (dict obfuscation)"
                elif _attr_call is not None and _ast_name_matches(
                    _attr_call[0], self._attr_obfuscation_targets()
                ):
                    # Stage 2 refinement: a benign constant attr (getattr(os, "getpid"))
                    # is allowed; only a dynamic attr or a dangerous constant attr blocks.
                    # Covers getattr/setattr and object.__getattribute__(builtins, 'eval').
                    if _analyzer_on:
                        attr_val = _const_fold(_attr_call[1], _const_env)
                        if isinstance(attr_val, str):
                            if attr_val in _DANGEROUS_ATTR_NAMES or attr_val == "__dict__":
                                # getattr(__builtins__, '__dict__')['__import__'] exposes the
                                # module namespace the same way vars()/direct .__dict__ do, so
                                # a constant '__dict__' on a sensitive module is dangerous too.
                                dynamic_desc = (
                                    "dynamic attribute access on a sensitive module "
                                    "(attribute-name obfuscation)"
                                )
                        else:
                            dynamic_desc = (
                                "dynamic attribute access on a sensitive module "
                                "(attribute-name obfuscation)"
                            )
                    else:
                        dynamic_desc = (
                            "dynamic attribute access on a sensitive module "
                            "(attribute-name obfuscation)"
                        )
                elif (
                    # sys.modules.get('os') -- the .get() twin of sys.modules['os'], incl. a
                    # single-assignment alias (m = sys.modules; m.get('os')).
                    isinstance(func, ast.Attribute)
                    and func.attr == "get"
                    and self._is_sys_modules(func.value)
                    and node.args
                ):
                    # Constant-fold the key so sys.modules.get('o' + 's') is caught, not
                    # just a bare literal (the module is already loaded by the prelude).
                    _key = _const_fold(node.args[0], _const_env)
                    if isinstance(_key, str) and _key.split(".")[0] in _DANGEROUS_IMPORT_NAMES:
                        dynamic_desc = "sys.modules.get(...) access to a sensitive module"
                elif (
                    # sys.modules.pop('_io', None) / .clear() / .update(...) / .setdefault(...)
                    # mutate the loader table just like `del sys.modules[...]`: dropping a
                    # guarded module entry lets `import _io` / `import posix` reload a fresh,
                    # UNWRAPPED C module (the prelude patched only the old object), bypassing
                    # filesystem confinement. The subscript-Store/Del check misses method calls.
                    isinstance(func, ast.Attribute)
                    and func.attr
                    in (
                        "pop",
                        "popitem",
                        "clear",
                        "setdefault",
                        "update",
                        "__setitem__",
                        "__delitem__",
                    )
                    and self._is_sys_modules(func.value)
                ):
                    dynamic_desc = (
                        f"sys.modules.{func.attr}(...) mutates the loader table "
                        "(can drop a guarded module for reimport)"
                    )
                elif (
                    # The same loader-table mutation through an UNBOUND dict method:
                    # dict.pop(sys.modules, '_io') / type(sys.modules).__delitem__(sys.modules,
                    # ...). The receiver is `dict` / `type(sys.modules)`, not `sys.modules`
                    # itself, so the check above misses it; here sys.modules is the first arg.
                    isinstance(func, ast.Attribute)
                    and func.attr
                    in (
                        "pop",
                        "popitem",
                        "clear",
                        "setdefault",
                        "update",
                        "__setitem__",
                        "__delitem__",
                    )
                    and node.args
                    and self._is_sys_modules(node.args[0])
                    and (
                        (isinstance(func.value, ast.Name) and func.value.id == "dict")
                        or (
                            isinstance(func.value, ast.Call)
                            and isinstance(func.value.func, ast.Name)
                            and func.value.func.id == "type"
                        )
                        # sys.modules.__class__.pop(sys.modules, ...): the receiver is the dict
                        # TYPE reached via .__class__, not the bare `dict` name / type(...) call.
                        or (
                            isinstance(func.value, ast.Attribute)
                            and func.value.attr == "__class__"
                            and self._is_sys_modules(func.value.value)
                        )
                    )
                ):
                    dynamic_desc = (
                        f"unbound dict.{func.attr}(sys.modules, ...) mutates the loader table "
                        "(can drop a guarded module for reimport)"
                    )
                elif (
                    # sys.meta_path.pop(0) / .clear() / .remove(...) / .insert(...) / .append(...)
                    # / .extend(...) / .reverse() / .sort() removes or reorders the import finder
                    # chain, dropping the sandbox's workdir-module vetter so a planted workdir
                    # helper (import evil) loads without source review and runs an unguarded sink.
                    # No sandboxed compute legitimately mutates the import finder chain.
                    isinstance(func, ast.Attribute)
                    and func.attr
                    in (
                        "pop",
                        "clear",
                        "remove",
                        "insert",
                        "append",
                        "extend",
                        "reverse",
                        "sort",
                        "__setitem__",
                        "__delitem__",
                        "__iadd__",
                    )
                    and self._is_sys_meta_path(func.value)
                ):
                    dynamic_desc = (
                        f"sys.meta_path.{func.attr}(...) mutates the import finder chain "
                        "(can remove the sandbox workdir-module vetter)"
                    )
                elif (
                    # The same mutation via an UNBOUND list method: list.pop(sys.meta_path, 0) /
                    # list.insert(sys.meta_path, ...). The receiver is `list`, not sys.meta_path,
                    # so the bound-method check above misses it; here sys.meta_path is the first arg.
                    isinstance(func, ast.Attribute)
                    and func.attr
                    in (
                        "pop",
                        "clear",
                        "remove",
                        "insert",
                        "append",
                        "extend",
                        "reverse",
                        "sort",
                        "__setitem__",
                        "__delitem__",
                        "__iadd__",
                    )
                    and node.args
                    and self._is_sys_meta_path(node.args[0])
                    and (
                        (isinstance(func.value, ast.Name) and func.value.id == "list")
                        or (
                            isinstance(func.value, ast.Call)
                            and isinstance(func.value.func, ast.Name)
                            and func.value.func.id == "type"
                        )
                        # sys.meta_path.__class__.pop(sys.meta_path, 0): the receiver is the list
                        # TYPE reached via .__class__, not the bare `list` name / type(...) call.
                        or (
                            isinstance(func.value, ast.Attribute)
                            and func.value.attr == "__class__"
                            and self._is_sys_meta_path(func.value.value)
                        )
                    )
                ):
                    dynamic_desc = (
                        f"unbound list.{func.attr}(sys.meta_path, ...) mutates the import finder "
                        "chain (can remove the sandbox workdir-module vetter)"
                    )
                elif (
                    # globals().get('__builtins__') / locals().get(...) / vars().get(...)
                    # -- the .get() twin of the globals()['__builtins__'] subscript form.
                    isinstance(func, ast.Attribute)
                    and func.attr == "get"
                    and isinstance(func.value, ast.Call)
                    and isinstance(func.value.func, ast.Name)
                    and func.value.func.id in ("globals", "locals", "vars")
                    and not func.value.args
                    and node.args
                ):
                    _key = _const_fold(node.args[0], _const_env)
                    if isinstance(_key, str) and (
                        _key in ("__builtins__", "__builtin__")
                        or _key.split(".")[0] in _DANGEROUS_IMPORT_NAMES
                    ):
                        dynamic_desc = (
                            "namespace-dict .get() access to builtins / a sensitive module"
                        )
                elif (
                    # dict.__getitem__(globals(), '__builtins__') / dict.get(locals(), ...): the
                    # unbound dict-method twin of globals()['__builtins__'], pulling the builtins
                    # namespace (or a dangerous module) out of the namespace dict without a
                    # subscript node. The receiver is `dict`, not the namespace dict itself, so
                    # the subscript scan misses it; here the namespace dict is the first argument.
                    isinstance(func, ast.Attribute)
                    and func.attr in ("__getitem__", "get")
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "dict"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name)
                    and node.args[0].func.id in ("globals", "locals", "vars")
                    and not node.args[0].args
                ):
                    _key = _const_fold(node.args[1], _const_env)
                    if isinstance(_key, str) and (
                        _key in ("__builtins__", "__builtin__")
                        or _key.split(".")[0] in _DANGEROUS_IMPORT_NAMES
                    ):
                        dynamic_desc = (
                            "unbound dict access to builtins / a sensitive module namespace dict"
                        )
                elif (
                    (
                        # types.FunctionType(compile(src, ...), {})() runs a code object WITHOUT
                        # eval/exec, so a dynamic compile() payload reaches execution here even
                        # though compile() alone is allowed. Flag when a FunctionType call takes
                        # a compile()-derived code object as its first argument.
                        (
                            isinstance(func, ast.Attribute)
                            and func.attr == "FunctionType"
                            and _ast_name_matches(func.value, self.types_aliases)
                        )
                        or (isinstance(func, ast.Name) and func.id in self.functiontype_aliases)
                        or (
                            # The same constructor is reachable as type(lambda: None): the
                            # type of any function IS types.FunctionType, so
                            # type(lambda: None)(code, {})() executes a code object too.
                            isinstance(func, ast.Call)
                            and not func.keywords
                            and len(func.args) == 1
                            and isinstance(func.args[0], ast.Lambda)
                            and (
                                (isinstance(func.func, ast.Name) and func.func.id == "type")
                                or (
                                    isinstance(func.func, ast.Attribute)
                                    and func.func.attr == "type"
                                    and _ast_name_matches(func.func.value, self.builtins_aliases)
                                )
                            )
                        )
                    )
                    and node.args
                    and not self._functiontype_arg_is_vetted(node.args[0])
                ):
                    dynamic_desc = (
                        "types.FunctionType() executes an unvetted code object "
                        "(bypasses the eval/exec gate)"
                    )
                elif (
                    # runpy.run_path('evil.py') / runpy.run_module('evil') execute a
                    # file/module in the guarded interpreter WITHOUT the recursive source
                    # analysis exec/eval receive, so a sandboxed snippet can write a local
                    # evil.py and run it. Treat these as direct execution sinks. Covers the
                    # attribute form, a `from runpy import run_path` bare-name alias, and a
                    # single-assignment alias (r = runpy.run_path; r('evil.py')).
                    (
                        isinstance(func, ast.Attribute)
                        and func.attr in ("run_path", "run_module")
                        and _ast_name_matches(func.value, self.runpy_aliases)
                    )
                    or (isinstance(func, ast.Name) and func.id in self.runpy_func_aliases)
                    or self._rhs_module_attr(func, ("run_path", "run_module"), self.runpy_aliases)
                ):
                    if isinstance(func, ast.Attribute):
                        _rn = func.attr
                    elif func.id in self.runpy_func_aliases:
                        _rn = func.id
                    else:
                        _rn = self._rhs_module_attr(
                            func, ("run_path", "run_module"), self.runpy_aliases
                        )
                    dynamic_desc = f"runpy.{_rn}() executes a file/module without static analysis"
                elif (
                    # pty.spawn([...]) / pty.fork() run an unguarded child process (typically a
                    # shell) outside the sandbox, the same escape as subprocess / os.system.
                    # Covers the attribute form, a `from pty import spawn` bare-name alias, and a
                    # single-assignment alias (s = pty.spawn; s([...])).
                    (
                        isinstance(func, ast.Attribute)
                        and func.attr in ("spawn", "fork")
                        and _ast_name_matches(func.value, self.pty_aliases)
                    )
                    or (isinstance(func, ast.Name) and func.id in self.pty_func_aliases)
                    or self._rhs_module_attr(func, ("spawn", "fork"), self.pty_aliases)
                ):
                    if isinstance(func, ast.Attribute):
                        _pn = func.attr
                    elif func.id in self.pty_func_aliases:
                        _pn = func.id
                    else:
                        _pn = self._rhs_module_attr(func, ("spawn", "fork"), self.pty_aliases)
                    dynamic_desc = f"pty.{_pn}() spawns an unguarded child process"
                elif (
                    # inspect.getclosurevars(open).nonlocals['real'] recovers the original
                    # unguarded callable a guard wrapper closes over, without spelling
                    # __closure__ / cell_contents. Block the introspection primitive
                    # (attribute form plus a `from inspect import getclosurevars` alias).
                    (
                        isinstance(func, ast.Attribute)
                        and func.attr == "getclosurevars"
                        and _ast_name_matches(func.value, self.inspect_aliases)
                    )
                    or (isinstance(func, ast.Name) and func.id in self.getclosurevars_aliases)
                ):
                    dynamic_desc = "inspect.getclosurevars() recovers a guarded wrapper's closure"
                elif (
                    # gc.get_referents / get_referrers / get_objects walk the object graph to a
                    # guard wrapper's closure cell (the original unguarded open/os.* callable)
                    # without spelling __closure__ / cell_contents, so a recovered original can
                    # then write/read outside the workdir. Block the graph-traversal APIs.
                    (
                        isinstance(func, ast.Attribute)
                        and func.attr in ("get_referents", "get_referrers", "get_objects")
                        and _ast_name_matches(func.value, self.gc_aliases)
                    )
                    or (isinstance(func, ast.Name) and func.id in self.gc_walk_aliases)
                ):
                    _gn = func.attr if isinstance(func, ast.Attribute) else func.id
                    dynamic_desc = (
                        f"gc.{_gn}() walks the object graph to a guarded wrapper's closure"
                    )
                elif (
                    # cls.mro().__getitem__(1) / .pop(1) / cls.__mro__.__getitem__(1): the
                    # method-call twin of the subscripted-mro base extraction
                    # (visit_Subscript). Same gadget shape (io.FileIO.mro().pop(1) recovers
                    # the original FileIO base), so flag an element-extraction method on an
                    # mro()/__mro__ receiver.
                    isinstance(func, ast.Attribute)
                    and func.attr in ("__getitem__", "pop")
                    and (
                        (
                            isinstance(func.value, ast.Call)
                            and isinstance(func.value.func, ast.Attribute)
                            and func.value.func.attr == "mro"
                            and not func.value.args
                        )
                        or (isinstance(func.value, ast.Attribute) and func.value.attr == "__mro__")
                    )
                    and (
                        # pop() / pop(i) always extract an element; __getitem__ only when the
                        # index is a plain integer (not a slice object).
                        func.attr == "pop"
                        or (
                            len(node.args) == 1
                            and isinstance(_const_fold(node.args[0], _const_env), int)
                        )
                    )
                ):
                    dynamic_desc = f"mro().{func.attr}(...) extracts a base class (gadget)"
                elif isinstance(func, ast.Attribute) and func.attr in (
                    "runcode",
                    "runsource",
                    "load_module",
                    "exec_module",
                ):
                    # code.InteractiveInterpreter().runcode(c) / InteractiveConsole()
                    # .runsource(src) execute a code object / source string; an importlib file
                    # loader (SourceFileLoader(...).load_module() / spec.loader.exec_module(m))
                    # executes a local file. None run through the recursive analysis exec/eval
                    # receive, so an opaque payload (a written evil.py, a compile() result, or
                    # raw source) runs un-analyzed. These method names are unique to those
                    # interpreters / loaders, so flag the call regardless of receiver.
                    dynamic_desc = (
                        f"{func.attr}() executes code / a file without static analysis "
                        "(code interpreter / importlib file loader)"
                    )
                if dynamic_desc:
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": dynamic_desc,
                        }
                    )

            self.generic_visit(node)

        def visit_Attribute(self, node):
            # Introspection gadget dunders (``__subclasses__``, ``__globals__``, ...) are the
            # standard way to walk from a benign object to os/builtins, bypassing the name-based
            # checks. Flag the attribute access itself, then keep descending.
            if node.attr in _GADGET_DUNDERS:
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": f"introspection gadget attribute {node.attr}",
                    }
                )
            elif node.attr == "__dict__" and _ast_name_matches(
                node.value, self._attr_obfuscation_targets()
            ):
                # os.__dict__['system']('id') reaches the sink with no getattr call for
                # the name-based checks to see. __dict__ on ordinary objects stays allowed.
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": "__dict__ access on a sensitive module",
                    }
                )
            elif node.attr in ("__mro__", "mro") and self._is_fileclass_recovery_expr(node.value):
                # io.FileIO.mro() / io.FileIO.__mro__ / open.__class__.mro(): the guard
                # replaces io.FileIO with a confining subclass, but its MRO still exposes the
                # UNGUARDED C base. Plain iteration recovers it (for c in io.FileIO.mro(): c(
                # '/etc/x','w')) without ever indexing, so flag any whole-MRO access on a
                # file-class-recovery receiver. Benign int.mro() / cls.__mro__ do not match.
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": "MRO access on a file class recovers an unguarded base (gadget)",
                    }
                )
            elif (
                node.attr == "meta_path"
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and _ast_name_matches(node.value, self.sys_aliases)
            ):
                # Reassigning / deleting sys.meta_path (sys.meta_path = []) replaces the whole
                # import finder chain, dropping the sandbox's workdir-module vetter. Reading it
                # (Load) stays allowed; only Store / Del is a mutation.
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": (
                            "sys.meta_path reassignment can remove the sandbox workdir-module vetter"
                        ),
                    }
                )
            self.generic_visit(node)

        def _is_sqlite_module_ref(self, node):
            # sqlite3 / _sqlite3 (Name) or sqlite3.dbapi2 (Attribute): the modules that export the
            # guarded Connection subclass whose MRO still exposes the unguarded _sqlite3.Connection.
            if isinstance(node, ast.Name):
                return node.id in ("sqlite3", "_sqlite3")
            if isinstance(node, ast.Attribute):
                return (
                    node.attr == "dbapi2"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "sqlite3"
                )
            return False

        def _is_type_of_guarded_instance(self, expr):
            # ``type(<x>)`` where ``<x>`` constructs a guarded file / sqlite instance, so ``type(
            # <x>)`` IS the guarded subclass and iterating its MRO recovers the unguarded base:
            # type(io.FileIO('x')).mro(), type(sqlite3.connect(':memory:')).mro(). Only a single
            # positional construction arg is matched, so type(x) on an opaque value does not.
            if not (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Name)
                and expr.func.id == "type"
                and len(expr.args) == 1
                and not expr.keywords
            ):
                return False
            arg = expr.args[0]
            if not isinstance(arg, ast.Call):
                return False
            f = arg.func
            if isinstance(f, ast.Attribute) and f.attr == "FileIO":
                return True  # io.FileIO(...) / _io.FileIO(...)
            if (
                isinstance(f, ast.Attribute)
                and f.attr in ("connect", "Connection")
                and self._is_sqlite_module_ref(f.value)
            ):
                return True  # sqlite3.connect(...) / sqlite3.Connection(...)
            return False

        def _is_fileclass_recovery_direct(self, expr):
            # io.FileIO / _io.FileIO (.FileIO) or a file object's type via .__class__
            # (open.__class__, f.__class__).
            if isinstance(expr, ast.Attribute) and expr.attr in ("FileIO", "__class__"):
                return True
            # The guarded sqlite3.Connection / _sqlite3.Connection / sqlite3.dbapi2.Connection
            # subclass, whose MRO still exposes the unguarded _sqlite3.Connection base.
            if (
                isinstance(expr, ast.Attribute)
                and expr.attr == "Connection"
                and self._is_sqlite_module_ref(expr.value)
            ):
                return True
            # type(<guarded file / sqlite instance>) is that same guarded subclass.
            if self._is_type_of_guarded_instance(expr):
                return True
            return False

        def _is_fileclass_recovery_expr(self, expr):
            """True when ``expr`` denotes a class whose MRO walk recovers an UNGUARDED primitive:
            the guarded ``io.FileIO`` / ``_io.FileIO`` (``.FileIO`` attribute), the type of a file
            object via ``.__class__`` (``open.__class__``, ``f.__class__``), the guarded
            ``sqlite3.Connection`` subclass (whose base is the unguarded ``_sqlite3.Connection``),
            or ``type(<guarded file / sqlite instance>)``. A single-assignment alias of any of these
            (``t = type(io.FileIO('x')); t.mro()``) resolves through the scope index. Ordinary class
            receivers (``int``, ``cls``, ``type(42)``, ``type('X', (), {})``) do not match, so benign
            MRO introspection stays allowed."""
            if self._is_fileclass_recovery_direct(expr):
                return True
            if _analyzer_on and isinstance(expr, ast.Name):
                rhs = _scope_idx.resolve(expr.id, expr, "rhsnode")
                if rhs is not None and self._is_fileclass_recovery_direct(rhs):
                    return True
            return False

        def _is_unbound_mro_gadget(self, node):
            """True when ``node`` is an UNBOUND MRO / getattribute call that recovers a file
            class's MRO without spelling ``.mro`` / ``.__mro__`` on the receiver:
            ``type.mro(io.FileIO)``, ``type.__getattribute__(io.FileIO, '__mro__')``,
            ``object.__getattribute__(io.FileIO, 'mro')`` or ``getattr(io.FileIO, '__mro__')``.
            Iterating the result exposes the unguarded ``_io.FileIO`` C base, so treat it as the
            same recovery gadget as ``io.FileIO.__mro__``."""
            f = node.func
            # getattr(<fileclass>, 'mro' | '__mro__')
            if (
                isinstance(f, ast.Name)
                and f.id == "getattr"
                and len(node.args) >= 2
                and self._is_fileclass_recovery_expr(node.args[0])
                and _const_fold(node.args[1], _const_env) in ("mro", "__mro__")
            ):
                return True
            if not isinstance(f, ast.Attribute):
                return False
            # type.mro(<fileclass>)
            if (
                f.attr == "mro"
                and isinstance(f.value, ast.Name)
                and f.value.id == "type"
                and node.args
                and self._is_fileclass_recovery_expr(node.args[0])
            ):
                return True
            # type.__getattribute__(<fileclass>, 'mro' | '__mro__') / object.__getattribute__(...)
            if (
                f.attr in ("__getattribute__", "__getattr__")
                and isinstance(f.value, ast.Name)
                and f.value.id in ("type", "object")
                and len(node.args) >= 2
                and self._is_fileclass_recovery_expr(node.args[0])
                and _const_fold(node.args[1], _const_env) in ("mro", "__mro__")
            ):
                return True
            return False

        def visit_Subscript(self, node):
            # An INTEGER-indexed __mro__ (cls.__mro__[1]) or the equivalent method call
            # (cls.mro()[1]) extracts a specific base class the way __bases__[0] does -- the
            # shape used to reach the original FileIO C base class (io.FileIO.mro()[1]) or
            # walk to object/subclasses. Plain iteration (for c in cls.__mro__ / cls.mro())
            # and slicing (cls.__mro__[1:]) yield the whole tuple/list for legitimate
            # introspection, so only a non-slice index is flagged.
            if (
                isinstance(node.ctx, ast.Load)
                and not isinstance(node.slice, ast.Slice)
                and (
                    (isinstance(node.value, ast.Attribute) and node.value.attr == "__mro__")
                    or (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "mro"
                        and not node.value.args
                    )
                )
            ):
                _mro_shape = "__mro__" if isinstance(node.value, ast.Attribute) else "mro()"
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": f"subscripted {_mro_shape} extracts a base class (gadget)",
                    }
                )
            # type(open).__dict__['__closure__'].__get__(open) / type(cell).__dict__[
            # 'cell_contents'].__get__(cell): fetch a gadget descriptor from a type's __dict__
            # BY NAME (a subscript, not an attribute node), then invoke __get__ to recover the
            # guarded wrapper's original callable. Flag a __dict__ subscript keyed by a gadget
            # dunder so the attribute-node gadget scan cannot be side-stepped this way. The same
            # mapping is reachable through vars(type(obj))[...], so cover that form too.
            _dunder_dict = (
                isinstance(node.value, ast.Attribute) and node.value.attr == "__dict__"
            ) or (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "vars"
                and node.value.args
            )
            if isinstance(node.ctx, ast.Load) and _dunder_dict:
                _dk = _const_fold(node.slice, _const_env)
                if isinstance(_dk, str) and _dk in _GADGET_DUNDERS:
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": f"__dict__[{_dk!r}] descriptor lookup (gadget)",
                        }
                    )
            # sys.modules['os'] pulls an already-loaded dangerous module out of the
            # loader table (os/subprocess are loaded by the host). Scope to a Load of a
            # dangerous LITERAL key so legit uses ("x" in sys.modules, sys.modules.get(
            # name), sys.modules[name] = ...) stay allowed.
            v = node.value
            # sys.modules[...] (attribute form), getattr(sys, 'modules')[...], or
            # object.__getattribute__(sys, 'modules')[...] all index the loader table -- as
            # does a single-assignment alias (m = sys.modules; m['os']), so use the alias-aware
            # helper the mutation checks already use.
            is_sys_modules = self._is_sys_modules(v)
            if isinstance(node.ctx, ast.Load) and is_sys_modules:
                # Constant-fold the key so sys.modules['o' + 's'] is caught, not just a
                # bare literal; a truly dynamic key (sys.modules[name]) stays allowed.
                key = _const_fold(node.slice, _const_env)
                if isinstance(key, str) and key.split(".")[0] in _DANGEROUS_IMPORT_NAMES:
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": "sys.modules[...] access to a sensitive module",
                        }
                    )
            if isinstance(node.ctx, (ast.Store, ast.Del)) and is_sys_modules:
                # `del sys.modules['posix']; import posix` (or reassigning the entry) drops
                # the guard-patched module object so a fresh, UNWRAPPED C module is imported,
                # bypassing the Stage 5 wrappers. Mutating the loader table has no legitimate
                # use in sandboxed compute, so deny any Store/Del on sys.modules[...].
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": "sys.modules mutation (del / assign) can drop a guarded module",
                    }
                )
            if isinstance(node.ctx, (ast.Store, ast.Del)) and self._is_sys_meta_path(v):
                # `sys.meta_path[:] = []` / `del sys.meta_path[0]` / `sys.meta_path[0] = x`
                # removes or reorders the import finder chain, dropping the sandbox's
                # workdir-module vetter so a planted helper imports without source review.
                dynamic_exec.append(
                    {
                        "type": "dynamic_exec",
                        "line": getattr(node, "lineno", -1),
                        "description": (
                            "sys.meta_path mutation (del / assign) can remove the sandbox "
                            "workdir-module vetter"
                        ),
                    }
                )
            # globals()['__builtins__'] / locals()[...] / vars()[...] pulls the builtins
            # namespace (or a dangerous module) out of the namespace dict, e.g.
            # getattr(globals()['__builtins__'], '__import__')('os'). Flag a Load of a
            # dangerous literal key off a bare globals()/locals()/vars() call.
            if isinstance(node.ctx, ast.Load) and self._is_namespace_dict_expr(v):
                key = _const_fold(node.slice, _const_env)
                if isinstance(key, str):
                    _ns_hit = (
                        key in ("__builtins__", "__builtin__")
                        or key.split(".")[0] in _DANGEROUS_IMPORT_NAMES
                    )
                    if not _ns_hit and _analyzer_on:
                        # The namespace dict also exposes a module-level / local ALIAS bound to a
                        # sink (import os; f = os.system; globals()['f']('touch /tmp/p')), which
                        # the literal-key check above misses. Resolve the key through the alias
                        # index -- a shell / exec-builtin / deserializer sink alias makes the
                        # namespace-dict lookup the sink itself. resolve() walks local->module,
                        # matching globals() (module) and locals()/vars() (local) in the usual case.
                        _ns_hit = (
                            _scope_idx.resolve(key, node, "shell") is not None
                            or _scope_idx.resolve(key, node, "execb") is not None
                            or _scope_idx.resolve(key, node, "deser") is not None
                        )
                    if _ns_hit:
                        dynamic_exec.append(
                            {
                                "type": "dynamic_exec",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    "namespace-dict access to builtins / a sensitive module "
                                    "or a sink alias"
                                ),
                            }
                        )
            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            if self.loop_depth == 0:
                self.generic_visit(node)
                return
            if node.type is None:
                exception_catching.append(
                    {
                        "type": "bare_except_in_loop",
                        "line": node.lineno,
                        "description": "Bare except in loop catches TimeoutError and continues looping",
                    }
                )
            elif isinstance(node.type, ast.Name):
                # Flag BaseException/TimeoutError but NOT Exception: `except
                # Exception` can't catch SystemExit/KeyboardInterrupt, so it
                # can't suppress timeout enforcement.
                if node.type.id in ("TimeoutError", "BaseException"):
                    exception_catching.append(
                        {
                            "type": f"catches_{node.type.id}_in_loop",
                            "line": node.lineno,
                            "description": f"Catches {node.type.id} in loop - may suppress timeout and continue",
                        }
                    )
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        if elt.id in ("TimeoutError", "BaseException"):
                            exception_catching.append(
                                {
                                    "type": f"catches_{elt.id}_in_loop",
                                    "line": node.lineno,
                                    "description": f"Catches {elt.id} in loop - may suppress timeout and continue",
                                }
                            )
            self.generic_visit(node)

    visitor = SignalEscapeVisitor()
    visitor.visit(tree)

    if visitor.imports_signal and not signal_tampering:
        warnings.append("Code imports 'signal' module - review manually for safety")

    # Static host policy: block metadata hosts and any literal host outside the
    # trusted allowlist; uploads blocked regardless of host. Dynamic hosts are
    # caught by the bash blocklist.
    network_calls: list[dict] = []
    sensitive_file_reads: list[dict] = []
    _NETWORK_FQ_PREFIXES = (
        "socket.socket",
        "socket.create_connection",
        "socket.getaddrinfo",
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
        "urllib3.",
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.delete",
        "requests.patch",
        "requests.head",
        "requests.request",
        "requests.Session",
        "http.client.HTTPConnection",
        "http.client.HTTPSConnection",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "httpx.get",
        "httpx.post",
        "httpx.put",
        "httpx.patch",
        "httpx.delete",
        "httpx.request",
        "httpx.Client",
        "httpx.AsyncClient",
        "aiohttp.ClientSession",
    )
    _UPLOAD_HTTP_METHODS = (
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.delete",
        "requests.request",
        "httpx.post",
        "httpx.put",
        "httpx.patch",
        "httpx.delete",
        "httpx.request",
        "urllib.request.urlopen",
        "urllib.request.Request",
    )
    _UPLOAD_HF_FQ = (
        "huggingface_hub.upload_file",
        "huggingface_hub.upload_folder",
        "huggingface_hub.upload_large_folder",
        "huggingface_hub.create_commit",
    )
    _UPLOAD_HF_METHODS = frozenset(
        {
            "upload_file",
            "upload_folder",
            "upload_large_folder",
            "create_commit",
        }
    )
    # Cloud-metadata / link-local hosts.
    _METADATA_HOST_LITERALS = {
        "169.254.169.254",
        "fd00:ec2::254",
        "metadata.google.internal",
        "metadata",
        "metadata.tencentyun.com",
        "100.100.100.200",
        "100.100.100.110",
        "169.254.170.2",
        "169.254.170.23",
    }
    _METADATA_HOST_PREFIXES = (
        "169.254.",
        "100.64.",
    )
    # Allowlist kept explicit so each entry is auditable.
    _TRUSTED_PUBLIC_HOST_LITERALS = frozenset(
        {
            # search
            "www.google.com",
            "google.com",
            "www.bing.com",
            "bing.com",
            "duckduckgo.com",
            "html.duckduckgo.com",
            # encyclopedic / reference
            "wikipedia.org",
            "www.wikipedia.org",
            "wikimedia.org",
            "www.wikimedia.org",
            "wikidata.org",
            "www.wikidata.org",
            "commons.wikimedia.org",
            "www.britannica.com",
            "openlibrary.org",
            "www.openstreetmap.org",
            # ML / dev / data
            "huggingface.co",
            "hf.co",
            "github.com",
            "api.github.com",
            "raw.githubusercontent.com",
            "gist.github.com",
            "docs.github.com",
            "pypi.org",
            "files.pythonhosted.org",
            "www.npmjs.com",
            "registry.npmjs.org",
            "crates.io",
            "static.crates.io",
            # docs
            "docs.python.org",
            "python.org",
            "www.python.org",
            "developer.mozilla.org",
            "developer.apple.com",
            "learn.microsoft.com",
            "docs.docker.com",
            "pytorch.org",
            "docs.pytorch.org",
            "tensorflow.org",
            "www.tensorflow.org",
            "numpy.org",
            "pandas.pydata.org",
            "scipy.org",
            "scikit-learn.org",
            "matplotlib.org",
            "fastapi.tiangolo.com",
            "starlette.io",
            # academic
            "arxiv.org",
            "export.arxiv.org",
            "scholar.google.com",
            "openreview.net",
            "semanticscholar.org",
            "www.semanticscholar.org",
            "biorxiv.org",
            "www.biorxiv.org",
            "medrxiv.org",
            "www.medrxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "www.ncbi.nlm.nih.gov",
            # Q&A / community
            "stackoverflow.com",
            "stackexchange.com",
            "askubuntu.com",
            "superuser.com",
            "serverfault.com",
            # standards
            "www.w3.org",
            "tools.ietf.org",
            "datatracker.ietf.org",
            "www.rfc-editor.org",
            # reputable news
            "www.bbc.com",
            "www.bbc.co.uk",
            "www.reuters.com",
            "apnews.com",
            "www.nature.com",
            "www.science.org",
            # government / open data
            "data.gov",
            "catalog.data.gov",
            "www.census.gov",
            "www.nasa.gov",
            "data.nasa.gov",
            "www.cdc.gov",
            "www.nih.gov",
            "www.who.int",
            # weather / time
            "api.weather.gov",
            "worldtimeapi.org",
        }
    )
    _TRUSTED_PUBLIC_HOST_SUFFIXES = (
        ".wikipedia.org",
        ".wikimedia.org",
        ".wiktionary.org",
        ".wikibooks.org",
        ".wikiquote.org",
        ".wikisource.org",
        ".wikiversity.org",
        ".wikivoyage.org",
        ".stackexchange.com",
        ".hf.co",
        ".huggingface.co",
        ".githubusercontent.com",
        ".github.io",
        ".arxiv.org",
        ".readthedocs.io",
        ".readthedocs.org",
    )
    _SENSITIVE_FILE_PREFIXES = (
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/ssh/",
    )
    _SENSITIVE_FILE_RE = re.compile(r"^/proc/(?:self|\d+)/(?:environ|cmdline|task/\d+/environ)$")

    def _normalize_host(host: str) -> str:
        if not host:
            return ""
        h = host.strip().lower().rstrip(".")
        if "@" in h:
            h = h.split("@", 1)[1]
        if h.startswith("[") and "]" in h:
            h = h[1 : h.index("]")]
        elif h.count(":") == 1:
            h = h.split(":", 1)[0]
        return h

    def _is_metadata_host(host: str) -> bool:
        h = _normalize_host(host)
        if not h:
            return False
        if h in _METADATA_HOST_LITERALS:
            return True
        if any(h.startswith(p) for p in _METADATA_HOST_PREFIXES):
            return True
        return False

    def _is_trusted_host(host: str) -> bool:
        h = _normalize_host(host)
        if not h:
            return False
        if h in _TRUSTED_PUBLIC_HOST_LITERALS:
            return True
        return any(h.endswith(s) for s in _TRUSTED_PUBLIC_HOST_SUFFIXES)

    def _call_is_upload_shape(node: ast.Call, fq: str) -> bool:
        """True for statically obvious upload shapes (files=, data=open(), bytes literal)."""
        if fq in _UPLOAD_HF_FQ:
            return True
        if fq not in _UPLOAD_HTTP_METHODS:
            return False
        for kw in node.keywords or []:
            if kw.arg == "files":
                return True
            if kw.arg == "data":
                v = kw.value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "open":
                    return True
                if isinstance(v, ast.Constant) and isinstance(v.value, (bytes, bytearray)):
                    return True
        return False

    # Bare method-name fallback (`x.upload_file(...)`) is fuzzy, so it fires only
    # when huggingface_hub/hf_api is imported; else paramiko.upload_file,
    # boto3.create_commit, etc. would false-positive. Pre-scan for the imports.
    _HF_IMPORT_MODULES = (
        "huggingface_hub",
        "hf_api",
        "huggingface_hub.hf_api",
    )

    def _module_has_hf_import(tree: ast.AST) -> bool:
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    if alias.name.split(".", 1)[0] in _HF_IMPORT_MODULES:
                        return True
            elif isinstance(n, ast.ImportFrom):
                root = (n.module or "").split(".", 1)[0]
                if root in _HF_IMPORT_MODULES:
                    return True
            elif isinstance(n, ast.Call) and n.args:
                # __import__('huggingface_hub'), importlib.import_module(...),
                # and bare import_module(...) (via `from importlib import ...`).
                arg0 = n.args[0]
                if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)):
                    continue
                if arg0.value.split(".", 1)[0] not in _HF_IMPORT_MODULES:
                    continue
                func = n.func
                if isinstance(func, ast.Name) and func.id in {
                    "__import__",
                    "import_module",
                }:
                    return True
                if isinstance(func, ast.Attribute) and func.attr == "import_module":
                    return True
        return False

    _hf_in_scope = _module_has_hf_import(tree)

    def _method_call_hf_upload_name(node: ast.Call) -> str | None:
        """Return the HF upload method name (`upload_file`, ...) or None. Covers
        the Attribute and bare-Name forms; the bare-name branch fires only when
        an HF import is in scope so paramiko/boto3 don't false-positive."""
        if not _hf_in_scope:
            return None
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in _UPLOAD_HF_METHODS:
            return f.attr
        if isinstance(f, ast.Name) and f.id in _UPLOAD_HF_METHODS:
            return f.id
        return None

    # Kwargs that ship a credential over the wire. The sandbox env strips
    # credentials up front, so any value here is hard-coded or lifted from parent.
    _HF_SENSITIVE_KWARGS = frozenset(
        {
            "token",
            "hf_token",
            "api_token",
            "api_key",
            "auth_token",
            "access_token",
            "password",
            "secret",
        }
    )

    def _is_os_environ(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        )

    def _reads_env_or_secret(node: ast.AST | None) -> bool:
        """True if any node in the subtree resolves to an env/process read.

        Walks the whole subtree (not just the root) to catch wrappers like
        `str(os.environ)`. Covers os.environ[/.get]/os.getenv, bare getenv, and
        subprocess.{run,check_output,...} that could lift parent env via printenv.
        """
        if node is None:
            return False
        for sub in ast.walk(node):
            if _is_os_environ(sub):
                return True
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Attribute):
                    if (
                        f.attr in {"getenv", "getenvb"}
                        and isinstance(f.value, ast.Name)
                        and f.value.id == "os"
                    ):
                        return True
                    if (
                        f.attr
                        in {
                            "check_output",
                            "run",
                            "Popen",
                            "getoutput",
                            "getstatusoutput",
                        }
                        and isinstance(f.value, ast.Name)
                        and f.value.id in {"subprocess", "commands"}
                    ):
                        return True
                if isinstance(f, ast.Name) and f.id in {"getenv", "getenvb"}:
                    return True
        return False

    def _is_safe_relative_path(path: str) -> bool:
        """Relative path with no leading `/`, `~`, drive letter, or `..` segments."""
        if not isinstance(path, str) or not path:
            return False
        if path[0] in ("/", "\\", "~"):
            return False
        if len(path) >= 2 and path[1] == ":":
            return False
        return ".." not in path.replace("\\", "/").split("/")

    def _path_arg_is_sandbox_local(node: ast.AST | None) -> bool:
        """Whether the path argument resolves to a sandbox-local literal."""
        if node is None:
            return False
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, bytearray)):
            return True  # inline bytes, no file access
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _is_safe_relative_path(node.value)
        if isinstance(node, ast.Call):
            f = node.func
            is_open = (isinstance(f, ast.Name) and f.id == "open") or (
                isinstance(f, ast.Attribute) and f.attr == "open"
            )
            if is_open and node.args:
                a0 = node.args[0]
                return (
                    isinstance(a0, ast.Constant)
                    and isinstance(a0.value, str)
                    and _is_safe_relative_path(a0.value)
                )
        return False

    def _hf_upload_violation(node: ast.Call, method_name: str) -> str | None:
        """Inspect an HF upload call; return a violation reason or None.

        Policy: HF uploads are allowed only when (a) no sensitive kwarg is set,
        (b) no positional / keyword value reads `os.environ` or related env
        readers, and (c) the path arg is a sandbox-local literal: a relative
        string with no `..`, an `open(<literal>)`, or inline bytes. Dynamic /
        variable paths are rejected since safety can't be proven statically and
        a wrong-allow means credential exfiltration.
        """
        for kw in node.keywords or []:
            if kw.arg in _HF_SENSITIVE_KWARGS:
                return (
                    f"HF upload {kw.arg}= cannot be set from sandboxed code; "
                    "uploads run with the sandbox identity only"
                )
        all_values = list(node.args or []) + [kw.value for kw in (node.keywords or [])]
        for v in all_values:
            if _reads_env_or_secret(v):
                return (
                    "HF upload cannot include os.environ / os.getenv / subprocess "
                    "env reads; secrets and tokens must not be exfiltrated"
                )
        if method_name == "create_commit":
            for kw in node.keywords or []:
                if kw.arg == "operations" and isinstance(kw.value, ast.List):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Call):
                            inner = _hf_upload_violation(elt, "upload_file")
                            if inner:
                                return inner
            return None
        path_node: ast.AST | None = node.args[0] if node.args else None
        for kw in node.keywords or []:
            if kw.arg in ("path_or_fileobj", "folder_path"):
                path_node = kw.value
                break
        if not _path_arg_is_sandbox_local(path_node):
            return (
                "HF upload path must be a sandbox-local relative-path literal "
                "(no absolute paths, no '..' segments, no dynamic expressions)"
            )
        return None

    # Network-module import aliases so the FQ prefix match sees the canonical module even
    # when it is renamed: import requests as r -> {"r": "requests"}, import urllib.request as
    # u -> {"u": "urllib.request"}, from urllib import request as req -> {"req":
    # "urllib.request"}. Without this, r.get('http://169.254.169.254/') builds fq="r.get"
    # and skips every metadata / allowlist / upload check.
    _NET_TOP_MODULES = ("socket", "urllib", "urllib3", "requests", "http", "httpx", "aiohttp")
    _net_aliases: dict[str, str] = {}
    for _n in ast.walk(tree):
        if isinstance(_n, ast.Import):
            for _a in _n.names:
                if _a.asname and _a.name.split(".")[0] in _NET_TOP_MODULES:
                    _net_aliases[_a.asname] = _a.name
        elif isinstance(_n, ast.ImportFrom) and _n.module:
            if _n.module.split(".")[0] in _NET_TOP_MODULES:
                for _a in _n.names:
                    _net_aliases[_a.asname or _a.name] = f"{_n.module}.{_a.name}"

    # The URL / address keyword arguments the stdlib + common HTTP clients accept, so a
    # keyword host (requests.get(url=...), urlopen(url=...), create_connection(address=...))
    # is extracted the same as a positional one.
    _NET_URL_KWARGS = ("url",)
    _NET_ADDR_KWARGS = ("address", "sock_addr")
    # APIs whose FIRST positional (or host= keyword) argument is a bare HOST string rather than a
    # URL: http.client.HTTP(S)Connection(host[, port]) and the socket name-resolution helpers. For
    # these the literal arg is checked directly as a host (there is no scheme to parse out first).
    _NET_HOST_APIS = frozenset(
        {
            "http.client.HTTPConnection",
            "http.client.HTTPSConnection",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyname_ex",
        }
    )
    _NET_HOST_KWARGS = ("host",)
    # Network-client constructors whose INSTANCES expose request methods. A call chained directly
    # off such a constructor -- requests.Session().get(url), httpx.Client().get(url),
    # urllib.request.build_opener().open(url) -- has a short fq (just the method name), so it is
    # matched by its receiver constructor instead of the module-rooted fq prefix.
    _NET_CLIENT_CTORS = frozenset(
        {
            "requests.Session",
            "requests.sessions.Session",
            "httpx.Client",
            "httpx.AsyncClient",
            "aiohttp.ClientSession",
            "urllib.request.build_opener",
            "urllib3.PoolManager",
            "urllib3.HTTPConnectionPool",
            "urllib3.HTTPSConnectionPool",
            "urllib3.connectionpool.HTTPConnectionPool",
            "urllib3.connectionpool.HTTPSConnectionPool",
        }
    )
    # Instance request methods. `.request(method, url)` carries the URL at arg1 (see below); the
    # others take it at arg0. `.get`/`.open` etc. on a non-client receiver (dict.get, file.open)
    # are excluded because the receiver must be a client-ctor Call or a tracked client alias.
    _NET_CLIENT_METHODS = frozenset(
        {
            "get",
            "post",
            "put",
            "delete",
            "head",
            "patch",
            "options",
            "request",
            "open",
        }
    )
    _NET_CLIENT_URL_AT_ARG1 = frozenset({"request"})

    def _net_call_fq(_call):
        # The alias-resolved fully-qualified name of a Call's callee (import requests as r ->
        # r.Session() folds to requests.Session), or "" if it is not an attribute/name call.
        if not isinstance(_call, ast.Call):
            return ""
        _parts: list[str] = []
        _cur = _call.func
        while isinstance(_cur, ast.Attribute):
            _parts.insert(0, _cur.attr)
            _cur = _cur.value
        if isinstance(_cur, ast.Name):
            _parts.insert(0, _cur.id)
        if _parts and _parts[0] in _net_aliases:
            _parts = _net_aliases[_parts[0]].split(".") + _parts[1:]
        return ".".join(_parts) if _parts else ""

    # Variables bound by a same-name single assignment to a network-client constructor, so a
    # method call on the stored instance (s = requests.Session(); s.get(url)) is matched like the
    # chained form. A name also assigned to any non-client value is excluded, so an unrelated
    # .get/.open on a reused name is not mis-flagged.
    _net_client_aliases: set[str] = set()
    _net_client_disqualified: set[str] = set()
    for _asn in ast.walk(tree):
        if (
            isinstance(_asn, ast.Assign)
            and len(_asn.targets) == 1
            and isinstance(_asn.targets[0], ast.Name)
        ):
            _nm = _asn.targets[0].id
            if _net_call_fq(_asn.value) in _NET_CLIENT_CTORS:
                _net_client_aliases.add(_nm)
            else:
                _net_client_disqualified.add(_nm)
    _net_client_aliases -= _net_client_disqualified

    def _net_fold_str(_n):
        # Fold a network target node to a concrete string: a module-level constant (via _const_env)
        # or a function-local single-assignment string (u = 'http://x'; urlopen(u)).
        _v = _const_fold(_n, _const_env)
        if isinstance(_v, str):
            return _v
        if isinstance(_n, ast.Name):
            _sv = _scope_idx.resolve(_n.id, _n, "strconst")
            if isinstance(_sv, str):
                return _sv
        return None

    def _net_leading_literal(_n):
        # The LEADING literal text of an f-string / concatenation, up to its first dynamic part.
        if isinstance(_n, ast.Constant) and isinstance(_n.value, str):
            return _n.value
        if isinstance(_n, ast.JoinedStr):
            _out = ""
            for _p in _n.values:
                if isinstance(_p, ast.Constant) and isinstance(_p.value, str):
                    _out += _p.value
                else:
                    break
            return _out
        if isinstance(_n, ast.BinOp) and isinstance(_n.op, ast.Add):
            return _net_leading_literal(_n.left)
        return ""

    def _net_literal_host_prefix(_n):
        # A host extracted from the leading literal of a non-fully-literal URL (f'https://hf.co/{x}',
        # 'https://hf.co/' + p): the host must be terminated by a / ? # WITHIN the literal, so a
        # dynamic tail cannot extend it (f'https://evil{x}.co/' has no literal host and returns None).
        _pre = _net_leading_literal(_n)
        if not _pre:
            return None
        _m = re.match(r"^\w+://([^/?#]+)[/?#]", _pre)
        return _m.group(1) if _m else None

    def _net_check_target(
        _node,
        _a0,
        is_host_api = False,
    ):
        # Resolve a network call's target argument to a concrete host and record a block if it is
        # a cloud-metadata host, an unresolved (opaque) target, or a host outside the allowlist.
        # ``is_host_api`` marks callees whose literal arg is a bare host (HTTPConnection('h'),
        # getaddrinfo('h', 80)) rather than a URL, so no scheme is parsed out.
        _host = None
        _url = None
        _opaque = False
        if _a0 is not None:
            if isinstance(_a0, ast.Tuple) and _a0.elts:
                _e0 = _a0.elts[0]
                if isinstance(_e0, ast.Constant) and isinstance(_e0.value, str):
                    _host = _e0.value
                else:
                    _folded = _net_fold_str(_e0)
                    if _folded is not None:
                        _host = _folded
                    else:
                        _opaque = True
            elif isinstance(_a0, ast.Constant) and isinstance(_a0.value, str):
                if is_host_api:
                    _host = _a0.value
                else:
                    _url = _a0.value
            else:
                _folded = _net_fold_str(_a0)
                if _folded is not None:
                    if is_host_api:
                        _host = _folded
                    else:
                        _url = _folded
                else:
                    _pref = _net_literal_host_prefix(_a0)
                    if _pref is not None:
                        _host = _pref
                    else:
                        _opaque = True
        if _url and _host is None:
            _m = re.match(r"^\w+://([^/?#]+)", _url)
            if _m:
                _host = _m.group(1)
        if _opaque:
            network_calls.append(
                {
                    "type": "untrusted_host_blocked",
                    "line": getattr(_node, "lineno", -1),
                    "description": (
                        "Blocked: non-literal network target cannot be checked "
                        "against the sandbox allowlist"
                    ),
                }
            )
        elif _host:
            if _is_metadata_host(_host):
                network_calls.append(
                    {
                        "type": "metadata_host_blocked",
                        "line": getattr(_node, "lineno", -1),
                        "description": "Blocked: cloud-metadata host",
                    }
                )
            elif not _is_trusted_host(_host):
                network_calls.append(
                    {
                        "type": "untrusted_host_blocked",
                        "line": getattr(_node, "lineno", -1),
                        "description": (
                            "Blocked: host not in sandbox allowlist; "
                            "use an allowed informational source"
                        ),
                    }
                )

    class NetworkAndIoVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            parts: list[str] = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                parts.insert(0, cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.insert(0, cur.id)
            # Resolve a renamed network module (import requests as r) to its canonical name
            # so the FQ-prefix match below still fires.
            if parts and parts[0] in _net_aliases:
                parts = _net_aliases[parts[0]].split(".") + parts[1:]
            fq = ".".join(parts) if parts else ""

            hf_upload_name = _method_call_hf_upload_name(node)
            if hf_upload_name is not None:
                violation = _hf_upload_violation(node, hf_upload_name)
                if violation is not None:
                    network_calls.append(
                        {
                            "type": "upload_blocked",
                            "line": getattr(node, "lineno", -1),
                            "description": f"Blocked: {violation}",
                        }
                    )

            # Direct sock.connect((host, port)) bypasses the FQ-prefix branch. Only the
            # (host, port) TUPLE form is an AF_INET network connect; a bare-string arg to
            # .connect() is an AF_UNIX socket PATH or a DB connector path (sqlite3.connect(
            # 'local.db'), duckdb.connect(':memory:')), not a network host, so restrict host
            # classification to the tuple form (the bare-string branch only mis-flagged benign
            # local database opens; filesystem escape for those is enforced at runtime instead).
            # connect_ex((host, port)) opens the SAME outbound connection but returns an errno
            # instead of raising, so classify it identically.
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("connect", "connect_ex"):
                a0 = node.args[0] if node.args else None
                if a0 is None:
                    for _kw in node.keywords or []:
                        if _kw.arg == "address":
                            a0 = _kw.value
                            break
                host_lit = None
                host_lit_opaque = False
                if isinstance(a0, ast.Tuple) and a0.elts:
                    e0 = a0.elts[0]
                    if isinstance(e0, ast.Constant) and isinstance(e0.value, str):
                        host_lit = e0.value
                    else:
                        _folded = _net_fold_str(e0)
                        if _folded is not None:
                            host_lit = _folded
                        else:
                            # A raw AF_INET connect to an unresolved host (sock.connect(
                            # (user_host, port))) is an egress the runtime cannot filter,
                            # so fail closed exactly like the urllib / requests branch.
                            host_lit_opaque = True
                if host_lit_opaque:
                    network_calls.append(
                        {
                            "type": "untrusted_host_blocked",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                "Blocked: non-literal network target cannot be checked "
                                "against the sandbox allowlist"
                            ),
                        }
                    )
                elif host_lit:
                    if _is_metadata_host(host_lit):
                        network_calls.append(
                            {
                                "type": "metadata_host_blocked",
                                "line": getattr(node, "lineno", -1),
                                "description": "Blocked: cloud-metadata host",
                            }
                        )
                    elif not _is_trusted_host(host_lit):
                        network_calls.append(
                            {
                                "type": "untrusted_host_blocked",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    "Blocked: host not in sandbox allowlist; "
                                    "use an allowed informational source"
                                ),
                            }
                        )

            if fq and any(fq.startswith(p) for p in _NETWORK_FQ_PREFIXES):
                # 1) Upload-shape check (host-independent).
                if _call_is_upload_shape(node, fq):
                    network_calls.append(
                        {
                            "type": "upload_blocked",
                            "line": getattr(node, "lineno", -1),
                            "description": ("Blocked: file upload disallowed in sandbox"),
                        }
                    )

                # 2) Extract the host (URL string, bare host, or (host, port) tuple) and check
                # it. The target may be a positional first arg OR a keyword (requests.get(url=...),
                # urlopen(url=...), create_connection(address=(host, port)), HTTPConnection(
                # host=...)). Bare-host callees (HTTPConnection, getaddrinfo) treat the literal
                # arg as a host directly; everything else parses a scheme://host URL. A target
                # that stays fully opaque fails closed. See _net_check_target.
                a0 = node.args[0] if node.args else None
                if a0 is None:
                    for _kw in node.keywords or []:
                        if _kw.arg in _NET_URL_KWARGS or _kw.arg in _NET_ADDR_KWARGS:
                            a0 = _kw.value
                            break
                        if fq in _NET_HOST_APIS and _kw.arg in _NET_HOST_KWARGS:
                            a0 = _kw.value
                            break
                _net_check_target(node, a0, is_host_api = (fq in _NET_HOST_APIS))

            # Client-instance request call: requests.Session().get(url), httpx.Client().get(url),
            # build_opener().open(url), or s.get(url) where s was bound to a client constructor.
            # The chained fq is just the method name, so match by the receiver being a client-ctor
            # Call or a tracked client alias. .get/.open on a plain dict/file receiver is excluded.
            if isinstance(node.func, ast.Attribute) and node.func.attr in _NET_CLIENT_METHODS:
                _recv = node.func.value
                _is_client = (
                    isinstance(_recv, ast.Call) and _net_call_fq(_recv) in _NET_CLIENT_CTORS
                ) or (isinstance(_recv, ast.Name) and _recv.id in _net_client_aliases)
                if _is_client:
                    _idx = 1 if node.func.attr in _NET_CLIENT_URL_AT_ARG1 else 0
                    _ca0 = node.args[_idx] if len(node.args) > _idx else None
                    if _ca0 is None:
                        for _kw in node.keywords or []:
                            if _kw.arg in _NET_URL_KWARGS:
                                _ca0 = _kw.value
                                break
                    _net_check_target(node, _ca0, is_host_api = False)

            is_open_call = (
                (isinstance(node.func, ast.Name) and node.func.id == "open")
                or fq in ("io.open", "pathlib.Path.open")
                or fq.endswith(".open")
            )
            if is_open_call and node.args:
                a0 = node.args[0]
                path_lit = None
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    path_lit = a0.value
                if path_lit:
                    flagged = False
                    if any(path_lit.startswith(p) for p in _SENSITIVE_FILE_PREFIXES):
                        flagged = True
                    elif _SENSITIVE_FILE_RE.match(path_lit):
                        flagged = True
                    if flagged:
                        sensitive_file_reads.append(
                            {
                                "type": "sensitive_file_read",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    f"open({path_lit!r}) targets a host identity / "
                                    "credential file; sandboxed code may not read it"
                                ),
                            }
                        )
            self.generic_visit(node)

    def _fs_block(node, description):
        filesystem_violations.append(
            {
                "type": "filesystem_violation",
                "line": getattr(node, "lineno", -1),
                "description": description,
            }
        )

    # Read-only scanner: filesystem WRITES are confined at runtime by the Stage 5
    # realpath backstop, so this static pass only blocks host-secret READS (the
    # backstop leaves reads unpatched). A sensitive absolute / ~-rooted literal in
    # ANY call arg is flagged -- this covers open()/os.open and library loaders that
    # internally open the path (pandas.read_csv('/etc/shadow'), numpy.load('/etc/passwd')).
    # The `..` / `~` traversal-escape form is flagged only for the dedicated
    # open()/read callees, so benign relative-path building (os.path.join('..','x'))
    # is not caught. Dynamic (non-foldable) paths are left to the runtime backstop.
    _READ_METHODS = ("read_text", "read_bytes")
    # Pathlib read methods carry the path on the RECEIVER, not in an argument:
    # Path('../../.ssh/id_rsa').read_text() has no call args, so the constructor path
    # must be inspected separately.
    _PATHLIB_READ_METHODS = ("read_text", "read_bytes", "open")
    _PATHLIB_CTORS = (
        "Path",
        "PurePath",
        "PosixPath",
        "PurePosixPath",
        "WindowsPath",
        "PureWindowsPath",
    )
    # shutil.copy*/move read their SOURCE (first arg) from the host, so a `..` traversal
    # or ~ source copies a host secret into the workdir even though it is not an open()/
    # read callee. Treat them as read callees so the traversal/sensitive check applies.
    _SHUTIL_COPY_SINKS = (
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
    )
    _SHUTIL_COPY_METHODS = ("copy", "copy2", "copyfile", "copytree", "move")
    # Import aliases so the traversal check still recognizes a renamed callee:
    #   from pathlib import Path as P  -> P('../../etc/passwd').read_text()
    #   import shutil as sh            -> sh.copy('../../etc/passwd', 'x')
    _pathlib_ctor_aliases = set(_PATHLIB_CTORS)
    _shutil_aliases = {"shutil"}
    # from shutil import copy as c / copyfile / move -> bare-name aliases whose SOURCE (first
    # arg) is a host read, e.g. c('../../../etc/passwd', 'x'). Tracked so the traversal check
    # treats them as read callees like the attribute form shutil.copy(...).
    _shutil_copy_from_aliases: set[str] = set()
    # from os import open as oo / from io import open as X / from builtins import open as X:
    # the read-only os.open is deliberately allowed OUTSIDE the workdir by the runtime
    # guard, so a traversal read via such an alias must be caught statically.
    _open_from_aliases: set[str] = set()
    # Receiver-module aliases for the open() attribute form (import builtins as b; b.open(...),
    # import io as i; i.open(...), import os as o; o.open(...)), so an aliased-module read is
    # recognized like the literal builtins/io/os.open forms.
    _open_mod_aliases = {"builtins", "__builtins__", "io", "os"}
    # os/subprocess module aliases + from-import shell-name aliases, so a shell command
    # string that reads a host secret (os.system('cat /etc/passwd')) is scanned even when
    # os/subprocess is renamed.
    _os_mod_aliases = {"os"}
    _subprocess_mod_aliases = {"subprocess"}
    _shell_name_aliases: dict[str, str] = {}
    # from subprocess import run as r / call / check_call / check_output / Popen -> bare-name
    # aliases that run an unguarded child, so r(['cat', '../../etc/shadow']) reads a host
    # secret. Tracked so the read-callee traversal check recognizes them like subprocess.run.
    _subprocess_exec_from_aliases: set[str] = set()
    # from os.path import join as j / normpath / abspath -> {alias: 'join'} so a path builder
    # folder recognizes the bare-name form open(join('/etc', 'passwd')).
    _pathfunc_from_aliases: dict[str, str] = {}
    # operator module + `from operator import methodcaller` aliases, so a deferred method
    # applied to an os/subprocess receiver (methodcaller('popen', 'cat /etc/passwd')(os)) is
    # rewritten to the direct call before the read scanner runs.
    _operator_mod_aliases = {"operator"}
    _methodcaller_from_aliases: set[str] = set()
    for _imp in ast.walk(tree):
        if isinstance(_imp, ast.ImportFrom) and _imp.module == "pathlib":
            for _a in _imp.names:
                if _a.name in _PATHLIB_CTORS:
                    _pathlib_ctor_aliases.add(_a.asname or _a.name)
        elif isinstance(_imp, ast.ImportFrom) and _imp.module in ("os", "io", "builtins"):
            for _a in _imp.names:
                if _a.name == "open":
                    _open_from_aliases.add(_a.asname or "open")
                # `from os import system as s` / popen: record the os shell-exec alias here too
                # (this elif consumes the `os` module, so the subprocess branch below never sees
                # it), else _scan_shell_string_reads skips s('cat /etc/passwd').
                _fq = f"{_imp.module}.{_a.name}"
                if _fq in _SHELL_EXEC_FUNCS:
                    _shell_name_aliases[_a.asname or _a.name] = _fq
        elif isinstance(_imp, ast.ImportFrom) and _imp.module == "shutil":
            for _a in _imp.names:
                if _a.name in _SHUTIL_COPY_METHODS:
                    _shutil_copy_from_aliases.add(_a.asname or _a.name)
        elif isinstance(_imp, ast.ImportFrom) and _imp.module == "subprocess":
            for _a in _imp.names:
                _fq = f"subprocess.{_a.name}"
                if _fq in _SHELL_EXEC_FUNCS:
                    _shell_name_aliases[_a.asname or _a.name] = _fq
                if _a.name in ("run", "call", "check_call", "check_output", "Popen"):
                    _subprocess_exec_from_aliases.add(_a.asname or _a.name)
        elif isinstance(_imp, ast.ImportFrom) and _imp.module in (
            "os.path",
            "posixpath",
            "ntpath",
        ):
            for _a in _imp.names:
                if _a.name in ("join", "normpath", "abspath"):
                    _pathfunc_from_aliases[_a.asname or _a.name] = _a.name
        elif isinstance(_imp, ast.ImportFrom) and _imp.module == "operator":
            for _a in _imp.names:
                if _a.name == "methodcaller":
                    _methodcaller_from_aliases.add(_a.asname or _a.name)
        elif isinstance(_imp, ast.Import):
            for _a in _imp.names:
                if _a.name == "shutil":
                    _shutil_aliases.add(_a.asname or "shutil")
                elif _a.name == "operator":
                    _operator_mod_aliases.add(_a.asname or "operator")
                elif _a.name == "os":
                    _os_mod_aliases.add(_a.asname or "os")
                    _open_mod_aliases.add(_a.asname or "os")  # o.open(...)
                elif _a.name in ("posix", "nt"):
                    # posix / nt are the os C backend (posix.system == os.system), so a shell
                    # string passed to them must be scanned for embedded secret reads too.
                    _os_mod_aliases.add(_a.asname or _a.name)
                elif _a.name in ("io", "builtins"):
                    _open_mod_aliases.add(_a.asname or _a.name)  # i.open(...) / b.open(...)
                elif _a.name == "subprocess":
                    _subprocess_mod_aliases.add(_a.asname or "subprocess")

    def _fold_pathjoin_call(call):
        # Fold an os.path.join/normpath/abspath call that _const_fold's owner check misses
        # because os is aliased (import os as o -> o.path.join) or the function is
        # from-imported (from os.path import join -> join(...)). Recurses through
        # _fold_read_arg so scope-local constants inside the args still resolve.
        if not isinstance(call, ast.Call):
            return None
        fn = call.func
        pname = None
        if isinstance(fn, ast.Attribute) and fn.attr in ("join", "normpath", "abspath"):
            owner = fn.value
            if (
                isinstance(owner, ast.Attribute)
                and owner.attr == "path"
                and isinstance(owner.value, ast.Name)
                and owner.value.id in _os_mod_aliases
            ):
                pname = fn.attr
            elif isinstance(owner, ast.Name) and owner.id in ("posixpath", "ntpath"):
                pname = fn.attr
        elif isinstance(fn, ast.Name) and fn.id in _pathfunc_from_aliases:
            pname = _pathfunc_from_aliases[fn.id]
        if pname is None or not call.args:
            return None
        parts = []
        for a in call.args:
            v = _fold_read_arg(a)
            if v is None:
                return None
            parts.append(v)
        try:
            if pname == "join":
                return os.path.join(*parts)
            if len(parts) == 1:
                return (
                    os.path.normpath(parts[0]) if pname == "normpath" else os.path.abspath(parts[0])
                )
        except Exception:
            return None
        return None

    def _unwrap_container_node(n):
        # `[open][0]` / `(open,)[0]` / `{'k': open}['k']`: resolve an inline literal-container
        # index to the element node so a container-hidden alias is seen through.
        if not isinstance(n, ast.Subscript):
            return n
        container = n.value
        ci = _const_fold(n.slice, _const_env)
        if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
            if -len(container.elts) <= ci < len(container.elts):
                return container.elts[ci]
        if isinstance(container, ast.Dict) and ci is not None:
            for k, v in zip(container.keys, container.values):
                if k is not None and _const_fold(k, _const_env) == ci:
                    return v
        return n

    def _resolves_to_open(fn):
        # A callee that is `open`, a `from os/io/builtins import open as X` alias, a
        # single-assignment alias (o = open; o('../../etc/passwd').read()), a
        # container-hidden alias (o = [open][0]; o(...)), the attribute forms
        # builtins.open / __builtins__.open / io.open / os.open, or any of these behind a
        # trailing .__call__ (open.__call__('../../etc/passwd')).
        while isinstance(fn, ast.Attribute) and fn.attr == "__call__":
            fn = fn.value
        if isinstance(fn, ast.Name):
            if fn.id == "open" or fn.id in _open_from_aliases:
                return True
            rhs = _unwrap_container_node(_scope_idx.resolve(fn.id, fn, "rhsnode"))
            if isinstance(rhs, ast.Name) and rhs.id == "open":
                return True
            if (
                isinstance(rhs, ast.Attribute)
                and rhs.attr == "open"
                and isinstance(rhs.value, ast.Name)
                and rhs.value.id in _open_mod_aliases
            ):
                return True
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "open"
            and isinstance(fn.value, ast.Name)
            and fn.value.id in _open_mod_aliases
        ):
            return True
        return False

    def _is_shutil_copy_callee(fn):
        while isinstance(fn, ast.Attribute) and fn.attr == "__call__":
            fn = fn.value
        # A single-assignment alias (c = shutil.copy; c('../../etc/passwd', 'x')) hides the
        # shutil.copy attribute form behind a bare Name, so resolve the RHS before matching.
        if isinstance(fn, ast.Name):
            fn = _unwrap_container_node(_scope_idx.resolve(fn.id, fn, "rhsnode"))
        return (
            isinstance(fn, ast.Attribute)
            and fn.attr in _SHUTIL_COPY_METHODS
            and isinstance(fn.value, ast.Name)
            and fn.value.id in _shutil_aliases
        )

    def _is_subprocess_exec_callee(fn):
        # subprocess.run/call/check_call/check_output/Popen run an unguarded child, so a
        # `..` traversal in a literal argv (subprocess.run(['cat', '../../root/.ssh/id_rsa']))
        # reads a host secret. Treat these as read callees so the traversal check fires on
        # their argv path elements (absolute-sensitive elements already block regardless).
        while isinstance(fn, ast.Attribute) and fn.attr == "__call__":
            fn = fn.value
        # A from-import (from subprocess import run as r -> r([...])) or a single-assignment
        # alias (r = subprocess.run) both hide the exec attribute form behind a bare Name;
        # resolve/recognize them before matching the attribute form.
        if isinstance(fn, ast.Name):
            if fn.id in _subprocess_exec_from_aliases:
                return True
            fn = _unwrap_container_node(_scope_idx.resolve(fn.id, fn, "rhsnode"))
        return (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id in _subprocess_mod_aliases
            and fn.attr in ("run", "call", "check_call", "check_output", "Popen")
        )

    def _fold_read_arg(arg):
        # Fold a read-path argument to a concrete string, resolving a module-level
        # constant (via _const_env) OR a function-local single-assignment string
        # constant (p = '/etc/passwd' inside a def) via the scope index.
        v = _const_fold(arg, _const_env)
        if isinstance(v, (str, bytes, bytearray)):
            return _to_text(v)
        if isinstance(arg, ast.Name):
            sv = _scope_idx.resolve(arg.id, arg, "strconst")
            if isinstance(sv, (str, bytes, bytearray)):
                return _to_text(sv)
            return None
        # os-aliased / from-imported path builder (o.path.join(...), join(...)) that
        # _const_fold's literal-`os` owner check misses.
        pj = _fold_pathjoin_call(arg)
        if isinstance(pj, (str, bytes, bytearray)):
            return _to_text(pj)
        # A path-builder call (os.path.join(p, 'passwd'), normpath, ...) whose arguments
        # include function-local single-assignment string constants stays opaque to the
        # module-level _const_env. Augment the fold env with those scope-local names' RHS
        # NODES and re-fold so `p = '/etc'; open(os.path.join(p, 'passwd'))` is caught.
        # (_const_fold maps names to RHS nodes, not values.)
        _local_env = None
        for _sub in ast.walk(arg):
            if isinstance(_sub, ast.Name) and isinstance(_sub.ctx, ast.Load):
                if _const_env is not None and _sub.id in _const_env:
                    continue
                _svn = _scope_idx.resolve(_sub.id, _sub, "rhsnode")
                if _svn is not None:
                    if _local_env is None:
                        _local_env = dict(_const_env or {})
                    _local_env[_sub.id] = _svn
        if _local_env is not None:
            v = _const_fold(arg, _local_env)
            if isinstance(v, (str, bytes, bytearray)):
                return _to_text(v)
        return None

    def _pathlib_receiver_path(recv, _seen = None):
        # Resolve a pathlib receiver to a concrete path: Path(...) / pathlib.Path(...)
        # (all constructor args joined), a `/` join (Path('/etc') / 'passwd'), a
        # .joinpath(...) chain, or a single-assignment name bound to any of these
        # (p = Path('..') / 'etc' / 'passwd'; p.read_text()).
        if isinstance(recv, ast.Name):
            # Resolve the name to its single-assignment RHS (cycle-guarded).
            if _seen is None:
                _seen = set()
            if recv.id in _seen:
                return None
            _seen.add(recv.id)
            rhs = _scope_idx.resolve(recv.id, recv, "rhsnode")
            if rhs is None:
                return None
            return _pathlib_receiver_path(rhs, _seen)
        if isinstance(recv, ast.BinOp) and isinstance(recv.op, ast.Div):
            base = _pathlib_receiver_path(recv.left, _seen)
            rv = _fold_read_arg(recv.right)
            if base is None or rv is None:
                return None
            try:
                return os.path.join(base, rv)
            except Exception:
                return None
        if not isinstance(recv, ast.Call):
            return None
        rf = recv.func
        # No-op path-identity methods (resolve/absolute/expanduser) return the same file, so
        # look through them: Path('/etc').joinpath('passwd').resolve().read_text() still
        # reads /etc/passwd. expanduser() only makes a leading ~ concrete, which the
        # sensitive check already handles on the pre-expansion form.
        if isinstance(rf, ast.Attribute) and rf.attr in ("resolve", "absolute", "expanduser"):
            return _pathlib_receiver_path(rf.value, _seen)
        if isinstance(rf, ast.Attribute) and rf.attr == "joinpath":
            base = _pathlib_receiver_path(rf.value, _seen)
            if base is None:
                return None
            parts = [base]
            for a in recv.args:
                v = _fold_read_arg(a)
                if v is None:
                    return None
                parts.append(v)
            try:
                return os.path.join(*parts)
            except Exception:
                return None
        # A single-assignment alias of the constructor (P = pathlib.Path / P = Path) is not
        # in _pathlib_ctor_aliases, so resolve a Name callee's RHS to see if it binds a
        # pathlib constructor before giving up.
        _ctor_name = isinstance(rf, ast.Name) and rf.id in _pathlib_ctor_aliases
        if not _ctor_name and isinstance(rf, ast.Name):
            _crhs = _unwrap_container_node(_scope_idx.resolve(rf.id, rf, "rhsnode"))
            _ctor_name = (isinstance(_crhs, ast.Name) and _crhs.id in _pathlib_ctor_aliases) or (
                isinstance(_crhs, ast.Attribute) and _crhs.attr in _PATHLIB_CTORS
            )
        ctor = _ctor_name or (isinstance(rf, ast.Attribute) and rf.attr in _PATHLIB_CTORS)
        if not ctor or not recv.args:
            return None
        parts = []
        for a in recv.args:
            v = _fold_read_arg(a)
            if v is None:
                return None
            parts.append(v)
        if not parts:
            return None
        try:
            return os.path.join(*parts)
        except Exception:
            return None

    def _flag_read_path(node, s, is_read_callee):
        norm = s.replace("\\", "/")
        # Collapse redundant separators / '.' segments and resolve '..' so equivalent
        # spellings (/etc//passwd, /etc/./passwd, /tmp/../etc/passwd) still match the
        # sensitive exact / dir checks. Keep the raw form for the traversal check below.
        try:
            canon = os.path.normpath(norm)
        except Exception:
            canon = norm
        if _is_sensitive_abs_path(norm) or _is_sensitive_abs_path(canon):
            _fs_block(node, f"{s!r} is a sensitive host identity / credential file")
            return True
        if is_read_callee and (s[:1] == "~" or ".." in norm.split("/")):
            _fs_block(node, f"{s!r} escapes the session workdir via path traversal")
            return True
        return False

    # Shell sinks whose first argument is always interpreted as a shell command STRING
    # (os.system('cat /etc/passwd') runs an unguarded child that leaks the file in stdout).
    _STRING_SHELL_SINKS = frozenset(
        {
            "os.system",
            "os.popen",
            "os.popen2",
            "os.popen3",
            "os.popen4",
            "subprocess.getoutput",
            "subprocess.getstatusoutput",
        }
    )

    def _shell_string_sink_fq(f):
        # Resolve a callee to its fq shell-sink name honoring os/subprocess module aliases
        # and from-import name aliases (from subprocess import getoutput as g), else None.
        if isinstance(f, ast.Attribute):
            v = f.value
            # os / posix / nt receiver as a simple name (posix.system) or a module that
            # re-exports os as an attribute (pathlib.os.system, tempfile.os.system).
            if (isinstance(v, ast.Name) and v.id in _os_mod_aliases) or (
                isinstance(v, ast.Attribute) and v.attr in ("os", "posix", "nt")
            ):
                cand = f"os.{f.attr}"
            elif (isinstance(v, ast.Name) and v.id in _subprocess_mod_aliases) or (
                isinstance(v, ast.Attribute) and v.attr == "subprocess"
            ):
                cand = f"subprocess.{f.attr}"
            else:
                cand = None
            if cand in _SHELL_EXEC_FUNCS:
                return cand
        elif isinstance(f, ast.Name):
            return _shell_name_aliases.get(f.id)
        return None

    def _rewrite_methodcaller_call(node):
        # operator.methodcaller('popen', 'cat /etc/passwd')(os) applies a deferred method to a
        # module receiver; rewrite it to the direct os.popen('cat /etc/passwd') call so the
        # read scanner tokenizes the embedded secret read. Only os/subprocess receivers are
        # rewritten, so a methodcaller aimed at a benign object is left untouched.
        if len(node.args) != 1 or node.keywords:
            return None
        mc = node.func
        while isinstance(mc, ast.Attribute) and mc.attr == "__call__":
            mc = mc.value
        if not isinstance(mc, ast.Call) or not mc.args:
            return None
        mf = mc.func
        is_mc = (
            isinstance(mf, ast.Attribute)
            and mf.attr == "methodcaller"
            and isinstance(mf.value, ast.Name)
            and mf.value.id in _operator_mod_aliases
        ) or (isinstance(mf, ast.Name) and mf.id in _methodcaller_from_aliases)
        if not is_mc:
            return None
        meth = _fold_read_arg(mc.args[0])
        if not isinstance(meth, str) or not meth.isidentifier():
            return None
        receiver = node.args[0]
        if not (
            isinstance(receiver, ast.Name)
            and (receiver.id in _os_mod_aliases or receiver.id in _subprocess_mod_aliases)
        ):
            return None
        synth = ast.Call(
            func = ast.Attribute(value = receiver, attr = meth, ctx = ast.Load()),
            args = list(mc.args[1:]),
            keywords = list(mc.keywords),
        )
        ast.copy_location(synth, node)
        ast.fix_missing_locations(synth)
        return synth

    def _is_exec_family_callee(f):
        # os.execv / os.execl / os.spawnv / os.posix_spawn ... replace or fork the guarded
        # process with an unguarded program, so a `..` traversal in their argv reads a host
        # secret the same way subprocess argv does (os.execv('/bin/cat', ['cat',
        # '../../etc/shadow'])). Treat them as read callees for the traversal check.
        while isinstance(f, ast.Attribute) and f.attr == "__call__":
            f = f.value
        fq = _shell_string_sink_fq(f)
        return fq is not None and (fq.startswith("os.exec") or fq.startswith("os.spawn"))

    def _iter_call_kwargs(call):
        # Yield (name, value_node) for every keyword argument, EXPANDING a literal **{...}
        # unpack (subprocess.run(cmd, **{'shell': True, 'cwd': p})) so a shell / cwd argument
        # smuggled through a dict unpack is seen exactly like an explicit shell= / cwd= kwarg.
        for _kw in call.keywords or []:
            if _kw.arg is not None:
                yield _kw.arg, _kw.value
            elif isinstance(_kw.value, ast.Dict):
                for _dk, _dv in zip(_kw.value.keys, _kw.value.values):
                    if (
                        _dk is not None
                        and isinstance(_dk, ast.Constant)
                        and isinstance(_dk.value, str)
                    ):
                        yield _dk.value, _dv

    def _scan_shell_string_reads(node, f):
        # os.system('cat /etc/passwd') / subprocess.run('cat /etc/passwd', shell=True): the
        # read scanner otherwise treats the whole command as one opaque path candidate, and
        # _is_sensitive_abs_path ignores strings with whitespace. Tokenize the command and
        # check each token as a read path so an embedded host-secret read is caught.
        def _first_cmd_arg():
            # The command may be positional OR the public `args=` keyword
            # (subprocess.run(args='cat /etc/passwd', shell=True) / run(args=['cat', p])),
            # including a literal **{'args': ...} unpack.
            if node.args:
                return node.args[0]
            for _name, _val in _iter_call_kwargs(node):
                if _name == "args":
                    return _val
            return None

        # subprocess.run('cat passwd', shell=True, cwd='/etc') runs the payload in an unguarded
        # shell whose cwd is /etc, so a relative reader arg reads /etc/passwd; a NON-literal cwd
        # cannot be proven sandbox-local. Extract cwd= once and thread it into the payload scan.
        _cwd_lit = None
        _cwd_dyn = False
        if _is_subprocess_exec_callee(f):
            for _name, _val in _iter_call_kwargs(node):
                if _name == "cwd":
                    _cv = _fold_read_arg(_val)
                    if isinstance(_cv, str):
                        _cwd_lit = _cv
                    elif not (isinstance(_val, ast.Constant) and _val.value is None):
                        _cwd_dyn = True
                    break

        def _scan_one_command(cmd):
            # Scan a shell command STRING (folded to a literal) for an embedded host-secret
            # read and record a violation. Delegates to the shared scanner in strict-traversal
            # mode (the os.system() shell-string policy blocks ANY .. / ~ read path), which also
            # resolves the reader past assignment / wrapper prefixes and recurses nested shells.
            if cmd is None:
                return False
            _r = _scan_command_string_for_reads(
                cmd, strict_traversal = True, cwd = _cwd_lit, cwd_dynamic = _cwd_dyn
            )
            if _r is not None:
                _fs_block(node, _r)
                return True
            return False

        # A subprocess argv that invokes a shell with -c runs the payload in an unguarded
        # child (subprocess.run(['sh', '-c', 'head -1 /etc/passwd'])). The blocked-command
        # scanner finds no blocked command (head is benign), so scan the -c payload for
        # sensitive reads here the same way a string shell sink is scanned.
        if _is_subprocess_exec_callee(f):
            argv = _first_cmd_arg()
            if isinstance(argv, (ast.List, ast.Tuple)) and argv.elts:
                _elts = [_fold_read_arg(_e) for _e in argv.elts]
                # Resolve the executed command word past wrapper prefixes (env / timeout /
                # nice / ...): subprocess.run(['env', 'bash', '-c', payload]) runs the nested
                # shell just like subprocess.run(['bash', '-c', payload]), so scan the -c
                # payload regardless of the wrapper hiding argv[0].
                _ci = _blocked_in_argv(_elts)[1]
                _sh = _elts[_ci] if _ci is not None and _ci < len(_elts) else None
                if _sh is not None and os.path.basename(_sh).lower() in _SHELL_BINARIES:
                    # An env -C DIR earlier in the SAME argv chdirs the child before the nested
                    # shell runs (['env', '-C', '/etc', 'bash', '-c', 'cat passwd']), so the -c
                    # payload's relative reads resolve against DIR, not the workdir. Fold the
                    # argv env -C into the payload's cwd (or fail closed on a dynamic DIR).
                    _pl_cwd, _pl_dyn = _cwd_lit, _cwd_dyn
                    _envc = _argv_env_chdir(_elts)
                    if _envc is not None:
                        _rdir, _rdyn = _resolve_read_chdir(_envc, {})
                        if _rdyn:
                            _pl_dyn = True
                        else:
                            _pl_cwd = _join_chdir(_cwd_lit, _rdir)
                    for _k in range(_ci + 1, len(_elts)):
                        _ev = _elts[_k]
                        if _ev is not None and (
                            _ev == "-c"
                            or (
                                _ev.startswith("-")
                                and not _ev.startswith("--")
                                and _ev.endswith("c")
                            )
                        ):
                            if _k + 1 < len(_elts) and _elts[_k + 1] is not None:
                                _rr = _scan_command_string_for_reads(
                                    _elts[_k + 1],
                                    strict_traversal = True,
                                    cwd = _pl_cwd,
                                    cwd_dynamic = _pl_dyn,
                                )
                                if _rr is not None:
                                    _fs_block(node, _rr)
                                    return True
                            break
                # env -S 'payload' / --split-string in an argv (subprocess.run(['env', '-S',
                # 'cat /etc/passwd'])) runs the split payload as the child; the -c block above
                # only covers shell binaries, so reconstruct the argv and read-scan it when the
                # resolved command word is env with a split-string flag.
                if (
                    all(_x is not None for _x in _elts)
                    and any(os.path.basename(_x).lower() == "env" for _x in _elts)
                    and any(
                        _x in ("-S", "--split-string")
                        or _x.startswith("--split-string=")
                        or (_x.startswith("-S") and len(_x) > 2)
                        for _x in _elts
                    )
                ):
                    if _scan_one_command(" ".join(shlex.quote(_x) for _x in _elts)):
                        return True

        _fq = _shell_string_sink_fq(f)
        _is_str = _fq in _STRING_SHELL_SINKS
        if not _is_str:
            # subprocess.run/call/Popen/check_output/check_call(cmd, shell=True): a string
            # command with shell=True runs through /bin/sh (these are in _SHELL_EXEC_FUNCS
            # but not in _STRING_SHELL_SINKS, so check the shell= kwarg explicitly). Uses the
            # subprocess-exec callee resolver so the attribute, from-import (from subprocess
            # import run as r) and single-assignment (r = subprocess.run) forms are all seen.
            if _is_subprocess_exec_callee(f):
                for _name, _val in _iter_call_kwargs(node):
                    if _name == "shell" and not (
                        isinstance(_val, ast.Constant) and _val.value is False
                    ):
                        _is_str = True
        _cmd_node = _first_cmd_arg()
        if not _is_str or _cmd_node is None:
            return False
        return _scan_one_command(_fold_read_arg(_cmd_node))

    class _SensitiveReadVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            _rw = _rewrite_methodcaller_call(node)
            if _rw is not None:
                # methodcaller('popen', 'cat /etc/passwd')(os): analyze the direct call form.
                self.visit_Call(_rw)
                return
            f = node.func
            fq = _fq_attr_name(f)
            method = (
                f.attr
                if isinstance(f, ast.Attribute)
                else (f.id if isinstance(f, ast.Name) else "")
            )
            # A shell-command STRING sink: scan the command for embedded sensitive reads.
            if _scan_shell_string_reads(node, f):
                return
            _is_child_exec = _is_subprocess_exec_callee(f) or _is_exec_family_callee(f)
            is_read_callee = (
                _resolves_to_open(f)
                or fq in ("io.open", "os.open")
                or fq in _SHUTIL_COPY_SINKS
                or _is_shutil_copy_callee(f)
                or (isinstance(f, ast.Name) and f.id in _shutil_copy_from_aliases)
                or _is_child_exec
                or method in _READ_METHODS
            )
            # subprocess.run(['cat', 'passwd'], cwd='/etc') reads /etc/passwd in an unguarded
            # child: the argv entry is relative and /etc alone is not sensitive, so combine a
            # literal cwd= with each relative argv path before the sensitivity check. A
            # NON-literal cwd (cwd=P) cannot be proven sandbox-local, so a relative read under
            # it fails closed (handled below).
            _sub_cwd = None
            _sub_cwd_dynamic = False
            if _is_child_exec:
                for _name, _val in _iter_call_kwargs(node):
                    if _name == "cwd":
                        _cv = _fold_read_arg(_val)
                        if isinstance(_cv, str):
                            _sub_cwd = _cv
                        elif not (isinstance(_val, ast.Constant) and _val.value is None):
                            _sub_cwd_dynamic = True
                        break
                # An env -C DIR / --chdir=DIR at the front of the argv chdirs the child before
                # the reader runs (['env', '-C', '/etc', 'cat', 'passwd']), just like the
                # shell-string env -C case; fold that dir into the cwd used for relative reads.
                _argv_for_cd = node.args[0] if node.args else None
                if _argv_for_cd is None:
                    for _name, _val in _iter_call_kwargs(node):
                        if _name == "args":
                            _argv_for_cd = _val
                            break
                if isinstance(_argv_for_cd, (ast.List, ast.Tuple)):
                    _ec = _argv_env_chdir([_fold_read_arg(_e) for _e in _argv_for_cd.elts])
                    if _ec is not None:
                        if _ec.startswith("/") or _ec.startswith("~"):
                            _sub_cwd = _ec  # absolute env -C overrides the ambient cwd
                            _sub_cwd_dynamic = False
                        elif not _sub_cwd_dynamic:
                            _sub_cwd = _join_chdir(_sub_cwd, _ec)
            # A file-reading child (cat / head / ...) with a relative argv path under a
            # non-literal cwd could read a host secret (cwd=P; P evaluates to /etc); the child
            # is unguarded, so fail closed unless the cwd is proven sandbox-local. The argv may
            # be positional OR the public args= keyword (run(args=['cat', p], cwd=P)).
            if _sub_cwd_dynamic:
                _argv0 = None
                _argv_node = node.args[0] if node.args else None
                if _argv_node is None:
                    for _name, _val in _iter_call_kwargs(node):
                        if _name == "args":
                            _argv_node = _val
                            break
                if isinstance(_argv_node, (ast.List, ast.Tuple)):
                    _av = _argv_node.elts
                    # Resolve the real command word past wrappers (timeout / env / nice / ...), so
                    # a wrapper-hidden reader (['timeout', '1', 'cat', 'passwd']) is checked, not
                    # just argv[0]. Then scan the reader's own relative args for a host read.
                    _folded = [_fold_read_arg(_e) for _e in _av]
                    _ci = _argv_command_word_index(_folded)
                    _p0 = _folded[_ci] if _ci is not None else None
                    if (
                        isinstance(_p0, str)
                        and os.path.basename(_p0).lower() in _SHELL_READ_COMMANDS
                    ):
                        for _ae in _av[_ci + 1 :]:
                            _av_s = _fold_read_arg(_ae)
                            if (
                                isinstance(_av_s, str)
                                and _av_s
                                and not _av_s.startswith("-")
                                and not _av_s.startswith("/")
                                and not _av_s.startswith("~")
                            ):
                                _fs_block(
                                    node,
                                    "child reader with a relative path under a non-literal cwd",
                                )
                                _argv0 = True
                                break
                if _argv0:
                    return
            # A `find` child-exec argv (subprocess.run(['find','/etc','-name','passwd','-exec',
            # 'cat','{}',';'])) reads host files the flat per-element scan misses: the {} placeholder
            # loses the escaping search root. Reconstruct the argv from the find command word into a
            # shell string and run it through the read scanner, which carries the find-root + -exec
            # logic. Only when every reconstructed element folds to a literal (else best-effort skip).
            if _is_child_exec:
                _fargv = node.args[0] if node.args else None
                if _fargv is None:
                    for _name, _val in _iter_call_kwargs(node):
                        if _name == "args":
                            _fargv = _val
                            break
                if isinstance(_fargv, (ast.List, ast.Tuple)):
                    _ffolded = [_fold_read_arg(_e) for _e in _fargv.elts]
                    _fci = _argv_command_word_index(_ffolded)
                    if (
                        _fci is not None
                        and isinstance(_ffolded[_fci], str)
                        and os.path.basename(_ffolded[_fci]).lower() == "find"
                        and all(isinstance(_x, str) for _x in _ffolded[_fci:])
                    ):
                        _freason = _command_reads_sensitive(shlex.join(_ffolded[_fci:]))
                        if _freason is not None:
                            _fs_block(node, f"find child-exec reads a host file ({_freason})")
                            return
            # Pathlib read on a Path(...) / join receiver: check the resolved path.
            if isinstance(f, ast.Attribute) and f.attr in _PATHLIB_READ_METHODS:
                rp = _pathlib_receiver_path(f.value)
                if rp is not None and _flag_read_path(node, rp, True):
                    return
            # Build the arg list, expanding a literal **{...} unpack so its path value is
            # scanned (open(**{'file': '../../etc/passwd'}) reads the same file that
            # open('../../etc/passwd') would, which is otherwise treated as opaque).
            scan_args = list(node.args)
            for kw in node.keywords or []:
                if kw.arg is None and isinstance(kw.value, ast.Dict):
                    scan_args.extend(v for v in kw.value.values if v is not None)
                else:
                    scan_args.append(kw.value)
            # Descend into a literal list/tuple argv so a sensitive path element is
            # scanned: subprocess.run(['cat', '/etc/passwd']) reads the host file in an
            # unguarded child even though the top-level arg is a list, not a string.
            # A literal *[...] / *(...) starred arg is unpacked positionally, so scan its
            # elements too: open(*['/etc/passwd']) reads the same file open('/etc/passwd')
            # would, and os.open(*['/etc/shadow', os.O_RDONLY]) is otherwise opaque.
            _expanded = []
            for a in scan_args:
                inner = a.value if isinstance(a, ast.Starred) else a
                if isinstance(inner, (ast.List, ast.Tuple)):
                    _expanded.extend(inner.elts)
                else:
                    _expanded.append(inner)
            scan_args = _expanded
            for arg in scan_args:
                s = _fold_read_arg(arg)
                if s is None:
                    # A pathlib expression carries no foldable string constant
                    # (open(Path('/etc') / 'passwd')), so resolve it the same way a
                    # Path(...).read_text() receiver is resolved before skipping.
                    rp = _pathlib_receiver_path(arg)
                    if rp is not None and _flag_read_path(node, rp, is_read_callee):
                        break
                    # An opaque read path assembled from obfuscation primitives
                    # (open(''.join(map(chr, [...]))).read()) can still target a host
                    # secret, and reads are not runtime-confined. Apply the same
                    # fail-closed obfuscation policy exec payloads get: block a read
                    # callee whose path is built from chr/join(map)/decode/fetch/... .
                    if is_read_callee and _payload_has_obfuscation_primitive(arg):
                        _fs_block(node, "read path assembled from obfuscation primitives")
                        break
                    continue
                if _flag_read_path(node, s, is_read_callee):
                    break
                # Resolve a relative argv entry against a literal subprocess cwd= (cat passwd
                # + cwd='/etc' -> /etc/passwd) so the combined host-secret read is caught.
                if _sub_cwd is not None and not s.startswith("/") and not s.startswith("~"):
                    if _flag_read_path(node, os.path.join(_sub_cwd, s), is_read_callee):
                        break
            self.generic_visit(node)

    NetworkAndIoVisitor().visit(tree)

    if _analyzer_on:
        try:
            _SensitiveReadVisitor().visit(tree)
        except Exception:  # pragma: no cover - never crashier than legacy
            logger.warning("sandbox filesystem analyzer failed; skipping", exc_info = True)
            filesystem_violations.clear()

    is_safe = (
        len(signal_tampering) == 0
        and len(exception_catching) == 0
        and len(shell_escapes) == 0
        and len(dynamic_exec) == 0
        and len(network_calls) == 0
        and len(sensitive_file_reads) == 0
        and len(filesystem_violations) == 0
    )
    return is_safe, {
        "signal_tampering": signal_tampering,
        "exception_catching": exception_catching,
        "shell_escapes": shell_escapes,
        "dynamic_exec": dynamic_exec,
        "network_calls": network_calls,
        "sensitive_file_reads": sensitive_file_reads,
        "filesystem_violations": filesystem_violations,
        "warnings": warnings,
    }


def _check_code_safety(code: str) -> str | None:
    """Validate code safety via static analysis.

    Returns an error message string if the code is unsafe, or None if OK.
    """
    safe, info = _check_signal_escape_patterns(code)
    if not safe:
        # Let SyntaxError from ast.parse through so the subprocess produces a
        # normal Python traceback instead of a misleading "unsafe code" message.
        if info.get("error"):
            return None

        reasons = [item.get("description", "") for item in info.get("signal_tampering", [])]
        shell_reasons = [item.get("description", "") for item in info.get("shell_escapes", [])]
        exception_reasons = [
            item.get("description", "") for item in info.get("exception_catching", [])
        ]
        dynamic_reasons = [item.get("description", "") for item in info.get("dynamic_exec", [])]
        network_reasons = [item.get("description", "") for item in info.get("network_calls", [])]
        file_reasons = [
            item.get("description", "") for item in info.get("sensitive_file_reads", [])
        ]
        fs_reasons = [item.get("description", "") for item in info.get("filesystem_violations", [])]
        all_reasons = [
            r
            for r in reasons
            + shell_reasons
            + exception_reasons
            + dynamic_reasons
            + network_reasons
            + file_reasons
            + fs_reasons
            if r
        ]
        if all_reasons:
            return (
                f"Error: unsafe code detected ({'; '.join(all_reasons)}). "
                f"Please remove unsafe patterns from your code."
            )

    return None


def _kill_process_tree(proc) -> None:
    """SIGKILL the setsid process group; fall back to single-pid kill."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _cancel_watcher(
    proc,
    cancel_event,
    poll_interval = 0.2,
):
    """Daemon thread that kills a process when cancel_event is set."""
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            _kill_process_tree(proc)
            return
        cancel_event.wait(poll_interval) if cancel_event else None


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


# --------------------------------------------------------------------------
# Stage 5: runtime realpath backstop injected into sandboxed Python.
#
# This child-side guard resolves the true realpath (following symlinks) of every
# MUTATING file op and refuses it unless it lands inside the session workdir. It is
# the PRIMARY filesystem-write boundary: it sees dynamic paths, pre-existing
# symlinks, and library writers that funnel through builtins.open / io.open / os.open
# (numpy.save, torch.save, pandas.to_csv, savefig, ...). Reads are left unpatched
# here -- host-secret reads are handled by the static sensitive-read scanner. Skipped
# entirely under disable_sandbox (Bypass Permissions).
#
# Accepted residuals (a Python monkeypatch layer cannot close these; OS-level
# isolation is the real boundary): native-C writers that never call a patched Python
# entry point (cv2.imwrite, some pyarrow/zipfile writers), ctypes/cffi direct syscalls,
# and the realpath-before-open TOCTOU window under adversarial in-sandbox threading.
# --------------------------------------------------------------------------
_SANDBOX_GUARD_SRC = r"""
import sys as _sys
# The exec script lives INSIDE the workdir, so Python prepends the workdir to sys.path[0].
# A malicious workdir/os.py / io.py / pathlib.py / re.py (dropped by a prior run or upload)
# would otherwise shadow the guard's OWN imports below and execute unguarded at import time,
# before any patch is installed. Import the guard's stdlib deps with the workdir / cwd
# stripped from the path, then restore it so ordinary user imports still resolve (os / io /
# pathlib / re are now cached as the real, patched modules). `import sys` is safe: sys is a
# built-in module, never loaded from a file.
_saved_path = list(_sys.path)
_sys.path = [_p for _p in _sys.path if _p not in ("", ".", __WORKDIR__, __WORKDIR__ + "/")]
import os as _os, builtins as _bi, io as _io, pathlib as _pl, re as _re
# Pin the builtins the guard predicates consult (isinstance / int / bytes / str / any) into
# THIS namespace so a sandboxed `builtins.isinstance = lambda *a: True` (etc.) cannot make a
# guard check lie -- e.g. isinstance(path, int) treating an outside path as an fd and
# approving an absolute write. Every guard function below resolves these names from here, not
# the mutable builtins module.
isinstance = _bi.isinstance
int = _bi.int
bytes = _bi.bytes
str = _bi.str
any = _bi.any
# NOTE: sys.path stays stripped for the WHOLE guard setup below (it also imports shutil,
# which is pure-Python and equally shadowable); it is restored at the very END of this
# prelude, just before user code runs, so ordinary user imports still resolve.
# io + pathlib are imported BEFORE any patching on purpose: on Python <= 3.11
# pathlib._NormalAccessor captures io.open / os.* into class attributes at import
# time. A C builtin captured there does not bind on instance access, but a Python
# wrapper does (self shifts into the next arg), which would corrupt Path.open /
# Path.write_text. Importing here makes the accessor capture the originals; the
# confinement is applied on the public os / io / Path.* APIs below instead.
# Capture the path helpers/separator into IMMUTABLE guard-local names BEFORE user code
# runs. _within() otherwise reads os.path.realpath / os.fspath / os.sep off the live
# module every call, so sandboxed code could reassign os.path.realpath (e.g. to a lambda
# that echoes an in-workdir path) right before a write and have the guard approve an
# outside target while the real open() still writes there. These references cannot be
# rebound by mutating the os module.
_realpath = _os.path.realpath
_fspath = _os.fspath
_fsdecode = _os.fsdecode
_sep = _os.sep
# os.path.realpath resolves symlinks by consulting the LIVE os.lstat / os.readlink (and
# os.getcwd for relative paths). Capture the originals so a monkeypatch of any of them --
# e.g. os.lstat raising so realpath stops FOLLOWING an in-workdir symlink that points
# outside -- cannot make _within() approve a path the real open() then escapes through.
_lstat = _os.lstat
_readlink = _os.readlink
_getcwd = _os.getcwd
_stat = _os.stat
# posixpath.realpath decides whether to FOLLOW a component by calling os.path.stat.S_ISLNK
# on the live stat module. Sandboxed code can set os.path.stat.S_ISLNK = lambda mode: False
# (or reassign os.path.stat) so realpath stops following an in-workdir symlink that escapes,
# leaving the target under _WD while the real open() follows it outside. Capture the module +
# S_ISLNK so both can be re-pinned before each resolution.
_stat_mod = _os.path.stat
_S_ISLNK = _stat_mod.S_ISLNK
_WD = _realpath(__WORKDIR__)
# Standard device sinks cannot persist data outside the workspace, so a write to one is
# allowed (mirrors the terminal shell redirect allowlist); benign patterns like
# open('/dev/null', 'w') to suppress output would otherwise be denied by the workdir check.
_SAFE_DEV_SINKS = frozenset(
    {"/dev/null", "/dev/zero", "/dev/full", "/dev/stdout", "/dev/stderr", "/dev/tty"}
)

def _within(p):
    try:
        if isinstance(p, int):
            return True
        # Allow a write to an exact device sink. Checked on the REQUESTED path, not its
        # realpath, so /dev/stdout is not followed to a redirected outside file. Normalize
        # via the base str.replace (not p.replace): a str subclass could override replace()
        # to return "/dev/null" while its real value is an outside file, so call the genuine
        # method on the underlying buffer, which yields a plain str immune to the override.
        _ps = p if isinstance(p, str) else (_fsdecode(p) if isinstance(p, (bytes, bytearray)) else None)
        if _ps is not None and str.replace(_ps, "\\", "/") in _SAFE_DEV_SINKS:
            return True
        # os.path.realpath internally calls the LIVE os.fspath (posixpath.realpath does
        # `filename = os.fspath(filename)`) and os.lstat / os.readlink / os.getcwd, so a
        # sandboxed reassignment of any of them would poison the resolution even though we
        # hold the original realpath. Re-pin them to the captured originals before
        # resolving; the target then resolves truthfully (symlinks followed, cwd honest).
        # Re-pinning per check keeps it self-healing if user code re-patches; the real
        # open() receives the already-materialized str and does not route through these,
        # so restoring them has no effect on the write itself.
        _os.fspath = _fspath
        _os.lstat = _lstat
        _os.readlink = _readlink
        _os.getcwd = _getcwd
        _os.stat = _stat
        _os.path.stat = _stat_mod
        _stat_mod.S_ISLNK = _S_ISLNK
        rp = _realpath(_fspath(p))
        # A bytes path resolves to bytes; normalize to str so the prefix compare against
        # the str _WD does not raise (which would deny a legitimate in-workdir bytes write
        # such as open(b'local.txt', 'w')).
        if isinstance(rp, bytes):
            rp = _fsdecode(rp)
    except Exception:
        return False
    return rp == _WD or rp.startswith(_WD + _sep)

def _deny(p, what):
    raise PermissionError(
        "sandbox: %s outside the session workdir is not permitted: %r" % (what, p)
    )

def _gwraps(real):
    # Like functools.wraps but WITHOUT publishing __wrapped__: functools.wraps stores
    # the ORIGINAL unguarded callable on w.__wrapped__, and sandboxed code could reach
    # it (builtins.open.__wrapped__('/etc/x', 'w'), os.rename.__wrapped__(...)) to call
    # straight through every confinement below. Copy only the cosmetic metadata.
    def _deco(w):
        for _a in ("__module__", "__name__", "__qualname__", "__doc__"):
            try:
                setattr(w, _a, getattr(real, _a))
            except Exception:
                pass
        return w
    return _deco

def _fspath1(p):
    # Materialize a path-like ONCE so a stateful __fspath__ cannot return a workdir
    # path for the _within() check and a different path for the real syscall (TOCTOU).
    # Uses the captured _fspath so a reassigned os.fspath cannot interpose here.
    if isinstance(p, int):
        return p
    try:
        return _fspath(p)
    except Exception:
        return p

def _mode_is_write(mode):
    # Coerce through the *base* str: a str-subclass __contains__/__str__ must not be
    # able to lie about whether the mode requests a write.
    m = str.__str__(mode) if isinstance(mode, str) else "r"
    return any(c in m for c in "wax+")

# Runtime sensitive-read backstop. The static scanner cannot fold every read path
# (open(globals()['x']), open(fetch_name()), open(''.join(...))), and reads are otherwise
# unconfined, so an opaque path could name a host secret. Deny a read whose REALPATH
# resolves to a known-sensitive host file OUTSIDE the workdir. In-workdir files are the
# sandbox's own and always allowed. The loose 'credentials' / '.pem' / '/root/' signals the
# static layer uses are intentionally NOT applied here: importing common libraries reads
# site-packages files such as google/auth/credentials.py and certifi/cacert.pem (and, under
# a root home, /root/.local/.../site-packages), so matching them at runtime would break
# imports. The specific SSH / cloud / kube / netrc / HF-token signals stay.
_SENS_EXACT = frozenset({
    "/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/gshadow", "/etc/master.passwd",
})
_SENS_DIRS = (
    "/etc/ssh/", "/.ssh/", "/.aws/", "/.config/gcloud", "/.kube/", "/.docker/",
    "/var/run/secrets/kubernetes.io/", "/run/secrets/kubernetes.io/",
)
_SENS_TOKENS = (
    "id_rsa", "id_ed25519", ".netrc", ".git-credentials", "/.huggingface/token", ".kube/config",
)
_SENS_PROC = _re.compile(r"^/proc/(?:self|\d+)/(?:environ|cmdline|maps|mem|task/\d+/environ)$")

def _is_sensitive_read(rp):
    n = rp.replace("\\", "/")
    if n in _SENS_EXACT:
        return True
    # _SENS_DIRS entries carry a trailing slash to match a file UNDER the dir
    # (/root/.ssh/id_rsa). Append one to n so the sensitive directory ITSELF
    # (os.listdir('/root/.ssh') -> '/root/.ssh', no trailing slash) matches too.
    if any(part in (n + "/") for part in _SENS_DIRS):
        return True
    if _SENS_PROC.match(n):
        return True
    # Dotfiles / caches under a root home hold credentials (/root/.bashrc, /root/.cache/...);
    # an opaque path the static /root/ rule cannot fold could read them at runtime. Restore
    # the /root/ protection here (including the root home ITSELF, /root, which a directory
    # reader would enumerate), but carve out package / library trees so importing a library
    # installed under a root home (site-packages, the stdlib) is not broken.
    if (n == "/root" or n.startswith("/root/")) and not any(
        _seg in n
        for _seg in ("/site-packages/", "/dist-packages/", "/lib/python", "/lib64/python")
    ):
        return True
    low = n.lower()
    return any(tok in low for tok in _SENS_TOKENS)

def _read_realpath(p):
    # Resolve to a truthful realpath the same self-healing way _within does, so a
    # sandboxed reassignment of os.fspath / os.lstat / os.readlink / os.getcwd cannot
    # poison the resolution.
    try:
        _os.fspath = _fspath
        _os.lstat = _lstat
        _os.readlink = _readlink
        _os.getcwd = _getcwd
        _os.stat = _stat
        _os.path.stat = _stat_mod
        _stat_mod.S_ISLNK = _S_ISLNK
        rp = _realpath(_fspath(p))
        if isinstance(rp, bytes):
            rp = _fsdecode(rp)
        return rp
    except Exception:
        return None

def _deny_sensitive_read(p):
    if isinstance(p, int):
        return
    rp = _read_realpath(p)
    if rp is None:
        return
    # In-workdir files are the sandbox's own; never treat them as host secrets.
    if rp == _WD or rp.startswith(_WD + _sep):
        return
    if _is_sensitive_read(rp):
        raise PermissionError(
            "sandbox: reading a sensitive host path is not permitted: %r" % (rp,)
        )

def _guard_open_like(real):
    @_gwraps(real)
    def w(file, mode="r", *a, **k):
        f = _fspath1(file)
        if _mode_is_write(mode):
            if not _within(f):
                _deny(f, "write")
        else:
            _deny_sensitive_read(f)
        return real(f, mode, *a, **k)
    return w

_bi.open = _guard_open_like(_bi.open)

# Low-level os.open: builtins.open does not route through it, so it needs its own
# guard. Any mutating open flag confines the target; ANY dir_fd call (read or write) fails
# closed -- a string realpath against cwd is wrong for an fd-relative path, and a read-only
# dir_fd open can still read a host file under a directory fd opened outside the workdir
# (d = os.open('/etc', O_RDONLY); os.open('passwd', O_RDONLY, dir_fd=d)).
_WRITE_OFLAGS = (
    getattr(_os, "O_WRONLY", 0) | getattr(_os, "O_RDWR", 0)
    | getattr(_os, "O_CREAT", 0) | getattr(_os, "O_TRUNC", 0) | getattr(_os, "O_APPEND", 0)
)
def _make_osopen_guard(real_open):
    @_gwraps(real_open)
    def _guarded(path, flags, *a, **k):
        if k.get("dir_fd") is not None:
            _deny(path, "os.open (dir_fd)")
        try:
            fi = int.__index__(flags)  # base int: an int-subclass __and__ must not lie
        except Exception:
            fi = None
        mutating = (fi is None) or bool(fi & _WRITE_OFLAGS)
        if mutating:
            p = _fspath1(path)
            if not _within(p):
                _deny(p, "os.open write")
            return real_open(p, flags, *a, **k)
        # Read-only os.open: reads are unconfined, but a host secret is still off limits.
        p = _fspath1(path)
        _deny_sensitive_read(p)
        return real_open(p, flags, *a, **k)
    return _guarded
_os.open = _make_osopen_guard(_os.open)

def _wrap1(mod, name, what):
    orig = getattr(mod, name, None)
    if orig is None:
        return
    @_gwraps(orig)
    def w(*a, **k):
        # The path may be positional OR a public keyword: os.mkdir(path=...),
        # os.makedirs(name=...), os.removedirs(name=...). Extract it from whichever slot it
        # arrived in so a keyword call is not broken (missing positional) while still confined.
        _pk = None
        if a:
            path = a[0]
        elif "path" in k:
            path, _pk = k["path"], "path"
        elif "name" in k:
            path, _pk = k["name"], "name"
        else:
            return orig(*a, **k)  # let the original raise its own TypeError
        if any(k.get(_f) is not None for _f in ("dir_fd", "src_dir_fd", "dst_dir_fd")):
            _deny(path, what + " (dir_fd)")  # fd-relative target: a realpath check is meaningless
        if isinstance(path, int):
            # A mutating op given an fd (os.chmod(fd), os.truncate(fd), ...) can hit a
            # file opened read-only outside the workdir; a string realpath cannot
            # confine an fd, so deny it (fchmod/fchown are already denied separately).
            _deny(path, what + " (fd)")
        p = _fspath1(path)
        if not _within(p):
            _deny(p, what)
        # Pass the MATERIALIZED path back in the same slot it arrived (a stateful __fspath__
        # cannot then return a different outside path to the real call).
        if a:
            return orig(p, *a[1:], **k)
        k = dict(k)
        k[_pk] = p
        return orig(*a, **k)
    setattr(mod, name, w)

# Path-first single-arg mutators. mkfifo/utime/setxattr/removexattr create or mutate
# host files/metadata; the platform-specific ones no-op via _wrap1 when absent.
_OS_MUTATORS1 = (
    "remove", "unlink", "rmdir", "removedirs", "truncate", "chmod", "chown",
    "mkdir", "makedirs", "mknod", "mkfifo", "utime", "setxattr", "removexattr",
    "lchmod", "lchown", "chflags", "lchflags",
)
for _n in _OS_MUTATORS1:
    _wrap1(_os, _n, _n)

# Directory readers (os.listdir / os.scandir) enumerate a directory's names/entries
# WITHOUT routing through open(), so a sensitive host directory whose path is opaque to
# the static scanner (P = globals()['P']; os.listdir(P)) would leak its contents past the
# open-like backstop. Apply the same sensitive-read check to the directory path. A bare
# call (cwd), in-workdir paths, and an fd argument (os.open already screens the fd's read)
# stay allowed.
def _guard_dir_reader(name, mod=_os):
    orig = getattr(mod, name, None)
    if orig is None:
        return
    @_gwraps(orig)
    def w(path=".", *a, **k):
        if isinstance(path, int):
            return orig(path, *a, **k)
        # Materialize the path ONCE and pass that same value to the real call, so a stateful
        # __fspath__ cannot return an in-workdir path for the check and a sensitive outside
        # directory for the real listdir/scandir.
        p = _fspath1(path)
        _deny_sensitive_read(p)
        return orig(p, *a, **k)
    setattr(mod, name, w)

for _n in ("listdir", "scandir"):
    _guard_dir_reader(_n)

def _wrap2(mod, name, both):
    orig = getattr(mod, name, None)
    if orig is None:
        return
    @_gwraps(orig)
    def w(src, dst, *a, **k):
        if any(k.get(_f) is not None for _f in ("dir_fd", "src_dir_fd", "dst_dir_fd")):
            _deny(dst, name + " (dir_fd)")  # fd-relative target: a realpath check is meaningless
        s, d = _fspath1(src), _fspath1(dst)
        if both and not _within(s):
            _deny(s, name + " source")
        if not _within(d):
            _deny(d, name + " destination")
        return orig(s, d, *a, **k)
    setattr(mod, name, w)

for _n in ("rename", "renames", "replace", "link", "symlink"):
    _wrap2(_os, _n, True)

# posix (POSIX) / nt (Windows) is the low-level C module os re-exports from; patching
# os.* leaves posix.open / posix.rename / ... importable with the originals, so apply
# the same guards to that module too.
for _lowosname in ("posix", "nt"):
    try:
        _lowos = __import__(_lowosname)
    except Exception:
        _lowos = None
    if _lowos is not None:
        try:
            if hasattr(_lowos, "open"):
                _lowos.open = _make_osopen_guard(_lowos.open)
            for _n in _OS_MUTATORS1:
                _wrap1(_lowos, _n, _lowosname + "." + _n)
            for _n in ("rename", "renames", "replace", "link", "symlink"):
                _wrap2(_lowos, _n, True)
        except Exception:
            pass

# io.open is a separate reference from the (now patched) builtins.open -- guard
# direct io.open() writers (e.g. zipfile-based) the same way. (Path.open is handled
# explicitly below, not via this patch.)
try:
    _io.open = _guard_open_like(_io.open)
except Exception:
    pass

# The low-level C module _io is where io.open / builtins.open originate; patching the
# io alias above leaves _io.open untouched, so `import _io; _io.open(p, 'w')` would
# escape. Guard the underlying entry point too.
try:
    import _io as _lowio
    _lowio.open = _guard_open_like(_lowio.open)
except Exception:
    _lowio = None

# io.FileIO / _io.FileIO is a C constructor that opens a file WITHOUT routing through
# open(), so `io.FileIO('/tmp/escape', 'w')` bypasses _guard_open_like. Subclass it to
# confine mutating modes (subclassing keeps guard-built objects real FileIO instances).
def _guard_fileio(_realcls):
    class _GuardedFileIO(_realcls):
        def __init__(self, name, mode="r", *a, **k):
            f = _fspath1(name)
            if _mode_is_write(mode):
                if not _within(f):
                    _deny(f, "FileIO write")
            else:
                _deny_sensitive_read(f)
            # Pass the MATERIALIZED path so a stateful __fspath__ cannot return a
            # different (outside) path to the real constructor than we checked.
            super().__init__(f, mode, *a, **k)
    return _GuardedFileIO

for _iomod in (_io, _lowio):
    try:
        if _iomod is not None and hasattr(_iomod, "FileIO"):
            _iomod.FileIO = _guard_fileio(_iomod.FileIO)
    except Exception:
        pass

# The guards above patch the EXISTING posix / nt / _io module objects, but sandboxed code
# can mint a FRESH copy with the original unwrapped C functions via
# _imp.create_builtin(posix.__spec__) (or the BuiltinImporter path) and call its open()
# directly. Wrap _imp.create_builtin / create_dynamic so a freshly created guard-relevant
# module (posix / nt with os.open-style open + mutators, _io / io with open + FileIO) gets
# the same wrappers re-applied before it is handed back. Other builtin modules carry no file
# primitives, so they pass through unchanged and ordinary lazy imports keep working.
def _reguard_created(m):
    try:
        _nm = getattr(m, "__name__", "") or ""
    except Exception:
        return m
    try:
        if _nm in ("posix", "nt"):
            if hasattr(m, "open"):
                m.open = _make_osopen_guard(m.open)
            for _rn in _OS_MUTATORS1:
                _wrap1(m, _rn, _nm + "." + _rn)
            for _rn in ("rename", "renames", "replace", "link", "symlink"):
                _wrap2(m, _rn, True)
            # A fresh posix/nt module also re-exposes the ORIGINAL fd metadata mutators and
            # directory readers; reapply the same fd deniers + read confinement applied to the
            # already-loaded module (else fresh fchmod(fd, ...) / fresh listdir('/root') slip).
            if hasattr(m, "chdir"):
                _wrap1(m, "chdir", _nm + ".chdir")
            for _rn in ("fchmod", "fchown"):
                if hasattr(m, _rn):
                    setattr(m, _rn, _make_fd_denier(_nm + "." + _rn, getattr(m, _rn)))
            for _rn in ("listdir", "scandir"):
                if hasattr(m, _rn):
                    _guard_dir_reader(_rn, m)
        elif _nm in ("_io", "io"):
            if hasattr(m, "open"):
                m.open = _guard_open_like(m.open)
            if hasattr(m, "FileIO"):
                m.FileIO = _guard_fileio(m.FileIO)
    except Exception:
        pass
    return m

try:
    import _imp as _lowimp

    def _guard_create(_orig):
        @_gwraps(_orig)
        def w(spec, *a, **k):
            return _reguard_created(_orig(spec, *a, **k))
        return w

    for _cn in ("create_builtin", "create_dynamic"):
        _co = getattr(_lowimp, _cn, None)
        if _co is not None:
            setattr(_lowimp, _cn, _guard_create(_co))
except Exception:
    pass

# Confine the current working directory: os.chdir to a dir outside the workdir would
# let a later relative write/read (which the static read scan treats as local) escape.
# os.fchdir takes an fd whose target we cannot cheaply realpath, so deny it outright.
_wrap1(_os, "chdir", "chdir")
try:
    _real_fchdir = _os.fchdir
    @_gwraps(_real_fchdir)
    def _guarded_fchdir(fd):
        _deny(fd, "fchdir")
    _os.fchdir = _guarded_fchdir
except Exception:
    pass

# fd-based metadata mutators operate on an already-open descriptor, so a read-only
# os.open of an outside file (allowed -- reads are not confined) could still be reused
# to mutate host metadata. Deny them; sandboxed compute has no need to chmod/chown by fd.
def _make_fd_denier(_name, _orig):
    @_gwraps(_orig)
    def _w(fd, *a, **k):
        _deny(fd, _name)
    return _w
for _n in ("fchmod", "fchown"):
    try:
        setattr(_os, _n, _make_fd_denier(_n, getattr(_os, _n)))
    except Exception:
        pass

# The low-level posix / nt modules re-export chdir / fchdir / fchmod / fchown with the
# ORIGINALS, so patching os.* leaves posix.chdir (cwd escape -> unconfined relative
# reads) and posix.fchmod / posix.fchown (host-metadata mutation on a read-only outside
# fd) reachable. Apply the same confinement / deniers to those module objects too.
for _lowosname in ("posix", "nt"):
    try:
        _lowos = __import__(_lowosname)
    except Exception:
        _lowos = None
    if _lowos is None:
        continue
    try:
        if hasattr(_lowos, "chdir"):
            _wrap1(_lowos, "chdir", _lowosname + ".chdir")
        if hasattr(_lowos, "fchdir"):
            _lowos.fchdir = _make_fd_denier(_lowosname + ".fchdir", _lowos.fchdir)
        for _n in ("fchmod", "fchown"):
            if hasattr(_lowos, _n):
                setattr(_lowos, _n, _make_fd_denier(_lowosname + "." + _n, getattr(_lowos, _n)))
        # posix.listdir / posix.scandir re-export the ORIGINAL enumerators, so the os.* dir
        # guard leaves them reachable (posix.listdir('/root')); apply the same sensitive-read
        # confinement to the low-level module.
        for _n in ("listdir", "scandir"):
            if hasattr(_lowos, _n):
                _guard_dir_reader(_n, _lowos)
    except Exception:
        pass

try:
    import shutil as _sh
    _wrap1(_sh, "rmtree", "rmtree")
    _wrap1(_sh, "chown", "chown")
    _wrap2(_sh, "move", True)
    for _n in ("copy", "copy2", "copyfile", "copytree", "copymode", "copystat"):
        _wrap2(_sh, _n, False)
except Exception:
    pass

try:
    # sqlite3.connect(database) CREATES / opens the DB file via the native _sqlite3 C
    # extension, not builtins.open, so the open-like realpath backstop never sees it and an
    # absolute / traversal / dynamically built path (sqlite3.connect(os.sep+'tmp/x.db'))
    # would write a persistent database outside the session workdir. Confine the database
    # path to the workdir at runtime; :memory: / an empty (private temp) / an in-memory URI
    # stay allowed. Both public bindings (sqlite3.connect and sqlite3.dbapi2.connect) are the
    # same re-exported _sqlite3.connect, so wrap once and reassign every reachable attribute.
    import sqlite3 as _sq3

    def _sqlite_uri_path(_body):
        # Resolve a file: URI body (already stripped of the 'file:' prefix) to the concrete path
        # SQLite opens, or None for an in-memory / private target. Strips a //authority and
        # percent-decodes the filename (SQLite decodes file:%2Ftmp%2Fx -> /tmp/x itself), using
        # the captured _bi.chr / _bi.int so a sandboxed rebind of chr/int cannot skew the decode.
        _pth, _, _params = _body.partition("?")
        if _pth == ":memory:" or _pth == "" or "mode=memory" in _params.lower():
            return None
        if _pth.startswith("//"):
            _slash = _pth.find("/", 2)
            _pth = _pth[_slash:] if _slash != -1 else ""
        return _re.sub("%([0-9A-Fa-f]{2})", lambda _m: _bi.chr(_bi.int(_m.group(1), 16)), _pth)

    def _sqlite_target_path(_db, _uri):
        # The concrete filesystem path to confine for a sqlite database argument, or None when it
        # never touches disk (:memory: / '' private temp / in-memory URI). A file: filename is
        # URI-decoded ONLY when uri mode is on; otherwise it is a literal filename.
        if isinstance(_db, str):
            if _db == ":memory:" or _db == "":
                return None
            if _uri and _db[:5].lower() == "file:":
                return _sqlite_uri_path(_db[5:])
            return _db
        return _db

    def _sqlite_path_ok(_db, _uri):
        _p = _sqlite_target_path(_db, _uri)
        return True if _p is None else _within(_p)

    def _make_sqlite_authorizer(_uri_on):
        # ATTACH DATABASE '<file>' and VACUUM ... INTO '<file>' create/open a file via the native
        # extension (NOT the wrapped connect / open), and both fire the SQLITE_ATTACH authorizer
        # action (24) with the target filename as arg1. Deny a target that escapes the workdir; an
        # in-workdir / :memory: / '' (temp) attach and every other action stay allowed.
        def _auth(_action, _a1, _a2, _dbname, _source):
            if _action == 24 and isinstance(_a1, str):
                _p = _sqlite_target_path(_a1, _uri_on)
                if _p is not None and not _within(_p):
                    return 1  # SQLITE_DENY
            return 0  # SQLITE_OK
        return _auth

    _SqliteConnBase = _sq3.Connection  # the original (unguarded) Connection class

    class _GuardedSqliteConnection(_SqliteConnBase):
        # A Connection subclass that (1) confines the database path AT CONSTRUCTION, so the direct
        # constructor forms sqlite3.Connection('/tmp/x') / _sqlite3.Connection(...) are guarded just
        # like connect(); and (2) makes the ATTACH / VACUUM INTO authorizer DURABLE:
        # set_authorizer(cb) composes the workdir confinement AHEAD of the caller's callback, and
        # set_authorizer(None) keeps the confinement -- so sandboxed code cannot drop the hook and
        # then ATTACH DATABASE '/tmp/escape.db' / VACUUM INTO an outside file via native code.
        def __init__(self, *a, **k):
            if a:
                _db = a[0]
            elif "database" in k:
                _db = k["database"]
            else:
                _db = None
            _uri = bool(k.get("uri", False))
            # Materialize a path-like once so a stateful __fspath__ cannot pass the check with an
            # in-workdir value and then hand sqlite a different outside path.
            if _db is not None and not isinstance(_db, (str, bytes)):
                _db = _fspath1(_db)
                if a:
                    a = (_db,) + tuple(a[1:])
                else:
                    k = dict(k)
                    k["database"] = _db
            if _db is not None and not _sqlite_path_ok(_db, _uri):
                _deny(_db, "sqlite3.Connection")
            _SqliteConnBase.__init__(self, *a, **k)
            self._sandbox_uri_on = _uri
            # Install the initial confinement authorizer through the durable override below.
            try:
                self.set_authorizer(None)
            except Exception:
                pass

        def set_authorizer(self, callback, *a, **k):
            _confine = _make_sqlite_authorizer(getattr(self, "_sandbox_uri_on", False))

            def _composed(_action, _a1, _a2, _dbname, _source):
                if _confine(_action, _a1, _a2, _dbname, _source) != 0:
                    return 1  # SQLITE_DENY -- an escaping ATTACH / VACUUM INTO target
                if callback is None:
                    return 0  # SQLITE_OK
                return callback(_action, _a1, _a2, _dbname, _source)

            # Route through the ORIGINAL C method (not the possibly-reassigned module attribute) so
            # the confinement is always reinstalled and this override cannot recurse.
            return _SqliteConnBase.set_authorizer(self, _composed, *a, **k)

    _guard_conn_cache = {}

    def _combined_guard_conn(_user):
        # A caller-supplied Connection factory is COMBINED with the guard subclass (guard methods
        # take MRO precedence) so the path confinement + durable authorizer still apply.
        _g = _guard_conn_cache.get(_user)
        if _g is None:
            try:
                _g = type("SandboxGuardedConnection", (_GuardedSqliteConnection, _user), {})
            except Exception:
                _g = _GuardedSqliteConnection
            _guard_conn_cache[_user] = _g
        return _g

    def _guard_sqlite_connect(_orig):
        @_gwraps(_orig)
        def w(*a, **k):
            # Force our guarded Connection subclass as the factory so the returned connection is
            # path-confined and carries the durable authorizer; a caller factory is combined in.
            _fac = k.get("factory")
            if _fac is None:
                k = dict(k)
                k["factory"] = _GuardedSqliteConnection
            elif not (isinstance(_fac, type) and issubclass(_fac, _GuardedSqliteConnection)):
                k = dict(k)
                k["factory"] = _combined_guard_conn(_fac)
            return _orig(*a, **k)
        return w

    _sq3_guarded_connect = _guard_sqlite_connect(_sq3.connect)
    _sq3.connect = _sq3_guarded_connect
    try:
        _sq3.dbapi2.connect = _sq3_guarded_connect
    except Exception:
        pass
    # Confine the direct constructor forms too (sqlite3.Connection('/tmp/escape.db') /
    # sqlite3.dbapi2.Connection / _sqlite3.Connection), which never go through connect(). Replacing
    # the module attribute with the guarded subclass keeps isinstance() working (it IS a Connection)
    # while routing construction through the confining __init__.
    _sq3.Connection = _GuardedSqliteConnection
    try:
        _sq3.dbapi2.Connection = _GuardedSqliteConnection
    except Exception:
        pass
    # The native _sqlite3 C extension still exposes the ORIGINAL connect / Connection, importable
    # directly (import _sqlite3; _sqlite3.connect('/tmp/escape.db') / _sqlite3.Connection(...)),
    # bypassing the bindings above. Wrap them too; module attribute assignment on a C extension is
    # allowed, but guard it in case a build disallows it.
    try:
        import _sqlite3 as _lowsq3

        _lowsq3.connect = _guard_sqlite_connect(_lowsq3.connect)
        _lowsq3.Connection = _GuardedSqliteConnection
    except Exception:
        pass
except Exception:
    pass

try:
    # Path.open("w"): wrap the public method directly (mode-aware). Version-robust
    # because pathlib's accessor holds the original io.open (captured at the top).
    _real_path_open = _pl.Path.open
    @_gwraps(_real_path_open)
    def _guarded_path_open(self, mode="r", *a, **k):
        # Coerce mode through the base str (a str-subclass __contains__ must not lie).
        if _mode_is_write(mode):
            if not _within(self):
                _deny(str(self), "Path.open")
        else:
            # A dynamically assembled Path (Path(globals()['P']).read_text()) has no literal
            # receiver for the static scanner. On Python <= 3.11 pathlib holds the ORIGINAL
            # io.open, so the io.open sensitive-read backstop would not fire for pathlib
            # reads; apply it here so Path reads are confined version-robustly. read_text /
            # read_bytes route through this same self.open(), so they are covered too.
            _deny_sensitive_read(self)
        return _real_path_open(self, mode, *a, **k)
    _pl.Path.open = _guarded_path_open

    def _wrapp(name, targ):
        orig = getattr(_pl.Path, name, None)
        if orig is None:
            return
        @_gwraps(orig)
        def w(self, *a, **k):
            if not _within(self):
                _deny(str(self), "Path." + name)
            if targ:
                # rename/replace/symlink_to/hardlink_to accept the target as the
                # `target=` keyword too; on Python <= 3.10 pathlib routes through the
                # accessor's ORIGINAL os.rename, so this wrapper is the only
                # confinement -- check the keyword as well as the positional arg.
                _t = a[0] if a else k.get("target")
                if _t is not None:
                    # Materialize the target once so a stateful __fspath__ cannot return
                    # an in-workdir path here and an outside one to the real call.
                    _tm = _fspath1(_t)
                    if not _within(_tm):
                        _deny(str(_tm), "Path." + name + " target")
                    if a:
                        a = (_tm,) + tuple(a[1:])
                    else:
                        k = dict(k)
                        k["target"] = _tm
            return orig(self, *a, **k)
        setattr(_pl.Path, name, w)
    for _n in ("write_text", "write_bytes", "unlink", "mkdir", "rmdir", "chmod", "touch"):
        _wrapp(_n, False)
    for _n in ("rename", "replace", "symlink_to", "hardlink_to"):
        _wrapp(_n, True)

    # Path.iterdir / glob / rglob enumerate a directory; a dynamically built receiver
    # (Path(globals()['P']).iterdir(), Path(P).glob('*')) has no literal path for the static
    # scanner and, on some CPython versions, routes through pathlib's captured original
    # os.scandir rather than the patched one, so screen the directory read on the RECEIVER dir.
    def _guard_path_dir_reader(_name):
        _real = getattr(_pl.Path, _name, None)
        if _real is None:
            return
        @_gwraps(_real)
        def w(self, *a, **k):
            _deny_sensitive_read(self)
            return _real(self, *a, **k)
        setattr(_pl.Path, _name, w)
    for _n in ("iterdir", "glob", "rglob"):
        _guard_path_dir_reader(_n)
except Exception:
    pass

# Gate workdir MODULE imports: user code may `import helper` a sibling .py it wrote, but that
# module's source was never seen by the static analyzer, so a planted workdir/evilmod.py could
# run os.system('cat /etc/passwd') / subprocess at import time in the guarded interpreter (the
# CHILD it spawns is unguarded). Install a meta-path finder that, for a module resolved FROM the
# workdir, parses its source and refuses the import if it reaches a command-execution sink or
# eval/exec/compile the runtime guard cannot confine. File reads/writes in the module are already
# runtime-guarded, and library imports (site-packages) are not workdir-sourced so they pass
# through untouched. (Direct sinks only; deeper obfuscation in a workdir module is an accepted
# residual -- OS isolation remains the real boundary.)
try:
    import ast as _gast
    import importlib.machinery as _gimach
    import importlib.util as _gimportutil
    _GUARD_WORKDIR_REAL = _os.path.realpath(__WORKDIR__)
    # Network-capable modules: a workdir helper that opens a socket / HTTP client bypasses the
    # static network policy (there is no runtime network backstop), so refuse importing one.
    # (urllib / http bare tops are left out -- urllib.parse etc. are benign, OS isolation remains
    # the boundary for the dotted network submodules.)
    _GUARD_NET_MODS = frozenset({
        "socket", "ssl", "ftplib", "smtplib", "telnetlib", "poplib", "imaplib", "nntplib",
        "requests", "httpx", "aiohttp", "urllib3", "pycurl", "websocket", "websockets", "paramiko",
    })
    # Network-capable stdlib SUBMODULES whose bare top (urllib / http / xmlrpc) is benign
    # (urllib.parse, http.cookies) but whose dotted form opens outbound connections the static
    # network policy never saw (urllib.request.urlopen, http.client.HTTPConnection). Matched on
    # the full dotted name so the benign siblings stay importable.
    _GUARD_NET_DOTTED = frozenset({
        "urllib.request", "urllib.robotparser", "http.client", "xmlrpc.client",
    })
    _GUARD_EXEC_ATTRS = frozenset({
        "system", "popen", "popen2", "popen3", "popen4", "startfile",
        "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
        "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "posix_spawn", "posix_spawnp",
    })
    _GUARD_EXEC_MODS = frozenset({"subprocess", "pty"})
    # Child-spawning methods of the exec modules above. A workdir helper that IMPORTS subprocess /
    # pty is already refused, but one that receives the module as an argument (def f(subprocess):
    # subprocess.run([...])) has no import to reject, so a call rooted at a receiver literally named
    # subprocess / pty (the injected module) is refused here regardless of import.
    _GUARD_EXEC_MOD_ATTRS = {
        "subprocess": frozenset(
            {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
        ),
        "pty": frozenset({"spawn", "fork"}),
    }
    # os / posix expose the exec-attr sinks (os.system, os.execv, ...); a sink attribute rooted at
    # one of these is a command-exec sink even without a direct call (x = os.system; x('id')).
    _GUARD_EXEC_RECEIVERS = frozenset({"os", "posix"})
    # Deserializers that run an attacker-controlled reduce payload (which can spawn an unguarded
    # child via posix.system in the reducer): pickle & friends, marshal, dill, cloudpickle,
    # jsonpickle. The malicious bytes live OUTSIDE this source, so a workdir helper that calls one
    # is refused. (yaml / torch / numpy have safe modes and are left to the top-level analyzer;
    # importing pickle for pickle.dumps stays allowed -- only the load sinks are refused.)
    _GUARD_DESER_MODS = frozenset(
        {"pickle", "_pickle", "cpickle", "marshal", "dill", "cloudpickle", "jsonpickle"}
    )
    _GUARD_DESER_ATTRS = frozenset({"loads", "load", "Unpickler", "decode"})
    # Native-code / dynamic-execution modules: a workdir helper importing one gets UNGUARDED native
    # syscalls (ctypes libc write bypassing the patched open/os.open) or runs source / files outside
    # the recursive analysis (runpy / code / codeop), so the import is refused too.
    _GUARD_NATIVE_MODS = frozenset({"ctypes", "_ctypes", "cffi", "runpy", "code", "codeop"})
    # Modules whose DYNAMIC import (importlib.import_module('subprocess')) re-obtains an otherwise
    # denied module without a literal `import` statement.
    _GUARD_IMPORT_DENIED = (
        _GUARD_EXEC_MODS
        | _GUARD_NET_MODS
        | _GUARD_NATIVE_MODS
        | _GUARD_EXEC_RECEIVERS
        | {"sys", "builtins", "importlib"}
    )
    # Introspection / frame gadget attributes that recover a runtime guard wrapper's ORIGINAL
    # unguarded callable (open.__closure__[0].cell_contents, frame.f_locals['real']) or walk to
    # os / builtins. Mirrors the top-level _GADGET_DUNDERS; refuse them in a workdir helper too.
    _GUARD_GADGET_ATTRS = frozenset({
        "__subclasses__", "__bases__", "__base__", "__globals__", "__builtins__",
        "__closure__", "cell_contents", "f_locals", "f_globals", "f_back", "f_builtins",
        "tb_frame", "tb_next", "gi_frame", "cr_frame", "ag_frame",
        "settrace", "setprofile", "_getframe", "_current_frames", "currentframe",
    })
    # sys attributes that reach the import machinery: mutating them removes the guard's import
    # vetter so a sibling `import evil` loads unscanned.
    _GUARD_IMPORT_MACHINERY = frozenset({"meta_path", "path_hooks", "path_importer_cache"})
    def _guard_attr_root(_v):
        # Base Name id of an attribute chain (os.path -> 'os'); None if not Name-rooted.
        while isinstance(_v, _gast.Attribute):
            _v = _v.value
        return _v.id if isinstance(_v, _gast.Name) else None
    def _guard_str_fold(_n):
        # A statically foldable string: a literal or a concatenation of literals ('ev' + 'al').
        if isinstance(_n, _gast.Constant) and isinstance(_n.value, str):
            return _n.value
        if isinstance(_n, _gast.BinOp) and isinstance(_n.op, _gast.Add):
            _l = _guard_str_fold(_n.left)
            _r = _guard_str_fold(_n.right)
            if _l is not None and _r is not None:
                return _l + _r
        return None
    def _guard_subscript_key(_sub):
        return _guard_str_fold(_sub.slice)
    def _guard_module_src_unsafe(_src):
        try:
            _tree = _gast.parse(_src)
        except _bi.BaseException:
            return True  # unparseable workdir module -> fail closed
        # Pre-pass: record os / posix import ALIASES (import os as o) so an aliased sink reference
        # that is only assigned (s = o.system) -- not directly called -- is still recognized.
        # Also record builtins aliases (import builtins as b) so the execution builtins reached
        # as an attribute (builtins.eval / b.exec) are recognized alongside the bare names.
        _recv = set(_GUARD_EXEC_RECEIVERS)
        _bi = {"builtins", "__builtins__"}
        _deser = set(_GUARD_DESER_MODS)
        _sysmod = {"sys"}
        _implib = {"importlib"}
        for _nd in _gast.walk(_tree):
            if isinstance(_nd, _gast.Import):
                for _al in _nd.names:
                    if _al.name in ("os", "posix"):
                        _recv.add(_al.asname or _al.name)
                    elif _al.name == "builtins":
                        _bi.add(_al.asname or _al.name)
                    elif _al.name in _GUARD_DESER_MODS:
                        _deser.add(_al.asname or _al.name)
                    elif _al.name == "sys":
                        _sysmod.add(_al.asname or _al.name)
                    elif _al.name == "importlib":
                        _implib.add(_al.asname or _al.name)
        # Modules whose dynamic attribute / namespace-dict access (getattr / vars) is obfuscation.
        _obf = _recv | _bi | _deser | _sysmod | _implib
        for _nd in _gast.walk(_tree):
            if isinstance(_nd, _gast.Import):
                for _al in _nd.names:
                    _top = _al.name.split(".")[0]
                    if (
                        _top in _GUARD_EXEC_MODS
                        or _top in _GUARD_NET_MODS
                        or _top in _GUARD_NATIVE_MODS
                    ):
                        return True
                    # import urllib.request / import http.client -- benign top, network submodule.
                    if _al.name in _GUARD_NET_DOTTED:
                        return True
            elif isinstance(_nd, _gast.ImportFrom):
                _mod = _nd.module or ""
                _mroot = _mod.split(".")[0]
                if (
                    _mroot in _GUARD_EXEC_MODS
                    or _mroot in _GUARD_NET_MODS
                    or _mroot in _GUARD_NATIVE_MODS
                ):
                    return True
                # from urllib.request import urlopen -- the module itself is a network submodule.
                if _mod in _GUARD_NET_DOTTED:
                    return True
                # from urllib import request / from http import client -- the network submodule is
                # bound by name, so the dotted target is (package + . + imported name).
                for _al in _nd.names:
                    if (_mod + "." + _al.name) in _GUARD_NET_DOTTED:
                        return True
                # `from os import system` / `from os import *` binds a BARE sink name into the
                # module namespace; a later bare system('id') call has no os. attribute to catch.
                if _mroot in _GUARD_EXEC_RECEIVERS:
                    for _al in _nd.names:
                        if _al.name == "*" or _al.name in _GUARD_EXEC_ATTRS:
                            return True
                # `from pickle import loads` / `from pickle import *` binds a bare deserializer
                # sink; a later bare loads(evil) has no module attribute to catch. (dumps stays
                # allowed -- only the load sinks are refused.)
                if _mroot in _GUARD_DESER_MODS:
                    for _al in _nd.names:
                        if _al.name == "*" or _al.name in _GUARD_DESER_ATTRS:
                            return True
            elif isinstance(_nd, _gast.Call):
                # An invocation of an os / posix command-exec sink (os.system(...), an aliased
                # o.system(...), os.execv / os.posix_spawn) spawns an UNGUARDED child. Root it at
                # an os / posix receiver -- like the sink-attribute REFERENCE check below -- so a
                # benign same-named call on an unrelated object (platform.system(), a workdir
                # helper's own obj.system() method, df.eval()) is not misread as a shell escape.
                if (
                    isinstance(_nd.func, _gast.Attribute)
                    and _nd.func.attr in _GUARD_EXEC_ATTRS
                    and _guard_attr_root(_nd.func.value) in _recv
                ):
                    return True
                # A subprocess / pty child-spawn (subprocess.run([...]) / pty.spawn(...)) rooted at
                # a receiver literally named subprocess / pty. A helper that IMPORTS these is already
                # refused above; this catches the dependency-injected form (def f(subprocess):
                # subprocess.run(...)) that has no import statement to reject.
                if isinstance(_nd.func, _gast.Attribute):
                    _mroot = _guard_attr_root(_nd.func.value)
                    if (
                        _mroot in _GUARD_EXEC_MOD_ATTRS
                        and _nd.func.attr in _GUARD_EXEC_MOD_ATTRS[_mroot]
                    ):
                        return True
                if isinstance(_nd.func, _gast.Name) and _nd.func.id in (
                    "eval", "exec", "compile", "__import__"):
                    return True
                # builtins.eval(...) / b.exec(...) -- the execution builtins reached as an
                # attribute of the builtins module (or an alias). Require a builtins root so a
                # benign .compile()/.eval() on some other object (model.compile, df.eval) is
                # not misread as a sink.
                if (
                    isinstance(_nd.func, _gast.Attribute)
                    and _nd.func.attr in ("eval", "exec", "compile", "__import__")
                    and _guard_attr_root(_nd.func.value) in _bi
                ):
                    return True
                # A deserializer call (pickle.loads / marshal.load / dill.load /
                # pickle.Unpickler(f) / jsonpickle.decode) runs an attacker-controlled reduce
                # payload whose bytes live outside this source, so refuse it -- rooted at a
                # deserializer module / alias so a benign json.load / config.load is untouched.
                if (
                    isinstance(_nd.func, _gast.Attribute)
                    and _nd.func.attr in _GUARD_DESER_ATTRS
                    and _guard_attr_root(_nd.func.value) in _deser
                ):
                    return True
                # importlib.import_module('subprocess') / importlib.reload(subprocess) dynamically
                # re-obtain a denied module without a literal `import`. Refuse when the target is a
                # denied module (constant name or module Name); a dynamic import_module target
                # (non-constant) fails closed.
                if (
                    isinstance(_nd.func, _gast.Attribute)
                    and _nd.func.attr in ("import_module", "reload")
                    and _guard_attr_root(_nd.func.value) in _implib
                    and _nd.args
                ):
                    _a0 = _nd.args[0]
                    if isinstance(_a0, _gast.Constant) and isinstance(_a0.value, str):
                        if _a0.value.split(".")[0] in _GUARD_IMPORT_DENIED:
                            return True
                    elif isinstance(_a0, _gast.Name) and _a0.id in _GUARD_IMPORT_DENIED:
                        return True
                    elif _nd.func.attr == "import_module":
                        return True  # dynamic import target -> fail closed
                # getattr(os, 'system')(...) / getattr(sys, 'meta_path') / vars(sys)['meta_path']
                # -- dynamic attribute / namespace-dict access is the obfuscated twin of the direct
                # sink (the name-based checks above never see it). A constant sink name on a sink
                # receiver is refused; a NON-constant name on such a receiver, and vars() of one,
                # are refused too (the attribute cannot be proven benign).
                if (
                    isinstance(_nd.func, _gast.Name)
                    and _nd.func.id == "getattr"
                    and len(_nd.args) >= 2
                    and isinstance(_nd.args[0], (_gast.Name, _gast.Attribute))
                ):
                    _grecv = _guard_attr_root(_nd.args[0])
                    _gname = (
                        _nd.args[1].value
                        if isinstance(_nd.args[1], _gast.Constant)
                        and isinstance(_nd.args[1].value, str)
                        else None
                    )
                    if _gname is None:
                        if _grecv in _obf:
                            return True
                    else:
                        # An introspection / frame gadget dunder via getattr reaches an escape on
                        # ANY receiver -- getattr(open, '__closure__'), getattr(cell,
                        # 'cell_contents') recover the guard wrapper's original unguarded open --
                        # so reject the gadget name regardless of receiver (mirrors the direct
                        # attribute check below).
                        if _gname in _GUARD_GADGET_ATTRS:
                            return True
                        if _grecv in _recv and _gname in _GUARD_EXEC_ATTRS:
                            return True
                        if _grecv in _bi and _gname in ("eval", "exec", "compile", "__import__"):
                            return True
                        if _grecv in _deser and _gname in _GUARD_DESER_ATTRS:
                            return True
                        if _grecv in _sysmod and _gname in _GUARD_IMPORT_MACHINERY:
                            return True
                        if _grecv in _implib and _gname in (
                            "import_module", "reload", "__import__"):
                            return True
                # vars(sys) / vars(os) / vars(builtins) exposes the module namespace dict for
                # indirect access (vars(sys)['meta_path'][:] = [...], vars(os)['system']).
                if (
                    isinstance(_nd.func, _gast.Name)
                    and _nd.func.id == "vars"
                    and len(_nd.args) == 1
                    and isinstance(_nd.args[0], (_gast.Name, _gast.Attribute))
                    and _guard_attr_root(_nd.args[0]) in _obf
                ):
                    return True
            elif isinstance(_nd, _gast.Subscript):
                # __builtins__['eval'] / builtins.__dict__['exec'] -- imported helpers run with
                # __builtins__ as a dict, so subscript access reaches the execution builtins the
                # attribute checks miss. A constant exec-builtin key is refused; a NON-constant key
                # on a builtins receiver fails closed.
                _sroot = _guard_attr_root(_nd.value)
                if _sroot in _bi:
                    _skey = _guard_subscript_key(_nd)
                    if _skey is None:
                        return True
                    if _skey in ("eval", "exec", "compile", "__import__"):
                        return True
            elif isinstance(_nd, _gast.Attribute):
                # An introspection / frame gadget attribute (open.__closure__[0].cell_contents,
                # frame.f_locals['real'], ().__class__.__bases__[0].__subclasses__()) recovers a
                # runtime guard wrapper's original unguarded callable or walks to os / builtins.
                # These reach an escape on ANY receiver, so flag the attribute itself.
                if _nd.attr in _GUARD_GADGET_ATTRS:
                    return True
                # A sink-named attribute REFERENCE (even uncalled) rooted at os / posix / an
                # os alias (x = os.system, s = o.system). A same-named attribute on an unrelated
                # object (p.system = 'linux') is NOT a sink, so require a sink-module receiver.
                if _nd.attr in _GUARD_EXEC_ATTRS and _guard_attr_root(_nd.value) in _recv:
                    return True
                # A builtins-rooted execution-builtin REFERENCE (e = builtins.eval; e(...)),
                # even uncalled, is the same sink as calling it directly.
                if (
                    _nd.attr in ("eval", "exec", "compile", "__import__")
                    and _guard_attr_root(_nd.value) in _bi
                ):
                    return True
                # A workdir module that touches the import machinery (sys.meta_path /
                # sys.path_hooks / sys.path_importer_cache) can remove THIS vetter, then a
                # sibling `import evil` loads unscanned. The top-level analyzer blocks such
                # mutation in submitted code; refuse it inside a vetted workdir module too.
                if _nd.attr in ("meta_path", "path_hooks", "path_importer_cache"):
                    return True
        return False
    def _guard_under_workdir(_p):
        return _p == _GUARD_WORKDIR_REAL or _p.startswith(_GUARD_WORKDIR_REAL + _os.sep)

    class _GuardVettedSourceLoader:
        # Executes the EXACT source string the vetter scanned, so the loader can never satisfy
        # the import from a planted bytecode cache (.pyc) or re-decode the file differently than
        # it was vetted. __path__ for a package still comes from the spec's search locations.
        def __init__(self, _name, _path, _src, _is_pkg):
            self._n = _name
            self._p = _path
            self._s = _src
            self._pkg = _is_pkg

        def create_module(self, _spec):
            return None

        def exec_module(self, _module):
            exec(compile(self._s, self._p, "exec"), _module.__dict__)

        def get_filename(self, _name=None):
            return self._p

        def is_package(self, _name=None):
            return self._pkg

        def get_source(self, _name=None):
            return self._s

    class _GuardWorkdirImportVetter:
        def find_spec(self, _name, _path=None, _target=None):
            try:
                _spec = _gimach.PathFinder.find_spec(_name, _path, _target)
            except _bi.BaseException:
                return None
            _orig = getattr(_spec, "origin", None) if _spec is not None else None
            if not _orig:
                return None  # namespace / builtin / frozen: no file to vet, not workdir-sourced
            try:
                _abs = _os.path.abspath(_orig)
                _rp = _os.path.realpath(_orig)
            except _bi.BaseException:
                return None
            _orig_in_wd = _guard_under_workdir(_abs) or _guard_under_workdir(
                _os.path.dirname(_abs)
            )
            _rp_in_wd = _guard_under_workdir(_rp)
            if not _orig_in_wd and not _rp_in_wd:
                return None  # genuinely not a workdir module; let the default finders load it
            # A workdir-sourced origin whose REALPATH escapes the workdir (a symlink to an outside
            # file) must fail closed, not be handed to the default loader unvetted.
            if not _rp_in_wd:
                raise _bi.ImportError(
                    "sandbox: refusing symlinked workdir module " + _name)
            # A workdir module must be a .py we can read + scan. A sourceless .pyc / native .so /
            # any other non-source file under the workdir cannot be statically vetted, so refuse
            # it: a planted legacy evil.pyc would otherwise run its bytecode via the default
            # sourceless loader, never reaching the source scan below.
            if not _orig.endswith(".py"):
                raise _bi.ImportError(
                    "sandbox: refusing to import non-source workdir module " + _name)
            # Decode with Python's PEP 263 source encoding (importlib.util.decode_source), NOT a
            # fixed utf-8: a `# coding: utf_7` module the loader would decode as UTF-7 must be
            # vetted as UTF-7, or a payload hidden in what a UTF-8 scan sees as a comment runs.
            try:
                _fb = _io.open(_orig, "rb")
                try:
                    _raw = _fb.read()
                finally:
                    _fb.close()
                _msrc = _gimportutil.decode_source(_raw)
            except _bi.BaseException:
                raise _bi.ImportError("sandbox: cannot vet workdir module " + _name)
            if _guard_module_src_unsafe(_msrc):
                raise _bi.ImportError(
                    "sandbox: refusing to import unvetted workdir module " + _name)
            # Run the EXACT vetted source via our loader so a planted matching .pyc can never be
            # executed instead (the default SourceFileLoader would satisfy the import from a
            # __pycache__ .pyc whose header matches the harmless source).
            _is_pkg = _spec.submodule_search_locations is not None or _os.path.basename(
                _orig
            ) == "__init__.py"
            _spec.loader = _GuardVettedSourceLoader(_name, _orig, _msrc, _is_pkg)
            return _spec
    _sys.meta_path.insert(0, _GuardWorkdirImportVetter())
except _bi.BaseException:
    pass

# All guard dependencies are now imported (and cached as the real, patched modules) with the
# workdir kept off sys.path, so no workdir/*.py could shadow them. Restore the original path
# for user code so ordinary sibling imports still resolve (workdir modules are vetted above).
_sys.path = _saved_path
"""


def _sandbox_runtime_prelude(workdir: str) -> str:
    """One physical line that runs the realpath backstop before the user code.

    The guard executes in its own namespace (helper names never leak into user
    globals) while its monkeypatches persist on the os/shutil/pathlib/builtins
    module objects. Emitting it on a single line keeps user traceback line numbers
    shifted by exactly one."""
    src = _SANDBOX_GUARD_SRC.replace("__WORKDIR__", repr(workdir))
    return (
        "exec(compile(%r, '<studio-sandbox-guard>', 'exec'), {'__builtins__': __builtins__})\n"
        % src
    )


def _inject_sandbox_guard(code: str, prelude: str) -> str:
    """Splice the runtime guard into ``code`` without displacing leading directives.

    ``from __future__`` imports must be the first statement of a module (only a
    docstring and comments may precede them), so blindly prepending the guard line
    turns any user program that opens with a future import into a SyntaxError. A
    leading string literal is likewise the module docstring only while it is the FIRST
    statement, so prepending the guard ahead of it would make ``__doc__`` None. Parse
    the code, keep a leading module docstring and any ``from __future__`` imports on
    top, and insert the (inert compile-time) guard immediately after them -- it still
    runs before the first real user statement, so the sandbox is established before
    any file operation. Everything else (including unparsable code, where we want the
    natural SyntaxError traceback) falls back to a plain prepend.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return prelude + code
    body = getattr(tree, "body", [])
    idx = 0
    split = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        idx = 1
        split = body[0].end_lineno or 0
    while (
        idx < len(body)
        and isinstance(body[idx], ast.ImportFrom)
        and body[idx].module == "__future__"
    ):
        split = body[idx].end_lineno or split
        idx += 1
    # Splice after the head only when there is something that MUST stay first (a leading
    # docstring and/or future imports); otherwise a plain prepend is correct and cheaper.
    if idx == 0 or split <= 0:
        return prelude + code
    # Split at the last head statement's exact END COLUMN, not the whole physical line: a
    # `from __future__ import annotations; open('/tmp/x','w')` puts a real statement on the SAME
    # line as the future import, and a line-granular split would copy that write into the head
    # (before the guard prelude) and run it unguarded. Slice the head line at end_col_offset so the
    # `;`-separated tail moves AFTER the prelude, then drop the leading `; ` so the tail is a valid
    # statement. (Only future-import lines reach here -- pure ASCII -- so character slicing matches
    # the byte col_offset.)
    _last = body[idx - 1]
    _el = _last.end_lineno or split
    _ec = _last.end_col_offset or 0
    lines = code.splitlines(keepends = True)
    head = "".join(lines[: _el - 1]) + lines[_el - 1][:_ec]
    tail = lines[_el - 1][_ec:] + "".join(lines[_el:])
    _m = re.match(r"[ \t]*;[ \t]*", tail)
    if _m:
        tail = tail[_m.end() :]  # a same-line `; stmt` tail -> valid statement after the prelude
    if head and not head.endswith(("\n", "\r")):
        head += "\n"
    if prelude and not prelude.endswith(("\n", "\r")):
        prelude += "\n"
    return head + prelude + tail


def _python_exec(
    code: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
    disable_sandbox: bool = False,
) -> str:
    """Execute Python code in a subprocess sandbox.

    disable_sandbox (Bypass Permissions): skip the safety analysis and rlimit
    pre-exec, and use the host env minus secrets.
    """
    if not code or not code.strip():
        return "No code provided."

    # Validate imports and code safety (skipped when the sandbox is disabled)
    if not disable_sandbox:
        error = _check_code_safety(code)
        if error:
            return error
    elif not _harden_parent_against_proc_env_leak():
        # Close the /proc/<parent>/environ secret-recovery path first; if it
        # cannot be applied, fail closed rather than leak the parent environ.
        return (
            "Execution error: could not harden the Studio process against "
            "/proc environment reads; refusing bypass execution."
        )

    tmp_path = None
    workdir = _get_workdir(session_id)
    # Snapshot image mtimes to detect new and overwritten files.
    _before: dict[str, int] = {}
    if os.path.isdir(workdir):
        for _name in os.listdir(workdir):
            if os.path.splitext(_name)[1].lower() in _IMAGE_EXTS:
                _p = os.path.join(workdir, _name)
                if os.path.isfile(_p):
                    try:
                        _before[_name] = os.stat(_p).st_mtime_ns
                    except OSError:
                        pass
    try:
        fd, tmp_path = tempfile.mkstemp(suffix = ".py", prefix = "studio_exec_", dir = workdir)
        # utf-8 so non-ASCII in model-written code survives the OS default codec
        # (Windows cp1252 would otherwise raise UnicodeEncodeError).
        # Sandboxed runs get the realpath backstop prepended (Stage 5); bypass
        # runs execute the code verbatim.
        file_body = (
            code
            if disable_sandbox
            else _inject_sandbox_guard(code, _sandbox_runtime_prelude(workdir))
        )
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.write(file_body)

        safe_env = _build_bypass_env(workdir) if disable_sandbox else _build_safe_env(workdir)
        if disable_sandbox:
            # Match the sandboxed Python path without changing bypass shell I/O.
            safe_env = dict(safe_env)
            safe_env["PYTHONIOENCODING"] = "utf-8"
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            # Decode child output as utf-8 (it emits utf-8 via PYTHONIOENCODING);
            # replace so non-ASCII output never crashes the read on Windows.
            encoding = "utf-8",
            errors = "replace",
            cwd = workdir,
            env = safe_env,
        )
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _bypass_preexec if disable_sandbox else _sandbox_preexec
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        # -s disables the per-user site directory (~/.local/.../site-packages): the guard is
        # injected into the SCRIPT body and runs after site initialization, so a prior run that
        # dropped .local/.../site-packages/usercustomize.py (HOME points at the workdir) would
        # otherwise import it during startup and run writers with unpatched stdlib. Bypass keeps
        # the host default. The real site-packages stays available for user imports.
        _py_argv = (
            [sys.executable, tmp_path] if disable_sandbox else [sys.executable, "-s", tmp_path]
        )
        proc = subprocess.Popen(_py_argv, **popen_kwargs)

        # Spawn cancel watcher if we have a cancel event
        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, args = (proc, cancel_event), daemon = True
            )
            watcher.start()

        try:
            output, _ = proc.communicate(timeout = timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            try:
                proc.communicate(timeout = 5)
            except subprocess.TimeoutExpired:
                pass
            return _truncate(f"Execution timed out after {timeout} seconds.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        result = _truncate(result) if result.strip() else "(no output)"

        # Detect new/overwritten images and append sentinel for the frontend
        if session_id and os.path.isdir(workdir):
            new_images = []
            for _name in os.listdir(workdir):
                if os.path.splitext(_name)[1].lower() not in _IMAGE_EXTS:
                    continue
                _p = os.path.join(workdir, _name)
                if not os.path.isfile(_p):
                    continue
                try:
                    _mtime = os.stat(_p).st_mtime_ns
                except OSError:
                    continue
                if _name not in _before or _mtime != _before[_name]:
                    new_images.append(_name)
            if new_images:
                import json as _json
                result += f"\n__IMAGES__:{_json.dumps(sorted(new_images))}"

        return result

    except Exception as e:
        return f"Execution error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _bash_exec(
    command: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
    disable_sandbox: bool = False,
) -> str:
    """Execute a bash command in a subprocess sandbox.

    disable_sandbox (Bypass Permissions): skip the command blocklist and rlimit
    pre-exec, and use the host env minus secrets.
    """
    if not command or not command.strip():
        return "No command provided."

    # Block dangerous commands (skipped when the sandbox is disabled)
    if not disable_sandbox:
        blocked = _find_blocked_commands(command)
        if blocked:
            return f"Blocked command(s) for safety: {', '.join(sorted(blocked))}"
        # The command runs in an unguarded shell child, so the Python-tool open() backstop
        # does not confine its reads; refuse an embedded host-secret read the same way an
        # os.system('cat /etc/passwd') shell string is refused in the Python tool.
        _read = _command_reads_sensitive(command)
        if _read is not None:
            return f"Blocked command for safety: sensitive file read ({_read})"
    elif not _harden_parent_against_proc_env_leak():
        # Close the /proc/<parent>/environ secret-recovery path first; if it
        # cannot be applied, fail closed rather than leak the parent environ.
        return (
            "Execution error: could not harden the Studio process against "
            "/proc environment reads; refusing bypass execution."
        )

    try:
        workdir = _get_workdir(session_id)
        safe_env = _build_bypass_env(workdir) if disable_sandbox else _build_safe_env(workdir)
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            cwd = workdir,
            env = safe_env,
        )
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _bypass_preexec if disable_sandbox else _sandbox_preexec
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(_get_shell_cmd(command), **popen_kwargs)

        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, args = (proc, cancel_event), daemon = True
            )
            watcher.start()

        try:
            output, _ = proc.communicate(timeout = timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            try:
                proc.communicate(timeout = 5)
            except subprocess.TimeoutExpired:
                pass
            return _truncate(f"Execution timed out after {timeout} seconds.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        return _truncate(result) if result.strip() else "(no output)"

    except Exception as e:
        return f"Execution error: {e}"
