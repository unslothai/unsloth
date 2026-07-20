# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool definitions and executors for LLM tool calling: web search
(DuckDuckGo), Python code execution, and terminal commands."""

import ast
import codecs
import fnmatch
import http.client
import os
import signal

os.environ["UNSLOTH_IS_PRESENT"] = "1"

import asyncio
import queue
import random
import re
import shlex
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

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
from core.inference.sandbox_static_fs import (
    check_static_fs,
    host_pathmod,
    static_screen_enabled,
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


def _env_int(name: str, default: int) -> int:
    """Read an int env override; fall back to ``default`` on unset/garbage."""
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Model-visible cap on python/terminal tool results (protects the context
# window). The live UI stream is capped separately and higher, so _truncate's
# notice stays mode-neutral (see tool_stream_exec.TOOL_OUTPUT_STREAM_MAX_CHARS).
_MAX_OUTPUT_CHARS = _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000)
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
    }
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
_BLOCKED_COMMANDS = (
    _BLOCKED_COMMANDS_COMMON | _BLOCKED_COMMANDS_WIN
    if sys.platform == "win32"
    else _BLOCKED_COMMANDS_COMMON
)


_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "\n", "(", ")", "`", "{", "}"})
# Bash keywords starting a new command position (then $cmd, do $cmd, etc.).
_SHELL_KEYWORDS_AS_SEP = frozenset({"then", "do", "else", "elif"})
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
    }
)
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Env-assignment prefixes that change command lookup or code loading, so
# `LD_PRELOAD=x ls` / `PATH=. ls` run attacker code before the read-only
# utility. LD_*/DYLD_* and any *PATH are covered by the prefix/suffix check.
_AUTO_UNSAFE_ENV_ASSIGN = frozenset(
    {
        "IFS",
        "BASH_ENV",
        "ENV",
        "SHELLOPTS",
        "BASHOPTS",
        "GLOBIGNORE",
        "PROMPT_COMMAND",
        "PS4",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "PERL5OPT",
        "PERL5LIB",
        "RUBYOPT",
        "RUBYLIB",
        # LESSOPEN/LESSCLOSE run an input preprocessor command for less.
        "LESSOPEN",
        "LESSCLOSE",
    }
)


def _env_assignment_is_unsafe(name: str) -> bool:
    """True if a NAME=value prefix affects command lookup/loading."""
    return (
        name in _AUTO_UNSAFE_ENV_ASSIGN
        or name.startswith(("LD_", "DYLD_"))
        or name.endswith("PATH")
    )


_FIND_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


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

    # punctuation_chars splits separators into their own tokens, so command
    # position is detected even in `echo done; rm -rf x` (no whitespace).
    try:
        if sys.platform == "win32":
            tokens = shlex.split(command, posix = False)
        else:
            lexer = shlex.shlex(command, posix = True, punctuation_chars = ";&|()`")
            lexer.whitespace_split = True
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
    for token in tokens:
        if token in _SHELL_SEPARATORS or token in _SHELL_KEYWORDS_AS_SEP:
            expect_command = True
            prefix_pending = False
            continue
        if token.startswith("-"):
            # Flags belong to the active command, but keep expect_command while a
            # wrapper prefix awaits its command (`stdbuf -oL cmd`, `xargs -- cmd`).
            if not prefix_pending:
                expect_command = False
            continue
        if not expect_command:
            continue
        # FOO=bar assignment prefix; next non-assignment token is the command.
        if _ASSIGNMENT_RE.match(token):
            continue
        # Numeric wrapper arg: `timeout 1 cmd` / `nice -n 5 cmd`.
        if prefix_pending and token.lstrip("-").isdigit():
            continue
        base = _token_basename(token)
        if base in _BLOCKED_COMMANDS:
            blocked.add(base)
        # Wrappers (env/time/xargs/sudo) consume one command; the next non-flag,
        # non-numeric token is the real command. sudo is also in _BLOCKED_COMMANDS.
        if base in _COMMAND_PREFIXES:
            prefix_pending = True
            continue
        expect_command = False
        prefix_pending = False

    # `find ... -exec CMD ... ;` and `-execdir CMD ... ;` invoke CMD directly.
    for i, tok in enumerate(tokens):
        if tok in _FIND_EXEC_FLAGS and i + 1 < len(tokens):
            base = _token_basename(tokens[i + 1])
            if base in _BLOCKED_COMMANDS:
                blocked.add(base)

    # Regex catches blocked words at command boundaries shlex misses: inside
    # $(rm -rf), <(rm), backtick chains, or "foo;rm". Anchored to command-position
    # delimiters, so it doesn't match in argument position.
    lowered = command.lower()
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
    _SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish"}
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

    return blocked


# Directory holding the sandbox ``sitecustomize.py`` shim (code-interpreter
# path remap); placed on the sandboxed child's PYTHONPATH in _build_safe_env.
_SANDBOX_SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site")
# ── "Approve for me" (permission_mode="auto") safety detection ──────────────
# Auto mode pauses only calls classified here as potentially unsafe. The sandbox
# and hard blocks (blocklist, rlimits) still apply at run time; this gate only
# decides prompting, and fails closed: anything not provably read-only asks.

# Read-only commands allowed to run without confirmation in auto mode.
_AUTO_SAFE_TERMINAL_COMMANDS = frozenset(
    {
        "ls",
        "dir",
        "pwd",
        # cd absent: `cd /; cat etc/passwd` escapes the workdir for a later
        # relative read the path scan cannot see, so cd always asks.
        "cat",
        "head",
        "tail",
        # less/more absent: their pager escapes (+cmd, !shell, -o, LESSOPEN) can
        # run a command or write a file, so they always ask.
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "find",
        "fd",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "diff",
        "cmp",
        "file",
        "stat",
        "du",
        "df",
        # ps absent: BSD env flags (ps auxe, ps eww) dump a parent's unscrubbed
        # env and can't be flag-parsed reliably, so ps always asks.
        "date",
        "cal",
        "whoami",
        "id",
        "uname",
        "hostname",
        "uptime",
        "which",
        "whereis",
        "type",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "md5",
        "md5sum",
        "shasum",
        "sha1sum",
        "sha256sum",
        "cksum",
        "tree",
        "printenv",
        "echo",
        "printf",
        "true",
        "false",
        "test",
        "[",
        "seq",
        "nl",
        "od",
        "xxd",
        "hexdump",
        "strings",
        "column",
        "paste",
        "join",
        "comm",
        "expand",
        "unexpand",
        "fold",
        "fmt",
        "rev",
        "tac",
        "locale",
        "arch",
        "nproc",
        "sw_vers",
        "jq",
    }
)
# Flags that turn an otherwise read-only command into a writer or executor
# (sort -o FILE, tree -o FILE, xxd -r IN OUT, find -exec/-delete/...).
_AUTO_UNSAFE_COMMAND_FLAGS = {
    # --files0-from=F makes sort read the NUL-separated list of input files
    # named in F, so a crafted list reads arbitrary host files indirectly.
    "sort": frozenset(
        {"-o", "--output", "--compress-program", "-T", "--temporary-directory", "--files0-from"}
    ),
    "tree": frozenset({"-o"}),
    "xxd": frozenset({"-r"}),
    # -c/--check makes a checksum tool read a manifest file and then read every
    # path it names, so a manifest listing /etc/passwd turns `sha256sum -c list`
    # into an indirect host-file read; the digest form (sha256sum file) only reads
    # the named files.
    "md5sum": frozenset({"-c", "--check"}),
    "sha1sum": frozenset({"-c", "--check"}),
    "sha256sum": frozenset({"-c", "--check"}),
    "shasum": frozenset({"-c", "--check"}),
    "cksum": frozenset({"-c", "--check"}),
    # GNU time -o/--output/-a/--append FILE writes timing output; time is a
    # wrapper, so the flag is checked before the wrapped command like env -C.
    "time": frozenset({"-o", "--output", "-a", "--append"}),
    # rg runs an arbitrary program per file with --pre/--hostname-bin.
    "rg": frozenset({"--pre", "--hostname-bin"}),
    # env -C/--chdir escapes the workdir; -S/--split-string builds a command.
    "env": frozenset({"-C", "--chdir", "-S", "--split-string"}),
    # ionice -p/-P/-u change the I/O priority of an already running process /
    # group / user instead of forwarding to a wrapped read-only command, so a
    # bare `ionice -c 3 -p <pid>` mutates another process. ionice stays a safe
    # wrapper for `ionice -c 3 <cmd>`; only the process-target flags ask.
    "ionice": frozenset({"-p", "-P", "-u"}),
    # printf -v NAME assigns to a shell var, so `printf -v PATH %s .; ls` runs
    # ./ls from the workdir.
    "printf": frozenset({"-v"}),
    # wc/du/find --files0-from=F read the NUL-separated list of input paths named
    # in F, so a crafted list reads arbitrary host files past the literal path /
    # root checks, like sort --files0-from. find spells it -files0-from (a primary).
    "wc": frozenset({"--files0-from"}),
    "du": frozenset({"--files0-from"}),
    "find": frozenset(
        {
            "-exec",
            "-execdir",
            "-ok",
            "-okdir",
            "-delete",
            "-fprint",
            "-fprint0",
            "-fprintf",
            "-fls",
            "-files0-from",
        }
    ),
    # fd -x/--exec/-X/--exec-batch run a command per result;
    # --base-directory/--search-path move the search root outside the workdir.
    "fd": frozenset({"-x", "--exec", "-X", "--exec-batch", "--base-directory", "--search-path"}),
    # date -s/--set writes the clock; display forms (+FORMAT, -d/-u/-R/-r) read.
    "date": frozenset({"-s", "--set"}),
    # file -C/--compile writes a compiled .mgc magic database; ident forms read.
    "file": frozenset({"-C", "--compile"}),
    # hostname -F/--file, -b/--boot set the hostname; display flags only read.
    "hostname": frozenset({"-F", "--file", "-b", "--boot"}),
}
# Commands safe only without a mutating positional: `hostname NAME` sets the
# hostname, `date MMDDhhmm...` sets the clock (a +FORMAT token or a display
# flag's value stays read-only), so any other positional asks.
_AUTO_ARG_SENSITIVE_COMMANDS = frozenset({"hostname", "date"})
# date display flags taking a value token (-d STRING, -r FILE, -f FILE); the
# value is not a clock-setting positional, so it is skipped.
_DATE_DISPLAY_VALUE_FLAGS = frozenset({"-d", "--date", "-r", "--reference", "-f", "--file"})
# Commands that write their 2nd positional (uniq [INPUT [OUTPUT]], xxd [infile
# [outfile]]): the 1st file reads to stdout, but a second file positional
# overwrites it, like `sort -o`.
_AUTO_SECOND_POSITIONAL_WRITES = frozenset({"uniq", "xxd"})
# Value-taking option flags for those commands whose argument is a separate token
# (uniq -f 2, xxd -c 16). The value must be consumed so a numeric option value is
# not miscounted as the output-file positional, and, conversely, a file that is
# literally named with digits (uniq 123 out) is still counted.
_SECOND_POSITIONAL_VALUE_FLAGS = {
    "uniq": frozenset({"-f", "--skip-fields", "-s", "--skip-chars", "-w", "--check-chars"}),
    "xxd": frozenset(
        {"-c", "--cols", "-s", "--seek", "-l", "--len", "-g", "--groupsize", "-o", "--offset"}
    ),
}
# find/fd group with (...) which resets command context, so scan every token for
# these once find/fd appears anywhere.
_AUTO_UNSAFE_FIND_LIKE_FLAGS = _AUTO_UNSAFE_COMMAND_FLAGS["find"] | _AUTO_UNSAFE_COMMAND_FLAGS["fd"]
# Recursive readers with an absolute-path target escape the workdir onto host
# files (grep -R TOKEN /home, rg TOKEN /), so they ask.
_AUTO_RECURSIVE_SEARCH = frozenset({"grep", "egrep", "fgrep", "rg", "ug", "find", "fd"})
# Directory walkers that always recurse (tree /home, du /) read the whole host
# subtree under an absolute/tilde root, like a recursive search. ls only recurses
# with -R/--recursive, so it is gated separately when that flag is present.
_AUTO_RECURSIVE_LISTERS = frozenset({"tree", "du"})
# Benign wrappers: safe AND forward command position to their target (checked in
# turn). sudo/su/chroot/etc. are absent, so they classify as unsafe. xargs is
# absent too: it appends arguments read from stdin that this scan never sees, so
# `echo -o out /etc/passwd | xargs sort` forwards to `sort -o out /etc/passwd`
# (a write + sensitive read) while only the allow-listed literals are visible.
_AUTO_SAFE_WRAPPERS = frozenset(
    {"env", "command", "time", "timeout", "nice", "ionice", "stdbuf", "nohup"}
)

# MCP tools whose names look read-only auto-run; anything else asks.
_AUTO_SAFE_MCP_TOOL_RE = re.compile(
    r"^(get|list|search|read|fetch|query|find|describe|show|view|lookup|"
    r"retrieve|count|status|info|help|check)(?:[_\-].*)?$",
    re.IGNORECASE,
)
# A mutating verb anywhere in the name overrides a read-only prefix, so a
# compound name like get_or_create_issue or read_and_delete_file still asks.
_AUTO_UNSAFE_MCP_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:create|update|delete|remove|write|set|add|send|post|put|"
    r"patch|insert|drop|kill|exec|execute|run|deploy|publish|move|rename|edit|"
    r"modify|upload|replace|revoke|grant|approve|merge|close|cancel|pay|"
    r"transfer|buy|sell|reset|clear|purge|destroy|terminate|revert|rollback|"
    r"trigger|enable|disable|install|uninstall|restart|stop|start|"
    r"save|archive|submit|commit|push|sync|register|"
    r"clone|checkout|comment|fork|tag|invite|share|append|prepend|"
    r"copy|duplicate|import|export|download|backup|restore|snapshot|mirror|"
    r"upsert|assign|mark|subscribe|unsubscribe|reply|notify)(?:[_\-]|$)",
    re.IGNORECASE,
)
# A read-named MCP tool that returns a secret is still a sensitive read, so a
# credential noun anywhere in the name (read_secret, list_tokens,
# get_credentials, fetch_api_key) asks even without a mutating verb or a path/SQL
# argument. Scoped nouns (api/access/private/... _key) avoid flagging benign
# keys like a primary_key or keyboard lookup.
_AUTO_SENSITIVE_MCP_NOUN_RE = re.compile(
    r"(?:^|[_\-])(?:"
    r"secret|token|credential|password|passwd|passphrase|apikey|"
    r"(?:api|access|private|secret|signing|encryption|auth|session)[_\-]?keys?"
    r")s?(?:[_\-]|$)",
    re.IGNORECASE,
)

# Python: modules whose import alone signals side effects auto mode should ask
# about (process spawning, network, bulk file ops, low-level memory).
_AUTO_UNSAFE_PY_MODULES = frozenset(
    {
        "subprocess",
        "shutil",
        "socket",
        "ctypes",
        "multiprocessing",
        "pty",
        "fcntl",
        "requests",
        "urllib",
        "urllib3",
        "http",
        "httpx",
        "aiohttp",
        # huggingface_hub.hf_hub_download / snapshot_download fetch remote repo
        # files over the network and write them to an on-disk cache.
        "huggingface_hub",
        # websockets opens a network connection; socketserver binds a listener.
        "websockets",
        "socketserver",
        "ftplib",
        "smtplib",
        "telnetlib",
        "paramiko",
        # mail/news/rpc/browser stdlib clients open outbound connections
        # (imaplib, poplib, xmlrpc.client, webbrowser.open).
        "imaplib",
        "poplib",
        "nntplib",
        "xmlrpc",
        "webbrowser",
        "tempfile",
        # deserialization that can execute arbitrary code on load.
        "pickle",
        "marshal",
        "shelve",
        "dill",
        # dbm.open(file, "c"/"n") creates files; treat the family as writers.
        "dbm",
        # sqlite3.connect(path) creates/mutates a database file (and runs DDL/DML
        # without an open()/writer attribute), like dbm.
        "sqlite3",
        # runpy runs a script/module as code.
        "runpy",
        # ensurepip.bootstrap installs pip and venv.create builds an environment;
        # both write to disk and can fetch/install packages.
        "ensurepip",
        "venv",
    }
)
# Attribute calls that mutate the filesystem / spawn processes (os.remove,
# Path.write_text, sock.connect, ...) regardless of how the module was bound.
_AUTO_UNSAFE_PY_ATTRS = frozenset(
    {
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "renames",
        "replace",
        "rmtree",
        "move",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "chmod",
        "chown",
        "system",
        "popen",
        "execv",
        "execve",
        "execl",
        "execlp",
        "execvp",
        "spawnl",
        "spawnv",
        # os.startfile launches a program via its Windows association.
        "startfile",
        "fork",
        "kill",
        "killpg",
        "symlink",
        "link",
        "mkdir",
        "makedirs",
        "truncate",
        "touch",
        "write_text",
        "write_bytes",
        "urlopen",
        "urlretrieve",
        "connect",
        "bind",
        "sendall",
        # pathlib link creators, os node/metadata mutators, dynamic import.
        "symlink_to",
        "hardlink_to",
        "link_to",
        "mkfifo",
        "mknod",
        "utime",
        # os.setxattr / os.removexattr mutate extended attributes, like chmod.
        "setxattr",
        "removexattr",
        "import_module",
        # loader.exec_module runs a module's code like import_module; archive
        # extractall/extract write arbitrary files (zip-slip): extract takes a
        # single member but an attacker-controlled member path still escapes.
        "exec_module",
        "extractall",
        "extract",
        "FileIO",
        # asyncio subprocess spawners run a program past the terminal blocklist.
        "create_subprocess_exec",
        "create_subprocess_shell",
        "subprocess_exec",
        "subprocess_shell",
        # asyncio outbound connections / listeners (open_connection,
        # create_connection/server and unix variants), like socket.connect.
        "open_connection",
        "create_connection",
        "create_server",
        "create_unix_connection",
        "create_unix_server",
        # more asyncio listen/connect + UDP/raw socket helpers.
        "start_server",
        "start_unix_server",
        "open_unix_connection",
        "create_datagram_endpoint",
        "sock_connect",
        # os.chdir escapes the workdir; runpy helpers run arbitrary code.
        "chdir",
        "fchdir",
        "run_path",
        "run_module",
        # types.FunctionType wraps a compiled code object into a callable, a
        # dynamic-execution vector; pandas read_pickle deserializes (runs code).
        "FunctionType",
        "read_pickle",
    }
)
# Pickle-backed loaders that can execute code embedded in the file; gated by
# receiver module (torch.load, joblib.load) since bare `load` is too common.
_AUTO_UNSAFE_PY_LOAD_MODULES = frozenset({"torch", "joblib", "cloudpickle"})
# Writer methods that persist to disk without going through open() (numpy.save,
# Image.save, plt.savefig, DataFrame.to_csv, json.dump). Gated as method calls
# only, so a bare attribute reference is not mistaken for a write.
_AUTO_UNSAFE_PY_WRITE_METHODS = frozenset(
    {
        "save",
        "savefig",
        "savez",
        "savez_compressed",
        "savetxt",
        "tofile",
        "dump",
        "to_csv",
        "to_parquet",
        "to_pickle",
        "to_json",
        "to_feather",
        "to_hdf",
        "to_excel",
        "to_stata",
        "to_sql",
        "to_xml",
        # pandas text exporters that write when given a path/buffer (to_html /
        # to_markdown / to_latex mirror to_csv); to_clipboard / to_gbq persist
        # off-process. to_string is omitted: it is overwhelmingly display-only.
        "to_html",
        "to_markdown",
        "to_latex",
        "to_clipboard",
        "to_gbq",
        "imwrite",
        "imsave",
        "write_image",
        "write_html",
        # ML persistence helpers (transformers/peft/safetensors/keras) that
        # export adapters or weights to disk without an open()/writer attribute.
        "save_pretrained",
        "save_file",
        "save_model",
        "save_weights",
        "save_lora",
        "save_checkpoint",
        # logging file handlers open a log file for write on construction (even
        # default mode "a" creates); matched as attribute call and bare import.
        "FileHandler",
        "WatchedFileHandler",
        "RotatingFileHandler",
        "TimedRotatingFileHandler",
        # numpy.memmap(..., mode="w+") and pandas writers create/truncate a file
        # on construction, like open(..., "w").
        "memmap",
        "open_memmap",
        "ExcelWriter",
        "HDFStore",
        # pydoc.writedoc(name) writes name.html to the workdir.
        "writedoc",
    }
)
# Archive / compressed-file constructors taking the mode as their 2nd arg like
# open: ZipFile(name, "w") / gzip.GzipFile(name, "w") write, so gated only in
# write mode (reading a .gz is fine, so the modules are not blanket-unsafe).
_ARCHIVE_CTOR_NAMES = frozenset({"ZipFile", "TarFile", "GzipFile", "BZ2File", "LZMAFile"})
# The stdlib module each archive constructor is imported from.
_ARCHIVE_CTOR_MODULES = {
    "zipfile": "ZipFile",
    "tarfile": "TarFile",
    "gzip": "GzipFile",
    "bz2": "BZ2File",
    "lzma": "LZMAFile",
}
# Modules whose top-level open() takes the mode as its 2nd arg like builtin open,
# so `from gzip import open as gopen` binds an open alias gated on write mode.
_OPEN_ALIAS_MODULES = frozenset({"gzip", "bz2", "lzma"})
# Builtins/itertools helpers that call their first argument once per item, so a
# writer/open alias handed to one runs without a direct call(...) site
# (list(map(open, names, modes)), starmap(np.save, ...)). filter's predicate is
# also invoked, so a writer smuggled there runs too.
_HIGHER_ORDER_INVOKERS = frozenset({"map", "filter", "starmap", "reduce"})
_PY_WRITE_MODE_RE = re.compile(r"[wax+]")
# A file-mode literal ("w", "rb", "a+"): letters/flags only, no path chars.
# Used to tell a Path.open("w") mode from a ZipFile.open("name.txt") filename.
_PY_MODE_LITERAL_RE = re.compile(r"^[rwxa][btru+]*$")

# Reading these off the host escapes the intent of "read-only is safe": they
# hold credentials. Path traversal (../) escapes the per-session workdir.
_SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[/\\])\.(?:ssh|aws|azure|gnupg|docker|kube|config/gcloud|config/gh)(?:[/\\]|$)"
    r"|\.(?:netrc|npmrc|pypirc|git-credentials|env)(?:$|[/\\.\s'\"])"
    r"|id_rsa|id_ed25519|id_ecdsa|id_dsa"
    # Hugging Face stores the login token at ~/.cache/huggingface/token and the
    # legacy ~/.huggingface/token (plus the multi-token store stored_tokens); the
    # rest of that cache is model data, so only the credential files match. The
    # optional leading dot covers the .huggingface dotdir form.
    r"|(?:^|[/\\])\.?huggingface[/\\](?:token|stored_tokens)(?:$|[/\\.\s'\"])"
    # /etc/ssh holds the host private keys (ssh_host_*_key); the whole dir is
    # sensitive, not just passwd/shadow/sudoers.
    r"|credentials|/etc/(?:passwd|shadow|sudoers|ssh(?:[/\\]|$))"
    # Bash opens /dev/tcp/host/port and /dev/udp/host/port as network sockets,
    # so a redirection to one reaches the network without the confirm prompt.
    r"|/dev/(?:tcp|udp)/"
    # Docker/Kubernetes secret mounts hold injected credentials.
    r"|/(?:var/)?run/secrets(?:[/\\]|$)"
    # procfs leaks a (possibly parent) process env/args/memory to a read,
    # including the per-thread aliases under /proc/<pid>/task/<tid>/. The fd/
    # dir holds symlinks to a process's open files (a held credential/db file).
    r"|/proc/[^/\s'\"]+/(?:task/[^/\s'\"]+/)?(?:environ|cmdline|mem|maps|fd)\b"
    # A .pem/.key file (basename before the extension), not a bare ".key"
    # (e.g. a jq '.key' filter).
    r"|\w[\w.-]*\.(?:pem|key)(?:$|[\s'\"])",
    re.IGNORECASE,
)
# A shell redirection with no following space (cat <../../notes) keeps `..`
# adjacent to `<`/`>`, so those count as leading delimiters here too.
_PARENT_TRAVERSAL_RE = re.compile(r"(?:^|[\s/\\'\"=:<>])\.\.(?:[/\\]|$|[\s'\"])")
# A sensitive directory: a dynamic segment under it (open(f"/etc/{name}")) is
# not provably safe, so fail closed when a folded path has a dynamic piece here.
_SENSITIVE_DIR_RE = re.compile(
    r"/etc/|/(?:var/)?run/secrets[/\\]|(?:^|[/\\])\.(?:ssh|aws|azure|gnupg|docker|kube)[/\\]"
    r"|(?:^|[/\\])\.config/(?:gcloud|gh)[/\\]",
    re.IGNORECASE,
)
# Collapse /./ and repeated slashes so /etc/./passwd and /etc//passwd, which
# the OS resolves to /etc/passwd, still match the sensitive-path regex.
_REDUNDANT_SLASH_RE = re.compile(r"/\.?(?=/)")
# $name, ${name}, and operator/substring forms (${name:-x}, ${name:0:6}) all
# reference `name`; substituting the assigned value catches paths hidden behind
# a substring expansion (p=passwd; cat /etc/${p:0:6}).
_SHELL_VAR_RE = re.compile(r"\$\{(\w+)(?::[^{}]*)?\}|\$(\w+)")
# Pattern replacement (${p/X/w}, global ${p//X/w}) transforms the value before
# the path is used; apply it so p=passXd; cat /etc/${p/X/w} is scanned.
_SHELL_PARAM_REPL_RE = re.compile(r"\$\{(\w+)/(/)?([^/{}]*)/([^{}]*)\}")
# Case modification (${p^^} upper, ${p,,} lower, ${p^}/${p,} first char) also
# transforms the value, so p=PASSWD; cat /etc/${p,,} builds /etc/passwd.
_SHELL_PARAM_CASE_RE = re.compile(r"\$\{(\w+)(\^\^|,,|\^|,)\}")
# Indirect expansion ${!p} yields the value of the variable *named* by $p, so
# x=passwd; p=x; cat /etc/${!p} builds /etc/passwd.
_SHELL_PARAM_INDIRECT_RE = re.compile(r"\$\{!(\w+)\}")
_SHELL_ASSIGN_RE = re.compile(r"(?:^|[\s;&|(])([A-Za-z_]\w*)=([^\s;&|)]+)")
# Bash ANSI-C quoting ($'\x77' -> 'w') is expanded after this classifier, so
# decode $'...' bodies before the sensitive-path scan.
_ANSI_C_RE = re.compile(r"\$'((?:[^'\\]|\\.)*)'")
# Shell quotes only delimit; bash concatenates the pieces (cat /proc/x/enviro''n
# reads .../environ), so strip them before the sensitive-path scan.
_SHELL_QUOTE_RE = re.compile(r"['\"]")
# A glob bracket class [s] -> s, so .s[s]h de-obfuscates to .ssh for the scan.
_GLOB_BRACKET_RE = re.compile(r"\[([^!\]][^\]]*)\]")
# Bash POSIX character classes ([[:lower:]]) each match one char; Python fnmatch
# does not understand them, so normalize to `?` before the glob check.
_POSIX_CLASS_RE = re.compile(r"\[\[:\w+:\]\]")
# Canonical sensitive files a ? / * / [..] glob could expand to; fnmatch tests
# whether the pattern reaches one (cat /e??/passwd -> /etc/passwd).
_SENSITIVE_GLOB_TARGETS = (
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/root/.ssh/id_rsa",
    "/root/.aws/credentials",
    "/home/u/.ssh/id_rsa",
    "/home/u/.ssh/id_ed25519",
    "/home/u/.aws/credentials",
    "/home/u/.netrc",
    "/home/u/.git-credentials",
)
# Directories whose every file is a credential/secret; a glob resolving into one
# (cat /r?n/secrets/hf_token, cat /root/.s??/id_rsa) reads a secret even though
# the exact filename is never enumerated, so a globbed token here asks.
_SENSITIVE_GLOB_DIRS = (
    "/run/secrets",
    "/var/run/secrets",
    "/root/.ssh",
    "/root/.aws",
    "/root/.azure",
    "/root/.gnupg",
    "/root/.docker",
    "/root/.kube",
    "/root/.config/gcloud",
    "/root/.config/gh",
    "/home/u/.ssh",
    "/home/u/.aws",
    "/home/u/.azure",
    "/home/u/.gnupg",
    "/home/u/.docker",
    "/home/u/.kube",
    "/home/u/.config/gcloud",
    "/home/u/.config/gh",
)
# Credential basenames a glob can reach even when the directory is not wholly
# sensitive (cat ~/.huggingface/tok?n -> token, cat ~/.netr? -> .netrc); the
# canonical-target list only covers a few fixed home paths, so match the globbed
# basename against these directly.
_SENSITIVE_GLOB_BASENAMES = frozenset(
    {
        "token",
        "stored_tokens",
        "credentials",
        ".netrc",
        "netrc",
        ".pypirc",
        ".npmrc",
        ".git-credentials",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "passwd",
        "shadow",
        # A project .env holds secrets; the literal path is gated elsewhere, so a
        # glob that expands to it (cat .e?v) must be too.
        ".env",
    }
)
# A leading shell redirection (<, >, 2>, >>) hides the path from a plain glob
# scan (cat </e??/passwd); strip it before matching.
_REDIR_PREFIX_RE = re.compile(r"^\d*[<>]+")
# Bash brace expansion (cat /etc/pass{w,}d -> /etc/passwd /etc/passd, and the
# sequence form cat /etc/pass{w..w}d -> /etc/passwd) runs after this classifier;
# expand comma groups and .. sequences to scan each result.
_BRACE_COMMA_RE = re.compile(r"^\{([^{}]*,[^{}]*)\}$")
_BRACE_SEQ_RE = re.compile(r"^\{([^{}]+)\.\.([^{}]+)(?:\.\.(-?\d+))?\}$")
_BRACE_ANY_RE = re.compile(r"\{[^{}]*,[^{}]*\}|\{[^{}]+\.\.[^{}]+(?:\.\.-?\d+)?\}")
# Parameter expansion with a default/alternate operator (${x:-passwd},
# ${x:+passwd}, ${x=passwd}) can synthesize a path after approval; the operand
# is substituted so the resulting path is scanned.
_SHELL_PARAM_OP_RE = re.compile(r"\$\{[A-Za-z_]\w*:?[-=+]([^{}]*)\}")


def _references_sensitive_path(text: str) -> bool:
    """True if a command or string literal reads a credential path or escapes
    the sandbox workdir via parent traversal."""
    norm = _REDUNDANT_SLASH_RE.sub("", text)
    debracket = _GLOB_BRACKET_RE.sub(lambda m: m.group(1)[0], text)
    return bool(
        _PARENT_TRAVERSAL_RE.search(text)
        or _SENSITIVE_PATH_RE.search(text)
        or _SENSITIVE_PATH_RE.search(norm)
        or _SENSITIVE_PATH_RE.search(debracket)
    )


def _pattern_matches_dir(pattern: str, target: str) -> bool:
    """Segment-wise fnmatch so a glob segment does not cross a '/' boundary
    (`/home/*` must not match `/home/u/.ssh`)."""
    p = pattern.split("/")
    t = target.split("/")
    if len(p) != len(t):
        return False
    return all(fnmatch.fnmatch(tseg, pseg) for pseg, tseg in zip(p, t))


def _glob_token_sensitive(token: str) -> bool:
    """True if a single ? / * / [..] glob token could expand to a sensitive file
    or a file under a secret/credential directory. Shared by the terminal scan
    and the Python glob check (glob.glob('/e??/passwd'))."""
    token = _REDIR_PREFIX_RE.sub("", _SHELL_QUOTE_RE.sub("", token))
    # A POSIX class ([[:lower:]]) matches one char, like `?`, but fnmatch treats
    # it as a literal set; normalize so cat /etc/pass[[:lower:]]d resolves.
    token = _POSIX_CLASS_RE.sub("?", token)
    if not any(c in token for c in "?*["):
        return False
    if any(fnmatch.fnmatch(target, token) for target in _SENSITIVE_GLOB_TARGETS):
        return True
    # A glob that resolves to a credential basename is sensitive wherever it
    # lives (cat ~/.huggingface/tok?n -> token, cat proj/.netr? -> .netrc); the
    # fixed-target list only covers a handful of home paths.
    base = token.rsplit("/", 1)[-1]
    if any(c in base for c in "?*[") and any(
        fnmatch.fnmatch(name, base) for name in _SENSITIVE_GLOB_BASENAMES
    ):
        return True
    # A globbed directory that resolves into a secret/credential dir makes every
    # file below it sensitive (cat /r?n/secrets/hf_token).
    head = token.rsplit("/", 1)[0] if "/" in token else token
    return any(
        _pattern_matches_dir(token, d) or _pattern_matches_dir(head, d)
        for d in _SENSITIVE_GLOB_DIRS
    )


def _glob_hits_sensitive(command: str) -> bool:
    """True if any glob token in a command could expand to a sensitive file, so
    `cat /e??/passwd` and `cat /r?n/secrets/hf_token` ask even without a literal
    sensitive path."""
    return any(
        _glob_token_sensitive(token)
        for token in command.replace(";", " ").replace("|", " ").split()
    )


def _expand_shell_assignments(command: str) -> str:
    """Best-effort substitution of `NAME=value ... $NAME`, so a sensitive path
    split across an assignment and an argument (p=/etc; cat $p/passwd) is still
    visible to the sensitive-path scan. Also applies pattern replacement
    (p=passXd; cat /etc/${p/X/w}). Fail-open: only adds detections."""
    env = dict(_SHELL_ASSIGN_RE.findall(command))
    if not env:
        return command

    def repl_pattern(m):
        var, is_global, pat, rep = m.group(1), m.group(2), m.group(3), m.group(4)
        if var not in env or not pat:
            return m.group(0)
        return env[var].replace(pat, rep) if is_global else env[var].replace(pat, rep, 1)

    def repl_case(m):
        var, op = m.group(1), m.group(2)
        if var not in env:
            return m.group(0)
        v = env[var]
        if op == ",,":
            return v.lower()
        if op == "^^":
            return v.upper()
        if op == ",":
            return v[:1].lower() + v[1:]
        return v[:1].upper() + v[1:]

    def repl_indirect(m):
        # ${!p} -> value of the variable named by $p (env[env[p]]).
        pointed = env.get(m.group(1))
        return env.get(pointed, m.group(0)) if pointed is not None else m.group(0)

    command = _SHELL_PARAM_INDIRECT_RE.sub(repl_indirect, command)
    command = _SHELL_PARAM_REPL_RE.sub(repl_pattern, command)
    command = _SHELL_PARAM_CASE_RE.sub(repl_case, command)
    return _SHELL_VAR_RE.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), command)


def _expand_param_defaults(command: str) -> str:
    """Substitute the operand of a default/alternate parameter expansion
    (cat /etc/pass${x:-wd} -> cat /etc/passwd), which bash applies after this
    classifier. Fail-open: only adds detections."""
    return _SHELL_PARAM_OP_RE.sub(lambda m: m.group(1), command)


def _decode_ansi_c(command: str) -> str:
    """Decode bash ANSI-C quoted words (cat $'/etc/pass\\x77d' -> cat /etc/passwd)
    so an escape-obfuscated path is visible to the scan. Fail-open: only adds
    detections."""

    def dec(m):
        try:
            return bytes(m.group(1), "utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            return m.group(0)

    return _ANSI_C_RE.sub(dec, command)


def _brace_range(lo: str, hi: str, step: "str | None") -> "list[str]":
    """Expand a bash sequence brace endpoint pair ({1..3}, {a..c}, {w..w})."""
    try:
        istep = abs(int(step)) if step else 1
        istep = istep or 1
        if re.fullmatch(r"-?\d+", lo) and re.fullmatch(r"-?\d+", hi):
            a, b = int(lo), int(hi)
            rng = range(a, b + 1, istep) if a <= b else range(a, b - 1, -istep)
            return [str(x) for x in rng][:64]
        if len(lo) == 1 and len(hi) == 1 and lo.isalpha() and hi.isalpha():
            a, b = ord(lo), ord(hi)
            rng = range(a, b + 1, istep) if a <= b else range(a, b - 1, -istep)
            return [chr(x) for x in rng][:64]
    except (ValueError, TypeError):
        pass
    return []


def _brace_options(text: str) -> "list[str]":
    """Options a single brace group expands to (comma list or .. sequence)."""
    m = _BRACE_COMMA_RE.match(text)
    if m:
        return m.group(1).split(",")
    m = _BRACE_SEQ_RE.match(text)
    if m:
        return _brace_range(m.group(1), m.group(2), m.group(3)) or [text]
    return [text]


def _expand_braces(command: str) -> str:
    """Best-effort bash brace expansion (cat /etc/pass{w,}d -> cat /etc/passwd
    /etc/passd, cat /etc/pass{w..w}d -> cat /etc/passwd) so a sensitive path
    split across a brace group is scanned. Bounded. Fail-open: only detects."""
    results = [command]
    for _ in range(6):
        if not any(_BRACE_ANY_RE.search(s) for s in results):
            break
        expanded = []
        for s in results:
            m = _BRACE_ANY_RE.search(s)
            if not m:
                expanded.append(s)
                continue
            for opt in _brace_options(m.group(0)):
                expanded.append(s[: m.start()] + opt + s[m.end() :])
        results = expanded[:64]
    return " ".join(results)


def _mode_arg_writes(mode_node) -> bool:
    """True if an AST node used as a file mode requests write/append."""
    if mode_node is None:
        return False  # default "r"
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return bool(_PY_WRITE_MODE_RE.search(mode_node.value))
    return True  # dynamic mode: cannot prove read-only


def _has_kwarg_splat(node) -> bool:
    """True if the call has a ``**kwargs`` splat, which can hide a write mode."""
    return any(kw.arg is None for kw in node.keywords or [])


def _builtin_open_writes(node) -> bool:
    """Write check for builtin ``open(file, mode)`` (mode is the 2nd arg)."""
    if _has_kwarg_splat(node):
        return True  # **{"mode": "w"} could request a write
    if any(isinstance(a, ast.Starred) for a in node.args):
        return True  # *("f", "w") could splat a write mode into the positionals
    mode = node.args[1] if len(node.args) >= 2 else None
    for kw in node.keywords or []:
        if kw.arg == "mode":
            mode = kw.value
    return _mode_arg_writes(mode)


def _attr_open_writes(node) -> bool:
    """Write check for ``x.open(...)`` (e.g. ``Path.open(mode)`` where mode is
    the 1st arg). Only a mode-looking string is read as the mode, so a
    ``ZipFile.open("name.txt")`` read is not mistaken for a write."""
    if _has_kwarg_splat(node):
        return True  # **{"mode": "w"} could request a write
    for kw in node.keywords or []:
        if kw.arg == "mode":
            return _mode_arg_writes(kw.value)
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if _PY_MODE_LITERAL_RE.match(first.value):
                return bool(_PY_WRITE_MODE_RE.search(first.value))
            # A 2nd positional arg is either a mode (x.open(name, "w")) or
            # os.open(path, O_CREAT) flags via an alias: honor a string mode,
            # otherwise cannot prove read-only, so ask.
            if len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    return _mode_arg_writes(second)
                return True
            return False
        return True  # dynamic first arg: cannot prove read-only
    return False  # no args: read


_PATH_CTORS = (
    "Path",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "PosixPath",
    "WindowsPath",
)
# Deterministic path pass-through/normalizer calls that return the same location
# (os.path.abspath('/etc') -> /etc, Path('/etc').resolve() -> /etc), so folding
# through them keeps a sensitive root visible to the scan.
_PATH_PASSTHROUGH_ATTRS = frozenset(
    {"abspath", "normpath", "realpath", "expanduser", "expandvars", "resolve", "absolute"}
)
# pathlib methods that rewrite only the final path component, so the sensitive
# target is never spelled out as a literal (Path('/etc/x').with_name('passwd')
# -> /etc/passwd). Folded below so the rewritten path is still scanned.
_PATH_NAME_REWRITES = frozenset({"with_name", "with_stem", "with_suffix"})
# Mapping-style %-format conversion specifier: %(name)s / %(n)5.2f. Used to fold
# '/etc/%(f)s' % {'f': 'passwd'} to /etc/passwd (a dynamic value becomes NUL).
_PERCENT_NAMED_RE = re.compile(r"%\((\w+)\)[-#0 +]*\d*(?:\.\d+)?[a-zA-Z]")


def _folded_path(
    node,
    literals = None,
    ctors = None,
    join_names = None,
) -> "str | None":
    """Best-effort value of a path built from string literals, so a sensitive
    path assembled from pieces (os.path.join('/etc', 'passwd'), '/etc'+'/passwd',
    Path('/etc') / 'passwd', f'/proc/{pid}/environ', f'/etc/{name}') is still
    visible to the scan. A dynamic piece becomes NUL, a non-slash placeholder,
    so a dynamic segment under a sensitive dir (/etc/NUL) is still detectable.
    ``literals`` maps names bound to string literals (base = '/etc'); ``ctors``
    is the set of pathlib constructor names (incl. import aliases); ``join_names``
    are bare names bound to os.path.join (from os.path import join)."""
    literals = literals or {}
    ctors = ctors or _PATH_CTORS
    join_names = join_names or frozenset()

    def fold(node) -> "str | None":
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            # bytes paths are valid too (open(b'/etc/passwd')); decode for scan.
            return (
                node.value.decode("latin-1", "ignore")
                if isinstance(node.value, bytes)
                else node.value
            )
        if isinstance(node, ast.Name):
            return literals.get(node.id)
        if isinstance(node, ast.Attribute) and node.attr in ("parent", "parents"):
            # A pathlib .parent/.parents walks above the current dir, escaping
            # the per-session workdir without a literal '..'; mark it so a read
            # folds to unsafe (\x02 is a non-slash escape sentinel).
            return "\x02"
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and (node.value.attr == "parents")
        ):
            return "\x02"  # Path(...).parents[1]
        if isinstance(node, ast.JoinedStr):
            return "".join(
                v.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
                else (fold(v.value) or "\x00")
                if isinstance(v, ast.FormattedValue)
                else "\x00"
                for v in node.values
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            left = fold(node.left)
            right = fold(node.right)
            left = "\x00" if left is None else left
            right = "\x00" if right is None else right
            # Path('/etc') / 'passwd' joins with a separator; '+' concatenates.
            return left + "/" + right if isinstance(node.op, ast.Div) else left + right
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            # Old-style formatting: '%s/%s' % ('/etc', 'passwd') -> /etc/passwd.
            template = fold(node.left)
            if template is not None and "%" in template:
                rhs = node.right
                if "%(" in template:
                    # Mapping-style: '/etc/%(f)s' % {'f': 'passwd'} -> /etc/passwd.
                    # A literal dict resolves each name; an unresolved value or a
                    # non-literal mapping leaves the NUL marker so /etc/<dynamic>
                    # still fails closed under a sensitive dir.
                    mapping: "dict[str, str]" = {}
                    if isinstance(rhs, ast.Dict):
                        for k, v in zip(rhs.keys, rhs.values):
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                fv = fold(v)
                                mapping[k.value] = fv if fv is not None else "\x00"
                    return _PERCENT_NAMED_RE.sub(
                        lambda m: mapping.get(m.group(1), "\x00"), template
                    )
                if isinstance(rhs, ast.Tuple):
                    args = tuple((fold(e) or "\x00") for e in rhs.elts)
                else:
                    single = fold(rhs)
                    args = (single if single is not None else "\x00",)
                try:
                    return template % args
                except (TypeError, ValueError, KeyError):
                    return None
            return None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "joinpath":
                # Path('/etc').joinpath('passwd') -> receiver and args are pieces.
                base = fold(func.value)
                parts = [base if base is not None else "\x00"]
                parts += [(fold(a) or "\x00") for a in node.args]
                return "/".join(parts)
            if isinstance(func, ast.Attribute) and func.attr in ("glob", "rglob", "iglob"):
                # Path('/etc').glob('passw?') -> the receiver dir joined with the
                # glob pattern; _glob_token_sensitive then tests /etc/passw?.
                base = fold(func.value)
                pattern = fold(node.args[0]) if node.args else "\x00"
                return (base if base is not None else "\x00") + "/" + (pattern or "\x00")
            if isinstance(func, ast.Attribute) and func.attr in _PATH_NAME_REWRITES:
                # Path('/etc/x').with_name('passwd') -> /etc/passwd; with_stem /
                # with_suffix rewrite only the final component. Fold to the
                # rewritten path so a sensitive target that no literal spells out
                # is still caught. An unresolved receiver stays None (untracked,
                # like a bare variable), and a dynamic arg becomes the NUL marker.
                base = fold(func.value)
                if base is None:
                    return None
                arg = fold(node.args[0]) if node.args else None
                arg = "\x00" if arg is None else arg
                idx = base.rfind("/")
                head = base[: idx + 1] if idx >= 0 else ""
                name = base[idx + 1 :] if idx >= 0 else base
                dot = name.rfind(".")
                stem = name[:dot] if dot > 0 else name
                suffix = name[dot:] if dot > 0 else ""
                if func.attr == "with_name":
                    name = arg
                elif func.attr == "with_stem":
                    name = arg + suffix
                else:  # with_suffix
                    name = stem + arg
                return head + name
            if isinstance(func, ast.Attribute) and func.attr in _PATH_PASSTHROUGH_ATTRS:
                # Deterministic normalizers keep the same path: os.path.abspath(
                # '/etc') -> /etc, Path('/etc').resolve() -> /etc. When called with
                # a path arg fold it, else fold the receiver (Path method form).
                return fold(node.args[0]) if node.args else fold(func.value)
            if isinstance(func, ast.Attribute) and func.attr == "join":
                # str.join has the separator as the receiver and the pieces in
                # one iterable arg ("".join(['/etc', '/passwd']) -> /etc/passwd);
                # tell it apart from os.path.join(*pieces).
                sep = fold(func.value)
                if (
                    sep is not None
                    and len(node.args) == 1
                    and isinstance(node.args[0], (ast.List, ast.Tuple))
                ):
                    pieces = [(fold(e) or "\x00") for e in node.args[0].elts]
                    return sep.join(pieces)
                parts = [(fold(a) or "\x00") for a in node.args]
                return "/".join(parts)
            # A bare os.path.join alias (from os.path import join): join(*pieces).
            if isinstance(func, ast.Name) and func.id in join_names:
                parts = [(fold(a) or "\x00") for a in node.args]
                return "/".join(parts)
            # A bare/qualified/aliased pathlib constructor (Path(...), P(...)).
            if (isinstance(func, ast.Attribute) and func.attr in ctors) or (
                isinstance(func, ast.Name) and func.id in ctors
            ):
                parts = [(fold(a) or "\x00") for a in node.args]
                return "/".join(parts)
            # '/etc/{}'.format('passwd') -> /etc/passwd (literal template + args).
            if isinstance(func, ast.Attribute) and func.attr == "format":
                template = fold(func.value)
                if template is not None and "{" in template:
                    parts = []
                    for a in node.args:
                        if isinstance(a, ast.Constant):
                            parts.append(str(a.value))
                        else:
                            folded = fold(a)
                            parts.append("\x00" if folded is None else folded)
                    try:
                        return template.format(*parts)
                    except (IndexError, KeyError, ValueError):
                        return None
        return None

    return fold(node)


def _dynamic_name_hits_sensitive(folded) -> bool:
    """True if a folded path with a dynamic piece (NUL) inside a path segment
    could spell a credential target, e.g. open('/et' + chr(99) + '/passwd')
    folds to '/et\\x00/passwd'. NUL matches any run of non-separator chars so the
    dynamic split of a sensitive name resolves, while an all-dynamic ('\\x00\\x00')
    or segment-spanning ('\\x00/\\x00') path cannot form a single credential name
    and stays safe."""
    if not folded or "\x00" not in folded:
        return False
    pattern = "".join(r"[^/\\]*" if ch == "\x00" else re.escape(ch) for ch in folded)
    try:
        rx = re.compile(pattern + r"\Z")
    except re.error:
        return True  # pathological pattern: fail closed
    return any(rx.match(t) for t in _SENSITIVE_GLOB_TARGETS)


def _folded_is_sensitive(folded) -> bool:
    """A folded path is sensitive if it names a credential file, has a dynamic
    segment (NUL) directly under a sensitive directory (/etc/NUL), walks out of
    the sandbox via a pathlib .parent/.parents escape (\\x02), or is a glob that
    could resolve to a credential path (glob.glob('/e??/passwd'))."""
    if not folded:
        return False
    return (
        "\x02" in folded
        or _references_sensitive_path(folded)
        or ("\x00" in folded and bool(_SENSITIVE_DIR_RE.search(folded)))
        # A dynamic segment (NUL) can be the "/" forming a sensitive root:
        # open(os.sep + "etc/passwd") folds to "\x00etc/passwd", so re-scan with
        # NUL as "/" (a benign "\x00data/file" -> "/data/file" stays safe).
        or ("\x00" in folded and _references_sensitive_path(folded.replace("\x00", "/")))
        # A dynamic piece can also sit INSIDE a sensitive name: open('/et' +
        # chr(99) + '/passwd') folds to "/et\x00/passwd", which none of the above
        # catch. Match the literals around each NUL against a credential target,
        # treating NUL as "any run of non-separator chars" so /et<dyn>/passwd
        # resolves while an all-dynamic ("\x00\x00" from 1 + 1) or segment-spanning
        # ("\x00/\x00" from a + '/' + b) path stays safe.
        or _dynamic_name_hits_sensitive(folded)
        or _glob_token_sensitive(folded)
    )


def _terminal_is_potentially_unsafe(command: str) -> bool:
    """Classify a terminal command for auto mode (fail closed)."""
    if not command or not command.strip():
        return False
    # Redirections and substitutions can hide writes or nested commands; a
    # quoted ">" false-positives into a prompt, which is the safe direction.
    if ">" in command or "`" in command or "$(" in command or "<(" in command:
        return True
    # Reads that escape the sandbox workdir (../) or hit credential paths are
    # not "safe" reads; ask before running them. Strip shell quotes/backslash
    # escapes and expand NAME=value prefixes first so `cat /proc/$PPID/enviro''n`,
    # `cat /et\c/passwd`, and `p="/proc/$PPID"; cat $p/environ` are caught too.
    stripped = _SHELL_QUOTE_RE.sub("", command).replace("\\", "")
    # Bash applies brace/parameter/ANSI-C expansion after this classifier, so a
    # path split across a brace group (/etc/pass{w,}d), a default/substring param
    # (${x:-wd}, ${p:0:6}), or an escape ($'...') is invisible to the raw scan;
    # expand first (ANSI-C decoded from the raw command, before backslash strip).
    candidates = []
    for c in (command, stripped, _decode_ansi_c(command)):
        c_param = _expand_param_defaults(c)
        candidates.extend((c, c_param, _expand_braces(c_param), _expand_shell_assignments(c_param)))
    # Run both the literal and glob-sensitive scans over every candidate, so a
    # brace-expanded glob (cat /e{t,}c/pass?d -> /etc/pass?d) is caught.
    if any(_glob_hits_sensitive(c) or _references_sensitive_path(c) for c in candidates):
        return True
    # Newlines (and CR) separate commands in a shell but read as plain
    # whitespace to shlex, which would demote "ls\nrm x" to argument position.
    command = command.replace("\r\n", ";").replace("\n", ";").replace("\r", ";")
    try:
        lexer = shlex.shlex(command, posix = True, punctuation_chars = ";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return True
    # A root can also hide behind an assignment (p=/; grep -R TOKEN $p) or a
    # default parameter (grep -R TOKEN ${root:-/home}); re-lex the fully expanded
    # command so the find/fd and recursive-search scans see the resolved token.
    expanded_command = _expand_shell_assignments(_expand_param_defaults(command))
    if expanded_command != command:
        try:
            elexer = shlex.shlex(expanded_command, posix = True, punctuation_chars = ";&|()")
            elexer.whitespace_split = True
            scan_tokens = list(elexer)
        except ValueError:
            return True
    else:
        scan_tokens = tokens
    # find/fd group with (...) which resets command context, so a trailing
    # -delete/-exec could slip past; scan every token when find/fd appears.
    if any(os.path.basename(t.strip(";&|()`{}")).lower() in ("find", "fd") for t in scan_tokens):
        if any(t.split("=", 1)[0] in _AUTO_UNSAFE_FIND_LIKE_FLAGS for t in scan_tokens):
            return True
    # A recursive reader rooted outside the sandbox reads host files (grep -R
    # TOKEN /home, rg TOKEN /, grep -R TOKEN ~root, p=/; grep -R TOKEN $p, and
    # the always-recursive walkers tree /home / du /); ask. Bash expands
    # ~/~user to a home dir after this decision, so a tilde root is a sandbox
    # escape too. A path-qualified command token starts with "/" as well, but
    # that already asks below.
    if any(t.startswith("/") or t.startswith("~") for t in scan_tokens):
        token_bases = [os.path.basename(t.strip(";&|()`{}")).lower() for t in tokens]
        if any(b in _AUTO_RECURSIVE_SEARCH or b in _AUTO_RECURSIVE_LISTERS for b in token_bases):
            return True
        # ls only walks the whole subtree with -R/--recursive (ls -R /home,
        # ls -laR /); a non-recursive ls /home lists one level and stays here.
        if "ls" in token_bases and any(
            t.split("=", 1)[0] in ("-R", "--recursive")
            or (t[:1] == "-" and t[:2] != "--" and "=" not in t and "R" in t[1:])
            for t in tokens
        ):
            return True
    expect_command = True
    prefix_pending = False
    current_command = ""
    positional_args = 0
    pending_flag_value = False
    for token in tokens:
        # Runs of punctuation (";;", ";&") lex as one token; any token made
        # purely of separator characters still separates commands.
        if (
            token in _SHELL_SEPARATORS
            or token in _SHELL_KEYWORDS_AS_SEP
            or not set(token) - set(";&|()")
        ):
            expect_command = True
            prefix_pending = False
            current_command = ""
            positional_args = 0
            pending_flag_value = False
            continue
        if token.startswith("-"):
            # A write/exec flag on an otherwise read-only command asks
            # (sort -o, tree -o, xxd -r, find -exec/-delete/...). Match
            # "--output=x", an attached short option "-o/tmp/out", and a short
            # option bundled in a cluster (sort -uo out => -u -o).
            flag_head = token.split("=", 1)[0]
            cluster = token[1:] if token[:2] != "--" and "=" not in token else ""
            # GNU tools accept unambiguous abbreviations of a long option, so
            # `sort --out=` reaches --output and `env --ch=/` reaches --chdir;
            # a "--x" prefix of an unsafe long flag fails closed.
            is_long_abbrev = flag_head.startswith("--") and len(flag_head) > 2
            for uf in _AUTO_UNSAFE_COMMAND_FLAGS.get(current_command, ()):
                if flag_head == uf or (len(uf) == 2 and (token.startswith(uf) or uf[1] in cluster)):
                    return True
                if is_long_abbrev and uf.startswith("--") and uf.startswith(flag_head):
                    return True
            # A flag that takes a following value (date -d STRING / -r FILE;
            # uniq -f N; xxd -c N) so the value token is not mistaken for a
            # clock-setting positional or an output-file positional.
            pending_flag_value = "=" not in token and (
                (current_command == "date" and flag_head in _DATE_DISPLAY_VALUE_FLAGS)
                or flag_head in _SECOND_POSITIONAL_VALUE_FLAGS.get(current_command, ())
            )
            if not prefix_pending:
                expect_command = False
            continue
        if not expect_command:
            raw_pos = token.strip(";&|()`{}")
            # uniq [INPUT [OUTPUT]] writes its second file positional; count file
            # positionals and ask on the second one. A preceding option's value
            # (uniq -f 2) is consumed via pending_flag_value, so a file literally
            # named with digits (uniq 123 out) is still counted.
            if current_command in _AUTO_SECOND_POSITIONAL_WRITES:
                if pending_flag_value:
                    pending_flag_value = False
                elif raw_pos:
                    positional_args += 1
                    if positional_args >= 2:
                        return True
            # hostname NAME sets the hostname; date <timestamp> sets the clock. A
            # positional past a display flag's value therefore mutates state and
            # asks (date's +FORMAT display token stays read-only).
            elif current_command in _AUTO_ARG_SENSITIVE_COMMANDS:
                if pending_flag_value:
                    pending_flag_value = False
                elif raw_pos and not (current_command == "date" and raw_pos.startswith("+")):
                    return True
            continue
        if _ASSIGNMENT_RE.match(token):
            # Benign NAME=value prefixes are skipped, but ones that change
            # command lookup/loading (PATH, LD_PRELOAD, ...) fail closed.
            if _env_assignment_is_unsafe(token.split("=", 1)[0]):
                return True
            continue
        if prefix_pending and token.lstrip("-").isdigit():
            continue
        raw = token.strip(";&|()`{}")
        # A path-qualified command (./ls, /tmp/cat) is an arbitrary executable,
        # not the trusted system utility its basename matches; ask first.
        if "/" in raw or "\\" in raw:
            return True
        base = os.path.basename(raw).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        if base in _AUTO_SAFE_WRAPPERS:
            prefix_pending = True
            # Track the wrapper so its own flags (env --chdir) are checked;
            # the real command overwrites this when it is reached.
            current_command = base
            pending_flag_value = False
            continue
        if base not in _AUTO_SAFE_TERMINAL_COMMANDS:
            return True
        current_command = base
        expect_command = False
        prefix_pending = False
        positional_args = 0
        pending_flag_value = False
    return False


def _python_is_potentially_unsafe(code: str) -> bool:
    """Classify python-tool code for auto mode (fail closed)."""
    if not code or not code.strip():
        return False
    # Anything the sandbox's static analysis already objects to would be
    # refused at execution time; surface it as a confirmation first.
    if _check_code_safety(code) is not None:
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False  # runs into a normal traceback; nothing to guard
    # Names bound to the builtin open (f = open; from builtins import open as f;
    # f, _ = (open, print)) so an aliased writer call is still checked below.
    # builtins_aliases tracks `import builtins [as b]` for builtins.exec/eval.
    open_aliases = {"open"}
    # Attribute names bound to open (box.f = open), so a later box.f('out', 'w')
    # write is still gated even though the callable is an attribute, not a name.
    attr_open_aliases: "set[str]" = set()
    builtins_aliases = {"builtins", "__builtins__"}
    # Names bound to a dynamic lookup (rm = getattr(os, "remove");
    # f = globals()["open"]) whose calls cannot be proven read-only, so they
    # fail closed.
    dynamic_aliases = set()
    # Names bound to a dynamic-code builtin, including aliased ones
    # (from builtins import eval as e; e = builtins.exec), so a call or
    # reference through the alias fails closed too. compile() builds a code
    # object that FunctionType/exec can then run.
    code_exec_aliases = {"exec", "eval", "__import__", "breakpoint", "compile"}
    # Names bound to a string literal (base = '/etc'), so a sensitive path
    # split through a variable (base + '/passwd') folds and is caught.
    literal_str_vars: "dict[str, str]" = {}
    # Pathlib constructor names incl. import aliases (from pathlib import Path as
    # P), os.path.join names bound directly (from os.path import join as j), and
    # writer functions imported as bare names (from numpy import save).
    path_ctor_aliases = set(_PATH_CTORS)
    pathjoin_aliases: "set[str]" = set()
    writer_aliases: "set[str]" = set()
    # Module names bound to os/posix (import os as o), so o.open(...) is still
    # recognized as the low-level create/write that os.open is.
    os_aliases = {"os", "posix"}
    # Module names bound to a pickle-backed loader (import torch as t), so
    # t.load(...) is still gated as a code-executing deserialize.
    load_module_aliases = set(_AUTO_UNSAFE_PY_LOAD_MODULES)
    # Names bound to the builtin getattr (g = getattr), so a dynamic lookup
    # aliased through it (rm = g(os, "remove"); rm("f")) still fails closed.
    getattr_aliases = {"getattr"}
    # Names bound to functools.partial, so a partial that wraps open/a writer
    # (w = partial(open, mode="w"); w("out.txt")) fails closed when w is called.
    partial_aliases: "set[str]" = set()
    # Archive constructors imported bare (from zipfile import ZipFile), so
    # ZipFile(name, "w") is gated like the zipfile.ZipFile attribute call.
    archive_ctor_aliases: "set[str]" = set()
    # operator.methodcaller("write_text") is dynamic dispatch, like getattr.
    operator_aliases = {"operator"}
    methodcaller_aliases: "set[str]" = set()
    # logging.basicConfig(filename=...) opens a log file for write.
    basicconfig_aliases: "set[str]" = set()
    # fileinput.input(..., inplace=True) rewrites a file in place.
    fileinput_aliases = {"fileinput"}
    # Higher-order invokers (map/filter/starmap/reduce) call their first arg, so
    # one handed a writer (map(open, ...)) writes without a direct open() site.
    # Track aliases (m = map; from itertools import starmap as sm) so an aliased
    # invoker is still checked; the write-callable gate keeps map(len, ...) safe.
    invoker_aliases = set(_HIGHER_ORDER_INVOKERS)

    def _is_dynamic_namespace(node) -> bool:
        # A namespace mapping whose .get/.pop/.setdefault (or subscript) can return
        # open/eval/a mutator: globals()/locals()/vars(...), any X.__dict__,
        # __builtins__, sys.modules. Looking a name up through one is as dynamic as
        # getattr, so a value fetched from it fails closed.
        if isinstance(node, ast.Attribute):
            if node.attr == "__dict__":
                return True
            return (
                node.attr == "modules"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            )
        if isinstance(node, ast.Name):
            return node.id in builtins_aliases
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id in ("globals", "locals", "vars")
        return False

    def _methodcaller_writes(call) -> bool:
        # operator.methodcaller("write_text", ...) / methodcaller(name): unsafe
        # when the method name is a known writer/mutator, or non-constant (cannot
        # be proven read-only).
        if not call.args:
            return False
        first = call.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            return True
        return first.value in _AUTO_UNSAFE_PY_ATTRS or first.value in _AUTO_UNSAFE_PY_WRITE_METHODS

    def _fileinput_inplace(call) -> bool:
        # fileinput.input(..., inplace=True) opens each file for in-place rewrite.
        if _has_kwarg_splat(call):
            return True
        for kw in call.keywords or []:
            if kw.arg == "inplace":
                v = kw.value
                if isinstance(v, ast.Constant):
                    return bool(v.value)
                return True  # dynamic inplace flag: cannot prove read-only
        return False

    def _basicconfig_writes(call) -> bool:
        # logging.basicConfig(filename=...) creates/opens a log file for writing.
        if _has_kwarg_splat(call):
            return True
        return any(kw.arg == "filename" for kw in call.keywords or [])

    def _wraps_write_callable(arg) -> bool:
        # The callable a partial wraps (partial(open, ...)); True when calling it
        # could create/overwrite a file or resolve a dynamic/mutating function.
        if isinstance(arg, ast.Name):
            return (
                arg.id in open_aliases
                or arg.id in dynamic_aliases
                or arg.id in code_exec_aliases
                or arg.id in getattr_aliases
                or arg.id in writer_aliases
                or arg.id in archive_ctor_aliases
            )
        if isinstance(arg, ast.Attribute):
            return (
                arg.attr == "open"
                or arg.attr in _AUTO_UNSAFE_PY_ATTRS
                or arg.attr in _AUTO_UNSAFE_PY_WRITE_METHODS
                or arg.attr in _ARCHIVE_CTOR_NAMES
            )
        return False

    def _passed_write_callable(arg) -> bool:
        # A concrete write callable handed as an argument to another call: a
        # name bound to open / a writer / an archive constructor, or an
        # attribute reference to a writer method / mutating os attr / archive
        # ctor / .open. Unlike _wraps_write_callable this omits the fail-closed
        # dynamic / getattr / code-exec poison aliases, which are already gated
        # where they are *called* and would over-trigger when a benign alias is
        # merely passed or printed (print(getattr(o, 'name'))).
        if isinstance(arg, ast.Name):
            return (
                arg.id in open_aliases or arg.id in writer_aliases or arg.id in archive_ctor_aliases
            )
        if isinstance(arg, ast.Attribute):
            return (
                arg.attr == "open"
                or arg.attr in _AUTO_UNSAFE_PY_ATTRS
                or arg.attr in _AUTO_UNSAFE_PY_WRITE_METHODS
                or arg.attr in _ARCHIVE_CTOR_NAMES
            )
        return False

    # Names bound more than once cannot be folded to a single literal: this scan
    # visits every assignment before any call is checked, so a later benign
    # reassignment (base = '/etc'; open(base + '/passwd'); base = 'data') would
    # otherwise mask the earlier sensitive value and auto-approve. Count every
    # binding target up front and poison multiply-bound names to the escape
    # sentinel so any path folded from them fails closed (asks) instead.
    assign_counts: "dict[str, int]" = {}
    for node in ast.walk(tree):
        binding_targets = []
        if isinstance(node, ast.Assign):
            binding_targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            binding_targets = [node.target]
        for target in binding_targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    assign_counts[sub.id] = assign_counts.get(sub.id, 0) + 1
    multi_assigned_names = {name for name, count in assign_counts.items() if count > 1}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or "builtins")
                elif alias.name in ("os", "posix"):
                    os_aliases.add(alias.asname or alias.name)
                elif alias.name in _AUTO_UNSAFE_PY_LOAD_MODULES:
                    load_module_aliases.add(alias.asname or alias.name)
                elif alias.name == "operator":
                    operator_aliases.add(alias.asname or "operator")
                elif alias.name == "fileinput":
                    fileinput_aliases.add(alias.asname or "fileinput")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "operator":
                for alias in node.names:
                    if alias.name == "methodcaller":
                        methodcaller_aliases.add(alias.asname or "methodcaller")
            if node.module == "logging":
                for alias in node.names:
                    if alias.name == "basicConfig":
                        basicconfig_aliases.add(alias.asname or "basicConfig")
            if node.module == "builtins":
                for alias in node.names:
                    if alias.name == "open":
                        open_aliases.add(alias.asname or "open")
                    elif alias.name in code_exec_aliases:
                        code_exec_aliases.add(alias.asname or alias.name)
            if node.module in _OPEN_ALIAS_MODULES:
                for alias in node.names:
                    if alias.name == "open":
                        # gzip/bz2/lzma open(file, mode) writes on "w"/"a"/"x",
                        # mode in the 2nd arg like builtin open.
                        open_aliases.add(alias.asname or "open")
            if node.module == "pathlib":
                for alias in node.names:
                    if alias.name in _PATH_CTORS:
                        path_ctor_aliases.add(alias.asname or alias.name)
            if node.module in ("os.path", "posixpath", "ntpath"):
                for alias in node.names:
                    if alias.name == "join":
                        pathjoin_aliases.add(alias.asname or "join")
            if node.module == "functools":
                for alias in node.names:
                    if alias.name == "partial":
                        partial_aliases.add(alias.asname or "partial")
            if node.module in _ARCHIVE_CTOR_MODULES:
                _ctor = _ARCHIVE_CTOR_MODULES[node.module]
                for alias in node.names:
                    if alias.name == _ctor:
                        archive_ctor_aliases.add(alias.asname or _ctor)
            for alias in node.names:
                if alias.name in _AUTO_UNSAFE_PY_WRITE_METHODS:
                    writer_aliases.add(alias.asname or alias.name)
                # from itertools import starmap as sm / from functools import
                # reduce as r: an aliased higher-order invoker.
                if alias.name in _HIGHER_ORDER_INVOKERS:
                    invoker_aliases.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            value = node.value
            # AnnAssign (f: object = open) has a single target, no destructuring.
            if isinstance(node, ast.AnnAssign):
                assign_targets = [node.target]
            else:
                assign_targets = node.targets
            targets = [t.id for t in assign_targets if isinstance(t, ast.Name)]
            attr_targets = [t.attr for t in assign_targets if isinstance(t, ast.Attribute)]
            if isinstance(value, ast.Name) and value.id in open_aliases:
                open_aliases.update(targets)
                attr_open_aliases.update(attr_targets)  # box.f = open
            elif isinstance(value, ast.Name) and value.id in getattr_aliases:
                getattr_aliases.update(targets)  # g = getattr
            elif isinstance(value, ast.Name) and value.id in partial_aliases:
                partial_aliases.update(targets)  # p = partial
            elif isinstance(value, ast.Name) and value.id in writer_aliases:
                writer_aliases.update(targets)  # s = save (numpy save alias)
            elif isinstance(value, ast.Name) and value.id in archive_ctor_aliases:
                archive_ctor_aliases.update(targets)  # z = ZipFile
            elif isinstance(value, ast.Name) and value.id in invoker_aliases:
                invoker_aliases.update(targets)  # m = map
            elif isinstance(value, ast.Name) and value.id in path_ctor_aliases:
                path_ctor_aliases.update(targets)  # P = Path
            elif isinstance(value, ast.Name) and value.id in pathjoin_aliases:
                pathjoin_aliases.update(targets)  # j = join
            elif isinstance(value, ast.Attribute) and value.attr == "join":
                pathjoin_aliases.update(targets)  # j = os.path.join
            elif isinstance(value, ast.Attribute) and value.attr in _PATH_CTORS:
                path_ctor_aliases.update(targets)  # P = pathlib.Path
            elif (
                isinstance(value, ast.Attribute)
                and value.attr == "open"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            ):
                open_aliases.update(targets)  # f = builtins.open
            elif (
                isinstance(value, ast.Attribute)
                and value.attr in code_exec_aliases
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            ):
                code_exec_aliases.update(targets)  # e = builtins.eval
            elif isinstance(value, ast.Attribute) and value.attr in _AUTO_UNSAFE_PY_WRITE_METHODS:
                writer_aliases.update(targets)  # s = np.save
            elif isinstance(value, ast.Attribute) and value.attr == "open":
                # A captured .open bound method (p = Path('out').open) opens a file
                # on any call; its mode position varies (Path.open mode is 1st arg,
                # builtin open's is 2nd), so fail closed on the call rather than
                # guess the write mode.
                dynamic_aliases.update(targets)  # p = Path('out').open; p('w')
            elif isinstance(value, ast.Attribute) and value.attr in _ARCHIVE_CTOR_NAMES:
                archive_ctor_aliases.update(targets)  # z = zipfile.ZipFile
            elif isinstance(value, ast.Subscript):
                dynamic_aliases.update(targets)  # f = globals()["open"]
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in getattr_aliases
            ):
                dynamic_aliases.update(targets)  # rm = getattr(os, "remove") / g(...)
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in ("get", "pop", "setdefault")
                and _is_dynamic_namespace(value.func.value)
            ):
                # f = __builtins__.__dict__.get("open") / globals().get("open"):
                # a namespace lookup can return open/eval, so poison like getattr.
                dynamic_aliases.update(targets)
            elif (
                isinstance(value, ast.Call)
                and (
                    (isinstance(value.func, ast.Name) and value.func.id in partial_aliases)
                    or (isinstance(value.func, ast.Attribute) and value.func.attr == "partial")
                )
                and value.args
                and _wraps_write_callable(value.args[0])
            ):
                dynamic_aliases.update(targets)  # w = partial(open, mode="w")
            elif (
                isinstance(value, ast.Call)
                and (
                    (isinstance(value.func, ast.Name) and value.func.id in methodcaller_aliases)
                    or (
                        isinstance(value.func, ast.Attribute)
                        and value.func.attr == "methodcaller"
                        and isinstance(value.func.value, ast.Name)
                        and value.func.value.id in operator_aliases
                    )
                )
                and _methodcaller_writes(value)
            ):
                dynamic_aliases.update(targets)  # w = methodcaller("write_text", ...)
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                # base = '/etc' -> resolve base in a later folded path. A name
                # bound more than once is poisoned (\x02) so it fails closed.
                for t in targets:
                    literal_str_vars[t] = "\x02" if t in multi_assigned_names else value.value
            elif isinstance(value, (ast.Call, ast.BinOp, ast.Name, ast.JoinedStr)):
                # p = Path('/etc'); q = p; r = os.path.join('/etc','x'): record a
                # fully-literal folded path so a later reuse (p / 'passwd') folds.
                folded = _folded_path(value, literal_str_vars, path_ctor_aliases, pathjoin_aliases)
                if folded is not None and "\x00" not in folded and "\x02" not in folded:
                    for t in targets:
                        literal_str_vars[t] = "\x02" if t in multi_assigned_names else folded
            elif isinstance(value, (ast.Tuple, ast.List)):
                # Destructuring binds each element like a single assignment, so an
                # aliased callable (f, _ = (open, print)) AND a string / path
                # literal (base, leaf = ('/etc', 'passwd')) both propagate; without
                # the latter a path folded from base/leaf would miss the sensitive
                # target and auto-approve.
                for target in assign_targets:
                    if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(
                        value.elts
                    ):
                        for tgt_el, val_el in zip(target.elts, value.elts):
                            if not isinstance(tgt_el, ast.Name):
                                continue
                            tid = tgt_el.id
                            if isinstance(val_el, ast.Name) and val_el.id in open_aliases:
                                open_aliases.add(tid)
                            elif isinstance(val_el, ast.Name) and val_el.id in getattr_aliases:
                                getattr_aliases.add(tid)
                            elif isinstance(val_el, ast.Name) and val_el.id in partial_aliases:
                                partial_aliases.add(tid)
                            elif isinstance(val_el, ast.Name) and val_el.id in writer_aliases:
                                writer_aliases.add(tid)  # s, _ = (save, 1)
                            elif isinstance(val_el, ast.Name) and val_el.id in archive_ctor_aliases:
                                archive_ctor_aliases.add(tid)  # z, _ = (ZipFile, 1)
                            elif isinstance(val_el, ast.Constant) and isinstance(val_el.value, str):
                                literal_str_vars[tid] = (
                                    "\x02" if tid in multi_assigned_names else val_el.value
                                )
                            elif isinstance(val_el, (ast.Call, ast.BinOp, ast.Name, ast.JoinedStr)):
                                folded = _folded_path(
                                    val_el, literal_str_vars, path_ctor_aliases, pathjoin_aliases
                                )
                                if (
                                    folded is not None
                                    and "\x00" not in folded
                                    and "\x02" not in folded
                                ):
                                    literal_str_vars[tid] = (
                                        "\x02" if tid in multi_assigned_names else folded
                                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # A callable captured as a parameter default (def f(o=open): o('x','w'))
            # binds that parameter to the same alias set, so a later call through
            # the parameter is still gated. defaults align to the tail of
            # posonlyargs+args; kw_defaults align 1:1 with kwonlyargs (None = none).
            _a = node.args
            _defaulted = list(
                zip(
                    (_a.posonlyargs + _a.args)[
                        len(_a.posonlyargs) + len(_a.args) - len(_a.defaults) :
                    ],
                    _a.defaults,
                )
            ) + [(p, d) for p, d in zip(_a.kwonlyargs, _a.kw_defaults) if d is not None]
            for _param, _default in _defaulted:
                if isinstance(_default, ast.Name):
                    _did = _default.id
                    if _did in open_aliases:
                        open_aliases.add(_param.arg)
                    elif _did in writer_aliases:
                        writer_aliases.add(_param.arg)
                    elif _did in archive_ctor_aliases:
                        archive_ctor_aliases.add(_param.arg)
                    elif _did in getattr_aliases:
                        getattr_aliases.add(_param.arg)
                    elif _did in partial_aliases:
                        partial_aliases.add(_param.arg)
                    elif _did in code_exec_aliases:
                        code_exec_aliases.add(_param.arg)
                    elif _did in dynamic_aliases:
                        dynamic_aliases.add(_param.arg)
                elif isinstance(_default, ast.Attribute):
                    # An attribute writer / archive ctor / captured .open used as
                    # a default (def f(s=np.save), def f(z=zipfile.ZipFile),
                    # def f(o=Path('x').open)) binds the parameter like the
                    # equivalent assignment; a benign attribute (np.mean) does not.
                    if _default.attr in _AUTO_UNSAFE_PY_WRITE_METHODS:
                        writer_aliases.add(_param.arg)
                    elif _default.attr in _ARCHIVE_CTOR_NAMES:
                        archive_ctor_aliases.add(_param.arg)
                    elif _default.attr == "open":
                        dynamic_aliases.add(_param.arg)
                elif (
                    isinstance(_default, ast.Call)
                    and (
                        (
                            isinstance(_default.func, ast.Name)
                            and _default.func.id in partial_aliases
                        )
                        or (
                            isinstance(_default.func, ast.Attribute)
                            and _default.func.attr == "partial"
                        )
                    )
                    and _default.args
                    and _wraps_write_callable(_default.args[0])
                ):
                    dynamic_aliases.add(_param.arg)  # def f(w=partial(open, mode="w"))
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _AUTO_UNSAFE_PY_MODULES:
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _AUTO_UNSAFE_PY_MODULES:
                    return True
                # from-imports can bind mutating callables to bare names
                # (from os import remove [as rm]); star imports hide anything.
                for alias in node.names:
                    if alias.name == "*" or alias.name in _AUTO_UNSAFE_PY_ATTRS:
                        return True
                    # os.open imported as a bare callable is a low-level
                    # create/write, like the os.open attribute call below.
                    if alias.name == "open" and node.module in ("os", "posix"):
                        return True
            elif isinstance(node, ast.Attribute):
                # Any reference to a mutating attribute fails closed, even
                # without an immediate call (rm = os.remove; rm("x")).
                if node.attr in _AUTO_UNSAFE_PY_ATTRS:
                    return True
                # builtins.exec / builtins.eval / builtins.__import__ (and
                # compile/breakpoint) are dynamic code execution, matching the
                # bare-name code_exec_aliases path; __builtins__.__import__(...)
                # is a dynamic import that dodges the static import check.
                if (
                    node.attr in ("exec", "eval", "__import__", "breakpoint", "compile")
                    and isinstance(node.value, ast.Name)
                    and node.value.id in builtins_aliases
                ):
                    return True
            elif isinstance(node, ast.Name):
                if node.id in code_exec_aliases:
                    return True
            elif isinstance(node, ast.Constant):
                # Credential paths / parent traversal in a string or bytes
                # literal (open('/etc/passwd') and open(b'/etc/passwd')), or a
                # glob that resolves to one (glob.glob('/e??/passwd')).
                val = node.value
                if isinstance(val, bytes):
                    val = val.decode("latin-1", "ignore")
                if isinstance(val, str) and (
                    _references_sensitive_path(val) or _glob_token_sensitive(val)
                ):
                    return True
            elif isinstance(node, (ast.BinOp, ast.JoinedStr)):
                # A sensitive path concatenated from literals ('/etc'+'/passwd'),
                # a pathlib / chain, an f-string (f'/proc/{pid}/environ'), a
                # dynamic segment under a sensitive dir (f'/etc/{name}'), or one
                # split through a literal variable (base = '/etc'; base+'/passwd').
                if _folded_is_sensitive(
                    _folded_path(node, literal_str_vars, path_ctor_aliases, pathjoin_aliases)
                ):
                    return True
            elif isinstance(node, ast.Call):
                # A sensitive path composed via os.path.join('/etc', name).
                if _folded_is_sensitive(
                    _folded_path(node, literal_str_vars, path_ctor_aliases, pathjoin_aliases)
                ):
                    return True
                func = node.func
                # x.__call__(args) is just x(args): unwrap so open.__call__('o',
                # 'w') / save.__call__(...) reach the open/writer checks below
                # instead of looking like a harmless ".__call__" attribute call.
                if isinstance(func, ast.Attribute) and func.attr == "__call__":
                    func = func.value
                if isinstance(func, (ast.Call, ast.Subscript)):
                    return True  # calling a call/subscript result is dynamic
                # A concrete write callable (open/writer/archive-ctor alias, or a
                # writer/mutating attribute) handed as an argument to any call
                # escapes into a helper that can invoke it without a direct
                # open()/writer site -- the same bypass the map/starmap/reduce
                # branches below gate, but through a user-defined helper
                # (def run(fn): fn('o','w').write('x'); run(open)). A benign
                # callable argument (run(len)) is unaffected.
                if any(_passed_write_callable(a) for a in node.args) or any(
                    _passed_write_callable(kw.value) for kw in node.keywords
                ):
                    return True
                if isinstance(func, ast.Name):
                    if func.id in dynamic_aliases:
                        return True  # call through a getattr alias is dynamic
                    if func.id in open_aliases and _builtin_open_writes(node):
                        return True
                    # A writer imported as a bare name (from numpy import save).
                    if func.id in writer_aliases:
                        return True
                    # A bare archive constructor (from zipfile import ZipFile)
                    # takes the mode as its 2nd arg like open, so ZipFile(x, "w")
                    # writes but ZipFile(x) reads.
                    if func.id in archive_ctor_aliases and _builtin_open_writes(node):
                        return True
                    # A bare-imported logging.basicConfig(filename=...) opens a
                    # log file for writing (from logging import basicConfig).
                    if func.id in basicconfig_aliases and _basicconfig_writes(node):
                        return True
                    # A writer/open alias handed to a higher-order invoker
                    # (map(open, names, modes), starmap(np.save, ...), or an
                    # aliased m = map / sm = starmap) is called without a direct
                    # open(...)/save(...) site; the callable is the first
                    # positional arg. A benign map(len, ...) is unaffected.
                    if (
                        func.id in invoker_aliases
                        and node.args
                        and _wraps_write_callable(node.args[0])
                    ):
                        return True
                elif isinstance(func, ast.Attribute):
                    # Writer methods persist to disk without open() (np.save,
                    # img.save, plt.savefig, df.to_csv, json.dump); ask before
                    # they mutate the workdir in auto mode.
                    if func.attr in _AUTO_UNSAFE_PY_WRITE_METHODS:
                        return True
                    # logging.basicConfig(filename=...) opens a log file for write.
                    if func.attr == "basicConfig" and _basicconfig_writes(node):
                        return True
                    # A qualified higher-order invoker (itertools.starmap(open, ...),
                    # functools.reduce(open, ...)) calls its first arg like the bare
                    # map/filter form; the writer-check on that arg keeps a benign
                    # itertools.starmap(len, ...) / df.map(transform) safe.
                    if (
                        func.attr in _HIGHER_ORDER_INVOKERS
                        and node.args
                        and _wraps_write_callable(node.args[0])
                    ):
                        return True
                    # fileinput.input(..., inplace=True) rewrites a file in place;
                    # the default fileinput.input(...) only reads, so gate inplace.
                    if (
                        func.attr == "input"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in fileinput_aliases
                        and _fileinput_inplace(node)
                    ):
                        return True
                    # os.open() always creates/writes a file descriptor
                    # (tracked through import aliases: import os as o; o.open()).
                    if (
                        func.attr == "open"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in os_aliases
                    ):
                        return True
                    # A pickle-backed loader (torch.load, joblib.load) can execute
                    # code embedded in the file it deserializes.
                    if (
                        func.attr == "load"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in load_module_aliases
                    ):
                        return True
                    if func.attr == "open" and _attr_open_writes(node):
                        return True
                    # An open bound onto an attribute (box.f = open; box.f('o','w'))
                    # writes on 'w'/'a'/'x' like the builtin, so gate the attr name.
                    if func.attr in attr_open_aliases and _builtin_open_writes(node):
                        return True
                    # ZipFile/TarFile/GzipFile/BZ2File/LZMAFile take the mode as
                    # the 2nd arg (like builtin open), so ZipFile(name, "w") writes
                    # but ZipFile(name) reads.
                    if func.attr in _ARCHIVE_CTOR_NAMES and _builtin_open_writes(node):
                        return True
                    # Enumerating a directory outside the sandbox reads host
                    # filenames (and enables reading their contents) the direct
                    # /etc/passwd checks would prompt for: Path('/etc').iterdir(),
                    # os.scandir('/etc'), os.listdir('/home'), os.walk('/'),
                    # Path('/home').glob('*'), glob.glob('/home/*'). Gate when the
                    # target dir folds to an absolute/tilde/sensitive path; a
                    # relative dir (Path('.').iterdir(), glob.glob('src/*')) stays
                    # safe, and an unresolved dynamic dir is left to other checks.
                    _enum_dir = None
                    if func.attr == "iterdir":
                        _enum_dir = func.value
                    elif func.attr in ("glob", "rglob", "iglob"):
                        # Path('/home').glob('*') enumerates the receiver dir;
                        # glob.glob('/home/*') enumerates the pattern's root dir.
                        _recv = _folded_path(
                            func.value, literal_str_vars, path_ctor_aliases, pathjoin_aliases
                        )
                        if isinstance(_recv, str) and _recv not in ("", "\x00"):
                            _enum_dir = func.value
                        elif node.args:
                            _enum_dir = node.args[0]
                    elif (
                        func.attr in ("scandir", "listdir", "walk")
                        and isinstance(func.value, ast.Name)
                        and func.value.id in os_aliases
                        and node.args
                    ):
                        _enum_dir = node.args[0]
                    if _enum_dir is not None:
                        _folded_dir = _folded_path(
                            _enum_dir, literal_str_vars, path_ctor_aliases, pathjoin_aliases
                        )
                        if isinstance(_folded_dir, str) and (
                            _folded_dir.startswith("/")
                            or _folded_dir.startswith("~")
                            or _folded_is_sensitive(_folded_dir)
                        ):
                            return True
    except Exception:
        return True  # unexpected AST shape: fail closed
    return False


# Cloud-metadata / link-local hosts (mirrors the sandbox SSRF blocklist): a
# read-named HTTP MCP tool pointed at one (fetch_url
# {"url": "http://169.254.169.254/..."}) reads instance credentials, so it asks.
_MCP_METADATA_HOST_RE = re.compile(
    r"169\.254\.\d{1,3}\.\d{1,3}|"
    r"100\.100\.100\.\d{1,3}|"
    r"fd00:ec2::254|"
    r"metadata\.google\.internal|"
    r"metadata\.tencentyun\.com|"
    r"://metadata(?=[:/])",
    re.IGNORECASE,
)


def _mcp_arguments_reference_sensitive(arguments) -> bool:
    """True if any string in an MCP call's arguments names a credential path, a
    credential/secret environment variable (get_env {"name": "OPENAI_API_KEY"}),
    or a cloud-metadata host (fetch_url {"url": "http://169.254.169.254/..."})."""

    def walk(value) -> bool:
        if isinstance(value, str):
            return (
                _references_sensitive_path(value)
                or bool(_AUTO_SENSITIVE_MCP_NOUN_RE.search(value))
                or bool(_MCP_METADATA_HOST_RE.search(value))
            )
        if isinstance(value, dict):
            return any(walk(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(walk(v) for v in value)
        return False

    return walk(arguments)


# DDL object types CREATE / DROP / ALTER share (DROP FUNCTION and ALTER INDEX
# mutate just like CREATE INDEX).
_SQL_DDL_OBJECTS = (
    r"table|database|schema|index|view|function|procedure|trigger|"
    r"sequence|role|user|extension|type|domain|aggregate|policy"
)
# Modifiers between the DDL verb and object (CREATE OR REPLACE VIEW, DROP
# MATERIALIZED VIEW, CREATE UNIQUE INDEX).
_SQL_DDL_MODIFIERS = (
    r"(?:(?:or\s+replace|unique|temp|temporary|global|local|materialized|recursive)\s+)*"
)
# A SQL identifier (bare, "quoted", `quoted`, [bracketed]), optionally
# schema-qualified, so UPDATE "users"/public.users/ONLY .../[users] SET all hit.
_SQL_IDENT = r'(?:\w+|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]+\])'
_SQL_UPDATE_TARGET = r"(?:only\s+)?" + _SQL_IDENT + r"(?:\s*\.\s*" + _SQL_IDENT + r")*"
# A read-named MCP tool (query_database, run_query) can still carry a mutating
# SQL statement; match DML/DDL as whole statements (DELETE FROM, DROP TABLE) so
# a natural-language query that merely contains the word "delete" stays safe.
_MCP_ARG_MUTATION_RE = re.compile(
    r"\b(?:delete\s+from|"
    r"drop\s+" + _SQL_DDL_MODIFIERS + r"(?:" + _SQL_DDL_OBJECTS + r")|"
    # Match the whole identifier (the outer trailing \b needs the alternative to
    # end on a word boundary, so a bare \w stops mid-name and TRUNCATE users slips
    # through); the optional opening quote/bracket/backtick covers "users"/[users].
    r"truncate\s+(?:table\s+)?[\"\[`]?\w+|"
    # UPDATE <target> [AS alias] SET: allow an explicit AS alias before SET so
    # UPDATE users AS u SET is caught, not just the bare form. The implicit-alias
    # form (UPDATE users u SET) is left out because it is indistinguishable from
    # the prose "update <noun> <noun> set" and would flag natural language.
    r"update\s+" + _SQL_UPDATE_TARGET + r"(?:\s+as\s+" + _SQL_IDENT + r")?\s+set\b|"
    r"insert\s+into|replace\s+into|"
    # SELECT ... INTO OUTFILE/DUMPFILE writes a file (MySQL); bare SELECT INTO
    # <table> is left out (PL/pgSQL uses it to read into a variable).
    r"select\s+[^;]*?\binto\s+(?:outfile|dumpfile)\b|"
    # ALTER SYSTEM persists PostgreSQL server configuration; SYSTEM is not one of
    # the DDL objects above, so match it explicitly.
    r"alter\s+system\b|"
    r"alter\s+" + _SQL_DDL_MODIFIERS + r"(?:" + _SQL_DDL_OBJECTS + r")|"
    r"create\s+" + _SQL_DDL_MODIFIERS + r"(?:" + _SQL_DDL_OBJECTS + r")|"
    r"grant\s+\w+|revoke\s+\w+|merge\s+into|"
    # Catalog mutations: COMMENT ON <obj>, SECURITY LABEL, and LOCK TABLE change
    # metadata or take a lock. Each needs a following keyword, so a "comment"
    # column (SELECT comment FROM t) or "locks" table stays safe.
    r"comment\s+on\b|security\s+label\b|lock\s+table\b|"
    # PostgreSQL maintenance writes: REFRESH MATERIALIZED VIEW rewrites the view,
    # REINDEX rebuilds an index. Both need a following object keyword/name, so a
    # column or word "refresh"/"reindex" in prose stays safe.
    r"refresh\s+materialized\s+view|reindex\s+\w+|"
    # CALL proc(...) / EXEC[UTE] name / VACUUM mutate; CALL needs a following
    # "(", ";", or end so natural-language "call me back" stays safe.
    r"call\s+\w+(?=\s*[(;]|\s*$)|exec(?:ute)?\s+\w+|vacuum|"
    # COPY ... FROM bulk-loads and COPY ... TO writes a file ([^;] stays in one
    # statement).
    r"copy\s+[^;]*?\b(?:from|to)\b)\b",
    re.IGNORECASE,
)
# SQLite statements the base regex misses: ATTACH/DETACH a database (DATABASE
# optional via the quoted-path form), a write-form PRAGMA (name=value / name(...),
# unlike the read-form PRAGMA name), and load_extension() which runs a shared
# library. These tokens are not natural language, so benign text does not trip.
_MCP_ARG_SQLITE_MUTATION_RE = re.compile(
    r"\b(?:attach|detach)\s+database\b"
    r"|\battach\s+(?:database\s+)?['\"]"
    r"|\bpragma\s+\w+(?:\.\w+)?\s*(?:=|\()"
    r"|\bload_extension\s*\(",
    re.IGNORECASE,
)
# State-changing SQL functions that mutate or write files inside a read-shaped
# SELECT (pg_terminate_backend, setval, pg_write_file, lo_export, ...). The
# trailing "(" is required, so a column named setval_count stays safe.
_MCP_ARG_SQL_FUNCTION_RE = re.compile(
    r"\b(?:pg_terminate_backend|pg_cancel_backend|pg_write_file|lo_export|"
    r"lo_import|setval|nextval|set_config|pg_notify|dblink_exec|pg_reload_conf|"
    r"pg_rotate_logfile|"
    # advisory locks change session/transaction lock state (read-shaped SELECT).
    r"pg_advisory_(?:lock|lock_shared|unlock|unlock_shared|unlock_all|"
    r"xact_lock|xact_lock_shared)|"
    r"pg_try_advisory_(?:lock|lock_shared|xact_lock|xact_lock_shared))\s*\(",
    re.IGNORECASE,
)
# SQL engines treat /* */ and -- comments as whitespace, so DELETE/**/FROM and
# UPDATE/**/users evade the \s+ in the mutation regex; collapse comments to a
# space before matching.
_SQL_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)
# A GraphQL mutation on a read-named tool. Directives are valid between the name
# and body (mutation M @audit { ... }), so allow @directive[(args)] before ( or {.
_GRAPHQL_MUTATION_RE = re.compile(
    r"\bmutation\b\s*\w*\s*(?:@\w+(?:\s*\([^)]*\))?\s*)*[({]", re.IGNORECASE
)
# GraphQL # comments run to end-of-line and count as whitespace, so a comment
# between `mutation` and the body (mutation # note\n { ... }) would otherwise
# hide it; collapse them to a space before matching.
_GRAPHQL_COMMENT_RE = re.compile(r"#[^\n]*")


# HTTP verbs that mutate the target resource; a generic HTTP MCP tool
# (mcp__http__get_url {"method": "DELETE"}) mutates an external service even
# though its name looks read-only. GET/HEAD/OPTIONS/TRACE only read.
_MUTATING_HTTP_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_HTTP_METHOD_KEYS = frozenset({"method", "http_method", "httpmethod", "verb", "http_verb"})


def _mcp_arguments_mutate(arguments) -> bool:
    """True if an MCP call's arguments carry a mutating command, so a read-named
    but write-capable tool (query_database {"query": "DELETE FROM runs"},
    query_graphql {"query": "mutation { deleteIssue(id: 1) }"}, or an HTTP tool
    {"method": "DELETE"}) asks."""

    def walk(value) -> bool:
        if isinstance(value, str):
            _sql = _SQL_COMMENT_RE.sub(" ", value)
            return (
                bool(_MCP_ARG_MUTATION_RE.search(_sql))
                or bool(_MCP_ARG_SQLITE_MUTATION_RE.search(_sql))
                or bool(_MCP_ARG_SQL_FUNCTION_RE.search(_sql))
                or bool(_GRAPHQL_MUTATION_RE.search(_GRAPHQL_COMMENT_RE.sub(" ", value)))
            )
        if isinstance(value, dict):
            for k, v in value.items():
                if (
                    isinstance(k, str)
                    and k.lower() in _HTTP_METHOD_KEYS
                    and isinstance(v, str)
                    and v.strip().upper() in _MUTATING_HTTP_METHODS
                ):
                    return True
            return any(walk(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(walk(v) for v in value)
        return False

    return walk(arguments)


# Tools that are read-only / non state-mutating regardless of their arguments,
# so auto mode never has to pause them (their safety needs no argument scan).
# render_html is NOT unconditionally safe: it runs arbitrary HTML/JS in the
# canvas preview frame. A static canvas (charts, layout, inline SVG) never
# reaches the network, but code that calls out can exfiltrate or fetch under the
# preview's CSP when artifact network access is enabled, so those ask; a canvas
# with no network construct still auto-runs. Matches JS egress APIs, a remote or
# root-relative <script src>/src=/href=/srcset, a CSS url()/@import that loads a
# resource, and ws(s) URLs. A leading "/" covers both //host (protocol-relative)
# and /path (root-relative, which the CSP resolves against the frame origin); a
# "./x" or bare relative ref and a url(#id)/data: ref are not matched, so an
# inline-SVG canvas (whose w3.org namespace lives in xmlns=) stays safe.
_RENDER_HTML_NETWORK_RE = re.compile(
    r"\bfetch\s*\(|"
    r"XMLHttpRequest|"
    r"\bWebSocket\b|"
    r"\bEventSource\b|"
    r"\bsendBeacon\b|"
    r"\bimportScripts\b|"
    r"navigator\s*\.\s*serviceWorker|"
    # new Worker(...) / new SharedWorker(...) run a script off the main thread
    # that this static scan cannot see: a module worker from a CORS-enabled CDN
    # executes remote code, and a blob/same-origin worker can fetch/importScripts
    # to egress, all reachable under worker-src http: https: blob:. Gate the
    # constructor like importScripts/serviceWorker; a var merely named myWorker
    # (no "new") stays static.
    r"\bnew\s+(?:Shared)?Worker\s*\(|"
    r"@import|"
    r"url\(\s*[\"']?\s*(?:https?:|/)|"
    r"<script[^>]*\bsrc\s*=|"
    r"\b(?:src|href|srcset)\s*=\s*[\"']?\s*(?:https?:|/)|"
    # Self-navigation sinks: location.assign/replace(...), window.open(...), and
    # assigning a URL to (window.)location(.href). location.reload()/history.back
    # do not navigate to a new URL, so they stay static.
    r"\blocation\s*\.\s*(?:assign|replace)\s*\(|"
    r"\bwindow\s*\.\s*open\s*\(|"
    r"\b(?:window\s*\.\s*)?location(?:\s*\.\s*href)?\s*=\s*[\"'`]?\s*(?:https?:|/)|"
    # Bracket-access obfuscation: window['fetch'](...), self["open"](...).
    r"\[\s*[\"'](?:fetch|open|XMLHttpRequest|WebSocket|EventSource|importScripts|"
    r"sendBeacon|serviceWorker)[\"']\s*\]|"
    # Computed bracket key spliced at runtime on a global host object
    # (window['fet'+'ch'](...)): a quoted fragment adjacent to a + inside the
    # index. Anchored to a host object so a plain obj['a'+'b'] key stays safe.
    r"\b(?:window|self|globalThis|top|parent|frames)\s*\[[^\]]*"
    r"(?:[\"']\s*\+|\+\s*[\"'])[^\]]*\]|"
    # Declarative meta-refresh navigation to a URL (order-tolerant); a bare
    # content="30" self-reload has no url= and stays static.
    r"<meta\b(?=[^>]*http-equiv\s*=\s*[\"']?\s*refresh)(?=[^>]*\burl\s*=)|"
    r"\bwss?://",
    re.IGNORECASE,
)
# Block comments can split an egress token (fetch/*x*/(...)); strip them before
# matching. Line // comments are left alone -- stripping them would eat the // in
# an https:// URL and hide a real load.
_JS_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _render_html_reaches_network(arguments: dict) -> bool:
    code = arguments.get("code")
    if not isinstance(code, str):
        return False
    return bool(_RENDER_HTML_NETWORK_RE.search(_JS_BLOCK_COMMENT_RE.sub("", code)))


# Tools that are read-only regardless of their arguments, so auto mode never has
# to pause them and their safety needs no argument scan. render_html is handled
# separately above because a networked canvas does need approval.
_ALWAYS_SAFE_TOOLS = frozenset({"web_search", "search_knowledge_base"})


def is_always_safe_tool(name: str) -> bool:
    """True for tools that never need an auto-mode prompt on any arguments, so a
    caller (e.g. the streaming provisional card) can allow them before the full
    arguments are known. render_html is intentionally excluded: a networked
    canvas needs approval, which cannot be judged before its arguments stream."""
    return name in _ALWAYS_SAFE_TOOLS


def is_potentially_unsafe_tool_call(name: str, arguments: dict) -> bool:
    """Whether a tool call must still pause for approval in auto mode.

    Used by permission_mode="auto" ("Approve for me"): read-only calls
    auto-run, anything that can mutate state, execute arbitrary code, or is
    simply unrecognized asks first. Unknown tools fail closed.
    """
    if name in _ALWAYS_SAFE_TOOLS:
        return False
    # render_html auto-runs a static canvas but asks once its HTML/JS reaches the
    # network (fetch/WebSocket/remote script), which can egress under the canvas
    # CSP when artifact network access is enabled.
    if name == "render_html":
        return _render_html_reaches_network(arguments)
    if name.startswith(MCP_TOOL_PREFIX):
        tool_name = name.split("__", 2)[-1]
        # A mutating verb anywhere (get_or_create_issue, read_and_delete)
        # overrides a read-only prefix.
        if _AUTO_UNSAFE_MCP_VERB_RE.search(tool_name):
            return True
        # A credential noun (read_secret, list_tokens, get_credentials) makes a
        # read-named tool a sensitive disclosure, so it asks too.
        if _AUTO_SENSITIVE_MCP_NOUN_RE.search(tool_name):
            return True
        # A read-named fs tool pointed at a credential path is still a
        # sensitive read (mcp__fs__read_file {"path": "/etc/passwd"}).
        if _mcp_arguments_reference_sensitive(arguments):
            return True
        # A read-named tool carrying a mutating query (query_database
        # {"query": "DELETE FROM runs"}) still mutates external state.
        if _mcp_arguments_mutate(arguments):
            return True
        return not _AUTO_SAFE_MCP_TOOL_RE.match(tool_name)
    if name == "terminal":
        return _terminal_is_potentially_unsafe(str(arguments.get("command", "")))
    if name == "python":
        return _python_is_potentially_unsafe(str(arguments.get("code", "")))
    return True


def _build_safe_env(workdir: str) -> dict[str, str]:
    """Build a minimal, credential-free environment for sandboxed subprocesses.

    Whitelist-built from scratch (parent env NOT inherited): only PATH/HOME/
    TMPDIR/LANG/TERM/PYTHONIOENCODING/PYTHONPATH (+VIRTUAL_ENV or Windows
    SystemRoot) reach the child; all credential vars (HF_TOKEN, AWS_*, etc.)
    are absent. HOME points at the sandbox workdir so SDKs can't read the
    operator's cached creds. PYTHONPATH carries only the sandbox sitecustomize
    shim directory.
    """
    # Start from the running interpreter's dir so 'python'/'pip' resolve to the
    # same environment the Unsloth server runs in.
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
        # sitecustomize shim: remaps ChatGPT code-interpreter paths (/mnt/data
        # etc.) onto the sandbox CWD; see sandbox_site/sitecustomize.py.
        "PYTHONPATH": _SANDBOX_SITE_DIR,
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
        # Auth brokers / capability handles: hand the child the operator's live
        # agent (ssh/gpg), kube config, or docker daemon. Listed by name (no
        # value signal). URL config vars are NOT name-listed: a credentialed
        # value is dropped by _is_secret_env_value() regardless of name.
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
    # Azure App Service connection strings carry DB/storage credentials.
    "CONNSTR",
    "CONNECTIONSTRING",
)
# Non-secret hardening flags that match a secret prefix/marker but must be KEPT
# so bypass mode does not undo an operator's opt-out (e.g.
# AWS_EC2_METADATA_DISABLED blocks the AWS SDK from pulling IMDS creds).
_BYPASS_ENV_KEEP_NAMES = frozenset(
    {
        "AWS_EC2_METADATA_DISABLED",
        "AWS_EC2_METADATA_V1_DISABLED",
    }
)
# Matches a URL embedding userinfo before the host ("scheme://user:pass@host"
# and token-only forms). The userinfo must precede the first '/', so an '@' in
# a path/query does not false-positive.
_URL_USERINFO_RE = re.compile(r"://[^/\s@]+@")
# Connection-string credential fields (ADO.NET / Azure storage / Service Bus)
# whose names dodge the name classifier. The Name fields (SharedAccessKeyName=)
# don't match since "=" must follow the keyword.
_SECRET_VALUE_RE = re.compile(r"(?i)(?:password|pwd|accountkey|accesskey)\s*=\s*[^\s;]")

# Names holding no secret value but pointing SDKs at the operator's real
# home/cache/config (cached tokens, cred files), defeating the HOME repoint.
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
    """Env for bypass exec: full host env minus credential vars, with HOME/TMPDIR
    repointed at the workdir so SDKs cannot read cached creds.

    Stripping the child env is necessary but not sufficient (a same-UID child can
    read the parent's env via procfs), so callers also harden the parent (see
    _harden_parent_against_proc_env_leak).
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
    # sitecustomize path shim (see _build_safe_env). Bypass inherits the
    # operator's PYTHONPATH, so prepend rather than replace.
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (_SANDBOX_SITE_DIR, inherited_pythonpath) if part
    )
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
    so without a new session a timeout/cancel would kill the Unsloth server too.
    """
    try:
        os.setsid()
    except OSError:
        pass


# Hardening the Unsloth parent is done once (PR_SET_DUMPABLE is process-global
# and sticky); guarded so repeated bypass calls do not re-issue the prctl.
_parent_proc_hardened = False


def _harden_parent_against_proc_env_leak() -> bool:
    """Make the Unsloth process's /proc/<pid>/environ unreadable to its children.

    Stripping the child env is not enough on Linux: a bypassed same-UID child
    can read /proc/<getppid()>/environ to recover the parent's unfiltered
    secrets. Clearing PR_SET_DUMPABLE reparents this process's /proc entries to
    root, closing that read.

    Returns True when hardened or unnecessary (off Linux), False when needed but
    unappliable (e.g. prctl denied by seccomp); callers must then fail closed.
    This is a mitigation, not a full boundary - a bypassed tool can still walk
    /proc to an ancestor or read creds by path. Applied lazily on first bypass.
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

# Appended to the python/terminal descriptions: models habitually write to
# /mnt/data (a ChatGPT code-interpreter path), which does not exist here.
_SANDBOX_PATHS_NOTE = (
    " Read and write files using relative paths in the current working "
    "directory, which persists for this conversation; absolute paths like "
    "/mnt/data or /tmp/outputs do not exist."
)

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code in a sandbox and return stdout/stderr."
        + _SANDBOX_PATHS_NOTE,
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
        "description": "Execute a terminal command and return stdout/stderr." + _SANDBOX_PATHS_NOTE,
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


# OpenAI's function.name regex; MCP names that violate it would 400 the whole
# request, so validate up front and skip with a warning.
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
    # Never spawn stdio servers when stdio is disabled on this host.
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
        # An edit/delete can land while we await a probe; re-read and drop a
        # result whose server changed or was removed mid-probe, else a stale
        # tool list (or cool-off on a just-fixed server) persists.
        current = {s["id"]: s for s in mcp_servers_db.list_servers()}
        for server, payload in zip(uncached, results):
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
    thread_id: str | None = None,
    rag_scope: dict | None = None,
    disable_sandbox: bool = False,
    output_callback = None,
) -> str:
    """Execute a tool by name with the given arguments; returns a string.

    ``timeout``: int seconds, ``None`` = no limit, unset = ``_EXEC_TIMEOUT``.
    ``session_id``: optional ID for per-conversation sandbox isolation.
    ``thread_id``: optional conversation ID; scopes stateful MCP stdio sessions
    per thread (session_id alone can be shared project-wide).
    ``rag_scope``: hidden per-request RAG context the model never sees; consumed
    by ``search_knowledge_base``.
    ``disable_sandbox``: Bypass Permissions; run python/terminal without the
    safety checks, blocklist, or resource caps (secrets still stripped). Only
    affects local code tools; web_search / MCP are unchanged.
    ``output_callback``: optional ``callable(str)`` invoked with incremental
    stdout/stderr chunks while python/terminal executions run (UI live
    output). Purely observational: the returned result string is identical
    with or without it. Tools without incremental output ignore it.
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
        # Persist a stateful stdio session only per conversation (thread_id).
        # session_id is the project-wide sandbox id, so scoping by it alone leaks
        # browser/DB/REPL state across conversations; fall back to one-shot. Tag +
        # percent-quote the parts so ids can't collide or ":" merge conversations.
        if thread_id:
            mcp_scope = "s={}:t={}".format(
                urllib.parse.quote(session_id or "", safe = ""),
                urllib.parse.quote(thread_id, safe = ""),
            )
        else:
            mcp_scope = None
        headers = parse_server_headers(server)
        url = server["url"]

        def _config_current() -> bool:
            # Re-read before a stdio session is cached: this call may have read
            # the row just before an update/delete closed its sessions.
            row = mcp_servers_db.get_server(server_id)
            return (
                row is not None
                and bool(row.get("is_enabled"))
                and row.get("url") == url
                and parse_server_headers(row) == headers
            )

        return call_tool_sync(
            url = url,
            headers = headers,
            name = tool_name,
            args = arguments,
            timeout = effective_timeout,
            use_oauth = bool(server.get("use_oauth")),
            cancel_event = cancel_event,
            scope = mcp_scope,
            config_check = _config_current,
        )
    if name == "web_search":
        return _web_search(
            arguments.get("query", ""),
            url = arguments.get("url"),
            timeout = effective_timeout,
            cancel_event = cancel_event,
        )
    if name == "python":
        return _python_exec(
            arguments.get("code", ""),
            cancel_event,
            effective_timeout,
            session_id,
            disable_sandbox = disable_sandbox,
            output_callback = output_callback,
        )
    if name == "terminal":
        return _bash_exec(
            arguments.get("command", ""),
            cancel_event,
            effective_timeout,
            session_id,
            disable_sandbox = disable_sandbox,
            output_callback = output_callback,
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

    # Whole-document mode: a thread-attached file under budget is injected in
    # full. A KB selection is exclusive so whole-doc never preempts it; project
    # sources are still retrieved top-K and appended under one citation
    # numbering. Oversized/absent thread docs fall through to top-K below.
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
# Raw download cap > _MAX_PAGE_CHARS since SSR pages embed large <head> sections
# stripped during conversion; 512 KB still reaches article content.
_MAX_FETCH_BYTES = 512 * 1024
# PDF cross-reference data lives at EOF, so extraction needs the whole body.
_MAX_PDF_FETCH_BYTES = 10 * 1024 * 1024
_MAX_WEB_PDF_PAGES = 50
# Control/undecodable chars, excluding text whitespace and ESC (for ANSI logs).
# Binary when they exceed 12.5%, after allowing 16 minor encoding glitches.
_BINARY_CHAR_RE = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1a\\x1c-\\x1f\\x7f-\\x9f\\ufffd]")
_MIN_BINARY_CHARS = 16
_BINARY_CHAR_DIVISOR = 8
# Common binary signatures that can otherwise look text-heavy when mislabeled.
_PDF_MAGIC = b"%PDF-"
_BINARY_MAGIC = (
    _PDF_MAGIC,
    b"PK\x03\x04",  # zip / docx / xlsx / pptx / epub / jar
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",  # OLE / legacy Office
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"GIF87a",
    b"GIF89a",
    b"\x1f\x8b",  # gzip
    b"BZh",  # bzip2
    b"\xfd7zXZ\x00",  # xz
    b"\x28\xb5\x2f\xfd",  # zstd
)

# Check UTF-32 first because its little-endian BOM starts with the UTF-16 BOM.
_UNICODE_BOM_CODECS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)

# A cp1252 retry needs 75% ASCII structure so it cannot rescue high-byte binary.
_MIN_SINGLE_BYTE_ASCII_RATIO = 3 / 4
_ASCII_TEXT_BYTES = frozenset((*range(0x20, 0x7F), 0x09, 0x0A, 0x0D, 0x1B))


def _looks_binary(text: str) -> bool:
    """Whether control or undecodable characters exceed the binary threshold."""
    return len(_BINARY_CHAR_RE.findall(text)) > max(
        _MIN_BINARY_CHARS, len(text) // _BINARY_CHAR_DIVISOR
    )


def _magic_head(data: bytes) -> bytes:
    head = data[:1024].lstrip()
    for bom, _codec in _UNICODE_BOM_CODECS:
        if head.startswith(bom):
            head = head.removeprefix(bom).lstrip()
            break
    return head


def _has_pdf_magic(data: bytes) -> bool:
    return _magic_head(data).startswith(_PDF_MAGIC)


def _has_binary_magic(data: bytes) -> bool:
    """Whether a common binary signature follows optional BOM or whitespace."""
    return _magic_head(data).startswith(_BINARY_MAGIC)


def _has_single_byte_text_evidence(data: bytes) -> bool:
    """True when *data* has enough ASCII structure for a cp1252 text retry."""
    if not data:
        return True
    ascii_text_bytes = sum(byte in _ASCII_TEXT_BYTES for byte in data)
    return ascii_text_bytes / len(data) >= _MIN_SINGLE_BYTE_ASCII_RATIO


def _extract_pdf_text(data: bytes) -> str:
    """Extract page-delimited text with the same parser used by RAG ingestion."""
    from ..rag.parsers import parse_pdf_bytes

    pages, total_pages = parse_pdf_bytes(data, max_pages = _MAX_WEB_PDF_PAGES)
    page_limit_reached = total_pages > _MAX_WEB_PDF_PAGES
    parts: list[str] = []
    length = 0
    text_limited = False
    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue
        section = f"## Page {page.page_number}\n\n{page_text}"
        piece = ("\n\n" if parts else "") + section
        remaining = _MAX_PAGE_CHARS - length
        if len(piece) > remaining:
            parts.append(piece[:remaining])
            text_limited = True
            break
        parts.append(piece)
        length += len(piece)

    text = "".join(parts).rstrip()
    if not text:
        if page_limit_reached:
            return f"(PDF contains no extractable text in the first {_MAX_WEB_PDF_PAGES} pages)"
        return ""
    limits = []
    if text_limited:
        limits.append(f"text limited to {_MAX_PAGE_CHARS:,} characters")
    if page_limit_reached:
        limits.append(f"page processing capped at {_MAX_WEB_PDF_PAGES} pages")
    if limits:
        marker = f"\n\n... (PDF extraction {'; '.join(limits)})"
        text = text[: _MAX_PAGE_CHARS - len(marker)].rstrip() + marker
    return text


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
        # `not ip.is_global` is the source of truth (also rejects CGNAT and
        # benchmarking/doc ranges); the explicit predicates only label the error.
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


# Binary application subtypes rejected by MIME; other application types are
# sniffed so textual artifacts such as SQL stay usable.
_BINARY_APPLICATION_SUBTYPES = frozenset(
    {
        "epub+zip",
        "gzip",
        "java-archive",
        "pdf",
        "vnd.apple.installer+xml",
        "wasm",
        "x-7z-compressed",
        "x-bzip2",
        "x-gzip",
        "x-rar-compressed",
        "x-tar",
        "x-xz",
        "zip",
        "zstd",
    }
)


def _is_text_candidate_content_type(content_type: str | None) -> bool:
    """Whether a MIME type is textual or ambiguous enough for byte sniffing."""
    match = re.match(r"[\w.+-]+/[\w.+-]+", content_type or "")
    if not match:
        return True
    ct = match.group(0).lower()
    if ct.startswith("text/"):
        return True
    if ct.startswith("application/"):
        subtype = ct[len("application/") :]
        return subtype not in _BINARY_APPLICATION_SUBTYPES
    return False


# First path segments on github.com that are site pages, not repo owners.
_GITHUB_NON_OWNER_SEGMENTS = frozenset(
    {
        "about",
        "apps",
        "codespaces",
        "collections",
        "contact",
        "customer-stories",
        "dashboard",
        "discussions",
        "enterprise",
        "explore",
        "features",
        "issues",
        "join",
        "login",
        "marketplace",
        "new",
        "notifications",
        "organizations",
        "orgs",
        "pricing",
        "pulls",
        "search",
        "security",
        "settings",
        "signup",
        "site",
        "sponsors",
        "team",
        "topics",
        "trending",
    }
)
_GITHUB_NAME_RE = re.compile(r"\A[A-Za-z0-9_.\-]{1,100}\Z")


def _github_repo_readme_api_url(url: str) -> str | None:
    """README API URL for a ``github.com/{owner}/{repo}`` page, else None.

    A repo root page rendered as HTML is mostly UI chrome (nav, file table,
    stats); the ``/readme`` API returns the raw README markdown unauthenticated,
    which is what the model actually wants to read.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in ("github.com", "www.github.com"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 2:
        return None
    owner, repo = parts
    if owner.lower() in _GITHUB_NON_OWNER_SEGMENTS:
        return None
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    if not (_GITHUB_NAME_RE.match(owner) and _GITHUB_NAME_RE.match(repo)):
        return None
    return f"https://api.github.com/repos/{owner}/{repo}/readme"


# A single fetch can chain several steps (README API attempt, HTML fallback, up
# to five redirect hops, each reading a body). A per-operation socket timeout
# bounds one stalled step but not their sum, and nothing aborts on client
# disconnect, so one overall wall-clock deadline (plus a cooperative
# cancel_event) bounds the whole fetch instead.
def _fetch_budget_exceeded(deadline, cancel_event):
    """User-facing error string when the fetch must stop early, else None."""
    if cancel_event is not None and cancel_event.is_set():
        return "Failed to fetch URL: cancelled."
    if deadline is not None and time.monotonic() >= deadline:
        return "Failed to fetch URL: timed out."
    return None


def _fetch_hop_timeout(timeout, deadline):
    """Per-operation socket timeout: the lesser of the caller's per-op timeout
    and the time left on the deadline, so one slow hop cannot overrun the whole
    budget. Callers check ``_fetch_budget_exceeded`` first, so remaining time is
    positive here; the tiny floor only guards a race."""
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        remaining = 0.001
    return remaining if timeout is None else min(timeout, remaining)


def _resolve_with_budget(hostname, port, deadline, cancel_event):
    """``_validate_and_resolve_host`` bounded by the overall fetch budget.

    ``getaddrinfo`` is blocking with no deadline of its own, so a slow resolver
    (or a request cancelled before dispatch) could run past the budget. Resolve
    on a daemon thread and poll the budget so the fetch aborts on time; the
    abandoned lookup is discarded. With no deadline and no cancel_event this is a
    plain synchronous call, so opt-out callers keep the old behavior and cost.
    """
    budget_error = _fetch_budget_exceeded(deadline, cancel_event)
    if budget_error is not None:
        return False, budget_error, ""
    if deadline is None and cancel_event is None:
        return _validate_and_resolve_host(hostname, port)

    result: "queue.Queue" = queue.Queue(maxsize = 1)

    def _resolve():
        try:
            result.put(_validate_and_resolve_host(hostname, port))
        except Exception as exc:  # defensive: never let the worker die silently
            result.put((False, f"Failed to resolve host: {exc}", ""))

    threading.Thread(target = _resolve, name = "web-fetch-dns", daemon = True).start()
    while True:
        budget_error = _fetch_budget_exceeded(deadline, cancel_event)
        if budget_error is not None:
            return False, budget_error, ""
        try:
            return result.get(timeout = 0.05)
        except queue.Empty:
            continue


def _read_capped_body(resp, max_bytes, timeout, deadline, cancel_event):
    """Read up to ``max_bytes``, enforcing the overall budget between chunks.

    A single ``resp.read(max_bytes)`` can block for the whole transfer if the
    server dribbles bytes just inside each socket-inactivity timeout, so the body
    is read in chunks with the budget re-checked (and the socket timeout
    re-tightened toward the deadline) each round. The joined bytes are identical
    to one capped read. Returns ``(error_or_None, body_bytes)``.
    """
    # Best-effort handle on the underlying socket so its timeout tightens as the
    # deadline nears; absent on test doubles, where the between-chunk budget
    # check still bounds the read.
    sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
    chunks = []
    remaining = max_bytes
    while remaining > 0:
        budget_error = _fetch_budget_exceeded(deadline, cancel_event)
        if budget_error is not None:
            try:
                resp.close()
            except Exception:
                pass
            return budget_error, b""
        if sock is not None:
            try:
                sock.settimeout(_fetch_hop_timeout(timeout, deadline))
            except Exception:
                pass
        chunk = resp.read(min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    budget_error = _fetch_budget_exceeded(deadline, cancel_event)
    if budget_error is not None:
        try:
            resp.close()
        except Exception:
            pass
        return budget_error, b""
    return None, b"".join(chunks)


def _fetch_url_raw(
    url: str,
    timeout: int = 30,
    extra_headers: dict | None = None,
    deadline: float | None = None,
    cancel_event = None,
) -> tuple[str | None, str, str]:
    """Fetch a URL with SSRF protection; return ``(error, body_text, content_type)``.

    ``error`` is a user-facing message string when the fetch failed (the
    existing "Blocked:" / "Failed to fetch URL:" wording), else ``None``.
    Blocks private/loopback/link-local targets and caps the download size.

    ``deadline`` is an optional ``time.monotonic`` cutoff for the whole fetch
    (redirect hops and body read included) and ``cancel_event`` aborts it when
    the caller goes away; both default off so callers keep the old behavior.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Blocked: only http/https URLs are allowed (got {parsed.scheme!r}).", "", ""
    if not parsed.hostname:
        return "Blocked: URL is missing a hostname.", "", ""

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ok, reason, pinned_ip = _resolve_with_budget(
        parsed.hostname,
        port,
        deadline,
        cancel_event,
    )
    if not ok:
        return reason, "", ""

    try:
        from urllib.error import HTTPError as _HTTPError
        from urllib.parse import urljoin, urlunparse

        max_bytes = _MAX_FETCH_BYTES
        current_url = url
        current_host = parsed.hostname
        ua = random.choice(_USER_AGENTS)

        for _hop in range(5):
            budget_error = _fetch_budget_exceeded(deadline, cancel_event)
            if budget_error is not None:
                return budget_error, "", ""
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

            headers = {
                "User-Agent": ua,
                "Host": current_host,
            }
            if extra_headers:
                headers.update(extra_headers)
            req = urllib.request.Request(pinned_url, headers = headers)
            try:
                # Cap the socket timeout at the time left on the overall deadline
                # so a single slow hop cannot outlast the whole fetch budget.
                resp = opener.open(req, timeout = _fetch_hop_timeout(timeout, deadline))
            except _HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    return f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}", "", ""
                location = e.headers.get("Location")
                if not location:
                    return "Failed to fetch URL: redirect missing Location header.", "", ""
                current_url = urljoin(current_url, location)
                rp = urlparse(current_url)
                if rp.scheme not in ("http", "https") or not rp.hostname:
                    return "Blocked: redirect target is not a valid http/https URL.", "", ""
                rp_port = rp.port or (443 if rp.scheme == "https" else 80)
                ok2, reason2, pinned_ip = _resolve_with_budget(
                    rp.hostname,
                    rp_port,
                    deadline,
                    cancel_event,
                )
                if not ok2:
                    return reason2, "", ""
                current_host = rp.hostname
                continue

            # get_content_type() defaults to "text/plain" when the header is
            # absent (RFC 2045); report "" instead so callers can tell a missing
            # header apart from a server that really declared text/plain.
            if resp.headers.get("Content-Type") is None:
                content_type = ""
            else:
                content_type = (resp.headers.get_content_type() or "").lower()

            # Success: read the capped body enforcing the budget between chunks
            # (see _read_capped_body), so a slow-drip server can't stretch a
            # single resp.read past the deadline.
            declared_pdf = content_type == "application/pdf"
            read_limit = _MAX_PDF_FETCH_BYTES + 1 if declared_pdf else max_bytes
            body_error, raw_bytes = _read_capped_body(
                resp,
                read_limit,
                timeout,
                deadline,
                cancel_event,
            )
            if body_error is not None:
                return body_error, "", ""

            # A missing or wrong PDF MIME type is common: once the initial text-sized
            # read identifies PDF magic, finish the bounded download to reach the EOF xref.
            if not declared_pdf and len(raw_bytes) == max_bytes and _has_pdf_magic(raw_bytes):
                tail_error, tail = _read_capped_body(
                    resp,
                    _MAX_PDF_FETCH_BYTES - max_bytes + 1,
                    timeout,
                    deadline,
                    cancel_event,
                )
                if tail_error is not None:
                    return tail_error, "", ""
                raw_bytes += tail
            break
        else:
            return "Failed to fetch URL: too many redirects.", "", ""

        is_pdf = declared_pdf or _has_pdf_magic(raw_bytes)
        if is_pdf:
            if len(raw_bytes) > _MAX_PDF_FETCH_BYTES:
                return (
                    "(PDF content exceeds the download limit; not readable as text)",
                    "",
                    content_type,
                )
            budget_error = _fetch_budget_exceeded(deadline, cancel_event)
            if budget_error is not None:
                return budget_error, "", content_type
            try:
                pdf_text = _extract_pdf_text(raw_bytes)
            except Exception as exc:
                logger.debug("web PDF text extraction failed (%s)", type(exc).__name__)
                return "(PDF content could not be read as text)", "", content_type
            budget_error = _fetch_budget_exceeded(deadline, cancel_event)
            if budget_error is not None:
                return budget_error, "", content_type
            if not pdf_text:
                pdf_text = "(PDF contains no extractable text)"
            # Report the true type even for a mislabeled body so the caller's "html"
            # check routes the extracted text to the plain-text path, not html_to_markdown.
            return None, pdf_text, "application/pdf"

        # Reject known-binary MIME types before decoding. Binary is returned as the
        # error string so the caller surfaces the placeholder, not replacement chars.
        if not _is_text_candidate_content_type(content_type):
            # Only echo a clean MIME token back to the model.
            m = re.match(r"[\w.+-]+/[\w.+-]+", content_type or "")
            safe_type = m.group(0) if m else "unknown type"
            return (
                f"(non-text content: {safe_type}, {len(raw_bytes)} bytes; not readable as text)",
                "",
                content_type,
            )

        # Catch text-labeled binary via its magic signature.
        if _has_binary_magic(raw_bytes):
            return (
                f"(binary content, {len(raw_bytes)} bytes; not readable as text)",
                "",
                content_type,
            )

        declared = resp.headers.get_content_charset()
        declared_codec = codecs.lookup(declared).name if declared else None
        bom_codec = next(
            (codec for bom, codec in _UNICODE_BOM_CODECS if raw_bytes.startswith(bom)),
            None,
        )
        raw_html = raw_bytes.decode(declared or bom_codec or "utf-8", errors = "replace")

        # Catch mislabeled or unlabeled binary, including valid UTF-8 controls.
        if _looks_binary(raw_html):
            # Rescue undeclared cp1252 only when the bytes have text structure.
            alt = (
                raw_bytes.decode("cp1252", "replace")
                if declared_codec in (None, "iso8859-1")
                and _has_single_byte_text_evidence(raw_bytes)
                else None
            )
            if alt is not None and not _looks_binary(alt):
                raw_html = alt
            else:
                return (
                    f"(binary content, {len(raw_bytes)} bytes; not readable as text)",
                    "",
                    content_type,
                )

        return None, raw_html, content_type
    except _HTTPError as e:
        return f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}", "", ""
    except Exception as e:
        return f"Failed to fetch URL: {e}", "", ""


# Tags that, at the very START of a body, mark it as HTML. Excludes ambiguous
# tags (<div>/<p>/<span>/<a>/<img>/<h1>..<h6>/<table>) that legitimately open
# centered-logo or badge-layout Markdown READMEs and must stay Markdown.
_HTML_LEADING_TAGS = (
    "html",
    "head",
    "body",
    "title",
    "meta",
    "link",
    "script",
    "style",
    "article",
    "section",
    "main",
    "header",
    "footer",
    "nav",
    "aside",
    "figure",
    "form",
    "ul",
    "ol",
    "dl",
    "pre",
    "blockquote",
)
_HTML_LEADING_RE = re.compile(r"<(?:!doctype\s+html|/?(?:" + "|".join(_HTML_LEADING_TAGS) + r")\b)")


def _looks_like_html(body: str) -> bool:
    """True only when the document ITSELF opens with HTML.

    Matches an HTML doctype or a leading document/structure tag after optional
    whitespace, not a mere substring, so a Markdown README with a fenced HTML
    example or tags further down stays Markdown. Also detects bare fragments
    (``<body>``/``<article>``/...) with no doctype, so a page with a
    missing/wrong Content-Type is still converted.
    """
    probe = body.lstrip()[:256].lower()
    return bool(_HTML_LEADING_RE.match(probe))


# Stricter than _HTML_LEADING_RE: only a real document opener (doctype or leading
# <html>/<head>/<body>), never a block tag a Markdown file can open with. Used on
# the raw GitHub README body so a Markdown README starting with an HTML block is
# not run through html_to_markdown, which would collapse its headings, lists and
# fenced code onto one line.
_HTML_DOCUMENT_RE = re.compile(r"<(?:!doctype\s+html\b|/?(?:html|head|body)\b)")


def _looks_like_html_document(body: str) -> bool:
    """True only when the body opens as a full HTML document (e.g. a .html README)."""
    probe = body.lstrip()[:256].lower()
    return bool(_HTML_DOCUMENT_RE.match(probe))


def _truncate_page_text(text: str, max_chars: int) -> str:
    if not text:
        return "(page returned no readable text)"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


def _fetch_page_text(
    url: str,
    max_chars: int = _MAX_PAGE_CHARS,
    timeout: int = 30,
    cancel_event = None,
) -> str:
    """Fetch a URL and return readable text content.

    HTML responses are converted to Markdown with a main-content heuristic
    (``<article>``/``<main>`` scoping, hidden-element and boilerplate
    stripping); non-HTML text responses are returned as-is. GitHub repo root
    pages are rewritten to the README API so the model reads the README
    instead of the repo page's UI chrome. Blocks private/loopback/link-local
    targets (SSRF protection) and caps the download size.
    """
    # One wall-clock budget for the whole fetch. The README API attempt and its
    # HTML fallback both draw from it, so a slow/failed API call cannot hand the
    # fallback a fresh full timeout and double the worst case.
    deadline = None if timeout is None else time.monotonic() + timeout
    readme_api_url = _github_repo_readme_api_url(url)
    if readme_api_url:
        err, body, _ctype = _fetch_url_raw(
            readme_api_url,
            timeout = timeout,
            extra_headers = {
                "Accept": "application/vnd.github.raw+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            deadline = deadline,
            cancel_event = cancel_event,
        )
        # The README API is unauthenticated and rate-limited; on any failure fall
        # back to the HTML page fetch. A 200 body is authoritative even when it is
        # HTML (a .html README): convert it rather than falling back to the repo
        # page's UI chrome, keeping the raw body if extraction yields nothing.
        if err is None and body.strip():
            readme_body = body
            # The raw file is almost always Markdown. Only a real HTML document (a
            # .html README) is converted; a Markdown README that merely opens with
            # a block tag is kept as-is (see _HTML_DOCUMENT_RE).
            if _looks_like_html_document(body):
                from ._html_to_md import html_to_markdown
                converted = html_to_markdown(body, main_content = True)
                readme_body = converted if converted.strip() else body
            if readme_body.strip():
                return _truncate_page_text(
                    f"README of {url} (fetched via the GitHub README API):\n\n" + readme_body,
                    max_chars,
                )

    err, body, content_type = _fetch_url_raw(
        url,
        timeout = timeout,
        deadline = deadline,
        cancel_event = cancel_event,
    )
    if err is not None:
        return err

    # Trust a declared HTML type, and otherwise sniff the body: servers with a
    # missing or wrong Content-Type (e.g. text/plain on an HTML page) still get
    # converted, matching the pre-extraction behavior of always converting.
    is_html = "html" in content_type or _looks_like_html(body)
    if not is_html:
        # Plain text / markdown / JSON (e.g. raw.githubusercontent.com):
        # converting through the HTML renderer would collapse its whitespace.
        return _truncate_page_text(body.strip(), max_chars)

    # Convert HTML to Markdown with the builtin converter (no external deps).
    from ._html_to_md import html_to_markdown

    return _truncate_page_text(html_to_markdown(body, main_content = True), max_chars)


def _web_search(
    query: str,
    max_results: int = 5,
    timeout: int = _EXEC_TIMEOUT,
    url: str | None = None,
    cancel_event = None,
) -> str:
    """Search the web using DuckDuckGo and return formatted results.

    If ``url`` is provided, fetches that page directly instead of searching.
    """
    # Direct URL fetch mode.
    if url and url.strip():
        fetch_timeout = 60 if timeout is None else min(timeout, 60)
        return _fetch_page_text(
            url.strip(),
            timeout = fetch_timeout,
            cancel_event = cancel_event,
        )

    if not query or not query.strip():
        return "No query provided."
    # A disconnect sets cancel_event; DDGS.text() is blocking and cannot be
    # interrupted mid-flight, so gate on either side: skip an already-cancelled
    # request, and discard results that land after the client has gone.
    if cancel_event is not None and cancel_event.is_set():
        return "Search cancelled."
    try:
        from ddgs import DDGS

        results = DDGS(timeout = timeout).text(query, max_results = max_results)
        if cancel_event is not None and cancel_event.is_set():
            return "Search cancelled."
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


def _check_signal_escape_patterns(code: str):
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

    signal_tampering = []
    exception_catching = []
    shell_escapes = []
    warnings = []

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

    # Dangerous os/subprocess functions that can execute shell commands.
    _SHELL_EXEC_FUNCS = frozenset(
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

    def _extract_string_from_node(node):
        """Extract a plain string value from an AST node, if it is a constant."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

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

    def _check_args_for_blocked(args_nodes):
        """Check if any call arguments contain blocked commands."""
        found = set()
        for arg in args_nodes:
            s = _extract_string_from_node(arg)
            if s is not None:
                found |= _find_blocked_commands(s)
            strs = _extract_strings_from_list(arg)
            for s in strs:
                found |= _find_blocked_commands(s)
        return found

    class SignalEscapeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.imports_signal = False
            self.signal_aliases = {"signal"}
            self.os_aliases = {"os"}
            self.subprocess_aliases = {"subprocess"}
            # Bare name -> fully-qualified form for from-import tracking
            # (e.g. "system" -> "os.system").
            self.shell_exec_aliases: dict[str, str] = {}
            self.loop_depth = 0

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == "signal":
                    self.imports_signal = True
                    if alias.asname:
                        self.signal_aliases.add(alias.asname)
                elif alias.name == "os":
                    self.os_aliases.add(alias.asname or "os")
                elif alias.name == "subprocess":
                    self.subprocess_aliases.add(alias.asname or "subprocess")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
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
            elif node.module in ("os", "subprocess"):
                if node.module == "os":
                    self.os_aliases.add("os")
                else:
                    self.subprocess_aliases.add("subprocess")
                # Track from-imports of dangerous functions.
                for alias in node.names:
                    fq = f"{node.module}.{alias.name}"
                    if fq in _SHELL_EXEC_FUNCS:
                        self.shell_exec_aliases[alias.asname or alias.name] = fq
            self.generic_visit(node)

        def visit_While(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_For(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Call(self, node):
            func = node.func
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
            # Resolve the FQ function name for os.*/subprocess.*
            shell_func = None
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in self.os_aliases:
                        shell_func = f"os.{func.attr}"
                    elif func.value.id in self.subprocess_aliases:
                        shell_func = f"subprocess.{func.attr}"
            elif isinstance(func, ast.Name):
                # from-import aliases: from os import system; system(...)
                shell_func = self.shell_exec_aliases.get(func.id)

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
                blocked_in_args = _check_args_for_blocked(all_call_args)

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

    class NetworkAndIoVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            parts: list[str] = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                parts.insert(0, cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.insert(0, cur.id)
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

            # Direct sock.connect((host, port)) bypasses the FQ-prefix branch.
            if isinstance(node.func, ast.Attribute) and node.func.attr == "connect" and node.args:
                a0 = node.args[0]
                host_lit = None
                if isinstance(a0, ast.Tuple) and a0.elts:
                    e0 = a0.elts[0]
                    if isinstance(e0, ast.Constant) and isinstance(e0.value, str):
                        host_lit = e0.value
                elif isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    host_lit = a0.value
                if host_lit:
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

                # 2) Extract literal host (URL string or (host, port) tuple).
                host_arg = None
                url_arg = None
                if node.args:
                    a0 = node.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        url_arg = a0.value
                    elif isinstance(a0, ast.Tuple) and a0.elts:
                        e0 = a0.elts[0]
                        if isinstance(e0, ast.Constant) and isinstance(e0.value, str):
                            host_arg = e0.value
                if url_arg and host_arg is None:
                    m = re.match(r"^\w+://([^/?#]+)", url_arg)
                    if m:
                        host_arg = m.group(1)

                if host_arg:
                    if _is_metadata_host(host_arg):
                        network_calls.append(
                            {
                                "type": "metadata_host_blocked",
                                "line": getattr(node, "lineno", -1),
                                "description": "Blocked: cloud-metadata host",
                            }
                        )
                    elif not _is_trusted_host(host_arg):
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

    NetworkAndIoVisitor().visit(tree)

    is_safe = (
        len(signal_tampering) == 0
        and len(exception_catching) == 0
        and len(shell_escapes) == 0
        and len(network_calls) == 0
        and len(sensitive_file_reads) == 0
    )
    return is_safe, {
        "signal_tampering": signal_tampering,
        "exception_catching": exception_catching,
        "shell_escapes": shell_escapes,
        "network_calls": network_calls,
        "sensitive_file_reads": sensitive_file_reads,
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
        network_reasons = [item.get("description", "") for item in info.get("network_calls", [])]
        file_reasons = [
            item.get("description", "") for item in info.get("sensitive_file_reads", [])
        ]
        all_reasons = [
            r
            for r in reasons + shell_reasons + exception_reasons + network_reasons + file_reasons
            if r
        ]
        if all_reasons:
            return (
                f"Error: unsafe code detected ({'; '.join(all_reasons)}). "
                f"Please remove unsafe patterns from your code."
            )

    return None


def _capture_process_group(proc):
    """Return the setsid process-group id, or ``None`` when unavailable.

    Captured right after ``Popen`` so a later ``poll()`` / ``wait()`` that reaps
    the leader cannot make ``os.getpgid(proc.pid)`` fail first. POSIX-only:
    Windows has no process groups (and no ``os.getpgid``), so return ``None``
    there and let the single-pid ``proc.kill()`` fallback handle cleanup.
    """
    if os.name != "posix" or not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(proc.pid)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        return None


def _kill_process_tree(proc) -> None:
    """SIGKILL the setsid process group; fall back to single-pid kill."""
    if proc.poll() is not None:
        return
    pgid = None
    if hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None
    if pgid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _killpg_captured(pgid) -> None:
    """SIGKILL a process group captured before its leader was waited on.

    Once ``proc`` exits, ``os.getpgid(proc.pid)`` fails and ``_kill_process_tree``
    short-circuits, so a stdout-holding grandchild that outlived the parent could
    not otherwise be signaled. The pre-captured setsid group id still targets the
    whole tree. No-op with no ``os.killpg`` (Windows) or nothing captured.
    """
    if pgid is None or not hasattr(os, "killpg"):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _cancel_watcher(
    proc,
    cancel_event,
    poll_interval = 0.2,
    pgid = None,
):
    """Daemon thread that kills a process when cancel_event is set.

    ``pgid`` is the group id captured right after spawn; killing it directly
    reaps a stdout-holding grandchild even when the watcher's own ``poll()``
    already reaped the leader (which makes ``_kill_process_tree`` short-circuit).
    """
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            _killpg_captured(pgid)
            _kill_process_tree(proc)
            return
        cancel_event.wait(poll_interval) if cancel_event else None


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    # Mode-neutral notice: this result serves both the streaming UI and
    # non-streaming callers and must stay byte-identical with and without an
    # output_callback (a regression-tested invariant), so it can't claim the
    # user saw the full output.
    if len(text) > limit:
        return text[:limit] + (
            f"\n\n... (truncated to {limit} chars for the model; {len(text)} chars "
            "total. The full output is not retained here; any files the code wrote "
            "persist in the working directory.)"
        )
    return text


# ChatGPT code-interpreter path conventions models write out of habit; none
# exist in the Unsloth sandbox, so a failure on one earns the retry hint.
_MISSING_PATH_PREFIXES = (
    "/mnt/data",
    "/mnt/outputs",
    "/home/sandbox",
    "/workspace",
    "/tmp/outputs",
)

# Matches the quoted path in a Python OSError str and the bare path in a bash
# "No such file or directory" error; applied only to the error line.
_QUOTED_ABS_PATH_RE = re.compile(r"""['"](/[^'"\n]+)['"]""")
_BASH_ABS_PATH_RE = re.compile(r"(/[^\s:'\"]+):\s*No such file or directory")

# The sandbox CWD is a per-thread dir under ~/studio_sandbox; an absolute path
# under it is a genuine local miss, not a hallucinated out-of-sandbox write.
_SANDBOX_ROOT = os.path.join(os.path.expanduser("~"), "studio_sandbox")


def _missing_error_lines(output: str) -> list[str]:
    """The lines that actually name a missing file (a FileNotFoundError message
    or a bash "No such file or directory"). Traceback frame lines such as
    ``File "/workspace/proj/script.py"`` are excluded, so an unrelated absolute
    path mentioned elsewhere in the output is never treated as the failing one."""
    return [
        line
        for line in output.splitlines()
        if "No such file or directory" in line or "FileNotFoundError" in line
    ]


def _extract_missing_abs_path(output: str) -> str | None:
    """Pull the absolute path a FileNotFoundError / bash error named, if any."""
    for line in reversed(_missing_error_lines(output)):
        m = _QUOTED_ABS_PATH_RE.search(line)
        if m:
            return m.group(1)
        m = _BASH_ABS_PATH_RE.search(line)
        if m:
            return m.group(1)
    return None


def _is_outside_workdir(abs_path: str, workdir: str | None = None) -> bool:
    """True when ``abs_path`` is not the working directory or under it.

    ``workdir`` is the executor's actual working directory (defaults to the
    sandbox root). Project-backed sessions run under a root OUTSIDE
    ``~/studio_sandbox`` (see ``_get_workdir``), so a legitimate miss inside a
    project must be judged against the real workdir, not a static sandbox root,
    or it is wrongly classed as an external habit path.
    """
    try:
        root = os.path.realpath(workdir or _SANDBOX_ROOT)
        rp = os.path.realpath(abs_path)
    except (OSError, ValueError):
        return True
    return rp != root and not rp.startswith(root + os.sep)


def _missing_path_hint(output: str, workdir: str | None = None) -> str:
    """Model-visible healing when an execution fails on an absolute path missing
    in the sandbox (a code-interpreter habit path, or one invented from the CWD).
    Detected on the full pre-truncation output; the hint echoes the failing path
    so the model retries with the right relative name."""
    error_lines = _missing_error_lines(output)
    if not error_lines:
        return ""
    abs_path = _extract_missing_abs_path(output)
    # A convention prefix is an out-of-sandbox signal only when the exact failing
    # path could not be isolated; scoped to the failing-path error line(s) so a
    # prefix mentioned elsewhere doesn't trigger a misleading hint.
    convention = any(prefix in line for line in error_lines for prefix in _MISSING_PATH_PREFIXES)
    if abs_path is not None:
        # Judge the isolated path against the real workdir even when it matches a
        # convention prefix, so a genuine miss inside a project rooted under such
        # a prefix (e.g. /workspace/proj) is not steered out of its subdirectory.
        if not _is_outside_workdir(abs_path, workdir):
            return ""
    elif not convention:
        # Nothing marks this as an out-of-sandbox miss; stay silent.
        return ""
    if abs_path:
        example = f"'{os.path.basename(abs_path)}', not '{abs_path}'"
    else:
        example = "'output.html', not '/mnt/data/output.html'"
    return (
        "\nHint: that absolute path does not exist in this sandbox. The current "
        "working directory is writable and persists for this conversation; retry "
        f"with a relative path (for example {example})."
    )


def _drain_process_output(
    proc,
    timeout,
    output_callback,
    cancel_event = None,
    *,
    pgid = None,
) -> tuple[str, bool]:
    """``proc.communicate(timeout=...)`` equivalent that also streams each
    stdout line to ``output_callback`` as it is produced.

    Returns ``(output, timed_out)``. The joined output is identical to what
    ``communicate`` would return: the same TextIOWrapper decodes the stream,
    so encoding, error replacement, and newline translation all match. On
    timeout the process tree is killed (mirroring the non-streaming path).
    With ``timeout=None`` the drain waits for EOF like ``communicate`` would,
    stopping early only when ``cancel_event`` is set.
    """
    chunks: list[str] = []

    # Captured before waiting so a stdout-holding grandchild can still be killed
    # after the leader is reaped (getpgid then fails). Callers pass it in from
    # right after Popen; fall back to capturing here for direct callers.
    if pgid is None:
        pgid = _capture_process_group(proc)

    def _reader() -> None:
        try:
            for line in iter(proc.stdout.readline, ""):
                chunks.append(line)
                if output_callback is not None:
                    try:
                        output_callback(line)
                    except Exception:  # noqa: BLE001 - observer must never kill the tool
                        logger.debug("tool output_callback raised", exc_info = True)
        except (ValueError, OSError):
            pass  # pipe closed during kill

    reader = threading.Thread(target = _reader, daemon = True)
    reader.start()
    started_at = time.monotonic()
    timed_out = False
    try:
        proc.wait(timeout = timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        # Also kill the pre-captured group in case the leader was reaped in the
        # window before _kill_process_tree sampled its pgid, reaping a
        # stdout-holding grandchild (matches the non-streaming timeout path).
        _killpg_captured(pgid)
        try:
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            pass
    # A grandchild that inherited stdout can hold the pipe open past the main
    # process's exit.
    if not timed_out:
        if timeout is not None:
            # Wait out the remaining budget like communicate() would, polling
            # cancel_event in slices (the cancel watcher is gone once the leader
            # exits) so a chatty grandchild doesn't keep draining after a Stop.
            # The normal path still reaches EOF on its own with the same bytes.
            deadline = started_at + timeout
            while reader.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _killpg_captured(pgid)
                    break
                if cancel_event is not None and cancel_event.is_set():
                    _killpg_captured(pgid)
                    break
                reader.join(timeout = min(0.5, remaining))
        else:
            # Unlimited timeout: drain until the pipe closes (like
            # communicate(timeout=None)), stopping early only on cancellation.
            while reader.is_alive():
                if cancel_event is not None and cancel_event.is_set():
                    _killpg_captured(pgid)
                    break
                reader.join(timeout = 0.5)
    reader.join(timeout = 5)
    return "".join(chunks), timed_out


def _python_exec(
    code: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
    disable_sandbox: bool = False,
    output_callback = None,
) -> str:
    """Execute Python code in a subprocess sandbox.

    disable_sandbox (Bypass Permissions): skip the safety analysis and rlimit
    pre-exec, and use the host env minus secrets.
    output_callback: optional callable(str) streamed each stdout line as it is
    produced; the returned result is unchanged.
    """
    if not code or not code.strip():
        return "No code provided."

    # Validate imports and code safety (skipped when the sandbox is disabled)
    if not disable_sandbox:
        error = _check_code_safety(code)
        if error:
            return error
        # Portable, best-effort string screen for out-of-workdir filesystem access;
        # shares the disable_sandbox bypass and the FS confinement env switch.
        if static_screen_enabled():
            _wd = _get_workdir(session_id)
            static_error = check_static_fs("python", code, _wd, _build_safe_env(_wd), host_pathmod())
            if static_error:
                return static_error
    elif not _harden_parent_against_proc_env_leak():
        # Close the /proc/<parent>/environ secret-recovery path first; if it
        # cannot be applied, fail closed rather than leak the parent environ.
        return (
            "Execution error: could not harden the Unsloth process against "
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
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.write(code)

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

        # -u forces unbuffered child stdout so a bare print() streams live
        # instead of sitting in the pipe's block buffer until exit. Applied
        # unconditionally to stay byte-identical with and without streaming;
        # unlike PYTHONUNBUFFERED=1 it never pollutes the child's os.environ.
        proc = subprocess.Popen([sys.executable, "-u", tmp_path], **popen_kwargs)

        # Capture the group before any watcher can reap the leader (see
        # _capture_process_group); None on Windows.
        pgid = _capture_process_group(proc)

        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher,
                args = (proc, cancel_event, 0.2, pgid),
                daemon = True,
            )
            watcher.start()

        # Always drain via _drain_process_output (output_callback may be None):
        # it kills the captured group on cancellation, reaping a grandchild that
        # outlived the leader, and returns bytes identical to communicate() so
        # the streaming vs non-streaming result stays byte-identical.
        output, timed_out = _drain_process_output(
            proc, timeout, output_callback, cancel_event, pgid = pgid
        )
        if timed_out:
            return _truncate(f"Execution timed out after {timeout} seconds.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        # Detect the missing-path pattern on the full output (truncation could
        # hide the trailing traceback); append the hint after truncation. External
        # paths are judged against the real workdir (project sessions live outside
        # the default sandbox root).
        hint = _missing_path_hint(result, workdir)
        result = _truncate(result) if result.strip() else "(no output)"
        result += hint

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
    output_callback = None,
) -> str:
    """Execute a bash command in a subprocess sandbox.

    disable_sandbox (Bypass Permissions): skip the command blocklist and rlimit
    pre-exec, and use the host env minus secrets.
    output_callback: optional callable(str) streamed each stdout line as it is
    produced; the returned result is unchanged.
    """
    if not command or not command.strip():
        return "No command provided."

    # Block dangerous commands (skipped when the sandbox is disabled)
    if not disable_sandbox:
        blocked = _find_blocked_commands(command)
        if blocked:
            return f"Blocked command(s) for safety: {', '.join(sorted(blocked))}"
        # Portable, best-effort string screen for out-of-workdir filesystem access;
        # shares the disable_sandbox bypass and the FS confinement env switch.
        if static_screen_enabled():
            _wd = _get_workdir(session_id)
            static_error = check_static_fs("shell", command, _wd, _build_safe_env(_wd), host_pathmod())
            if static_error:
                return static_error
    elif not _harden_parent_against_proc_env_leak():
        # Close the /proc/<parent>/environ secret-recovery path first; if it
        # cannot be applied, fail closed rather than leak the parent environ.
        return (
            "Execution error: could not harden the Unsloth process against "
            "/proc environment reads; refusing bypass execution."
        )

    try:
        workdir = _get_workdir(session_id)
        safe_env = _build_bypass_env(workdir) if disable_sandbox else _build_safe_env(workdir)
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            # Match _python_exec: decode utf-8 with "replace" so invalid output
            # bytes never raise UnicodeDecodeError (which the streaming reader
            # thread would swallow), keeping both paths byte-identical.
            encoding = "utf-8",
            errors = "replace",
            cwd = workdir,
            env = safe_env,
        )
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _bypass_preexec if disable_sandbox else _sandbox_preexec
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(_get_shell_cmd(command), **popen_kwargs)

        # Capture the group before any watcher can poll/reap the leader (see
        # _python_exec); None on Windows.
        pgid = _capture_process_group(proc)

        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher,
                args = (proc, cancel_event, 0.2, pgid),
                daemon = True,
            )
            watcher.start()

        # Always drain via _drain_process_output (see _python_exec): kills the
        # captured group on cancellation and returns bytes identical to
        # communicate(), keeping streaming vs non-streaming byte-identical.
        output, timed_out = _drain_process_output(
            proc, timeout, output_callback, cancel_event, pgid = pgid
        )
        if timed_out:
            return _truncate(f"Execution timed out after {timeout} seconds.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        # Same missing-path healing as _python_exec.
        hint = _missing_path_hint(result, workdir)
        result = _truncate(result) if result.strip() else "(no output)"
        return result + hint

    except Exception as e:
        return f"Execution error: {e}"
