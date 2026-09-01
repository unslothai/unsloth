# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool definitions and executors for LLM tool calling: web search
(DuckDuckGo), Python code execution, and terminal commands."""

import ast
import codecs
import copy
import fnmatch
import functools
import hashlib
import json
import http.client
import os
import signal

os.environ["UNSLOTH_IS_PRESENT"] = "1"

import asyncio
import queue
import random
import re
import shlex
import shutil
import ssl
import stat
from stat import S_ISREG
import subprocess
import sys
import tempfile
import contextlib
import threading
from contextvars import ContextVar

# What a truncated result costs besides its body, charged where the cut is decided rather
# than held back from the room in advance. See its definition for why that matters.
from .context_window import _RESULT_NOTICE_RESERVE

# The window of the model THIS request is served by, set by execute_tool for the call's
# duration. Left unset, the budget falls back to the process-global probe, which is right
# for the local loops and wrong for anything else: an external-provider request runs
# Unsloth's tool loop without touching a resident GGUF, so inheriting that GGUF's window
# let a small resident model truncate pages for a large cloud model, and a large resident
# model hand the full 16,000 characters to a small OpenAI-compatible endpoint.
_UNSET_CONTEXT_TOKENS = object()
_REQUEST_CONTEXT_TOKENS: ContextVar = ContextVar(
    "unsloth_request_context_tokens",
    default = _UNSET_CONTEXT_TOKENS,
)

# What the CONVERSATION has left, as opposed to how big the window is. The window alone
# cannot size a result: it does not fall as the thread fills, so the last result before an
# overflow is allowed exactly as much room as the first. None means the caller could not
# say, and every cap then behaves exactly as it did before this existed.
_REQUEST_RESULT_BUDGET: ContextVar = ContextVar(
    "unsloth_request_result_budget_tokens",
    default = None,
)

import uuid
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
from storage import mcp_servers_db

from loggers import get_logger

logger = get_logger(__name__)

_EXEC_TIMEOUT = 300  # 5 minutes
_RAG_SEARCH_SLOT = threading.BoundedSemaphore(1)
# Candidate multiplier when a website policy will filter the results after the search.
_POLICY_OVERFETCH = 4
_DISABLE_DNS_PINNING_ENV = "UNSLOTH_STUDIO_DISABLE_DNS_PINNING"

# Splits the UI source-map from the result; loops strip it (like __IMAGES__).
RAG_SOURCES_SENTINEL = "\n__RAG_SOURCES__:"

# A search that ran but produced nothing usable. Not a tool error, so callers that judge a step
# by its evidence (deep research) have to test for these explicitly.
EMPTY_SEARCH_RESULTS = (
    "No results found.",
    "No results found within the website access limits.",
)
# ddgs signals an empty sweep by raising rather than returning [].
_DDGS_EMPTY_SWEEP = "No results found"

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
        "slogin",
        "scp",
        "sftp",
        "rsync",
        "eval",
        "source",
        # `.` is the POSIX synonym for `source`: `. ./script.sh` runs the file's
        # contents in the current shell, past a classifier that never sees them.
        # Matched at command position only, so `find . -type f` / `cd .` are fine.
        ".",
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
# `if`/`while`/`until` are followed by a CONDITION the shell executes, so a
# command right after them is at command position (if rm -rf x; then :; fi).
_SHELL_KEYWORDS_AS_SEP = frozenset({"then", "do", "else", "elif", "if", "while", "until", "!"})
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
        "setpriv",
        "sudo",
        "doas",
        "su",
        "xargs",
    }
)
# Wrapper options whose VALUE is a separate token (env -u NAME, nice -n 5).
# Unconsumed, the value is mistaken for the wrapped command: `env -u FOO rm -rf x`
# reads as command `FOO`. Shared by the auto gate and the blocklist walk.
_WRAPPER_VALUE_FLAGS_BY_CMD = {
    # env -i/--ignore-environment is VALUELESS; only -u/--unset takes a name.
    "env": frozenset({"-u", "--unset"}),
    "stdbuf": frozenset({"-i", "--input", "-o", "--output", "-e", "--error"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "ionice": frozenset({"-c", "--class", "-n", "--classdata", "-p", "--pid"}),
    "xargs": frozenset(
        {"-I", "-L", "-P", "-d", "--delimiter", "-a", "--arg-file", "-n", "-s", "-E"}
    ),
    "chroot": frozenset({"--userspec", "--groups"}),
    # setpriv <options> <program>: only the value-taking options consume a token.
    "setpriv": frozenset(
        {
            "--reuid",
            "--regid",
            "--groups",
            "--inh-caps",
            "--ambient-caps",
            "--bounding-set",
            "--securebits",
            "--pdeathsig",
            "--selinux-label",
            "--apparmor-profile",
            "--landlock-access",
            "--landlock-rule",
        }
    ),
    # exec -a NAME runs cmd under NAME, so NAME is a value, not the command.
    "exec": frozenset({"-a"}),
    "setsid": frozenset(),
    "nohup": frozenset(),
}
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


# A search-path entry that can shadow a real binary or module: absolute, home or
# a parent escape. A relative entry (`PYTHONPATH=src`) points inside the session
# workdir, the agent's own directory, and is the common spelling in ordinary work.
_PATH_ENTRY_ESCAPES_RE = re.compile(r"(?:^|:)\s*(?:/|~|\$|[A-Za-z]:[\\/]|\.\.)")


def _env_assignment_is_unsafe(name: str, value: str = "") -> bool:
    """True if a NAME=value prefix affects command lookup/loading."""
    if name in _AUTO_UNSAFE_ENV_ASSIGN or name.startswith(("LD_", "DYLD_")):
        return True
    if name == "PATH":
        # Every value counts: PATH picks the BINARY, and a relative entry is the
        # sharpest form of that (`PATH=. ls` runs ./ls).
        return True
    # The other search paths (PYTHONPATH, NODE_PATH, ...) only shadow a real
    # module when the entry escapes the workdir.
    return name.endswith("PATH") and bool(_PATH_ENTRY_ESCAPES_RE.search(value))


# Container CLIs start or reach into a container (docker run -v /:/host), but
# their read subcommands are ordinary inspection and must not interrupt. An
# unrecognised subcommand still asks, so the list can only be too small.
_CONTAINER_CLIS = frozenset({"docker", "podman", "nerdctl", "ctr", "crictl", "lxc", "kubectl"})
_CONTAINER_READ_SUBCOMMANDS = frozenset(
    {
        "ps",
        "images",
        "logs",
        "inspect",
        "version",
        "info",
        "stats",
        "top",
        "port",
        "diff",
        "history",
        "search",
        "events",
        "ls",
        "list",
        "get",
        "describe",
        "df",
        "help",
        "explain",
        "api-resources",
        "api-versions",
    }
)
# Windows `if exist FILE cmd` / `if defined VAR cmd` put an operand between the
# keyword and the command, so the command word is two tokens along.
# awk runs its program text, which can shell out through the system() builtin
# or by piping to a shell ("cmd" | "sh"). Screening the program keeps ordinary
# field work (awk '{print $1}') running while the escape hatches ask.
_AWK_COMMANDS = frozenset({"awk", "gawk", "mawk", "nawk", "busybox-awk"})
_AWK_SHELL_ESCAPE_RE = re.compile(
    r"\bsystem\s*\(|\|\s*&?\s*[\"']\s*(?:/\S*/)?(?:sh|bash|zsh|ksh|dash|cmd)\b|"
    r"\bENVIRON\s*\[|\bprintf\s*\|"
)
# sed shells out like awk: GNU's `e` runs the rest of its line through popen and
# the `s///e` flag runs the pattern space, hiding a command inside a text-editing
# argument. Screened so ordinary editing (sed 's/a/b/g') stays unprompted.
_SED_COMMANDS = frozenset({"sed", "gsed", "ssed"})
# `s///` flags that may precede `e`. `w` is absent: it takes the rest of the
# line as a filename, so the e in `s/a/b/w report.txt` is part of that name.
_SED_SUBST_FLAGS = frozenset("0123456789gpiImMe")
# sed short options that consume text, so no later letter in the cluster is a
# flag: -e/-f take a script and -l a length (attached or next token), while -i's
# backup suffix is ATTACHED ONLY (`-ifoo` otherwise reads as an attached `-f oo`).
_SED_VALUE_FLAGS = "efl"
_SED_ATTACHED_VALUE_FLAGS = "i"
# A backslash in a sed text argument escapes the next character, newline
# included, so it is stripped before the payload is read as a shell command.
_SED_TEXT_ESCAPE_RE = re.compile(r"\\([\s\S])")
# A plain parameter reference in a sed program (`sed "$p" f`). Bare `$NAME` /
# `${NAME}` only: anything with an operator is a transformation this scan does
# not model, so the program is judged UNREAD (see _sed_program_unresolved).
_PROGRAM_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")
# An unbraced expansion bash performs: a name (`$p`), a positional (`$1`) or a
# special parameter ($@ $* $# $? $- $$ $!). Any other `$` is literal (verified:
# `printf '%s' "$ d"` prints `$ d`), which keeps sed's `$` address out of scope.
_UNBRACED_PARAM_RE = re.compile(r"\$(?:[A-Za-z_]\w*|[0-9]+|[@*#?$!-])")
# Arithmetic evaluates to an INTEGER, so it spells no sed command. A digit in its
# place keeps `sed -n "1,$((n + 1))p" f` silent while still exposing the `e` in
# `sed "$((c+1))e rm -f victim"`, which runs rm.
_ARITHMETIC_VALUE = "0"
# The FLOOR every invocation gets for its argument walk, which keeps a line
# padded with `-exec sed` words linear. A flat cap is padding an attacker
# controls: `sed -n ...x128 '1e rm -f victim'` pushed the script past 128.
_MAX_SED_ARG_SCAN = 128
# Argument tokens the sed screen may walk across ONE command line, split over the
# sed words on it, so a lone sed reads its whole list and the work stays linear.
_SED_SCAN_BUDGET = 200_000
# Wrappers may sit between `find -exec` and the command it runs; bounded so a
# line padded with `-exec env -exec env ...` cannot make the scan quadratic.
_MAX_EXEC_PREFIX_SCAN = 32
# First window tried when balancing a `$(...)`, quadrupled until the span closes
# (_substitution_span), so a line of many short substitutions stays linear.
_SUBSTITUTION_SPAN_STEP = 64
# Quote state (_shell_quote_states) of a backslash and the character behind it.
# Distinct from the surrounding quoting because bash expands neither: the `$(` in
# `sed "s/\$(CC)/gcc/" Makefile` opens no command substitution.
_ESCAPED_CHAR_STATE = "\\"
_WIN_CONDITIONAL_KEYWORDS = frozenset({"exist", "defined", "errorlevel", "not"})
_FIND_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
# A find action is COMPLETE at its terminator: words after it are find's next
# predicate, not CMD's. Reading past it took a following `-exec grep -e safe {} +`
# for sed's script. `\;` is listed too, for the non-posix lexer.
_FIND_EXEC_TERMINATORS = frozenset({"+", ";", "\\;"})
# The `;` spellings END the action wherever they stand: a quoted `';'` and an
# escaped `\;` reach find as the same word. `+` is absent because find reads it
# as the batched terminator only directly after a `{}` (see _exec_scan_layout).
_FIND_EXEC_SEMICOLONS = frozenset({";", "\\;"})
# ...but ONLY inside such an action. shlex strips quoting, so a sed FILE operand
# spelled `';'` or `'+'` arrives as the same token as a real separator, and
# ending the scan there dropped the `-e` script behind it: verified that
# `sed -n ';' -e '1e rm -f victim' input` really runs rm. Outside an action only
# an UNQUOTED `;` ends the invocation.

# The characters a separator token can be built from, masked while the command
# is lexed a second time so a quoted one is told apart from a real one.
_SEPARATOR_CHARS = frozenset("".join(_SHELL_SEPARATORS))
# Placeholder for a quoted separator character during that second lex. Any
# non-whitespace, non-quote, non-punctuation_chars character serves, so the
# masked text splits into the same words and the token lists line up.
_QUOTED_SEPARATOR_MARK = "\x00"
# The characters bash expands a word against the filesystem for, and the
# placeholder standing in for a QUOTED one during the same second lex.
_GLOB_CHARS = frozenset("*?[")
_QUOTED_GLOB_MARK = "\x01"
# The characters a redirection is built from, and the placeholder standing in
# for a QUOTED one. A redirection is something the shell PERFORMS, so a quoted
# spelling is an ordinary word the command receives instead.
_REDIRECT_CHARS = frozenset("<>")
_QUOTED_REDIRECT_MARK = "\x02"
# The characters that open an expansion, and the placeholder for one the quoting
# made literal. Double quoting is NOT literal here (`sed "$p" f` expands), so
# only single-quoted and escaped states count (see _unquoted_expansion_indexes).
_EXPANSION_CHARS = frozenset("$`")
_QUOTED_EXPANSION_MARK = "\x04"
# The characters punctuation_chars glues into one token. A run like `|&` matches
# no _SHELL_SEPARATORS entry, so the sed screen read past the end of the command
# (`sed '1e rm -f victim' input |& grep -e safe` runs rm). `{`/`}` are absent so
# find's `{}` stays an ordinary word.
_OPERATOR_TOKEN_CHARS = frozenset(";&|()`")
# One shell redirection, as the lexer hands it over. The target may be glued on
# (`2>/dev/null`) or be the next token (`> out.txt`); `&` splits off under
# punctuation_chars, so `2>&1` arrives as three.
_REDIRECTION_RE = re.compile(r"^(?:\d+|&)?(?:<<<|<<-|<<|<>|>>|>\||<&|>&|<|>)")


def _looks_like_separator(token: str) -> bool:
    """Whether a lexed token is a shell operator rather than a word a command
    receives. A known separator, or a RUN of punctuation_chars characters, which
    is how bash builds `|&`, `;;` and `;&`."""
    if token in _SHELL_SEPARATORS:
        return True
    return bool(token) and not (set(token) - _OPERATOR_TOKEN_CHARS)


def _redirection_span(
    tokens: "list[str]",
    index: int,
    quoted: "frozenset[int]" = frozenset(),
    quoted_redirects: "frozenset[int]" = frozenset(),
) -> "tuple[int, ...]":
    """The token indexes one shell redirection at ``index`` occupies, or ``()``.

    The shell REMOVES a redirection before the command sees its arguments, so
    leaving the words in place made it the command's first operand: verified that
    `sed </dev/null '1e rm -f victim' input` and `> out.txt rm -rf victim` both
    run for real. A detached target is claimed only when it is an ordinary word.
    """
    if tokens[index] == "&" and index + 1 < len(tokens) and tokens[index + 1][:1] in "<>":
        # `&>out.txt` splits in two, and reading the `&` as a background
        # operator ended the command early. Only a redirection may follow, so
        # `echo hi & rm -rf victim` keeps its separator.
        tail = _redirection_span(tokens, index + 1, quoted, quoted_redirects)
        return (index, *tail) if tail else ()
    if index in quoted_redirects:
        # The quoting makes it a WORD the command receives: `sed -f '>prog' -e
        # '1e rm -f victim' input` takes `>prog` as the script FILE and really
        # runs the payload, while removing it as a redirection left -e unread.
        return ()
    match = _REDIRECTION_RE.match(tokens[index])
    if not match:
        return ()
    if tokens[index][match.end() :]:
        return (index,)  # target glued on: `2>/dev/null`, `>out.txt`
    span = [index]
    nxt = index + 1
    if nxt >= len(tokens):
        return tuple(span)
    if tokens[nxt] in {"&", "|"}:
        # `2>&1` and `>|out.txt` each arrive as three tokens, and the middle one
        # was read as the end of the command (verified: both run the payload).
        span.append(nxt)
        nxt += 1
    if nxt < len(tokens) and not (_looks_like_separator(tokens[nxt]) and nxt not in quoted):
        # The shell hands the target to open(), not to sed: `sed > --sandbox
        # '1e touch MARKER' input` and its `> ';'` twin both really run it. Only
        # a BARE operator is refused, since that line is malformed anyway.
        span.append(nxt)
    return tuple(span)


# `[` and `[[` are the test builtins, not patterns.
_TEST_BUILTINS = frozenset({"[", "[[", "]", "]]"})


def _is_unresolved_command_glob(base: str) -> bool:
    """Whether a command word is a glob bash expands to some other name
    (`/bin/r[m]` runs rm). A pattern with no literal character (a bare `*`) is
    not one, and the test builtins are not patterns."""
    if base in _TEST_BUILTINS or not any(ch in base for ch in "*?["):
        return False
    return any(ch.isalnum() for ch in base)


def _blocked_matching_glob(base: str) -> "set[str]":
    """Blocked command names a command-position glob can expand to."""
    if not _is_unresolved_command_glob(base):
        return set()
    return {name for name in _BLOCKED_COMMANDS if fnmatch.fnmatchcase(name, base)}


def _is_sed_command(base: str) -> bool:
    """Whether a command word runs sed: an exact name, or a command-position GLOB
    that could expand to one, since bash resolves `/usr/bin/s[e]d` to sed after
    this scan. Fail closed: a non-sed program holds no `e` and yields no
    payload."""
    if base in _SED_COMMANDS:
        return True
    return _is_unresolved_command_glob(base) and any(
        fnmatch.fnmatchcase(name, base) for name in _SED_COMMANDS
    )


def _sed_short_flag(token: str) -> "tuple[str, str] | None":
    """The first value-taking short option in a sed flag cluster, as
    ``(letter, text glued after it)``, or ``None``. The scan stops there because
    the rest of the token is that option's value: `-ifoo` is -i with backup
    suffix "foo", not an attached -f."""
    if not token.startswith("-") or token.startswith("--"):
        return None
    for index, ch in enumerate(token[1:]):
        if ch in _SED_VALUE_FLAGS or ch in _SED_ATTACHED_VALUE_FLAGS:
            return ch, token[index + 2 :]
    return None


def _sed_long_flag(name: str) -> str:
    """Which value-taking sed long option ``--name`` is: "e" for --expression,
    "f" for --file, "l" for --line-length, "" otherwise. getopt allows unambiguous
    abbreviations, so --e/--ex are --expression and --fi upwards is --file (--f is
    ambiguous with --follow-symlinks). --in-place's suffix is always attached."""
    if len(name) <= 2:
        return ""
    if "--expression".startswith(name):
        return "e"
    if len(name) > 3 and "--file".startswith(name):
        return "f"
    if "--line-length".startswith(name):
        return "l"
    return ""


def _sed_disables_exec(name: str) -> bool:
    """Whether the long option ``name`` puts sed in a mode that REFUSES to shell
    out. --sandbox disables e/r/w and --posix drops the GNU extensions `e` belongs
    to, so a script COMPILED under either aborts the run (exit 1) and its payload
    is inert. WHICH scripts that covers depends on where the flag sits: see
    _sed_invocation. Only unambiguous abbreviations count (`--s` is ambiguous and
    sed exits on it), and an `=` spelling is rejected by sed too.
    """
    if len(name) >= 4 and "--sandbox".startswith(name):
        return True
    return len(name) >= 3 and "--posix".startswith(name)


def _sed_scan_limit(sed_words: int) -> int:
    """How many argument tokens ONE sed invocation may walk looking for its
    script. A lone sed gets the whole budget, so padding cannot push the script
    out of view; a line packed with sed words falls back to the floor, which
    keeps the walk linear (`-exec sed ` repeated to 16KB: 39s against 3s)."""
    if sed_words <= 1:
        return _SED_SCAN_BUDGET
    return max(_MAX_SED_ARG_SCAN, _SED_SCAN_BUDGET // sed_words)


# An -f operand naming a STREAM rather than a file on disk, so the script arrives
# on stdin and "no program found" is ignorance rather than safety:
# `sed -f - input <<EOF ... 1e touch MARKER ... EOF` really runs the payload.
_SED_STREAM_PROGRAM_SOURCES = frozenset({"-", "/dev/stdin", "/dev/fd/0"})


def _sed_program_source_is_stream(value: str) -> bool:
    """Whether an `-f` operand reads the script from a stream this scan cannot
    follow. A named file (`sed -f prog.sed input`) stays out: it is documented
    residue rather than something to fail on. A process substitution counts, since
    `sed -f <(printf 'e rm -f victim') input` really runs rm; the lexer splits
    that operand at the `(`, which is why the bare `<`/`>` are here too."""
    if value in _SED_STREAM_PROGRAM_SOURCES or value.startswith("/dev/fd/"):
        return True
    return value[:1] in "<>"


def _end_program_source(programs: "list[str]", exec_disabled: bool) -> None:
    """Close the script source the pieces collected so far belong to, by appending
    the blank line the join needs.

    A source BOUNDARY ends any line continuation open across it, so a trailing
    `a\\` appends a blank line instead of swallowing the next source's first line.
    Verified on GNU sed 4.9: `sed -e '1a\\' -f /dev/null -e 'e touch MARKER' input`
    creates the file while the same line without the -f does not.
    """
    if programs and programs[-1] and not exec_disabled:
        programs.append("")


def _sed_invocation(
    tokens: "list[str]",
    start: int,
    limit: int = _MAX_SED_ARG_SCAN,
    stops: "frozenset[int]" = frozenset(),
    skips: "frozenset[int]" = frozenset(),
    globs: "frozenset[int]" = frozenset(),
    expandable: "frozenset[int]" = frozenset(),
) -> "tuple[list[str], bool, bool]":
    """The sed invocation whose command word sits at ``start``, as
    ``(program alternatives, unread, live_program)``.

    sed joins its -e values with newlines, so `sed -e '1a\\' -e 'e rm -rf x'`
    appends a line instead of executing it and the pieces are judged together.
    With no -e or -f the first positional is the script.

    --sandbox / --posix abort at COMPILE time, and sed compiles each -e as it is
    parsed while the positional waits for the whole option list, so the flag
    suppresses exactly the scripts written after it (verified on GNU sed 4.9:
    `sed -e '1e touch MARKER' --sandbox input` still runs). One written after the
    POSITIONAL suppresses only while getopt permutes, and POSIXLY_CORRECT turns
    that off from outside the command text, so it is not read as suppressing.
    `--` is honoured: a `--sandbox` behind it is an input FILENAME.

    ``unread`` says the program is at best a PREFIX of the real one, so an empty
    result proves nothing and callers fail closed on it.

    ``stops`` and ``skips`` are token INDEXES, not text: where the invocation
    ends (a separator the shell performs, or the `+` / `;` closing this sed's
    find action) and which words are a redirection the shell removes before sed
    runs. Both distinctions need the original quoting, which the text has lost.
    A skip yields to a pending -e/-f/-l value, since that word is sed's.
    """
    programs: "list[str]" = []
    first_positional = ""
    positional_disabled = False  # a mode flag preceded the positional script
    positional_globbed = False  # ...and bash rewrites it before sed is started
    positional_live = False  # ...and it holds an expansion the shell performs
    # A program flag AHEAD of the positional word makes that word an input FILE.
    # One BEHIND it does so only while getopt permutes, and POSIXLY_CORRECT turns
    # permutation off from outside the command text, so the positional is still
    # read as a script then (verified on GNU sed 4.9 that
    # `POSIXLY_CORRECT=1 sed '1e touch MARKER' input -f /dev/null` creates it).
    program_flag_before_positional = False
    # A mode flag has been seen, so every script COMPILED after it is inert.
    # Monotone by construction, so the live pieces are always a PREFIX rather
    # than a hole in the middle of one `-e '1a\' -e 'e rm -rf x'` program.
    exec_disabled = False
    end_of_options = False  # `--` seen: no later word is an option
    value_pending = ""  # "e", "f" or "l": the next token is that flag's value
    hit_separator = False  # the invocation ended before the window ran out
    stream_program = False  # an -f names a stream, so the script is not in argv
    glob_program = False  # the script word is one bash rewrites before sed sees it
    live_program = False  # ...and it holds an expansion the shell really performs
    window = tokens[start + 1 : start + 1 + limit]
    for offset, token in enumerate(window):
        if start + 1 + offset in stops:
            hit_separator = True
            break
        if start + 1 + offset in skips:
            # A redirection: the shell removed it before sed ran. Checked AHEAD
            # of the pending value, because one standing where that value goes is
            # removed too and the value is the word BEHIND it (`sed -n -e >out
            # '1e touch MARKER' input` really runs the payload).
            continue
        if value_pending:
            # The value is consumed either way; only a script sed still compiles
            # goes into the program.
            if value_pending == "e" and not exec_disabled:
                programs.append(token)
                glob_program = glob_program or start + 1 + offset in globs
                live_program = live_program or start + 1 + offset in expandable
            elif value_pending == "f" and _sed_program_source_is_stream(token):
                stream_program = True
            value_pending = ""
            continue
        if not end_of_options and token == "--":
            end_of_options = True
            continue
        if not end_of_options and token.startswith("--"):
            name, sep, value = token.partition("=")
            if not sep and _sed_disables_exec(name):
                exec_disabled = True
                continue
            letter = _sed_long_flag(name)
            if not letter:
                continue
            # -l only matters so its operand is not mistaken for the script.
            if letter in "ef" and not first_positional:
                program_flag_before_positional = True
            if letter == "f":
                _end_program_source(programs, exec_disabled)
                stream_program = stream_program or (
                    bool(sep) and _sed_program_source_is_stream(value)
                )
            if not sep:
                value_pending = letter
            elif letter == "e" and not exec_disabled:
                programs.append(value)
                glob_program = glob_program or start + 1 + offset in globs
                live_program = live_program or start + 1 + offset in expandable
            continue
        if not end_of_options and token.startswith("-"):
            # A cluster glues the value on (-ne'1p') or takes the next (-ne '1p').
            found = _sed_short_flag(token)
            if found is None:
                continue
            letter, attached = found
            if letter in _SED_ATTACHED_VALUE_FLAGS:
                # -i's suffix is the rest of the token; it never takes the next
                # one, so the script is still the positional ahead.
                continue
            if letter in "ef" and not first_positional:
                program_flag_before_positional = True
            if letter == "f":
                _end_program_source(programs, exec_disabled)
                stream_program = stream_program or (
                    bool(attached) and _sed_program_source_is_stream(attached)
                )
            if not attached:
                value_pending = letter
            elif letter == "e" and not exec_disabled:
                programs.append(attached)
                glob_program = glob_program or start + 1 + offset in globs
                live_program = live_program or start + 1 + offset in expandable
            continue
        if not first_positional:
            first_positional = token
            positional_disabled = exec_disabled
            positional_globbed = start + 1 + offset in globs
            positional_live = start + 1 + offset in expandable
    joined = ["\n".join(programs)] if programs else []
    if first_positional and not positional_disabled and not program_flag_before_positional:
        glob_program = glob_program or positional_globbed
        live_program = live_program or positional_live
        if not programs:
            joined = [first_positional]
        else:
            # A program option stands BEHIND the positional, so which of the two
            # sed compiles depends on permutation. They are ALTERNATIVES, not one
            # program: joining them let an unterminated command in one swallow
            # the other, and `POSIXLY_CORRECT=1 sed '1e touch MARKER' input -e
            # safe` read as safe although it really runs the payload.
            joined.append(first_positional)
    # Complete when a separator closed the invocation, or when the window
    # already covered every remaining argument.
    scan_overflowed = not hit_separator and len(tokens) > start + 1 + limit
    # A still-pending -f value means the invocation ended before its operand was
    # read at all -- a process substitution ends it at the `(` -- so the program
    # is unknown rather than absent.
    joined = [piece.replace(_ANSI_C_NEWLINE_MARK, "\n") for piece in joined]
    unread = scan_overflowed or stream_program or glob_program or value_pending == "f"
    return joined, unread, live_program


def _sed_text(text: str) -> str:
    """Unescape one sed text argument the way read_text does: every backslash
    drops away and the character behind it stays, so `e touch MARK\\ER` runs
    MARKER."""
    return _SED_TEXT_ESCAPE_RE.sub(r"\1", text).strip()


def _sed_exec_payloads(program: str) -> "list[str]":
    """Shell payloads a sed program executes, in order.

    `e COMMAND` runs COMMAND. A bare `e` and the `s///e` flag run the pattern
    space, which only exists at run time, so they yield an EMPTY payload:
    executes, but nothing to screen. An empty list means it only edits text.

    The walk skips every region where an `e` is data (regexes, replacements,
    a/i/c text, r/w filenames, b/t labels, comments), keeping `:e;N;$!be;...`,
    `sed 's/e/E/g'` and `sed 's/a/b/w report.txt'` out of the results.
    """
    payloads: "list[str]" = []
    n = len(program)

    def _end_of_line(pos: int) -> int:
        end = program.find("\n", pos)
        return n if end < 0 else end

    def _end_of_text(pos: int) -> int:
        # read_text, which collects `e`/`a`/`i`/`c` text: a backslash escapes
        # the next character, so a line ending in one carries the text onto the
        # NEXT line instead of stopping there.
        while pos < n and program[pos] != "\n":
            pos += 2 if program[pos] == "\\" else 1
        return min(pos, n)

    def _skip_bracket(pos: int) -> int:
        # A bracket expression, where the delimiter is data (`s/[/]/x/` really
        # substitutes a slash). A leading `]` is literal; [:class:] nests.
        pos += 1
        if pos < n and program[pos] == "^":
            pos += 1
        if pos < n and program[pos] == "]":
            pos += 1
        while pos < n and program[pos] != "]":
            if program[pos] == "[" and pos + 1 < n and program[pos + 1] in ":.=":
                end = program.find(program[pos + 1] + "]", pos + 2)
                pos = n if end < 0 else end + 2
                continue
            pos += 1
        return pos + 1

    def _skip_section(pos: int, delim: str, brackets: bool) -> int:
        # One delimited section of a regex / s/// / y///, through its closing
        # delimiter. Brackets apply to regex halves only; elsewhere `[` is data.
        while pos < n and program[pos] != delim:
            if program[pos] == "\\":
                pos += 2
            elif brackets and program[pos] == "[":
                pos = _skip_bracket(pos)
            else:
                pos += 1
        return pos + 1

    def _skip_address(pos: int) -> int:
        # A line number (GNU's first~step included), `$`, /regex/ or \%regex%,
        # each allowing I/M modifiers.
        if pos < n and program[pos] == "$":
            return pos + 1
        if pos < n and program[pos].isdigit():
            while pos < n and (program[pos].isdigit() or program[pos] == "~"):
                pos += 1
            return pos
        if pos < n and program[pos] == "/":
            pos = _skip_section(pos + 1, "/", brackets = True)
        elif pos < n and program[pos] == "\\" and pos + 1 < n:
            pos = _skip_section(pos + 2, program[pos + 1], brackets = True)
        else:
            return pos
        while pos < n and program[pos] in "IM":
            pos += 1
        return pos

    i = 0
    while i < n:
        if program[i] in " \t\n;{}":
            # Separators and block braces carry no command.
            i += 1
            continue
        if program[i] == "#":
            i = _end_of_line(i)
            continue
        i = _skip_address(i)
        if i < n and program[i] == ",":
            i += 1
            while i < n and program[i] in " \t":
                i += 1
            if i < n and program[i] in "+~":
                # `addr,+N` / `addr,~N` end the range relative to the first match.
                i += 1
                while i < n and program[i].isdigit():
                    i += 1
            else:
                i = _skip_address(i)
        while i < n and program[i] in " \t!":
            # `1!e cmd`: negation, the command word is still ahead.
            i += 1
        if i >= n:
            break
        cmd, i = program[i], i + 1
        if cmd == "e":
            # The payload ends at an UNESCAPED newline, so a `;` inside it is
            # shell text and `e\` + newline hands the next line to the same
            # shell (`1e\` / `rm -f victim` really runs rm).
            end = _end_of_text(i)
            payloads.append(_sed_text(program[i:end]))
            i = end
        elif cmd in "sy" and i < n:
            delim, i = program[i], i + 1
            i = _skip_section(i, delim, brackets = cmd == "s")
            i = _skip_section(i, delim, brackets = False)
            if cmd == "s":
                executes = False
                while i < n and program[i] in _SED_SUBST_FLAGS:
                    executes = executes or program[i] == "e"
                    i += 1
                if executes:
                    payloads.append("")
                if i < n and program[i] == "w":
                    i = _end_of_line(i)
        elif cmd in "aic":
            # Literal text; the `a\` + newline form continues on a trailing "\".
            i = _end_of_text(i)
        elif cmd in "rRwW":
            i = _end_of_line(i)  # the filename runs to the end of the line
        elif cmd in "btT:v":
            # A label (or `v` version) ends at the next separator.
            while i < n and program[i] not in ";\n}":
                i += 1
    return payloads


def _assignment_bindings(
    tokens: "list[str]", quoted: "frozenset[int]" = frozenset()
) -> "list[tuple[int, str, str | None]]":
    """Every `NAME=value` word as ``(token index, name, value)``, in the order
    the shell performs the assignments.

    An ordered LIST, not a map, because bash uses the binding performed most
    recently BEFORE the reference: first-wins let
    `p='1,3p'; p='1e rm -f victim'; sed "$p" input` read as `1,3p` while rm
    really runs. The index rides along so _bindings_before can drop the
    assignments that only happen after the sed.

    A non-literal value is recorded as ``None``, which CLEARS the name rather
    than leaving a stale earlier one standing, since resolving to that would
    invent a program rather than read one.

    Only a word that really changes SHELL state counts. An assignment-shaped
    ARGUMENT (`echo p='1,3p'`), one in a subshell and one used as a command's
    environment prefix all leave `$p` alone, and recording them overwrote a
    payload with a value bash never assigned; all three run rm for real. A
    conditional one after `&&` may or may not run, so it is UNRESOLVED instead.
    """
    bindings: "list[tuple[int, str, str | None]]" = []
    pending: "list[tuple[int, str, str | None]]" = []  # the run at this position
    at_command = True  # an assignment here is a prefix, not an argument
    depth = 0  # inside ( ... ), where an assignment does not escape
    conditional = False  # after && / || : the assignment may never run
    function_body = 0  # inside f() { ... }, which bash has not run yet
    saw_parens = False  # the `()` of a function definition just went past
    for index, token in enumerate(tokens):
        if token == "{" and saw_parens:
            function_body += 1
            saw_parens = False
            continue
        if token == "}" and function_body:
            function_body -= 1
            at_command = True
            continue
        if _looks_like_separator(token) and index not in quoted:
            # Nothing followed the run, so it changed the shell's own state.
            bindings.extend(pending)
            pending = []
            saw_parens = set(token) <= {"(", ")"} and ")" in token
            depth = max(0, depth + token.count("(") - token.count(")"))
            conditional = "&&" in token or "||" in token
            at_command = True
            continue
        if function_body and _ASSIGNMENT_RE.match(token):
            # A body bash has not run yet, and may never run: `p='1e rm -f
            # victim'; f() { p='1,3p'; }; sed "$p" input` really runs rm.
            # Clearing the name is right whether or not f is ever called.
            name = token.partition("=")[0]
            pending.append((index, name, None))
            continue
        if at_command and _ASSIGNMENT_RE.match(token):
            if depth == 0:
                name, _, value = token.partition("=")
                literal = None if "$" in value or "`" in value else value
                pending.append((index, name, None if conditional else literal))
            continue
        if at_command:
            # A command word: the run in front of it is that command's
            # ENVIRONMENT, which bash hands the CHILD and not itself.
            pending = []
            at_command = False
    bindings.extend(pending)
    return bindings


def _bindings_before(
    bindings: "list[tuple[int, str, str | None]]", cursor: int, limit: int, env: "dict[str, str]"
) -> int:
    """Fold into ``env`` every binding at a token index below ``limit``, starting
    at ``cursor``, and return the cursor to pass in next time. Later bindings
    overwrite earlier ones, so ``env`` holds what the shell would have in scope
    at token ``limit``. Seds are visited left to right, so the cursor only moves
    forward and the whole line costs ONE walk of the binding list."""
    while cursor < len(bindings) and bindings[cursor][0] < limit:
        _index, name, value = bindings[cursor]
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
        cursor += 1
    return cursor


def _resolve_program_vars(program: str, env: "dict[str, str]") -> str:
    """``program`` with each `$NAME` / `${NAME}` replaced by its assigned value.

    A sed script held in a variable (`p='# note<newline>e CMD'; sed "$p" f`) is
    only a program once the reference is resolved, and only in a pass that KEEPS
    the quoted newline: the blanket newline pass turns the value into one long
    sed comment. An unassigned name is left as written, so nothing is invented.
    """
    return _PROGRAM_VAR_RE.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), program)


def _sed_program_variants(program: str, env: "dict[str, str]") -> "list[str]":
    """The sed program as written, plus the variable-resolved and
    arithmetic-collapsed forms. All are screened, because any spelling can be the
    one holding the `e`: the raw text in `sed "e $file"`, the resolved one in
    `sed "$p"`, the collapsed one in `sed "$((c+1))e rm -f victim"`."""
    if "$" not in program:
        return [program]
    variants = [program]
    resolved = _resolve_program_vars(program, env)
    if resolved != program:
        variants.append(resolved)
    for form in list(variants):
        collapsed = _collapse_shell_arithmetic(form)
        if collapsed not in variants:
            variants.append(collapsed)
    return variants


def _expansion_key(text: str) -> str:
    """One expansion, keyed so the raw-command spelling and the post-lex one
    compare equal. Only the escaping differs between them, so it is dropped."""
    return text.replace("\\", "")


def _sed_program_unresolved(variants: "list[str]", live: "set[str]") -> bool:
    """Whether NO spelling of the sed program is one this scan actually READ,
    because every one still holds an expansion bash would rewrite.

    The program is knowable only when each expansion reduces to text:
    `p='1,3p'; sed "$p" f` does, `sed "${p#x }" f` does not. The parameter
    transformations (`${p%y}`, `${p/a/b}`, `${p:-z}`, `${p^^}`, `${!p}`, ...) are
    not modelled one at a time; an unread program is UNKNOWN and the auto gate
    asks, which makes every unmodelled form safe by default rather than a way
    past (`p='x e rm -f victim'; sed "${p#x }" input` really runs rm).

    Only expansions the shell RUNS count, and only where they land in the
    PROGRAM, so one the program merely quotes (`sed 's/$(x)/y/' f`), an escaped
    one (`sed "s/\\$(CC)/gcc/" Makefile`) and one in a FILE operand
    (`sed -n '1,3p' $(ls)`) are all left running.
    """
    if not live:
        return False
    # shlex removes the escaping as it splits, so the SAME expansion is spelled
    # one way in the raw command and another in the token, and an exact
    # comparison read a generated program as one already read. Keying both sides
    # without backslashes can only make a spelling MATCH, so it fails closed.
    keys = {_expansion_key(found) for found in live}
    return not any(
        all(_expansion_key(found) not in keys for found in _shell_expansions(variant, quoted = False))
        for variant in variants
    )


def _quoted_separator_indexes(text: str, tokens: "list[str]", punctuation: str) -> "frozenset[int]":
    """Indexes of ``tokens`` that only LOOK like a shell separator because the
    quoting has been stripped off them.

    shlex hands back the identical token `;` for a real separator and for a
    quoted `';'` a command receives as data, so `sed -n ';' -e '1e rm -f victim'
    input` looked like a sed that had already ended and the `-e` script behind
    the `;` was never read (verified on GNU sed 4.9: it runs rm).

    Told apart by masking every separator character the shell QUOTES and lexing
    a second time. Only those characters change, and each inside the word it
    already belonged to, so the two token lists line up; the alignment is
    asserted by the length check, and anything unexpected reports nothing.
    """
    if not any(_looks_like_separator(token) for token in tokens):
        # Nothing to tell apart: skip the quote walk and the second lex.
        return frozenset()
    if _QUOTED_SEPARATOR_MARK in text:
        return frozenset()  # the mark is not ours to read back
    states = _shell_quote_states(text)
    masked = "".join(
        _QUOTED_SEPARATOR_MARK if char in _SEPARATOR_CHARS and states[index] else char
        for index, char in enumerate(text)
    )
    if _QUOTED_SEPARATOR_MARK not in masked:
        return frozenset()  # every separator character was bare
    try:
        lexer = shlex.shlex(masked, posix = True, punctuation_chars = punctuation)
        lexer.whitespace_split = True
        marked = list(lexer)
    except ValueError:
        return frozenset()
    if len(marked) != len(tokens):
        return frozenset()
    return frozenset(
        index
        for index, token in enumerate(marked)
        if _QUOTED_SEPARATOR_MARK in token and _looks_like_separator(tokens[index])
    )


def _masked_tokens(
    text: str, tokens: "list[str]", punctuation: str, chars: "frozenset[str]", mark: str
) -> "list[str] | None":
    """``tokens`` re-lexed with every one of ``chars`` the QUOTING made literal
    replaced by ``mark``, or ``None`` when the two lexes do not line up and
    nothing can be said. Each replacement stays inside the word it already
    belonged to, so the second lex yields the same words; the alignment is
    asserted by the length check rather than assumed."""
    if not any(char in chars for char in text) or mark in text:
        return None
    states = _shell_quote_states(text)
    masked = "".join(
        mark if char in chars and states[index] else char for index, char in enumerate(text)
    )
    try:
        lexer = shlex.shlex(masked, posix = True, punctuation_chars = punctuation)
        lexer.whitespace_split = True
        marked = list(lexer)
    except ValueError:
        return None
    return marked if len(marked) == len(tokens) else None


def _quoted_redirection_indexes(
    text: str, tokens: "list[str]", punctuation: str
) -> "frozenset[int]":
    """Indexes of ``tokens`` that only LOOK like a redirection because the
    quoting has been stripped off them.

    A QUOTED redirection is a word the shell hands the command: `sed -f '>prog'
    -e '1e rm -f victim' input` takes `>prog` as the script FILE and really runs
    the payload. Decided on the operator the token OPENS with, so `2>'/dev/null'`
    keeps its bare `2>` and stays a redirection while `'>prog'` does not.
    """
    marked = _masked_tokens(text, tokens, punctuation, _REDIRECT_CHARS, _QUOTED_REDIRECT_MARK)
    if marked is None:
        return frozenset()
    return frozenset(
        index
        for index, token in enumerate(tokens)
        if _REDIRECTION_RE.match(token) and not _REDIRECTION_RE.match(marked[index])
    )


def _unquoted_expansion_indexes(
    text: str, tokens: "list[str]", punctuation: str
) -> "frozenset[int]":
    """Indexes of ``tokens`` holding an expansion the shell really PERFORMS.

    Live expansions are collected over the whole command, so matching a sed
    program against them by text alone attributed another command's expansion to
    a program that merely spells the same thing, and the read-only
    `echo "$p"; sed 's/$p/x/' f` asked. This supplies the missing occurrence.

    Double quoting is deliberately not literal: `sed "$p" f` expands and must
    stay in. Only single, ANSI-C and backslash quoting make these characters
    data.
    """
    if not any(char in _EXPANSION_CHARS for char in text) or _QUOTED_EXPANSION_MARK in text:
        return frozenset()
    states = _shell_quote_states(text)
    masked = "".join(
        _QUOTED_EXPANSION_MARK
        if char in _EXPANSION_CHARS and states[index] and states[index] != '"'
        else char
        for index, char in enumerate(text)
    )
    try:
        lexer = shlex.shlex(masked, posix = True, punctuation_chars = punctuation)
        lexer.whitespace_split = True
        marked = list(lexer)
    except ValueError:
        return frozenset()
    if len(marked) != len(tokens):
        return frozenset()
    return frozenset(
        index
        for index, token in enumerate(marked)
        if any(char in _EXPANSION_CHARS for char in token)
    )


def _unquoted_glob_indexes(text: str, tokens: "list[str]", punctuation: str) -> "frozenset[int]":
    """Indexes of ``tokens`` holding a pathname-expansion metacharacter the shell
    will EXPAND, rather than one the quoting made literal.

    bash expands after this scan, so a word it rewrites is not the word the
    command receives: in a directory holding a file named `1e rm -f victim`,
    `sed *` hands sed that filename as its script and really runs rm. The quoted
    spellings a sed program uses must stay readable (`sed 's/a*/b/' f` expands
    nothing). Told apart by masking and re-lexing, as in
    _quoted_separator_indexes.
    """
    if not any(char in _GLOB_CHARS for char in text) or _QUOTED_GLOB_MARK in text:
        return frozenset()
    states = _shell_quote_states(text)
    masked = "".join(
        _QUOTED_GLOB_MARK if char in _GLOB_CHARS and states[index] else char
        for index, char in enumerate(text)
    )
    try:
        lexer = shlex.shlex(masked, posix = True, punctuation_chars = punctuation)
        lexer.whitespace_split = True
        marked = list(lexer)
    except ValueError:
        return frozenset()
    if len(marked) != len(tokens):
        return frozenset()
    return frozenset(
        index for index, token in enumerate(marked) if any(char in _GLOB_CHARS for char in token)
    )


def _xargs_replacement(tokens: "list[str]", start: int, end: int) -> str:
    """The placeholder the xargs word at ``start`` substitutes into the command
    words behind it, or "" when it replaces nothing. GNU xargs takes it attached
    (`-I{}`), as the next word (`-I {}`) or after an `=` (`--replace={}`); `-i`
    and a bare `--replace` default to `{}`."""
    index = start + 1
    while index < end:
        token = tokens[index]
        name, sep, value = token.partition("=")
        if name in {"--replace", "--replace-str"}:
            return value if sep and value else "{}"
        if token.startswith("-I"):
            if len(token) > 2:
                return token[2:]
            return tokens[index + 1] if index + 1 < end else "{}"
        if token.startswith("-i") and len(token.rstrip()) >= 2:
            return token[2:] or "{}"
        index += 1
    return ""


def _xargs_hides_sed_program(tokens: "list[str]", xargs: int, sed: int, program: str) -> bool:
    """Whether an xargs is the one deciding what program its sed runs.

    xargs appends the words it reads on stdin, and with -I substitutes them into
    the words already there, so the program need not be in the command TEXT at
    all. Both of these run rm for real, one holding no program and the other only
    the placeholder, so the sed fails closed:
        printf '1e rm -f victim\\0input\\0' | xargs -0 sed
        printf '1e rm -f victim\\n' | xargs -I{} sed '{}' input
    The ordinary idioms are untouched, since their program is right there and the
    placeholder stands where the FILE goes:
        find . -name '*.py' | xargs sed -i 's/a/b/g'
        find . -name '*.py' | xargs -I{} sed -i 's/a/b/' {}
    """
    if not program.strip():
        return True
    placeholder = _xargs_replacement(tokens, xargs, sed)
    return bool(placeholder) and placeholder in program


def _sed_program_is_a_placeholder(program: str) -> bool:
    """Whether the whole sed program is a token another tool REWRITES before sed
    starts. find replaces `{}` with the pathname it found, so with a file named
    `1e rm -f victim` the line
    `printf 'input' | find '1e rm -f victim' -exec xargs sed {} +` really runs rm
    while `{}` read as an already-known program. A `{}` among the FILE operands
    (`find . -exec sed -i 's/a/b/' {} +`) is not the program and is untouched."""
    return program.strip() == "{}"


def _forwards_exec_flags(base: str) -> bool:
    """Whether a command word runs a tool whose `-exec` / `-x` options hand the
    words behind them to a child command. Exact names, plus any command-position
    GLOB that could expand to one, so `/usr/bin/fin[d] . -exec rm {} \\;` is not
    read as an ordinary word."""
    if base in _EXEC_FLAG_FORWARDING_COMMANDS:
        return True
    return _is_unresolved_command_glob(base) and any(
        fnmatch.fnmatchcase(name, base) for name in _EXEC_FLAG_FORWARDING_COMMANDS
    )


def _exec_scan_layout(
    tokens: "list[str]",
    quoted: "frozenset[int]",
    quoted_redirects: "frozenset[int]" = frozenset(),
) -> "tuple[frozenset[int], frozenset[int], frozenset[int]]":
    """``(exec-flag indexes, invocation-stop indexes, redirection indexes)`` for
    one token list, in a single left-to-right pass.

    An exec-flag index is a `find`/`fd` option whose following words are a
    COMMAND that tool runs. Recognised only while a find/fd word the shell
    really RUNS is in scope: those letters belong to too many other tools, so
    `grep -x rm file` and the grep `-x` in `find . -exec grep -x rm {} \\;` must
    not have rm hard-blocked.

    A stop index ends a sed invocation: a separator the shell PERFORMS, or the
    `;` / `{} +` closing an open exec action. Outside an action those are
    ordinary operands, which keeps `sed -n ';' -e '1e rm -f victim' input`
    readable while a real terminator still stops the scan.

    A redirection index is a word the shell consumes and never hands to the
    command. Taken FIRST, so the `&` in `sed 2>&1 '1e rm -f victim' input` reads
    as part of that redirection rather than as the end of the invocation.
    """
    exec_flags: "set[int]" = set()
    stops: "set[int]" = set()
    redirects: "set[int]" = set()
    forwarding = False  # a find/fd command word is in scope
    in_action = False  # inside its `-exec CMD ...` action
    at_command = True  # the next ordinary word is one the shell RUNS
    wrapper = ""  # a command prefix (env/timeout/sudo) awaiting that word
    skip_operand = False  # ...and its option's value stands in between
    index = 0
    while index < len(tokens):
        token = tokens[index]
        span = _redirection_span(tokens, index, quoted, quoted_redirects)
        if span:
            redirects.update(span)
            index = span[-1] + 1
            continue
        here = index
        index += 1
        if _looks_like_separator(token) and here not in quoted:
            stops.add(here)
            forwarding = in_action = False
            at_command = True
            wrapper = ""
            skip_operand = False
            continue
        if in_action and (
            token in _FIND_EXEC_SEMICOLONS or (token == "+" and here and tokens[here - 1] == "{}")
        ):
            # find ends the batched form at `{} +` only: a `+` anywhere else is
            # an ordinary argument it hands the child, so
            # `find . -exec sed -n '+' -e '1e touch MARKER' {} +` really runs the
            # payload. The `;` forms need no such test: a quoted `';'` and an
            # escaped `\\;` reach find as the same word and both terminate.
            stops.add(here)
            in_action = False
            continue
        if forwarding and token == "--" and not in_action:
            # Nothing behind fd's `--` is an option: `fd -- -x rm` merely lists
            # `rm/-x` and was being refused.
            forwarding = False
            at_command = False
            continue
        flag = token.split("=", 1)[0]
        if forwarding and (
            flag in _FIND_EXEC_FLAGS or (not in_action and flag in _EXEC_FORWARD_FLAGS)
        ):
            exec_flags.add(here)
            in_action = True
            continue
        if forwarding and not in_action and token[:2] in {"-x", "-X"} and len(token) > 2:
            # fd takes the command attached to the short option too:
            # `fd '^victim$' . -xrm` deletes the match for real (fdfind 9.0.0).
            exec_flags.add(here)
            in_action = True
            continue
        if at_command and token in _SHELL_KEYWORDS_AS_SEP:
            continue  # `then find ...` / `do find ...`: still a command position
        if skip_operand:
            skip_operand = False  # a wrapper option's value (env -u NAME)
            continue
        if token.startswith("-") or _ASSIGNMENT_RE.match(token):
            # A wrapper option whose value is a SEPARATE token precedes that
            # value and not the wrapped command, so `env -u FOO find ...` keeps
            # looking for find rather than stopping at FOO.
            skip_operand = token in _WRAPPER_VALUE_FLAGS_BY_CMD.get(wrapper, frozenset())
            continue
        if wrapper and token.lstrip("-").isdigit():
            continue  # `timeout 5 find ...`: the wrapper's own operand
        base = os.path.basename(token.strip(";&|()`{}")).lower()
        if at_command and base in _COMMAND_PREFIXES:
            wrapper = base
            continue
        if at_command and _forwards_exec_flags(base):
            # Only a find/fd the shell really RUNS forwards its exec flags. Any
            # token spelled `fd`/`find` used to turn one on, so `echo fd -x rm`
            # and `grep fd -x rm file` came back with rm and were refused.
            forwarding = True
        at_command = False
        wrapper = ""
    return frozenset(exec_flags), frozenset(stops), frozenset(redirects)


def _win_switch(token: str) -> str:
    """Collapse a Git Bash `//x` switch to the `/x` cmd.exe actually receives."""
    return token[1:] if token.startswith("//") else token


# `start` launches its argument as a program, so that argument is a command
# position. These switches precede it; the value-taking ones eat a token.
_START_SWITCHES_WITH_VALUE = {"/d", "/node", "/affinity", "/machine"}

# A slash, one letter, optional `:value` (/s, /v:on, /t:0a). Matched in full so
# a program spelled as a path (/bin/bash) is never skipped as a switch.
_CMD_SWITCH_RE = re.compile(r"/[a-zA-Z](?::[\w.]+)?")

# START's documented switches. Matched by name rather than by a leading slash,
# because MSYS rewrites a POSIX path argument and hands cmd back something like
# /c/Windows/.../powershell.exe, which is a program and not a switch.
_START_SWITCHES = frozenset(
    {
        "/min",
        "/max",
        "/separate",
        "/shared",
        "/low",
        "/normal",
        "/high",
        "/realtime",
        "/abovenormal",
        "/belownormal",
        "/wait",
        "/b",
        "/i",
        "/d",
        "/node",
        "/affinity",
        "/machine",
    }
)


def _is_start_title(token: str) -> bool:
    """True when START would read ``token`` as its window title, not the program.

    The cmd lexer keeps the quote marks, so a title still arrives quoted. The
    posix lexer has stripped them, leaving two spellings a bare program name
    cannot have: the empty ``start ""`` idiom, and a title containing
    whitespace. A single-word posix title (``start "job" prog``) is
    indistinguishable from a program name and is deliberately not guessed.
    """
    return (
        token == ""
        or any(char.isspace() for char in token)
        or (len(token) >= 2 and token[0] == '"' and token[-1] == '"')
    )


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

    # Decode ANSI-C quoting first ($'ssh' -> ssh) so a blocked name hidden behind
    # it is still detected at command position.
    command = _decode_ansi_c(command, keep_one_word = True)

    # punctuation_chars splits separators into their own tokens, so command
    # position is detected even in `echo done; rm -rf x` (no whitespace).
    # Keyed to the shell that will actually run this, not to the OS: on a
    # Windows host with bash the non-posix lexer never split on `;`, so the
    # command after a control-flow keyword stayed unread and
    # `if true; then rm -rf x; fi` came back with nothing blocked.
    lexed_posix = _shell_is_posix()
    try:
        if not lexed_posix:
            tokens = shlex.split(command, posix = False)
        else:
            lexer = shlex.shlex(command, posix = True, punctuation_chars = ";&|()`")
            lexer.whitespace_split = True
            tokens = list(lexer)
    except ValueError:
        tokens = command.split()
        lexed_posix = False
    # Which separator tokens the shell only produced because the quoting was
    # stripped. The non-posix (cmd) lexer KEEPS the quote marks, so a quoted
    # `';'` never looks like a separator there and nothing has to be recovered;
    # the split() fallback has no quoting model at all, so it reports nothing
    # either and both shells reach the same verdict.
    quoted_separators = (
        _quoted_separator_indexes(command, tokens, ";&|()`") if lexed_posix else frozenset()
    )
    quoted_redirects = (
        _quoted_redirection_indexes(command, tokens, ";&|()`") if lexed_posix else frozenset()
    )
    exec_flag_indexes, invocation_stops, redirect_indexes = _exec_scan_layout(
        tokens, quoted_separators, quoted_redirects
    )
    # Built only when a sed is actually reached, since it costs a second lex.
    glob_indexes: "frozenset[int] | None" = None

    def _token_basename(tok: str) -> str:
        # Strip glued-on meta-chars (`rm;`) so the basename still matches `rm`.
        tok = tok.strip(";&|()`{}")
        base = os.path.basename(tok).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        return base

    def _exec_child_index(start: int) -> "tuple[int, bool]":
        """The command a `find -exec` actually runs, as ``(index, overflowed)``;
        the index is -1 when the action holds no command word at all.

        Command prefixes forward to their target, so `-exec env sed ...` runs
        sed. Wrapper flags, assignment prefixes and duration operands are
        stepped over as the walk above does, and a wrapper option taking a
        SEPARATE value consumes it too, else that value reads as the command
        (`-exec env -u FOO sed ...` came back with `FOO`). The hop is bounded so
        `-exec env -exec env ...` cannot make this quadratic.

        ``overflowed`` says the bound ran out with words still ahead. That is
        NOT the same as finding nothing, and reporting both as "no child" let a
        long enough chain read as safe: `-exec` + 33 `env` + `rm -f victim ;`
        really deletes. The caller fails closed on it.
        """
        i, steps, wrapper = start, 0, ""
        while i < len(tokens) and steps < _MAX_EXEC_PREFIX_SCAN:
            token = tokens[i]
            if token in _SHELL_SEPARATORS or token in _FIND_EXEC_TERMINATORS:
                return -1, False
            steps += 1
            if wrapper and token in _WRAPPER_VALUE_FLAGS_BY_CMD.get(wrapper, frozenset()):
                # `env -u NAME`, `stdbuf -o L`: the option and its operand, both
                # consumed in ONE step -- the budget bounds the work done per
                # -exec, and stepping over two tokens costs no more than one.
                # An attached spelling (-uNAME, --unset=NAME) carries its own
                # value and is skipped by the plain-option branch below.
                i += 2
                continue
            if wrapper and (
                token.startswith("-") or _ASSIGNMENT_RE.match(token) or token.lstrip("-").isdigit()
            ):
                # `env -i`, `env A=b`, `timeout 5`: the wrapper's own argument.
                i += 1
                continue
            base = _token_basename(token)
            if base in _COMMAND_PREFIXES:
                wrapper = base
                i += 1
                continue
            return i, False
        # Walking off the end means the action really held nothing; stopping on
        # the bound with words still ahead means the child is merely UNREAD.
        return -1, steps >= _MAX_EXEC_PREFIX_SCAN and i < len(tokens)

    expect_command = True  # start of string is a command position
    prefix_pending = False  # last cmd-position token was a wrapper (env/time/xargs/...)
    prefix_command = ""  # which wrapper that was, for its own value-taking options
    skip_operand = False  # consume a wrapper/conditional operand, not the command
    sed_indexes: "list[int]" = []  # command-position sed words, for the `e` scan below
    sed_xargs: "dict[int, int]" = {}  # sed word -> the xargs that builds its argv
    xargs_index = -1  # an xargs awaiting the command it wraps
    for token_index, token in enumerate(tokens):
        if skip_operand:
            # `exec -a NAME cmd` and `if exist FILE cmd` both put an operand
            # where the command word would otherwise be.
            skip_operand = False
            continue
        if expect_command and token.lower() in _WIN_CONDITIONAL_KEYWORDS:
            skip_operand = token.lower() != "not"
            continue
        if prefix_pending and token == "-a":
            skip_operand = True
            continue
        if token_index in redirect_indexes:
            # The shell performs the redirection and hands the command neither
            # word, so command position is unchanged by it: `> out.txt rm -rf
            # victim` and `2>&1 rm -rf victim` both really delete, while reading
            # `out.txt` (and the `1`) as the command word left the `rm` behind
            # it in argument position and the blocklist came back empty.
            continue
        # A keyword only separates where a COMMAND may start (see below).
        # A quoted operator is DATA the command receives, not a separator, so it
        # leaves command position alone: `printf '%s' '|&' rm` and
        # `grep '|&' rm file` run nothing and must not be refused.
        if (_looks_like_separator(token) and token_index not in quoted_separators) or (
            token in _SHELL_KEYWORDS_AS_SEP and expect_command
        ):
            expect_command = True
            prefix_pending = False
            prefix_command = ""
            xargs_index = -1
            continue
        if token.startswith("-"):
            # A wrapper option whose value is a SEPARATE token precedes that
            # value, not the wrapped command. Without consuming it the value is
            # read as the command word and the real command behind it is never
            # reached: `env -u PATH rm -rf x` and `xargs -I {} rm -rf build`
            # both came back empty. An attached spelling (-uPATH, --unset=PATH)
            # carries its own value and falls through to the plain-flag case.
            if prefix_pending and token in _WRAPPER_VALUE_FLAGS_BY_CMD.get(
                prefix_command, frozenset()
            ):
                skip_operand = True
                continue
            # Flags belong to the active command, but keep expect_command while a
            # wrapper prefix awaits its command (`stdbuf -oL cmd`, `xargs -- cmd`).
            if not prefix_pending:
                expect_command = False
            continue
        if not expect_command:
            continue
        # A redirection may precede the command word (`</dev/null rm -rf x`).
        if _REDIR_PREFIX_RE.match(token):
            continue
        # FOO=bar assignment prefix; next non-assignment token is the command.
        if _ASSIGNMENT_RE.match(token):
            continue
        # Numeric wrapper arg: `timeout 1 cmd` / `nice -n 5 cmd`.
        if prefix_pending and token.lstrip("-").isdigit():
            continue
        base = _token_basename(token)
        if _is_sed_command(base):
            sed_indexes.append(token_index)
            if xargs_index >= 0:
                sed_xargs[token_index] = xargs_index
        if base in _BLOCKED_COMMANDS:
            blocked.add(base)
        else:
            blocked |= _blocked_matching_glob(base)
        # Wrappers (env/time/xargs/sudo) consume one command; the next non-flag,
        # non-numeric token is the real command. sudo is also in _BLOCKED_COMMANDS.
        if base in _COMMAND_PREFIXES:
            if base == "xargs" and xargs_index < 0:
                xargs_index = token_index
            prefix_pending = True
            prefix_command = base
            continue
        expect_command = False
        prefix_pending = False
        prefix_command = ""
        xargs_index = -1

    # `alias zap='rm -rf'` stores a command bash runs when the alias is invoked,
    # so the body is scanned as a command in its own right.
    for i, tok in enumerate(tokens):
        if _token_basename(tok) != "alias":
            continue
        for nxt in tokens[i + 1 :]:
            if nxt in _SHELL_SEPARATORS:
                break
            _name, _sep, _value = nxt.partition("=")
            if _sep and _value:
                blocked |= _find_blocked_commands(_value)

    # `find ... -exec CMD ... ;`, `-execdir CMD ... ;` and fd's `-x` / `-X` /
    # `--exec` / `--exec-batch` all invoke CMD directly (_exec_scan_layout picks
    # which spellings count where). Reading only find's own flags left every fd
    # form unscanned, so `fd -x rm -rf x` and `fd -x sed '1e rm -f victim' {}`
    # -- both verified to run -- reached the hard blocklist as nothing at all.
    for i, tok in enumerate(tokens):
        # The long flags also carry the command attached (fd --exec=rm), where
        # the value is command position rather than a discarded option argument.
        attached = ""
        if tok[:2] in {"-x", "-X"} and len(tok) > 2 and i in exec_flag_indexes:
            # fd takes the command attached to the short option (`fd ... -xrm`),
            # where the value is command position rather than an option argument.
            attached = tok[2:].strip("\"'")
        elif "=" in tok and tok.split("=", 1)[0] in _ATTACHED_EXEC_FLAGS:
            attached = tok.split("=", 1)[1].strip("\"'")
        if attached:
            attached_base = _token_basename(attached.split()[0])
            if _is_sed_command(attached_base):
                # The words after the flag are that sed's arguments, so its
                # program is screened from the FLAG. fd 9 actually takes them
                # as search paths and runs nothing, so this only ever blocks
                # a command that could not have worked anyway; a spelling
                # that does forward them would otherwise be a free pass.
                sed_indexes.append(i)
            if attached_base in _BLOCKED_COMMANDS:
                blocked.add(attached_base)
            else:
                blocked |= _blocked_matching_glob(attached_base)
        if i in exec_flag_indexes and i + 1 < len(tokens):
            # The word right after the flag AND the command it forwards to: a
            # wrapper is a command in its own right (`-exec sudo ls`) as well as
            # a step on the way to another one (`-exec env rm -rf x`), so
            # dropping either half loses a real detection.
            child, prefix_overflowed = _exec_child_index(i + 1)
            if prefix_overflowed:
                # The wrapper chain outran the hop budget, so the command that
                # finally runs was never reached: block the chain itself rather
                # than let `-exec env ...x33 rm -f victim ;` ride in behind it.
                blocked.add(_token_basename(tokens[i + 1]))
                continue
            exec_words = [i + 1] if child in (-1, i + 1) else [i + 1, child]
            for word in exec_words:
                base = _token_basename(tokens[word])
                if _is_sed_command(base):
                    # find runs its -exec child directly, but the walk above only
                    # reaches `find`, so a sed there never got its program
                    # screened (`find . -exec sed '1e rm -f victim' {} +`, and
                    # behind a wrapper `find . -exec env sed '1e ...' {} +`).
                    sed_indexes.append(word)
                if base in _BLOCKED_COMMANDS:
                    blocked.add(base)
                else:
                    blocked |= _blocked_matching_glob(base)

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
        # Git Bash mangles a lone /c into a path, so models write //c and MSYS
        # hands cmd back a single slash; /k runs the payload the same way.
        is_win_c = _win_switch(tok_lower) in ("/c", "/k")
        if not (is_unix_c or is_win_c) or i < 1 or i + 1 >= len(tokens):
            continue
        # Look back past flags for the shell binary. Windows flags and absolute
        # paths both start with /, so only skip things shaped like a whole
        # switch (/s, /v:on) and never a program spelled as a path (/bin/bash).
        for j in range(i - 1, -1, -1):
            prev = tokens[j]
            if prev.startswith("-"):
                continue  # skip Unix flags like --login, -l
            # Git Bash doubles the slash on these switches too, so normalise
            # them like the trigger: else `cmd //v:on //c powershell` stops here.
            if is_win_c and _CMD_SWITCH_RE.fullmatch(_win_switch(prev)):
                continue  # skip Windows switches like /s, /q, /v:on
            prev_base = os.path.basename(prev).lower()
            if is_unix_c and prev_base in _SHELLS:
                blocked |= _find_blocked_commands(tokens[i + 1])
            elif is_win_c and prev_base in _SHELLS_WIN:
                # The cmd lexer keeps the marks, so `cmd /c "powershell ls"`
                # would recurse on a first word of `"powershell` and match nothing.
                payload = tokens[i + 1]
                if len(payload) > 1 and payload[0] == '"' and payload[-1] == '"':
                    payload = payload[1:-1]
                blocked |= _find_blocked_commands(payload)
            break  # stop at first non-flag token

    # `cmd /c start "" prog` puts prog in a command position the scan above
    # sees only as an argument, so screen what start actually launches.
    for i, token in enumerate(tokens):
        if os.path.basename(token).lower() not in ("start", "start.exe"):
            continue
        j = i + 1
        while j < len(tokens) and _win_switch(tokens[j].lower()) in _START_SWITCHES:
            # /d C:\dir and friends carry their value in the next token.
            j += 2 if _win_switch(tokens[j].lower()) in _START_SWITCHES_WITH_VALUE else 1
        # cmd reads a quoted first argument as the window title, putting the
        # program one token further on. Reading that second token only when the
        # first is recognisably a title keeps `echo start notepad powershell`
        # runnable; deciding whether `start` itself is executed is deliberately
        # not attempted, since every local approximation of it under-approximated
        # and let a real launch through.
        if j < len(tokens):
            blocked |= _find_blocked_commands(tokens[j])
        if j + 1 < len(tokens) and _is_start_title(tokens[j]):
            k = j + 1
            # `start "my window" /min prog` puts switches after the title too.
            while k < len(tokens) and _win_switch(tokens[k].lower()) in _START_SWITCHES:
                k += 2 if _win_switch(tokens[k].lower()) in _START_SWITCHES_WITH_VALUE else 1
            if k < len(tokens):
                blocked |= _find_blocked_commands(tokens[k])

    # sed's `e COMMAND` hands COMMAND to the shell, a real command position the
    # scan above sees only as a text argument, so screen it like `bash -c`. The
    # pattern-space forms yield an empty payload; the auto gate prompts on those.
    sed_limit = _sed_scan_limit(len(sed_indexes))
    # Built at most once per call, and only when some program actually names a
    # variable, so a line packed with sed words stays linear.
    sed_vars: "dict[str, str] | None" = None
    sed_bindings: "list[tuple[int, str, str | None]] | None" = None
    sed_cursor = 0
    # Visited left to right so the binding cursor below only moves forward.
    for i in sorted(set(sed_indexes)):
        # A script --sandbox / --posix stops sed compiling is already left out of
        # the program (_sed_invocation), so a name inside one is never blocked.
        if glob_indexes is None:
            glob_indexes = (
                _unquoted_glob_indexes(command, tokens, ";&|()`") if lexed_posix else frozenset()
            )
        alternatives, scan_overflowed, _live = _sed_invocation(
            tokens, i, sed_limit, invocation_stops, redirect_indexes, glob_indexes
        )
        program = "\n".join(alternatives)
        if scan_overflowed:
            # The script sits past the scan window, so an empty program here is
            # only ignorance: block the sed itself rather than let an
            # `e rm -rf ~` ride in behind enough padding options.
            blocked.add(_token_basename(tokens[i]))
            continue
        if _sed_program_is_a_placeholder(program):
            # find rewrites `{}` before the child starts, so this is not a
            # program that was read (see _sed_program_is_a_placeholder).
            blocked.add(_token_basename(tokens[i]))
            continue
        if i in sed_xargs and _xargs_hides_sed_program(tokens, sed_xargs[i], i, program):
            # The program comes off stdin or out of an -I placeholder, so it is
            # not in the text to read at all (see _xargs_hides_sed_program).
            blocked.add(_token_basename(tokens[i]))
            continue
        if "$" in program:
            # A program held in a variable (p='...e rm -f victim'; sed "$p" f)
            # only shows its `e` once the reference is resolved. shlex kept the
            # quoted value whole, newlines and all, so the binding is exact.
            # Only the assignments AHEAD of this sed are in scope, and the last
            # of them wins, which is the pair that `p='1,3p';
            # p='1e rm -f victim'; sed "$p" input` turns on.
            if sed_bindings is None:
                sed_bindings = _assignment_bindings(tokens, quoted_separators)
                sed_vars = {}
            sed_cursor = _bindings_before(sed_bindings, sed_cursor, i, sed_vars)
        for alternative in alternatives:
            for variant in _sed_program_variants(alternative, sed_vars or {}):
                for payload in _sed_exec_payloads(variant):
                    if payload:
                        blocked |= _find_blocked_commands(payload)

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
# setsid/exec/builtin forward to a child command just like env/nohup, so
# classification continues at the child rather than stopping at the wrapper.
_AUTO_SAFE_WRAPPERS = frozenset(
    {
        "env",
        "command",
        "builtin",
        "exec",
        "time",
        "timeout",
        "nice",
        "ionice",
        "stdbuf",
        "nohup",
        "setsid",
    }
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
# Split a camelCase boundary with an underscore (runCommand -> run_Command) so
# the term-boundary MCP regexes match camelCase tool names too.
_CAMEL_CASE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# A name that reads (get_release, search_code, list_invoices) names its SUBJECT,
# not the action, so the impact and runtime-noun patterns below must not fire on
# it, or the everyday read tools of every server would prompt.
_AUTO_READ_MCP_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:get|list|read|search|find|fetch|query|describe|show|view|"
    r"inspect|status|info|count|exists|lookup|browse|preview|download|export|"
    r"history|log|logs|diff|compare|summarize|summarise)(?:[_\-]|$)",
    re.IGNORECASE,
)
# The runtime nouns alone (python, code, script, notebook) name a subject as
# often as an action, so they only count when nothing reads.
_AUTO_EXEC_MCP_VERB_ONLY_RE = re.compile(
    r"(?:^|[_\-])(?:exec|execute|run|eval|spawn|invoke|launch|shell|bash|zsh|"
    r"powershell|pwsh|terminal|subprocess|interpreter)(?:[_\-]|$)",
    re.IGNORECASE,
)
_AUTO_EXEC_MCP_RUNTIME_NOUN_RE = re.compile(
    r"(?:^|[_\-])(?:python[0-9.]*|node|nodejs|deno|bun|ruby|perl|php|code|"
    r"script|repl|sandbox|notebook)(?:[_\-]|$)",
    re.IGNORECASE,
)
# An MCP tool that runs arbitrary commands/code (run_command, eval_code, bash)
# is as unsafe as a terminal call and runs on the server, outside the terminal
# sandbox, so auto gates it. Whole name segments only, so get_command and
# list_shells stay read.
_AUTO_EXEC_MCP_TOOL_RE = re.compile(
    r"(?:^|[_\-])(?:"
    r"exec|execute|run|eval|spawn|invoke|launch|"
    r"shell|bash|zsh|powershell|pwsh|terminal|subprocess|interpreter|"
    # A bare runtime name (mcp__srv__python, __node, __code) is an execution
    # tool even without a verb: its payload runs on the MCP server.
    r"python[0-9.]*|node|nodejs|deno|bun|ruby|perl|php|code|script|repl|sandbox|notebook"
    r")(?:[_\-]|$)",
    re.IGNORECASE,
)
# A destructive verb as a whole name segment: an honestly-named MCP tool
# (delete_file, delete_repo, drop_table, purge_index) runs outside the terminal
# sandbox and causes data loss, so auto prompts on it even when the arguments
# carry no SQL/HTTP mutation marker. Non-destructive mutations (create/update/
# add/set/insert/patch) still run; a read that merely contains one of these as
# a substring (undelete, list_removed) does not match on the segment boundary.
_AUTO_DESTRUCTIVE_MCP_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:"
    r"delete|destroy|drop|purge|wipe|truncate|erase|remove|unlink|"
    r"teardown|revoke|terminate|uninstall|clear|reset|empty|flush|prune|expire"
    r")(?:[_\-]|$)",
    re.IGNORECASE,
)
# A name without separators (mcp__srv__runcommand, __shellexec) never reaches the
# segment boundaries above, so match the verb+object compounds directly.
_MCP_EXEC_VERBS = r"execute|exec|run|eval|spawn|invoke|launch|start"
_MCP_EXEC_OBJECTS = r"command|cmd|shell|script|code|process|program|bash|terminal|proc|task|job"
_AUTO_EXEC_MCP_COMPOUND_RE = re.compile(
    r"(?:^|[_\-])(?:"
    rf"(?:{_MCP_EXEC_VERBS})(?:{_MCP_EXEC_OBJECTS})"
    rf"|(?:{_MCP_EXEC_OBJECTS})(?:{_MCP_EXEC_VERBS})"
    r")(?:[_\-]|$)",
    re.IGNORECASE,
)
# The verbs an MCP tool name may carry and still run without a prompt: reads, and
# ordinary writes that create or edit a record. Destructive, privilege and
# money-moving verbs are caught by the patterns above before this is consulted.
_AUTO_KNOWN_MCP_VERBS = frozenset(
    {
        # read / inspect
        "get",
        "list",
        "read",
        "search",
        "find",
        "fetch",
        "query",
        "describe",
        "show",
        "view",
        "inspect",
        "status",
        "info",
        "count",
        "exists",
        "resolve",
        "lookup",
        "browse",
        "diff",
        "log",
        "logs",
        "history",
        "summarize",
        "summarise",
        "analyze",
        "analyse",
        "validate",
        "check",
        "test",
        "ping",
        "preview",
        "head",
        "stat",
        "download",
        "export",
        "render",
        "format",
        "parse",
        "compare",
        "explain",
        "select",
        "retrieve",
        "audit",
        "review",
        "monitor",
        "trace",
        "profile",
        "benchmark",
        "lint",
        "detect",
        "classify",
        "rank",
        "score",
        "predict",
        "infer",
        "evaluate",
        # ordinary writes
        "create",
        "add",
        "insert",
        "update",
        "edit",
        "modify",
        "set",
        "put",
        "patch",
        "post",
        "send",
        "write",
        "append",
        "upload",
        "comment",
        "assign",
        "label",
        "tag",
        "move",
        "rename",
        "copy",
        "clone",
        "sync",
        "merge",
        "close",
        "reopen",
        "open",
        "start",
        "stop",
        "pause",
        "resume",
        "cancel",
        "schedule",
        "notify",
        "register",
        "save",
        "store",
        "apply",
        "submit",
        "request",
        "generate",
        "convert",
        "translate",
        "complete",
        "index",
        "ingest",
        "embed",
        "train",
        "call",
        "load",
        "init",
        "configure",
        "config",
        "upsert",
        "retry",
        "replay",
        "approve",
        "reject",
        "acknowledge",
        "annotate",
        "draft",
        "subscribe",
        "watch",
        "listen",
        "poll",
        "wait",
        "sleep",
        # browser / ui drivers
        "navigate",
        "click",
        "type",
        "scroll",
        "hover",
        "press",
        "screenshot",
        "capture",
        "snapshot",
        "extract",
        "crawl",
        "scrape",
        "fill",
        "focus",
        # data shaping
        "sort",
        "filter",
        "group",
        "aggregate",
        "split",
        "chunk",
        "tokenize",
        "encode",
        "decode",
        "hash",
        "sign",
        "verify",
        "compress",
        "decompress",
        "dedupe",
        "normalize",
        "normalise",
        "sanitize",
        "sanitise",
        "redact",
        "mask",
        "compute",
        "calculate",
        "solve",
        "simulate",
        "plot",
        "chart",
        # build / ship
        "build",
        "compile",
        "bundle",
        "package",
        "backup",
        "restore",
        "ask",
        "answer",
        "chat",
        "prompt",
        "respond",
        "reply",
        "transcribe",
    }
)


# Verbs the patterns above already gate. A name carrying one is still screenable
# even though reaching this point means it did not match: `undelete` is the
# reverse of a verb this classifier knows.
_AUTO_GATED_MCP_VERBS = frozenset(
    {
        "delete",
        "remove",
        "drop",
        "destroy",
        "purge",
        "wipe",
        "truncate",
        "clear",
        "reset",
        "empty",
        "flush",
        "prune",
        "expire",
        "revoke",
        "grant",
        "authorize",
        "authorise",
        "elevate",
        "escalate",
        "impersonate",
        "promote",
        "transfer",
        "payout",
        "charge",
        "refund",
        "publish",
        "deploy",
        "release",
        "install",
        "uninstall",
        "lock",
        "mount",
    }
)
_AUTO_MCP_VERB_VOCAB = _AUTO_KNOWN_MCP_VERBS | _AUTO_GATED_MCP_VERBS


def _mcp_verb_is_known(tool_name: str) -> bool:
    """Whether any term of an MCP tool name is a verb this classifier knows.
    A name with none of them cannot be screened, so the caller fails closed."""
    for part in re.split(r"[_\-]+", tool_name.lower()):
        if not part:
            continue
        if part in _AUTO_KNOWN_MCP_VERBS:
            return True
        # The reverse or the repeat of a recognised verb (undelete, reopen,
        # resend) is just as screenable as the verb itself.
        for prefix in ("un", "re"):
            if part.startswith(prefix) and part[len(prefix) :] in _AUTO_MCP_VERB_VOCAB:
                return True
    return False


# Privilege escalation over MCP: granting a role/permission/policy hands out
# access the operator never approved. An unambiguous privilege verb matches on
# its own; the soft verbs below (assign/add/set/attach/bind) only count next to a
# privilege noun, so assign_issue / add_label keep running.
_AUTO_PRIVILEGE_MCP_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:grant|authorize|authorise|elevate|escalate|impersonate|sudo|promote)(?:[_\-]|$)",
    re.IGNORECASE,
)
# Money movement and other irreversible external side effects: an MCP call
# that pays, refunds, wires or transfers funds cannot be undone by the
# operator, so it asks even though it is not "destructive" in the fs sense.
_AUTO_HIGH_IMPACT_MCP_RE = re.compile(
    r"(?:^|[_\-])(?:transfer|payout|payment|pay|charge|refund|wire|remit|"
    r"withdraw|deposit|invoice|subscription|subscriptions|billing|"
    r"publish|deploy|release)(?:[_\-]|$)",
    re.IGNORECASE,
)
_AUTO_PRIVILEGE_MCP_NOUN_RE = re.compile(
    r"(?:^|[_\-])(?:role|roles|permission|permissions|privilege|privileges|acl|acls|"
    r"policy|policies|scope|scopes|grant|grants|membership|member|members|"
    r"collaborator|collaborators|admin|owner)(?:[_\-]|$)",
    re.IGNORECASE,
)
_AUTO_PRIVILEGE_MCP_SOFT_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:assign|add|set|attach|bind|put|update|create)(?:[_\-]|$)",
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
# Loaders that can execute code embedded in the data they deserialize; gated by
# receiver module (torch.load, yaml.load) since bare `load` is too common.
_AUTO_UNSAFE_PY_LOAD_MODULES = frozenset({"torch", "joblib", "cloudpickle", "yaml"})
# The load entry points on those modules. yaml.load runs whatever its Loader=
# builds, and !!python/object/apply in the data is a call, so it asks like the
# pickle-backed ones do. yaml.safe_load is untouched.
_AUTO_UNSAFE_PY_LOAD_ATTRS = frozenset({"load", "load_all"})
# Loader classes: the same deserialize one level down (yaml.Loader(s).get_data())
# and what a custom loader subclasses. Ordinary words, so matched by receiver.
_AUTO_UNSAFE_PY_LOAD_CLASSES = frozenset({"Loader", "Constructor"})
# Names no other library uses, so they are matched wherever they appear rather
# than by receiver: an alias, a subclass, a loop, a helper or a factory all name
# one of these somewhere, and following the value through every binding form is
# a game without an end.
_AUTO_UNSAFE_YAML_LOADERS = frozenset(
    {
        "unsafe_load",
        "unsafe_load_all",
        "full_load",
        "full_load_all",
        "UnsafeLoader",
        "CUnsafeLoader",
        "FullLoader",
        "CFullLoader",
        "CLoader",
        "UnsafeConstructor",
        "FullConstructor",
    }
)
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
# Destructive filesystem calls in the python tool pair with the terminal `rm`
# gate, so auto prompts. `rmtree`/`unlink`/`rmdir`/`removedirs` name only fs
# deletion, so any receiver counts; `remove` is gated on the `os` module alone so
# a benign list.remove() stays out. A bare import binding is caught separately.
_PY_DESTRUCTIVE_FS_ATTRS = frozenset({"unlink", "rmtree", "rmdir", "removedirs"})
# psutil ends a process exactly as os.kill does, which is already gated.
_PY_PROCESS_KILL_ATTRS = frozenset({"kill", "terminate", "send_signal", "suspend"})
_PY_PROCESS_MODULES = frozenset({"psutil"})
# Gated only on the os module (or an alias) so a truncate/remove-like method on
# another receiver stays out. os.truncate zeroes a file like the gated terminal
# `truncate`; os.kill/os.killpg terminate like the blocked `kill`.
_PY_DESTRUCTIVE_FS_OS_ATTRS = frozenset({"remove", "truncate", "ftruncate", "kill", "killpg"})
_PY_DESTRUCTIVE_FS_IMPORT_NAMES = frozenset(
    {
        "remove",
        "unlink",
        "rmtree",
        "rmdir",
        "removedirs",
        "truncate",
        "ftruncate",
        "kill",
        "killpg",
    }
)
# Modules whose destructive names are the same calls: posix/nt are os's
# platform twins (from posix import unlink; nt.remove(...)).
_PY_DESTRUCTIVE_FS_MODULES = ("os", "posix", "nt", "shutil", "pathlib")

# Reading these off the host escapes the intent of "read-only is safe": they
# hold credentials. Path traversal (../) escapes the per-session workdir.
_SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[/\\])\.(?:ssh|aws|azure|gnupg|docker|kube|config/gcloud|config/gh)(?:[/\\]|$)"
    r"|\.(?:netrc|npmrc|pypirc|git-credentials|env)(?:$|[/\\.\s'\"])"
    # User-level persistence: a write into a shell startup file or an XDG
    # autostart/user-service dir runs on the next login, the /etc boot-hook risk
    # without root, and the sandbox does not confine absolute paths (>> ~/.bashrc
    # reaches the real file). Rarely read in a dev session, so gating any
    # reference does not over-prompt.
    r"|(?:^|[/\\\s'\"=])\.(?:bashrc|bash_profile|bash_login|bash_logout|bash_aliases"
    r"|profile|zshrc|zprofile|zshenv|zlogin|zlogout|kshrc|cshrc|tcshrc|login"
    r"|xprofile|xinitrc|xsession)(?:$|[/\\\s'\"])"
    r"|(?:^|[/\\])\.config[/\\](?:autostart|systemd[/\\]user|environment\.d)(?:[/\\]|$)"
    r"|id_rsa|id_ed25519|id_ecdsa|id_dsa"
    # Hugging Face stores the login token at ~/.cache/huggingface/token and the
    # legacy ~/.huggingface/token (plus the multi-token store stored_tokens); the
    # rest of that cache is model data, so only the credential files match. The
    # optional leading dot covers the .huggingface dotdir form.
    r"|(?:^|[/\\])\.?huggingface[/\\](?:token|stored_tokens)(?:$|[/\\.\s'\"])"
    # /etc/ssh holds the host private keys (ssh_host_*_key); the whole dir is
    # sensitive, not just passwd/shadow/sudoers. The trailing group is the system
    # persistence set: a write there (tee /etc/ld.so.preload, a drop into
    # /etc/cron.d or /etc/systemd) installs a boot/login/preload hook, and the
    # sandbox keeps host-fs access. Effectively write-only in a dev session, so
    # gating any reference does not over-prompt.
    r"|credentials|/etc/(?:passwd|shadow|sudoers|ssh(?:[/\\]|$)"
    r"|cron[^/\\]*(?:[/\\]|$)|profile\.d(?:[/\\]|$)|systemd(?:[/\\]|$)"
    r"|ld\.so\.preload(?:$|[/\\.\s'\"])|ld\.so\.conf|rc\.local|init\.d(?:[/\\]|$))"
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


# The credential-path pattern is superlinear in the text length and a real path
# is short, so text far past any real path fails closed: the caller asks rather
# than spending unbounded time. Ordinary commands are far below these bounds.
_MAX_PATH_SCAN_CHARS = 2048
_MAX_TERMINAL_SCAN_CHARS = 4096


def _references_sensitive_path(text: str) -> bool:
    """True if a command or string literal reads a credential path or escapes
    the sandbox workdir via parent traversal."""
    if len(text) > _MAX_PATH_SCAN_CHARS:
        return True
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


# Bash expands $'...' to a single word, so a separator inside it is data. Callers
# that tokenize the decoded text neutralize these first, otherwise
# `printf '%s' $'a\\nrm -rf x'` reads as two commands and the printf is refused.
_ANSI_C_SEPARATOR_RE = re.compile(r"[\s;&|()<>`]")
# A newline revealed by ANSI-C decoding, and the mark standing in for it. Any
# character shlex leaves inside a quoted word serves, as long as the boundary
# regex in _find_blocked_commands does not read it as the start of a command.
_ANSI_C_NEWLINE_MARK = "\x03"
_ANSI_C_NEWLINE_RE = re.compile(r"[\n\r]")


def _folded_str_literal(node) -> "str | None":
    """The string an expression evaluates to when built only from string literals
    ("un" + "link", f"un{'link'}"), else None. Resolves a name spelled
    dynamically but fully known at parse time."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _folded_str_literal(node.left)
        right = _folded_str_literal(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            piece = _folded_str_literal(value)
            if piece is None:
                return None
            parts.append(piece)
        return "".join(parts)
    if isinstance(node, ast.FormattedValue) and node.format_spec is None:
        return _folded_str_literal(node.value)
    return None


def _decode_ansi_c(command: str, *, keep_one_word: bool = False) -> str:
    """Decode bash ANSI-C quoted words (cat $'/etc/pass\\x77d' -> cat /etc/passwd)
    so an escape-obfuscated path is visible to the scan. Fail-open: only adds
    detections. With ``keep_one_word`` the decoded text cannot introduce new
    shell syntax, which is what bash does with it."""

    def dec(m):
        try:
            text = bytes(m.group(1), "utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            return m.group(0)
        if not keep_one_word:
            return text
        if _ANSI_C_NEWLINE_MARK not in text:
            # Re-quote rather than flatten: bash gives the command ONE word
            # however much whitespace the decoding reveals, and a sed program
            # ends its COMMENT at a newline, so the spaces and the `#` around it
            # all carry meaning. An apostrophe is re-quoted `'\''` for the same
            # reason. The newline stands as a MARK because it is data for the
            # command bash starts, not a place a new one begins, and the
            # boundary regex below would read a bare one as the latter;
            # _sed_invocation puts it back where its meaning matters.
            body = _ANSI_C_NEWLINE_RE.sub(_ANSI_C_NEWLINE_MARK, text)
            return "'" + body.replace("'", "'\\''") + "'"
        return _ANSI_C_SEPARATOR_RE.sub("_", text)

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


def _command_references_sensitive(command: str) -> bool:
    """True if a shell command reads/writes a credential path or escapes the
    sandbox workdir (../), after undoing the shell expansions that would hide it:
    quotes/backslash escapes, brace/parameter/ANSI-C expansion and NAME=value
    prefixes, so `cat /et\\c/passwd`, `p="/proc/$PPID"; cat $p/environ` and
    `cat /e{t,}c/pass?d` are all caught."""
    stripped = _SHELL_QUOTE_RE.sub("", command).replace("\\", "")
    candidates = []
    for c in (command, stripped, _decode_ansi_c(command)):
        c_param = _expand_param_defaults(c)
        candidates.extend((c, c_param, _expand_braces(c_param), _expand_shell_assignments(c_param)))
    return any(_glob_hits_sensitive(c) or _references_sensitive_path(c) for c in candidates)


def _terminal_is_potentially_unsafe(command: str) -> bool:
    """Classify a terminal command for auto mode (fail closed)."""
    if not command or not command.strip():
        return False
    # Redirections and substitutions can hide writes or nested commands; a
    # quoted ">" false-positives into a prompt, which is the safe direction.
    if ">" in command or "`" in command or "$(" in command or "<(" in command:
        return True
    # Reads that escape the sandbox workdir (../) or hit credential paths are
    # not "safe" reads; ask before running them.
    if _command_references_sensitive(command):
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
            or (token in _SHELL_KEYWORDS_AS_SEP and expect_command)
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
    # Naming a code-executing loader asks, wherever the name appears. Presence
    # rather than dataflow: a loader can be aliased, subclassed, packed into a
    # container, returned from a helper or picked by a conditional, and following
    # it through all of those is a game without an end. A safe read names none of
    # these, so yaml.safe_load and json.load are unaffected.
    _module_names = set(_AUTO_UNSAFE_PY_LOAD_MODULES)  # receivers: yaml.load
    _bare_names = set(_AUTO_UNSAFE_YAML_LOADERS)  # loaders named on their own
    _imported_modules = set()  # only the module itself, for the return rule
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _root = alias.name.split(".")[0]
                if _root in _AUTO_UNSAFE_PY_LOAD_MODULES:
                    _imported_modules.add(alias.asname or _root)
        elif isinstance(node, ast.ImportFrom):
            _root = (node.module or "").split(".")[0]
            for alias in node.names:
                if alias.name in _AUTO_UNSAFE_YAML_LOADERS or (
                    _root in _AUTO_UNSAFE_PY_LOAD_MODULES
                    and (
                        alias.name in _AUTO_UNSAFE_PY_LOAD_ATTRS
                        or alias.name in _AUTO_UNSAFE_PY_LOAD_CLASSES
                    )
                ):
                    _bare_names.add(alias.asname or alias.name)
                elif _root in _AUTO_UNSAFE_PY_LOAD_MODULES:
                    # from yaml import loader as yl, so yl.Loader still reads as one.
                    _module_names.add(alias.asname or alias.name)
    _module_names |= _imported_modules

    def _loader_receiver(node) -> bool:
        # yaml, a submodule of it (yaml.loader.Loader), or an import alias.
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id in _module_names

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _AUTO_UNSAFE_YAML_LOADERS:
                return True
            if (
                node.attr in _AUTO_UNSAFE_PY_LOAD_ATTRS or node.attr in _AUTO_UNSAFE_PY_LOAD_CLASSES
            ) and _loader_receiver(node.value):
                return True
        elif isinstance(node, ast.Name) and node.id in _bare_names:
            return True
        # Handing the module out of a function hands out every loader on it, and
        # the caller's name for it cannot be seen from here.
        elif isinstance(node, (ast.Return, ast.Lambda)):
            _out = node.value if isinstance(node, ast.Return) else node.body
            if isinstance(_out, ast.Name) and _out.id in _imported_modules:
                return True
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


# Argument names that carry a credential outward regardless of their value.
_MCP_CREDENTIAL_KEY_RE = re.compile(
    r"^(?:authorization|proxy-authorization|cookie|set-cookie|"
    r"x-api-key|api[-_]?key|apikey|x-auth-token|auth[-_]?token|access[-_]?token|"
    r"refresh[-_]?token|id[-_]?token|bearer|private[-_]?key|secret[-_]?key|"
    r"client[-_]?secret|password|passwd|session[-_]?token)$",
    re.IGNORECASE,
)


def _mcp_arguments_reference_sensitive(arguments) -> bool:
    """True if any string in an MCP call's arguments names a credential path, a
    credential/secret environment variable (get_env {"name": "OPENAI_API_KEY"}),
    or a cloud-metadata host (fetch_url {"url": "http://169.254.169.254/..."})."""

    def key_is_credential(key) -> bool:
        return isinstance(key, str) and bool(_MCP_CREDENTIAL_KEY_RE.match(key.strip()))

    def walk(value, is_prose: bool = False) -> bool:
        if isinstance(value, str):
            # A path can be carried under any argument name, so prose keys are
            # skipped rather than path keys allowlisted: an issue body mentioning
            # a credential file is text to store, not a file to open.
            if is_prose:
                return False
            return (
                _references_sensitive_path(value)
                or bool(_AUTO_SENSITIVE_MCP_NOUN_RE.search(value))
                or bool(_MCP_METADATA_HOST_RE.search(value))
            )
        if isinstance(value, dict):
            if any(key_is_credential(k) for k in value):
                return True
            return any(
                walk(v, is_prose or (isinstance(k, str) and k.lower() in _MCP_PROSE_KEYS))
                for k, v in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(walk(v, is_prose) for v in value)
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


# Argument names that carry free text the tool stores or displays rather than
# acts on, so a path or a statement mentioned inside them is a mention.
_MCP_PROSE_KEYS = frozenset(
    {
        "text",
        "body",
        "message",
        "msg",
        "description",
        "comment",
        "content",
        "title",
        "summary",
        "note",
        "notes",
        "prompt",
        "caption",
        "reason",
        "markdown",
        "blocks",
        "detail",
        "details",
        "context",
    }
)
# Argument names that carry a statement the tool will execute, as opposed to
# free text the tool will merely store or display.
_MCP_QUERY_KEYS = frozenset(
    {
        "query",
        "sql",
        "statement",
        "stmt",
        "command",
        "cmd",
        "script",
        "expression",
        "expr",
        "filter",
        "pipeline",
        "aggregate",
        "mutation",
        "operation",
        "graphql",
        "queries",
        "statements",
        "commands",
    }
)


def _mcp_arguments_mutate(arguments) -> bool:
    """True if an MCP call's arguments carry a mutating command, so a read-named
    but write-capable tool (query_database {"query": "DELETE FROM runs"},
    query_graphql {"query": "mutation { deleteIssue(id: 1) }"}, or an HTTP tool
    {"method": "DELETE"}) asks."""

    def walk(value, in_query: bool = False) -> bool:
        if isinstance(value, str):
            # Prose that merely mentions DELETE FROM (a chat message, an issue
            # body) is not a statement this call will run.
            if not in_query:
                return False
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
            return any(
                walk(v, in_query or (isinstance(k, str) and k.lower() in _MCP_QUERY_KEYS))
                for k, v in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(walk(v, in_query) for v in value)
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
    # The same for the navigation sinks: location['assign'](...),
    # location["href"] = URL. Anchored to location (dotted or bracketed) so an
    # ordinary str['replace'](...) or obj['href'] read stays static.
    r"(?:\blocation|\[\s*[\"']location[\"']\s*\])\s*\[\s*[\"'](?:assign|replace)[\"']\s*\]\s*\(|"
    r"(?:\blocation|\[\s*[\"']location[\"']\s*\])\s*\[\s*[\"']href[\"']\s*\]"
    r"\s*=\s*[\"'`]?\s*(?:https?:|/)|"
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
# search_conversation only reads this chat's own past turns, so auto mode would otherwise
# prompt for approval on every call. deep_research runs no code and reaches nothing either: it
# only reports that the user's own armed research is starting, and without it here
# is_high_risk_tool_call's unknown-name default would prompt on every handoff.
_ALWAYS_SAFE_TOOLS = frozenset(
    {"web_search", "search_knowledge_base", "search_conversation", "deep_research"}
)


def is_always_safe_tool(name: str) -> bool:
    """True for tools that never need an auto-mode prompt on any arguments, so a
    caller (e.g. the streaming provisional card) can allow them before the full
    arguments are known. render_html is intentionally excluded: a networked
    canvas needs approval, which cannot be judged before its arguments stream."""
    return name in _ALWAYS_SAFE_TOOLS


# Tools whose provisional card is only a text preview of the arguments, so it can stream
# while awaiting approval.
_TEXT_PREVIEW_TOOLS = frozenset({"python", "terminal", "edit_file"})


def has_text_only_provisional_card(name: str) -> bool:
    """True when streaming this tool's arguments before approval shows only text.

    A large code payload takes a minute or more to write, and suppressing the
    card until the call completes leaves the chat blank the whole time. Nothing
    runs before the decision either way, and you have to read the code to make
    it.
    """
    return name in _TEXT_PREVIEW_TOOLS


def _web_search_fetches_url(name: str, arguments: dict) -> bool:
    """web_search carrying a ``url`` fetches that exact page instead of searching.

    That fetch is egress to a host the *call* names, the one such case in the built-in
    set, so it asks even though plain search stays always-safe. Name-only
    ``is_always_safe_tool`` is deliberately unchanged: it runs before arguments exist
    (provisional card, stream requirement), where a query-only search must not prompt."""
    return name == "web_search" and bool(str(arguments.get("url", "") or "").strip())


def is_potentially_unsafe_tool_call(name: str, arguments: dict) -> bool:
    """Whether a tool call must still pause for approval in auto mode.

    Used by permission_mode="auto" ("Approve for me"): read-only calls
    auto-run, anything that can mutate state, execute arbitrary code, or is
    simply unrecognized asks first. Unknown tools fail closed.
    """
    if _web_search_fetches_url(name, arguments):
        return True
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
    # Always writes, and python's open(..., "w") already prompts, so the cheaper
    # tool must not become the quiet way around that. Stated rather than left to
    # the fail-closed default, so a later clause cannot drop it.
    if name == "edit_file":
        return True
    return True


# Terminal commands that are high risk regardless of their arguments, so auto
# ("Approve for me") pauses them while ordinary dev commands (pip install, mkdir,
# cp, make, git, ...) run. The hard-block command set, rlimits, secret-env
# stripping and the per-session scratch workdir stay on beneath this prompt.
_HIGH_RISK_COMMANDS = frozenset(
    {
        # privilege escalation
        "sudo",
        "su",
        "doas",
        "pkexec",
        # destructive filesystem / storage devices (mkfs* matched by prefix)
        "rm",
        "rmdir",
        "shred",
        "dd",
        "wipefs",
        "fdisk",
        "parted",
        "blkdiscard",
        "chattr",
        "truncate",
        # Windows cmd.exe built-ins that delete files / trees (reachable when the
        # terminal executor falls back to `cmd /c`; not in _BLOCKED_COMMANDS_WIN)
        "del",
        "erase",
        "rd",
        # Ending a process kills work in progress (a training run, the server
        # itself); a power command ends every process at once.
        "kill",
        "pkill",
        "killall",
        "taskkill",
        "tskill",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        # setcap grants file capabilities, a privilege change without sudo.
        "setcap",
        # accounts / persistence / system services
        "crontab",
        # at/batch hand the payload to atd, which runs it later as this user and
        # outside this invocation's blocklist, rlimits, timeout and cancellation.
        "at",
        "batch",
        "atrm",
        "systemctl",
        "service",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "groupdel",
        "groupmod",
        "adduser",
        "deluser",
        "addgroup",
        "delgroup",
        "gpasswd",
        "newusers",
        "chgpasswd",
        "passwd",
        "chpasswd",
        "visudo",
        "chsh",
        # firewall / mounts
        "iptables",
        "ip6tables",
        "nft",
        "ufw",
        "mount",
        "umount",
        # remote exec / raw network transfer
        "ssh",
        "slogin",
        "scp",
        "sftp",
        "telnet",
        "nc",
        "ncat",
        "netcat",
        "socat",
        "ftp",
        "tftp",
        # POSIX unlink(1) deletes a file exactly like rm, which is gated above.
        "unlink",
        # Windows / macOS storage destruction, the platform twins of the POSIX
        # mkfs/wipefs/dd family already gated above.
        "format",
        "diskpart",
        "diskutil",
        # Windows / macOS scheduled tasks, registry and service control: the twins
        # of crontab/systemctl. Gated wholesale (a read-only `reg query` prompts
        # too) because the destructive subcommand lives in the arguments.
        "systemd-run",
        "schtasks",
        "reg",
        "sc",
        "launchctl",
        # container/VM runtimes: the daemon acts with host privileges, so
        # `docker run -v /:/host ...` writes the real filesystem, escaping the
        # child's workdir and rlimit sandbox entirely. chroot/nsenter/unshare
        # cross a privilege or namespace boundary and then exec a nested command,
        # so the wrapper hides the real action.
        "chroot",
        "nsenter",
        "unshare",
        "docker",
        "podman",
        "nerdctl",
        "ctr",
        "crictl",
        "lxc",
        "machinectl",
        "kubectl",
    }
)
# sysctl's write and load forms change kernel parameters; a read-only query
# (sysctl -a, sysctl net.ipv4.ip_forward) stays automatic.
_SYSCTL_WRITE_FLAGS = frozenset({"-w", "--write", "-p", "--load", "--system"})
# setpriv changes privilege state and then execs its remaining arguments, so the
# real command sits behind it. Kept out of _AUTO_SAFE_WRAPPERS (it is not safe in
# its own right) and instead made transparent only for the high-risk scan, where
# the flags that raise privilege are gated on their own.
_PRIVILEGE_EXEC_WRAPPERS = frozenset({"setpriv"})
_SETPRIV_PRIVILEGE_FLAGS = frozenset(
    {
        "--reuid",
        "--regid",
        "--ruid",
        "--euid",
        "--rgid",
        "--egid",
        "--groups",
        "--init-groups",
        "--inh-caps",
        "--ambient-caps",
        "--bounding-set",
        "--securebits",
        "--selinux-label",
        "--apparmor-profile",
    }
)
# fallocate replaces a range with a hole, zeroes it or removes it, destroying
# file contents in place. Plain allocation (-l SIZE) only grows a file.
_FALLOCATE_DESTRUCTIVE_FLAGS = frozenset(
    {"-p", "--punch-hole", "-z", "--zero-range", "-c", "--collapse-range", "-d", "--dig-holes"}
)
# High risk only with a recursive flag (chmod -R 777 .); a scoped
# `chmod +x build.sh` stays out.
_HIGH_RISK_RECURSIVE_COMMANDS = frozenset({"chmod", "chown", "chgrp"})
# Commands that forward command position to a following command name
# (find . -exec rm, echo x | xargs rm, parallel rm, watch rm), so the wrapped
# command is checked against the high-risk sets too.
_HIGH_RISK_FORWARDING_COMMANDS = frozenset(
    {
        "find",
        "fd",
        "xargs",
        "parallel",
        "watch",
        "strace",
        "ltrace",
        "ktrace",
        "dtruss",
        "perf",
        "valgrind",
    }
)
# Of those, find/fd only execute a child after an explicit -exec-style flag.
# A tracer or profiler runs the rest of the line as a child process, so the
# real command sits in argument position behind it.
_TRACER_LAUNCHERS = frozenset({"strace", "ltrace", "ktrace", "dtruss", "perf", "valgrind"})
_EXEC_FLAG_FORWARDING_COMMANDS = frozenset({"find", "fd"})
_EXEC_FORWARD_FLAGS = frozenset(
    {"-exec", "-execdir", "-ok", "-okdir", "--exec", "--exec-batch", "-x", "-X"}
)
# The long forms also accept the command attached to the flag (fd --exec=rm),
# where the value is command position rather than a discarded option argument.
_ATTACHED_EXEC_FLAGS = frozenset({"-exec", "-execdir", "--exec", "--exec-batch"})
# find/fd flags that delete matches outright (a bare `find . -delete`, with no
# separate command token to catch); an `-exec rm` is caught via forwarding.
_HIGH_RISK_FIND_FLAGS = frozenset({"-delete"})
# Flags whose VALUE is a command the tool then executes, so a payload (even a
# hard-blocked one) rides inside an argument instead of at command position.
# GNU tar --checkpoint-action=exec=CMD, rsync/scp -e REMOTE_SHELL.
_HIGH_RISK_ARG_EXEC_FLAGS = frozenset({"--checkpoint-action", "--rsh", "--rsync-path"})
# ...but only for the utilities that actually run them; otherwise a mere
# mention (printf '%s' --rsh, a grep for the flag name) would prompt.
_ARG_EXEC_FLAG_OWNERS = frozenset({"tar", "gtar", "bsdtar", "rsync", "scp", "sftp"})
# An interpreter run as a network server (python -m http.server, uvicorn app:api)
# listens on a socket; the sandbox has no network namespace, so the session
# workdir becomes reachable wherever that port is exposed. Position-scoped, since
# a bare mention (pip install uvicorn, grep uvicorn reqs.txt) starts no listener.
_LISTENER_PY_MODULES = (
    r"http\.server|SimpleHTTPServer|uvicorn|gunicorn|waitress|flask|"
    r"twisted|websockets|aiohttp\.web"
)
_LISTENER_PY_MODULE_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:\S*/)?"
    r"(?:python|pypy)[0-9.]*\s+(?:-\S+\s+)*-m\s+(?:" + _LISTENER_PY_MODULES + r")\b",
    re.IGNORECASE,
)
# The same modules as the command-position regex, matched after wrapper
# resolution so `env python -m http.server` and `timeout 60 python -m ...`
# are seen too.
_LISTENER_PY_MODULE_NAMES = frozenset(
    {
        "http.server",
        "simplehttpserver",
        "uvicorn",
        "gunicorn",
        "waitress",
        "flask",
        "twisted",
        "websockets",
        "aiohttp.web",
    }
)
_LISTENER_BINARIES = frozenset({"uvicorn", "gunicorn", "waitress-serve", "hypercorn", "daphne"})
_LISTENER_BIN_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*"
    r"(?:uvicorn|gunicorn|waitress-serve|hypercorn|daphne)\b"
)
# curl upload/POST flags: local data sent out (exfiltration surface). The short
# forms may be attached (-d@f, -Ffile=@dump.sql), so they match prefix-wise.
_CURL_UPLOAD_LONG_FLAGS = frozenset(
    {
        "--data",
        "--data-ascii",
        "--data-binary",
        "--data-raw",
        "--data-urlencode",
        "--form",
        "--upload-file",
    }
)
_CURL_UPLOAD_SHORT_FLAGS = ("-d", "-F", "-T")
# curl's explicit-method flags and the methods that mutate/delete a remote
# resource (a plain GET download stays out). POST is omitted: it is the ordinary
# upload verb and is already caught by the body/upload flags above.
# wget spells the request method --method=DELETE.
_WGET_METHOD_FLAGS = frozenset({"--method"})
_CURL_METHOD_FLAGS = frozenset({"-X", "--request"})
_CURL_DESTRUCTIVE_METHODS = frozenset({"delete", "put", "patch"})
# wget upload/POST flags. Kept separate from curl's so a benign wget short option
# (wget -T 10 timeout, wget -F force-html) is not misread as an upload.
_WGET_UPLOAD_FLAGS = frozenset({"--post-data", "--post-file", "--body-data", "--body-file"})
# curl/wget output piped straight into an interpreter is remote code execution.
_PIPE_TO_INTERPRETER_RE = re.compile(
    r"\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh|fish|python[0-9.]*|node|ruby|perl|php)\b"
)
_BARE_TRUNCATING_REDIRECT_RE = re.compile(r"(?:^|[;&|\n(]|&&|\|\|)\s*(?::|true)?\s*>(?!>)\s*\S")
_HERESTRING_TO_INTERPRETER_RE = re.compile(
    r"\b(?:sh|bash|zsh|dash|ksh|fish|ash|python[0-9.]*|node|ruby|perl|php)\b[^\n]*<<<"
)
# An interpreter that executes a process substitution's output as a script
# (bash <(printf 'rm -rf x'), source <(...)): the generated content is never
# literal text, so it is unscreenable and fails closed. A non-interpreter consumer
# (diff <(sort a) <(sort b)) only reads the file and stays out.
_PROC_SUBST_EXEC_RE = re.compile(
    r"\b(?:sh|bash|zsh|dash|ksh|fish|ash|source|eval|python[0-9.]*|node|nodejs|bun|ruby|perl|php)\b"
    r"[^\n]*<\("
    r"|(?:^|[;&|\n(]|&&|\|\|)\s*\.\s+<\("
)
# Network clients beyond curl/wget that open a socket to a remote host: the
# sandbox has no network namespace, so they can exfil the workdir or fetch and run
# remote code. Command position only, so a filename argument (scp ./ssh_notes.txt)
# is not misread as the command.
_NETWORK_CLIENT_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*"
    r"(?:nc|ncat|netcat|telnet|socat|ssh|slogin|scp|sftp)\b"
)
# openssl's s_client/s_server open a TLS socket, the classic no-curl exfil channel
# (tar czf - . | openssl s_client -connect host:443). Plain openssl (dgst, enc) is
# local and stays out. Matched on the resolved command segment, so the wrapped
# forms (env openssl s_client) are seen too.
_OPENSSL_NETWORK_SUBCOMMANDS = frozenset({"s_client", "s_server"})
# `getent shadow` returns password hashes straight from NSS, so the read
# never spells out /etc/shadow for the path check to find.
_GETENT_CREDENTIAL_DATABASES = frozenset({"shadow", "gshadow"})
_OPENSSL_NETWORK_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:\S*/)?openssl\s+s_(?:client|server)\b"
)
# An array expansion (${x[*]}, ${x[@]}) builds a command from elements the static
# scan cannot resolve; fed to a shell -c/eval it runs an unscreened payload.
# Paired with the var-executed-as-command test so `echo "${a[@]}"` is left alone.
_ARRAY_EXPANSION_RE = re.compile(r"\$\{\w+\[[@*]\]\}")
# A wrapper's bare duration/count argument (timeout 5 rm, timeout 1.5s rm) that
# precedes the real command, so it is not mistaken for the command itself.
_WRAPPER_DURATION_RE = re.compile(r"\d+(?:\.\d+)?[smhd]?$")
# Non-shell interpreters running an inline program (python -c, node -e, php -r):
# the terminal path never screens that program the way the python tool does.
# sh/bash -c are omitted, the hard-block already recurses into their payloads.
_INLINE_CODE_INTERPRETERS = frozenset(
    {
        "python",
        "python2",
        "python3",
        "pypy",
        "pypy3",
        "node",
        "nodejs",
        "deno",
        "bun",
        "ruby",
        "perl",
        "php",
    }
)
_INLINE_CODE_FLAGS = frozenset({"-c", "-e", "-E", "-r", "--eval", "--exec"})
# Inline-code flags are per-interpreter: a flag that evaluates code for one runtime
# is an ordinary option for another (`python -E` ignores PYTHON* env, it is not
# eval). Value is (exact flags, short letters that may appear in a cluster).
_INLINE_CODE_FLAG_SPEC = {
    "python": (frozenset({"-c"}), "c"),
    "pypy": (frozenset({"-c"}), "c"),
    "node": (frozenset({"-e", "--eval"}), "e"),
    "nodejs": (frozenset({"-e", "--eval"}), "e"),
    "deno": (frozenset({"-e", "--eval"}), "e"),
    "bun": (frozenset({"-e", "--eval"}), "e"),
    "ruby": (frozenset({"-e"}), "e"),
    # perl -e and -E both run a one-liner (-E also enables feature bundles).
    "perl": (frozenset({"-e", "-E"}), "eE"),
    # php -r runs code; -B / -R / -E run begin / per-line / end code.
    "php": (frozenset({"-r", "-B", "-R", "-E"}), "rBRE"),
}


def _inline_code_flag_spec(name: str):
    """(exact flags, short-cluster letters) that make `name` run inline code."""
    base = name
    if _VERSIONED_INTERPRETER_RE.match(base):
        base = re.sub(r"\d+(?:\.\d+)*$", "", base)
    else:
        base = re.sub(r"^(python|pypy)[23]$", r"\1", base)
    return _INLINE_CODE_FLAG_SPEC.get(base)


# node/bun evaluate and print the argument to -p / --print, arbitrary code just
# like -e/--eval. Scoped to the JS runtimes: -p is a print-loop switch for
# perl/ruby/sed, not inline eval.
_NODE_PRINT_INTERPRETERS = frozenset({"node", "nodejs", "bun"})
# Runtimes that expose inline evaluation as a SUBCOMMAND (deno eval "...",
# bun eval "..."), which the flag scan above never sees.
_EVAL_SUBCOMMAND_INTERPRETERS = frozenset({"deno", "bun"})
_NODE_PRINT_FLAGS = frozenset({"-p", "--print"})
# Windows cmd.exe runs the rest of the line as a nested command after /c (or /k),
# so the payload is screened recursively like a shell -c payload. cmd is not in
# the hard-block set, and del/erase/rd were added to the high-risk set for it.
_CMD_SHELLS = frozenset({"cmd"})
# PowerShell runs an arbitrary inline program passed to -Command /
# -EncodedCommand (and their unambiguous prefixes), which the terminal path cannot
# parse. On Windows both names are hard-blocked; elsewhere pwsh is not, so gate an
# inline-command invocation there. A bare `pwsh script.ps1` file run stays out.
_POWERSHELL_INTERPRETERS = frozenset({"powershell", "pwsh"})
# Versioned interpreter binaries (python3.11, python2.7, pypy3.10) are the same
# inline-code risk as their unversioned names, so recognise the version suffix.
_VERSIONED_INTERPRETER_RE = re.compile(r"^(?:python|pypy|perl|ruby|php|node)\d+(?:\.\d+)*$")
# busybox / toybox dispatch to an applet given as the first argument, so the
# applet, not the multicall binary, is the command whose risk is judged.
_MULTICALL_BINARIES = frozenset({"busybox", "toybox"})
# `cd /proc/$PPID; cat environ` reads a sensitive path after the chdir even though
# no single token spells it out, so a chdir into a sensitive dir is gated.
_CHDIR_COMMANDS = frozenset({"cd", "pushd", "chdir"})
# The absolute system dirs are anchored so an unrelated user dir (/home/x/etc)
# does not match; the credential dotfile dirs match anywhere in the path.
_SENSITIVE_CHDIR_RE = re.compile(
    r"^~?/proc/[^/\s'\"]+"
    r"|^~?/etc(?:/|$)"
    r"|^~?/root(?:/|$)"
    r"|^~?/(?:var/)?run/secrets(?:/|$)"
    r"|(?:^|[/\\])\.(?:ssh|aws|azure|gnupg|docker|kube)(?:[/\\]|$)"
    r"|(?:^|[/\\])\.config[/\\](?:gcloud|gh)(?:[/\\]|$)",
    re.IGNORECASE,
)


def _is_inline_code_interpreter(name: str) -> bool:
    """True for an interpreter whose ``-c`` / ``-e`` runs an inline program the
    terminal path never screens, including versioned python/pypy binaries."""
    return name in _INLINE_CODE_INTERPRETERS or bool(_VERSIONED_INTERPRETER_RE.match(name))


def _short_flag_cluster(token: str) -> "list[str]":
    """Split a combined short-option token into its individual flags
    (`-qf` -> ['-q', '-f']). A long option, a `-x=value` form or a bare `-`
    yields nothing, so only genuine clusters are expanded."""
    if len(token) < 3 or not token.startswith("-") or token.startswith("--") or "=" in token:
        return []
    return ["-" + ch for ch in token[1:]]


def _short_flag_arg(token: str, letters: str) -> "str | None":
    """For a short-flag cluster (``-lc``, ``-Bc``, ``-c``), if one of ``letters``
    appears as a flag in it, return the text glued after that letter -- ``""`` when
    the value is the next token, or the attached payload for ``-c'cmd'``. ``None``
    when no such flag is present, or for long options / non-flags. Catches combined
    forms (``bash -lc 'git clean'``) an exact ``-c`` match would miss."""
    if not token.startswith("-") or token.startswith("--"):
        return None
    body = token[1:]
    for i, ch in enumerate(body):
        if ch in letters:
            return body[i + 1 :]
    return None


def _shell_quote_states(command: str) -> "list[str]":
    """The quote context of every character: ``""`` outside quoting, ``"'"``
    (or ``"$'"`` for ANSI-C, which honours backslash escapes) inside single
    quoting, ``'"'`` inside double quoting, and ``_ESCAPED_CHAR_STATE`` for a
    backslash and the character it quotes. A quote mark itself reports the
    context it opens from, so a character is text bash expands exactly when its
    state is ``""`` or ``'"'``.

    Tracked character by character rather than paired off with a regex, because
    a regex matches the apostrophe in `echo "it's"` against the next quote,
    inverting the state for everything after it.
    """
    states: "list[str]" = []
    quote = ""
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote in ("'", "$'"):
            # A plain single quote protects even backslashes; ANSI-C does not,
            # so `\'` there is a quote character rather than the end of the word.
            if quote == "$'" and ch == "\\" and i + 1 < n:
                states += [quote, quote]
                i += 2
                continue
            states.append(quote)
            if ch == "'":
                quote = ""
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            # Reported under its OWN state rather than the surrounding one:
            # marking `\$` as ordinary double-quoted text made `$(` there look
            # like a live substitution, so an everyday `sed "s/\$(CC)/gcc/"
            # Makefile` asked for confirmation while real bash hands sed a
            # literal `$(CC)` and nothing runs (verified: it prints CC=cc).
            states += [_ESCAPED_CHAR_STATE, _ESCAPED_CHAR_STATE]
            i += 2
            continue
        states.append(quote)
        if quote == '"':
            # Only the closing quote ends it; an apostrophe here is text.
            if ch == '"':
                quote = ""
        elif ch == "'":
            quote = "$'" if i and command[i - 1] == "$" else "'"
        elif ch == '"':
            quote = '"'
        i += 1
    return states


def _substitution_span(command: str, start: int) -> int:
    """Index just past the `)` that closes the `$(` at ``start``.

    The body of a substitution is a FRESH shell context -- bash re-parses it, so
    quoting reopens inside even when the whole thing sits in double quotes --
    and a paren the body QUOTES is text, not nesting. Counting it raised the
    depth, the real `)` then never brought the depth back to zero, and the span
    ran on past the end of the word: `sed "$(printf '(' >/dev/null; printf 'e
    rm -f victim')" input` yielded a span with ` input` glued on, which no
    longer matched the sed program it had to be found inside, so the generated
    script went unnoticed.

    _shell_quote_states is a left-to-right machine, so the states it reports for
    a prefix are the ones it reports for the whole string; the window is grown
    until the span closes, which keeps the cost a constant multiple of the
    substitution's own length rather than a walk to the end of the line for
    every one of them.
    """
    n = len(command)
    width = _SUBSTITUTION_SPAN_STEP
    while True:
        stop = min(n, start + 1 + width)
        body = command[start + 1 : stop]
        depth = 0
        for offset, state in enumerate(_shell_quote_states(body)):
            if state:
                continue  # quoted: data to the nested shell, not a delimiter
            char = body[offset]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return start + 2 + offset
        if stop >= n:
            return n
        width *= 4


def _arithmetic_span(command: str, start: int) -> int:
    """Index just past the `))` / `]` closing the arithmetic expansion at
    ``start`` -- `$((...))`, or the deprecated `$[...]` bash 5.2 still
    evaluates (`echo $[1+2]` prints 3)."""
    opener = command[start + 1]
    closer = ")" if opener == "(" else "]"
    depth, i, n = 0, start + 1, len(command)
    while i < n:
        if command[i] == opener:
            depth += 1
        elif command[i] == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _brace_param_span(command: str, start: int) -> int:
    """Index just past the `}` closing the `${` at ``start``. Braces nest
    (`${a:-${b}}`) and a backslash quotes the one behind it."""
    depth, i, n = 0, start + 1, len(command)
    while i < n:
        if command[i] == "\\":
            i += 2
            continue
        if command[i] == "{":
            depth += 1
        elif command[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _collapse_shell_arithmetic(program: str) -> str:
    """``program`` with each arithmetic expansion replaced by a digit
    (_ARITHMETIC_VALUE), which is a faithful stand-in because arithmetic always
    evaluates to an integer.

    Without it the expansion's own punctuation is read as sed source and hides
    the command behind it: `sed "$((c+1))e rm -f victim"` runs rm for real
    (`$((c+1))` is 1), while the raw text takes the `c` for an append-text
    command and swallows the payload as its operand. An expansion holding a
    COMMAND substitution is left alone, so the substitution stays visible to
    _sed_program_unresolved rather than being collapsed out of sight.
    """
    out: "list[str]" = []
    i, n = 0, len(program)
    while i < n:
        if program.startswith("$((", i) or program.startswith("$[", i):
            end = _arithmetic_span(program, i)
            if not _HAS_COMMAND_SUBST_RE.search(program[i:end]):
                out.append(_ARITHMETIC_VALUE)
                i = end
                continue
        out.append(program[i])
        i += 1
    return "".join(out)


def _shell_expansions(command: str, quoted: bool = True) -> "list[str]":
    """Every expansion bash performs, as the exact text each one occupies:
    `$(...)`, backticks, `${...}` in ANY form and a bare `$NAME` / `$?`.

    With ``quoted`` (the default) the text is a whole command line, so a
    single-quoted or backslash-escaped expansion is literal and reported as
    nothing -- ``sed 's/`//g' NOTES.md`` and `sed "s/\\$(CC)/gcc/" Makefile`
    both yield an empty list. With ``quoted`` False the text is a token shlex
    has already unquoted, where every character counts; comparing the two tells
    an expansion the shell RUNS from one a sed program merely quotes.

    ARITHMETIC is skipped: it evaluates to an integer, so it can spell no sed
    command (_ARITHMETIC_VALUE). One holding a command substitution is stepped
    INTO instead, so the substitution inside `sed "$(( $(cat n) ))p"` is still
    reported.
    """
    found: "list[str]" = []
    states = _shell_quote_states(command) if quoted else None
    i, n = 0, len(command)
    while i < n:
        if states is not None and states[i] not in ("", '"'):
            i += 1
            continue
        if command[i] == "`":
            end = command.find("`", i + 1)
            end = n if end < 0 else end + 1
            found.append(command[i:end])
            i = end
            continue
        if command.startswith("$((", i) or command.startswith("$[", i):
            end = _arithmetic_span(command, i)
            # Stepping over the `$` alone would report the arithmetic's own
            # `(name)` as a substitution; stepping over the whole span would
            # hide a `$(...)` nested inside it. Do each where it applies.
            i = i + 2 if _HAS_COMMAND_SUBST_RE.search(command[i:end]) else end
            continue
        if command.startswith("$(", i):
            end = _substitution_span(command, i)
            found.append(command[i:end])
            i = end
            continue
        if command.startswith("${", i):
            end = _brace_param_span(command, i)
            found.append(command[i:end])
            i = end
            continue
        match = _UNBRACED_PARAM_RE.match(command, i)
        if match:
            found.append(match.group(0))
            i = match.end()
            continue
        i += 1
    return found


def _separate_unquoted_newlines(text: str) -> str:
    """``text`` with each UNQUOTED newline replaced by `;`, which shlex reads as
    a command boundary. A newline inside quotes is DATA -- a sed comment ends at
    one -- so it survives, unlike a blanket replacement. A BACKSLASH-escaped
    newline is a line continuation bash deletes rather than a separator, so it
    survives too; the blanket pass still supplies that boundary if one is
    wanted, since it replaces every newline unconditionally."""
    states = _shell_quote_states(text)
    out = []
    for i, ch in enumerate(text):
        if ch in "\r\n" and states[i] == "":
            # \r\n is one boundary, not two.
            if not (ch == "\n" and i and text[i - 1] == "\r"):
                out.append(";")
        else:
            out.append(ch)
    return "".join(out)


# git subcommands that discard or overwrite work: `clean` deletes untracked files,
# `restore` overwrites the worktree from the index/HEAD, `rm` deletes tracked
# files, and the plumbing entries delete refs/reflogs/objects or rewrite history.
# `reset`/`push`/`checkout` only qualify with a destructive flag or pathspec, so
# `git reset --soft`, a plain `git push` and ordinary git (add/commit/log) run.
_HIGH_RISK_GIT_SUBCOMMANDS = frozenset(
    {"clean", "restore", "rm", "update-ref", "filter-branch", "prune", "gc", "reflog"}
)
_HIGH_RISK_GIT_RESET_FLAGS = frozenset({"--hard"})
_HIGH_RISK_GIT_PUSH_FLAGS = frozenset(
    # --delete/-d removes a remote ref; --mirror and --prune delete remote refs
    # that are absent locally. All are remote data loss, like a force push.
    {"-f", "--force", "--force-with-lease", "-d", "--delete", "--mirror", "--prune"}
)
# `git worktree remove --force` deletes a linked worktree even when it holds
# uncommitted work or is locked. An unforced remove refuses on a dirty worktree,
# so it stays out.
_HIGH_RISK_GIT_WORKTREE_FLAGS = frozenset({"-f", "--force"})
# `git switch -f/--discard-changes` throws away tracked working-tree edits.
_HIGH_RISK_GIT_SWITCH_FLAGS = frozenset({"-C", "-f", "--force", "--discard-changes"})
# `git branch -D` force-deletes a branch, discarding unmerged commits; -M
# force-renames over an existing branch. Plain -d/--delete refuses to drop
# unmerged work, so it stays out.
_HIGH_RISK_GIT_BRANCH_FLAGS = frozenset({"-D", "-M", "-f", "--force"})
# `git stash clear` / `drop` destroy stashed work with no reflog to recover it.
_HIGH_RISK_GIT_STASH_ACTIONS = frozenset({"clear", "drop"})
# `git checkout -- <path>` / `git checkout .` / `git checkout -f` discard tracked
# working-tree changes; a bare `git checkout <branch>` (switching) does not.
_HIGH_RISK_GIT_CHECKOUT_FLAGS = frozenset({"-f", "--force", "-B"})
# `git checkout-index -f` overwrites working-tree files from the index.
_HIGH_RISK_GIT_CHECKOUT_INDEX_FLAGS = frozenset({"-f", "--force"})
# `git tag -d` deletes a ref; `git tag -f` replaces one that already exists.
_HIGH_RISK_GIT_TAG_FLAGS = frozenset({"-d", "--delete", "-f", "--force"})
# `git -c alias.NAME=PAYLOAD` defines an alias git then runs; a leading `!` makes
# the payload a shell command.
_GIT_ALIAS_ASSIGN_RE = re.compile(r"^alias\.[^=]+=(.*)$", re.DOTALL)
# `git --config-env=alias.n=VAR n` names an environment variable whose value
# becomes the alias body, so the code is never present in the command text.
_GIT_CONFIG_ENV_ALIAS_RE = re.compile(r"(?:^|=)alias\.", re.IGNORECASE)
# git global options taking a separate value token (git -C repo clean); the value
# must be consumed so it is not mistaken for the subcommand.
_GIT_GLOBAL_VALUE_FLAGS = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env"}
)
# Shells whose `-c PAYLOAD` runs an inline program: the payload is recursively
# screened, so a high-risk command wrapped in `bash -c '...'` is still caught. The
# hard-block only recurses for its own smaller command set.
_SHELL_C_INTERPRETERS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "ash"})
# A command synthesized by a command substitution at command position
# ($(printf rm) -rf build) cannot be read statically. A substitution in argument
# position (echo $(date), make $(FILES)) is left alone.
_COMMAND_SUBST_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=[^\s;&|()]*\s+)*(?:\$\(|`)"
)

# A command substitution appearing anywhere ($(...) that is not arithmetic
# $((...)), or a backtick). Used to catch a substitution stashed in a variable
# (x=`...`) that a later dynamic exec runs, which never surfaces as literal text.
_HAS_COMMAND_SUBST_RE = re.compile(r"\$\((?!\()|`")
# The same as below, but only when the expansion is the WHOLE command word. A
# variable used as a path prefix (${VENV}/bin/python) still leaves a literal
# basename the scan can screen, so it is not unresolvable.
_BARE_VAR_AS_COMMAND_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*\$\{?\w+\}?(?=\s|$)"
)
# A variable expansion executed as a command: $VAR at command position, or a shell
# `-c` / eval whose payload contains a `$` expansion. Paired with
# _HAS_COMMAND_SUBST_RE this flags `x=`printf 'git clean -fd'`; bash -c "$x"`,
# assembled at runtime and so unscreenable statically.
_VAR_EXECUTED_AS_COMMAND_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*\$\{?\w"
    r"|\b(?:sh|bash|zsh|dash|ksh|ash)\b[^\n]*?\s-c\b[^\n]*\$"
    r"|\beval\b[^\n]*\$"
)


_SHELL_SEGMENT_SPLIT_RE = re.compile(r"^(?:;|&&|\|\||\||&)$")


# Wrappers that may sit in front of a network client without changing what it
# does, so the client is still at command position behind them.
_CLIENT_WRAPPERS = frozenset(
    {"env", "command", "timeout", "nohup", "nice", "ionice", "stdbuf", "setsid", "exec"}
)
_CLIENT_WRAPPER_PREFIX = (
    r"(?:(?:env|command|timeout|nohup|nice|ionice|stdbuf|setsid|exec)\s+"
    r"(?:-\S+\s+|\d+(?:\.\d+)?[smhd]?\s+)*)*"
)
# The terminal sandbox shares the backend's installed environment, so removing
# a package (pip uninstall torch) breaks the running process. Installing does
# not, and is ordinary work, so only the removal verbs are gated.
_PKG_REMOVE_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:\S*/)?"
    r"(?:(?:python[0-9.]*\s+-m\s+)?pip[0-9]*|uv\s+pip|pipx|conda|mamba|micromamba)"
    r"\s+(?:uninstall|remove)\b",
    re.IGNORECASE,
)
_CURL_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*"
    + _CLIENT_WRAPPER_PREFIX
    + r"(?:\S*/)?curl\b",
    re.IGNORECASE,
)
_WGET_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*"
    + _CLIENT_WRAPPER_PREFIX
    + r"(?:\S*/)?wget\b",
    re.IGNORECASE,
)


def _tokens_for_client_segment(tokens: list, has_curl: bool, has_wget: bool):
    """Tokens of the segments whose command is curl/wget, or None if there is no
    such segment. Keeps an unrelated command's option letters out of the upload
    scan (`ls -T && echo curl`)."""
    segments: list = []
    current: list = []
    for t in tokens:
        if _SHELL_SEGMENT_SPLIT_RE.match(t):
            segments.append(current)
            current = []
        else:
            current.append(t)
    segments.append(current)
    kept: list = []
    for seg in segments:
        # Skip leading NAME=value prefixes to find the command word.
        i = 0
        while i < len(seg) and re.match(r"^[A-Za-z_]\w*=", seg[i]):
            i += 1
        if i >= len(seg):
            continue
        # Step past a wrapper (env curl, timeout 5 curl) to the real client.
        while i < len(seg):
            base = os.path.basename(seg[i].strip(";&|()`{}")).lower()
            if base not in _CLIENT_WRAPPERS:
                break
            i += 1
            while i < len(seg) and (seg[i].startswith("-") or _WRAPPER_DURATION_RE.match(seg[i])):
                i += 1
        if i >= len(seg):
            continue
        base = os.path.basename(seg[i].strip(";&|()`{}")).lower()
        if (has_curl and base == "curl") or (has_wget and base == "wget"):
            kept.extend(seg[i:])
    return kept or None


def _command_is_network_exec_or_exfil(command: str) -> bool:
    """curl/wget used to run remote code (piped into a shell, or via process
    substitution) or to upload local data. Plain downloads (curl -O, wget URL)
    are ordinary and stay out. Fails closed on an unparseable command."""
    low = command.lower()
    # A non-curl/wget client (nc/ssh/socat) or openssl's TLS socket is a remote
    # reach in its own right, so gate it before the upload-flag logic below.
    if _NETWORK_CLIENT_AT_CMD_RE.search(command) or _OPENSSL_NETWORK_RE.search(low):
        return True
    # A mention in argument position (`grep curl notes.txt`) is not an invocation,
    # and treating it as one lends another command's option letters to the scan.
    has_curl = bool(_CURL_AT_CMD_RE.search(command))
    has_wget = bool(_WGET_AT_CMD_RE.search(command))
    if not has_curl and not has_wget:
        return False
    if _PIPE_TO_INTERPRETER_RE.search(low):
        return True
    if "<(" in command:  # bash <(curl ...) process substitution
        return True
    try:
        tokens = shlex.split(command.replace("\n", " "), posix = True)
    except ValueError:
        return True
    # Scope the flag scan to the segment that actually runs curl/wget: a shared
    # option letter from an unrelated command (`ls -T && echo curl`) is not an
    # upload flag.
    tokens = _tokens_for_client_segment(tokens, has_curl, has_wget)
    if tokens is None:
        return False
    method_pending = False
    for t in tokens:
        name = t.split("=", 1)[0]
        # curl -X DELETE / --request PUT mutates a remote resource, not a plain
        # download. Separated, attached (-XDELETE) and --request=DELETE forms.
        if has_curl:
            if method_pending:
                method_pending = False
                if t.lower() in _CURL_DESTRUCTIVE_METHODS:
                    return True
            if name in _CURL_METHOD_FLAGS:
                if "=" in t and t.split("=", 1)[1].lower() in _CURL_DESTRUCTIVE_METHODS:
                    return True
                method_pending = True
                continue
            if t.startswith("-X") and t[2:].lower() in _CURL_DESTRUCTIVE_METHODS:
                return True
        if has_wget:
            # wget --method=DELETE / --method DELETE is the same remote mutation.
            if method_pending:
                method_pending = False
                if t.lower() in _CURL_DESTRUCTIVE_METHODS:
                    return True
            if name in _WGET_METHOD_FLAGS:
                if "=" in t and t.split("=", 1)[1].lower() in _CURL_DESTRUCTIVE_METHODS:
                    return True
                method_pending = True
                continue
        if has_curl and (
            name in _CURL_UPLOAD_LONG_FLAGS
            # a curl short upload flag, attached or not (-d@f, -Ffile=@dump.sql)
            or (not name.startswith("--") and name.startswith(_CURL_UPLOAD_SHORT_FLAGS))
        ):
            return True
        if has_wget and name in _WGET_UPLOAD_FLAGS:
            return True
    return False


# `git clean -n` / `--dry-run` only lists what would be removed.
_GIT_CLEAN_DRY_RUN_FLAGS = frozenset({"-n", "--dry-run"})


def _container_subcommand_is_read_only(tokens: list, start: int) -> bool:
    """Whether a container CLI's first positional is a read subcommand. A bare
    `docker` or `docker --version` prints help and runs nothing."""
    for t in tokens[start + 1 :]:
        if t in _SHELL_SEPARATORS or not set(t) - set(";&|()"):
            break
        if t.startswith("-"):
            continue
        return t.lower() in _CONTAINER_READ_SUBCOMMANDS
    return True


def _segment_has_command_after(tokens: list, start: int) -> bool:
    """Whether a command word follows an assignment in the same segment. A bare
    `export PATH=...` or `FOO=bar` runs nothing: every terminal call gets its own
    shell process, so an assignment with no command dies with it."""
    for t in tokens[start + 1 :]:
        if t in _SHELL_SEPARATORS or not set(t) - set(";&|()"):
            return False
        if _ASSIGNMENT_RE.match(t) or t.startswith("-"):
            continue
        return True
    return False


def _segment_has_flag(
    tokens: list,
    start: int,
    exact: frozenset,
    letters: str = "",
) -> bool:
    """Whether a flag appears in the same command segment as ``start``, so a
    later command's options are not read as this command's."""
    for t in tokens[start + 1 :]:
        if t in _SHELL_SEPARATORS or not set(t) - set(";&|()"):
            break
        if t in exact:
            return True
        if letters and t[:1] == "-" and t[:2] != "--" and "=" not in t:
            if any(ch in letters for ch in t[1:]):
                return True
    return False


def _segment_is_recursive(tokens: list, start: int) -> bool:
    """Whether a recursive flag (-R / --recursive / an -rf style cluster) belongs
    to the command starting at ``start``: scan only up to the next separator, so
    `grep -R x . && chmod +x f` does not make the chmod look recursive."""
    for t in tokens[start + 1 :]:
        if t in _SHELL_SEPARATORS or not set(t) - set(";&|()"):
            break
        if t in ("-R", "--recursive"):
            return True
        if t[:1] == "-" and t[:2] != "--" and "=" not in t and "R" in t[1:]:
            return True
    return False


def _inline_python_is_high_risk(code: str) -> bool:
    """Screen a `python -c` payload with the same analyzer the python tool uses,
    so an ordinary one-liner runs and a destructive one still asks. Source that
    does not parse fails closed: shell quoting may have mangled it, leaving
    nothing to screen."""
    try:
        ast.parse(code)
    except SyntaxError:
        return True
    return _python_is_high_risk(code)


def _terminal_is_high_risk(command: str, _depth: int = 0) -> bool:
    """High-risk terminal command for auto mode: credential/secret access,
    privilege escalation, destructive/persistence changes, or network
    exec/exfil. Ordinary dev commands run without a prompt. Fails closed
    (prompts) on an unparseable command. ``_depth`` bounds the recursion into
    shell ``-c`` payloads."""
    if len(command) > _MAX_TERMINAL_SCAN_CHARS:
        # Far longer than any ordinary command, and screening it is superlinear,
        # so it asks instead.
        return True
    if not command or not command.strip():
        return False
    # A credential/secret path read or write, or a sandbox escape (../), asks.
    if _command_references_sensitive(command):
        return True
    # A bare redirection with no command (`> notes.txt`, `: > notes.txt`) truncates
    # the file to zero bytes, the same loss as the gated `truncate -s 0`. A
    # redirect after a real command (`python train.py > out.log`) stays out.
    if _BARE_TRUNCATING_REDIRECT_RE.search(command):
        return True
    # A process substitution an interpreter executes runs a script the static scan
    # cannot read, so fail closed.
    if _PROC_SUBST_EXEC_RE.search(command):
        return True
    # A script piped into a shell (printf '...' | bash) or fed as a herestring
    # (bash <<< '...') is executed without ever appearing at command position.
    if _PKG_REMOVE_AT_CMD_RE.search(command):
        return True
    if _PIPE_TO_INTERPRETER_RE.search(command.lower()):
        return True
    _herestring = _HERESTRING_TO_INTERPRETER_RE.search(command)
    if _herestring:
        return True
    # Newlines separate commands in a shell but read as whitespace to shlex, and
    # ANSI-C quoting ($'rm') hides the real command name.
    decoded = _decode_ansi_c(command, keep_one_word = True)
    normalized = decoded.replace("\r\n", ";").replace("\n", ";").replace("\r", ";")
    # Identical to the blanket form unless a newline is actually present, so the
    # usual single-line command never pays for the quote walk.
    quoted_newlines_kept = (
        _separate_unquoted_newlines(decoded) if "\n" in decoded or "\r" in decoded else normalized
    )
    # Matched against a sed program below to tell an expansion the shell RUNS
    # from one the program merely quotes. Held in both newline forms so the
    # match works whichever pass produced the tokens.
    live_expansions: "set[str]" = set()
    if "$" in command or "`" in command:
        live_expansions = {
            form
            for expansion in _shell_expansions(command)
            for form in (
                expansion,
                expansion.replace("\r\n", ";").replace("\n", ";").replace("\r", ";"),
            )
        }
    # A verb hidden behind an assignment (c=rm; $c x) or a default parameter
    # (${c:-rm}) is expanded so the resolved token is scanned too.
    expanded = _expand_shell_assignments(_expand_param_defaults(normalized))
    # Run the network exfil check over the expanded form too, so a curl/wget
    # name assembled from variables (c=cu d=rl; $c$d -F ...) is still seen.
    if _command_is_network_exec_or_exfil(command) or _command_is_network_exec_or_exfil(expanded):
        return True
    # A command substitution at command position generates the command Bash runs.
    if _COMMAND_SUBST_AT_CMD_RE.search(command):
        return True
    # A variable executed at command position hides the name that actually runs. A
    # plain assignment is resolved by the expansion above, so reaching here means
    # the binding came from somewhere this scan cannot follow (a command
    # substitution, or `printf -v c rm`). No name left to screen: fail closed.
    if _HAS_COMMAND_SUBST_RE.search(command) and _VAR_EXECUTED_AS_COMMAND_RE.search(command):
        return True
    if _BARE_VAR_AS_COMMAND_RE.search(expanded):
        return True
    # An array run as a command (x=(git clean -fd); bash -c "${x[*]}") carries no
    # command substitution, and assignment expansion does not resolve arrays, so
    # the check above misses it. A benign array print is untouched.
    if _ARRAY_EXPANSION_RE.search(command) and _VAR_EXECUTED_AS_COMMAND_RE.search(command):
        return True
    # A newline inside a QUOTED argument is data, not a separator, and turning
    # it into `;` rewrites that data: a sed comment ends at a real newline, so
    # `sed '# note<newline>e CMD'` reads as one long comment once the newline is
    # gone. So a pass that only separates the UNQUOTED ones is scanned too. It
    # keeps every command boundary the blanket form has, so the token stream is
    # the same and only quoted content differs: the pass adds detections without
    # merging two commands into one segment. The set collapses to a single scan
    # for the usual single-line command.
    for text in {normalized, expanded, quoted_newlines_kept}:
        try:
            lexer = shlex.shlex(text, posix = True, punctuation_chars = ";&|()")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return True
        recursive = any(
            t in ("-R", "--recursive")
            or (t[:1] == "-" and t[:2] != "--" and "=" not in t and "R" in t[1:])
            for t in tokens
        )
        find_like = any(
            os.path.basename(t.strip(";&|()`{}")).lower() in ("find", "fd") for t in tokens
        )
        # Shared out over the sed words present, so a lone sed reads its whole
        # argument list and a line packed with them stays linear (_sed_scan_limit).
        sed_scan_limit = _sed_scan_limit(
            sum(1 for t in tokens if os.path.basename(t.strip(";&|()`{}")).lower() in _SED_COMMANDS)
        )
        # Built at most once per pass, and only when a sed program actually
        # names a variable, so a line packed with sed words stays linear.
        sed_vars: "dict[str, str] | None" = None
        sed_bindings: "list[tuple[int, str, str | None]] | None" = None
        sed_cursor = 0
        # Where a sed invocation really ends. Built at most once per pass, and
        # only once a sed is actually reached, so a line without one never pays
        # for the quote walk it needs (_quoted_separator_indexes).
        sed_stops: "frozenset[int] | None" = None
        sed_skips: "frozenset[int]" = frozenset()
        sed_quoted: "frozenset[int]" = frozenset()
        sed_globs: "frozenset[int]" = frozenset()
        sed_expandable: "frozenset[int]" = frozenset()
        if find_like and any(t.split("=", 1)[0] in _HIGH_RISK_FIND_FLAGS for t in tokens):
            return True
        # GNU tar runs --checkpoint-action=exec=CMD at each checkpoint, hiding a
        # command (including hard-blocked ones) inside an argument.
        if any(
            os.path.basename(t.strip(";&|()`{}")).lower() in _ARG_EXEC_FLAG_OWNERS for t in tokens
        ) and any(t.split("=", 1)[0] in _HIGH_RISK_ARG_EXEC_FLAGS for t in tokens):
            return True
        # An interpreter serving on the network exposes the session workdir; the
        # sandbox keeps no network namespace.
        if _LISTENER_PY_MODULE_RE.search(text) or _LISTENER_BIN_AT_CMD_RE.search(text):
            return True
        expect_command = True  # at the start of a command (after a separator)
        prefix_pending = False  # inside a wrapper (env/timeout/...) still seeking the command
        scan_forward = False  # a forwarding command (find/xargs/...) precedes another command
        current_command = ""  # the resolved command whose flags / git subcommand we judge
        git_subcommand = ""  # the first positional after `git`
        shell_c_pending = False  # a shell `-c` precedes its inline payload
        wrapper_value_pending = False  # a wrapper option precedes its value
        exec_flag_pending = False  # inside find/fd, waiting for -exec
        git_checkout_positionals = 0  # positionals seen after `git checkout`
        git_worktree_action = ""  # the action after `git worktree`
        win_operand_pending = False  # operand of a Windows `if exist`/`if defined`
        inline_python_pending = False  # next token is a `python -c` payload
        py_module_pending = False  # next token is the module after `python -m`
        git_submodule_action = ""  # the action after `git submodule`
        awk_program_pending = False  # next positional is an awk program
        git_config_alias_pending = False  # `git config alias.x` precedes its body
        git_glob_pending = False  # a git global option (-C repo) precedes its value
        chdir_pending = False  # a cd/pushd precedes its target directory
        xargs_index = -1  # an xargs awaiting the command whose argv it builds
        for _tok_idx, token in enumerate(tokens):
            if (
                token in _SHELL_SEPARATORS
                or (token in _SHELL_KEYWORDS_AS_SEP and expect_command)
                or not set(token) - set(";&|()")
            ):
                expect_command = True
                prefix_pending = False
                xargs_index = -1
                # A dangling wrapper option (env -u ; rm ...) must not consume
                # the next segment's command word.
                wrapper_value_pending = False
                scan_forward = False
                current_command = ""
                git_subcommand = ""
                git_worktree_action = ""
                win_operand_pending = False
                inline_python_pending = False
                py_module_pending = False
                git_submodule_action = ""
                awk_program_pending = False
                shell_c_pending = False
                git_glob_pending = False
                chdir_pending = False
                continue
            if py_module_pending:
                py_module_pending = False
                if token.strip("\"'").lower() in _LISTENER_PY_MODULE_NAMES:
                    return True
            if inline_python_pending:
                inline_python_pending = False
                if _depth >= 3 or _inline_python_is_high_risk(token):
                    return True
                continue
            if expect_command and token.lower() in _WIN_CONDITIONAL_KEYWORDS:
                # `if exist FILE del FILE`: the operand sits where the command
                # word would be, so the real command is still ahead.
                win_operand_pending = token.lower() != "not"
                continue
            if win_operand_pending:
                win_operand_pending = False
                continue
            if expect_command and _REDIR_PREFIX_RE.match(token):
                # Bash accepts a redirection before the command word
                # (`</dev/null rm -rf build`); the command is still to come.
                continue
            if exec_flag_pending and token == "--":
                # fd tells a user whose PATTERN starts with a dash to write
                # `fd -- '-foo'`, so nothing behind the marker is an option and
                # `fd -- -x rm` merely lists a file called `-x`.
                exec_flag_pending = False
                continue
            if token.startswith("-"):
                flag = token.split("=", 1)[0]
                # find/fd: the command after -exec/-ok is the one that runs.
                if exec_flag_pending and flag in _EXEC_FORWARD_FLAGS:
                    # `fd . --exec=rm` attaches the command to the flag, so the
                    # value is the command that runs, not an option argument.
                    if "=" in token and flag in _ATTACHED_EXEC_FLAGS:
                        attached = token.split("=", 1)[1].strip("\"'")
                        if attached and (
                            _depth >= 3 or _terminal_is_high_risk(attached, _depth + 1)
                        ):
                            return True
                    scan_forward = True
                    expect_command = True
                    continue
                if exec_flag_pending and token[:2] in {"-x", "-X"} and len(token) > 2:
                    # fd takes the command attached to the SHORT option too, and
                    # only the exact spellings were read as one: `fd '^victim$'
                    # . -xrm` deletes the match for real (fdfind 9.0.0).
                    attached = token[2:].strip("\"'")
                    if attached and (_depth >= 3 or _terminal_is_high_risk(attached, _depth + 1)):
                        return True
                    scan_forward = True
                    expect_command = True
                    continue
                if current_command == "setpriv" and flag in _SETPRIV_PRIVILEGE_FLAGS:
                    # Ahead of the wrapper-value skip below, which would otherwise
                    # swallow `--reuid 0` before it is judged.
                    return True
                # A wrapper option taking a SEPARATE value (env -u NAME): the next
                # token is that value, not the wrapped command.
                if (
                    prefix_pending
                    and "=" not in token
                    and flag in _WRAPPER_VALUE_FLAGS_BY_CMD.get(current_command, frozenset())
                ):
                    wrapper_value_pending = True
                    continue
                # An interpreter running inline code (python -c, node -e) executes
                # a program the terminal path never screens. Matches the long
                # --eval/--exec forms and any short cluster carrying -c.
                _inline_spec = (
                    _inline_code_flag_spec(current_command)
                    if _is_inline_code_interpreter(current_command)
                    else None
                )
                _current_is_python_family = current_command.startswith(("python", "pypy"))
                if _current_is_python_family and flag == "-m":
                    py_module_pending = True
                    continue
                if _inline_spec is not None and (
                    flag in _inline_spec[0] or _short_flag_arg(token, _inline_spec[1]) is not None
                ):
                    # Python payloads go through the python tool's analyzer, so an
                    # ordinary one-liner runs and a destructive one asks. The other
                    # runtimes have no analyzer here, so they stay gated.
                    if _current_is_python_family:
                        # A bare `-c` yields an EMPTY attached value, not None,
                        # so the payload is the next token; only a non-empty
                        # value is the attached form (python -c'print(1)').
                        _attached = _short_flag_arg(token, _inline_spec[1])
                        if _attached:
                            if _depth >= 3 or _inline_python_is_high_risk(_attached):
                                return True
                            continue
                        inline_python_pending = True
                        continue
                    return True
                # node/bun -p / --print evaluate and print arbitrary source, the
                # same inline-code risk as -e/--eval (attached node -p'...' too).
                if current_command in _NODE_PRINT_INTERPRETERS and (
                    flag in _NODE_PRINT_FLAGS or _short_flag_arg(token, "p") is not None
                ):
                    return True
                # PowerShell -Command / -EncodedCommand run an inline program the
                # terminal path cannot screen; a bare `pwsh script.ps1` still runs.
                if current_command in _POWERSHELL_INTERPRETERS and flag.lower().startswith(
                    ("-c", "-e")
                ):
                    return True
                # A shell `-c PAYLOAD` runs its quoted payload; screen it
                # recursively. Combined clusters (bash -lc) carry -c too.
                if current_command in _SHELL_C_INTERPRETERS:
                    payload = _short_flag_arg(token, "c")
                    if payload is not None:
                        # A short run of plain letters after `c` (bash -ce) is more
                        # bash OPTIONS, not an attached payload: the command string
                        # still comes from the next token.
                        if payload and payload.isalpha() and len(payload) <= 4:
                            shell_c_pending = True
                        elif payload:
                            if _depth >= 3:
                                return True
                            if _terminal_is_high_risk(payload, _depth + 1):
                                return True
                        else:
                            shell_c_pending = True
                # env -S 'cmd' runs the string as a new command, so screen it;
                # env -C chdirs (enabling a relative sensitive read), so it asks.
                if current_command == "env":
                    if flag in ("-C", "--chdir"):
                        return True
                    payload = None
                    if token.startswith("-S") and token != "-S":
                        payload = token[2:]  # attached: -S'cmd'
                    elif flag == "--split-string" and "=" in token:
                        payload = token.split("=", 1)[1]
                    elif token == "-S" or flag == "--split-string":
                        shell_c_pending = True  # payload is the next token
                    if (
                        payload is not None
                        and _depth < 3
                        and _terminal_is_high_risk(payload, _depth + 1)
                    ):
                        return True
                if current_command == "sysctl" and flag in _SYSCTL_WRITE_FLAGS:
                    return True
                if current_command == "fallocate" and (
                    flag in _FALLOCATE_DESTRUCTIVE_FLAGS
                    or any(f in _FALLOCATE_DESTRUCTIVE_FLAGS for f in _short_flag_cluster(token))
                ):
                    return True
                if (
                    current_command == "git"
                    and git_subcommand == "worktree"
                    and git_worktree_action == "remove"
                    and flag in _HIGH_RISK_GIT_WORKTREE_FLAGS
                ):
                    return True
                if current_command == "git":
                    # reset --hard discards the working tree; push --force
                    # overwrites a remote ref.
                    if git_subcommand == "reset" and flag in _HIGH_RISK_GIT_RESET_FLAGS:
                        return True
                    if git_subcommand == "push" and (
                        flag in _HIGH_RISK_GIT_PUSH_FLAGS
                        or any(f in _HIGH_RISK_GIT_PUSH_FLAGS for f in _short_flag_cluster(token))
                    ):
                        return True
                    # git checkout -f / --force, or an explicit `--` path
                    # separator (git checkout -- file), discards tracked edits.
                    if git_subcommand == "checkout" and (
                        flag in _HIGH_RISK_GIT_CHECKOUT_FLAGS
                        or any(
                            f in _HIGH_RISK_GIT_CHECKOUT_FLAGS for f in _short_flag_cluster(token)
                        )
                        or token == "--"
                        or flag == "--pathspec-from-file"
                    ):
                        return True
                    if git_subcommand == "checkout-index" and (
                        flag in _HIGH_RISK_GIT_CHECKOUT_INDEX_FLAGS
                        or any(
                            f in _HIGH_RISK_GIT_CHECKOUT_INDEX_FLAGS
                            for f in _short_flag_cluster(token)
                        )
                    ):
                        return True
                    if git_subcommand == "tag" and (
                        flag in _HIGH_RISK_GIT_TAG_FLAGS
                        or any(f in _HIGH_RISK_GIT_TAG_FLAGS for f in _short_flag_cluster(token))
                    ):
                        return True
                    if git_subcommand == "switch" and (
                        flag in _HIGH_RISK_GIT_SWITCH_FLAGS
                        or any(f in _HIGH_RISK_GIT_SWITCH_FLAGS for f in _short_flag_cluster(token))
                    ):
                        return True
                    # git branch -D / -M drops or overwrites unmerged commits.
                    if git_subcommand == "branch" and (
                        flag in _HIGH_RISK_GIT_BRANCH_FLAGS
                        or any(f in _HIGH_RISK_GIT_BRANCH_FLAGS for f in _short_flag_cluster(token))
                    ):
                        return True
                    # --config-env=<key>=<envvar> reads the value from the
                    # environment, unresolvable here, so an alias key would store
                    # unscreened code git runs on the next call.
                    if flag == "--config-env" and _GIT_CONFIG_ENV_ALIAS_RE.search(token):
                        return True
                    # A git global option with a separate value (git -C repo clean)
                    # precedes its value, not the subcommand.
                    if not git_subcommand and "=" not in token and flag in _GIT_GLOBAL_VALUE_FLAGS:
                        git_glob_pending = True
                continue
            if _ASSIGNMENT_RE.match(token):
                _assign_name, _, _assign_value = token.partition("=")
                # `alias zap='rm -rf'` stores a command bash runs when the alias
                # is invoked, the same shape as a git alias body.
                if current_command == "alias" and _assign_value:
                    if _depth >= 3 or _terminal_is_high_risk(_assign_value, _depth + 1):
                        return True
                # PATH/LD_PRELOAD-style assignments hijack command lookup, but only
                # for the command they prefix: a bare `export PATH=...` runs
                # nothing, and the shell it was set in exits immediately.
                if _env_assignment_is_unsafe(
                    _assign_name, _assign_value
                ) and _segment_has_command_after(tokens, _tok_idx):
                    return True
                continue
            raw = token.strip(";&|()`{}")
            if not raw:
                continue
            # cmd.exe /c (or /k) runs the following token as a nested command. /c is
            # not a `-`-flag, so it is handled here in argument position after cmd.
            if current_command in _CMD_SHELLS and raw.lower() in ("/c", "/k"):
                shell_c_pending = True
                continue
            # The payload of a shell `-c`, screened recursively (bounded depth).
            if shell_c_pending:
                shell_c_pending = False
                # An unquoted payload (cmd /c git clean -fd) spans the remaining
                # tokens, so screen the whole remainder.
                payload = " ".join(tokens[_tok_idx:])
                if _depth >= 3:
                    # Too deeply nested to screen: fail closed.
                    return True
                if _terminal_is_high_risk(payload, _depth + 1):
                    return True
                if payload != raw and _terminal_is_high_risk(raw, _depth + 1):
                    return True
                expect_command = False
                continue
            # The value of a git global option (git -C repo clean): not the subcommand.
            if git_glob_pending:
                git_glob_pending = False
                # `git -c alias.x=BODY` defines an alias git later executes, so the
                # payload is real code hiding in an option value: screen it.
                m = _GIT_ALIAS_ASSIGN_RE.match(raw)
                if m and _depth < 3:
                    alias_body = m.group(1)
                    # A `!` alias runs through a shell; a plain one is a git
                    # subcommand, so screen it as `git <body>` to reach the git
                    # gates (alias.n='clean -fd' really runs `git clean -fd`).
                    nested = alias_body[1:] if alias_body.startswith("!") else "git " + alias_body
                    if _terminal_is_high_risk(nested, _depth + 1):
                        return True
                continue
            # The value of a wrapper option (env -u FOO, stdbuf -o L): not the
            # command, so skip it and keep looking for the wrapped command.
            if wrapper_value_pending:
                wrapper_value_pending = False
                continue
            # A wrapper's bare duration argument (timeout 5 rm) is not the command.
            if prefix_pending and _WRAPPER_DURATION_RE.fullmatch(raw):
                continue
            base = os.path.basename(raw).lower()
            stem, ext = os.path.splitext(base)
            if ext in {".exe", ".com", ".bat", ".cmd"}:
                base = stem
            if (expect_command or prefix_pending) and (
                base in _AUTO_SAFE_WRAPPERS
                or base in _MULTICALL_BINARIES
                or base in _PRIVILEGE_EXEC_WRAPPERS
            ):
                # A wrapper (env/timeout) or a multicall binary (busybox rm)
                # precedes the real command; keep seeking it, but track it so its
                # own flags (env -S / -C) are judged in the meantime.
                prefix_pending = True
                expect_command = False
                current_command = base
                continue
            if expect_command or prefix_pending or scan_forward:
                if base in _HIGH_RISK_COMMANDS or base.startswith("mkfs"):
                    # A container CLI reading its own state (docker ps, docker
                    # logs) inspects; anything else starts or enters a container.
                    if not (
                        base in _CONTAINER_CLIS
                        and _container_subcommand_is_read_only(tokens, _tok_idx)
                    ):
                        return True
                # Bash expands a command-position glob after this scan, so the name
                # here is not the one that runs (`/bin/r[m] -rf x`): ask.
                if _is_unresolved_command_glob(base):
                    return True
                # A server binary resolved here covers the wrapped and absolute
                # forms (env uvicorn app:api, timeout 60 gunicorn, /usr/bin/uvicorn).
                if base in _LISTENER_BINARIES:
                    return True
                if base in _HIGH_RISK_RECURSIVE_COMMANDS and _segment_is_recursive(
                    tokens, _tok_idx
                ):
                    return True
                if base in _HIGH_RISK_FORWARDING_COMMANDS:
                    if base == "xargs" and xargs_index < 0:
                        # It builds the argv of whatever follows, so a sed there
                        # may be handed a program this scan cannot see.
                        xargs_index = _tok_idx
                    # find/fd only run a child at -exec/-ok; forwarding from the
                    # command itself would make `find . -name rm` prompt.
                    if base in _EXEC_FLAG_FORWARDING_COMMANDS:
                        scan_forward = False
                        exec_flag_pending = True
                    else:
                        scan_forward = True
                elif base == "git":
                    # Only git needs the forwarding scan to stop: its risk lives in
                    # the SUBCOMMAND (git clean), so following tokens are git's own
                    # arguments. Others keep scanning, since find's predicates sit
                    # between `find` and `-exec rm`.
                    scan_forward = False
                # Remember the resolved command so its own flags (python -c), git
                # subcommand or chdir target can be judged as they follow.
                current_command = base
                if base in _CHDIR_COMMANDS:
                    chdir_pending = True
                if base in _AWK_COMMANDS:
                    awk_program_pending = True
                if base in _SED_COMMANDS:
                    # `e` / `s///e` shell out from inside the script, which may
                    # ride on -e/--expression rather than the next positional.
                    # A script --sandbox / --posix stops sed compiling is already
                    # left out of the program (_sed_invocation), so a payload
                    # inside one never reaches this screen.
                    if sed_stops is None:
                        # A quoted `';'` / `'+'` operand is a sed FILE, not the
                        # end of the invocation; reading it as one dropped the
                        # `-e` script behind it (`sed -n ';' -e '1e rm -f
                        # victim' input` really runs rm). A redirection is the
                        # other way round: those words never reach sed at all.
                        sed_quoted = _quoted_separator_indexes(text, tokens, ";&|()")
                        _flags, sed_stops, sed_skips = _exec_scan_layout(
                            tokens, sed_quoted, _quoted_redirection_indexes(text, tokens, ";&|()")
                        )
                        sed_globs = _unquoted_glob_indexes(text, tokens, ";&|()")
                        sed_expandable = _unquoted_expansion_indexes(text, tokens, ";&|()")
                    sed_alternatives, sed_overflowed, sed_live = _sed_invocation(
                        tokens,
                        _tok_idx,
                        sed_scan_limit,
                        sed_stops,
                        sed_skips,
                        sed_globs,
                        sed_expandable,
                    )
                    sed_program = "\n".join(sed_alternatives)
                    if sed_overflowed:
                        # The script was pushed past the scan window by padding
                        # options, so "no payload found" only means "not looked
                        # at": ask instead of falling through to safe.
                        return True
                    if _sed_program_is_a_placeholder(sed_program):
                        # find rewrites `{}` before the child starts.
                        return True
                    if xargs_index >= 0 and _xargs_hides_sed_program(
                        tokens, xargs_index, _tok_idx, sed_program
                    ):
                        # xargs builds the argv from stdin or an -I placeholder,
                        # so the program is not in the text to read at all.
                        return True
                    if "$" in sed_program:
                        # A program held in a variable (p='# note<newline>e CMD';
                        # sed "$p" f) is only a program once the reference is
                        # resolved, and only THIS pass keeps the quoted newline
                        # that ends the comment: the blanket one turns the whole
                        # value into a single inert comment line. Only the
                        # assignments ahead of this sed can reach it, and the
                        # last of them is the one bash uses.
                        if sed_bindings is None:
                            sed_bindings = _assignment_bindings(tokens, sed_quoted)
                            sed_vars = {}
                        sed_cursor = _bindings_before(sed_bindings, sed_cursor, _tok_idx, sed_vars)
                    sed_variants = [
                        variant
                        for alternative in sed_alternatives
                        for variant in _sed_program_variants(alternative, sed_vars or {})
                    ]
                    if any(_sed_exec_payloads(variant) for variant in sed_variants):
                        return True
                    # A program the shell still has to build is not knowable
                    # here -- sed splices the result straight into the program
                    # text, where it can open `;e CMD` from any position -- so
                    # an unread one asks rather than being assumed to only edit
                    # text (_sed_program_unresolved).
                    # Only where the program's OWN occurrence is one the
                    # shell expands: the live set covers the whole command, so
                    # matching by text alone made the read-only
                    # `echo "$p"; sed 's/$p/x/' f` ask for an expansion another
                    # command performs.
                    if sed_live and _sed_program_unresolved(sed_variants, live_expansions):
                        return True
            elif current_command == "git" and not git_subcommand:
                # The first positional after `git` is its subcommand.
                git_subcommand = base
                if base == "clean" and _segment_has_flag(
                    tokens, _tok_idx, _GIT_CLEAN_DRY_RUN_FLAGS, "n"
                ):
                    # A dry run lists what would go and removes nothing.
                    expect_command = False
                    prefix_pending = False
                    continue
                if base in _HIGH_RISK_GIT_SUBCOMMANDS:
                    return True
            elif awk_program_pending:
                awk_program_pending = False
                if _AWK_SHELL_ESCAPE_RE.search(raw):
                    return True
            elif (
                current_command == "git"
                and git_subcommand == "submodule"
                and git_submodule_action == "foreach"
            ):
                # `git submodule foreach '<cmd>'` runs the argument in every
                # submodule, so it is a command in its own right.
                git_submodule_action = ""
                if _depth >= 3 or _terminal_is_high_risk(raw, _depth + 1):
                    return True
            elif (
                current_command == "git"
                and git_subcommand == "submodule"
                and not git_submodule_action
            ):
                git_submodule_action = base
            elif current_command == "getent" and base in _GETENT_CREDENTIAL_DATABASES:
                # The database name is the whole request; no path is mentioned.
                return True
            elif current_command == "openssl" and base in _OPENSSL_NETWORK_SUBCOMMANDS:
                # openssl s_client/s_server open a TLS socket. The regex above is
                # anchored at command position, so it misses the wrapped forms.
                return True
            elif current_command == "sysctl" and "=" in raw:
                # `sysctl net.ipv4.ip_forward=1` writes without needing -w.
                return True
            elif (
                current_command == "git"
                and git_subcommand == "worktree"
                and not git_worktree_action
            ):
                git_worktree_action = base
            elif current_command in _EVAL_SUBCOMMAND_INTERPRETERS and base == "eval":
                # `deno eval "..."` / `bun eval "..."` run inline code as a
                # subcommand rather than a flag, the same risk as -e.
                return True
            elif current_command == "git" and git_subcommand == "checkout" and base == ".":
                # `git checkout .` discards every tracked working-tree change.
                return True
            elif current_command == "git" and git_subcommand == "checkout":
                # A SECOND positional means the first was a commit-ish and this is
                # a pathspec (git checkout HEAD file), which overwrites the file. A
                # single one is ambiguous with a branch name and is left alone.
                git_checkout_positionals += 1
                if git_checkout_positionals >= 2:
                    return True
            elif (
                current_command == "git" and git_subcommand == "config" and git_config_alias_pending
            ):
                git_config_alias_pending = False
                # The stored alias body is code git runs on the next invocation.
                nested = raw[1:] if raw.startswith("!") else "git " + raw
                if _depth >= 3 or _terminal_is_high_risk(nested, _depth + 1):
                    return True
            elif (
                current_command == "git"
                and git_subcommand == "config"
                and raw.lower().startswith("alias.")
            ):
                git_config_alias_pending = True
            elif (
                current_command == "git"
                and git_subcommand == "stash"
                and base in _HIGH_RISK_GIT_STASH_ACTIONS
            ):
                # `git stash clear` / `drop` destroys stashed work unrecoverably.
                return True
            elif current_command == "git" and git_subcommand == "push" and raw[:1] in ("+", ":"):
                # A refspec forcing (+src:dst) or deleting (:dst) a remote ref is
                # the punctuation form of --force / --delete.
                if len(raw) > 1:
                    return True
            elif chdir_pending:
                # A chdir into a sensitive directory sets up a relative read that no
                # single token spells out (cd /proc/$PPID; cat environ).
                chdir_pending = False
                if any(
                    _SENSITIVE_CHDIR_RE.search(cand)
                    for cand in (raw, _expand_param_defaults(raw), _expand_shell_assignments(raw))
                ):
                    return True
            expect_command = False
            prefix_pending = False
    return False


def _python_is_high_risk(code: str) -> bool:
    """High-risk python for auto mode: code the sandbox static analysis would
    refuse anyway (shell escape, network egress, a sensitive read), that
    reads/writes a credential path, or that runs dynamically built code past
    those static checks. Ordinary in-workdir file writes and computation run
    without a prompt."""
    if not code or not code.strip():
        return False
    # _check_code_safety objecting means execution would be refused outright, so a
    # confirmation first beats a silent refusal.
    if _check_code_safety(code) is not None:
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Unparsable code never runs, but scan the raw text anyway.
        return _references_sensitive_path(code)
    # A credential basename only names a file when it appears in a string, so match
    # it there rather than across the source: `credentials = {}` and
    # `def load_credentials()` do no I/O and must not prompt.
    for _node in ast.walk(tree):
        if (
            isinstance(_node, ast.Constant)
            and isinstance(_node.value, str)
            and _references_sensitive_path(_node.value)
        ):
            return True
    # A destructive filesystem call (shutil.rmtree, Path.unlink) asks, for parity
    # with the terminal `rm` gate. Collect bare import aliases first.
    destructive_fs_aliases: "set[str]" = set()
    # Modules whose handles end processes; tracked so an unrelated .kill() on a
    # user-defined object is not mistaken for one.
    psutil_names: "set[str]" = set()
    for _node in ast.walk(tree):
        if isinstance(_node, ast.Import):
            for _a in _node.names:
                if _a.name.split(".")[0] in _PY_PROCESS_MODULES:
                    psutil_names.add("psutil")
        elif (
            isinstance(_node, ast.ImportFrom)
            and (_node.module or "").split(".")[0] in _PY_PROCESS_MODULES
        ):
            psutil_names.add("psutil")
    # `import os as filesystem` rebinds the module, so os.remove reached through
    # the alias (filesystem.remove) must resolve too; posix is os's low-level twin.
    os_module_aliases: "set[str]" = {"os", "posix", "nt"}

    def _is_os_module_ref(value) -> bool:
        # A Name bound to os/posix/nt, a walrus binding one, or a literal
        # __import__("os") call used directly. builtins.__import__ is the same
        # callable reached through the module, so both spellings resolve.
        if isinstance(value, ast.Name):
            return value.id in os_module_aliases
        if isinstance(value, ast.NamedExpr):
            return _is_os_module_ref(value.value)
        if not isinstance(value, ast.Call):
            return False
        func = value.func
        is_import = (isinstance(func, ast.Name) and func.id == "__import__") or (
            isinstance(func, ast.Attribute) and func.attr == "__import__"
        )
        return (
            is_import
            and bool(value.args)
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value in ("os", "posix", "nt")
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _PY_DESTRUCTIVE_FS_MODULES:
            for alias in node.names:
                if alias.name in _PY_DESTRUCTIVE_FS_IMPORT_NAMES:
                    destructive_fs_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("os", "posix", "nt") and alias.asname:
                    os_module_aliases.add(alias.asname)
        elif isinstance(node, ast.Assign) and _is_os_module_ref(node.value):
            # m = __import__("os") binds the module under a new name.
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    os_module_aliases.add(tgt.id)
        elif isinstance(node, ast.NamedExpr) and _is_os_module_ref(node.value):
            # (fs := os).remove(...) binds it in an expression instead.
            if isinstance(node.target, ast.Name):
                os_module_aliases.add(node.target.id)

    def _is_fs_module_ref(value) -> bool:
        # os/posix/nt (including aliases), or a literal shutil/pathlib name.
        if _is_os_module_ref(value):
            return True
        return isinstance(value, ast.Name) and value.id in _PY_DESTRUCTIVE_FS_MODULES

    def _is_process_kill(node) -> bool:
        # psutil.Process(pid).kill() / .terminate(), including a handle bound to
        # a name first. Keyed on the psutil import so an unrelated .kill() on a
        # user object does not prompt.
        if "psutil" not in psutil_names:
            return False
        return isinstance(node, ast.Attribute) and node.attr in _PY_PROCESS_KILL_ATTRS

    def _is_destructive_attr(attr: str, value) -> bool:
        # A destructive-name attribute (unlink/rmtree/...) on any receiver, or
        # `remove` specifically on the os module (or an alias of it).
        if attr in _PY_DESTRUCTIVE_FS_ATTRS:
            return True
        return attr in _PY_DESTRUCTIVE_FS_OS_ATTRS and _is_os_module_ref(value)

    def _module_dict_target(value):
        # The module namespace as a dict: vars(os) or os.__dict__.
        if isinstance(value, ast.Attribute) and value.attr == "__dict__":
            return value.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "vars"
            and len(value.args) == 1
        ):
            return value.args[0]
        return None

    def _is_module_dict_lookup(node) -> bool:
        # vars(os)["remove"] / os.__dict__["unlink"] is getattr spelled through
        # the namespace dict, so screen the key the same way. Anchored to a
        # filesystem module, leaving an ordinary d["remove"] alone.
        if not isinstance(node, ast.Subscript):
            return False
        module = _module_dict_target(node.value)
        if module is None:
            return False
        attr = _folded_str_literal(node.slice)
        if attr is None:
            return _is_fs_module_ref(module)
        return _is_destructive_attr(attr, module)

    # `rm = getattr(os, "remove")` stores the lookup and calls it later, so the
    # direct getattr(...)(...) shape never sees it. Bind the name here instead.
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "getattr"
            and len(node.value.args) >= 2
        ):
            continue
        _attr = _folded_str_literal(node.value.args[1])
        _hit = (
            _is_fs_module_ref(node.value.args[0])
            if _attr is None
            else _is_destructive_attr(_attr, node.value.args[0])
        )
        if _hit:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    destructive_fs_aliases.add(tgt.id)

    # `f = open(path, "r+")` then `f.truncate(0)` zeroes the file. Gated via the
    # handle name, not the bare `.truncate` attribute: pandas DataFrame.truncate()
    # is common here and non-destructive.
    file_handles: "set[str]" = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "open"
        ):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    file_handles.add(tgt.id)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            # `with open(p, "r+") as f:` binds the handle like an assignment.
            for item in node.items:
                ctx = item.context_expr
                if (
                    isinstance(ctx, ast.Call)
                    and isinstance(ctx.func, ast.Name)
                    and ctx.func.id == "open"
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    file_handles.add(item.optional_vars.id)
    if file_handles:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "truncate"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in file_handles
            ):
                return True
    # A bound reference (f = os.remove; f(x)) hides the call site behind a plain
    # Name, so record the target name as a destructive alias to catch f(...) below.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript):
            if _is_module_dict_lookup(node.value):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        destructive_fs_aliases.add(tgt.id)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            if _is_destructive_attr(node.value.attr, node.value.value):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        destructive_fs_aliases.add(tgt.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.target, ast.Name)
        ):
            # An annotated binding (f: object = os.remove) is the same alias.
            if _is_destructive_attr(node.value.attr, node.value.value):
                destructive_fs_aliases.add(node.target.id)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if _is_destructive_attr(func.attr, func.value):
                return True
            if _is_process_kill(func):
                return True
        elif isinstance(func, ast.Subscript):
            if _is_module_dict_lookup(func):
                return True
        elif isinstance(func, ast.Name) and func.id in destructive_fs_aliases:
            return True
        elif isinstance(func, ast.NamedExpr):
            # (f := os.remove)(...) binds and calls in one expression.
            inner = func.value
            if isinstance(inner, ast.Attribute) and _is_destructive_attr(inner.attr, inner.value):
                return True
            if isinstance(inner, ast.Name) and inner.id in destructive_fs_aliases:
                return True
            if _is_module_dict_lookup(inner):
                return True
        # getattr(os, "remove")(x) resolves the attribute at runtime. The name is
        # folded first ("un" + "link"); one that cannot be folded at all on a
        # filesystem module fails closed, since there is nothing left to screen.
        if (
            isinstance(func, ast.Call)
            and isinstance(func.func, ast.Name)
            and func.func.id == "getattr"
            and len(func.args) >= 2
        ):
            attr_name = _folded_str_literal(func.args[1])
            if attr_name is None:
                if _is_fs_module_ref(func.args[0]):
                    return True
            elif _is_destructive_attr(attr_name, func.args[0]):
                return True
    # A sensitive path split across names or joins (p = "/etc"; open(p + "/shadow"))
    # is not a contiguous literal above, so fold the string-literal variables
    # through _folded_path and re-check. An unresolved fragment folds to a sentinel
    # so a partial fold never false-positives.
    str_vars: "dict[str, str]" = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            str_vars[node.targets[0].id] = value.value
        elif isinstance(value, (ast.Call, ast.BinOp, ast.JoinedStr, ast.Name)):
            # Record a fully-literal folded path so a later reuse (p / "shadow")
            # resolves; a dynamic fold is skipped so only known paths bind.
            folded = _folded_path(value, str_vars)
            if folded and "\x00" not in folded and "\x02" not in folded:
                str_vars[node.targets[0].id] = folded

    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.JoinedStr, ast.Call)):
            folded = _folded_path(node, str_vars)
            if folded and _folded_is_sensitive(folded):
                return True
    # exec/eval/compile/__import__ of a non-literal (exec(b64decode(...)),
    # eval(input()), __import__(name)) runs whatever it builds at runtime, past
    # the static checks above; ask. A literal eval("1+1") is harmless and runs.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            if func.attr == "import_module":  # importlib.import_module(name)
                name = "__import__"
            elif func.attr in ("exec", "eval", "compile"):  # builtins.exec(...)
                name = func.attr
        if name not in ("exec", "eval", "compile", "__import__"):
            continue
        # The source is the first positional, or the source=/name= keyword when
        # called by keyword (compile(source=x), importlib.import_module(name=x)).
        arg = node.args[0] if node.args else None
        if arg is None:
            for kw in node.keywords:
                if kw.arg in ("source", "name"):
                    arg = kw.value
                    break
        if arg is None:
            continue
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (str, bytes)):
            # A literal source is only as safe as the code it runs, so screen it
            # recursively.
            if name == "__import__":
                # A module name is not analyzable as code, but a literal
                # __import__("socket") binds a side-effecting module just like a
                # static import, so apply the same module screen.
                mod = (
                    arg.value.decode("utf-8", "replace")
                    if isinstance(arg.value, bytes)
                    else arg.value
                )
                if isinstance(mod, str) and mod.split(".")[0] in _AUTO_UNSAFE_PY_MODULES:
                    return True
                continue
            inner = (
                arg.value.decode("utf-8", "replace") if isinstance(arg.value, bytes) else arg.value
            )
            if _python_is_high_risk(inner):
                return True
            continue
        return True
    return False


def is_high_risk_tool_call(name: str, arguments: dict) -> bool:
    """Whether a tool call is sensitive enough to pause for approval in auto
    ("Approve for me") mode.

    Unlike is_potentially_unsafe_tool_call (which prompts on anything not
    read-only), this prompts only on genuinely sensitive actions - credential
    access, privilege escalation, destructive/persistence changes, and network
    exec/exfil - and lets ordinary development commands run. The hard-block command
    set, rlimits and secret-env stripping remain in force underneath. Unknown tools
    fail closed (prompt).
    """
    if _web_search_fetches_url(name, arguments):
        return True
    if name in _ALWAYS_SAFE_TOOLS:
        return False
    if name == "render_html":
        # A static canvas is fine; only a networked canvas can egress.
        return _render_html_reaches_network(arguments)
    if name.startswith(MCP_TOOL_PREFIX):
        tool_name = name.split("__", 2)[-1]
        # Split camelCase into `_`-delimited terms so the term-boundary regexes
        # below match camelCase names too.
        tool_name = _CAMEL_CASE_RE.sub("_", tool_name)
        # An execution tool runs arbitrary commands on the MCP server, outside the
        # terminal sandbox; a credential noun discloses secrets; a read/write
        # pointed at a sensitive path is a sensitive access. All prompt, while
        # ordinary create/update/delete MCP calls run.
        _reads = bool(_AUTO_READ_MCP_VERB_RE.search(tool_name))
        if _AUTO_EXEC_MCP_COMPOUND_RE.search(tool_name):
            return True
        if _AUTO_EXEC_MCP_TOOL_RE.search(tool_name) and not (
            _reads and not _AUTO_EXEC_MCP_VERB_ONLY_RE.search(tool_name)
        ):
            return True
        if _AUTO_DESTRUCTIVE_MCP_VERB_RE.search(tool_name):
            return True
        if _AUTO_PRIVILEGE_MCP_VERB_RE.search(tool_name):
            return True
        if _AUTO_HIGH_IMPACT_MCP_RE.search(tool_name) and not _reads:
            return True
        if _AUTO_PRIVILEGE_MCP_NOUN_RE.search(
            tool_name
        ) and _AUTO_PRIVILEGE_MCP_SOFT_VERB_RE.search(tool_name):
            return True
        if _AUTO_SENSITIVE_MCP_NOUN_RE.search(tool_name):
            return True
        if _mcp_arguments_reference_sensitive(arguments):
            return True
        # A read-named tool carrying a destructive payload (query_database
        # {"query": "DELETE FROM runs"}) masks a destructive external action behind
        # a read-looking name. Honestly-named create/update calls still run.
        if _mcp_arguments_mutate(arguments):
            return True
        # MCP names are an open vocabulary, not the finite set of POSIX utilities,
        # so the denylists above cannot be complete: an unfamiliar verb
        # (nuke_database) would sail through as ordinary. A name carrying no
        # recognised verb at all therefore asks.
        if not _mcp_verb_is_known(tool_name):
            return True
        return False
    if name == "terminal":
        return _terminal_is_high_risk(str(arguments.get("command", "")))
    if name == "python":
        return _python_is_high_risk(str(arguments.get("code", "")))
    return True


def _canon_win_path(p: str) -> str:
    """Canonical form for trust comparison: realpath (expands 8.3 aliases and
    resolves junctions/symlinks) + normcase/normpath."""
    return os.path.normcase(os.path.normpath(os.path.realpath(p)))


def _augment_native_program_roots(roots: list[str]) -> list[str]:
    """Add the native Program Files sibling for any x86 root by stripping the
    `` (x86)`` suffix, so a 32-bit process (whose known-folder ids map only to
    the x86 root) still trusts a 64-bit Git install."""
    out = list(roots)
    for root in roots:
        base = root.rstrip("\\/")
        if base.lower().endswith(" (x86)"):
            native = base[: -len(" (x86)")]
            if native and native not in out:
                out.append(native)
    return out


def _windows_program_roots() -> list[str]:
    """Program Files install roots, resolved ONLY from the Windows known-folder
    API (SHGetKnownFolderPath). Fails closed (returns ``[]``) if the API is
    unavailable: env vars (%ProgramFiles%, even %SystemDrive%) are caller-
    overrideable and could relocate the trust boundary, so we never derive a
    trusted root from them. On any real Windows host shell32 is present, so
    this only returns empty in a broken/non-Windows environment where the
    sandbox git-PATH feature is not needed anyway (#7317).
    """
    roots: list[str] = []
    try:
        import ctypes
        from ctypes import wintypes

        # FOLDERID_ProgramFiles, _ProgramFilesX86, _ProgramFilesX64. The X64
        # id (Win10 1703+) yields the native root even from a 32-bit process,
        # where the first two both map to Program Files (x86).
        folder_ids = (
            "{905e63b6-c1bf-494e-b29c-65b732d3d21a}",
            "{7C5A40EF-A0FB-4BFC-874A-C0F2E0B9FA8E}",
            "{6D809377-6AF0-444b-8957-A3773F02200E}",
        )
        _SHGet = ctypes.windll.shell32.SHGetKnownFolderPath
        _CoTaskMemFree = ctypes.windll.ole32.CoTaskMemFree
        for fid in folder_ids:
            guid = ctypes.create_string_buffer(16)
            ctypes.windll.ole32.CLSIDFromString(wintypes.LPCWSTR(fid), ctypes.byref(guid))
            ptr = ctypes.c_wchar_p()
            if _SHGet(ctypes.byref(guid), 0, None, ctypes.byref(ptr)) == 0:
                if ptr.value:
                    roots.append(ptr.value)
                _CoTaskMemFree(ptr)
    except Exception:
        return []
    return _augment_native_program_roots(roots)


def _resolve_trusted_windows_git() -> tuple[str, str]:
    """Find a git launcher in a TRUSTED Program Files dir. Returns
    ``(canonical_dir, ext)`` or ``("", "")``.

    ``shutil.which`` returns only the first PATH match, which may be an
    untrusted user shim; scan the remaining PATH entries for a later trusted
    Git so bare ``git`` still resolves (#7317).
    """
    exts = [e for e in (os.environ.get("PATHEXT") or ".EXE;.CMD;.BAT;.COM").split(os.pathsep)]
    candidates: list[str] = []
    primary = shutil.which("git")
    if primary:
        candidates.append(primary)
    for entry in (os.environ.get("PATH") or "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry or not os.path.isabs(entry):
            continue
        for ext in exts:
            cand = os.path.join(entry, "git" + ext)
            if os.path.isfile(cand):
                candidates.append(cand)
    for git_exe in candidates:
        git_dir = os.path.dirname(git_exe)
        if os.path.isabs(git_dir) and _is_trusted_windows_program_dir(git_dir):
            return os.path.realpath(git_dir), os.path.splitext(git_exe)[1].upper()
    return "", ""


def _is_trusted_windows_program_dir(path: str) -> bool:
    """True when ``path`` sits under a system-managed Program Files root.

    Only the Program Files roots are trusted (admin-writable only), resolved
    via the known-folder API so an overridden env var cannot relocate them,
    never ``%SystemRoot%`` (Git does not install there and it holds
    world-writable subdirs like ``Windows\\Temp``). Per-user managers
    (Scoop/Choco shims under the profile) are refused. Paths are canonicalized
    so 8.3 aliases and junctions still resolve to their real root (#7317).
    """
    norm = _canon_win_path(path)
    for root in _windows_program_roots():
        root_norm = _canon_win_path(root)
        if norm == root_norm or norm.startswith(root_norm + os.sep):
            return True
    return False


# not dot-named: the walks skip dot-dirs, which would hide a model's /tmp write.
# not "tmp": too common in a workspace, and adopting one is what broke the walks.
_SANDBOX_TEMP_DIRNAME = "unsloth-tmp"


def _sandbox_temp_dir(workdir: str) -> str:
    """The scratch directory for a sandboxed child, created when missing.

    Inside the workdir, not the workdir itself: Git for Windows mounts /tmp at
    %TEMP% (the msys2 ``usertemp`` fstab entry), so pointing TEMP at the workdir
    made /tmp its shortest POSIX name and ``pwd`` printed /tmp, leaving the user
    no way to find the real folder (#8892). One level down still sits where the
    listings reach, so a /tmp write is offered exactly as before.

    Falls back to the workdir when the name is unusable, since a TMPDIR that
    does not exist breaks every tempfile call in the child. os.mkdir, never
    os.makedirs, so a workdir deleted mid-call is not silently recreated.
    """
    temp_dir = os.path.join(workdir, _SANDBOX_TEMP_DIRNAME)
    try:
        os.mkdir(temp_dir, 0o700)
    except FileExistsError:
        if not _reusable_sandbox_temp_dir(temp_dir, workdir):
            return workdir
    except OSError:
        return workdir
    return temp_dir


def _is_sandbox_temp_dir(temp_dir: str, workdir: str) -> bool:
    """Whether *temp_dir* is the workdir's own scratch directory.

    Exact stored spelling, because the walks read the name off os.walk: on a
    case-insensitive volume (default APFS, every NTFS) our lowercase probe lands
    on a directory stored as ``TMP``, and realpath does not canonicalise case.
    And the real directory, not a link or junction to one, since os.walk does
    not follow links and the artifacts would land where both walks skip.
    """
    try:
        with os.scandir(workdir) as entries:
            if not any(entry.name == _SANDBOX_TEMP_DIRNAME for entry in entries):
                return False
        # realpath, not islink: a Windows junction is not a link to either.
        if os.path.realpath(temp_dir) != os.path.join(
            os.path.realpath(workdir), _SANDBOX_TEMP_DIRNAME
        ):
            return False
    except OSError:
        return False
    return os.path.isdir(temp_dir)


def _reusable_sandbox_temp_dir(temp_dir: str, workdir: str) -> bool:
    """Whether an existing entry may serve as the scratch directory.

    Identity plus writability, since tempfile abandons an unwritable TMPDIR for
    the platform default. Path accounting asks _is_sandbox_temp_dir instead:
    which directory a segment sits in does not depend on writability.
    """
    return _is_sandbox_temp_dir(temp_dir, workdir) and os.access(temp_dir, os.W_OK | os.X_OK)


def _build_safe_env(workdir: str) -> dict[str, str]:
    """Build a minimal, credential-free environment for sandboxed subprocesses.

    Whitelist-built from scratch (parent env NOT inherited): only PATH/HOME/
    TMPDIR/LANG/TERM/PYTHONIOENCODING/PYTHONPATH (+VIRTUAL_ENV or Windows
    SystemRoot and a minimal PATHEXT) reach the child; all credential vars
    (HF_TOKEN, AWS_*, etc.) are absent. HOME points at the sandbox workdir so SDKs can't read the
    operator's cached creds, and the temp vars at _sandbox_temp_dir just inside
    it. PYTHONPATH carries only the sandbox sitecustomize shim directory.

    PATH starts with the Unsloth interpreter / venv and OS system dirs so
    ``python``/``pip`` stay pinned. On Windows only, Git-for-Windows install
    dirs from the host PATH are appended so bare ``git`` resolves (#7317).
    User-writable host PATH entries (venv, ``node_modules/.bin``, etc.) are
    never inherited — they could shadow auto-safe terminal commands.
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
        # Ahead of System32 and its DOS twins (bare `find` would hit FIND.EXE,
        # not GNU find), behind the interpreter dirs so a Git-shipped
        # python.exe cannot shadow the environment this server runs in.
        path_entries.extend(_windows_bash_userland_dirs())
        path_entries.extend([os.path.join(sysroot, "System32"), sysroot])
    else:
        path_entries.extend(["/usr/local/bin", "/usr/bin", "/bin"])

    # Windows Git installs live outside System32; inherit the dir of the git
    # the HOST shell resolves, but ONLY when it sits under a system install
    # root (Program Files, windir). A user-writable dir (Scoop/Choco shims)
    # is refused: it would let an attacker drop rg.exe/jq.exe beside git and
    # have an auto-approved bare command execute it (#7317).
    git_ext = ""
    if sys.platform == "win32":
        # Append the CANONICAL (realpath) trusted git dir, scanning past any
        # untrusted user shim that sorts first on PATH; the canonical path
        # cannot be retargeted via a junction after the trust check.
        _trusted_git_dir, git_ext = _resolve_trusted_windows_git()
        if _trusted_git_dir:
            path_entries.append(_trusted_git_dir)

    # Deduplicate, preserving order.
    deduped = list(dict.fromkeys(p for p in path_entries if p))

    temp_dir = _sandbox_temp_dir(workdir)
    env = {
        "PATH": os.pathsep.join(deduped),
        "HOME": workdir,
        "TMPDIR": temp_dir,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": "dumb",
        "PYTHONIOENCODING": "utf-8",
        # A GUI backend opens a native window and blocks plt.show() until closed.
        "MPLBACKEND": "Agg",
        # sitecustomize shim: remaps ChatGPT code-interpreter paths (/mnt/data
        # etc.) onto the sandbox CWD; see sandbox_site/sitecustomize.py.
        "PYTHONPATH": _SANDBOX_SITE_DIR,
    }
    if venv:
        env["VIRTUAL_ENV"] = venv
    # Windows needs SystemRoot for Python/subprocess to work.
    if sys.platform == "win32":
        env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
        # Windows tempfile / native SDKs honour TEMP/TMP, not TMPDIR; without
        # these a child falls back to GetTempPath and writes outside the workdir.
        env["TEMP"] = temp_dir
        env["TMP"] = temp_dir
        # Restrict PATHEXT so cwd .BAT/.CMD cannot hijack bare names (#7317).
        pathext = ".EXE;.COM"
        if git_ext and git_ext not in (".EXE", ".COM"):
            # Keep the host git launcher (e.g. a .CMD shim) resolvable.
            pathext += ";" + git_ext
        env["PATHEXT"] = pathext
        # cmd/CreateProcess search cwd before PATH for bare names; disable so
        # a workdir rg.exe/git.exe cannot shadow auto-approved commands.
        env["NoDefaultCurrentDirectoryInExePath"] = "1"
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
    """Env for bypass exec: full host env minus credential vars, with HOME at the
    workdir and TMPDIR just inside it so SDKs cannot read cached creds.

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
    temp_dir = _sandbox_temp_dir(workdir)
    env["HOME"] = workdir
    env["TMPDIR"] = temp_dir
    # Windows tempfile / SDKs honour TEMP/TMP, not TMPDIR; repoint all three so
    # the bypassed tool writes under the per-session sandbox dir on every OS.
    env["TEMP"] = temp_dir
    env["TMP"] = temp_dir
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
    env.setdefault("MPLBACKEND", "Agg")
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


# System32\bash.exe and the WindowsApps shim are the WSL launcher, which runs the
# command inside the WSL filesystem: the sandbox workdir and the blocklist's path
# checks would both apply to the wrong tree. Only a native Win32 bash is usable.
_WSL_BASH_MARKERS = ("\\system32\\", "\\windowsapps\\")
# os.path.join, not a literal `Git\bin\...`, so the probe uses the host
# separator and the resolver stays testable on POSIX.
_WIN_BASH_RELATIVE = (
    os.path.join("Git", "bin", "bash.exe"),
    os.path.join("Git", "usr", "bin", "bash.exe"),
)


def _is_trusted_windows_bash(path: str) -> bool:
    """True when ``path`` is a native Win32 bash under a system install root.

    The shell runs every sandboxed command, so an untrusted one defeats the
    PATH / PATHEXT / NoDefaultCurrentDirectoryInExePath hardening in
    _build_safe_env: a bash.exe in a user-writable dir (Scoop, a per-user Git,
    a project checkout) would execute attacker-controlled code for a command
    that already passed the blocklist. Same Program Files trust boundary as
    the sandbox git PATH entry (#7317), and fails closed.
    """
    lowered = path.replace("/", "\\").lower()
    if any(marker in lowered for marker in _WSL_BASH_MARKERS):
        return False
    return _is_trusted_windows_program_dir(os.path.dirname(path))


@functools.lru_cache(maxsize = 1)
def _windows_bash() -> "str | None":
    """Path to a trusted native Win32 bash, or None when only cmd is available."""
    candidates: list[str] = []
    for root in _windows_program_roots():
        candidates.extend(os.path.join(root, relative) for relative in _WIN_BASH_RELATIVE)
    # shutil.which returns only the FIRST PATH match, which may be an untrusted
    # shim; scan the rest too so it cannot mask a trusted Git install (#7317).
    primary = shutil.which("bash")
    if primary:
        candidates.append(primary)
    for entry in (os.environ.get("PATH") or "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if entry and os.path.isabs(entry):
            candidates.append(os.path.join(entry, "bash.exe"))
    for candidate in candidates:
        if os.path.isfile(candidate) and _is_trusted_windows_bash(candidate):
            return candidate
    return None


def _windows_bash_userland_dirs() -> list[str]:
    """Trusted dirs holding the resolved bash and the POSIX tools beside it.

    ``bash -c`` is non-login, so /etc/profile never runs and Git for Windows'
    ``usr\\bin`` stays off PATH, leaving ls / cat / grep "command not found".
    bash.exe ships under ``Git\\bin`` or ``Git\\usr\\bin``, so both parents are
    probed. Candidates clear the same Program Files trust boundary as the git
    entry (#7317) and are canonicalised against junctions. Fails closed: no
    trusted bash, no entries, PATH unchanged.
    """
    bash = _windows_bash()
    if not bash:
        return []
    bin_dir = os.path.dirname(bash)
    candidates = [bin_dir]
    for root in (os.path.dirname(bin_dir), os.path.dirname(os.path.dirname(bin_dir))):
        if root:
            candidates.append(os.path.join(root, "usr", "bin"))
    dirs: list[str] = []
    for candidate in candidates:
        if not os.path.isdir(candidate) or not _is_trusted_windows_program_dir(candidate):
            continue
        real = os.path.realpath(candidate)
        if real not in dirs:
            dirs.append(real)
    return dirs


def _shell_is_posix() -> bool:
    """True when the shell that will run a command parses POSIX syntax."""
    return sys.platform != "win32" or _windows_bash() is not None


def _get_shell_cmd(command: str) -> list[str]:
    """Return the platform-appropriate shell invocation for a command string."""
    if sys.platform == "win32":
        # why: the model is told this tool is bash and writes bash. cmd /c runs
        # only the first line of a multi-line command, keeps single quotes
        # literal, and does not understand bash quoting, so a correct script
        # silently half-executes. Use a real bash when the host has one.
        bash = _windows_bash()
        if bash:
            return [bash, "-c", command]
        return ["cmd", "/c", command]
    return ["bash", "-c", command]


# Per-session working directories so each chat thread gets its own sandbox.
# Falls back to ~/studio_sandbox/_default for callers without a session_id.
_workdirs: dict[str, str] = {}
# Sessions with a tool call in flight. Deleting a chat unlinks its workdir, and
# a process whose cwd has been removed fails every relative write with ENOENT.
_active_sessions: "dict[str, int]" = {}
# Deletions that arrived mid-call: the thread has gone from history, so nothing
# would ask for the folder again. Keyed like the above and holding every exact
# id that folded onto the key, since each can be its own directory.
_pending_removals: "dict[str, dict[str, bool]]" = {}
_active_sessions_lock = threading.Lock()
# Sessions whose sandbox is being removed right now. A start for one of these
# waits on the condition rather than on the lock, so only that chat is held up.
_removing_sessions: "set[str]" = set()
_sessions_free = threading.Condition(_active_sessions_lock)


def _session_key(session_id: "str | None") -> str:
    """Lifecycle key for a session id.

    Case-folded: two ids differing only in case are one directory on Windows and
    on a default macOS volume, and keying them apart let a delete land while the
    other chat was running a tool in there.
    """
    return (session_id or _ANON_KEY).casefold()


@contextlib.contextmanager
def _session_in_flight(session_id: "str | None"):
    key = _session_key(session_id)
    with _sessions_free:
        # A removal for this session runs with the lock released, so a call
        # starting in that window would be handed the directory it is about to
        # rename away. Only this session waits; every other chat is untouched.
        while key in _removing_sessions:
            _sessions_free.wait()
        _active_sessions[key] = _active_sessions.get(key, 0) + 1
    try:
        yield
    finally:
        pending: "dict[str, bool]" = {}
        with _sessions_free:
            if _active_sessions.get(key, 0) <= 1:
                _active_sessions.pop(key, None)
                pending = _pending_removals.pop(key, {})
                if pending:
                    _removing_sessions.add(key)
            else:
                _active_sessions[key] -= 1
        # Not `if not pending: return` -- a return here swallows whatever the
        # tool raised, and the caller reports it as an unknown tool instead.
        if pending:
            # Outside the lock: deciding whether the tree holds files can take
            # seconds, and no other chat could start a call meanwhile. This
            # session stays closed, so nothing starts in the directory removed.
            try:
                for pending_id, pending_files in pending.items():
                    if _thread_exists(pending_id, unknown = True):
                        # Recreated while that call ran: this delete belongs to
                        # the chat that went, the folder is the new one's now.
                        continue
                    _remove_session_sandbox_locked(pending_id, pending_files)
            finally:
                with _sessions_free:
                    _removing_sessions.discard(key)
                    _sessions_free.notify_all()


# Non-matching session_ids collapse to ``_invalid`` to block cross-session escapes.
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9_\-]{1,64}\Z")
# Reserved on Windows even as a directory name, and an API caller picks this id.
_WINDOWS_DEVICE_NAMES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)
_PROJECT_SESSION_PREFIX = "project-"


def _usable_session_id(session_id: str) -> bool:
    """Matches the id charset and is a name every OS can hold as a directory."""
    if not _SESSION_ID_RE.match(session_id):
        return False
    return session_id.split(".")[0].lower() not in _WINDOWS_DEVICE_NAMES


def _orphan_records_dir() -> str:
    """Where a preserved project workspace's path is written down.

    One small file per project, named by its id: the row that knew the path is
    gone, and a workspace the user pointed somewhere custom cannot be derived
    from anything else.
    """
    try:
        from utils.paths.storage_roots import studio_root
        return os.path.join(str(studio_root()), "orphaned-projects")
    except Exception:
        # Only if the studio home cannot be resolved at all: beside the sandbox
        # root, whose parent an administrator may have made read-only.
        return os.path.join(
            os.path.dirname(os.path.realpath(sandbox_root())),
            "orphaned-projects",
        )


# One record per kept folder, named by kind and a digest of the id, with the
# exact id inside: chats and projects are different tables and can carry the
# same client-supplied id, which is not always one a filename can hold.
_ORPHAN_CHAT = "chat"
_ORPHAN_PROJECT = "project"
# A pass reads every record: they are a few hundred bytes each, one per deleted
# folder still kept, and a cap here would strand everything past it for good.
_MAX_ORPHAN_RECORDS = 10_000


def _orphan_record_name(kind: str, record_id: str) -> str:
    """The filename a record is kept under."""
    digest = hashlib.sha256(record_id.encode("utf-8", "surrogatepass")).hexdigest()[:32]
    return f"{kind}-{digest}"


def _read_orphan_record(kind: str, record_id: str) -> "dict | None":
    """One record by key, without listing the directory."""
    import json as _json

    path = os.path.join(_orphan_records_dir(), _orphan_record_name(kind, record_id))
    try:
        with open(path, encoding = "utf-8") as fh:
            record = _json.loads(fh.read(4096).strip())
    except (OSError, ValueError, TypeError):
        return None
    return record if isinstance(record, dict) and record.get("path") else None


def record_orphaned_project(
    project_id: str,
    workspace: str,
    pending_delete: bool = False,
    root_path: "str | None" = None,
) -> None:
    """Remember where a deleted project's kept workspace lives.

    Written whether or not files were to be deleted: the row that knew the path
    has gone either way, and a chat forked out of the project still shows cards
    for it. ``pending_delete`` is what separates "keep this, just make it
    reachable" from "the user asked for it, finish when nothing is using it".
    """
    if not project_id or not workspace:
        return
    _write_orphan_record(
        _ORPHAN_PROJECT,
        project_id,
        {
            "path": os.path.realpath(workspace),
            # The whole workspace is what the delete dialog offers, and the sandbox
            # is one directory inside it.
            "rootPath": os.path.realpath(root_path) if root_path else None,
            "pendingDelete": bool(pending_delete),
        },
    )


def _write_orphan_record(kind: str, record_id: str, record: dict) -> None:
    """One small JSON file per kept folder, under its kind and id."""
    import json as _json

    record = {**record, "id": record_id, "chat": kind == _ORPHAN_CHAT}
    try:
        os.makedirs(_orphan_records_dir(), exist_ok = True)
        name = _orphan_record_name(kind, record_id)
        with open(os.path.join(_orphan_records_dir(), name), "w", encoding = "utf-8") as fh:
            fh.write(_json.dumps(record))
    except OSError:
        logger.warning("Could not record kept folder for %s", record_id)


def record_kept_sandbox(session_id: str) -> None:
    """Remember a chat sandbox kept because a fork still shows its files.

    The user asked for those files and the chat is gone, so nothing would come
    back to that folder: the fork's own delete finishes the job instead.
    """
    if not session_id:
        return
    try:
        workdir = os.path.realpath(resolve_sandbox_workdir(session_id))
    except OSError:
        return
    if not os.path.isdir(workdir):
        return
    _write_orphan_record(
        _ORPHAN_CHAT,
        session_id,
        {"path": workdir, "rootPath": None, "pendingDelete": True},
    )


def forget_orphaned_project(project_id: str, is_chat: bool = False) -> None:
    """Drop the record once the folder has gone."""
    if not project_id:
        return
    kind = _ORPHAN_CHAT if is_chat else _ORPHAN_PROJECT
    try:
        os.unlink(os.path.join(_orphan_records_dir(), _orphan_record_name(kind, project_id)))
    except OSError:
        pass


def list_orphaned_projects() -> "list[tuple[str, str, str | None, bool, bool]]":
    """Every recorded (id, folder, project root, pending, is a chat) still there."""
    import json as _json

    records = []
    try:
        names = sorted(os.listdir(_orphan_records_dir()))
    except OSError:
        return records
    if len(names) > _MAX_ORPHAN_RECORDS:
        logger.warning(
            "%d kept-folder records; reading the first %d", len(names), _MAX_ORPHAN_RECORDS
        )
        names = names[:_MAX_ORPHAN_RECORDS]
    for name in names:
        try:
            with open(os.path.join(_orphan_records_dir(), name), encoding = "utf-8") as fh:
                raw = fh.read(4096).strip()
        except OSError:
            continue
        try:
            record = _json.loads(raw)
            path, pending = record["path"], bool(record.get("pendingDelete"))
            root = record.get("rootPath") or None
            is_chat = bool(record.get("chat"))
            record_id = record["id"]
        except (ValueError, TypeError, KeyError):
            continue
        if _recorded_workspace_remains(path, root):
            records.append((record_id, path, root, pending, is_chat))
        else:
            forget_orphaned_project(record_id, is_chat)
    return records


def _recorded_workspace_remains(workspace: str, root: "str | None") -> bool:
    """Whether anything a record names is still on disk.

    The project root as well as its sandbox: a delete that removed the sandbox
    and stopped at a locked file elsewhere leaves the rest of the workspace,
    and dropping the record here loses both the path and the user's request.
    """
    for path in (workspace, root):
        if path and os.path.isdir(path) and not os.path.islink(path):
            return True
    return False


def forget_orphaned_project_if_gone(
    project_id: str,
    workspace: str,
    root: "str | None",
    is_chat: bool = False,
) -> None:
    """Drop the record only once the folder has gone."""
    if _recorded_workspace_remains(workspace, root):
        logger.warning("Workspace for %s is still there; left pending", project_id)
        return
    forget_orphaned_project(project_id, is_chat)


def _delete_recorded_workspace(project_id: str, workspace: str, root: "str | None") -> None:
    """Remove a recorded workspace the way the immediate delete would.

    Always through the storage helper, whose folder-name and denied-path checks
    decide what may go: a record is a file on disk, and a stale or edited one
    naming an unrelated directory must not become an rmtree of it. Without a
    recorded root the workspace's own parent is offered, which is what the
    default layout puts the sandbox in; anything else it refuses, and the
    record stays pending rather than being deleted on our own authority.
    """
    from storage.studio_db import delete_project_workspace

    target = root or os.path.dirname(os.path.realpath(workspace))
    delete_project_workspace({"id": project_id, "rootPath": target})


def collect_orphaned_project_workspaces() -> None:
    """Finish the workspace deletes the user asked for.

    Only records marked pending: one kept merely so a fork's cards resolve is
    not something anybody asked to remove. Skipped while a tool call is still
    running in there, or while a chat still shows its files.
    """
    from storage.studio_db import sandbox_is_referenced_elsewhere
    for record_id, workspace, root, pending, is_chat in list_orphaned_projects():
        if not pending:
            continue
        try:
            session = record_id if is_chat else project_session_id(record_id)
            # This runs minutes after the row went, and the id is the client's
            # to reuse: a chat or a project created since owns that folder, and
            # a card of its own may not be stored yet.
            recreated = (
                _thread_exists(record_id, unknown = True)
                if is_chat
                else live_project_owns(record_id, workspace, root)
            )
            if recreated:
                logger.info("Kept %s: it was created again", record_id)
                continue
            if not wait_for_sessions_idle([session], timeout = 0.0):
                continue
            if sandbox_is_referenced_elsewhere(session):
                continue
            if is_chat:
                # Its own ownership checks decide whether that directory is
                # ours to take, exactly as the chat's own delete would.
                remove_session_sandbox(session, delete_files = True)
            else:
                _delete_recorded_workspace(record_id, workspace, root)
            # A locked file on Windows, or a network volume having a bad
            # moment: the record stays so the next launch tries again.
            forget_orphaned_project_if_gone(record_id, workspace, root, is_chat)
        except Exception:  # noqa: BLE001 - a stuck record must not break a delete
            logger.warning("Could not collect workspace for %s", record_id, exc_info = True)


def finish_workspace_delete_when_idle(
    project_id: str, timeout: float = 600.0
) -> "threading.Thread":
    """Wait out the tool call still using a workspace, then delete it.

    The delete dialog promised those files would go, and nothing else would
    come back to them: the collection otherwise runs only on the next delete.
    """

    def _wait_and_collect() -> None:
        session = project_session_id(project_id)
        wait_for_sessions_idle([session], timeout = timeout)
        collect_orphaned_project_workspaces()

    thread = threading.Thread(
        target = _wait_and_collect,
        name = "workspace-delete",
        daemon = True,
    )
    thread.start()
    return thread


def _recorded_project_workdir(project_id: str) -> "str | None":
    """The kept workspace of a deleted project, wherever the user put it.

    By key: a resolve happens on every tool call for such a project, and no
    number of other records may keep it from finding its own.
    """
    record = _read_orphan_record(_ORPHAN_PROJECT, project_id)
    if not record:
        return None
    path = record["path"]
    # Only a sandbox still there: a record kept alive by the rest of its
    # workspace names a directory nothing can be served from.
    return path if os.path.isdir(path) else None


def _orphaned_project_workdir(project_id: str) -> "str | None":
    """A deleted project's workspace, when its files were kept.

    The record answers for any id, since it is keyed by a digest. Only the
    guess below builds a directory name, so only that needs an id a filename
    can hold.
    """
    recorded = _recorded_project_workdir(project_id)
    if recorded:
        return recorded
    if not _usable_session_id(project_id):
        return None
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "-", project_id)[:8].strip("-_") or "project"
    try:
        from utils.paths import project_workspaces_root
        root = str(project_workspaces_root())
        names = sorted(os.listdir(root))[:_MAX_SNAPSHOT_DIRS]
    except Exception:
        return None
    for entry in names:
        if not entry.endswith(f"-{suffix}"):
            continue
        candidate = os.path.join(root, entry, "sandbox")
        if os.path.isdir(candidate) and not os.path.islink(candidate):
            return os.path.realpath(candidate)
    return None


def _thread_exists(thread_id: str, unknown: bool = False) -> bool:
    """Whether a chat of the user's is stored under this exact id.

    ``unknown`` is what a check that could not be made returns: a caller about
    to delete files passes True, so a database hiccup keeps them, while one
    merely routing a call passes False and treats the id as a project's.
    """
    try:
        from storage.studio_db import get_chat_thread
        return get_chat_thread(thread_id) is not None
    except Exception:  # noqa: BLE001 - see `unknown`
        return unknown


def live_project_owns(
    project_id: str,
    workspace: str,
    root: "str | None" = None,
) -> bool:
    """Whether a project with this id is the one those folders belong to.

    A reused id is not the same workspace: the default root carries the
    project's name, and renaming a project leaves its root where it was. A
    folder the live row does not own is still the deleted project's, and still
    the one the user asked to remove.
    """
    try:
        from storage.studio_db import get_chat_project
        project = get_chat_project(project_id)
    except Exception:  # noqa: BLE001 - an unanswerable check keeps the files
        return True
    if not project:
        return False
    live = [project.get("rootPath"), project.get("sandboxPath")]
    theirs = {os.path.realpath(path) for path in live if path}
    for path in (workspace, root):
        if not path:
            continue
        resolved = os.path.realpath(path)
        if any(resolved == one or resolved.startswith(one + os.sep) for one in theirs):
            return True
    return False


def _project_exists(project_id: str) -> bool:
    """Whether a project of the user's is stored under this exact id."""
    try:
        from storage.studio_db import get_chat_project
        return get_chat_project(project_id) is not None
    except Exception:  # noqa: BLE001 - a storage hiccup must not delete files
        return True


def _project_workdir_for(session_id: "str | None") -> "str | None":
    """The project workspace a session id names, if it names one.

    The prefixed id can be longer than a directory name may be, or carry a
    character one may not: it is the project part that has to be usable, and
    the workspace path comes from the row rather than from the id.
    """
    if not session_id:
        return None
    if not _usable_session_id(session_id) and not session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    return _get_project_workdir(session_id)


def _get_project_workdir(session_id: str) -> str | None:
    if not session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    project_id = session_id[len(_PROJECT_SESSION_PREFIX) :]
    if not project_id:
        return None
    if _thread_exists(session_id):
        # An API client picks its own thread ids, and a chat called this is a
        # chat: sharing the project's workspace would run its tool calls in
        # there and leave its delete refusing to remove anything.
        return None
    try:
        from storage.studio_db import ensure_chat_project_workspace
        project = ensure_chat_project_workspace(project_id)
    except Exception:
        logger.warning("Failed to resolve project sandbox for %s", session_id, exc_info = True)
        return None
    if not project:
        # The project is gone but a chat forked out of it still shows cards for
        # this sandbox, and the workspace was kept for exactly that: the record
        # answers for any id, and the folder-name guess needs a usable one.
        return _orphaned_project_workdir(project_id)
    root_path = project.get("rootPath")
    sandbox_path = project.get("sandboxPath")
    if not root_path or not sandbox_path:
        return None
    root_real = os.path.realpath(root_path)
    sandbox_real = os.path.realpath(sandbox_path)
    if sandbox_real != root_real and not sandbox_real.startswith(root_real + os.sep):
        return None
    return sandbox_real


# Dropped in every session directory we create. The root can be an existing
# shared folder the user pointed us at, and a chat id can name something already
# in there; this is the only evidence the directory is ours to delete.
_SANDBOX_MARKER = ".unsloth_sandbox"

# Reserved: a directory named this way belongs to the id that hashes to it, and
# never to a chat that happens to be called the same thing.
_DERIVED_PREFIX = "_id-"

# The directories a call with no usable session id runs in. A chat whose id is
# one of these gets a derived name instead of sharing that folder, and with it
# every session-less call's files and the delete that takes them.
_FALLBACK_NAMES = frozenset({"_default", "_invalid"})

# The cache and lifecycle key for a call with no session id. Holds a character
# the id charset forbids, so a chat cannot key onto the same entry and be handed
# the session-less sandbox out of the cache.
_ANON_KEY = "\x00_default"


def _sandbox_name(session_id: str) -> str:
    """The directory name for an id.

    An id the filesystem cannot hold gets a name derived from it rather than a
    shared bucket: those ids come from API clients, and one bucket meant every
    such chat could read and delete every other one's files.
    """
    if (
        _usable_session_id(session_id)
        and not session_id.startswith(_DERIVED_PREFIX)
        and session_id not in _FALLBACK_NAMES
    ):
        return session_id
    # An id that already looks derived is derived too, or it would land on the
    # directory of whichever unusable id hashes to it. surrogatepass because a
    # lone surrogate reaches here from JSON and from surrogateescape alike.
    encoded = session_id.encode("utf-8", "surrogatepass")
    return _DERIVED_PREFIX + hashlib.sha256(encoded).hexdigest()[:16]


def _preserve_foreign_marker(workdir: str, name: "str | None" = None) -> None:
    """Move aside a marker-named entry that this migration did not write.

    This name was not reserved before the change, so a chat that wrote its own
    .unsloth_sandbox has a real file there, and a short note like "notes" reads
    as a perfectly good session name. Only the exact marker this move is about
    to write is left alone; everything else is renamed, not removed.
    """
    marker = os.path.join(workdir, _SANDBOX_MARKER)
    if not os.path.lexists(marker):
        return
    if name is not None and _marker_owner(workdir) == _sandbox_name(name):
        return  # already ours, from a move that ran before
    for n in range(1, 100):
        kept = f"{marker}.saved" if n == 1 else f"{marker}.saved-{n}"
        if not os.path.lexists(kept):
            try:
                os.rename(marker, kept)
            except OSError:
                logger.warning("Could not preserve %s", marker)
            return


def _mark_sandbox(workdir: str, session_id: str) -> None:
    """(Re)write the marker. Never through a link.

    The file sits where tool code runs, so one replaced by a symlink would send
    this write to whatever it points at and truncate it.
    """
    marker = os.path.join(workdir, _SANDBOX_MARKER)
    try:
        if os.path.islink(marker):
            os.unlink(marker)
        flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(marker, flags, 0o600)
        try:
            os.write(fd, _sandbox_name(session_id).encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def _marker_owner(workdir: str) -> "str | None":
    """The id this directory was created for, when it says.

    Anything that is not an id reads as no owner rather than as somebody else's:
    a tool writing over this file would otherwise send its own chat to a fresh
    directory on the next launch, leaving its files behind.
    """
    marker = os.path.join(workdir, _SANDBOX_MARKER)
    if os.path.islink(marker):
        return None  # not a marker we wrote, and not one to follow
    try:
        with open(marker, encoding = "utf-8") as fh:
            owner = fh.read(256).strip()
    except (OSError, UnicodeDecodeError):
        return None
    return owner if owner and _usable_session_id(owner) else None


def _session_dir(root: str, session_id: str) -> str:
    """The directory for this exact id.

    Two ids differing only in case are one name on Windows and on a default
    macOS volume. The marker says which id made the directory, and anyone else
    gets one of their own rather than sharing files that either chat's deletion
    would then remove. A directory already sitting in a root the user pointed
    us at is nobody's sandbox: it is stepped around rather than run in, so no
    tool can write a marker into it and make it look like ours.
    """
    name = _sandbox_name(session_id)
    plain = os.path.join(root, name)
    # A link is never ours, whatever it points at: claiming through one writes
    # the marker into a directory somebody else made, inside the root or not.
    if not os.path.islink(plain):
        owner = _marker_owner(plain)
        if owner == name:
            return plain
        if owner is None and not (os.path.isdir(plain) and not _root_is_ours()):
            return plain
    # Taken by something that is not ours, so this chat gets a name of its own.
    return os.path.join(root, f"{name}-{_name_suffix(session_id)}")


def _name_suffix(session_id: str) -> str:
    """A short stable tail, so the same chat lands in the same directory.

    surrogatepass for the same reason _sandbox_name uses it: an id with a lone
    surrogate reaches here on the collision path, and a strict encode would
    raise rather than step aside.
    """
    encoded = session_id.encode("utf-8", "surrogatepass")
    return hashlib.sha256(encoded).hexdigest()[:8]


# Serialises the pick-then-create below. Two case-variant ids racing on a
# case-insensitive volume could otherwise both see an unowned name and take it.
_assign_lock = threading.Lock()


# Directories this run created and claimed. Not a record of ownership beyond
# this process: it is what lets a marker a tool removed be written again.
_claimed_here: "set[str]" = set()


def _claim_sandbox(workdir: str, session_id: str) -> bool:
    """Write the marker if nobody has, and report whether this id owns it.

    O_EXCL, so of two processes creating the same directory exactly one claims
    it and the other is told to go elsewhere.
    """
    marker = os.path.join(workdir, _SANDBOX_MARKER)
    name = _sandbox_name(session_id)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return _marker_owner(workdir) == name
    except OSError:
        return False
    try:
        os.write(fd, name.encode("utf-8"))
    finally:
        os.close(fd)
    _claimed_here.add(workdir)
    return True


def _ensure_session_dir(root: str, session_id: str) -> str:
    """Create this id's sandbox and claim it, stepping aside on a collision."""
    with _assign_lock:
        workdir = _session_dir(root, session_id)
        if not _contained_in_root(workdir, root):
            return _sandbox_fallback(root, "_invalid")
        # Before creating anything: _session_dir can hand back the fallback
        # name, which in a root the user pointed us at can be a directory of
        # theirs, and claiming it would run this chat inside their files.
        if not os.path.exists(workdir):
            # A tree an interrupted migration left marked but not moved in is
            # this chat's, at any root: starting fresh beside it would leave
            # those files unreachable for good.
            stranded = _marked_sandbox_in(root, session_id)
            if stranded:
                try:
                    os.rename(stranded, workdir)
                except OSError:
                    return stranded
                return workdir
        if not _free_for(workdir, _sandbox_name(session_id)) and not _root_is_ours():
            workdir = _free_fallback_dir(root, session_id)
            if workdir is None or not _contained_in_root(workdir, root):
                return _sandbox_fallback(root, "_invalid")
        os.makedirs(workdir, exist_ok = True)
        if _claim_sandbox(workdir, session_id):
            return workdir
        # A marker naming nobody is one a tool wrote over, and at our own root
        # nothing else could have put this directory here.
        if _root_is_ours() and _marker_owner(workdir) is None:
            _mark_sandbox(workdir, session_id)
            return workdir
        # Somebody else's, so take a name of our own rather than run inside it.
        # The fallback is in the same user-controlled root, so it gets the same
        # test: an unowned directory already sitting there is theirs too.
        workdir = _free_fallback_dir(root, session_id)
        if workdir is None or not _contained_in_root(workdir, root):
            return _sandbox_fallback(root, "_invalid")
        os.makedirs(workdir, exist_ok = True)
        _claim_sandbox(workdir, session_id)
        return workdir


def _marked_sandbox_in(root: str, session_id: str) -> "str | None":
    """A directory in *root* whose marker names this chat, if there is one.

    A fresh fallback has a name nothing can recompute, so this is what finds it
    again on a later launch, on a read and on a delete. Bounded: a root the
    user pointed us at can hold a lot of their own folders.
    """
    name = _sandbox_name(session_id)
    # Only names derived from this id, the plain name a half-finished migration
    # staged beside, and their staging names: tool code can write any owner into
    # the marker, so "whoever claims to be me" would hand over another's files.
    candidates = [os.path.join(root, name), *_fallback_candidates(root, session_id)]
    # Listed once: there are 33 candidate names, and a scan each would be 33
    # walks of a root that can hold a folder per chat, on a first call.
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        entries = []
    for candidate in candidates:
        base = os.path.basename(candidate)
        prefix = f"{base}{_STAGING_SUFFIX}"
        staged = [os.path.join(root, e) for e in entries if e.startswith(prefix)]
        for path in [candidate, *staged]:
            if os.path.islink(path) or not os.path.isdir(path):
                continue
            if _marker_owner(path) == name:
                return path
    return None


def _free_fallback_dir(root: str, session_id: str) -> "str | None":
    """A name in *root* this chat may take, or None when they are all spoken for.

    The deterministic one first, so the same chat comes back to the same folder,
    then a fresh one rather than running inside anything already there.
    """
    # One we already made wins, wherever it is: a fresh name every launch would
    # scatter this chat's files.
    ours = _marked_sandbox_in(root, session_id)
    if ours:
        return ours
    for candidate in _fallback_candidates(root, session_id):
        if _free_for(candidate, _sandbox_name(session_id)):
            return candidate
    return None


# How many derived names a chat may try in a root the user pointed us at. Each
# is recomputable, so a later launch finds the folder by name; a random name
# needed a scan, which a big enough root pushes past its bound.
_MAX_FALLBACK_NAMES = 32


def _fallback_candidates(root: str, session_id: str) -> "list[str]":
    """The names this chat may take, in the order it takes them."""
    name = _sandbox_name(session_id)
    stem = f"{name}-{_name_suffix(session_id)}"
    return [os.path.join(root, stem)] + [
        os.path.join(root, f"{stem}-{n}") for n in range(2, _MAX_FALLBACK_NAMES + 1)
    ]


def _free_for(path: str, name: str) -> bool:
    """Whether we may run in *path*: ours already, or not there at all."""
    if os.path.islink(path):
        return False
    if not os.path.exists(path):
        return True
    return _marker_owner(path) == name


def _root_is_ours() -> bool:
    """True unless the root is a directory the user pointed us at.

    A link is theirs as well: `<studio home>/sandbox` pointing somewhere else
    means the directories under it are the user's, and "ours by construction"
    is what lets a delete rename and remove one of them.
    """
    if (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip():
        return False
    try:
        return not os.path.islink(sandbox_root())
    except OSError:
        return False


def _sandbox_is_ours(target: str) -> bool:
    """Ours by construction at our own root, otherwise only with the marker.

    A real file: isfile() follows a link, so a marker symlinked at any existing
    file would make an unrelated directory in a shared root deletable.
    """
    if _root_is_ours():
        return True
    marker = os.path.join(target, _SANDBOX_MARKER)
    return os.path.isfile(marker) and not os.path.islink(marker)


def _legacy_sandbox_root() -> str:
    """Where the sandbox used to live: a third folder in the user's home."""
    return os.path.join(os.path.expanduser("~"), "studio_sandbox")


def sandbox_root() -> str:
    """Root of the per-session tool sandboxes.

    Under the studio home, so UNSLOTH_STUDIO_HOME keeps everything in one place
    instead of leaving a stray ~/studio_sandbox. Falls back to the legacy path
    only if the studio root cannot be resolved.
    """
    override = (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip()
    if override:
        return os.path.expanduser(override)
    try:
        from utils.paths.storage_roots import studio_root
        return os.path.join(str(studio_root()), "sandbox")
    except Exception:
        return _legacy_sandbox_root()


_legacy_sandbox_migrated = False
_legacy_sandbox_lock = threading.Lock()


# Where a cross-filesystem move is assembled. Not a session name, so nothing
# resolves to it while it is being filled.
_STAGING_SUFFIX = ".arriving-"


def _free_move_target(root: str, name: str) -> "str | None":
    """A name in *root* nothing occupies, for a legacy move to land on.

    The resolver's own answer first, so an untouched root keeps the plain name.
    Both derived names can be the user's in a root they pointed us at, and
    returning nothing there stranded the files at the legacy root for good: the
    marker the move writes is what finds this one again.
    """
    candidate = _session_dir(root, name)
    if not os.path.exists(candidate):
        return candidate
    if _marker_owner(candidate) == _sandbox_name(name):
        # This session's folder is already at the destination from an earlier
        # move. A duplicate legacy copy is left alone rather than overwritten,
        # or moved somewhere the user could no longer find it.
        return None
    if _root_is_ours():
        return None  # ours and occupied is a real collision, not a borrowed name
    # The same derived names the request path takes, so whichever runs first the
    # other one resolves to the folder that ended up holding the files.
    for candidate in _fallback_candidates(root, name):
        if not os.path.exists(candidate):
            return candidate
    return None


def _staged_move(source: str, target: str, name: str) -> None:
    """Move one session in, so an interruption cannot look like a collision.

    Across filesystems shutil.move copies as it goes, and a run killed part way
    leaves a partial destination with the original still in place. The next
    launch would read that as a session the new root already has and strand the
    files. Filled under a name nothing resolves to, then renamed, which on one
    filesystem is atomic.
    """
    staging = f"{target}{_STAGING_SUFFIX}{uuid.uuid4().hex[:8]}"
    try:
        shutil.move(source, staging)
    except OSError:
        # Half filled and ours, and the source is still where it was.
        shutil.rmtree(staging, ignore_errors = True)
        raise
    # Marked here, not after the rename: across filesystems the move has already
    # removed the legacy copy, so from this instant the staging tree is the only
    # one there is and a kill before the marker would leave it unfindable.
    _preserve_foreign_marker(staging, name)
    _mark_sandbox(staging, name)
    try:
        os.rename(staging, target)
    except OSError:
        # The move already took the legacy copy, so this tree is the only one
        # and deleting it would lose the files. It is marked, so
        # _marked_sandbox_in finds it; put it back and let the next pass retry.
        try:
            os.rename(staging, source)
        except OSError:
            logger.warning("Sandbox %s left at %s: could not be moved in", name, staging)
        raise
    _mark_sandbox(target, name)


# Bookkeeping only, never held across a move: starting the background pass and
# the sweep. Anything that copies a tree takes that session's own lock below.
_legacy_one_lock = threading.Lock()

# One lock per session being moved: a single lock made a first tool call wait
# out another chat's multi-gigabyte copy, which is what the background pass
# exists to avoid. Bounded by the chats that had a legacy folder.
_legacy_session_locks: "dict[str, threading.Lock]" = {}
_legacy_locks_guard = threading.Lock()


def _legacy_lock_for(name: str) -> threading.Lock:
    """The lock covering this one session's move."""
    with _legacy_locks_guard:
        return _legacy_session_locks.setdefault(name, threading.Lock())


# Where every id the old code could not use as a directory name went. One
# bucket for all of them, which is what this change stops doing, so it is never
# moved up as though it were a chat: several chats' files are in there.
_LEGACY_SHARED_BUCKET = "_invalid"


def _legacy_session_dir(session_id: str) -> "str | None":
    """This session's directory at the legacy root, while one is still there.

    Both names, like the migration itself: a chat from before the upgrade whose
    id starts with the derived prefix kept its folder under the literal id.
    """
    legacy_root = _legacy_sandbox_root()
    names = [_sandbox_name(session_id)]
    if not _usable_session_id(session_id):
        # Before this change an id the filesystem could not hold shared one
        # bucket with every other such chat: read where they are, never moved or
        # deleted, and only for such an id, or a chat reads somebody else's.
        names.append(_LEGACY_SHARED_BUCKET)
    elif session_id not in names and session_id not in _FALLBACK_NAMES:
        # Only the derived-prefix case: an id the old code could hold kept its
        # folder under the literal name while _sandbox_name now hashes it. A
        # fallback name is nobody's chat: every session-less call ran in there.
        names.append(session_id)
    for name in names:
        candidate = os.path.join(legacy_root, name)
        if not os.path.isdir(candidate) or os.path.islink(candidate):
            continue
        # Under this session's move lock, and checked again inside it: the move
        # is a rename, so a path handed back mid-move lists nothing and 404s
        # every card. One that already ran sends the caller to the destination.
        with _legacy_lock_for(name):
            if os.path.isdir(candidate) and not os.path.islink(candidate):
                return candidate
    return None


def _migrate_one_legacy_session(root: str, name: str) -> None:
    """Bring one session up from the legacy root, without waiting for the rest."""
    if _legacy_sandbox_migrated:
        return
    source = os.path.join(_legacy_sandbox_root(), name)
    if os.path.islink(source) or not os.path.isdir(source):
        return
    with _legacy_lock_for(name):
        if not os.path.isdir(source):
            return  # the background pass got there first
        # Through the resolver, like the whole-tree pass: at a shared root the
        # plain name can be the user's own, and stopping here would leave this
        # chat with an empty sandbox and its files stranded at the old root.
        target = _free_move_target(root, name)
        if target is None or not _contained_in_root(target, root):
            return  # nowhere free to land, left for the whole-tree pass
        try:
            os.makedirs(root, exist_ok = True)
            _staged_move(source, target, name)
        except OSError as error:
            logger.warning("Could not move sandbox %s: %s", name, error)


_legacy_background: "threading.Thread | None" = None


def _start_legacy_migration() -> "threading.Thread | None":
    """Carry the rest of the tree up, one pass at a time, off this request."""
    global _legacy_background
    if _legacy_sandbox_migrated:
        return None
    with _legacy_one_lock:
        if _legacy_background is not None and _legacy_background.is_alive():
            return _legacy_background
        _legacy_background = migrate_legacy_sandbox_in_background()
        return _legacy_background


def _migrate_legacy_sandbox(root: str) -> None:
    """Move sessions from ~/studio_sandbox into the studio home, once.

    Those files are the user's, so they move rather than being dropped. A
    session already present at the new root wins and its legacy copy is left
    alone, so nothing is silently overwritten.
    """
    global _legacy_sandbox_migrated
    if _legacy_sandbox_migrated:
        return
    # Flagged only once the move is done: setting it first let a concurrent
    # call create the destination, which then read as a collision.
    with _legacy_sandbox_lock:
        if _legacy_sandbox_migrated:
            return
        # Only when nothing movable is left: a file locked on Windows is
        # retryable, and one attempt strands it once the destination exists.
        if _migrate_legacy_sandbox_locked(root):
            _legacy_sandbox_migrated = True


def _migrate_legacy_sandbox_locked(root: str) -> bool:
    """True when the legacy root holds nothing that could still be moved.

    A collision is not a failure: the new root already has that session, and the
    legacy copy is deliberately left for the user to find.
    """
    legacy = _legacy_sandbox_root()
    try:
        if os.path.realpath(legacy) == os.path.realpath(root) or not os.path.isdir(legacy):
            return True
        os.makedirs(root, exist_ok = True)
        moved = 0
        complete = True
        for name in os.listdir(legacy):
            source = os.path.join(legacy, name)
            # A link is not a sandbox of ours: the move preserves it, and the
            # marker would then be written inside whatever it points at, which
            # is a directory outside both roots.
            if os.path.islink(source) or not os.path.isdir(source):
                continue
            if name == _LEGACY_SHARED_BUCKET:
                continue  # several chats' files, not one chat's
            # The same choice the request path makes: at a shared root both
            # derived names can be the user's, and skipping here while still
            # reporting the pass complete stranded that chat's files for good.
            target = _free_move_target(root, name)
            if target is None or not _contained_in_root(target, root):
                continue
            try:
                with _legacy_lock_for(name):
                    if not os.path.isdir(source) or os.path.exists(target):
                        continue  # a request path moved it while we waited
                    _staged_move(source, target, name)
                moved += 1
            except OSError as error:
                # A file locked on Windows is retryable, so this run reports
                # itself unfinished and the next launch tries again.
                complete = False
                logger.warning("Could not move sandbox %s: %s", name, error)
        if moved:
            logger.info("Moved %d chat sandbox folder(s) from %s to %s", moved, legacy, root)
        # Empty only: a leftover is a collision the user should still find.
        try:
            os.rmdir(legacy)
        except OSError:
            pass
        return complete
    except Exception as error:  # noqa: BLE001 - startup must survive this
        logger.warning("Sandbox migration skipped: %s", error)
        return False


def _sandbox_fallback(
    root: str,
    name: str,
    create: bool = False,
) -> str:
    """``_default`` / ``_invalid`` under the root, contained like any session.

    They are ordinary directories in a writable sandbox, so one replaced by a
    symlink would otherwise become the root every unchecked request reads from.
    Dropping that link is only ours to do at our own root; in a directory the
    user pointed us at, the entry is theirs and a fresh one is used instead.
    """
    owner = _sandbox_name(name)  # reserved, so it is never a chat's own name
    path = os.path.join(root, name)
    if os.path.islink(path):
        if _root_is_ours():
            try:
                os.unlink(path)
                return path
            except OSError:
                pass
    elif _root_is_ours() or not os.path.exists(path) or _marker_owner(path) == owner:
        return path
    # In a root the user pointed us at, a directory already sitting under this
    # name is theirs: a call with no session id would otherwise run in it. The
    # one we made instead is remembered, so it is not a new one every call.
    stem = f"{name}_{_name_suffix(name)}"
    candidates = [os.path.join(root, stem)] + [
        os.path.join(root, f"{stem}-{n}") for n in range(2, _MAX_FALLBACK_NAMES + 1)
    ]
    if not create:
        for made in candidates:
            if not os.path.islink(made) and _marker_owner(made) == owner:
                return made
        return _nothing_to_serve(name)
    for made in candidates:
        # exist_ok alone would take a directory of theirs that happens to carry
        # this name, and follow a link out of the root to run wherever it
        # points, so the entry has to be free and the claim has to succeed.
        if not _free_for(made, owner):
            continue
        try:
            os.makedirs(made, exist_ok = True)
        except OSError:
            continue
        if _claim_sandbox(made, name):
            return made
    return _nothing_to_serve(name)


# Where a read is sent when the chat owns nothing. Outside every sandbox root
# and inside a directory this process makes and empties, so a listing is empty
# and a download is a 404 rather than whatever the user keeps at that name.
_NOTHING_ROOT = None
_nothing_lock = threading.Lock()


def _nothing_to_serve(name: str) -> str:
    """A path that exists nowhere the user keeps files.

    The name is derived first: callers pass the id straight from the request,
    and an absolute one like ``/etc`` would make os.path.join drop the root it
    was given and hand back a directory of the system's.
    """
    global _NOTHING_ROOT
    leaf = _sandbox_name(name)
    with _nothing_lock:
        if _NOTHING_ROOT is None or not os.path.isdir(_NOTHING_ROOT):
            try:
                _NOTHING_ROOT = tempfile.mkdtemp(prefix = "unsloth-unowned-")
            except OSError:
                _NOTHING_ROOT = os.path.join(tempfile.gettempdir(), "unsloth-unowned")
        root = _NOTHING_ROOT
    resolved = os.path.join(root, leaf)
    # Belt and braces: a derived name cannot escape, and neither may anything
    # else that reaches here.
    return resolved if _contained_in_root(resolved, root) else root


def _contained_in_root(workdir: str, root: str) -> bool:
    """Whether a resolved session path is still inside the sandbox root.

    Applied to cached paths too: a directory replaced by a symlink after it was
    cached would otherwise keep serving from wherever it now points.
    """
    try:
        resolved, base = os.path.realpath(workdir), os.path.realpath(root)
        # commonpath, not a prefix test: a filesystem root already ends in a
        # separator, and appending another failed every real session path.
        return resolved != base and os.path.commonpath([resolved, base]) == base
    except (OSError, ValueError):
        return False


def _owned_by_session(workdir: str, session_id: str) -> bool:
    """Whether this session may read *workdir*, for a caller that creates nothing.

    ``_ensure_session_dir`` claims or steps aside; a read has to decide on what
    is already there, and the name it was given can be somebody else's too.
    """
    owner = _marker_owner(workdir)
    if owner is not None:
        return owner == _sandbox_name(session_id)
    # No marker, so the name is the only evidence, and `Foo` and `foo` are one
    # directory on Windows and on a default macOS volume. The delete path has
    # always asked this; a read that did not let the other chat's files through.
    return _root_is_ours() and os.path.basename(workdir) == _sandbox_name(session_id)


def _get_workdir(session_id: str | None = None) -> str:
    """Return a per-session sandbox dir at mode 0o700."""
    global _workdirs
    key = session_id or _ANON_KEY
    cached = _workdirs.get(key)
    if cached is not None and not os.path.isdir(cached):
        cached = None
    if cached is not None and not _get_project_workdir(session_id or ""):
        # The same checks a fresh resolve makes: the entry can have been
        # renamed and replaced with a link to another chat's directory since,
        # and containment alone accepts that.
        root_now = sandbox_root()
        # Tool code runs in here and can delete the marker. For a directory
        # this run claimed it is written again, rather than read as somebody
        # else's, which would strand the files in it and start the chat anew.
        if (
            session_id
            and cached in _claimed_here
            and not os.path.islink(cached)
            and _contained_in_root(cached, root_now)
            and os.path.isdir(cached)
            and _marker_owner(cached) != _sandbox_name(session_id)
        ):
            # Whatever it says now, this process made this directory for this
            # chat. A tool writing another id into the marker would otherwise
            # send the chat to a new folder and strand what it just wrote.
            _preserve_foreign_marker(cached, session_id)
            _mark_sandbox(cached, session_id)
        if (
            os.path.islink(cached)
            or not _contained_in_root(cached, root_now)
            or (session_id and not _owned_by_session(cached, session_id))
        ):
            cached = None
    if cached is None:
        _workdirs.pop(key, None)
        sandbox_root_path = sandbox_root()
        root_existed = os.path.isdir(sandbox_root_path)
        # The folder may still be at the legacy root right after an upgrade.
        # Only this chat's, so a first tool call never waits on the whole tree:
        # across filesystems that is a copy of every session.
        if session_id:
            # A pre-upgrade chat whose id already starts with the derived
            # prefix kept its folder under the literal id, so that name is
            # tried too. Only a usable one: the rest never named a directory.
            derived = _sandbox_name(session_id)
            if (
                derived != session_id
                and _usable_session_id(session_id)
                and session_id not in _FALLBACK_NAMES
            ):
                _migrate_one_legacy_session(sandbox_root_path, session_id)
            _migrate_one_legacy_session(sandbox_root_path, derived)
        _start_legacy_migration()
        _start_detached_sweep()
        project_workdir = _project_workdir_for(session_id)
        if project_workdir:
            workdir = project_workdir
        elif session_id:
            workdir = _ensure_session_dir(sandbox_root_path, session_id)
        else:
            workdir = _sandbox_fallback(sandbox_root_path, "_default", create = True)
        created = not os.path.isdir(workdir)
        os.makedirs(workdir, exist_ok = True)
        if not project_workdir and not session_id:
            # The fallbacks are directories like any other: claimed, so the next
            # run knows this one is the one we made.
            _claim_sandbox(workdir, "_default")
        # Only a root we just created: the override can name a shared
        # directory, and locking that down would cut off everything else.
        if not root_existed or not (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip():
            try:
                os.chmod(sandbox_root_path, 0o700)
            except OSError:
                pass
        # Only ours: a shared root can already hold a directory with this name,
        # and tightening it would cut off whoever else uses it.
        if created or _sandbox_is_ours(workdir):
            try:
                os.chmod(workdir, 0o700)
            except OSError:
                pass
        _workdirs[key] = workdir
    return _workdirs[key]


def get_sandbox_workdir(session_id: str | None = None) -> str:
    return _get_workdir(session_id)


def resolve_sandbox_workdir(session_id: str | None = None) -> str:
    """Where a session's sandbox would be, without creating it.

    For read-only callers: serving a file must not materialise a directory for
    every id someone asks about.
    """
    if session_id:
        project = _project_workdir_for(session_id)
        if project:
            return project
    root = sandbox_root()
    cached = _workdirs.get(session_id or _ANON_KEY)
    if (
        cached
        and not os.path.islink(cached)
        and _contained_in_root(cached, root)
        and (not session_id or _owned_by_session(cached, session_id))
    ):
        return cached
    if not session_id:
        return _sandbox_fallback(root, "_default")
    # A directory this process made for this chat is this chat's, whatever a
    # tool wrote into the marker since: the check above trusts that file, and
    # without this the files it holds are served from nowhere.
    claimed = _claimed_by_this_run(session_id, os.path.realpath(root))
    if claimed:
        return claimed
    workdir = _session_dir(root, session_id)
    # Same containment _get_workdir applies: a session entry symlinked out of
    # the root would otherwise serve whatever it points at.
    if not _contained_in_root(workdir, root):
        return _sandbox_fallback(root, "_invalid")
    if not os.path.isdir(workdir):
        # A migration that moved the tree but could not rename it into place
        # leaves the only copy under a marked name, at any root.
        ours = _marked_sandbox_in(root, session_id)
        if ours:
            return ours
        # Right after an upgrade the files can still be at the legacy root: the
        # move runs in the background and can take minutes. Read where they are
        # rather than 404 every card in the transcript until it finishes.
        legacy = _legacy_session_dir(session_id)
        if legacy:
            return legacy
    if not _root_is_ours() and not _owned_by_session(workdir, session_id):
        # In a root the user pointed us at this chat can be in a fallback whose
        # name nothing recomputes, and a read that stops here shows an empty
        # sandbox and 404s the file cards already in the transcript.
        ours = _marked_sandbox_in(root, session_id)
        if ours:
            return ours
    if os.path.isdir(workdir) and not _owned_by_session(workdir, session_id):
        # Somebody else's, so this session has nothing here to serve.
        return _nothing_to_serve(session_id)
    return workdir


def migrate_legacy_sandbox_in_background() -> "threading.Thread":
    """Move the legacy sandbox up at startup, off every request.

    Across filesystems this copies every session, which is not something a
    listing or a download can wait on: those run on the event loop.
    """

    def _run() -> None:
        try:
            _migrate_legacy_sandbox(sandbox_root())
        except Exception:  # noqa: BLE001 - best effort, like the rest of this
            logger.debug("legacy sandbox migration failed", exc_info = True)

    thread = threading.Thread(target = _run, name = "sandbox-migrate", daemon = True)
    thread.start()
    return thread


# A name no session resolves to, so a detached tree is inert until it is gone.
_DETACHED_SUFFIX = ".deleting-"
# The exact shape the rename produces. A substring test would also have matched
# a backup of the user's own named report.deleting-old.
_DETACHED_RE = re.compile(r"\A.+\.deleting-[0-9a-f]{8}\Z")


# One worker for every detached tree, rather than a thread per chat: clearing a
# thousand chats would otherwise start a thousand recursive deletes at once.
_delete_queue: "queue.Queue[tuple[str, int]]" = queue.Queue()
# Attempts at one tree, and how long the first wait is (doubled each time).
_MAX_DETACHED_DELETE_TRIES = 5
_DETACHED_RETRY_DELAY = 1.0
_delete_worker: "threading.Thread | None" = None
_delete_worker_lock = threading.Lock()


def _drain_detached_deletes() -> None:
    while True:
        target, tries = _delete_queue.get()
        try:
            shutil.rmtree(target, ignore_errors = True)
            if os.path.exists(target):
                _retry_detached_delete(target, tries)
        finally:
            _delete_queue.task_done()


def _retry_detached_delete(target: str, tries: int) -> None:
    """Queue another attempt at a tree ignore_errors left behind.

    A file held open by a scanner or a process still exiting is transient, and
    on Windows routine. The route has already told the user those files went,
    so waiting for the next launch's sweep is not an answer.
    """
    if tries + 1 >= _MAX_DETACHED_DELETE_TRIES:
        logger.warning("Could not delete %s; the next sweep retries it", target)
        return
    try:
        timer = threading.Timer(
            min(_DETACHED_RETRY_DELAY * 2**tries, 30.0),
            _delete_queue.put,
            [(target, tries + 1)],
        )
        timer.daemon = True
        timer.start()
    except RuntimeError:
        logger.warning("Could not delete %s; the next sweep retries it", target)


def _queue_detached_delete(target: str) -> None:
    """Hand a renamed tree to the sweeper, or delete it here if none can run."""
    global _delete_worker
    with _delete_worker_lock:
        if _delete_worker is None or not _delete_worker.is_alive():
            try:
                _delete_worker = threading.Thread(
                    target = _drain_detached_deletes,
                    name = "sandbox-delete",
                    daemon = True,
                )
                _delete_worker.start()
            except RuntimeError:
                # No thread to be had: better a slow call than a tree under a
                # name nothing resolves to and nothing deletes.
                _delete_worker = None
                shutil.rmtree(target, ignore_errors = True)
                return
    _delete_queue.put((target, 0))


def sweep_detached_sandboxes(root: "str | None" = None) -> None:
    """Finish deletes a previous run was killed part way through.

    The rename is what puts the tree out of reach, so a kill between it and the
    rmtree leaves a full copy of the files nothing resolves to.

    At our own root the name is enough: nothing but this code puts a
    ``.deleting-<hex>`` directory there, and a tool that had removed the marker
    before the delete would otherwise leave the tree unreachable for good. In a
    root the user pointed us at, the marker is still required, since a folder of
    theirs can carry any name.
    """
    base = os.path.realpath(root or sandbox_root())
    try:
        names = [name for name in os.listdir(base) if _DETACHED_RE.match(name)]
    except OSError:
        return
    for name in names:
        target = os.path.join(base, name)
        if os.path.islink(target) or not os.path.isdir(target):
            continue
        if _marker_owner(target) is None and not _root_is_ours():
            continue
        shutil.rmtree(target, ignore_errors = True)


_swept_detached = False


def start_sandbox_recovery() -> "threading.Thread | None":
    """Finish what an interrupted run left: renamed trees and pending deletes."""
    return _start_detached_sweep()


def _start_detached_sweep() -> "threading.Thread | None":
    """Run the sweep once per process, off the call that noticed."""
    global _swept_detached
    with _legacy_one_lock:
        if _swept_detached:
            return None
        _swept_detached = True

    def _sweep() -> None:
        sweep_detached_sandboxes()
        # A workspace the user asked to delete, left pending because a tool was
        # still running in it when the app went away.
        collect_orphaned_project_workspaces()

    thread = threading.Thread(target = _sweep, name = "sandbox-sweep", daemon = True)
    thread.start()
    return thread


def remove_session_sandbox(session_id: str, delete_files: bool = False) -> bool:
    """Drop a deleted chat's sandbox. True when something was removed.

    The chat was the only handle on that directory, so leaving it behind means
    one unreachable folder per chat forever. Empty folders always go; files
    need ``delete_files``, since they are the user's and are downloadable from
    the chat. Project workspaces are shared and have their own delete flow.
    """
    if not session_id:
        return False
    # Only a session that really resolves to a project workspace: an imported
    # chat whose id merely starts with the prefix gets an ordinary directory
    # from _get_workdir, and would otherwise never be cleaned up.
    if session_id.startswith(_PROJECT_SESSION_PREFIX) and _get_project_workdir(session_id):
        # Unless this id has a sandbox of its own: a chat named like a project
        # session had one while its row existed, and the row goes first, so the
        # project would inherit the question and those files stay behind.
        root_here = os.path.realpath(sandbox_root())
        if not _claimed_by_this_run(session_id, root_here) and not _marked_sandbox_in(
            root_here,
            session_id,
        ):
            return False
    # The folder may still be at the legacy root right after an upgrade. This
    # session only, or a delete would sit behind a copy of every chat, and
    # outside the lock below, since it moves a tree and takes its own.
    root_now = sandbox_root()
    _migrate_one_legacy_session(root_now, _sandbox_name(session_id))
    if _usable_session_id(session_id) and _sandbox_name(session_id) != session_id:
        _migrate_one_legacy_session(root_now, session_id)
    _start_legacy_migration()
    # Held across the decision AND the unlink: otherwise a tool can start in
    # between and run in a directory this call then removes.
    key = _session_key(session_id)
    with _sessions_free:
        while key in _removing_sessions:
            _sessions_free.wait()  # another delete for this chat is finishing
        if _active_sessions.get(key, 0) > 0:
            # Queued rather than dropped: the chat is already gone from history,
            # so no later delete or clear would ever name this session again.
            queued = _pending_removals.setdefault(key, {})
            queued[session_id] = delete_files or queued.get(session_id, False)
            return False  # see sandbox_removal_deferred
        # Closed for this chat, then released: deciding whether the sandbox is
        # empty walks the tree, and holding the global lock for that stops tool
        # calls starting in every unrelated chat.
        _removing_sessions.add(key)
    try:
        return _remove_session_sandbox_locked(session_id, delete_files)
    finally:
        with _sessions_free:
            _removing_sessions.discard(key)
            _sessions_free.notify_all()


def session_sandbox_has_files(session_id: str) -> bool:
    """Whether this chat's sandbox still holds files of the user's.

    For a delete that was not offered the choice: the chat was the only way to
    those files, so the caller can offer it afterwards rather than leave a
    folder nothing can reach.
    """
    if not session_id:
        return False
    try:
        target = os.path.realpath(resolve_sandbox_workdir(session_id))
        if not os.path.isdir(target) or not _sandbox_is_ours(target):
            claimed = _claimed_by_this_run(session_id, os.path.realpath(sandbox_root()))
            if not claimed:
                return False
            target = os.path.realpath(claimed)
        return not _holds_no_user_files(target, _sandbox_name(session_id))
    except OSError:
        return False


def _is_spill_artifact(sandbox: str, parent: str, name: str) -> bool:
    """Whether ``parent/name`` is a spill this process wrote, rather than anything else.

    Both halves are checked: the name has to be one `_spill_full_output` generates, and it
    has to sit at the spill root or one scope below it. A link is never one, whatever it
    is called.
    """
    root = os.path.join(sandbox, _SPILL_DIR)
    if parent != root and os.path.dirname(parent) != root:
        return False
    identity, owned = _spill_record(root)
    if identity is None or identity != _spill_identity(root):
        # No record of this directory: it came with the sandbox or was replaced, so
        # everything in it is the user's, however the files are named.
        return False
    return _is_recorded_spill(root, os.path.join(parent, name), owned)


def _holds_no_user_files(target: str, owner: "str | None" = None) -> bool:
    """Whether a sandbox holds nothing but (possibly empty) directories.

    Our own marker does not count, and only while it is still ours: tool code
    runs in there and can write its own content over that file, which is then
    the only copy of it. Bounded like every other walk here, and a tree too big
    to check is not one to remove without being asked.
    """
    budget = _MAX_SNAPSHOT_DIRS
    for parent, dirs, files in os.walk(target):
        # A link to a directory is listed here, not in files, and a tool made
        # it: a sandbox holding one is not empty.
        if any(os.path.islink(os.path.join(parent, name)) for name in dirs):
            return False
        for name in files:
            if parent == target and name in _INTERNAL_SANDBOX_FILES:
                if name != _SANDBOX_MARKER:
                    continue
                marker = _marker_owner(target)
                if marker is not None and owner in (None, marker):
                    continue
            # Unsloth's own, like the marker above: a spill is truncated tool output this
            # process wrote and deliberately kept off the file cards, so counting one as
            # the user's content leaves an unreachable sandbox behind, reported as holding
            # files the user never created. Only the artifacts themselves, by the name
            # `_spill_full_output` generates: the directory is writable, tool code can
            # create anything in it, and a real file there is the user's like any other.
            if _is_spill_artifact(target, parent, name):
                continue
            return False
        budget -= 1
        if budget <= 0:
            return False
    return True


def _claimed_by_this_run(session_id: str, root: str) -> "str | None":
    """The directory this process made for this chat, whatever the marker says.

    Tool code runs in there and can empty that file or write another id into
    it, and neither makes the directory somebody else's: this process wrote the
    marker with O_EXCL and remembers doing it. Put back here, so the ordinary
    routes find it too rather than leaving the files stranded until some later
    call happens to repair it.
    """
    cached = _workdirs.get(session_id)
    if not cached or cached not in _claimed_here:
        return None
    if os.path.islink(cached) or not os.path.isdir(cached):
        return None
    if not _contained_in_root(cached, root):
        return None
    if _marker_owner(cached) != _sandbox_name(session_id):
        _preserve_foreign_marker(cached, session_id)
        _mark_sandbox(cached, session_id)
    return cached


def project_session_id(project_id: str) -> str:
    """The sandbox session a project's chats share."""
    return f"{_PROJECT_SESSION_PREFIX}{project_id}"


def wait_for_sessions_idle(session_ids, timeout: float = 10.0) -> bool:
    """Wait until no tool call is running for these sessions. True if none is.

    Cancelling a generation only sets its event, so the call inside the executor
    is still using its working directory for a moment after.
    """
    keys = {_session_key(session_id) for session_id in session_ids or []}
    if not keys:
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with _active_sessions_lock:
            busy = any(_active_sessions.get(key, 0) > 0 for key in keys)
        if not busy:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def sandbox_removal_deferred(session_id: str) -> bool:
    """Whether this session's removal is queued behind a running tool call.

    The caller reports what it kept, and the answer is not known yet: the call
    still in flight can write a file after the sandbox looked empty, and the
    deferred removal would then keep it with nobody left to say so.
    """
    if not session_id:
        return False
    with _active_sessions_lock:
        return session_id in _pending_removals.get(_session_key(session_id), {})


def _remove_session_sandbox_locked(session_id: str, delete_files: bool) -> bool:
    root = os.path.realpath(sandbox_root())
    claimed = _claimed_by_this_run(session_id, root)
    entry = os.path.join(root, _sandbox_name(session_id))
    if not os.path.islink(entry):
        entry = _session_dir(root, session_id)
        # Neither computed name is ours, so this chat may be in a fallback, or
        # in a directory whose marker a tool removed since we made it.
        if not _owned_by_session(entry, session_id):
            entry = _marked_sandbox_in(root, session_id) or claimed or entry
    # The entry itself, not what it resolves to: a symlink to a sibling passes
    # the check below and would take that chat's files. Drop the link, but only
    # at our own root: in a shared one it is the user's entry, not a sandbox.
    if os.path.islink(entry):
        if not _root_is_ours():
            return False
        try:
            os.unlink(entry)
            return True
        except OSError:
            return False
    target = os.path.realpath(entry)
    if os.path.dirname(target) != root or not os.path.isdir(target):  # contained
        return False
    # Ours by the marker, or because this process is the one that made it.
    ours_here = bool(claimed) and target == os.path.realpath(claimed)
    if not ours_here and not _sandbox_is_ours(target):
        return False
    # Made for a different id: the same directory on a case-insensitive volume,
    # and those files are the other chat's.
    owner = _marker_owner(target)
    if owner is not None and owner != _sandbox_name(session_id):
        return False
    if owner is None and not ours_here and os.path.basename(target) != _sandbox_name(session_id):
        # Nothing says whose this is, and on a case-insensitive volume `foo`
        # and `Foo` are one directory: without the marker the name is the only
        # evidence, and it names the other chat.
        return False
    _workdirs.pop(session_id, None)
    # Resolved BEFORE anything is removed: the record is named by the real path of the
    # spill directory, which cannot be derived once the tree is gone. Without this every
    # deleted chat that ever truncated output leaves one small file behind for good.
    forget_record = _spill_record_path(os.path.join(target, _SPILL_DIR))
    try:
        if delete_files:
            # Renamed while locked, deleted after: every tool start takes this
            # lock, so an rmtree of a large tree in here stops calls in every
            # other chat for as long as it runs.
            detached = f"{target}{_DETACHED_SUFFIX}{uuid.uuid4().hex[:8]}"
            try:
                os.rename(target, detached)
            except OSError:
                shutil.rmtree(target, ignore_errors = True)
                gone = not os.path.isdir(target)
                if gone:
                    _forget_spill_record(forget_record)
                return gone
            _queue_detached_delete(detached)
            _forget_spill_record(forget_record)
            return True
        # Empty means no files of the user's: a tool that only ran mkdir, or
        # deleted what it wrote, leaves directories behind, and the chat record
        # is already gone by the time this runs.
        if not _holds_no_user_files(target, _sandbox_name(session_id)):
            return False
        shutil.rmtree(target, ignore_errors = True)
        gone = not os.path.isdir(target)
        if gone:
            _forget_spill_record(forget_record)
        return gone
    except OSError:
        return False


# edit_file
#
# Without it, changing a file means a whole-file `cat > f <<'EOF'` or
# open(...).write(...): ~7.7k output tokens to rewrite a 500-line file that a
# patch does in ~45, and anything the model fails to retype is lost.
#
# Exact-string replacement, not a unified diff: models corrupt @@ hunk headers
# far more often than they mis-copy a literal snippet, and a bad header patches
# the wrong place instead of failing. A missing or non-unique old_string is a
# hard error naming the match count, so the retry is "add context".

# The whole text is read to find the match, so the cap is on the file.
_EDIT_FILE_MAX_BYTES = _env_int("UNSLOTH_STUDIO_EDIT_FILE_MAX_BYTES", 16 * 1024 * 1024)

# Bounded receipt: lines alone are no bound, since one line of minified JS can
# be the whole file, so characters are capped per line and over the receipt.
_EDIT_FILE_DIFF_LINES = 40
_EDIT_FILE_DIFF_LINE_CHARS = 200
_EDIT_FILE_DIFF_CHARS = 4000
# Lines either side of the first change that are handed to difflib. Diffing the
# whole file would split it into one str per line: 8M of them at the 16MB cap.
_EDIT_FILE_DIFF_WINDOW_LINES = 120


def _edit_file_resolve(
    raw_path: str, session_id: "str | None", disable_sandbox: bool
) -> "tuple[str | None, str]":
    """Resolve the model's path the way python/terminal resolve theirs.

    Same rules as the sitecustomize shim: a code-interpreter habit prefix
    (/mnt/data, /workspace, ...) keeps its suffix under the workdir, everything
    else is relative to it. Containment is checked on the realpath, so a symlink
    planted inside cannot reach out.
    """
    raw = (raw_path or "").strip()
    if not raw:
        return None, "Error: 'path' is required."
    workdir = _get_workdir(session_id)
    candidate = raw
    # An absolute path already inside the workdir is a real path, not a habit
    # one: a project rooted at /workspace/repo would otherwise have its own
    # prefix stripped and be rejoined onto itself.
    already_inside = os.path.isabs(raw) and not _is_outside_workdir(raw, workdir)
    if not disable_sandbox and not already_inside:
        for prefix in _MISSING_PATH_PREFIXES:
            if candidate == prefix or candidate.startswith(prefix + "/"):
                candidate = candidate[len(prefix) :].lstrip("/")
                break
    if not candidate:
        return None, "Error: 'path' is required."
    target = candidate if os.path.isabs(candidate) else os.path.join(workdir, candidate)
    try:
        target = os.path.realpath(target)
    except (OSError, ValueError):
        return None, f"Error: cannot resolve path '{raw}'."
    # Full access runs python/terminal unsandboxed already; holding this one
    # tool to the workdir would just push the model back to cat.
    if not disable_sandbox and _is_outside_workdir(target, workdir):
        return None, (
            f"Error: '{raw}' is outside this conversation's working directory, "
            "which is the only place edit_file can write. Use a relative path "
            f"(for example '{os.path.basename(raw) or 'file.py'}')."
        )
    return target, ""


def _edit_file_decode(data: bytes, path: str) -> "tuple[str, str, str, str]":
    """Decode file bytes into (text, newline, bom, error).

    ``text`` is normalized to \\n so an old_string with plain newlines still
    matches a CRLF file; matching raw bytes would fail every edit of a
    Windows-authored source. The original convention is returned so the write
    puts it back instead of converting every line ending in the file.
    """
    bom = ""
    if data.startswith(codecs.BOM_UTF8):
        bom = codecs.BOM_UTF8.decode("utf-8")
        data = data[len(codecs.BOM_UTF8) :]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "", "\n", "", f"Error: '{os.path.basename(path)}' is not UTF-8 text."
    if "\x00" in text:
        return "", "\n", "", f"Error: '{os.path.basename(path)}' is a binary file."
    crlf = text.count("\r\n")
    # Judged against the total so a file with a couple of stray CRs is still
    # written back as LF; a mixed file is normalized to whichever dominates.
    newline = "\r\n" if crlf and crlf * 2 >= text.count("\n") else "\n"
    return text.replace("\r\n", "\n"), newline, bom, ""


def _edit_file_write(
    path: str,
    text: str,
    newline: str,
    bom: str,
    *,
    expect: "bytes | None" = None,
    workdir: "str | None" = None,
) -> str:
    """Write the new content, replacing the file atomically.

    Sibling temp file then rename: an interrupted write must not leave a source
    file half-replaced. The mode is carried over so an edit keeps the
    executable bit.

    ``expect`` are the bytes the edit was computed from, compared again here so
    a file another chat rewrote meanwhile is not reverted to this call's stale
    copy. ``workdir`` re-checks containment just before the rename, so a parent
    swapped for a symlink after the path was resolved is caught.
    """
    payload = (bom + text.replace("\n", newline)).encode("utf-8")
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok = True)
    except OSError as exc:
        return f"Error: cannot create directory for '{os.path.basename(path)}': {exc}"
    tmp = ""
    try:
        fd, tmp = tempfile.mkstemp(dir = directory, prefix = ".unsloth_edit_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        try:
            shutil.copymode(path, tmp)
        except OSError:
            pass  # new file, or a mode we cannot read; the default is fine
        if workdir is not None and _is_outside_workdir(path, workdir):
            return (
                f"Error: '{os.path.basename(path)}' moved outside the working "
                "directory while the edit was being prepared; nothing was written."
            )
        if expect is not None:
            try:
                with open(path, "rb") as fh:
                    current = fh.read(len(expect) + 1)
            except OSError:
                current = None
            if current != expect:
                return (
                    f"Error: '{os.path.basename(path)}' changed while this edit "
                    "was being prepared; nothing was written. Read it again and "
                    "redo the edit against the current contents."
                )
        os.replace(tmp, path)
        tmp = ""
    except OSError as exc:
        return f"Error: cannot write '{os.path.basename(path)}': {exc}"
    finally:
        if tmp:
            with contextlib.suppress(OSError):
                os.remove(tmp)
    return ""


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@")


def _edit_file_shift_hunk(line: str, offset: int) -> str:
    """Add ``offset`` to both line numbers in a @@ hunk header."""
    match = _HUNK_HEADER_RE.match(line)
    if not match:
        return line
    before_span = match.group(2) or ""
    after_span = match.group(4) or ""
    shifted = (
        f"@@ -{int(match.group(1)) + offset}{before_span} "
        f"+{int(match.group(3)) + offset}{after_span} @@"
    )
    return shifted + line[match.end() :]


def _edit_file_line_window(text: str, index: int, lines: int) -> "tuple[int, int]":
    """Offsets of a window of ``lines`` lines either side of ``index``."""
    start = index
    for _ in range(lines):
        newline = text.rfind("\n", 0, start)
        if newline == -1:
            start = 0
            break
        start = newline
    if start and text[start : start + 1] == "\n":
        start += 1
    end = index
    for _ in range(lines):
        newline = text.find("\n", end)
        if newline == -1:
            end = len(text)
            break
        end = newline + 1
    return start, max(end, index)


def _edit_file_receipt(
    before: str,
    old: str,
    new: str,
    name: str,
    count: int,
    change_at: int = 0,
) -> str:
    """A bounded unified diff of what changed.

    Line-numbered so the model can confirm the edit landed where it meant. Two
    separate bounds, because either alone leaks: difflib sees only a window
    around the first change, and the generator is consumed lazily.
    """
    import difflib
    import itertools

    window_start, window_end = _edit_file_line_window(
        before, change_at, _EDIT_FILE_DIFF_WINDOW_LINES
    )
    # The window is cut out of the old text and the replacement replayed on it,
    # rather than a second window of the same LINE COUNT cut out of the new one.
    # An edit that adds or removes lines shifts everything after it, so equal
    # windows end on different text and difflib calls that a second hunk --
    # deletions the edit never made, a window away from anything that changed.
    window_end = max(window_end, change_at + len(old))  # keep the match whole
    before_window = before[window_start:window_end]
    # Right in both modes: without replace_all the file held exactly one match.
    after_window = before_window.replace(old, new)
    first_line = before.count("\n", 0, window_start) + 1
    stream = difflib.unified_diff(
        before_window.split("\n"),
        after_window.split("\n"),
        lineterm = "",
        n = 2,
    )
    # drop the ---/+++ headers; the name is on the summary line
    taken = list(itertools.islice(stream, 2 + _EDIT_FILE_DIFF_LINES + 1))[2:]
    plural = "" if count == 1 else "s"
    head = f"Edited {name} ({count} replacement{plural})"
    if not taken:
        return head
    if len(taken) > _EDIT_FILE_DIFF_LINES:
        # An exact remaining count would cost the full diff this avoids.
        diff = taken[:_EDIT_FILE_DIFF_LINES] + ["... (more diff lines)"]
    else:
        diff = taken
    diff = [
        line
        if len(line) <= _EDIT_FILE_DIFF_LINE_CHARS
        else f"{line[:_EDIT_FILE_DIFF_LINE_CHARS]}... (+{len(line) - _EDIT_FILE_DIFF_LINE_CHARS} chars)"
        for line in diff
    ]
    # difflib numbered the hunks against the window, so shift them back to real
    # file lines; line 3 of a 9000-line file is worse than no numbers at all.
    if first_line > 1:
        diff = [_edit_file_shift_hunk(line, first_line - 1) for line in diff]
    body = "\n".join(diff)
    if len(body) > _EDIT_FILE_DIFF_CHARS:
        body = body[:_EDIT_FILE_DIFF_CHARS] + "\n... (receipt truncated)"
    return head + "\n" + body


def _edit_file_replace_all(value: object) -> "bool | None":
    """Read replace_all strictly; None means "not a boolean".

    bool("false") is True, and models do emit the JSON string. Coercing it that
    way turns the multi-match guard off and rewrites every occurrence.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def _edit_file_create(
    target: str,
    new: str,
    name: str,
    newline: str,
    workdir: "str | None" = None,
) -> str:
    """Handle the empty-old_string case: create a file, never clobber one.

    A zero-byte file is writable here on purpose: refusing every existing target
    would strand the model, since an empty old_string would be refused and no
    other old_string can match an empty file.

    The absent case is created with O_EXCL rather than checked and then written:
    two chats sharing a workspace can both pass a lexists() check and the later
    rename drops the earlier file. O_EXCL also gives the new file the usual
    umask-derived mode, where a mkstemp temp file would leave it 0600.
    """
    payload = (new.replace("\n", newline)).encode("utf-8")
    if not os.path.lexists(target):
        directory = os.path.dirname(target) or "."
        try:
            os.makedirs(directory, exist_ok = True)
        except OSError as exc:
            return f"Error: cannot create directory for '{name}': {exc}"
        if workdir is not None and _is_outside_workdir(target, workdir):
            return (
                f"Error: '{name}' moved outside the working directory while the "
                "edit was being prepared; nothing was written."
            )
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        # Never follow a symlink planted at the final component in the meantime.
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(target, flags, 0o666)
        except FileExistsError:
            return (
                f"Error: '{name}' was created by something else while this call "
                "was preparing it; nothing was written."
            )
        except OSError as exc:
            return f"Error: cannot write '{name}': {exc}"
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
        except OSError as exc:
            # ENOSPC partway through leaves the bytes that fit -- a file
            # truncated mid-token -- and the retry the message asks for cannot
            # clear it, since an empty old_string refuses a non-empty target.
            # Removing the inode puts the retry back on the create path.
            with contextlib.suppress(OSError):
                os.remove(target)
            return f"Error: cannot write '{name}': {exc}"
        return f"Created {name} ({new.count(chr(10)) + 1 if new else 0} lines)"
    try:
        st = os.stat(target)
    except OSError:
        st = None
    # Refused rather than measured: a FIFO reports st_size 0, so it fell into
    # the zero-byte branch below, whose write reopens the target -- and open()
    # on a FIFO with no writer blocks for ever.
    if st is None or not S_ISREG(st.st_mode) or st.st_size:
        return (
            f"Error: '{name}' already exists. An empty 'old_string' only "
            "creates a new file; to change this one, pass the exact text to "
            "replace."
        )
    # Guarded like any other edit, so a chat that filled it in is not clobbered.
    error = _edit_file_write(target, new, newline, "", expect = b"", workdir = workdir)
    if error:
        return error
    return f"Created {name} ({new.count(chr(10)) + 1 if new else 0} lines)"


# Each entry costs a full `count` scan of the file plus a search pass, and the file may be
# up to 16 MiB, so the work is entries x size. Unbounded, a model-generated batch of a few
# thousand one-line edits turns a single call into gigabytes of repeated scanning and holds
# the tool worker for minutes. High enough that no honest refactor hits it, low enough that
# the worst case stays bounded.
_MAX_EDITS_PER_CALL = 100

# And a bound on the matches ONE replace_all may expand into. The entry limit above caps
# how many patterns are searched, not how many places any of them hits, so a single
# pathological entry can still swamp the worker on its own.
_MAX_MATCH_SPANS = 10_000


def _edit_file_parse_edits(raw) -> "tuple[list[tuple[str, str, bool]], str]":
    """Validate the `edits` array into `(old, new, replace_all)` triples.

    Every string is normalized the way the file is, so a snippet copied out of `cat`
    output with plain newlines still matches a Windows-authored file.

    The surrogate check is per entry and up front, before anything is written: a
    truncated emoji escape ("\\ud83d") survives `json.loads` as a lone surrogate that
    cannot be encoded, and the UnicodeEncodeError the write raises is swallowed upstream
    into "Unknown tool: edit_file" -- the one answer that sends the model back to the
    whole-file rewrite. `old_string` needs no check, being only ever compared.
    """
    if not isinstance(raw, list) or not raw:
        return [], (
            "Error: 'edits' must be a non-empty array of {old_string, new_string} "
            "objects. Send every change to this file as separate entries in it."
        )
    if len(raw) > _MAX_EDITS_PER_CALL:
        return [], (
            f"Error: {len(raw)} edits in one call is over the limit of "
            f"{_MAX_EDITS_PER_CALL}; nothing was written. Send them in batches of "
            f"{_MAX_EDITS_PER_CALL} or fewer, applying each batch before the next."
        )
    edits: list[tuple[str, str, bool]] = []
    for index, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            return [], f"Error: edit {index} is not an object with old_string/new_string."
        old = entry.get("old_string")
        new = entry.get("new_string")
        # Checked, not coerced: str(None) would write the literal "None" into a file.
        if not isinstance(old, str) or not isinstance(new, str):
            return [], (
                f"Error: edit {index} needs 'old_string' and 'new_string' to both be strings."
            )
        try:
            new.encode("utf-8")
        except UnicodeEncodeError:
            return [], (
                f"Error: edit {index} has unpaired surrogate characters in "
                "'new_string', usually a half-written emoji; nothing was written. "
                "Send it again as plain text."
            )
        replace_all = _edit_file_replace_all(entry.get("replace_all"))
        if replace_all is None:
            return [], f"Error: edit {index} needs 'replace_all' to be true or false."
        edits.append((old.replace("\r\n", "\n"), new.replace("\r\n", "\n"), replace_all))
    return edits, ""


def _edit_file_apply_all(
    before: str, edits: "list[tuple[str, str, bool]]", name: str
) -> "tuple[str, int, str, str, int, str]":
    """Apply every edit against the ORIGINAL text, or none of them.

    Matched against `before` rather than against the running result, which is the rule
    llama.cpp's own `edit_file` states and the only one a model can reason about: it
    copied each `old_string` out of the file it read, so an entry that silently matched
    the output of an earlier entry would land somewhere it never saw.

    Spans are resolved for all entries first and checked for overlap, then applied right
    to left so the earlier offsets stay valid. Every failure returns before a single byte
    is written -- a partly applied batch is the one outcome worse than a refused one,
    because the model cannot tell which half landed.
    """
    spans: list[tuple[int, int, str, int]] = []
    for index, (old, new, replace_all) in enumerate(edits, 1):
        if not old:
            # Defence in depth, not a live path: `_edit_file` rejects every empty
            # `old_string` before calling this. It matters because the failure mode if
            # that guard ever moves is not a wrong answer but a HANG -- `find(old, start
            # + len(old))` cannot advance on a zero-length pattern, and the worker keeps
            # a core after its caller has timed out.
            return (
                "",
                0,
                "",
                "",
                0,
                (
                    f"Error: edit {index} has an empty 'old_string'. Only a single edit "
                    "may be empty, and only to create the file."
                ),
            )
        count = before.count(old)
        if count == 0:
            return (
                "",
                0,
                "",
                "",
                0,
                (
                    f"Error: edit {index}'s 'old_string' was not found in {name}. It must "
                    "match the file byte for byte, including indentation. Read the file and "
                    "copy the text to replace out of it."
                ),
            )
        if count > 1 and not replace_all:
            return (
                "",
                0,
                "",
                "",
                0,
                (
                    f"Error: edit {index}'s 'old_string' matches {count} places in {name}. "
                    "Include surrounding lines to make it unique, or set replace_all on that "
                    f"entry to change all {count}."
                ),
            )
        # One entry needs no spans at all. There is nothing to overlap with, so the whole
        # reason to enumerate matches disappears and `str.replace` does the work in linear
        # space. Enumerating cost about 16 million tuples plus a sort on a 16 MiB file of a
        # one-character pattern -- over a gigabyte for an edit the single-edit spelling had
        # always done cheaply. The batch limit does not help here: one entry is enough.
        if replace_all and len(edits) == 1:
            after = before.replace(old, new)
            first = before.find(old)
            return after, count, old, new, first, ""
        if replace_all and count > _MAX_MATCH_SPANS:
            return (
                "",
                0,
                "",
                "",
                0,
                (
                    f"Error: edit {index}'s 'old_string' matches {count} places in {name}, "
                    f"over the limit of {_MAX_MATCH_SPANS} for one entry in a batch; "
                    "nothing was written. Send it as a call of its own, or use a longer "
                    "'old_string'."
                ),
            )
        start = before.find(old)
        while start >= 0:
            spans.append((start, start + len(old), new, index))
            if not replace_all:
                break
            start = before.find(old, start + len(old))
    spans.sort()
    for (start, end, _, index), (next_start, _, _, next_index) in zip(spans, spans[1:]):
        if next_start < end:
            return (
                "",
                0,
                "",
                "",
                0,
                (
                    f"Error: edits {index} and {next_index} overlap in {name}. Every "
                    "old_string is matched against the file as it was before this call, so "
                    "two edits cannot cover the same text. Combine them into one entry."
                ),
            )
    # One pass with a cursor, not a slice-and-concat per span. Rebuilding the whole string
    # for every replacement is quadratic, and `replace_all` over a large file is exactly
    # where that bites: a file with tens of thousands of matches took 10s where the single
    # `str.replace` it succeeded took well under one.
    parts: list[str] = []
    cursor = 0
    for start, end, new, _ in spans:
        parts.append(before[cursor:start])
        parts.append(new)
        cursor = end
    parts.append(before[cursor:])
    after = "".join(parts)
    first_start, first_end, first_new, _ = spans[0]
    return after, len(spans), before[first_start:first_end], first_new, first_start, ""


def _edit_file(
    arguments: dict,
    session_id: "str | None" = None,
    disable_sandbox: bool = False,
) -> str:
    """Replace exact strings in a file. See the notes above."""
    edits, error = _edit_file_parse_edits(arguments.get("edits"))
    if error:
        return error
    target, error = _edit_file_resolve(
        str(arguments.get("path") or ""), session_id, disable_sandbox
    )
    if error:
        return error
    name = os.path.basename(target)
    # Decided before the no-op check below, not after: both strings empty is the
    # documented way to create __init__.py or .gitkeep, and read as "identical,
    # nothing to change" it was refused, leaving no way to write a zero-byte
    # file.
    if not edits[0][0]:
        if len(edits) > 1:
            return (
                "Error: an empty 'old_string' creates the file, so it cannot be "
                f"combined with the other {len(edits) - 1} edit(s). Create the file "
                "in one call, then edit it in the next."
            )
        return _edit_file_create(
            target,
            edits[0][1],
            name,
            "\n",
            workdir = None if disable_sandbox else _get_workdir(session_id),
        )
    for index, (old, new, _) in enumerate(edits, 1):
        if not old:
            return (
                f"Error: edit {index} has an empty 'old_string'. Only a single edit "
                "may be empty, and only to create the file."
            )
        if old == new:
            return (
                f"Error: edit {index} has identical 'old_string' and 'new_string'; "
                "nothing to change."
            )
    try:
        st = os.stat(target)
    except FileNotFoundError:
        return f"Error: '{name}' does not exist. Pass an empty 'old_string' to create it."
    except OSError as exc:
        return f"Error: cannot read '{name}': {exc}"
    if os.path.isdir(target):
        return f"Error: '{name}' is a directory."
    # A FIFO reads forever and a character device such as /dev/zero reads until
    # memory runs out. Neither reports a useful st_size, and this path carries
    # no timeout or cancel event, so the turn cannot be recovered.
    if not S_ISREG(st.st_mode):
        return f"Error: '{name}' is not a regular file."
    if st.st_size > _EDIT_FILE_MAX_BYTES:
        return (
            f"Error: '{name}' is larger than "
            f"{_EDIT_FILE_MAX_BYTES // (1024 * 1024)}MB; edit it with python instead."
        )
    try:
        with open(target, "rb") as fh:
            data = fh.read(_EDIT_FILE_MAX_BYTES + 1)
    except OSError as exc:
        return f"Error: cannot read '{name}': {exc}"
    before, newline, bom, error = _edit_file_decode(data, target)
    if error:
        return error
    after, total, first_old, first_new, change_at, error = _edit_file_apply_all(before, edits, name)
    if error:
        return error
    error = _edit_file_write(
        target,
        after,
        newline,
        bom,
        expect = data,
        workdir = None if disable_sandbox else _get_workdir(session_id),
    )
    if error:
        return error
    # Windowed around the first replacement, rather than diffing the whole file.
    return _edit_file_receipt(
        before,
        first_old,
        first_new,
        name,
        total,
        change_at = change_at,
    )


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


def web_search_tool_with_images() -> dict:
    # web_search plus image_queries, offered while the Search images setting is on.
    tool = copy.deepcopy(WEB_SEARCH_TOOL)
    fn = tool["function"]
    fn["description"] += (
        " To show pictures, pass image_queries: the exact names of the specific things you "
        'will mention, one per entry (e.g. ["German Shepherd", "Labrador"]), never a list '
        "title. Each returns an [[img:...]] token; put it on its own line under that item. "
        "image_queries may also be sent alone after the answer."
    )
    fn["parameters"]["properties"]["image_queries"] = {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 5,
        "description": "Specific things to fetch one picture each for, named exactly as in your answer.",
    }
    return tool


def _build_sandbox_paths_note() -> str:
    """Platform and working-directory note, on BOTH tool descriptions.

    Models habitually write to /mnt/data, a ChatGPT code-interpreter path that
    does not exist here, so the POSIX text names it. Naming only POSIX paths on
    Windows reads as "you are on Linux" and models then refuse to invoke Windows
    programs that are in fact available, so that text says where the code runs
    instead: without it a model assumes the pipe is its only output and declines
    to open a window it believes nobody can see.
    """
    # Without this a model assumes its output vanished into a scratch dir and
    # says so, which reads as "the file was not really created".
    created = (
        " Any file you create here is kept and shown to the user with a download "
        "link, so name the files you created in your reply -- by file name only, "
        "since you do not know their absolute path."
    )
    if sys.platform != "win32":
        return (
            " Read and write files using relative paths in the current working "
            "directory, which persists for this conversation; absolute paths like "
            "/mnt/data or /tmp/outputs do not exist." + created
        )
    return (
        " You are on Windows, and this runs on the user's own machine. Read and "
        "write files using relative paths in the current working directory, which "
        "persists for this conversation." + created
    )


# Full access (permission_mode='full') edits, applied to the sandboxed text
# rather than writing a second copy of it. Everything the two modes share -- the
# relative-path advice, the persisted workdir, the download-link note -- is prose
# that gets tuned over time, and a parallel copy would drift out of sync in
# silence. Only the claims the sandbox makes true are touched. A rewording that
# stops one of these matching is caught by test_full_access_tool_prompt.py, which
# asserts the sandboxed markers are gone from the result on both platforms.
_FULL_ACCESS_SUBSTITUTIONS = (
    # The only sentence in the python description that names the sandbox. Just
    # dropped, not reworded: where the code runs is said once, by the paths note
    # below, which both tools carry.
    ("Execute Python code in a sandbox and", "Execute Python code and"),
    # POSIX: a blanket denial is wrong with the sandbox off, and so is a blanket
    # promise. POSIX has no sentence saying where the code runs, so it is added
    # here; Windows already has one.
    (
        "; absolute paths like /mnt/data or /tmp/outputs do not exist.",
        ". This runs wherever Unsloth Studio is running, which may be a remote host "
        "or a container with only some paths mounted.{clause}",
    ),
    # Windows already says where the code runs and never denies absolute paths,
    # so there is nothing false to remove; state the capability instead. "the
    # user's own machine" is narrowed at the same time: --secure and -H 0.0.0.0
    # are documented remote modes (README), and the tools run on the host serving
    # Unsloth, which is then not the device the user is looking at.
    # _TERMINAL_SHELL_NOTE is carried through unchanged except here: its Git Bash
    # branch promises a detached program's window appears on the user's desktop,
    # which only holds while Unsloth is local.
    (
        "opens a window on the user's desktop.",
        "opens a window on that machine's desktop, which the user sees only if "
        "they are sitting at it.",
    ),
    (
        " You are on Windows, and this runs on the user's own machine.",
        " You are on Windows, and this runs wherever Unsloth Studio is running, "
        "which may be a remote host or a container with only some paths "
        "mounted.{clause}",
    ),
)


# What "the sandbox is off" actually means for paths, per tool. Shared by both
# platform substitutions: the split is the shim, not the OS. _build_bypass_env
# keeps _SANDBOX_SITE_DIR on PYTHONPATH for BOTH tools, so sitecustomize.py still
# loads, and being a CPython startup hook is what separates them. Measured:
#
#   python, parent exists    -> writes the real absolute path
#   python, absent prefix    -> _remap keeps the SUFFIX under the workdir
#                               (/mnt/data/reports/out.csv -> ./reports/out.csv)
#                               and returns before the generic fallback, so no
#                               basename collapse and no anti-clobber
#   python, other missing parent -> the fallback keeps only the base name, and
#                               raises when an UNRELATED file holds it; the
#                               .unsloth_sandbox_remap.json sidecar lets a rewrite
#                               of the same invented path re-serve its own target
#   python, prefix present   -> a real /mnt/data mount is never shadowed, so a
#                               prefix is special only while absent, which is why
#                               the clause names one inside a conditional and
#                               never categorically
#   API coverage differs per rewrite -> _makedirs calls _remap only, so the
#                               generic fallback is open/io.open/os.open alone and
#                               os.makedirs under a missing parent OUTSIDE the
#                               convention prefixes targets the REAL host path.
#                               Inside them _remap still rewrites, measured:
#                               makedirs("/mnt/data/reports") created ./reports and
#                               raised nothing, so the clause has to name the prefixes
#                               rather than say "not rewritten at all" -- a model told
#                               otherwise reports the host path for a directory that
#                               is in its working directory.
#                               An attempt, not an outcome: measured
#                               under this shim, makedirs into a mode-500 directory
#                               raised PermissionError and created nothing, neither
#                               on the host nor in the workdir, so the clause must
#                               not promise creation any more than it promises a
#                               prefix is absent. os.rename and os.symlink raise, while
#                               shutil.copy writes the rewritten file through open
#                               and then raises in copymode
#   terminal                 -> the shell's own rules, except for Python it
#                               launches, which is patched like the python tool
_FULL_ACCESS_CLAUSE = {
    "python": (
        " The code sandbox is disabled, so absolute paths under a directory that "
        "exists do resolve. Two different rewrites apply when the directory does "
        "not exist: under a code-interpreter convention prefix (/mnt/data, "
        "/mnt/outputs, /tmp/outputs, /home/sandbox, /workspace) the rest of the "
        "path is kept relative to the working directory, replacing any file "
        "already sitting there; under any other missing directory only the base "
        "name is kept, and the write fails outright if that name is taken by an "
        "unrelated file, though rewriting the same absolute path just replaces "
        "what your own earlier call left there. The convention rewrite covers "
        "open() and the mkdir calls; the other covers open() alone, so "
        "os.makedirs under a missing parent outside those prefixes is not "
        "rewritten and attempts the real host path, which then succeeds or fails "
        "on the filesystem's own permissions. "
        "Neither touches os.rename or os.symlink, which simply fail, and a helper "
        "such as shutil.copy can write the rewritten file and still raise on a "
        "later step. Report where a file actually landed rather than the path you "
        "asked for."
    ),
    "terminal": (
        " The code sandbox is disabled, so absolute paths do resolve as the shell "
        "resolves them. Python you launch from here is the exception: it loads the "
        "same shim as the python tool and gets the same rewrites, so a create "
        "under a directory that does not exist lands in the working directory."
    ),
}


def _to_full_access(description: str, tool_name: str) -> str:
    """Rewrite a sandboxed tool description for Full access.

    Under bypass_permissions the loops pass disable_sandbox=True:
    _build_bypass_env / _bypass_preexec skip the static analysis, the command
    blocklist and the rlimits, so the host filesystem really is reachable.
    Handing a model the sandboxed text in that mode makes it answer "I am
    sandboxed and cannot see your files" to a question one tool call would have
    answered. Untouched clauses are the ones still true in both modes: the
    workdir is the per-session dir either way (_build_bypass_env repoints HOME at
    it and TMPDIR / TEMP / TMP just inside it), and so is the download-link note.
    """
    clause = _FULL_ACCESS_CLAUSE[tool_name]
    for sandboxed, full_access in _FULL_ACCESS_SUBSTITUTIONS:
        description = description.replace(sandboxed, full_access.format(clause = clause))
    return description


def _build_terminal_shell_note() -> str:
    """Shell-specific note, on the TERMINAL description only.

    Which shell runs is read from the resolver, not assumed: telling a model it
    has bash on a host where _get_shell_cmd fell back to cmd reintroduces the
    multi-line half-execution this note exists to prevent. It stays off the
    python description because none of it applies to the python sandbox, and
    naming a shell there invites subprocess/os.system as a way past the terminal
    blocklist. No program is named either: powershell/pwsh are on that blocklist
    and `cmd /c start` is not, so recommending one hands back a hard block and
    recommending the other advertises the gap.
    """
    if sys.platform != "win32":
        return ""
    if _windows_bash():
        return (
            " The shell is bash (Git for Windows), and native Windows programs are "
            "available; a program you start detached opens a window on the user's "
            "desktop."
        )
    return (
        " The shell is cmd, not bash: send one command per call, chain with &&, and "
        "do not use bash syntax such as multi-line loops or single-quoted arguments."
    )


_SANDBOX_PATHS_NOTE = _build_sandbox_paths_note()
_TERMINAL_SHELL_NOTE = _build_terminal_shell_note()

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
        "description": "Execute a terminal command and return stdout/stderr."
        + _SANDBOX_PATHS_NOTE
        + _TERMINAL_SHELL_NOTE,
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

# Full access runs these two without the sandbox, so it gets its own pair of
# schemas rather than a per-request rebuild: the descriptions are
# platform-derived constants either way. The sandboxed pair stays the module
# default, so every existing importer keeps the safe wording. The shell note is
# unaffected by the substitutions and carries through as-is.
PYTHON_TOOL_FULL_ACCESS = {
    "type": "function",
    "function": {
        **PYTHON_TOOL["function"],
        "description": _to_full_access(PYTHON_TOOL["function"]["description"], "python"),
    },
}

TERMINAL_TOOL_FULL_ACCESS = {
    "type": "function",
    "function": {
        **TERMINAL_TOOL["function"],
        "description": _to_full_access(TERMINAL_TOOL["function"]["description"], "terminal"),
    },
}

# edit_file is registered below, once its schema exists.
_FULL_ACCESS_TOOL_BY_NAME = {
    "python": PYTHON_TOOL_FULL_ACCESS,
    "terminal": TERMINAL_TOOL_FULL_ACCESS,
}


def apply_full_access_tool_descriptions(tools: list[dict]) -> list[dict]:
    """Swap python/terminal/edit_file for their Full access schemas.

    Only the sandboxed built-ins are touched; web_search, render_html,
    search_knowledge_base and MCP tools are passed through untouched, and a list
    without any of them is returned as-is so callers can apply this
    unconditionally. The input list is never mutated -- ALL_TOOLS entries are
    module globals shared across requests.
    """
    if not tools:
        return tools
    swapped = False
    out: list[dict] = []
    for tool in tools:
        name = (tool.get("function") or {}).get("name") if isinstance(tool, dict) else None
        replacement = _FULL_ACCESS_TOOL_BY_NAME.get(name)
        if replacement is None:
            out.append(tool)
        else:
            out.append(replacement)
            swapped = True
    return out if swapped else tools


EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        # The description does the steering: given the tool but no preference,
        # a model keeps writing heredocs because that is what it was trained on.
        "description": (
            "Change a file by replacing exact strings. Prefer this over rewriting a "
            "file with python or a shell heredoc: it sends only what changes. Copy each "
            "old_string verbatim from the file, indentation included. Batch every change "
            "to one file into edits rather than calling repeatedly, since each call "
            "replays the whole conversation. Every old_string matches the file as it was "
            "BEFORE this call, not the result of earlier edits, and no two may overlap. "
            "Each must match exactly one place unless it sets replace_all; if any matches "
            "none or several, nothing is written. Paths are relative to the working "
            "directory. A successful call means the file holds what you sent, so do not "
            "read it back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to edit, relative to the working directory.",
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "One or more replacements to apply together. A single "
                        "entry whose old_string is empty creates a new file."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {
                                "type": "string",
                                "description": (
                                    "Exact text to replace, copied from the file. "
                                    "Empty creates a new file."
                                ),
                            },
                            "new_string": {
                                "type": "string",
                                "description": "Text to put in its place.",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": (
                                    "Replace every occurrence of this entry's "
                                    "old_string instead of requiring a unique "
                                    "match. Defaults to false."
                                ),
                            },
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
    },
}

# Appended, not substituted: the sandboxed text never claims absolute paths
# fail, so there is nothing false to rewrite, only a capability to add. A model
# that thinks it cannot reach a real checkout falls back to the whole-file rewrite.
_EDIT_FILE_FULL_ACCESS_CLAUSE = (
    " The code sandbox is disabled, so an absolute path resolves as written and "
    "edits the real file there, anywhere the Unsloth Studio process can reach."
)

EDIT_FILE_TOOL_FULL_ACCESS = {
    "type": "function",
    "function": {
        **EDIT_FILE_TOOL["function"],
        "description": EDIT_FILE_TOOL["function"]["description"] + _EDIT_FILE_FULL_ACCESS_CLAUSE,
    },
}

_FULL_ACCESS_TOOL_BY_NAME["edit_file"] = EDIT_FILE_TOOL_FULL_ACCESS

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

SEARCH_CONVERSATION_TOOL = {
    "type": "function",
    "function": {
        "name": "search_conversation",
        "description": (
            "Search earlier turns of THIS conversation that were removed from your "
            "context when it grew too long. Use it whenever the user refers to something "
            "discussed earlier that you cannot see, instead of saying you have no record "
            "of it."
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
                    "description": "Max earlier turns to return.",
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
    EDIT_FILE_TOOL,
    RENDER_HTML_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    SEARCH_CONVERSATION_TOOL,
]

# Deliberately an ordinary tool with an ordinary result. Studio runs three tool loops -- one for
# llama.cpp, one for safetensors, one for external providers -- and only a plain result behaves
# the same in all three. The client starts the run off the tool events every loop already
# publishes, so nothing here needs to know a research run exists.
#
# Never in ALL_TOOLS: it is offered only when the composer armed research.
#
# The client keys the handoff on this opening rather than on tool_end alone: a denied, skipped,
# truncated or budget-exhausted call is closed by the same event, and only the result says the
# call actually ran.
DEEP_RESEARCH_STARTED_MARKER = "Deep Research has started"
DEEP_RESEARCH_STARTED = (
    f"{DEEP_RESEARCH_STARTED_MARKER} on that question. Reply with one short sentence saying you "
    "are looking into it. Do not answer the question yourself and do not call this tool again; "
    "the researched report arrives separately."
)

DEEP_RESEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "deep_research",
        "description": (
            "Start Deep Research on the user's question: a multi-step web investigation that "
            "gathers current sources and writes a cited report, replacing your reply. The user "
            "turned this on because they want researched answers, so call it for any question "
            "about the world -- facts, events, laws, products, papers, prices, comparisons, "
            "anything that may have changed since your training -- even when you think you know "
            "the answer. Do not answer such questions from memory.\n"
            "Do not call it for a message with no question in it, such as a hello or a thanks, or "
            "for a request to write or transform text the user supplied.\n"
            "If the topic is too vague to research well, do not call this yet: ask one short "
            "clarifying question, then call it once the user has narrowed it down.\n"
            "The question you pass is what gets researched, so make it specific and "
            "self-contained: fold in what the conversation established rather than repeating "
            "the user's words."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The specific, self-contained question to research. Not the user's raw "
                        "message unless it already reads as one."
                    ),
                },
            },
            "required": ["question"],
        },
    },
}


# OpenAI's function.name regex; MCP names that violate it would 400 the whole
# request, so validate up front and skip with a warning.
_OPENAI_FN_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _mcp_tool_model_visible(tool: dict) -> bool:
    """False for MCP Apps tools marked app-only (_meta.ui.visibility without
    "model"): those exist for a server-rendered widget to call, not the LLM."""
    # model_dump() gives "meta", the wire "_meta"; unrelated keys in one must not mask the other.
    for key in ("meta", "_meta"):
        meta = tool.get(key)
        if not isinstance(meta, dict):
            continue
        ui = meta.get("ui")
        visibility = ui.get("visibility") if isinstance(ui, dict) else None
        if visibility is None:
            # Tolerated, not spec: only flat "ui/resourceUri" is deprecated.
            visibility = meta.get("ui/visibility")
        if isinstance(visibility, (list, tuple)):
            return "model" in visibility
    return True


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
        if not _mcp_tool_model_visible(tool):
            logger.debug("Skipping app-only MCP tool '%s' on '%s'.", raw_name, display)
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
                    # mcp<2 dumps "inputSchema", 2.x "input_schema"; accept both.
                    "parameters": tool.get("inputSchema")
                    or tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return specs


def cached_mcp_tools() -> tuple[list[dict], bool]:
    """The MCP schemas already in cache, and whether that is the whole set.

    get_enabled_mcp_tools() renders the same servers, but reaches the network to do it: it
    spawns stdio servers, blocks for a probe timeout on one that is down, and writes cache and
    cool-off state. A background token count must not do any of that, so this reads only what
    those probes have already filled in.

    ``complete`` is False when an enabled server has nothing cached and is not in its
    post-failure cool-off, meaning a completion would probe it and render schemas this cannot
    price. A cool-off server renders nothing on the completion path either, so skipping that one
    is exact rather than short. Callers that must not undercount should decline on False.
    """
    servers = [s for s in mcp_servers_db.list_servers() if s.get("is_enabled")]
    if not stdio_mcp_enabled():
        servers = [s for s in servers if not is_stdio(s["url"])]

    specs: list[dict] = []
    complete = True
    for server in servers:
        payload = get_cached_tools(server["id"])
        if payload is None:
            if not in_failure_cooloff(server["id"]):
                complete = False
            continue
        specs.extend(_mcp_specs_for_server(server, payload))
    return specs, complete


async def get_enabled_mcp_tools() -> list[dict]:
    # Keep the SQLite-backed server list off the event loop.
    servers = [
        s for s in await asyncio.to_thread(mcp_servers_db.list_servers) if s.get("is_enabled")
    ]
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
        # Keep this re-read on-loop so an edit cannot invalidate between it and
        # the cache writes below. Drop results for changed or removed servers.
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
    website_policy: dict | None = None,
    conversation_branch: list[dict] | None = None,
    conversation_budget_tokens: int | None = None,
    conversation_token_counter = None,
    context_tokens = _UNSET_CONTEXT_TOKENS,
    search_images: bool = False,
    result_budget_tokens: int | None = None,
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
    ``website_policy``: hidden server-validated domain limits for web_search.
    """
    logger.info(f"execute_tool: name={name}, session_id={session_id}, timeout={timeout}")
    # Set unconditionally, so a value from an earlier call on this thread can never be
    # read by a later one. That is what makes a try/finally reset unnecessary here.
    _REQUEST_CONTEXT_TOKENS.set(context_tokens)
    # Same rule, and it matters more here: a stale budget is a budget measured before this
    # turn's own tool exchanges existed, which is precisely the undercount that lets the
    # last result overflow.
    _REQUEST_RESULT_BUDGET.set(result_budget_tokens)
    # Arguments that never parsed, for a tool with no single argument they could be.
    # Answered here rather than by the tool, which can only report the keys it wanted and
    # would blame the model for omitting them: `edit_file` said "'old_string' and
    # 'new_string' must both be strings" about a call that sent neither because its JSON
    # was cut off mid-string. Naming the real fault is what makes the retry the right one.
    # Imported here for the same reason `strip_result_for_model` is: at module scope this
    # closes an import cycle, since the controller reads this module's own schemas.
    from .tool_loop_controller import UNPARSED_ARGUMENTS_KEY  # noqa: PLC0415

    if isinstance(arguments, dict) and UNPARSED_ARGUMENTS_KEY in arguments:
        raw = str(arguments.get(UNPARSED_ARGUMENTS_KEY) or "")
        truncated = raw.lstrip().startswith(("{", "[")) and not raw.rstrip().endswith(("}", "]"))
        cause = (
            "were cut off part-way and could not be read" if truncated else "were not valid JSON"
        )
        return (
            f"Error: {name} arguments {cause}, so nothing ran. Resend as complete JSON, "
            "split across smaller calls if the content is long."
        )
    effective_timeout = _EXEC_TIMEOUT if timeout is _TIMEOUT_UNSET else timeout
    if name == "search_knowledge_base":
        return _fit_result_to_room(
            _search_knowledge_base_with_budget(
                arguments,
                rag_scope,
                effective_timeout,
                cancel_event,
            ),
            name,
        )
    if name == "search_conversation":
        # Scoped by thread id alone: the archive is this chat's own evicted turns, so it
        # works with or without a document rag_scope.
        return _fit_result_to_room(
            _search_knowledge_base_with_budget(
                arguments,
                {
                    "thread_id": thread_id,
                    "branch_messages": conversation_branch,
                    "budget_tokens": conversation_budget_tokens,
                    "token_counter": conversation_token_counter,
                },
                effective_timeout,
                cancel_event,
                search_fn = _search_conversation,
            ),
            name,
        )
    if name == "render_html":
        return _fit_result_to_room(_render_html_result(arguments), name)
    if name.startswith(MCP_TOOL_PREFIX):
        try:
            _, server_id, tool_name = name.split("__", 2)
        except ValueError:
            return f"Error: malformed MCP tool name '{name}'"
        server = mcp_servers_db.get_server(server_id)
        if not server:
            return f"Error: MCP server for tool '{tool_name}' not found"
        display = server.get("display_name") or server_id
        if not server.get("is_enabled"):
            return f"Error: MCP server '{display}' is disabled"
        if is_stdio(server["url"]) and not stdio_mcp_enabled():
            return f"Error: stdio MCP server '{display}' is disabled on this host"
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

        return _fit_result_to_room(
            call_tool_sync(
                url = url,
                headers = headers,
                name = tool_name,
                args = arguments,
                timeout = effective_timeout,
                use_oauth = bool(server.get("use_oauth")),
                cancel_event = cancel_event,
                scope = mcp_scope,
                config_check = _config_current,
            ),
            name,
        )
    if name == "deep_research":
        if not str(arguments.get("question") or "").strip():
            return "Error: deep_research needs a question to investigate."
        return DEEP_RESEARCH_STARTED
    if name == "web_search":
        return _fit_result_to_room(
            _web_search(
                arguments.get("query", ""),
                url = arguments.get("url"),
                timeout = effective_timeout,
                cancel_event = cancel_event,
                website_policy = website_policy,
                include_images = search_images,
                image_queries = arguments.get("image_queries"),
            ),
            name,
        )
    # Both run with the session's sandbox as cwd, so a chat deleted mid-call
    # must not unlink it from under them.
    if name == "python":
        with _session_in_flight(session_id):
            return _python_exec(
                arguments.get("code", ""),
                cancel_event,
                effective_timeout,
                session_id,
                disable_sandbox = disable_sandbox,
                output_callback = output_callback,
                thread_id = thread_id,
            )
    if name == "terminal":
        with _session_in_flight(session_id):
            return _bash_exec(
                arguments.get("command", ""),
                cancel_event,
                effective_timeout,
                session_id,
                disable_sandbox = disable_sandbox,
                output_callback = output_callback,
                thread_id = thread_id,
            )
    # Same in-flight guard as the two above: it writes into the session workdir,
    # so a chat deleted mid-call must not unlink it underneath.
    if name == "edit_file":
        with _session_in_flight(session_id):
            return _fit_result_to_room(
                _edit_file(
                    arguments,
                    session_id = session_id,
                    disable_sandbox = disable_sandbox,
                ),
                name,
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


# Ceiling for a model-supplied top_k. Small on purpose: this returns whole archived turns
# into a protected exchange the rolling window cannot trim.
_MAX_CONVERSATION_SEARCH_TOP_K = 8


def _search_conversation(arguments: dict, rag_scope: dict | None) -> str:
    """Search this thread's archived turns. ``rag_scope`` carries only the thread id here;
    the model supplies ``query``/``top_k``."""
    scope = rag_scope or {}
    thread_id = scope.get("thread_id")
    query = (arguments or {}).get("query", "")
    if not query or not str(query).strip():
        return "Error: query is empty."
    if not thread_id:
        return "There is no earlier conversation to search."
    try:
        from core.rag import conversation_archive
    except Exception as exc:  # noqa: BLE001
        logger.warning("Conversation archive unavailable: %s", exc)
        return "Searching earlier conversation is unavailable on this server."
    if not conversation_archive.enabled():
        return "Searching earlier conversation is unavailable on this server."

    # Clamped, not trusted: top_k comes from the model, and a negative value reaches a
    # Python slice as out[:-1], returning nearly the whole candidate pool as a ~30k-token
    # tool result that the protected current exchange cannot evict.
    requested = _opt_int((arguments or {}).get("top_k"))
    # None, not the ceiling: an omitted top_k must fall through to the configured recall
    # default. Defaulting to the maximum returned eight chunks into that same protected
    # exchange, enough to fail the next pass on a 4K chat.
    top_k = (
        None if requested is None else max(1, min(_MAX_CONVERSATION_SEARCH_TOP_K, int(requested)))
    )
    # Then against the room actually left: the fixed cap bounds what the model may ask
    # for, not what fits. Eight chunks is roughly 4,000 tokens once wrapped, so on a 4K
    # chat overshooting here is a context-length error no later preflight can recover.
    budget = scope.get("budget_tokens")
    if budget is not None:
        default_k = 1
        try:
            from core.rag import config as rag_config
            affordable = max(0, int(budget)) // max(1, int(rag_config.CHUNK_TOKENS))
            default_k = max(1, int(rag_config.CONVERSATION_ARCHIVE_TOP_K))
        except Exception:
            affordable = 0
        if affordable <= 0:
            return "There is no room left in this context to search earlier conversation."
        # An omitted top_k still means the configured default; room is a cap on it, not a
        # target. Taking the room itself asked a 128K chat for 200 passages, past both the
        # default and the ceiling the model's own value is held to.
        top_k = (
            min(default_k, _MAX_CONVERSATION_SEARCH_TOP_K, affordable)
            if top_k is None
            else max(1, min(top_k, affordable))
        )

    # The branch this request is on, so a response replaced by Retry cannot be searched
    # back out of the archive. Absent callers fall back to the whole stored thread.
    def _recall(k):
        return conversation_archive.recall(
            str(thread_id),
            str(query),
            top_k = k,
            branch_messages = scope.get("branch_messages"),
        )

    found = _recall(top_k)
    if not found:
        return "No earlier turns of this conversation matched that query."

    # Then against what the result actually costs. CHUNK_TOKENS is what the chunker AIMS
    # at, not what a chunk weighs: chunks overlap, the chunker's tokenizer is not the
    # model's, and the rendered block adds markup, source metadata and the tool framing
    # around it. Measured on a 500-token budget: one chunk came back as 1,256 estimated
    # tokens. So the count is halved until the rendered result fits, the same backoff the
    # forced recall uses, and a single chunk that still does not fit is refused rather
    # than appended to an exchange the window is not allowed to evict.
    if budget is not None:
        counter = scope.get("token_counter")
        attempt = max(1, int(top_k or 1))
        while True:
            rendered = _rendered_conversation_search(found)
            if _conversation_search_cost(rendered, counter) <= int(budget):
                return rendered
            if attempt <= 1:
                return "There is no room left in this context to search earlier conversation."
            attempt = max(1, attempt // 2)
            found = _recall(attempt)
            if not found:
                return "No earlier turns of this conversation matched that query."
    return _rendered_conversation_search(found)


# What a `tool` message costs beyond its own text: the role, the call id and whatever the
# template wraps them in. Small, fixed, and left out entirely before, which is the wrong
# direction on a check whose whole job is to refuse a result that will not fit.
_TOOL_MESSAGE_FRAMING_TOKENS = 8


def _conversation_search_cost(text: str, counter = None) -> int:
    """What admitting this result really costs, exactly when the caller has a tokenizer.

    The estimate below is pessimistic for CJK and emoji but still optimistic for ASCII
    that tokenises densely -- source code, minified JSON, hashes, command output all run
    nearer two or three characters per token than four -- so a result could be admitted at
    well under its real cost and then land in the current tool exchange, which the window
    is not allowed to evict. A tokenizer-backed caller passes its own counter, and the
    GGUF path is one, so the check that already computes the budget exactly can now spend
    it exactly too.
    """
    if counter is not None:
        try:
            return int(counter(text)) + _TOOL_MESSAGE_FRAMING_TOKENS
        except Exception:
            logger.debug("conversation search: exact result count failed", exc_info = True)
    return _conversation_search_tokens(text) + _TOOL_MESSAGE_FRAMING_TOKENS


def _conversation_search_tokens(text: str) -> int:
    """A deliberately pessimistic size for a search result, in tokens.

    The shared estimator charges four characters per token, which is about right for
    English and badly wrong for text that tokenises densely: CJK and emoji run closer to
    one token per character, so a result could be accepted at a quarter of its real cost
    and then land in the current tool exchange, which the window cannot evict. No exact
    counter is reachable from here, the provider loop having no tokenizer at all, so
    non-ASCII characters are charged one token each and the rest at the usual rate.
    """
    dense = sum(1 for char in text if ord(char) > 127)
    return max(1, dense + (len(text) - dense) // 4)


def _rendered_conversation_search(found) -> str:
    """The tool result exactly as the model would receive it."""
    text, sources = found
    if sources:
        import json as _json
        return text + RAG_SOURCES_SENTINEL + _json.dumps(sources, ensure_ascii = False)
    return text


def _search_knowledge_base_with_budget(
    arguments: dict,
    rag_scope: dict | None,
    timeout: int | None,
    cancel_event = None,
    search_fn = None,
) -> str:
    """Admission-controlled RAG search.

    ``search_fn`` swaps in a different search over the same capacity-of-one slot, so
    archive lookups queue behind document lookups instead of racing for the embedder."""
    search_fn = search_fn or _search_knowledge_base
    if cancel_event is not None and cancel_event.is_set():
        return "Error: knowledge base search cancelled."
    deadline = time.monotonic() + timeout if timeout is not None else None
    while not _RAG_SEARCH_SLOT.acquire(timeout = 0.05):
        if cancel_event is not None and cancel_event.is_set():
            return "Error: knowledge base search cancelled."
        if deadline is not None and time.monotonic() >= deadline:
            return "Error: knowledge base search timed out."

    # The running search owns the admission slot until it actually stops; release it exactly once,
    # from whichever path terminates the work. Releasing on caller timeout/cancel would let a
    # second search in while the first worker is still doing embedding/index/GPU work, defeating
    # the capacity-of-one bound, so the worker frees the slot in its finally instead.
    _slot_lock = threading.Lock()
    _slot_released = False

    def release_slot() -> None:
        nonlocal _slot_released
        with _slot_lock:
            if _slot_released:
                return
            _slot_released = True
        _RAG_SEARCH_SLOT.release()

    if cancel_event is not None and cancel_event.is_set():
        release_slot()
        return "Error: knowledge base search cancelled."
    if deadline is not None and time.monotonic() >= deadline:
        release_slot()
        return "Error: knowledge base search timed out."

    if timeout is None and cancel_event is None:
        try:
            return search_fn(arguments, rag_scope)
        finally:
            release_slot()

    result: queue.Queue = queue.Queue(maxsize = 1)

    def search() -> None:
        try:
            result.put((True, search_fn(arguments, rag_scope)))
        except BaseException as exc:
            result.put((False, exc))
        finally:
            release_slot()

    try:
        threading.Thread(target = search, name = "rag-tool-search", daemon = True).start()
    except Exception:
        release_slot()
        raise
    while True:
        # Caller gives up, but the worker thread still holds the slot and releases it in its
        # finally when it truly finishes -- so concurrency stays bounded to one.
        if cancel_event is not None and cancel_event.is_set():
            return "Error: knowledge base search cancelled."
        if deadline is not None and time.monotonic() >= deadline:
            return "Error: knowledge base search timed out."
        wait = 0.05
        if deadline is not None:
            wait = min(wait, max(0.001, deadline - time.monotonic()))
        try:
            ok, value = result.get(timeout = wait)
        except queue.Empty:
            continue
        if ok:
            return value
        raise value


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


def _last_searchable_text(messages):
    """The most recent EARLIER user turn that names something to search for, or None."""
    try:
        from core.inference import instruction_pin
    except Exception:
        return None
    users = [m for m in (messages or []) if m.get("role") == "user"]
    for message in reversed(users[:-1] if users else []):
        text = _last_user_text([message])
        if text and not instruction_pin.is_thin_query(text):
            return text
    return None


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


def build_synthetic_search_exchange(
    *,
    tool_name: str,
    call_prefix: str,
    status_label: str,
    query: str,
    text: str,
    sources: list[dict],
) -> dict:
    """Render a retrieval the loop never asked for as a normal tool exchange.

    Returns ``{"events": [...], "messages": [...]}``: the messages are what the model
    reads, the events what the UI draws, so a forced retrieval shows up as an ordinary
    tool card with working citations instead of appearing from nowhere.
    """
    import json as _json
    import uuid as _uuid

    call_id = call_prefix + _uuid.uuid4().hex[:12]
    args = {"query": query}
    full_result = text + RAG_SOURCES_SENTINEL + _json.dumps(sources, ensure_ascii = False)
    events = [
        {"type": "status", "text": f"{status_label}: {query[:60]}"},
        {
            "type": "tool_start",
            "tool_name": tool_name,
            "tool_call_id": call_id,
            "arguments": args,
        },
        {
            "type": "tool_end",
            "tool_name": tool_name,
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
                        "name": tool_name,
                        "arguments": _json.dumps(args, ensure_ascii = False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": tool_name,
            "tool_call_id": call_id,
            "content": text,
        },
    ]
    return {"events": events, "messages": messages}


_RECALL_BLOCK = (
    "<recalled_conversation>\n"
    "This conversation was compacted and earlier turns were removed from your context. "
    "Relevant earlier turns of this chat are quoted below, retrieved verbatim.\n"
    "{text}\n"
    "</recalled_conversation>\n\n"
)


def build_conversation_recall(
    conversation: list[dict],
    thread_id: str | None,
    *,
    style: str = "tool",
    top_k: int | None = None,
    branch_messages: list[dict] | None = None,
) -> dict | None:
    """Retrieve the archived turns most relevant to the latest user message.

    Deliberately NOT gated on ``rag_scope``: compaction happens whether or not document
    RAG is on, and these turns are the conversation's own, not uploaded files.

    Forcing this one retrieval is the whole feature. Given only a search tool, a 35B on
    MRCR v2 declined on 56% of rows, scoring 0.099 when it skipped against 0.461 when it
    searched; forcing the lookup on the evicting turn took tool-only 0.258 to 0.604, and
    the model then called the tool on 0% of rows, so it costs nothing on the common path.

    ``style="tool"`` renders a tool exchange, for the tool loop which already carries a
    tools array. ``style="inline"`` prefixes the latest user message instead, for the
    plain path: forging tool_calls without a tools array breaks strict templates.
    """
    if not thread_id:
        return None
    try:
        from core.rag import conversation_archive
    except Exception:
        return None
    if not conversation_archive.enabled():
        return None

    # The BRANCH's latest user turn, not the loop conversation's: a later tool-loop
    # iteration can end with an internal user-role re-prompt (the plan-without-action
    # nudge), and searching for that controller instruction defeats the forced retrieval.
    # branch_messages is what the client sent, so its last user turn is the real request.
    query = _last_user_text(branch_messages or conversation) or _last_user_text(conversation)
    if not query:
        return None
    # A nudge ("continue", "yes") retrieves nothing, so the user's last real instruction is
    # asked for as a SECOND query. Not applied to the model's own `search_conversation`
    # calls: the model wrote that query, and overriding it answers a different question.
    anchor = None
    thin = False
    try:
        from core.inference import instruction_pin
        thin = instruction_pin.is_thin_query(query)
        if thin:
            _behind = branch_messages or conversation
            anchor = instruction_pin.last_substantive_instruction(_behind)
            if not anchor:
                # An instruction is 80 characters; a QUERY need only name something. On a
                # first reset a thread of short prompts ("Write a story about Mars", then
                # "continue") had no anchor at all, so recall was skipped, the block
                # carried nothing (the same length rule) and the archive was written after
                # tool selection, so the model saw the nudge alone with no way to reach
                # what it was continuing. `is_thin_query` already separates "names
                # nothing" from "short", so ask it instead.
                anchor = _last_searchable_text(_behind)
    except Exception:  # noqa: BLE001 -- a query refinement must never break a chat
        anchor = None
        thin = False
    if thin and not anchor:
        # A nudge with nothing behind it: searching for "continue" returns whatever shares
        # its stopwords, and under checkpoint compaction that block is the model's FIRST
        # sight of the search tool. Skip; the tool stays available.
        logger.info(
            "Conversation recall skipped: the latest message is a nudge with no "
            "earlier instruction to search for instead"
        )
        return None
    try:
        found = conversation_archive.recall(
            thread_id,
            query,
            top_k = top_k,
            branch_messages = branch_messages,
            extra_queries = [anchor] if anchor else None,
            # This is the automatic lookup, so it is the one the quality floor applies to.
            forced = True,
        )
    except Exception:
        logger.warning("Conversation recall failed", exc_info = True)
        return None
    if not found:
        return None
    text, sources = found

    if style == "inline":
        return {
            "events": [],
            "messages": [],
            "prefix": _RECALL_BLOCK.format(text = text),
            "sources": len(sources),
        }
    built = build_synthetic_search_exchange(
        tool_name = "search_conversation",
        call_prefix = "conv_recall_",
        status_label = "Recalling earlier conversation",
        query = query,
        text = text,
        sources = sources,
    )
    built["sources"] = len(sources)
    logger.info("Conversation recall: %d earlier passage(s) for %r", len(sources), query[:80])
    return built


def rag_autoinject_reaches_retrieval(
    conversation: list[dict], rag_scope: dict | None
) -> tuple[bool, bool]:
    """Everything checked before pre-retrieval searches: switched on, something to search
    for, somewhere to search, and a store to search it in. Whether a hit then clears the
    score floor is the one part not knowable without running the search.

    Shared with token counting, which cannot run it and so must not decline a turn that
    stops short of the search here.
    """
    if not rag_scope:
        return False, False
    enabled = rag_scope.get("autoinject")
    if enabled is None:
        enabled = _autoinject_enabled()
    thread_id = rag_scope.get("thread_id")
    whole_doc_requested = (
        bool(thread_id) and not rag_scope.get("kb_id") and _thread_whole_doc_enabled(rag_scope)
    )
    if not enabled and not whole_doc_requested:
        return False, False
    # What _resolve_scope resolves to nothing: an unpersisted New Chat carries a scope with
    # none of the three ids, and the search stops there.
    if not (rag_scope.get("kb_id") or rag_scope.get("project_id") or thread_id):
        return False, False
    if not _last_user_text(conversation):
        return False, False
    try:
        from storage import rag_db

        # rag_available(), not the import flag: the vec0 native library is a separate file a
        # venv can be missing, and nothing finds out until a connection tries.
        if not rag_db.rag_available():
            return False, False
    except Exception:  # noqa: BLE001
        return False, False
    return bool(enabled), whole_doc_requested


def build_rag_autoinject(conversation: list[dict], rag_scope: dict | None) -> dict | None:
    """Pre-retrieve the latest user turn; if a hit clears the cosine floor return
    ``{"events": [...], "messages": [...]}`` to splice into the loop, else ``None``.
    Toggle via ``rag_scope.autoinject`` (else env ``RAG_AUTOINJECT``); floor via
    ``rag_scope.autoinject_min_score`` (else env ``RAG_AUTOINJECT_MIN_SCORE``).

    Also the small-model fallback: models below ~4B often answer from memory
    instead of calling ``search_knowledge_base``, so forcing retrieval here keeps
    attachments consulted regardless of model size."""
    enabled, whole_doc_requested = rag_autoinject_reaches_retrieval(conversation, rag_scope)
    if not enabled and not whole_doc_requested:
        return None
    thread_id = rag_scope.get("thread_id")
    query = _last_user_text(conversation)
    try:
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
    budget: int | None = None
    # The window the budget was sized against, so `_text_token_cost` only trusts
    # a GGUF actually serving this same window.
    ctx_tokens = (
        _opt_int(rag_scope.get("context_length") or rag_scope.get("max_context_tokens")) or 0
    )

    # Whole-document mode: a thread-attached file under budget is injected in
    # full. A KB selection is exclusive so whole-doc never preempts it; project
    # sources are still retrieved top-K and appended under one citation
    # numbering. Oversized/absent thread docs fall through to top-K below.
    if whole_doc_requested:
        try:
            budget = _whole_doc_budget(rag_scope, conversation)
            whole = whole_document_context(scope_thread_id = thread_id, max_tokens = budget)
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

    def _fits(candidate_text, max_tokens) -> bool:
        # None means the estimate itself failed, so there is nothing to enforce.
        # Zero is the opposite: a measured "no room left".
        if max_tokens is None:
            return True
        if max_tokens <= 0:
            return False
        # Priced by the serving GGUF when it can, doubled when it cannot. The
        # doubling is what stops dense ASCII (source, minified JSON, hashes, all
        # nearer two characters per token) being charged the English four.
        return _text_token_cost(candidate_text, ctx_tokens) <= max_tokens

    def _trim(hit_text, hit_sources, max_tokens):
        """Drop passages from the tail until the rendered block fits, else None.

        Re-renders only when something is dropped, so an untrimmed result comes
        back exactly as retrieval built it.

        None when not even the top passage fits: the block joins the current
        turn, which the window may not evict, so an overflowing injection fails
        the request rather than degrading the answer. Losing the attachment is
        what this branch exists to prevent, but main already loses it here, and
        that beats an error instead of an answer.
        """
        kept, rendered = list(hit_sources), hit_text
        while len(kept) > 1 and not _fits(rendered, max_tokens):
            kept = kept[:-1]
            rendered = render_sources(kept)
        return (rendered, kept) if _fits(rendered, max_tokens) else None

    def retrieve(*, max_tokens = None, **scope):
        found = search_for_autoinject(query = query, top_k = top_k, **scope)
        return _trim(found[0], found[1], max_tokens) if found else None

    # An oversized thread attachment is mandatory grounding: with auto-injection
    # off, search it alone, without the optional-auto relevance floor, then add
    # project context if the combination still fits. The budget binds on that
    # path only -- with auto-injection on this stays the single combined
    # unbudgeted search, so a small context cannot silently switch RAG off.
    if text is None and (enabled or whole_doc_requested):
        try:
            if whole_doc_requested and not enabled:
                found = retrieve(
                    max_tokens = budget,
                    scope_thread_id = thread_id,
                    min_dense_score = None,
                    **_scope_retrieval_kwargs(rag_scope),
                )
                project_id = rag_scope.get("project_id")
                if found and project_id:
                    # Isolated like the whole-document companion above: an
                    # unavailable project index must not send the shared handler
                    # below into discarding thread grounding already in hand.
                    try:
                        proj = retrieve(
                            scope_project_id = project_id,
                            min_dense_score = floor,
                            **_scope_retrieval_kwargs(rag_scope),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("RAG project retrieval (fallback companion) failed: %s", exc)
                        proj = None
                    if proj:
                        # Trim the combination, not the project alone, which was
                        # never entitled to the whole budget. The tail is all
                        # project, so what fits beside the thread result survives.
                        merged = found[1] + proj[1]
                        found = _trim(render_sources(merged), merged, budget) or found
            else:
                found = retrieve(
                    scope_kb_id = rag_scope.get("kb_id"),
                    scope_thread_id = thread_id,
                    scope_project_id = rag_scope.get("project_id"),
                    min_dense_score = floor,
                    **_scope_retrieval_kwargs(rag_scope),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG auto-inject retrieval failed: %s", exc)
            return None
        if not found:
            logger.info("RAG auto-inject: no matching passage; skipping")
            return None
        text, sources = found
    if text is None:
        return None

    built = build_synthetic_search_exchange(
        tool_name = "search_knowledge_base",
        call_prefix = "rag_auto_",
        status_label = "Searching documents",
        query = query,
        text = text,
        sources = sources,
    )
    logger.info("RAG auto-inject: %d passage(s) for %r", len(sources), query[:80])
    return built


_MAX_PAGE_CHARS = 16000  # cap fetched page text (after HTML-to-MD conversion)

# Share of the loaded window one fetched page may claim. The same window also has to hold
# the system prompt, the carried-forward block, the user's turn, the call itself and room
# to answer, so a third is already generous.
_PAGE_CONTEXT_SHARE = 0.35
# Below this a page is too clipped to answer from, so the fetch is not worth making small.
# Half, when the room has to be converted to characters with no way to check the answer.
# See `_dense_char_limit`: the conversion charges ASCII an English four characters per
# token and the dense ASCII these tools print runs nearer two.
_UNMEASURED_ROOM_MARGIN = 0.5

_MIN_PAGE_CHARS = 2000
# A percent-escape is one non-ASCII byte written in ASCII, and tokenises like one.
_HEX_PAIR_RE = re.compile(r"[0-9A-Fa-f]{2}")
# Raw download cap > _MAX_PAGE_CHARS since SSR pages embed large <head> sections
# stripped during conversion; 512 KB still reaches article content.
_MAX_FETCH_BYTES = 512 * 1024
# "%" is safe so an already-encoded URL is not re-encoded into %25.
_IRI_PATH_SAFE = "/%:@!$&'()*+,;="
_IRI_QUERY_SAFE = "/%:@!$&'()*+,;=?"
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


def _explicit_proxy_applies(scheme: str, host: str) -> bool:
    """Whether urllib routes a *scheme* request for *host* through a proxy.

    Only a proxied fetch may keep the hostname in the request URL: the proxy
    resolves it, so this host never looks it up again. A direct one would, which
    is the DNS-rebinding window, so it stays pinned to the validated IP.

    *host* must be the ``host[:port]`` form ``Request.host`` carries, since that
    is what ``ProxyHandler`` passes to ``proxy_bypass``; probing the bare hostname
    instead would disagree with it on a port-qualified NO_PROXY entry.
    """
    from urllib.request import getproxies, proxy_bypass

    # ProxyHandler lowercases every mapping key, and the Windows registry can hand
    # back "HTTPS=...", so normalize before testing or a proxy-only host goes direct.
    if scheme not in {key.lower() for key in getproxies()}:
        return False
    try:
        return not proxy_bypass(host)
    except (OSError, ValueError):
        # proxy_bypass reads system config on macOS/Windows; failure falls back to pinning.
        return False


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
    except (OSError, UnicodeError) as e:
        # IDNA encoding rejects a hostname with UnicodeError, not OSError.
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


_DOTTED_HOST_RE = re.compile(r"[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+")
# ASCII-only because str.isdigit() is True for digits int() refuses ("²"), and
# capped at 5 digits so the range check never converts an unbounded integer.
_PORT_RE = re.compile(r"[0-9]{1,5}")


def _normalize_url_scheme(url: str) -> str:
    """Prepend ``https://`` to bare hosts (``google.com``, ``example.com:8443``).

    ``urlparse`` reads the host of a ``host:port`` input as the scheme, so those
    are recognised by a dotted host-like scheme with an empty netloc. Rewrites a
    dotted host with an optional in-range port, and the ``//host`` form. Real
    schemes (``file:``, ``javascript:``, including ``file:80``), root-relative
    paths (``/login``) and bad ports are returned untouched so the caller
    rejects them. A dotted scheme is indistinguishable from ``host:port``, so
    ``com.acme.app:443/cb`` is rewritten too; an empty port (``example.com:``)
    is kept as-is, matching ``https://example.com:``.

    The host is matched against the raw authority, never against what
    ``urlparse`` returned, because urlsplit strips tabs/newlines (3.10) and
    leading C0/space (3.12). Anything it would strip fails the match, so the
    decision and the rewritten string cannot disagree across versions."""
    from urllib.parse import urlparse

    url = url.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        # Unmatched IPv6 brackets, or an NFKC-decomposing netloc: not a bare host.
        return url
    if parsed.scheme:
        if parsed.netloc or not _DOTTED_HOST_RE.fullmatch(parsed.scheme):
            return url
        rest = url
    elif url.startswith("//"):
        rest = url[2:]
    elif url.startswith("/"):
        return url
    else:
        rest = url

    authority = re.split(r"[/?#]", rest, maxsplit = 1)[0]
    host, _, port = authority.partition(":")
    if not _DOTTED_HOST_RE.fullmatch(host):
        return url
    if port and not (_PORT_RE.fullmatch(port) and 1 <= int(port) <= 65535):
        return url
    return "https://" + rest


def _fetch_url_raw(
    url: str,
    timeout: int = 30,
    extra_headers: dict | None = None,
    deadline: float | None = None,
    cancel_event = None,
    website_policy: dict | None = None,
    raw_bytes_max: int | None = None,
) -> tuple[str | None, "str | bytes", str]:
    """Fetch a URL with SSRF protection; return ``(error, body_text, content_type)``.

    ``raw_bytes_max`` switches to binary mode: the body is returned as ``bytes``
    untouched (no PDF or text handling) and refused past that many bytes. The
    same scheme, host, redirect and budget gates apply either way.

    ``error`` is a user-facing message string when the fetch failed (the
    existing "Blocked:" / "Failed to fetch URL:" wording), else ``None``.
    Blocks private/loopback/link-local targets and caps the download size.
    No input reaches the caller as an exception: the URL is model-supplied, so
    every malformed form resolves to one of these strings.

    ``deadline`` is an optional ``time.monotonic`` cutoff for the whole fetch
    (redirect hops and body read included) and ``cancel_event`` aborts it when
    the caller goes away; both default off so callers keep the old behavior.
    """
    from urllib.parse import urlparse
    from .web_access_policy import check_url_access

    # Before the policy gate: it requires an http(s) scheme, so a bare host
    # would be refused there and never reach the fetch.
    url = _normalize_url_scheme(url)
    allowed, reason, canonical_host = check_url_access(url, website_policy)
    if not allowed:
        return reason, "", ""

    # check_url_access already parsed this and read .port, so this cannot raise.
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ok, reason, pinned_ip = _resolve_with_budget(
        canonical_host,
        port,
        deadline,
        cancel_event,
    )
    if not ok:
        return reason, "", ""

    try:
        from urllib.error import HTTPError as _HTTPError
        from urllib.parse import quote, urljoin, urlunparse

        max_bytes = _MAX_FETCH_BYTES
        current_url = url
        current_host = canonical_host
        ua = random.choice(_USER_AGENTS)

        for _hop in range(5):
            budget_error = _fetch_budget_exceeded(deadline, cancel_event)
            if budget_error is not None:
                return budget_error, "", ""
            cp = urlparse(current_url)
            # http.client rejects a non-ASCII selector outright.
            cp = cp._replace(
                path = quote(cp.path, safe = _IRI_PATH_SAFE),
                params = quote(cp.params, safe = _IRI_PATH_SAFE),
                query = quote(cp.query, safe = _IRI_QUERY_SAFE),
            )
            # Bracket IPv6 so the netloc stays a valid URL.
            validated_netloc = f"[{current_host}]" if ":" in current_host else current_host
            if cp.port:
                validated_netloc = f"{validated_netloc}:{cp.port}"
            # Decide routing once, on the netloc urllib tests: a pinned request
            # carries an IP, which no NO_PROXY entry matches, so the opener below
            # has to carry the decision rather than re-derive it.
            proxied = _explicit_proxy_applies(cp.scheme, validated_netloc)
            if os.environ.get(_DISABLE_DNS_PINNING_ENV) == "1" and proxied:
                # Enterprise proxies need the hostname in CONNECT for policy and TLS
                # interception, and they resolve it, so nothing rebinds behind us.
                request_url = urlunparse(cp._replace(netloc = validated_netloc))
            else:
                # Pin to the validated IP to prevent DNS rebinding.
                ip_str = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
                ip_netloc = f"{ip_str}:{cp.port}" if cp.port else ip_str
                request_url = urlunparse(cp._replace(netloc = ip_netloc))

            handlers = [_NoRedirect, _SNIHTTPSHandler(current_host)]
            if not proxied:
                # An empty ProxyHandler is the documented way to opt a request out.
                handlers.append(urllib.request.ProxyHandler({}))
            opener = urllib.request.build_opener(*handlers)

            headers = {
                "User-Agent": ua,
                "Host": validated_netloc,
            }
            if extra_headers:
                headers.update(extra_headers)
            req = urllib.request.Request(request_url, headers = headers)
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
                # Server-controlled, so never scheme-upgraded; the gate below
                # reads .port first, so the parse after it cannot raise.
                allowed, policy_reason, redirect_host = check_url_access(
                    current_url,
                    website_policy,
                )
                if not allowed:
                    return policy_reason, "", ""
                rp = urlparse(current_url)
                rp_port = rp.port or (443 if rp.scheme == "https" else 80)
                ok2, reason2, pinned_ip = _resolve_with_budget(
                    redirect_host,
                    rp_port,
                    deadline,
                    cancel_event,
                )
                if not ok2:
                    return reason2, "", ""
                current_host = redirect_host
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
            declared_pdf = raw_bytes_max is None and content_type == "application/pdf"
            if raw_bytes_max is not None:
                read_limit = raw_bytes_max + 1
            elif declared_pdf:
                read_limit = _MAX_PDF_FETCH_BYTES + 1
            else:
                read_limit = max_bytes
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
            if raw_bytes_max is not None:
                if len(raw_bytes) > raw_bytes_max:
                    return f"(content exceeds the {raw_bytes_max} byte limit)", "", content_type
                return None, raw_bytes, content_type
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


def _loaded_context_tokens() -> int | None:
    """The active model's context window, or None when it cannot be read.

    Mirrors `research_runs._loaded_context_length` and `routes.inference.
    _monitor_context_length`: llama.cpp first, then the orchestrator the API layer reads.
    Both branches are needed. A native/Transformers chat leaves `is_loaded` false, and
    stopping at that probe reported "unknown", which kept the full 16,000-character cap
    and reproduced on small native models exactly the overflow this budget exists to
    prevent.

    The ML backends live in a worker subprocess, so the in-process singleton is
    unpopulated here and importing it pulls in the ML stack; peek at the orchestrator
    instead of constructing one. Every failure is "unknown" so a fetch is never blocked by
    not knowing.
    """
    try:
        from routes.inference import get_llama_cpp_backend  # noqa: PLC0415
        llama = get_llama_cpp_backend()
        if getattr(llama, "is_loaded", False):
            ctx = getattr(llama, "context_length", None)
            if isinstance(ctx, int) and ctx > 0:
                return ctx
    except Exception:  # noqa: BLE001 -- an unreadable window is "unknown", never an error
        pass
    try:
        from core.research_runs import _peek_inference_backend  # noqa: PLC0415

        backend = _peek_inference_backend()
        name = getattr(backend, "active_model_name", None)
        models = getattr(backend, "models", {}) or {}
        info = models.get(name) if (name and isinstance(models, dict)) else None
        for candidate in (
            (info or {}).get("context_length"),
            getattr(backend, "context_length", None),
            getattr(backend, "max_seq_length", None),
        ):
            if isinstance(candidate, int) and candidate > 0:
                return candidate
    except Exception:  # noqa: BLE001 -- same rule: unknown, never an error
        return None
    return None


def _result_char_budget(cap: int) -> int:
    """`cap`, lowered to what the serving window can actually hold.

    Shared by fetched pages and by terminal/python results, because the failure is the
    same: a fixed character cap has no relation to the loaded context, so on a small
    window one result fills most of it. That result lands in the NEWEST turn, which the
    fit protects, so compaction cannot drop the very thing that does not fit and the
    request goes irreducible. Measured live on a 5120-token window: 7043 and 6684 token
    requests refused, both on the code tools, whose 16,000-character cap is about 4,000
    tokens on its own.
    """
    scoped = _REQUEST_CONTEXT_TOKENS.get()
    # An explicit 0/None means "asked, and unknowable" (external provider), and must NOT
    # fall through to the probe. Only an absent value keeps the process-global read.
    ctx = _loaded_context_tokens() if scoped is _UNSET_CONTEXT_TOKENS else scoped
    if not ctx:
        return cap
    # Clamped to `cap` on the way out, not only on the way in. The floor keeps a result
    # worth reading when the WINDOW is the thing making it small; it is not a licence to
    # hand the model more than the install configured. Unclamped, an install running
    # `UNSLOTH_TOOL_RESULT_MAX_CHARS=500` got 500 characters from the hosted path and
    # 2,000 from this one, the moment a local window became readable -- the one function
    # whose job is to LOWER the cap raising it fourfold instead.
    return min(cap, max(_MIN_PAGE_CHARS, int(ctx * 4 * _PAGE_CONTEXT_SHARE)))


def _tool_result_char_budget() -> int:
    """The terminal/python cap, sized to the window. See `_result_char_budget`."""
    return _result_char_budget(_MAX_OUTPUT_CHARS)


def _page_char_budget() -> int:
    """`_MAX_PAGE_CHARS`, lowered to what the serving window can actually hold.

    16,000 characters is roughly 4,000 tokens: fine on a 128k model, nonsensical on a
    4,864-token one. Measured there, a single fetched page came back at 12,295 characters,
    the request went irreducible at 8,995 tokens against a 3,648-token budget with
    `latest_turn_role: "tool"`, and the user was advised to shorten a conversation
    consisting of one 11-token question. Nothing downstream can recover from it either:
    the fit protects the newest turn, so compaction may not drop the very result that does
    not fit.

    Above roughly an 11k window this returns the old constant unchanged, so only the models
    that cannot afford a whole page are affected.
    """
    scoped = _REQUEST_CONTEXT_TOKENS.get()
    # An explicit 0/None means "asked, and unknowable" (external provider), and must NOT
    # fall through to the probe. Only an absent value keeps the process-global read.
    ctx = _loaded_context_tokens() if scoped is _UNSET_CONTEXT_TOKENS else scoped
    if not ctx:
        return _MAX_PAGE_CHARS
    return max(_MIN_PAGE_CHARS, min(_MAX_PAGE_CHARS, int(ctx * 4 * _PAGE_CONTEXT_SHARE)))


def _request_result_room() -> int | None:
    """Tokens this result may add before the NEXT prompt is over budget.

    None when the caller could not say, and every cap then behaves exactly as it did
    before this existed: external providers, the hosted path and any tool loop that does
    not price its own conversation all take that leg.
    """
    room = _REQUEST_RESULT_BUDGET.get()
    if room is None:
        return None
    try:
        return max(0, int(room))
    except (TypeError, ValueError):
        return None


def _window_context_tokens() -> int | None:
    """The window this request is served by, or None when it cannot be read."""
    scoped = _REQUEST_CONTEXT_TOKENS.get()
    # An explicit 0/None means "asked, and unknowable" (external provider), and must NOT
    # fall through to the probe. Only an absent value keeps the process-global read.
    ctx = _loaded_context_tokens() if scoped is _UNSET_CONTEXT_TOKENS else scoped
    return ctx if ctx else None


def _dense_prefix_chars(text: str, token_budget: float) -> int:
    """How many leading characters of `text` cost at most `token_budget` tokens.

    Four characters per token is an English rate. Measured with Qwen3, Llama 3.2 and
    tiktoken on real fetched pages, CJK prose runs 1.3-1.6 characters per token, and the
    percent-escaped links a CJK page is full of (`%E7%9F%A5`) run 1.3-1.5: both are the
    same non-ASCII bytes, one spelled in ASCII. Charging them a token each, the rule
    `context_window.estimate_messages_tokens_dense` already uses, keeps the share the
    caller asked to reserve a share instead of the whole budget.

    One pass, so it costs nothing next to the fetch it sizes.
    """
    spent = 0.0
    index = 0
    length = len(text)
    while index < length:
        start = index
        if text[index] == "%" and _HEX_PAIR_RE.match(text, index + 1):
            spent += 3.0  # a non-ASCII byte spelled in ASCII; charge it like one
            index += 3
        else:
            spent += 1.0 if ord(text[index]) > 127 else 0.25
            index += 1
        # Cut on whole characters (and whole escapes), so the tail is never half a
        # percent-escape the model has to guess at.
        if spent > token_budget:
            return start
    return length


# `count_chat_tokens` prices a chunk by rendering it through the model's chat template
# (/apply-template) and tokenizing the result, so the probe can only measure text the
# template actually RENDERS. A standalone tool message is not that text: the Gemma-4
# templates shipped in `assets/chat_templates` skip it outright -- `gemma-4.jinja:232` is
# `{%- if message['role'] != 'tool' -%}`, and a tool result is only emitted while scanning
# forward from an assistant tool call -- so a 600-character payload rendered to 46
# characters with the payload absent, and 7,168 characters of base64 priced as ~12 tokens
# of framing sailed under any budget on the first pass. A user turn is rendered by every
# template checked: both bundled Gemma-4 templates, Qwen3, Llama-3.2, Mistral and
# Hermes-3. The assistant-tool-call pair is not a safe alternative -- Mistral's template
# raises on any tool call id that is not nine alphanumeric characters.
_PROBE_ROLE = "user"

# The guard below: how few tokens a rendered chunk may cost before the count is treated as
# not having measured it. Deliberately far past anything real text reaches -- the densest
# packing measured with Qwen3 is 128 characters per token, for a chunk of nothing but
# spaces, and ordinary output runs 1-8. A template that drops the content lands at
# hundreds, or at infinity as the chunk grows, because its count does not move at all.
_MAX_PROBE_CHARS_PER_TOKEN = 256

# A measured count is a pure function of (model, chat template, window, chunk), and each
# `count_chat_tokens` is two llama-server calls (/apply-template then /tokenize) over a
# fresh connection. So the framing baseline -- one number for EVERY result the process
# truncates -- is worth remembering, and so is any prefix already priced.
#
# Keyed on the resident llama-server process, because the count depends on the EFFECTIVE
# chat template and the managed fields cannot reconstruct it: user pass-through args are
# appended verbatim after Unsloth's own flags (`llama_cpp.py`, "User pass-through args go
# last") and llama.cpp is last-wins, so `--chat-template` in extra args renders through a
# template `_chat_template_override` never sees. Reload the same GGUF into the same window
# with only those args changed and every managed field matches while the rendering does
# not, which would price a prefix by a template no longer serving it. `is_loaded` is
# `self._process is not None and self._healthy` and args reach llama-server only on its
# command line, so any change to them is a new process by construction -- which settles it
# without enumerating the flags that matter. The content fields ride along so a recycled
# pid still has to agree on everything before a count is reused.
_PROBE_COUNT_CACHE: dict = {}

# Tool calls run in worker threads (`tool_stream_exec.stream_tool_execution` runs each
# invocation in one), so concurrent chats reach this cache at the same time. A bare dict
# assignment is atomic under the GIL, but the LRU touch and the eviction below are
# read-then-mutate sequences and are not: measured with 24 threads over a 3-entry cache,
# `cache.pop(chunk)` raised KeyError after another thread evicted the same key, `del
# cache[victim]` raised on a victim already taken, and choosing a victim raised
# "dictionary changed size during iteration" -- 90 exceptions in one run, none of them
# caught on the way out of `_truncate`.
#
# Held only across the dict work, never across a `count_chat_tokens` call. Serialising the
# round trips themselves would trade a shared cache for a shared queue, which is the
# opposite of the point. Two threads may therefore measure the same chunk at once and both
# store it; the value is the same either way, so that costs one duplicate measurement,
# which is what the merge base did on every result anyway.
_PROBE_COUNT_LOCK = threading.Lock()

# The empty chunk: the framing baseline, and the entry eviction pins. Named so the two
# places that treat it specially cannot drift apart from a bare "".
_PROBE_BASELINE = ""

# One model's counts at a time (a new identity clears the map). Ten times the worst case
# for one result: `_EXACT_FIT_PASSES` prefixes plus the baseline.
_PROBE_COUNT_CACHE_ENTRIES = 64

# And a bound on what is HELD, since the entry count alone does not give one. Only a
# fetched page is capped at `_MAX_PAGE_CHARS`; a tool result's prefix is bounded by
# `min(UNSLOTH_TOOL_RESULT_MAX_CHARS, ctx * 4 * _PAGE_CONTEXT_SHARE)` and `_env_int`
# accepts any positive integer, so a large configured cap on a large window makes one
# prefix enormous. Measured: a cap of 1,000,000 on a 262k window cached 733,971 characters
# from a SINGLE result, which 64 entries would then multiply. This also drops an oversized
# prefix rather than storing it. The baseline is 0 characters, so the entry that earns the
# most is never the one squeezed out.
_PROBE_COUNT_CACHE_CHARS = 1_000_000


def _probe_identity(llama, ctx: int):
    """A key that changes whenever a measured count could, or None to disable the cache.

    None is the safe answer: it costs round trips, it never returns a stale number.
    """
    try:
        # The resident llama-server. No process is no key: a backend this module cannot
        # tie a count to keeps paying for its round trips, which is the safe direction.
        pid = getattr(getattr(llama, "_process", None), "pid", None)
        if not isinstance(pid, int):
            return None
        key = (
            pid,
            ctx,
            getattr(llama, "model_identifier", None),
            getattr(llama, "_gguf_load_identity", None),
            getattr(llama, "_chat_template_override", None),
            # The gap the process id closes, spelled out: whatever the user appended to
            # the command line, including a template that overrides the managed one.
            tuple(getattr(llama, "_extra_args", None) or ()),
        )
        hash(key)  # an unhashable field is also "do not cache", not a TypeError upstream
        return key
    except Exception:  # noqa: BLE001 -- an unreadable identity is "do not cache"
        return None


def _probe_cache(llama, ctx: int) -> dict:
    """The count cache for the model serving this request.

    A fresh per-call dict when the model has no identity, so the caller's code path is the
    same either way and an unidentifiable backend simply gets no reuse between calls.
    """
    identity = _probe_identity(llama, ctx)
    if identity is None:
        return {}
    with _PROBE_COUNT_LOCK:
        cache = _PROBE_COUNT_CACHE.get(identity)
        if cache is None:
            # A different model is serving now. Drop the previous one's numbers rather than
            # keep them around to be matched against.
            _PROBE_COUNT_CACHE.clear()
            cache = _PROBE_COUNT_CACHE[identity] = {}
    return cache


def _neutralized_for_prompt(chunk: str, llama) -> str:
    """``chunk`` as the outgoing request will carry it, rather than as it is in hand.

    The same sweep the request path applies, through the same helper and the backend's own
    markup profile, so the two cannot disagree about what a marker becomes. `_PROBE_ROLE`
    is a user role, which takes the full control rewrite a tool result takes rather than
    the boundary-only one an assistant turn takes.

    Best effort: a sweep that cannot run leaves the text as it was, which is the estimate
    this had before and never worse.
    """
    if not chunk:
        return chunk
    try:
        from .chat_template_helpers import neutralize_control_markup_in_messages  # noqa: PLC0415

        swept = neutralize_control_markup_in_messages(
            [{"role": _PROBE_ROLE, "content": chunk}],
            None,
            getattr(llama, "markup_profile", None),
        )
        content = swept[0].get("content") if swept else None
        return content if isinstance(content, str) else chunk
    except Exception:  # noqa: BLE001 -- measuring is never fatal
        logger.debug("result budget: markup sweep failed", exc_info = True)
        return chunk


def _loaded_token_counter(ctx: int):
    """The tokenizer of the model serving this request, or None when there is not one.

    Same probe as `_loaded_context_tokens`: whatever can answer for the window can also
    price a string exactly, and `llama_cpp` already hands this same counter to the RAG
    admission check for exactly this reason. Gated on the backend's own window matching
    the one the budget was sized against, so a resident GGUF never prices a request that
    a different model (native, or an external endpoint) is actually answering.
    """
    try:
        from routes.inference import get_llama_cpp_backend  # noqa: PLC0415

        llama = get_llama_cpp_backend()
        if not getattr(llama, "is_loaded", False):
            return None
        if getattr(llama, "context_length", None) != ctx:
            return None
        counter = getattr(llama, "count_chat_tokens", None)
        if not callable(counter):
            return None
    except Exception:  # noqa: BLE001 -- no tokenizer is "unknown", never an error
        return None

    cache = _probe_cache(llama, ctx)
    # Whether anything said here will outlive this call, and whether `/apply-template` has
    # already refused once. Both only gate the strict attempt, never a returned value.
    retained = bool(_probe_identity(llama, ctx))
    template_down: list[bool] = []

    def _remember(chunk: str, value: int) -> None:
        """Hold `value` for `chunk`, evicting least-recently-used entries to stay in bounds.

        Refusing new entries once full was worse than not caching at all. Most tool results
        are one-offs, so the first `_PROBE_COUNT_CACHE_ENTRIES` distinct prefixes froze the
        cache on text that would never be asked about again -- and because the baseline is
        only priced when a count comes in OVER budget, a process that handled 64 results
        that FIT first locked it out for good. Measured: after 64 English results, every
        later dense result paid 4 counter calls (8 HTTP) again, exactly the merge base's
        cost, for the life of the process.

        So evict, and pin the baseline: it is 0 characters, it is the same number for every
        result this process truncates, and it is the one entry a bounded cache most needs.
        """
        if len(chunk) > _PROBE_COUNT_CACHE_CHARS:
            return  # too large to hold at all; evicting the rest would not help
        with _PROBE_COUNT_LOCK:
            held = sum(map(len, cache))
            while len(cache) >= _PROBE_COUNT_CACHE_ENTRIES or (
                held + len(chunk) > _PROBE_COUNT_CACHE_CHARS
            ):
                # `list()` so the scan cannot trip over another thread's insert, and
                # `pop(..., None)` so a victim someone else already took is not an error.
                victim = next((key for key in list(cache) if key != _PROBE_BASELINE), None)
                if victim is None:
                    return  # only the pinned baseline is left, and it stays
                held -= len(victim)
                cache.pop(victim, None)
            cache[chunk] = value

    def _rendered(chunk: str):
        # Priced as the request will really carry it. A tool result is swept for control
        # markup before it is sent (`neutralize_control_markup_in_messages`, #7066), and
        # the sweep costs tokens: a live `<|eot_id|>` is one special token raw and several
        # ordinary ones once it has been broken up. A result full of them measured on the
        # raw text fits the room here and does not fit the prompt that follows, which is
        # the overflow this budget exists to prevent, reached through the leg that is
        # supposed to be the accurate one.
        chunk = _neutralized_for_prompt(chunk, llama)
        with _PROBE_COUNT_LOCK:
            hit = cache.get(chunk)
            if hit is not None and chunk != _PROBE_BASELINE:
                # Most recently used moves to the back. `pop(..., None)` and the re-check
                # keep this a no-op rather than a KeyError if it lost a race to an evictor.
                if cache.pop(chunk, None) is not None:
                    cache[chunk] = hit
        if hit is not None:
            return hit
        # Strict, so that a count is only retained when the chat template really rendered
        # it. With `strict = False` a failed `/apply-template` still returns the plain-text
        # fallback, which drops role markers and special tokens -- fine as a one-off answer,
        # but it prices a prompt the model will never be sent, and caching it would let one
        # bad moment quietly under-count that prefix for the life of the process.
        #
        # Asked at most once per counter, and not at all when nothing would be retained
        # anyway. Strictness exists only to decide whether a count may be KEPT, so paying
        # for it twice would spend round trips to answer a question already settled: a
        # template that would not render is not going to start, and this whole change is
        # about not making calls that cannot change an answer. So a template outage costs
        # one extra attempt for the first probe of a result rather than one for every probe.
        message = [{"role": _PROBE_ROLE, "content": chunk}]
        rendered = False
        if retained and not template_down:
            try:
                spent = counter(message, None, None, strict = True)
                rendered = True
            except Exception:  # noqa: BLE001 -- not fatal: the fallback still prices bytes
                template_down.append(True)
        if not rendered:
            try:
                spent = counter(message, None, None, strict = False)
            except Exception:  # noqa: BLE001 -- now it is: fall back to the estimate
                logger.debug("result budget: exact count failed", exc_info = True)
                return None
        value = int(spent) if isinstance(spent, (int, float)) and spent > 0 else None
        # The fallback's count is USED, exactly as before -- it still tokenizes the real
        # bytes, which is what catches dense ASCII, and the estimate it would otherwise fall
        # back to undercharges base64 several fold. It is simply not retained: like a
        # failure, it is a property of the moment rather than of the text.
        if value is not None and rendered:
            _remember(chunk, value)
        return value

    # What the turn costs with nothing in it: the baseline the guard measures growth
    # against, so a template that renders no content is caught by its count not moving
    # rather than by a guess about density. Left IN the total rather than subtracted -- 8
    # tokens on Qwen3 and 11 on the Gemma-4 templates, under 1% of a 1,792-token share, and
    # the real tool turn pays its own framing anyway, so counting it errs toward a smaller
    # result. Priced on demand: see `_count`.
    baseline: list[int] = []

    def _framing() -> int:
        if not baseline:
            baseline.append(_rendered(_PROBE_BASELINE) or 0)
        return baseline[0]

    def _count(chunk: str, token_budget: float = 0.0):
        """Tokens for `chunk`, or None when the count did not measure it.

        `token_budget` is an optimisation and nothing more. A count within budget and a
        count the guard rejects both make the caller return its own estimate unchanged, so
        when `spent` fits, the baseline that separates those two paths cannot change the
        answer and is not priced -- which is why an English result costs one round trip
        rather than two. The default of 0 means "no budget", so the guard always runs.
        """
        spent = _rendered(chunk)
        if spent is None:
            return None
        if spent <= token_budget:
            return spent
        # A count that barely moves off the framing measured nothing, whatever it reports.
        framing = _framing()
        if spent - framing < len(chunk) / _MAX_PROBE_CHARS_PER_TOKEN:
            logger.debug(
                "result budget: template priced %d chars at %d tokens over %d of framing; "
                "not a measurement, keeping the estimate",
                len(chunk),
                spent,
                framing,
            )
            return None
        return spent

    return _count


# Measured: English costs one pass (it fits on the first count), base64 two, and a mixed
# result -- dense output followed by prose -- three, with the last as slack. Bounded
# rather than a binary search because each pass is a llama-server round trip.
_EXACT_FIT_PASSES = 5


def _exact_prefix_chars(
    text: str,
    chars: int,
    token_budget: float,
    ctx: int,
    floor: int | None = None,
) -> int:
    """`chars`, shrunk until the prefix really costs `token_budget`. Never grown.

    The estimate below charges every ASCII character a flat 0.25 tokens, which is an
    English rate and wrong in the same direction for the ASCII the code tools print most:
    measured with Qwen3-4B and Llama-3.2 on a 5,120-token window, where the character cap
    admits 7,168 characters against a 1,792-token share, `base64 payload.bin` came back at
    5,361 tokens, `hexdump -C` at 5,540 and `sha256sum *` at 5,109 -- 105-108% of the
    WHOLE window, in the newest turn, which the fit protects. A four-message thread (one
    8-token question and one such result) was refused irreducible at 5,475 tokens against
    a 3,840-token prompt budget. No character rule closes that: the same rule that charges
    a 76-character base64 line its real 57 tokens charges English prose 40% more than it
    costs and shrinks every page that was already fine. So when a tokenizer is serving the
    request, ask it; when none is, keep the estimate exactly as it was.
    """
    # `floor` is the caller's, not this function's: when a thread has 100 tokens left, the
    # 2,000-character comfort floor is 666 tokens of dense output and the overflow this
    # whole path exists to prevent. Defaulted, so the page callers are unchanged.
    if floor is None:
        floor = _MIN_PAGE_CHARS
    # An estimate already at or below the floor the caller guarantees cannot be improved
    # on, so nothing measured here could change its answer. Every value below is at most
    # `chars` or is exactly `floor`, so with `chars <= floor` the caller lands on the same
    # number whichever branch is taken. Checked before the counter is even looked up: this
    # is the small-window case, and it used to spend a full set of round trips
    # rediscovering a number the caller already had. Against the caller's floor rather
    # than `_MIN_PAGE_CHARS` itself, since a room-aware caller passes a lower one and
    # returning early on the legacy constant would skip the measurement it asked for.
    if chars <= floor:
        return chars
    counter = _loaded_token_counter(ctx)
    if counter is None:
        return chars
    # Every value this returns is either a MEASURED fit, the caller's own estimate (when
    # nothing could be measured), or the floor. A proportional shrink assumes the retained
    # prefix keeps the average density of the whole, which is false for the shape the code
    # tools produce most: dense output followed by prose. Cutting prose off a
    # base64-then-English result raises the density of what is left, so each pass gains
    # less than it asked for and a fixed pass count used to hand back the last shrink
    # unmeasured -- 3,497 characters costing 1,978 tokens against a 1,792-token share
    # (110%), measured with Qwen3-4B, which is the irreducible overflow this budget exists
    # to prevent.
    previous = None  # the last (chars, tokens) pair, for the secant step below
    for _ in range(_EXACT_FIT_PASSES):
        # The budget goes with the chunk so a count that already fits can skip pricing the
        # framing baseline it would only be compared against. See `_count`.
        spent = counter(text[:chars], token_budget)
        if spent is None:
            return chars  # nothing to measure with; the estimate stands, as before
        if spent <= token_budget:
            return chars  # measured, not assumed
        fitted = int(chars * token_budget / spent)
        if previous is not None:
            # Two measurements price the TAIL that was cut rather than the whole prefix,
            # which is what the proportional step gets wrong. Take whichever is smaller:
            # this only ever shrinks faster, never grows.
            prior_chars, prior_spent = previous
            per_char = (prior_spent - spent) / (prior_chars - chars)
            if per_char > 0:
                fitted = min(fitted, chars - int((spent - token_budget) / per_char))
        previous = (chars, spent)
        # The floor still applies, and stopping here saves a round trip that cannot
        # change the answer.
        if fitted <= floor:
            return floor
        chars = min(fitted, chars - 1)  # always progress, so the loop cannot stall
    # Out of passes with the last shrink still unmeasured. Returning it would be the
    # unchecked prefix above, so fall back to the floor the caller guarantees anyway.
    return floor


def _can_measure_tokens(ctx: int, text: str) -> bool:
    """Whether this request's tokens can really be counted, not merely whether a counter
    is exposed.

    `_loaded_token_counter` returns a callable that answers None whenever the probe does
    not come back with a number: `/apply-template` failing, or a chat template that drops
    the probe role, or a backend that stopped serving between the check and the call.
    `_exact_prefix_chars` then hands back the caller's estimate untouched, which charges
    plain ASCII the English four characters per token; base64, minified JSON and hashes
    run nearer two, so a room that was never halved is spent about twice over. A counter
    that cannot measure has to be treated exactly like a counter that is not there.

    Probed on this text's own opening rather than a constant, so a template that refuses
    some content and not other content is judged on what is actually being sized.
    """
    counter = _loaded_token_counter(ctx)
    if counter is None:
        return False
    return counter(text[:_MEASURABILITY_PROBE_CHARS] or "x") is not None


# Enough to render as a real message and cheap enough to price on every call.
_MEASURABILITY_PROBE_CHARS = 64


def _text_token_cost(text: str, ctx: int) -> float:
    """What ``text`` really costs, measured when the serving model can measure it.

    The estimate is the inverse of `_dense_prefix_chars`: ASCII at the English four
    characters per token, everything else at one. Doubled when nothing can check it, for
    the same reason `_UNMEASURED_ROOM_MARGIN` halves a room that cannot be measured.
    """
    counter = _loaded_token_counter(ctx) if ctx else None
    measured = None
    if counter is not None:
        try:
            spent = counter(text)
            measured = None if spent is None else float(spent)
        except Exception:
            logger.debug("token count failed", exc_info = True)
    if measured is not None:
        return measured
    # A counter that could not answer is a counter that is not there: taking its presence
    # as proof the estimate is safe is what leaves dense ASCII priced at the English rate.
    estimate = sum(0.25 if character.isascii() else 1.0 for character in text)
    return estimate / _UNMEASURED_ROOM_MARGIN


def _dense_char_limit(
    text: str,
    max_chars: int,
    reserve_tokens: float = 0.0,
) -> int:
    """`max_chars`, lowered when `text` tokenises denser than four characters per token.

    Without this the window-derived caps above reserve their share only for English. On
    the 4,864-token window this PR was measured against, the 6,809-character page budget
    is 35% of the window in English and 3,800-4,500 real tokens of a Chinese or Japanese
    page: 80-95% of the whole prompt budget, in the newest turn, which the fit protects.
    That is the same irreducible refusal the budget exists to prevent.
    """
    ctx = _window_context_tokens()
    room = _request_result_room()
    if room is None and (not ctx or len(text) <= _MIN_PAGE_CHARS):
        # Nothing measured, so the caller's cap is the only budget there is, and the
        # reserve comes off it at the same four characters per token used everywhere else
        # the real rate is unknown.
        return max(0, max_chars - int(reserve_tokens * 4))
    if room is not None and not _can_measure_tokens(ctx or 0, text):
        # Nothing here can measure this model's tokens: `_can_measure_tokens` answers only
        # for a resident GGUF that just proved it can price a string, and a native
        # safetensors model is served through a loop with no rolling fit to recover if the
        # estimate is wrong. The estimate below
        # charges plain ASCII four characters per token, which is an English rate; base64,
        # minified JSON and hashes run nearer two, so the room could be spent twice over.
        # Halved so the optimistic rate becomes a pessimistic one. It costs a shorter
        # result on a path that cannot check its own arithmetic, which is the side to be
        # wrong on when being wrong the other way is an unrecoverable turn.
        room = int(room * _UNMEASURED_ROOM_MARGIN)
    # Kept a float, so English text lands on exactly the character budget rather than
    # one character short of it. Unknown window, known room: the room is the whole answer,
    # which is the native case where nothing here can see a context length.
    share = float(ctx * _PAGE_CONTEXT_SHARE) if ctx else float(room)
    # Whatever the caller will append is part of what has to fit, and it is taken off the
    # TOKEN budget rather than off the character cap: a punctuation-heavy path tokenises
    # far more densely than the prose characters that would otherwise be dropped to make
    # room for it, so subtracting its length in characters buys less room than it costs.
    share = max(0.0, share - reserve_tokens)
    if room is not None:
        room = max(0, int(room - reserve_tokens))
        # The share is a fraction of the WINDOW and does not fall as the thread fills, so
        # on its own it lets the last result before an overflow claim as much as the
        # first. Whichever of the two is smaller is the one that has to hold.
        share = min(share, float(room))
    # Never below the floor that keeps a result worth reading...
    floor = _MIN_PAGE_CHARS
    if room is not None:
        # ...but the floor is a comfort, not a right. 2,000 characters of dense output on
        # a thread with room for 100 tokens is the overflow this budget exists to prevent,
        # and the fit protects the newest turn so nothing downstream recovers. In the
        # extreme it leaves the stub alone, which is small enough to stay servable.
        #
        # MEASURED, not estimated: the flat four-characters-per-token rule hands back
        # about a third more than the room holds. Bottomed at one character rather than
        # the legacy floor, since going below it is the point.
        room_chars = _dense_prefix_chars(text, float(room))
        if not ctx:
            # No window to measure against, so the estimate above is the answer, already
            # halved for being unmeasurable.
            return min(max_chars, room_chars)
        # Bottomed at ZERO, not at one character. A thread at its budget measures a room of
        # zero, and a real tokenizer charges for the framing around even an empty string,
        # so the exact fit lands on nothing fitting. One character here is not a rounding
        # detail: it puts `_truncate` past its `limit <= 0` stub and back on the ordinary
        # notice, which is the ~90 tokens the stub exists to avoid spending when the
        # measurement just said there are none.
        floor = min(floor, _exact_prefix_chars(text, room_chars, float(room), ctx, 0))
    fitted = _dense_prefix_chars(text, share)
    # And measured rather than estimated when the serving model can measure it: the rule
    # above is honest about non-ASCII and still optimistic about dense ASCII. The floor
    # goes WITH it: the exact fit bottoms out on that value, so leaving it at the legacy
    # 2,000 would hand back 2,000 characters on a thread with room for a few hundred --
    # the same overflow, reached through the leg that is supposed to be the accurate one.
    fitted = _exact_prefix_chars(text, min(fitted, max_chars), share, ctx, floor)
    # An explicit cap smaller than the floor still wins.
    return min(max_chars, max(floor, fitted))


def _truncate_page_text(text: str, max_chars: int) -> str:
    if not text:
        return "(page returned no readable text)"
    max_chars = _dense_char_limit(text, max_chars)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


def _fetch_page_text(
    url: str,
    # Resolved per call rather than bound at import: the default would freeze the constant
    # before any model is loaded, which is exactly when the window is still unknown.
    max_chars: int | None = None,
    timeout: int = 30,
    cancel_event = None,
    website_policy: dict | None = None,
) -> str:
    """Fetch a URL and return readable text content.

    HTML responses are converted to Markdown with a main-content heuristic
    (``<article>``/``<main>`` scoping, hidden-element and boilerplate
    stripping); non-HTML text responses are returned as-is. GitHub repo root
    pages are rewritten to the README API so the model reads the README
    instead of the repo page's UI chrome. Blocks private/loopback/link-local
    targets (SSRF protection) and caps the download size.
    """
    if max_chars is None:
        max_chars = _page_char_budget()
    # One wall-clock budget for the whole fetch. The README API attempt and its
    # HTML fallback both draw from it, so a slow/failed API call cannot hand the
    # fallback a fresh full timeout and double the worst case.
    deadline = None if timeout is None else time.monotonic() + timeout
    from .web_access_policy import check_url_access

    # Before the policy gate (needs a scheme) and the README routing (reads host/path).
    url = _normalize_url_scheme(url)
    allowed, reason, _hostname = check_url_access(url, website_policy)
    if not allowed:
        return reason
    policy_kwargs = {"website_policy": website_policy} if website_policy is not None else {}
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
            **policy_kwargs,
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
        **policy_kwargs,
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


def _search_failure_message(exc: BaseException, timeout: int) -> str:
    """Turn a ddgs exception into text the model and the UI can act on.

    ddgs raises for an empty sweep as well as for refusals, so an unclassified
    ``Search failed: {exc}`` reports "nothing matched" and "every engine throttled us" the same
    way. Matched by class name because ddgs is imported lazily and tests stub the module.

    The RatelimitException arm is forward-looking: ddgs 9.14.4 defines the class but raises it
    nowhere, and no engine inspects the status code, so a throttled sweep parses to zero items
    and arrives here as the empty-sweep DDGSException instead.
    """
    name = type(exc).__name__
    if name == "RatelimitException":
        return (
            "Search failed: the search engines are rate limiting this machine. Wait a minute "
            'before searching again, or read a known page directly with {"url": "<URL>"}.'
        )
    if name == "TimeoutException":
        budget = f" within {timeout}s" if timeout else ""
        return f"Search failed: the search engines did not respond{budget}."
    # Only the base exception, so a subclass that happens to quote the phrase stays an error.
    if name == "DDGSException" and _DDGS_EMPTY_SWEEP in str(exc):
        return EMPTY_SEARCH_RESULTS[0]
    return f"Search failed: {exc}"


def _image_search_or_none(subjects: list, timeout, cancel_event, website_policy) -> "str | None":
    """``_image_search`` that reports a failure as None instead of raising.

    Every caller sits inside ``_web_search``'s own ``except``, which would turn a
    raise into "Search failed: ..." and throw away the text results the search had
    already found. A picture is a garnish: it must not become the answer.
    """
    try:
        return _image_search(
            subjects,
            timeout = timeout,
            cancel_event = cancel_event,
            website_policy = website_policy,
        )
    except Exception as exc:  # noqa: BLE001 - a garnish must not become the answer
        logger.debug("image lookup failed (%s)", type(exc).__name__)
        return None


def _empty_result_with_requested_images(
    empty_text: str, subjects: list, include_images: bool, timeout, cancel_event, website_policy
) -> str:
    """``empty_text`` plus the pictures the model asked for by name, if any.

    An ``image_queries`` call is an explicit request, and it succeeds on its own when
    sent without a query -- so returning the bare "No results found." because the TEXT
    sweep came back empty dropped images that were there to be had. Only the named
    subjects are looked up here; the per-query image pile has no answer to garnish.
    """
    if not subjects:
        return empty_text
    if not include_images:
        # Replayed history keeps teaching the parameter; say so, don't drop it.
        return empty_text + "\n\n---\n\n" + IMAGE_SEARCH_DISABLED
    if cancel_event is not None and cancel_event.is_set():
        return empty_text
    found = _image_search_or_none(subjects, timeout, cancel_event, website_policy)
    if found is None:
        return empty_text
    return empty_text + "\n\n---\n\n" + found


def _web_search(
    query: str,
    max_results: int = 5,
    timeout: int = _EXEC_TIMEOUT,
    url: str | None = None,
    cancel_event = None,
    website_policy: dict | None = None,
    include_images: bool = False,
    image_queries = None,
) -> str:
    """Search the web and return formatted results.

    ddgs fans the query out across its search engines, so a single engine refusing is already
    covered. If ``url`` is provided, fetches that page directly instead of searching.
    ``include_images`` adds image results registered server-side and offered to the model
    as ``[[img:<id>]]`` tokens, with a frontend-only envelope appended: one picture per
    ``image_queries`` subject when the model named them, else a handful for the query.
    ``image_queries`` alone (no query) is a pure image lookup.
    """
    # Direct URL fetch mode.
    if url and url.strip():
        fetch_timeout = 60 if timeout is None else min(timeout, 60)
        return _fetch_page_text(
            url.strip(),
            timeout = fetch_timeout,
            cancel_event = cancel_event,
            website_policy = website_policy,
        )

    subjects = _clean_image_queries(image_queries)
    if subjects and not (query and query.strip()):
        if not include_images:
            return IMAGE_SEARCH_DISABLED
        # Ahead of the try below, so this one has to carry its own guard: execute_tool
        # returns a string for every input, and a raise here would escape _web_search.
        found = _image_search_or_none(subjects, timeout, cancel_event, website_policy)
        if found is None:
            return "No images found for: " + ", ".join(subjects)
        return found

    if not query or not query.strip():
        return "No query provided."
    # A disconnect sets cancel_event; DDGS.text() is blocking and cannot be
    # interrupted mid-flight, so gate on either side: skip an already-cancelled
    # request, and discard results that land after the client has gone.
    if cancel_event is not None and cancel_event.is_set():
        return "Search cancelled."
    try:
        from ddgs import DDGS

        from .web_access_policy import check_url_access, scope_search_query

        effective_query = scope_search_query(query, website_policy)
        # The policy filters below, so ask for a deeper pool when one actually restricts: a page
        # whose top hits are all disallowed otherwise yields nothing even when valid results rank
        # just under them. Test the domain lists, not the dict: a run always stores a normalized
        # policy, which is truthy even when unrestricted.
        restricted = any(
            (website_policy or {}).get(key) for key in ("allowedDomains", "blockedDomains")
        )
        wanted = max_results * _POLICY_OVERFETCH if restricted else max_results
        client = DDGS(timeout = timeout)
        results = client.text(effective_query, max_results = wanted)
        if cancel_event is not None and cancel_event.is_set():
            return "Search cancelled."
        if not results:
            return _empty_result_with_requested_images(
                EMPTY_SEARCH_RESULTS[0],
                subjects,
                include_images,
                timeout,
                cancel_event,
                website_policy,
            )
        parts = []
        for r in results:
            if len(parts) >= max_results:
                break
            href = str(r.get("href") or "").strip()
            allowed, _reason, _hostname = check_url_access(href, website_policy)
            if not allowed:
                continue
            title = " ".join(str(r.get("title") or "").split())
            snippet = " ".join(str(r.get("body") or "").split())
            parts.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")
        if not parts:
            return _empty_result_with_requested_images(
                EMPTY_SEARCH_RESULTS[1],
                subjects,
                include_images,
                timeout,
                cancel_event,
                website_policy,
            )
        text = "\n\n---\n\n".join(parts)
        text += (
            "\n\n---\n\nIMPORTANT: These are only short snippets. "
            "To get the full page content, call web_search with "
            'the url parameter (e.g. {"url": "<URL>"}).'
        )
        if include_images and subjects:
            # The model named what it will show: one picture per subject, no generic pile.
            found = _image_search_or_none(subjects, timeout, cancel_event, website_policy)
            if found is not None:
                text += "\n\n---\n\n" + found
        elif include_images:
            text += _web_search_images_suffix(
                client,
                effective_query,
                wanted,
                cancel_event,
                website_policy,
            )
        elif subjects:
            # Replayed history keeps teaching the parameter; say so, don't drop it.
            text += "\n\n---\n\n" + IMAGE_SEARCH_DISABLED
        return text
    except Exception as e:
        failure = _search_failure_message(e, timeout)
        # ddgs signals an empty sweep by RAISING, so that exit is an empty result too
        # and owes the named subjects their pictures. A genuine failure keeps its
        # message alone: pictures under an error read as a partial answer.
        if failure == EMPTY_SEARCH_RESULTS[0]:
            return _empty_result_with_requested_images(
                failure,
                subjects,
                include_images,
                timeout,
                cancel_event,
                website_policy,
            )
        return failure


IMAGE_SEARCH_MAX_QUERIES = 5
IMAGE_SEARCH_PER_QUERY = 2
IMAGE_SEARCH_DISABLED = (
    "Image search is turned off. It can be enabled under Settings > Chat > Web search."
)


def _clean_image_queries(queries) -> list[str]:
    # Strings only, trimmed, deduped case-insensitively, capped; anything else is [].
    if isinstance(queries, str):
        queries = [queries]
    if not isinstance(queries, list):
        return []
    cleaned: list[str] = []
    for raw in queries:
        if not isinstance(raw, (str, int, float)):
            continue
        subject = " ".join(str(raw).split())[:80]
        if subject and subject.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(subject)
        if len(cleaned) >= IMAGE_SEARCH_MAX_QUERIES:
            break
    return cleaned


def _image_search(
    queries,
    timeout: int = _EXEC_TIMEOUT,
    cancel_event = None,
    website_policy: dict | None = None,
) -> str:
    # One lookup per subject, concurrently; same registry and tokens as web_search.
    from concurrent.futures import ThreadPoolExecutor

    from .search_images import cache_generation, images_envelope, register_images

    cleaned = _clean_image_queries(queries)
    if not cleaned:
        return "No subjects provided."
    if cancel_event is not None and cancel_event.is_set():
        return "Search cancelled."
    expected_generation = cache_generation()
    try:
        from ddgs import DDGS

        from .web_access_policy import scope_search_query
    except Exception as e:
        return _search_failure_message(e, timeout)
    if not callable(getattr(DDGS, "images", None)):
        return "Image search is unavailable in this install."

    def lookup(subject: str) -> list:
        # A client per call: ddgs instances are not documented thread-safe.
        try:
            return list(
                DDGS(timeout = timeout).images(
                    scope_search_query(subject, website_policy),
                    max_results = IMAGE_SEARCH_PER_QUERY * 4,
                    safesearch = "moderate",
                )
                or []
            )
        except Exception as exc:  # noqa: BLE001 - one subject failing must not lose the rest
            logger.debug("image lookup skipped %r (%s)", subject, type(exc).__name__)
            return []

    with ThreadPoolExecutor(max_workers = len(cleaned)) as pool:
        raw_by_subject = list(pool.map(lookup, cleaned))
    if cancel_event is not None and cancel_event.is_set():
        return "Search cancelled."

    sections: list[str] = []
    entries_all: list[dict[str, str]] = []
    for subject, raw in zip(cleaned, raw_by_subject):
        entries = register_images(
            raw,
            website_policy,
            max_images = IMAGE_SEARCH_PER_QUERY,
            subject = subject,
            expected_generation = expected_generation,
        )
        if not entries:
            sections.append(f"{subject}: no image found")
            continue
        entries_all.extend(entries)
        # One token per subject; spares ride along in the envelope as fallbacks.
        first = entries[0]
        domain = f" — {first['domain']}" if first["domain"] else ""
        sections.append(
            f"{subject}:\n- [[img:{first['id']}]] {first['title'] or '(untitled)'}{domain}"
        )
    if not entries_all:
        return "No images found for: " + ", ".join(cleaned)
    header = (
        "Images by subject. To show one, write its token exactly as given, e.g. "
        f"[[img:{entries_all[0]['id']}]], on its own line directly under the text about that "
        "subject. Use only these tokens; one per subject is enough."
    )
    return header + "\n\n" + "\n\n".join(sections) + images_envelope(entries_all)


def _web_search_images_suffix(client, query, wanted, cancel_event, website_policy) -> str:
    # "" when images are unavailable; never raises, the text results stand on their own.
    from .search_images import (
        MAX_IMAGES_PER_SEARCH,
        cache_generation,
        format_images_for_model,
        images_envelope,
        register_images,
    )

    images_fn = getattr(client, "images", None)
    if not callable(images_fn):
        return ""
    # Before the sweep, like _image_search: clear-all is what bumps this, and an
    # entry registered after one would keep serving a picture the user cleared.
    expected_generation = cache_generation()
    try:
        raw = images_fn(
            query, max_results = max(wanted, MAX_IMAGES_PER_SEARCH * 2), safesearch = "moderate"
        )
    except Exception as exc:  # noqa: BLE001 - optional extra; the text results stand on their own
        logger.debug("web_search image lookup skipped (%s)", type(exc).__name__)
        return ""
    if cancel_event is not None and cancel_event.is_set():
        return ""
    entries = register_images(
        list(raw or []), website_policy, expected_generation = expected_generation
    )
    if not entries:
        return ""
    return "\n\n---\n\n" + format_images_for_model(entries) + images_envelope(entries)


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
            "preupload_lfs_files",
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

    _HF_UPLOAD_PATH_VIOLATION = (
        "HF upload path must be a sandbox-local relative-path literal "
        "(no absolute paths, no '..' segments, no dynamic expressions)"
    )

    # Upload methods that take CommitOperation* objects rather than a path, and
    # the kwarg each one carries them in. `preupload_lfs_files` sends the file
    # bytes to the LFS store on its own, so it needs the same gate as a commit.
    _HF_OPERATIONS_KWARG = {
        "create_commit": "operations",
        "preupload_lfs_files": "additions",
    }

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
        if method_name in _HF_OPERATIONS_KWARG:
            ops_kwarg = _HF_OPERATIONS_KWARG[method_name]
            # A `*args` / `**kwargs` splat can smuggle in the operations or a token,
            # and either may sit after `operations=`, so scan before resolving.
            if any(isinstance(a, ast.Starred) for a in node.args or []):
                return _HF_UPLOAD_PATH_VIOLATION
            if any(kw.arg is None for kw in node.keywords or []):
                return _HF_UPLOAD_PATH_VIOLATION
            # Both methods take the operation list as their 2nd positional param.
            operations_node: ast.AST | None = node.args[1] if len(node.args or []) > 1 else None
            for kw in node.keywords or []:
                if kw.arg == ops_kwarg:
                    operations_node = kw.value
                    break
            if operations_node is None:
                return None  # no operations -> nothing is read off disk
            if not isinstance(operations_node, (ast.List, ast.Tuple)):
                return _HF_UPLOAD_PATH_VIOLATION
            for elt in operations_node.elts:
                if not isinstance(elt, ast.Call):
                    return _HF_UPLOAD_PATH_VIOLATION
                inner = _hf_upload_violation(elt, "commit_operation")
                if inner:
                    return inner
            return None
        if method_name == "commit_operation":
            # A CommitOperation* constructor from the list above. Delete and copy get
            # no exemption: a by-name one would trust a name sandboxed code can rebind.
            # Add is (path_in_repo, path_or_fileobj) so both positionals are checked,
            # and other keywords must be literals -- a computed one reads the file.
            path_nodes = list(node.args or [])
            for kw in node.keywords or []:
                if kw.arg is None:
                    return _HF_UPLOAD_PATH_VIOLATION
                if kw.arg == "path_or_fileobj":
                    path_nodes.append(kw.value)
                elif not isinstance(kw.value, ast.Constant):
                    return _HF_UPLOAD_PATH_VIOLATION
            if not path_nodes:
                return _HF_UPLOAD_PATH_VIOLATION
            for p in path_nodes:
                if not _path_arg_is_sandbox_local(p):
                    return _HF_UPLOAD_PATH_VIOLATION
            return None
        path_node: ast.AST | None = node.args[0] if node.args else None
        for kw in node.keywords or []:
            if kw.arg in ("path_or_fileobj", "folder_path"):
                path_node = kw.value
                break
        if not _path_arg_is_sandbox_local(path_node):
            return _HF_UPLOAD_PATH_VIOLATION
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


def _adopt_tool_pid(pid: "int | None") -> None:
    """Record a tool subprocess for the startup sweep.

    macOS has no parent-death signal, so a force quit mid-call would otherwise
    leave a session-leading tool (and whatever it spawned) with nothing able to
    find it. Best-effort: a failure here must never break a tool call.
    """
    if not pid:
        return
    try:
        from utils.process_lifetime import adopt_pid
        adopt_pid(pid)
    except Exception:
        pass


def _forget_tool_pid(proc) -> None:
    """Drop the record once the process has actually exited."""
    pid = getattr(proc, "pid", None)
    if not pid:
        return
    try:
        if getattr(proc, "poll", lambda: None)() is None:
            return
        from utils.process_lifetime import forget_pid
        forget_pid(pid)
    except Exception:
        pass


def _capture_process_group(proc):
    """Return the setsid process-group id, or ``None`` when unavailable.

    Captured right after ``Popen`` so a later ``poll()`` / ``wait()`` that reaps
    the leader cannot make ``os.getpgid(proc.pid)`` fail first.

    Windows has no process groups, so capture the wrapper pid instead, tagged
    for ``_killpg_captured`` to reach with ``taskkill /T``; returning ``None``
    there left a payload that outlived its wrapper unsignalled.
    """
    if os.name == "nt":
        job = _windows_job_capture(proc)
        if job is not None:
            return ("windows-job", job)
        # No job available, so fall back to the pid, carrying its creation-time
        # identity: a posix group id cannot be recycled while a member lives,
        # but this bare pid can, and the timeout path may fire much later.
        return ("windows-tree", proc.pid, _windows_pid_identity(proc.pid))
    if os.name != "posix" or not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(proc.pid)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        return None


class _WindowsToolJob:
    """A job object holding one tool call's process tree.

    Windows has no process groups, and ``taskkill`` cannot reach a tree whose
    root has already exited, which is exactly the case this capture exists for.
    The job stays a valid handle on every descendant either way. Created without
    kill-on-close, so releasing it never kills a process that outlived the call.
    """

    def __init__(self, handle, kernel32):
        self._handle = handle
        self._kernel32 = kernel32

    def terminate(self) -> bool:
        if not self._handle:
            return False
        return bool(self._kernel32.TerminateJobObject(self._handle, 1))

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            try:
                self._kernel32.CloseHandle(handle)
            except Exception:  # noqa: BLE001 - interpreter teardown
                pass

    def __del__(self) -> None:
        self.close()


def _windows_job_capture(proc) -> "_WindowsToolJob | None":
    """Put ``proc`` in its own job. ``None`` when that is not possible, leaving
    the pid-based fallback."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        H, BOOL, UINT = wintypes.HANDLE, wintypes.BOOL, wintypes.UINT
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        # Explicit widths: without them ctypes truncates a 64-bit handle to
        # c_int and every call silently works on a bogus one.
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.CreateJobObjectW.restype = H
        kernel32.AssignProcessToJobObject.argtypes = [H, H]
        kernel32.AssignProcessToJobObject.restype = BOOL
        kernel32.TerminateJobObject.argtypes = [H, UINT]
        kernel32.TerminateJobObject.restype = BOOL
        kernel32.CloseHandle.argtypes = [H]
        kernel32.CloseHandle.restype = BOOL

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        # The Popen handle, not a fresh OpenProcess: it already refers to this
        # child, so there is no window for the pid to be recycled first.
        if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
            kernel32.CloseHandle(job)
            return None
        return _WindowsToolJob(job, kernel32)
    except Exception:  # noqa: BLE001 - falls back to the pid-based kill
        return None


def _windows_pid_identity(pid: int) -> "str | None":
    """Process creation time, so a recycled pid is never mistaken for this one."""
    if os.name != "nt":
        return None
    try:
        from utils.process_lifetime import _pid_identity
        return _pid_identity(pid)
    except Exception:
        return None


def _windows_taskkill_tree(pid: int, identity: "str | None" = None) -> bool:
    """``taskkill /T /F`` a pid and its descendants. True when it succeeded.

    Every tool call runs under a shell wrapper, and Windows has no process
    groups, so a bare ``proc.kill()`` reaps the wrapper and orphans the payload
    (usually the venv python), which then blocks `unsloth studio update`.

    ``identity`` is the creation time captured at spawn; a mismatch means the pid
    now belongs to something else, so nothing is signalled.
    """
    if os.name != "nt":
        return False
    if identity is not None and _windows_pid_identity(pid) != identity:
        return False
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output = True,
            timeout = 15,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode in (0, 128)  # 128: already gone


def _kill_process_tree(proc) -> None:
    """SIGKILL the setsid process group; fall back to single-pid kill."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        if _windows_taskkill_tree(proc.pid):
            return
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
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
    whole tree. On Windows the capture is a tagged pid and the equivalent reach
    is ``taskkill /T /F``. No-op when nothing was captured.
    """
    if pgid is None:
        return
    if isinstance(pgid, tuple):
        if pgid[0] == "windows-job":
            pgid[1].terminate()
            return
        _tag, pid, identity = pgid
        # Fail closed: this runs long after the capture, so without a verified
        # identity the pid may be someone else's now. The job object still takes
        # the whole tree when Unsloth exits, which is the safe half to keep.
        if identity is not None:
            _windows_taskkill_tree(pid, identity)
        return
    if not hasattr(os, "killpg"):
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


def _appended_by_the_loop(text: str) -> float:
    """What the tool loop will add to this result after the tool has handed it back.

    `ToolCallCompletion.model_message` appends `TOOL_ERROR_NUDGE` to a result that opens
    with one of `TOOL_ERROR_PREFIXES`, after this budget has already let the body take the
    whole room, and a parallel batch of failed calls carries one nudge each. Priced in
    tokens like the retry hint, and only for the results that will really carry it.
    """
    try:
        from .tool_call_parser import TOOL_ERROR_NUDGE, TOOL_ERROR_PREFIXES  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- an unpriced nudge, not a failed tool call
        logger.debug("result budget: tool error nudge unavailable", exc_info = True)
        return 0.0
    if not text.startswith(TOOL_ERROR_PREFIXES):
        return 0.0
    return _text_token_cost(TOOL_ERROR_NUDGE, _window_context_tokens())


def _truncate(
    text: str,
    limit: int | None = None,
    workdir: str | None = None,
    scope: "str | None" = "",
    hint: str = "",
) -> str:
    # Resolved per call, not bound at import: the default would freeze the constant
    # before any model is loaded, which is exactly when the window is still unknown.
    if limit is None:
        limit = _tool_result_char_budget()
    # Same correction as a fetched page: a character cap reserves its share of the window
    # only for English, and a command that prints CJK or percent-escaped text costs two to
    # three times what the cap assumed.
    # Whatever the loop will append to this result once it has it: the tool-error nudge
    # goes on after the tool has returned, so a result sized to fill the room arrives at
    # the prompt with the nudge past the end of it. Charged only to the results that will
    # actually carry one, since a reserve taken from every result spends room the thread
    # has.
    cap, cost = limit, _appended_by_the_loop(text)
    if hint:
        # Priced in tokens, not characters, and taken off the budget before it is converted
        # (see `_dense_char_limit`). A failing absolute path is dense: subtracting its
        # LENGTH from the character cap frees fewer tokens than the hint then spends, which
        # is the unbudgeted overflow this whole change exists to prevent. It may take at
        # most half the room; past that the output is worth more than the advice about it.
        plain = _dense_char_limit(text, limit, cost)
        cost += _text_token_cost(hint, _window_context_tokens())
        with_hint = _dense_char_limit(text, limit, cost)
        if plain > 0 and (len(text) <= with_hint or with_hint * 2 >= plain):
            limit = with_hint
        else:
            # Nothing to spend on advice: at zero room the stub IS the message, and when
            # paying for it would cut the output in half the output is worth more than the
            # advice about it. Nothing is dropped while the result fits anyway.
            limit, hint, cost = plain, "", _appended_by_the_loop(text)
    else:
        limit = _dense_char_limit(text, limit, cost)
    # Mode-neutral notice: this result serves both the streaming UI and
    # non-streaming callers and must stay byte-identical with and without an
    # output_callback (a regression-tested invariant), so it can't claim the
    # user saw the full output.
    if len(text) <= limit:
        return text + hint
    # Only now is a notice certain, and only now is it charged. Held back before the
    # measurement above, it would cut results that fit: a 100-token result with 200 tokens
    # of room would be sized against 72 and come back as 72 tokens of body plus ~70 of
    # notice, which is more of the window spent to say less. See `_RESULT_NOTICE_RESERVE`.
    #
    # Against a priced room only. With none, the caller's character cap is the whole
    # budget and the notice has always been appended past it; taking a token reserve off a
    # character cap there would cut every legacy caller's output to nothing.
    if _request_result_room() is not None:
        limit = _dense_char_limit(text, cap, cost + _RESULT_NOTICE_RESERVE)
    if limit <= 0 and len(_zero_room_stub(len(text), None, True)) >= len(text):
        # Decided BEFORE the spill: a result this short is served whole below, and writing
        # a file (and creating the spill directory) for output that is never cut is a side
        # effect with nothing on the other side of it.
        return text + hint
    spill, complete = _spill_full_output(text, workdir, scope)
    if limit <= 0:
        # No room for a body, so no room for the usual notice either: at this point the
        # notice IS the message, and the full one costs ~90 tokens of a budget that just
        # reported none. Kept to a line so the thread stays servable and the next fit can
        # evict older turns and recover, which is the whole reason a stub beats a refusal.
        stub = _zero_room_stub(len(text), spill, complete)
        # A short result costs less than the notice explaining it is gone, and replacing
        # "done" with a longer sentence saves nothing and loses the answer.
        return (stub if len(stub) < len(text) else text) + hint
    head, on_boundary = _head_whole_lines(text, limit)
    if spill is None:
        return (
            head
            + (
                f"\n\n... (truncated to {limit} chars for the model; {len(text)} chars "
                "total. The full output is not retained here; any files the code wrote "
                "persist in the working directory.)"
            )
            + hint
        )
    # The rest is not advice, it is reachable: the sandbox persists between calls and the
    # model already has the terminal, so naming the exact next command turns a dead end
    # into paging. Truncating without one is what makes a model re-run the same command
    # and truncate identically.
    if on_boundary:
        # No "+1" when the head already ends in a newline: the count is of the lines
        # SHOWN, and a trailing break closes the last one rather than opening another.
        # Reachable at a limit of 1 on output that starts with a blank line, where the
        # head is "\n" alone and a count of two makes the hint resume at line 3, skipping
        # the first line the reader never saw.
        shown = 0 if not head else head.count("\n") + (0 if head.endswith("\n") else 1)
        total = text.count("\n") + 1
        resume = f"sed -n '{shown + 1},{shown + max(1, shown)}p' {spill}"
        where = f"showing lines 1-{shown} of {total}"
    else:
        # Cut inside a line, so a line number would name the NEXT one and skip the rest of
        # the line still unread -- on single-line output, everything. Bytes resume exactly
        # where this stopped. Measured in bytes, not characters, because that is what
        # `tail -c` counts and the two differ on any non-ASCII text.
        offset = len(head.encode("utf-8", "surrogatepass"))
        # The chunk is as many BYTES as the next `len(head)` characters actually occupy,
        # not as many bytes as this one did. `head -c` counts bytes, and a round number of
        # them lands inside a code point on any mixed-width text, so the model would read
        # back a mangled character at the end of every chunk (the runner decodes with
        # errors="replace"). The text is here, so the boundary is not a guess.
        chunk = text[len(head) : len(head) * 2 or None]
        span = len(chunk.encode("utf-8", "surrogatepass"))
        resume = f"tail -c +{offset + 1} {spill} | head -c {max(1, span)}"
        where = f"showing the first {len(head)} chars of {len(text)}"
    # The workdir sentence stays whatever else the notice says: it is about the files the
    # CODE wrote, not the spill, and it is the only thing telling the model those survive.
    common = (
        f"\n\n... (truncated to {limit} chars for the model; {where}, {len(text)} chars "
        f"total. {_capitalise(_spill_phrase(spill, complete))}, and any files the code "
        "wrote persist in the working directory"
    )
    if not _posix_tools_available():
        # A cmd-only Windows host has none of sed, tail or head, so the command would fail
        # and the model would most likely re-run the command that truncated. Name where
        # the output is and stop there, rather than promising paging that cannot happen.
        return head + common + ".)" + hint
    return head + common + f" -- continue with:\n  {resume})" + hint


def _fit_result_to_room(text, name = None):
    """Cap a tool that does not cap its own output, when this request priced its room.

    `python` and `terminal` truncate against this same budget before they return, so this
    is a no-op for them. The other tools hand their string back whole: an MCP response is
    unbounded, a fetched page or an edit receipt runs to a few thousand characters, and
    any of them can overflow a nearly full local thread and then be protected as its
    newest exchange, which is the failure the budget exists to prevent.

    No spill file: these tools have no sandbox of their own, and `edit_file` must not have
    a workdir created underneath a caller running with code execution off. So the cap
    comes with the plain notice and no paging hint, which is the scope limit this change
    states rather than hides.

    With no priced room -- external providers, the hosted path, any loop that does not
    measure its own conversation -- the text is returned untouched, so nothing outside a
    local tool loop changes.
    """
    if _request_result_room() is None or not isinstance(text, str) or not text:
        return text
    # Only the part the model will actually be shown is measured and cut. The rest is a
    # frontend-only envelope -- an MCP image array, web_search thumbnails, RAG sources --
    # which `strip_result_for_model` removes before the result is replayed, so it costs
    # the window nothing and must come back byte-identical: every consumer of the
    # __MCP_IMAGES__ envelope requires the whole valid JSON array, and a cut anywhere
    # inside a megabyte of base64 does not lose the image quietly, it replays the broken
    # fragment to the model instead.
    body, suffix = _split_frontend_suffix(text, name)
    if not body:
        return text
    fitted = _truncate(body)
    return fitted + suffix if fitted is not body else text


def _split_frontend_suffix(text: str, name: "str | None") -> "tuple[str, str]":
    """``text`` split into what the model sees and the trailing frontend-only envelope.

    `strip_result_for_model` is the same function the replay path uses, so the split
    follows its validation exactly -- a result that merely mentions a sentinel keeps it in
    the body and is capped with it, which is the conservative half.
    """
    from .tool_loop_controller import strip_result_for_model

    try:
        body = strip_result_for_model(text, name)
    except Exception:
        logger.debug("frontend suffix split failed", exc_info = True)
        return text, ""
    # It only ever strips a suffix, so the remainder is the exact bytes that were removed
    # (including whatever whitespace the strip rstripped away).
    if not isinstance(body, str) or not text.startswith(body):
        return text, ""
    return body, text[len(body) :]


def _head_whole_lines(text: str, limit: int) -> "tuple[str, bool]":
    """``text`` cut to at most ``limit`` characters, and whether it ended on a line break.

    Whole lines where possible, so the hint can name a line that resumes exactly where
    this stopped; a mid-line cut would repeat or lose one, and on a file printed verbatim
    that is a line of the user's own code.

    The flag is not decoration. One enormous line (minified JS, base64) has no boundary to
    cut on, and a line number would then name the line AFTER the one the reader is halfway
    through, skipping the rest of it. On single-line output that returns nothing at all.
    """
    head = text[:limit]
    cut = head.rfind("\n")
    # Only when a boundary is actually near the end: rewinding further would throw away
    # the whole result to keep the hint tidy.
    if cut > 0 and cut >= limit // 2:
        return head[:cut], True
    return head, head.endswith("\n")


def _posix_tools_available() -> bool:
    """Whether the shell these tools run in has sed/tail/head.

    `_get_shell_cmd` falls back to `cmd /c` on a Windows host with no trusted bash, and
    none of those exist there, so a hint naming them is a command the model cannot run.
    The spill is still worth naming; the command is not.
    """
    if sys.platform != "win32":
        return True
    return _windows_bash() is not None


# Dot-directory on purpose: `_snapshot_workdir_files` skips those, so the spill never
# appears as a file the model created and never earns a download card in the UI. A plain
# name here would put a phantom artifact beside every truncated result.
_SPILL_DIR = ".unsloth_tool_output"
_SPILL_KEEP = 20
# A result reaches here whole, so one `cat` of a multi-gigabyte file would be retained in
# full and twenty of them would fill the host's disk. The subprocess file-size limit does
# not apply: this output came through a pipe, not a file the sandbox wrote. A spill exists
# so the model can page through what it was shown, and 8 MB is already far more of that
# than any window can consume.
_SPILL_MAX_BYTES = 8 * 1024 * 1024

# How much of a result is encoded at a time when it is hashed. UTF-8 encodes one code
# point at a time, so a stream built from slices of the string is byte for byte the stream
# built from the whole of it, whatever the chunk size.
_SPILL_HASH_CHUNK_CHARS = 1 << 20


def _digest_and_head(text: str, max_bytes: int) -> "tuple[str, int, bytes]":
    """``(digest, encoded length, the first max_bytes of it)``, in one bounded pass.

    Encoding the whole result to hash it and again to cut it puts two more copies of it
    through memory, and at most `max_bytes` of the second is ever written. The output this
    runs on is by definition the output that did not fit: `cat` of a file the model just
    wrote can be hundreds of megabytes, and spending it twice more inside the backend, at
    the point the result is already in hand, risks the stall or the OOM instead of the
    bounded answer this whole path exists to return.
    """
    digest = hashlib.sha256()
    head = bytearray()
    total = 0
    for start in range(0, len(text), _SPILL_HASH_CHUNK_CHARS):
        chunk = text[start : start + _SPILL_HASH_CHUNK_CHARS].encode("utf-8", "surrogatepass")
        digest.update(chunk)
        total += len(chunk)
        if len(head) < max_bytes:
            head += chunk[: max_bytes - len(head)]
    return digest.hexdigest()[:12], total, bytes(head)


_SPILL_MAX_TOTAL_BYTES = 64 * 1024 * 1024
# Exactly the names `_spill_full_output` generates: twelve hex characters of a content
# digest. The prune below deletes what it matches, and the sandbox is the user's own
# directory -- a session may open on one that already holds a folder of this name, and
# anything in it that Unsloth did not write is not Unsloth's to remove.
_SPILL_NAME_RE = re.compile(r"[0-9a-f]{12}\.txt")
# Written once, when this process creates the spill directory. Ownership is RECORDED
# rather than inferred from the names inside: a sandbox can be a project the user opened,
# a directory of this name in it may be theirs, and a file name proves nothing about who
# wrote it. Without the marker nothing here writes, prunes or discounts anything there.
_SPILL_RECORD_HEADER = "unsloth-studio tool output "
# One lock per spill root. Appending a spill and rewriting the manifest after a prune are
# a read-modify-write over one shared file, and a project's chats share a sandbox: two
# calls spilling at once could otherwise have the pruner drop the entry the other just
# appended, leaving a file nothing counts, prunes, or recognises as Unsloth's.
_SPILL_LOCKS: "dict[str, threading.Lock]" = {}
_SPILL_LOCKS_GUARD = threading.Lock()


def _spill_lock(root: str) -> "threading.Lock":
    key = os.path.realpath(root)
    with _SPILL_LOCKS_GUARD:
        return _SPILL_LOCKS.setdefault(key, threading.Lock())


def _spill_records_dir() -> str:
    """Where the spill manifests live: Unsloth's own storage, NOT the sandbox.

    The sandbox is a directory tool code writes to, so nothing kept inside it can be
    evidence about the sandbox. A marker file there was replaceable by a link, and once it
    is a plain file the model can rewrite its contents and name the user's own files as
    Unsloth's, which turns the cleanup into a delete and the prune into an unlink. Held
    beside the other records this file already keeps outside the sandboxes.
    """
    try:
        from utils.paths.storage_roots import studio_root  # noqa: PLC0415
        return os.path.join(str(studio_root()), "tool-output-records")
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.realpath(sandbox_root())), "tool-output-records"
        )


def _spill_record_path(root: str) -> str:
    """The record for one spill root, named by a digest of its real path."""
    digest = hashlib.sha256(os.path.realpath(root).encode("utf-8", "surrogatepass")).hexdigest()
    return os.path.join(_spill_records_dir(), f"{digest[:24]}.txt")


def _spill_identity(root: str) -> "str | None":
    """The directory's device and inode, which is what the record claims ownership OF.

    Tool code can delete `.unsloth_tool_output` and make its own in the same place. That
    is a different directory with the same path, and a record that only knew the path
    would hand the new one's contents to the prune.
    """
    try:
        stat = os.lstat(root)
    except OSError:
        return None
    if not os.path.isdir(root) or os.path.islink(root):
        return None
    return f"{stat.st_dev}:{stat.st_ino}"


def _own_spill_root(root: str) -> bool:
    """Whether the spill directory is one this process made, creating it if it is absent.

    Ownership is recorded outside the sandbox (`_spill_records_dir`) and is of a specific
    directory, not of a path: a `.unsloth_tool_output` that came with the sandbox, or one
    tool code deleted and recreated, has no matching record and is left alone entirely.
    """
    if os.path.islink(root):
        return False
    try:
        # The existence check, the creation and the first record are ONE locked step. Two
        # first-time spills in a shared project sandbox can both see the directory absent,
        # and the slower one would otherwise write an empty record over the winner's, which
        # leaves the winner's spill owned by nobody: never pruned, counted as the user's
        # content on cleanup, and outside the byte budget for good.
        with _spill_lock(root):
            existed = os.path.isdir(root)
            if not existed:
                if os.path.exists(root):
                    return False
                os.makedirs(root, exist_ok = True)
            identity = _spill_identity(root)
            if identity is None:
                return False
            recorded = _spill_record(root)[0]
            if recorded is None:
                if existed and os.listdir(root):
                    # Not ours and not empty, so it came with the sandbox.
                    return False
                os.makedirs(_spill_records_dir(), exist_ok = True)
                _write_spill_manifest(root, {}, identity = identity)
                recorded = identity
            return recorded == identity
    except OSError:
        logger.debug("tool result spill ownership check failed", exc_info = True)
        return False


# Only where the platform can do the whole write through a directory descriptor: Windows
# has neither O_DIRECTORY nor dir_fd, and there the path-based write below stands, with the
# link checks it already makes.
_DIR_FD_WRITES = (
    hasattr(os, "O_DIRECTORY")
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.link in getattr(os, "supports_dir_fd", set())
    and os.unlink in getattr(os, "supports_dir_fd", set())
)


def _write_spill_file(target_dir: str, name: str, body: str) -> "str | None":
    """Write one spill into ``target_dir``, without following a link at any point.

    The directory is opened ONCE, O_NOFOLLOW, and every step after that is relative to
    that descriptor. Checking the path and then writing to it by name is a race a shared
    project sandbox can lose: another call can replace the directory with a symlink in
    between, and both the create and the rename would follow it, putting this output
    outside the sandbox with the backend's own permissions.

    Still written to a fresh O_EXCL file and installed under the real name rather than
    opened over whatever is there: the spill name comes from content the model produced, so
    it can predict it and pre-create it, as a symlink or as a hard link sharing an inode
    with some file elsewhere.

    Installed with `os.link`, which fails when the name is taken, rather than a rename,
    which on POSIX replaces silently. The caller checks the destination first, but between
    that check and this write another call sharing the workspace can put a file there, and
    a rename would then destroy it.

    Returns the stamp of what was installed, or None if nothing was. The stamp is taken
    here rather than re-read from the path afterwards, because by then another call can
    have replaced the file and the record would name its content as Unsloth's.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if not _DIR_FD_WRITES:
        tmp = os.path.join(target_dir, f".tmp-{uuid.uuid4().hex[:12]}.txt")
        try:
            with os.fdopen(
                os.open(tmp, flags | getattr(os, "O_NOFOLLOW", 0), 0o600),
                "w",
                encoding = "utf-8",
                newline = "",
            ) as handle:
                handle.write(body)
            installed = os.path.join(target_dir, name)
            os.link(tmp, installed)
            _quiet_unlink(tmp)
            return _spill_stamp(installed)
        except OSError:
            logger.debug("tool result spill write failed", exc_info = True)
            _quiet_unlink(tmp)
            return None
    dir_fd = None
    tmp_name = f".tmp-{uuid.uuid4().hex[:12]}.txt"
    try:
        dir_fd = os.open(target_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        with os.fdopen(
            os.open(tmp_name, flags, 0o600, dir_fd = dir_fd), "w", encoding = "utf-8", newline = ""
        ) as handle:
            handle.write(body)
        # `os.link` rather than a rename: rename replaces the destination silently on
        # POSIX, and the name may have been taken since the caller looked. link fails with
        # EEXIST instead, which is the answer this wants.
        os.link(tmp_name, name, src_dir_fd = dir_fd, dst_dir_fd = dir_fd)
        _quiet_unlink(tmp_name, dir_fd = dir_fd)
        # Through the same descriptor, so it is the file just linked rather than whatever
        # the name resolves to by the time this returns.
        stat = os.stat(name, dir_fd = dir_fd, follow_symlinks = False)
        return ":".join(
            str(part)
            for part in (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        )
    except OSError:
        logger.debug("tool result spill write failed", exc_info = True)
        if dir_fd is not None:
            _quiet_unlink(tmp_name, dir_fd = dir_fd)
        return None
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


def _quiet_unlink(path: str, dir_fd = None) -> None:
    try:
        os.unlink(path, dir_fd = dir_fd) if dir_fd is not None else os.unlink(path)
    except OSError:
        pass


def _forget_spill_record(path: str) -> None:
    """Drop the record for a sandbox that has been removed. See `_spill_records_dir`."""
    try:
        os.remove(path)
    except OSError:
        pass


def _spill_stamp(path: str) -> "str | None":
    """What a spill looked like when it was written: device, inode, size, mtime, ctime.

    A recorded PATH is not the file: tool code can write its own content over one, in
    place, keeping the inode. mtime alone is not enough either, since `os.utime` can put
    it back and a coarse-grained filesystem can leave it unchanged on its own; ctime moves
    on any write to the file OR its metadata and cannot be set back from userspace, so
    restoring the mtime is itself a change this sees.
    """
    try:
        stat = os.lstat(path)
    except OSError:
        return None
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    return ":".join(
        str(part)
        for part in (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    )


def _stamp_size(stamp: str) -> "int | None":
    """The size a stamp remembers. See `_spill_stamp`."""
    try:
        return int(stamp.split(":")[2])
    except (IndexError, ValueError):
        return None


def _file_digest(path: str, expected_size: "int | None" = None) -> "str | None":
    """The digest of what is on disk now, or None when it is not a file this may read.

    Opened O_NOFOLLOW and O_NONBLOCK and checked through the DESCRIPTOR, not the path. The
    stamp was taken a moment ago and this runs in the sandbox's own directory: between the
    two, tool code can put a symlink at the name, or a FIFO, or a device. A plain open
    would follow the first and block forever on the others, and this is called
    synchronously by the prune and by chat deletion, neither of which has a timeout.

    ``expected_size`` refuses anything that is not the size the record remembers, so a
    file swapped for an enormous one is not hashed before it is rejected.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = None
    try:
        fd = os.open(path, flags)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        if expected_size is not None and info.st_size != expected_size:
            return None
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb") as handle:
            fd = None
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _is_recorded_spill(root: str, path: str, owned: "dict[str, tuple[str, str]]") -> bool:
    """Whether ``path`` is still the file this wrote, rather than one written over it.

    Both the stamp and the CONTENT, because the stamp is only evidence about metadata: the
    content is what says this file is still the output this process put there, and it is
    the last word before anything is deleted. Read only after the cheap check passes, so
    the cost falls on the handful of files that already look like ours.
    """
    relative = os.path.relpath(path, root).replace(os.sep, "/")
    recorded = owned.get(relative)
    if recorded is None or recorded[0] != _spill_stamp(path):
        return False
    return recorded[1] == _file_digest(path, _stamp_size(recorded[0]))


def _spill_record(root: str) -> "tuple[str | None, dict[str, tuple[str, str]]]":
    """The recorded identity of ``root`` and the spills written into it.

    ``(None, {})`` when there is no record, which is the answer for any directory this
    process did not create. An unreadable or half-written record reads the same way, which
    retains too much rather than deleting something that was never ours.
    """
    try:
        with open(_spill_record_path(root), encoding = "utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None, {}
    if not lines or not lines[0].startswith(_SPILL_RECORD_HEADER):
        return None, {}
    identity = lines[0][len(_SPILL_RECORD_HEADER) :].strip() or None
    entries: "dict[str, tuple[str, str]]" = {}
    for line in lines[1:]:
        name, _, rest = line.strip().partition("\t")
        stamp, _, digest = rest.partition("\t")
        if name and stamp and digest:
            # Last line wins: the same content spilled twice rewrites the file, and the
            # stamp of the write actually on disk is the later one.
            entries[name] = (stamp, digest)
    return identity, entries


def _spill_manifest(root: str) -> "dict[str, tuple[str, str]]":
    """The spills this process wrote into ``root``: relative path to stamp and digest."""
    return _spill_record(root)[1]


def _write_spill_manifest(
    root: str,
    entries,
    identity: "str | None" = None,
) -> None:
    """Rewrite the record with ``entries`` (relative name to stamp), atomically."""
    if identity is None:
        identity = _spill_record(root)[0]
    path = _spill_record_path(root)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir = os.path.dirname(path), prefix = ".tmp-record-")
        with os.fdopen(fd, "w", encoding = "utf-8") as handle:
            handle.write(f"{_SPILL_RECORD_HEADER}{identity or ''}\n")
            for name in sorted(entries):
                stamp, digest = entries[name]
                handle.write(f"{name}\t{stamp}\t{digest}\n")
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _record_spill(root: str, relative: str, stamp: str, digest: str) -> None:
    """Append one written spill, with the stamp and digest of what was INSTALLED.

    Not re-read from the path: between the install and this, another call sharing the
    sandbox can replace the file, and stating the path then records that call's content as
    Unsloth's, which a later prune or cleanup would delete. The writer knows what it put
    there, so it says so.
    """
    try:
        with _spill_lock(root):
            with open(_spill_record_path(root), "a", encoding = "utf-8") as handle:
                handle.write(f"{relative}\t{stamp}\t{digest}\n")
    except OSError:
        logger.debug("tool result spill record append failed", exc_info = True)


def _spill_scope(session_id: "str | None", thread_id: "str | None") -> "str | None":
    """Where this call's spills belong, or None for "retain nothing".

    Nothing is retained in a sandbox that more than one chat runs in. A project's chats
    share one session by design (`project_session_id`) and a call with no session lands in
    the shared `_default` one, and in both the sandbox is a directory every sibling chat's
    model has a terminal in: a sub-directory is not access control, it is a name. Writing
    a result there puts output that existed only in one chat's response on disk where the
    next chat can read it, and lets that chat prune the files this one was told to page
    through.

    So the trade is stated rather than hidden: in a project, a large result is truncated
    with a notice and no continuation. Only a chat with a sandbox of its own gets paging.
    ``thread_id`` is taken and unused for that reason -- it identifies the chat, which is
    not the thing that has to be separate.
    """
    if not session_id or session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    return ""


def _spill_phrase(spill: str, complete: bool) -> str:
    """How the notice names the spill, which depends on whether all of it got there.

    Lower case, and capitalised by the caller that needs it: the phrase ends in a path,
    and case-folding a whole sentence to fit it into another would fold that too.
    """
    if complete:
        return f"full output saved to {spill}"
    return f"the first {_SPILL_MAX_BYTES} bytes of it are saved to {spill}"


def _capitalise(phrase: str) -> str:
    """First letter only. `str.capitalize` lower-cases the rest, including a path."""
    return phrase[:1].upper() + phrase[1:]


def _zero_room_stub(size: int, spill: "str | None", complete: bool) -> str:
    """The whole message when there is no room for a body. See `_truncate`."""
    located = f", {_spill_phrase(spill, complete)}" if spill else ""
    return f"(output omitted: {size} chars, no context room left{located})"


def _spill_full_output(
    text: str,
    workdir: str | None,
    scope: "str | None" = "",
) -> "tuple[str | None, bool]":
    """Write the result into the sandbox; return its relative path and whether it is whole.

    ``(None, True)`` whenever it cannot be done -- no workdir, a read-only mount, a full
    disk, a path that is not what it claims to be. The caller then falls back to the plain
    notice, because a hint naming a file that is not there is worse than admitting the
    output is gone.
    """
    if not workdir or not os.path.isdir(workdir) or scope is None:
        return None, True
    try:
        if not _own_spill_root(os.path.join(workdir, _SPILL_DIR)):
            return None, True
        relative = f"{_SPILL_DIR}/{scope}" if scope else _SPILL_DIR
        target_dir = os.path.join(workdir, *relative.split("/"))
        # The sandbox is a directory the model runs commands in, so `.unsloth_tool_output`
        # may already be a symlink it made, or one that came with a project opened as the
        # workdir. makedirs(exist_ok=True) and a plain open() both follow it, which writes
        # this result outside the sandbox with the backend's own permissions and then lets
        # the prune delete files there. Refuse instead: no spill means a notice without a
        # continuation hint, which is a great deal better than a write out of bounds.
        if any(
            os.path.islink(os.path.join(workdir, *relative.split("/")[: n + 1]))
            for n in range(len(relative.split("/")))
        ):
            return None, True
        os.makedirs(target_dir, exist_ok = True)
        expected = os.path.join(os.path.realpath(workdir), *relative.split("/"))
        if os.path.realpath(target_dir) != expected:
            return None, True
        # Named from the CONTENT, not at random. The result has to come back byte-identical
        # with and without an output_callback (the streaming invariant asserted by
        # test_truncated_result_identical_and_notice_neutral_with_streaming), and a random
        # name puts a different path in the notice on each of the two runs. Content
        # addressing also means asking for the same file twice reuses one spill instead of
        # filling the sandbox with copies, which is the repeat case this whole change is
        # about.
        # Bounded before it is written rather than after: pruning by count alone still
        # lets one enormous result through, and by then it is already on the disk.
        digest, spilled_bytes, head = _digest_and_head(text, _SPILL_MAX_BYTES)
        name = f"{digest}.txt"
        complete = spilled_bytes <= _SPILL_MAX_BYTES
        # Cut on a character boundary, so what lands is still decodable text.
        body = text if complete else head.decode("utf-8", "ignore")
        # newline="" so the bytes on disk are the bytes measured. The default translates
        # "\n" to os.linesep, which on Windows writes an extra byte per line, and the
        # byte offset in the continuation hint is counted from the untranslated text --
        # so a mid-line resume would start one byte early for every preceding newline.
        path = os.path.join(target_dir, name)
        # Written to a fresh O_EXCL file and moved into place, never opened O_TRUNC over
        # whatever is already at that path. The spill name is derived from content the
        # model produced, so it can predict it and pre-create it: as a symlink (refused by
        # O_NOFOLLOW and the check above) or as a HARD link, which reports islink() false
        # and shares the inode of some file outside the sandbox. Truncating that writes
        # through to it with the backend's privileges; replacing a directory entry does
        # not touch the linked file at all.
        # The name comes from the content, so re-running a command lands on the same path,
        # and the file there may no longer be the spill this wrote: a later call can have
        # put the user's own data at it. `_is_recorded_spill` is what says whether it is
        # still ours, and the rename below would otherwise replace it either way.
        path = os.path.join(target_dir, name)
        if os.path.exists(path):
            # The name is the digest of the text, so a recorded spill at it already HOLDS
            # this content: reuse it and write nothing. That is the repeat case this whole
            # change is about, and not writing is also the only way not to race with
            # another call that may be replacing the file right now.
            if _is_recorded_spill(
                os.path.join(workdir, _SPILL_DIR),
                path,
                _spill_manifest(os.path.join(workdir, _SPILL_DIR)),
            ):
                return f"{relative}/{name}", complete
            # Not ours: the user's code put something at that path, and the install below
            # would replace it.
            return None, True
        stamp = _write_spill_file(target_dir, name, body)
        if stamp is None:
            return None, True
        _record_spill(
            os.path.join(workdir, _SPILL_DIR),
            f"{scope}/{name}" if scope else name,
            stamp,
            hashlib.sha256(body.encode("utf-8", "surrogatepass")).hexdigest(),
        )
        _prune_spills(target_dir, os.path.join(workdir, _SPILL_DIR))
        # Relative, so the command works from the cwd the tools already run in, and so
        # the absolute sandbox path never reaches the model.
        return f"{relative}/{name}", complete
    except Exception:
        logger.debug("tool result spill failed", exc_info = True)
        return None, True


def _spill_files(root: str, directory: str, owned: "set[str]") -> "list[str]":
    """The spills in one directory: the ones ``owned`` records this process having written.

    Everything else there is the user's, whatever it is called. Links are never spills.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return [
        path
        for path in (os.path.join(directory, name) for name in names)
        if _is_recorded_spill(root, path, owned)
    ]


def _unlink_verified_spill(root: str, path: str, owned: "dict[str, tuple[str, str]]") -> bool:
    """Delete a spill, and only ever the file that was checked.

    Moved to a private name first. A rename is atomic, so from that point the inode this
    verifies is the inode this deletes: a sandbox writer that replaces the original name
    afterwards replaces nothing that is on its way out. Verifying and then unlinking by
    name cannot promise that, because the manifest lock orders Unsloth's own threads and
    the thing racing here is the sandbox.

    Checked again under the private name, and put back if it no longer matches, since at
    that point it is not the file this recorded and is not this to delete. The rename
    itself moves ctime, which is ours to move, so the second check compares the device,
    inode, size and mtime, and the content.

    Nothing but a regular file with the recorded identity is moved at all: a directory or
    a FIFO left at the name by the sandbox is rejected through its own descriptor before
    the rename, so the restore below is never asked to put back a kind of thing `os.link`
    cannot. That leaves only the vanishing window between the check and the rename, and
    the restore handles it by falling back to a rename when the name is free again.
    """
    directory, name = os.path.split(path)
    recorded = owned.get(os.path.relpath(path, root).replace(os.sep, "/"))
    if recorded is None:
        return False
    if not _is_stamped_regular(path, recorded[0]):
        return False
    private = os.path.join(directory, f".tmp-prune-{uuid.uuid4().hex[:12]}.txt")
    try:
        os.rename(path, private)
    except OSError:
        return False
    stamp = _spill_stamp(private)
    if (
        stamp is not None
        and stamp.split(":")[:4] == recorded[0].split(":")[:4]
        and recorded[1] == _file_digest(private, _stamp_size(recorded[0]))
    ):
        _quiet_unlink(private)
        return True
    _restore_pruned_path(private, path)
    return False


def _is_stamped_regular(path: str, stamp: str) -> bool:
    """Whether ``path`` is right now the regular file ``stamp`` recorded.

    Through a descriptor rather than the path, and for the same reasons `_file_digest`
    does it that way: O_NOFOLLOW refuses a symlink dropped at the name and O_NONBLOCK
    refuses to hang on a FIFO or a device. A directory opens, and is rejected by the
    mode check. ctime is deliberately not compared -- the caller compares it under the
    private name, where the rename it performs has already moved it.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = None
    try:
        fd = os.open(path, flags)
        info = os.fstat(fd)
    except OSError:
        return False
    finally:
        if fd is not None:
            os.close(fd)
    if not stat.S_ISREG(info.st_mode):
        return False
    return [str(part) for part in (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)] == (
        stamp.split(":")[:4]
    )


def _restore_pruned_path(private: str, path: str) -> None:
    """Put back something the prune moved but must not delete.

    `os.link` first, because it refuses to overwrite: if the sandbox has taken the name
    again in the meantime, the file it put there is not this to replace. A hard link
    cannot be made to a directory, so when it fails and the name is in fact free, the
    rename that moved this here is reversed instead, which works whatever was moved.
    Neither can promise the name is still free at the instant it runs, so the last resort
    is to leave the data under the private name and say where it went, rather than
    unlink it.
    """
    try:
        os.link(private, path)
        _quiet_unlink(private)
        return
    except OSError:
        pass
    try:
        if not os.path.lexists(path):
            os.rename(private, path)
            return
    except OSError:
        pass
    # The name is taken again, so whatever is there now is not this to overwrite either.
    # The file stays, under a name nothing will prune.
    logger.warning("tool result spill prune: kept a swapped file as %s", private)


def _prune_spills(target_dir: str, root: "str | None" = None) -> None:
    """Bound this scope by count and the whole spill tree by bytes.

    A long session prints many large results and each one is retained; without this the
    sandbox grows without bound for output the model has almost certainly finished paging
    through.

    The byte budget is enforced across ``root`` rather than per directory. A project's
    chats each get their own scope under one shared sandbox, so a per-directory limit is
    really a per-chat limit and a project with many tool-using chats multiplies it by
    however many there are.
    """
    root = root or target_dir
    try:
        # Held across the read and the rewrite below, so a spill recorded in between is
        # not dropped from the manifest by a prune that read it before the append.
        with _spill_lock(root):
            _prune_spills_locked(target_dir, root)
    except Exception:
        logger.debug("tool result spill prune failed", exc_info = True)


def _prune_spills_locked(target_dir: str, root: str) -> None:
    try:
        identity, owned = _spill_record(root)
        if identity is None or identity != _spill_identity(root):
            # No record of this directory, so it came with the sandbox or was replaced.
            return
        removed = set()
        # Newest first within this scope, so the count keeps the ones still being paged.
        for extra in sorted(
            _spill_files(root, target_dir, owned), key = os.path.getmtime, reverse = True
        )[_SPILL_KEEP:]:
            if _unlink_verified_spill(root, extra, owned):
                removed.add(extra)

        everything = [p for p in _spill_files(root, root, owned) if p not in removed]
        for name in sorted(os.listdir(root)):
            scope = os.path.join(root, name)
            if os.path.isdir(scope) and not os.path.islink(scope):
                everything.extend(p for p in _spill_files(root, scope, owned) if p not in removed)
        kept, total = 0, 0
        for path in sorted(everything, key = os.path.getmtime, reverse = True):
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            # The newest is always kept, whatever the budget says: it is the file the
            # notice about to be returned names, and a hint pointing at a path that was
            # deleted on the way out is worse than no hint at all.
            if kept == 0 or total + size <= _SPILL_MAX_TOTAL_BYTES:
                kept, total = kept + 1, total + size
                continue
            if _unlink_verified_spill(root, path, owned):
                removed.add(path)
        # The manifest follows the files: a name left in it after its file is gone would
        # keep being counted as something this owns.
        _write_spill_manifest(
            root,
            {
                name: stamp
                for name, stamp in owned.items()
                if os.path.join(root, *name.split("/")) not in removed
            },
        )
        # An emptied scope is a chat that stopped spilling, and leaving its directory
        # behind is what makes the sandbox look non-empty to the cleanup.
        for name in sorted(os.listdir(root)):
            scope = os.path.join(root, name)
            if os.path.isdir(scope) and not os.path.islink(scope) and not os.listdir(scope):
                try:
                    os.rmdir(scope)
                except OSError:
                    pass
    except Exception:
        logger.debug("tool result spill prune failed", exc_info = True)


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

# The sandbox CWD is a per-session dir under the studio home; an absolute path
# under it is a genuine local miss, not a hallucinated out-of-sandbox write.
# Resolved through the same helper as _get_workdir so the two cannot drift.


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
        root = os.path.realpath(workdir or sandbox_root())
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


_MAX_REPORTED_FILES = 25
# Bounded so an unpacked archive cannot turn a tool call into a filesystem
# crawl. In path segments, the unit the download route enforces, so a card can
# never advertise a file that route would refuse.
_MAX_SANDBOX_PATH_SEGMENTS = 4
_MAX_SNAPSHOT_FILES = 2000  # a shard-writing script must not blow up the result
_MAX_SNAPSHOT_DIRS = 2000  # nor a directory-writing one stall the next call


def _user_path_parts(parts: "list[str]", root: "str | None" = None) -> "list[str]":
    """The segments _MAX_SANDBOX_PATH_SEGMENTS applies to.

    The scratch container is Unsloth's, not a name the model chose, and on
    Windows it is what /tmp resolves to, so charging it a segment would drop one
    level of the /tmp artifacts served before the workdir stopped being %TEMP%.

    *root* is for callers that resolve the path afterwards. The walks read the
    stored spelling off os.walk and never follow links; the download route
    resolves, and without the root a link named unsloth-tmp (or a wrong-case
    entry on NTFS or APFS) would take the discount for a tree neither walk lists.
    """
    if not parts or parts[0] != _SANDBOX_TEMP_DIRNAME:
        return parts
    if root is not None and not _is_sandbox_temp_dir(
        os.path.join(root, _SANDBOX_TEMP_DIRNAME), root
    ):
        return parts
    return parts[1:]


# The same allowlist the download route applies per segment, so a name that
# route would refuse never reaches a file chip.
_SERVABLE_SEGMENT_RE = re.compile(r"\A[^/\\\x00-\x1f]{1,255}\Z")


def _servable_segment(name: str) -> bool:
    if name in (".", "..") or not _SERVABLE_SEGMENT_RE.match(name):
        return False
    # Non-UTF-8 bytes in a POSIX filename surface as lone surrogates, which
    # encodeURIComponent throws on, so the chip could never issue its download.
    return not any("\ud800" <= ch <= "\udfff" for ch in name)


# Unsloth's own bookkeeping, written by the sandbox sitecustomize. One exact
# name we write ourselves, not a pattern reserved over names a tool may pick.
_INTERNAL_SANDBOX_FILES = frozenset({".unsloth_sandbox_remap.json", _SANDBOX_MARKER})


# Above this a file is identified by mtime and size alone. Rewriting a large
# artifact byte for byte inside one filesystem tick is not worth reading every
# artifact twice per tool call.
_MAX_HASHED_SNAPSHOT_BYTES = 4 * 1024 * 1024


def _content_key(path: str, size: int) -> "str | None":
    """A digest for a file small enough to read, else None.

    On FAT/exFAT and some network volumes the timestamp granularity is a second
    or two, so an overwrite with different content of the same length inside one
    tick is invisible to mtime and size, and the call reported no file at all.
    """
    if size > _MAX_HASHED_SNAPSHOT_BYTES:
        return None
    try:
        digest = hashlib.blake2b(digest_size = 16)
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


# Volumes already known to timestamp finely, by device id. Only the positive is
# kept: a negative costs one stat to redo and hashing to get wrong.
_fine_mtime_devices: "set[int]" = set()


def _volume_timestamps_finely(workdir: str) -> bool:
    """Whether this volume records sub-second times, so digests can be skipped.

    That is the only thing the digest buys: at sub-second resolution a program
    cannot overwrite a file inside the tick of its own previous write, and
    reading every artifact twice per call was ~90% of a snapshot's cost.

    Read off the directory rather than probed with a file of our own: several
    chats can share a workdir, and a probe file appearing mid-walk is reported
    as a file the other call created. Three stamps because one whole-second
    reading on a fine volume is chance; three is not. Anything unreadable
    answers False, which hashes exactly as before.
    """
    try:
        stat = os.stat(workdir)
    except OSError:
        return False
    if stat.st_dev in _fine_mtime_devices:
        return True
    if not any(
        stamp % 1_000_000_000 for stamp in (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_atime_ns)
    ):
        return False
    _fine_mtime_devices.add(stat.st_dev)
    return True


def _defuse_sentinels(text: str) -> str:
    """Break a marker line the executed program printed itself.

    Both readers take the last one, so a call that created nothing and printed
    `__FILES__:[{"name": "report.csv", "size": 1}]` had that read as the
    envelope: the line was hidden from the model and the UI offered a download
    for a file nobody wrote. One space is enough, since both anchor on the line
    start, and it leaves the text otherwise as the program wrote it.
    """
    for marker in ("\n__FILES__:", "\n__IMAGES__:"):
        text = text.replace(marker, "\n " + marker[1:])
    return text


# What one snapshot may read to tell a same-size overwrite apart. The file cap
# alone allowed 2,000 x 4 MiB per walk, twice per call, which on a directory of
# artifacts made a trivial command look hung.
_MAX_SNAPSHOT_HASH_BYTES = 64 * 1024 * 1024


def _snapshot_workdir_files(workdir: str | None) -> "dict[str, tuple]":
    """relative path -> change key for every regular file, for the post-run diff.

    All files, not just images: a .csv the model wrote used to be invisible.
    Size rides along with mtime because FAT/exFAT and some network volumes have
    coarse timestamps, where an overwrite inside one tick would look unchanged,
    and on those volumes alone a digest rides along too, since a same-length
    rewrite inside one tick matches on all of it otherwise.
    """
    snapshot: "dict[str, tuple]" = {}
    if not workdir or not os.path.isdir(workdir):
        return snapshot
    # Directories are budgeted separately from files: thousands of empty output
    # folders never reach the file cap, and this walk runs twice per tool call.
    visited = 0
    hash_budget = 0 if _volume_timestamps_finely(workdir) else _MAX_SNAPSHOT_HASH_BYTES
    # Walked, not listed: a script writing outputs/report.csv is ordinary, and a
    # top-level listing saw only the directory and dropped it.
    for base, dirs, names in os.walk(workdir):
        visited += 1
        if visited > _MAX_SNAPSHOT_DIRS:
            return snapshot
        # depth 0 is the workdir itself, whose files are one segment.
        relative = base[len(workdir) :].strip(os.sep)
        depth = len(_user_path_parts(relative.split(os.sep) if relative else []))
        # Dot-directories stay out: .git, .cache and friends are where the noise
        # lives. Dot-FILES are reported, since .gitignore is a real artifact.
        dirs[:] = (
            []
            if depth >= _MAX_SANDBOX_PATH_SEGMENTS - 1
            else [d for d in dirs if not d.startswith(".") and _servable_segment(d)]
        )
        for name in names:
            # Only at the top: a tool that wrote archive/.unsloth_sandbox made an
            # ordinary file, and dropping it hid it from every listing while
            # still counting it as a reason to keep the sandbox.
            if base == workdir and name in _INTERNAL_SANDBOX_FILES:
                continue
            if not _servable_segment(name):
                continue
            path = os.path.join(base, name)
            try:
                # One lstat where isfile + islink + stat were three, on every
                # file of every walk. A link is not a regular file to lstat, so
                # this drops the same entries the pair did.
                stat = os.lstat(path)
                if not S_ISREG(stat.st_mode):
                    continue
                relative = os.path.relpath(path, workdir).replace(os.sep, "/")
                content = None
                if hash_budget and hash_budget >= stat.st_size:
                    content = _content_key(path, stat.st_size)
                    if content is not None:
                        hash_budget -= stat.st_size
                snapshot[relative] = (stat.st_mtime_ns, stat.st_size, content)
            except OSError:
                continue
            if len(snapshot) >= _MAX_SNAPSHOT_FILES:
                return snapshot
    return snapshot


# Scratch scripts of calls running right now. Chats in one project share a
# workdir and each snapshots all of it, so without this the other call's
# studio_exec_*.py is offered as this call's file and 404s once it is gone.
_active_scratch: "set[str]" = set()
_scratch_lock = threading.Lock()

# Tool calls running in each workdir. Chats in one project share one and each
# call diffs the whole tree, so the other call's file would land on this card.
# A call ever alongside another claims nothing, and no clock is involved.
_workdir_calls: "dict[str, list]" = {}
_calls_lock = threading.Lock()


def _call_started(workdir: "str | None") -> dict:
    """Register a call in *workdir* and hand back its token."""
    token = {"workdir": workdir, "shared": False}
    if not workdir:
        return token
    with _calls_lock:
        running = _workdir_calls.setdefault(workdir, [])
        if running:
            token["shared"] = True
            for other in running:
                other["shared"] = True
        running.append(token)
    return token


def _call_finished(token: "dict | None") -> None:
    """Drop a call's registration. Its token keeps whatever it learned."""
    if not token or not token.get("workdir"):
        return
    with _calls_lock:
        running = _workdir_calls.get(token["workdir"])
        if running is None:
            return
        try:
            running.remove(token)
        except ValueError:
            pass
        if not running:
            _workdir_calls.pop(token["workdir"], None)


def _snapshot_differs(before: tuple, after: tuple) -> bool:
    """Whether a file changed between two snapshots of its directory.

    The digest only when both snapshots have one: hashing stops at a byte
    budget, so a file added or removed earlier in the walk can push an
    untouched later file in or out of it, and comparing the tuples whole would
    then report that file as one this call wrote.
    """
    if before[:2] != after[:2]:
        return True
    return before[2] is not None and after[2] is not None and before[2] != after[2]


def _created_file_sentinels(
    workdir: str | None,
    before: "dict[str, tuple]",
    exclude: "str | None" = None,
    token: "dict | None" = None,
) -> str:
    """Sentinels naming the files this call created or overwrote.

    ``__IMAGES__`` renders inline as before; ``__FILES__`` carries every file
    with its size so the UI can offer a download. Both are stripped before the
    model sees the result.
    """
    if token is not None and token.get("shared"):
        # Another call ran in this directory while this one did, so nothing here
        # can be said to be ours. A missing card beats one that names, and
        # downloads, another chat's file.
        return ""
    after = _snapshot_workdir_files(workdir)
    if token is not None and token.get("shared"):
        # A call that started while that walk was running. What it wrote is in
        # `after` and cannot be told apart from ours, so the same rule applies:
        # the check above only saves the walk when the sharing was already known.
        return ""
    # ``exclude`` is this call's own scratch script by exact name, not a pattern
    # reserved over names a tool might pick.
    with _scratch_lock:
        scratch = set(_active_scratch)
    scratch.discard(exclude)  # this call's own is named below anyway
    changed = sorted(
        name
        for name, key in after.items()
        if name != exclude
        and name not in scratch
        and (name not in before or _snapshot_differs(before[name], key))
    )
    if not changed:
        return ""

    import json as _json

    # Same cap on both: a script writing a frame per step would otherwise put
    # every name in the result and the stored chat, and the UI would render them all.
    images = [n for n in changed if os.path.splitext(n)[1].lower() in _IMAGE_EXTS]
    images = images[:_MAX_REPORTED_FILES]
    entries = []
    for name in changed[:_MAX_REPORTED_FILES]:
        try:
            size = os.stat(os.path.join(workdir, name)).st_size
        except OSError:
            size = None
        entries.append({"name": name, "size": size})

    # __IMAGES__ stays LAST: older clients slice from it to the end of the
    # string, so anything after would land inside their JSON.
    out = f"\n__FILES__:{_json.dumps(entries)}"
    if images:
        out += f"\n__IMAGES__:{_json.dumps(images)}"
    return out


def _python_exec(
    code: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
    disable_sandbox: bool = False,
    output_callback = None,
    thread_id: str | None = None,
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
            # Capped like any other result: the analyzer names every occurrence it
            # found, so code that repeats a forbidden construct enough times reports
            # back something larger than the room that is left, which is the overflow
            # this budget exists to prevent. See `_fit_result_to_room`.
            return _truncate(error)
        # Stripping the child env is not enough: a same-UID child can read
        # /proc/<getppid()>/environ to recover the unfiltered secrets, so close
        # that read here too, not only in bypass mode. Best-effort: the child env
        # is already scrubbed, so a system where prctl is denied still runs.
        _harden_parent_against_proc_env_leak()
    elif not _harden_parent_against_proc_env_leak():
        # Close the /proc/<parent>/environ secret-recovery path first; if it
        # cannot be applied, fail closed rather than leak the parent environ.
        return (
            "Execution error: could not harden the Unsloth process against "
            "/proc environment reads; refusing bypass execution."
        )

    tmp_path = None
    _scratch_name = None
    workdir = _get_workdir(session_id)
    # `_get_workdir(None)` is the shared `_default` sandbox, and a project's chats share
    # one session by design. Retaining a result in either, under a path the next chat can
    # list, would leave behind output that existed only in this call's own response. See
    # `_spill_scope`, which returns None for exactly those cases.
    spill_scope = _spill_scope(session_id, thread_id)
    spill_dir = workdir if session_id else None
    call_token = _call_started(workdir)
    # Snapshot mtimes to detect new and overwritten files.
    _before = _snapshot_workdir_files(workdir)
    try:
        # In the workdir: Python puts it on sys.path[0], so an earlier call's
        # helper.py stays importable and __file__ resolves inside the sandbox.
        fd, tmp_path = tempfile.mkstemp(suffix = ".py", prefix = "studio_exec_", dir = workdir)
        # utf-8 so non-ASCII in model-written code survives the OS default codec
        # (Windows cp1252 would otherwise raise UnicodeEncodeError).
        _scratch_name = os.path.basename(tmp_path)
        with _scratch_lock:
            _active_scratch.add(_scratch_name)
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
        _adopt_tool_pid(proc.pid)

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
        # A run that wrote its file and then hung still produced that file, so
        # report it: `printf data > report.csv; sleep 999` is downloadable.
        if timed_out:
            ended = _truncate(f"Execution timed out after {timeout} seconds.")
            return ended + (
                _created_file_sentinels(workdir, _before, _scratch_name, call_token)
                if session_id
                else ""
            )

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled." + (
                _created_file_sentinels(workdir, _before, _scratch_name, call_token)
                if session_id
                else ""
            )

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        # Detect the missing-path pattern on the full output (truncation could
        # hide the trailing traceback); append the hint after truncation. External
        # paths are judged against the real workdir (project sessions live outside
        # the default sandbox root).
        hint = _missing_path_hint(result, workdir)
        # Before the fit, not after it. Defusing inserts a space into every line that
        # opens with a marker, so a result full of them grows after it has been measured
        # and the text replayed to the model is larger than the prefix that was admitted.
        # Before ours is appended, and whether or not one is: a program's own marker line
        # is not an envelope.
        result = _defuse_sentinels(result)
        result = (
            _truncate(result, workdir = spill_dir, scope = spill_scope, hint = hint)
            if result.strip()
            else "(no output)" + hint
        )

        # Only for a chat that has an id: without one every first turn shares
        # the _default workdir, so a card pinned to it would later download
        # whatever the next new chat wrote there.
        if session_id:
            result += _created_file_sentinels(workdir, _before, _scratch_name, call_token)

        return result

    except Exception as e:
        # An exception message carries whatever the failure put in it, so it is capped
        # like the result would have been.
        return _truncate(f"Execution error: {e}")
    finally:
        _call_finished(call_token)
        if _scratch_name:
            with _scratch_lock:
                _active_scratch.discard(_scratch_name)
        _forget_tool_pid(locals().get("proc"))
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
    thread_id: str | None = None,
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
            # Capped for the same reason the Python analyzer's error is: it lists what
            # it found in the command it was handed.
            return _truncate(f"Blocked command(s) for safety: {', '.join(sorted(blocked))}")
        # Stripping the child env is not enough: a same-UID child can read
        # /proc/<getppid()>/environ to recover the unfiltered secrets, so close
        # that read here too, not only in bypass mode. Best-effort: the child env
        # is already scrubbed, so a system where prctl is denied still runs.
        _harden_parent_against_proc_env_leak()
    elif not _harden_parent_against_proc_env_leak():
        # Close the /proc/<parent>/environ secret-recovery path first; if it
        # cannot be applied, fail closed rather than leak the parent environ.
        return (
            "Execution error: could not harden the Unsloth process against "
            "/proc environment reads; refusing bypass execution."
        )

    workdir = None
    spill_dir = None
    spill_scope = None
    call_token = None
    try:
        workdir = _get_workdir(session_id)
        # Same scoping as _python_exec: nothing is retained in a sandbox that is shared.
        spill_scope = _spill_scope(session_id, thread_id)
        spill_dir = workdir if session_id else None
        call_token = _call_started(workdir)
        # Same pre-run snapshot as _python_exec. A command that writes a file used
        # to produce "(no output)" and no other trace anywhere in the product.
        _before = _snapshot_workdir_files(workdir)
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
        _adopt_tool_pid(proc.pid)

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
        # A run that wrote its file and then hung still produced that file, so
        # report it: `printf data > report.csv; sleep 999` is downloadable.
        if timed_out:
            ended = _truncate(f"Execution timed out after {timeout} seconds.")
            return ended + (
                _created_file_sentinels(workdir, _before, None, call_token) if session_id else ""
            )

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled." + (
                _created_file_sentinels(workdir, _before, None, call_token) if session_id else ""
            )

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        # Same missing-path healing as _python_exec.
        hint = _missing_path_hint(result, workdir)
        result = _defuse_sentinels(result)  # before the fit; see _python_exec
        result = (
            _truncate(result, workdir = spill_dir, scope = spill_scope, hint = hint)
            if result.strip()
            else "(no output)" + hint
        )
        # Only for a chat that has an id (see _python_exec).
        if session_id:
            result += _created_file_sentinels(workdir, _before, None, call_token)
        return result

    except Exception as e:
        # An exception message carries whatever the failure put in it, so it is capped
        # like the result would have been.
        return _truncate(f"Execution error: {e}")
    finally:
        _call_finished(call_token)
        _forget_tool_pid(locals().get("proc"))
